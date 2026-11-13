"""Diagnostic unit and generated-output reconciliation tests; IS only."""
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from orb_diagnostics import (load_is, reconstruct, quantile_groups, cluster_stats,
                             SEED, REPLICATES, FEATURES, profit_factor)


class DiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(__file__).resolve().parent
        cls.t,cls.metrics=load_is(cls.root)
        cls.c=reconstruct(cls.t,cls.metrics['config'])
        cls.groups=pd.read_csv(cls.root/'orb_feature_conditioned_is.csv')
        cls.exits=pd.read_csv(cls.root/'orb_exit_distribution_is.csv')

    def test_original_is_count_and_boundary(self):
        self.assertEqual(len(self.t),1624)
        self.assertTrue(self.t.date.between('2015-01-01','2021-12-31').all())
        self.assertTrue(self.t.date.is_unique)

    def test_per_trade_reconstruction_and_sizing(self):
        for a,b in [('gross_points','gross_points'),('net_points','net_points'),('gross_r','gross_r'),
                    ('net_r','net_r'),('gross_pnl','gross_pnl'),('net_pnl','net_pnl'),
                    ('fees_pnl','transaction_cost_pnl'),('slippage_pnl','slippage_pnl')]:
            np.testing.assert_allclose(self.c[a],self.t[b],rtol=1e-10,atol=1e-8)
        np.testing.assert_allclose(self.c.original_units,self.t.units)
        np.testing.assert_allclose(self.c.net_pnl,self.c.gross_pnl-self.c.total_cost_pnl)

    def test_fee_and_slippage_components(self):
        self.assertTrue((self.c.entry_fee_points>0).all())
        self.assertTrue((self.c.exit_fee_points>0).all())
        self.assertTrue((self.c.entry_slippage_points>0).all())
        self.assertTrue((self.c.exit_slippage_points>0).all())
        np.testing.assert_allclose(self.c.gross_r-self.c.net_r,self.c.fees_r+self.c.slippage_r)

    def test_fixed_reference_prices_not_adverse_fills(self):
        sign=self.t.direction.map({'long':1,'short':-1})
        reference=sign*(self.t.exit_reference_price-self.t.entry_reference_open)
        adverse=sign*(self.t.exit_price-self.t.entry_price)
        np.testing.assert_allclose(self.c.gross_points,reference)
        self.assertTrue((self.c.gross_points>adverse).all())

    def test_total_net_pnl_reconciliation(self):
        self.assertAlmostEqual(self.c.net_pnl.sum(),self.metrics['net_pnl'],places=6)
        self.assertAlmostEqual(self.c.net_pnl.sum(),-162720.7869754629,places=6)
        exported=pd.read_csv(self.root/'orb_cost_decomposition.csv')
        np.testing.assert_allclose(exported.net_pnl,self.c.net_pnl)

    def test_every_partition_count_pnl_and_weighted_mean(self):
        for name,g in self.groups.groupby('feature'):
            with self.subTest(feature=name):
                self.assertEqual(g.number_of_trades.sum(),1624)
                self.assertAlmostEqual(g.total_net_pnl.sum(),self.t.net_pnl.sum(),places=6)
                self.assertAlmostEqual(np.average(g.net_expectancy_r,weights=g.number_of_trades),self.t.net_r.mean(),places=10)
                np.testing.assert_allclose(g.target_pct+g.stop_loss_pct+g.session_exit_pct+g.other_exit_pct,100)
                self.assertTrue((g.net_r_ci_low<=g.net_r_ci_high).all())
                self.assertTrue((g.bootstrap_replicates>=2000).all())
                self.assertTrue((g.bootstrap_seed==SEED).all())

    def test_is_only_quantile_edges_and_missing_counts(self):
        for feature in FEATURES:
            labels,edges=quantile_groups(self.t[feature])
            g=self.groups[self.groups.feature==feature]
            bins=g[g.group!='Missing']
            self.assertEqual(len(bins),5)
            np.testing.assert_allclose(bins.lower_bound,edges[:-1])
            np.testing.assert_allclose(bins.upper_bound,edges[1:])
            for row in g.itertuples():
                self.assertEqual(row.number_of_trades,int((labels==row.group).sum()))
                self.assertAlmostEqual(row.net_expectancy_r,self.t.loc[labels==row.group,'net_r'].mean(),places=10)
            missing=g[g.group=='Missing'].number_of_trades.sum()
            self.assertEqual(missing,self.t[feature].isna().sum())

    def test_quantile_ties_not_arbitrarily_split(self):
        s=pd.Series([1.]*30+[2.]*30+[3.]*30+[np.nan])
        labels,edges=quantile_groups(s)
        self.assertLess(len(edges)-1,5)
        for value in [1.,2.,3.]: self.assertEqual(labels[s==value].nunique(),1)
        self.assertEqual(labels.iloc[-1],'Missing')

    def test_timing_uses_signal_start_half_open(self):
        start=pd.to_datetime(self.t.signal_candle_start)
        minute=start.dt.hour*60+start.dt.minute
        g=self.groups[self.groups.feature=='breakout_timing']
        for row in g.itertuples():
            mask=minute.between(row.lower_bound,row.upper_bound-1)
            self.assertEqual(row.number_of_trades,int(mask.sum()))
            self.assertAlmostEqual(row.net_expectancy_r,self.t.loc[mask,'net_r'].mean(),places=10)
        prediction=pd.to_datetime(self.t.prediction_timestamp)
        self.assertTrue((prediction==start+pd.Timedelta(minutes=1)).all())

    def test_exit_categories_reconcile(self):
        e=self.exits
        self.assertEqual(e.trade_count.sum(),1624)
        self.assertAlmostEqual(e.percentage_of_trades.sum(),100)
        self.assertAlmostEqual(e.total_net_pnl.sum(),self.t.net_pnl.sum(),places=6)
        for row in e.itertuples():
            subset=self.t[self.t.exit_reason==row.exit_category]
            self.assertEqual(row.trade_count,len(subset))
            self.assertAlmostEqual(row.average_net_r,subset.net_r.mean(),places=10)

    def test_bootstrap_deterministic_and_true_day_clustering(self):
        # Different rows of the same day must travel together. Per-trade
        # bootstrap would produce many impossible intermediate estimates.
        fixture=pd.DataFrame({'date':['A','A','B'],'r':[0.,2.,8.]})
        a=cluster_stats(fixture,'r','fixture',replicates=2000)
        b=cluster_stats(fixture,'r','fixture',replicates=2000)
        self.assertEqual(a,b)
        self.assertEqual(a['days'],2)
        self.assertAlmostEqual(a['mean'],10/3)
        self.assertEqual(a['ci_low'],1.)
        self.assertEqual(a['ci_high'],8.)
        sums=np.array([2.,8.]); counts=np.array([2.,1.])
        expected=np.sqrt(2*np.sum((sums-(10/3)*counts)**2))/3
        self.assertAlmostEqual(a['standard_error'],expected)

    def test_overall_se_and_bootstrap(self):
        self.assertGreaterEqual(REPLICATES,2000)
        for col in ['gross_r','net_r']:
            s=cluster_stats(self.t,col,'test_'+col)
            self.assertAlmostEqual(s['standard_error'],self.t[col].std()/np.sqrt(1624))
            self.assertLess(s['ci_low'],s['mean'])
            self.assertGreater(s['ci_high'],s['mean'])

    def test_profit_factor_uses_original_currency_sizing(self):
        self.assertAlmostEqual(profit_factor(self.c.net_pnl),self.metrics['profit_factor_net'])
        values=pd.Series([100.,-25.,50.])
        self.assertEqual(profit_factor(values),6.)


if __name__=='__main__': unittest.main(verbosity=2)
