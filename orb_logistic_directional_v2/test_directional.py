"""IS-only tests of reconciliation, timing, chronology and fixed models."""
from pathlib import Path
import json
import unittest
import joblib
import numpy as np
import pandas as pd
import run as research


class DirectionalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f,cls.t,cls.X,cls.y,cls.pred,cls.end,_=research.v1.load_data(research.ROOT)
        cls.metrics=json.loads((research.OUT/'directional_model_metrics.json').read_text())
        cls.manifest=json.loads((research.OUT/'input_manifest.json').read_text())
        cls.diag=pd.read_csv(research.OUT/'directional_diagnostics.csv')

    def test_all_trade_reconciliation(self):
        self.assertEqual(len(self.t),1624)
        self.assertEqual(self.diag.n.tolist(),[759,865])
        self.assertAlmostEqual(self.diag.total_net_pnl.sum(),-162720.7869754629)
        self.assertTrue(self.t.date.between('2015-01-01','2021-12-31').all())
        for row in self.diag.itertuples():
            g=self.t[self.t.direction==row.direction]
            self.assertAlmostEqual(row.target_hit_rate,g.target_hit.mean())
            self.assertAlmostEqual(row.gross_expectancy_r,g.gross_r.mean())
            self.assertAlmostEqual(row.net_expectancy_r,g.net_r.mean())
            self.assertAlmostEqual(row.stop_pct+row.target_pct+row.session_exit_pct,100)
            self.assertAlmostEqual(row.gross_expectancy_r-row.net_expectancy_r,row.mean_fees_r+row.mean_slippage_r)

    def test_bootstrap_reproducible_day_clusters(self):
        a=research.bootstrap_mean(self.t,'net_r',research.SEED)
        b=research.bootstrap_mean(self.t,'net_r',research.SEED)
        self.assertEqual(a,b)
        self.assertEqual(research.REPLICATES,2000)
        fixture=pd.DataFrame({'date':['A','A','B'],'r':[0.,2.,8.]})
        ci=research.bootstrap_mean(fixture,'r',123)
        self.assertEqual(ci,[1.,8.])

    def test_exact_v1_features_minus_constant_and_no_leakage(self):
        self.assertEqual(research.FEATURES,[f for f in research.v1.FEATURES if f!='feature_breakout_direction'])
        self.assertEqual(len(research.FEATURES),6)
        self.assertIn('feature_or_direction',research.FEATURES)
        self.assertTrue(np.isfinite(self.X[research.FEATURES]).all().all())
        self.assertTrue(set(research.FEATURES).isdisjoint({research.v1.TARGET,'target_net_return_r','outcome_exit_reason','fill_entry_price'}))
        np.testing.assert_array_equal(self.y,(self.t.exit_reason=='target').astype(int))
        self.assertTrue((self.pred==pd.to_datetime(self.f.signal_candle_start)+pd.Timedelta(minutes=1)).all())

    def test_each_direction_has_five_expanding_mature_folds(self):
        for direction,m in self.metrics['models'].items():
            subset=self.t[self.t.direction==direction].reset_index(drop=True)
            previous=set(); test_indices=[]
            self.assertEqual(len(m['folds']),5)
            for fold in m['folds']:
                train,test=fold['train_indices'],fold['test_indices']
                self.assertTrue(previous.issubset(set(train)))
                self.assertTrue(set(train).isdisjoint(test))
                self.assertLess(pd.to_datetime(fold['latest_training_label_end']),pd.to_datetime(fold['first_test_prediction']))
                self.assertEqual(fold['purged_indices'],[])
                self.assertEqual(fold['train_positive_count']+fold['train_negative_count'],len(train))
                self.assertEqual(fold['test_positive_count']+fold['test_negative_count'],len(test))
                self.assertEqual(fold['train_positive_count'],subset.target_hit.iloc[train].sum())
                previous=set(train); test_indices+=test
            self.assertEqual(len(test_indices),len(set(test_indices)))
            self.assertEqual(m['warmup_n']+m['oof_n'],len(subset))

    def test_training_scaler_baseline_and_fixed_parameters(self):
        for direction,m in self.metrics['models'].items():
            subset=self.f[self.t.direction==direction].reset_index(drop=True)
            for fold in m['folds']:
                train,test=fold['train_indices'],fold['test_indices']
                model=joblib.load(research.OUT/f'{direction}_fold_{fold["fold"]}.joblib')
                np.testing.assert_allclose(model.named_steps['scaler'].mean_,subset[research.FEATURES].iloc[train].mean())
                self.assertEqual(model.named_steps['scaler'].n_samples_seen_,len(train))
                self.assertEqual(model.named_steps['logistic'].C,1.)
                self.assertIsNone(model.named_steps['logistic'].class_weight)
                self.assertAlmostEqual(fold['training_prevalence'],subset[research.v1.TARGET].iloc[train].mean())
                self.assertEqual(model.feature_names_in_.tolist(),research.FEATURES)

    def test_oof_reproducibility_and_alignment(self):
        for direction in ['long','short']:
            regenerated,metrics,_=research.fit_direction(self.f,self.t,direction)
            saved=pd.read_csv(research.OUT/f'{direction}_oof_predictions.csv')
            np.testing.assert_array_equal(regenerated.event_id,saved.event_id)
            np.testing.assert_allclose(regenerated.probability,saved.probability,rtol=1e-12,atol=1e-12)
            ledger=self.t.set_index('date').loc[saved.event_id]
            np.testing.assert_array_equal(saved.target,ledger.target_hit)
            np.testing.assert_allclose(saved.realized_net_r,ledger.net_r)
            np.testing.assert_allclose(saved.gross_r-saved.realized_net_r,saved.costs_r)
            self.assertTrue((ledger.direction==direction).all())

    def test_paired_same_events_and_no_in_sample_v1_substitution(self):
        pooled=pd.read_csv(research.ROOT/'orb_logistic_v1/oof_predictions.csv')
        for direction in ['long','short']:
            directional=pd.read_csv(research.OUT/f'{direction}_oof_predictions.csv')
            p,c=research.paired_comparison(directional,pooled)
            self.assertEqual(set(p.event_id),set(directional.event_id)&set(pooled.event_id))
            self.assertEqual(c['v1_same_events']['n'],c['v2_same_events']['n'])
            self.assertEqual(c['matched_n']+len(c['unmatched_v2_event_ids']),len(directional))
            self.assertEqual(c['matched_n'],self.metrics['paired_comparison'][direction]['matched_n'])
            np.testing.assert_allclose(p.probability_v1,pooled.set_index('event_id').loc[p.event_id,'probability'])

    def test_quintile_totals_and_returns(self):
        q=pd.read_csv(research.OUT/'directional_probability_quintiles.csv')
        for direction,m in self.metrics['models'].items():
            oof=pd.read_csv(research.OUT/f'{direction}_oof_predictions.csv')
            g=q[(q.model==m['model_name'])&(q['sample']=='full_direction_oof')]
            self.assertEqual(g.n.sum(),len(oof))
            self.assertEqual(len(g),5)
            self.assertAlmostEqual(np.average(g.mean_net_r,weights=g.n),oof.realized_net_r.mean())
            self.assertAlmostEqual(np.average(g.actual_target_rate,weights=g.n),oof.target.mean())
            np.testing.assert_allclose(g.mean_gross_r-g.mean_net_r,g.mean_costs_r)

    def test_final_models_fit_only_own_direction_and_existing_files_unchanged(self):
        for direction in ['long','short']:
            model=joblib.load(research.OUT/f'ORB_{direction.upper()}_LOGISTIC.joblib')
            mask=self.t.direction==direction
            self.assertEqual(model.named_steps['scaler'].n_samples_seen_,int(mask.sum()))
            np.testing.assert_allclose(model.named_steps['scaler'].mean_,self.X.loc[mask,research.FEATURES].mean())
        for name,h in self.manifest['protected_input_sha256'].items():
            self.assertEqual(research.v1.sha(research.ROOT/name),h)


if __name__=='__main__': unittest.main(verbosity=2)
