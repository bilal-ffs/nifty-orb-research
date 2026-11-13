"""Generated IS experiments: cross-file reconciliation and chronology."""
from pathlib import Path
import json
import unittest
import joblib
import numpy as np
import pandas as pd
import strategy as s
import experiment as e


@unittest.skipUnless((e.OUT/'ml_metrics.json').exists(),'Run experiment.py to produce results first')
class ResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metrics=json.loads((e.OUT/'ml_metrics.json').read_text())
        cls.windows=json.loads((e.OUT/'fold_windows.json').read_text())
        cls.audit=json.loads((e.OUT/'data_audit.json').read_text())
        cls.comparison=pd.read_csv(e.OUT/'atr_multiplier_comparison.csv')

    def test_all_ten_configs_both_execution_modes_and_direction_totals(self):
        self.assertEqual(len(self.comparison),20)
        self.assertEqual(set(self.comparison.multiplier),set(s.MULTIPLIERS))
        directions=pd.read_csv(e.OUT/'long_short_breakdown.csv')
        for row in self.comparison.itertuples():
            g=directions[(directions.multiplier==row.multiplier)&(directions['mode']==row.mode)]
            self.assertEqual(g.trades.sum(),row.trades)
            self.assertAlmostEqual(g.total_net_pnl.sum(),row.total_net_pnl,places=6)
            self.assertAlmostEqual(row.target_rate+row.stop_rate+row.square_off_rate,1)

    def test_ledger_alignment_costs_and_no_overnight(self):
        for multiplier in s.MULTIPLIERS:
            for mode in ['next_open','pine_close']:
                t=pd.read_csv(e.OUT/f'atr_{multiplier:.1f}_{mode}_trades.csv')
                self.assertTrue(t.date.is_unique)
                self.assertTrue(t.date.between('2015-01-01','2021-12-31').all())
                self.assertTrue((t.risk_points>0).all())
                np.testing.assert_array_equal(t.target_hit,t.exit_reason.isin(['target','target_gap_at_entry']).astype(int))
                np.testing.assert_allclose(t.gross_pnl-t.fees_pnl-t.slippage_pnl,t.net_pnl,atol=1e-7)
                np.testing.assert_allclose(t.gross_r-t.fees_r-t.slippage_r,t.net_r,atol=1e-8)
                signal=pd.to_datetime(t.signal_candle_start); prediction=pd.to_datetime(t.prediction_timestamp)
                self.assertTrue((prediction==signal+pd.Timedelta(minutes=5)).all())
                self.assertTrue((pd.to_datetime(t.entry_timestamp)==prediction).all())
                self.assertTrue((pd.to_datetime(t.label_end_timestamp)>=prediction).all())
                self.assertTrue((pd.to_datetime(t.label_end_timestamp).dt.strftime('%Y-%m-%d')==t.date).all())
                self.assertTrue((pd.to_datetime(t.label_end_timestamp).dt.hour*60+pd.to_datetime(t.label_end_timestamp).dt.minute<=900).all())
                sign=t.direction.map({'long':1,'short':-1})
                np.testing.assert_allclose(t.target_price, # recover signal close from price + breakout distance/OR relation is not needed
                    t.initial_stop+sign*t.feature_signal_risk_pct*(t.initial_stop/(1-sign*t.feature_signal_risk_pct))+
                    sign*1.5*t.feature_or_width_points,rtol=1e-8,atol=1e-6)

    def test_common_chronological_windows_scaling_and_label_maturity(self):
        self.assertEqual(s.FEATURES[:7],e.v1.FEATURES)
        for key,metrics in self.metrics.items():
            tag=f'atr_{float(key):.1f}'
            t=pd.read_csv(e.OUT/f'{tag}_next_open_trades.csv')
            for fold,window in zip(metrics['folds'],self.windows):
                self.assertEqual(fold['test_start'],window['test_start'])
                self.assertEqual(fold['test_end'],window['test_end'])
                self.assertEqual(fold['status'],'scored')
                boundary=pd.Timestamp(window['test_start']+' 00:00',tz='Asia/Kolkata')
                train=(t.date<=window['train_end'])&(pd.to_datetime(t.label_end_timestamp)<boundary)
                self.assertLess(pd.to_datetime(fold['latest_training_label_end']),boundary)
                self.assertEqual(train.sum(),fold['train_n'])
                model=joblib.load(e.OUT/f'{tag}_fold_{fold["fold"]}.joblib')
                self.assertEqual(model.feature_names_in_.tolist(),s.FEATURES)
                np.testing.assert_allclose(model.named_steps['scaler'].mean_,t.loc[train,s.FEATURES].mean(),rtol=1e-10)
                self.assertEqual(model.named_steps['logistic'].C,1.)
                self.assertIsNone(model.named_steps['logistic'].class_weight)

    def test_oof_reproduces_saved_models_baselines_and_corresponding_labels(self):
        for key,metrics in self.metrics.items():
            tag=f'atr_{float(key):.1f}'
            t=pd.read_csv(e.OUT/f'{tag}_next_open_trades.csv').set_index('date')
            oof=pd.read_csv(e.OUT/f'{tag}_oof.csv')
            self.assertTrue(oof.date.is_unique)
            aligned=t.loc[oof.date]
            np.testing.assert_array_equal(oof.target_hit,aligned.target_hit)
            np.testing.assert_allclose(oof.net_r,aligned.net_r)
            for fold in metrics['folds']:
                g=oof[oof.fold==fold['fold']]
                model=joblib.load(e.OUT/f'{tag}_fold_{fold["fold"]}.joblib')
                np.testing.assert_allclose(model.predict_proba(t.loc[g.date,s.FEATURES])[:,1],g.probability,rtol=1e-10,atol=1e-12)
                np.testing.assert_allclose(g.baseline_probability,fold['training_prevalence'])
            final=joblib.load(e.OUT/f'{tag}_final.joblib')
            self.assertEqual(final.named_steps['scaler'].n_samples_seen_,len(t))
            for suffix in ['calibration','quintiles']:
                g=pd.read_csv(e.OUT/f'{tag}_{suffix}.csv')
                self.assertEqual(g.n.sum(),len(oof))
                self.assertAlmostEqual(np.average(g.mean_net_r,weights=g.n),oof.net_r.mean())

    def test_source_unchanged_and_no_execution_fields_in_features(self):
        self.assertEqual(s.digest(e.ROOT/'NIFTY_50_minute.csv'),self.audit['source_sha256'])
        self.assertTrue(all(c.startswith('feature_') for c in s.FEATURES))
        self.assertTrue(set(s.FEATURES).isdisjoint({'entry_price','risk_points','exit_price','target_hit','net_r','final_active_stop'}))

    def test_training_reproducibility(self):
        t=pd.read_csv(e.OUT/'atr_0.5_next_open_trades.csv')
        window=self.windows[0]
        boundary=pd.Timestamp(window['test_start']+' 00:00',tz='Asia/Kolkata')
        train=(t.date<=window['train_end'])&(pd.to_datetime(t.label_end_timestamp)<boundary)
        test=t.date.between(window['test_start'],window['test_end'])
        rebuilt=e.v1.make_model().fit(t.loc[train,s.FEATURES],t.loc[train,'target_hit'])
        saved=joblib.load(e.OUT/'atr_0.5_fold_1.joblib')
        np.testing.assert_allclose(rebuilt.predict_proba(t.loc[test,s.FEATURES]),saved.predict_proba(t.loc[test,s.FEATURES]),rtol=1e-8,atol=1e-10)


if __name__=='__main__': unittest.main(verbosity=2)
