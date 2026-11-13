"""IS-only leakage, temporal-validation, reproducibility and artifact tests."""
from pathlib import Path
import json
import unittest
import joblib
import numpy as np
import pandas as pd
from train import (FEATURES, TARGET, INPUTS, load_data, validate_data, make_model,
                   make_folds, train_oof, scores, sha)


class LogisticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out=Path(__file__).resolve().parent
        cls.root=cls.out.parent
        cls.f,cls.t,cls.X,cls.y,cls.pred,cls.end,cls.proofs=load_data(cls.root)
        cls.oof=pd.read_csv(cls.out/'oof_predictions.csv')
        cls.folds=json.loads((cls.out/'folds.json').read_text())
        cls.metrics=json.loads((cls.out/'metrics.json').read_text())

    def test_feature_whitelist_no_outcomes_fills_or_raw_timestamps(self):
        self.assertEqual(self.X.columns.tolist(),FEATURES)
        self.assertEqual(len(FEATURES),7)
        for c in FEATURES:
            self.assertTrue(c.startswith('feature_'))
            self.assertNotIn(c,[TARGET,'feature_history_available'])
            self.assertNotIn('timestamp',c)
            self.assertNotIn('fill',c)
            self.assertNotIn('target',c)
        self.assertTrue(np.isfinite(self.X.to_numpy()).all())

    def test_label_verified_and_mutation_rejected(self):
        np.testing.assert_array_equal(self.y,(self.t.exit_reason=='target').astype(int))
        bad=self.f.copy()
        bad.loc[0,TARGET]=1-int(bad.loc[0,TARGET])
        with self.assertRaises(ValueError): validate_data(bad,self.t)

    def test_missing_selected_feature_stops_without_guessing(self):
        bad=self.f.copy()
        bad.loc[0,FEATURES[0]]=np.nan
        with self.assertRaises(ValueError): validate_data(bad,self.t)

    def test_non_is_and_duplicate_events_rejected(self):
        bad=self.f.copy()
        bad.loc[0,'event_id']='2022-01-01'
        with self.assertRaises(ValueError): validate_data(bad,self.t)
        bad=self.f.copy()
        bad.loc[0,'event_id']=bad.loc[1,'event_id']
        with self.assertRaises(ValueError): validate_data(bad,self.t)

    def test_alignment_survives_input_shuffling(self):
        f,t,X,y,pred,end=validate_data(self.f.sample(frac=1,random_state=3),self.t.sample(frac=1,random_state=4))
        pd.testing.assert_frame_equal(X,self.X)
        np.testing.assert_array_equal(y,self.y)
        self.assertTrue(pred.is_monotonic_increasing)
        np.testing.assert_array_equal(f.event_id,t.date)

    def test_five_expanding_folds_and_mature_labels(self):
        previous=set()
        tests=[]
        for number,train,test,purged in make_folds(self.pred,self.end):
            self.assertTrue(previous.issubset(set(train)))
            self.assertLess(self.pred.iloc[train].max(),self.pred.iloc[test].min())
            self.assertLess(self.end.iloc[train].max(),self.pred.iloc[test].min())
            self.assertEqual(len(purged),0)
            self.assertEqual(len(test),270)
            self.assertTrue(set(train).isdisjoint(test))
            previous=set(train); tests.extend(test)
        self.assertEqual(number,5)
        self.assertEqual(len(tests),1350)
        self.assertEqual(len(set(tests)),1350)
        self.assertEqual(min(tests),274)

    def test_overlap_purge_uses_label_end(self):
        prediction=pd.Series(pd.date_range('2020-01-01',periods=12,tz='Asia/Kolkata'))
        end=prediction+pd.Timedelta(hours=2)
        end.iloc[1]=prediction.iloc[2]+pd.Timedelta(minutes=1)
        number,train,test,purged=make_folds(prediction,end)[0]
        self.assertIn(1,purged)
        self.assertNotIn(1,train)
        self.assertLess(end.iloc[train].max(),prediction.iloc[test].min())

    def test_scalers_models_and_baselines_use_fold_train_only(self):
        for definition in self.folds:
            train=definition['train_indices']; test=definition['test_indices']
            model=joblib.load(self.out/f"fold_{definition['fold']}.joblib")
            np.testing.assert_allclose(model.named_steps['scaler'].mean_,self.X.iloc[train].mean().to_numpy(),rtol=1e-12)
            self.assertEqual(model.named_steps['scaler'].n_samples_seen_,len(train))
            g=self.oof[self.oof.fold==definition['fold']]
            np.testing.assert_array_equal(g.row_index,test)
            np.testing.assert_allclose(model.predict_proba(self.X.iloc[test])[:,1],g.probability,rtol=1e-12)
            np.testing.assert_allclose(g.baseline_probability,self.y.iloc[train].mean())
            self.assertEqual(model.named_steps['logistic'].C,1)
            self.assertIsNone(model.named_steps['logistic'].class_weight)

    def test_oof_probabilities_reproducible(self):
        oof,folds,models=train_oof(self.X,self.y,self.f,self.t,self.pred,self.end)
        np.testing.assert_allclose(oof.probability,self.oof.probability,rtol=1e-12,atol=1e-12)
        np.testing.assert_array_equal(oof.row_index,self.oof.row_index)

    def test_future_features_cannot_change_earlier_fit(self):
        definition=self.folds[0]
        train,test=definition['train_indices'],definition['test_indices']
        changed=self.X.copy()
        changed.iloc[test]=changed.iloc[test]*1000+500
        model=make_model().fit(changed.iloc[train],self.y.iloc[train])
        original=joblib.load(self.out/'fold_1.joblib')
        np.testing.assert_allclose(model.named_steps['logistic'].coef_,original.named_steps['logistic'].coef_,rtol=1e-12)
        np.testing.assert_allclose(model.named_steps['scaler'].mean_,original.named_steps['scaler'].mean_,rtol=1e-12)

    def test_oof_event_label_return_alignment_and_coverage(self):
        indices=self.oof.row_index.to_numpy()
        self.assertTrue(self.oof.event_id.is_unique)
        np.testing.assert_array_equal(self.oof.event_id,self.f.event_id.iloc[indices])
        np.testing.assert_array_equal(self.oof.target,self.y.iloc[indices])
        np.testing.assert_allclose(self.oof.realized_net_r,self.t.net_r.iloc[indices])
        self.assertEqual(len(self.oof)+self.metrics['warmup_events'],1624)
        self.assertTrue(self.oof.probability.between(0,1).all())

    def test_metric_calibration_and_quintile_reconciliation(self):
        for kind,col in [('oof_model','probability'),('oof_baseline','baseline_probability')]:
            reconstructed=scores(self.oof.target,self.oof[col])
            for key,value in reconstructed.items():
                if isinstance(value,list): self.assertEqual(value,self.metrics[kind][key])
                else: self.assertAlmostEqual(value,self.metrics[kind][key],places=12)
        for file in ['calibration_deciles.csv','baseline_calibration.csv','probability_quintiles.csv']:
            g=pd.read_csv(self.out/file)
            self.assertEqual(g.n.sum(),1350)
            self.assertAlmostEqual(np.average(g.observed_target_rate,weights=g.n),self.oof.target.mean())
            self.assertAlmostEqual(np.average(g.mean_realized_net_r,weights=g.n),self.oof.realized_net_r.mean())

    def test_final_model_trained_all_is_and_inputs_unchanged(self):
        final=joblib.load(self.out/'final_model.joblib')
        self.assertEqual(final.named_steps['scaler'].n_samples_seen_,1624)
        np.testing.assert_allclose(final.named_steps['scaler'].mean_,self.X.mean().to_numpy())
        self.assertEqual(final.feature_names_in_.tolist(),FEATURES)
        for name in INPUTS:
            self.assertEqual(sha(self.root/name),self.metrics['input_sha256'][name])


if __name__=='__main__': unittest.main(verbosity=2)
