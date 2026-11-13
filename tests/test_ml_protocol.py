"""Synthetic checks of actual research functions, without historical inputs."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('orb_v1_protocol', ROOT / 'orb_logistic_v1/train.py')
v1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v1)


class MLProtocolTests(unittest.TestCase):
    def test_fold_purges_immature_labels(self):
        prediction = pd.Series(pd.date_range('2020-01-01', periods=60, tz='Asia/Kolkata'))
        label_end = prediction + pd.Timedelta(hours=1)
        label_end.iloc[9] = prediction.iloc[10] + pd.Timedelta(minutes=1)
        folds = v1.make_folds(prediction, label_end)
        self.assertEqual(len(folds), 5)
        self.assertIn(9, folds[0][3])
        seen = set()
        for _, train, test, _ in folds:
            self.assertTrue(seen.issubset(set(train)))
            self.assertTrue(set(train).isdisjoint(test))
            self.assertLess(label_end.iloc[train].max(), prediction.iloc[test].min())
            seen = set(train)

    def test_scaler_and_oof_predictions_ignore_later_features(self):
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.normal(size=(60, len(v1.FEATURES))), columns=v1.FEATURES)
        y = pd.Series(np.tile([0, 1], 30))
        prediction = pd.Series(pd.date_range('2020-01-01', periods=60, tz='Asia/Kolkata'))
        label_end = prediction + pd.Timedelta(hours=1)
        f = pd.DataFrame({'event_id': prediction.dt.strftime('%Y-%m-%d'),
                          'prediction_timestamp': prediction})
        t = pd.DataFrame({'net_r': rng.normal(size=60)})
        a, folds, models = v1.train_oof(X, y, f, t, prediction, label_end)
        changed = X.copy()
        changed.iloc[20:] = changed.iloc[20:] * 1000 + 500
        b, _, changed_models = v1.train_oof(changed, y, f, t, prediction, label_end)
        np.testing.assert_allclose(a.loc[a.fold == 1, 'probability'],
                                   b.loc[b.fold == 1, 'probability'])
        model = models[0]
        np.testing.assert_allclose(model.named_steps['scaler'].mean_, X.iloc[:10].mean())
        np.testing.assert_allclose(model.named_steps['logistic'].coef_,
                                   changed_models[0].named_steps['logistic'].coef_)
        np.testing.assert_allclose(a.baseline_probability, .5)
        self.assertEqual(len(a), 50)
        self.assertTrue(a.event_id.is_unique)

    def test_feature_and_label_contract_is_verified_in_current_pipeline(self):
        proof = v1.verify_pipeline(ROOT / 'orb_pipeline.py')
        self.assertEqual(set(proof), {'first_event', 'simulate_trade', 'run_split'})
        self.assertTrue(all(name.startswith('feature_') for name in v1.FEATURES))
        self.assertTrue(all(not any(word in name for word in ('target', 'fill', 'outcome'))
                            for name in v1.FEATURES))


if __name__ == '__main__':
    unittest.main()
