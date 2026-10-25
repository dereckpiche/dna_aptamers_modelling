"""Checks saved calculations without training a model."""
from pathlib import Path
import json
import unittest
import numpy as np

P = Path(__file__).resolve().parent


class InferenceChecks(unittest.TestCase):
    def setUp(self):
        a = np.load(P / 'test_predictions.npz')
        self.y = a['target'].astype(float)
        self.pred = a['prediction'].astype(float)
        self.results = json.loads((P / 'statistical_results.json').read_text())

    def test_metrics_match_saved_predictions(self):
        loss = (self.y - self.pred)**2
        self.assertAlmostEqual(loss.mean(), self.results['mse'], places=12)
        r2 = 1 - loss.sum() / ((self.y - self.y.mean())**2).sum()
        self.assertAlmostEqual(r2, self.results['r2'], places=12)
        original = json.loads((P / 'final_results.json').read_text())
        self.assertLess(abs(original['test_mse']-loss.mean()), 1e-6)

    def test_bootstrap_preserves_pairs_and_centers_null(self):
        ref = json.loads((P / 'reference.json').read_text())
        y, pred = self.y, self.pred
        loss = (pred-y)**2
        gain = (y-ref['training_mean'])**2-loss
        idx = np.random.default_rng(751903).integers(0, len(y), size=(20,len(y)))
        expected_r2 = 1-loss[idx].mean(axis=1)/y[idx].var(axis=1)
        expected_t = ((gain[idx].mean(axis=1)-gain.mean()) /
                      (gain[idx].std(axis=1,ddof=1)/np.sqrt(len(y))))
        saved = np.load(P / 'bootstrap_replicates.npz')['statistics']
        np.testing.assert_allclose(saved[:20,0], expected_r2, rtol=0, atol=1e-12)
        np.testing.assert_allclose(saved[:20,4], expected_t, rtol=0, atol=1e-12)
        ci = np.quantile(saved[:,0],[.025,.975])
        np.testing.assert_allclose(ci,self.results['r2_ci95'])
        count = int((saved[:,4] >= self.results['primary_statistic']).sum())
        self.assertEqual(count,self.results['null_exceedances'])
        self.assertEqual((1+count)/(len(saved)+1),self.results['primary_bootstrap_one_sided_p'])

    def test_split_integrity(self):
        s = np.load(P / 'split_indices.npz')
        a = np.concatenate([s[k] for k in ['train','validation','test']])
        self.assertEqual(len(np.unique(a)),1000000)
        p = np.load(P / 'test_predictions.npz')
        np.testing.assert_array_equal(s['test'],p['original_row'])

    def test_r2_is_not_squared_correlation(self):
        y = np.arange(10,dtype=float)
        self.assertEqual(1-((y-y)**2).sum()/((y-y.mean())**2).sum(),1)
        shifted = y+100
        r2 = 1-((shifted-y)**2).sum()/((y-y.mean())**2).sum()
        self.assertLess(r2,0)
        self.assertAlmostEqual(np.corrcoef(y,shifted)[0,1]**2,1)

    def test_transformer_metrics_use_recovered_predictions(self):
        import pandas as pd
        df = pd.read_csv(P.parent/'src'/'data.csv',float_precision='round_trip')
        target = df.true.to_numpy(dtype=np.float32).astype(float)
        pred = df.predict.to_numpy(dtype=float)
        r = json.loads((P/'other_model_results.json').read_text())['larger_transformer']
        self.assertAlmostEqual(((pred-target)**2).mean(),r['mse'],places=12)
        self.assertAlmostEqual(1-((pred-target)**2).mean()/target.var(),r['r2'],places=12)
        self.assertAlmostEqual(r['mse'],0.15640475240276502,places=12)

    def test_matched_model_comparison_and_null_centering(self):
        a = np.load(P/'matched_predictions.npz')
        self.assertEqual(len(a['target']),9371)
        self.assertEqual(len(np.unique(a['original_row'])),9371)
        self.assertTrue(np.isin(a['original_row'],np.load(P/'split_indices.npz')['test']).all())
        idx = np.searchsorted(np.sort(np.load(P/'split_indices.npz')['test']),a['original_row'])
        target = a['target'].astype(float)
        left = (a['lstm']-target)**2
        right = (a['transformer']-target)**2
        gain = left-right
        result = json.loads((P/'other_model_results.json').read_text())['matched_comparison']
        self.assertAlmostEqual(gain.mean(),result['gain'],places=12)
        ix = np.random.default_rng(751905).integers(0,len(gain),size=(20,len(gain)))
        expected_t = (gain[ix].mean(axis=1)-gain.mean())/(gain[ix].std(axis=1,ddof=1)/np.sqrt(len(gain)))
        saved = np.load(P/'additional_bootstrap_replicates.npz')['matched']
        np.testing.assert_allclose(saved[:20,3],expected_t,rtol=0,atol=1e-12)

    def test_holm_adjustment_and_three_comparisons(self):
        import pandas as pd
        table = pd.read_csv(P/'model_statistical_tests.csv')
        self.assertEqual(len(table),3)
        raw = table['Raw Monte Carlo p'].to_numpy()
        order = np.argsort(raw)
        adjusted = np.empty_like(raw)
        running = 0
        for rank,i in enumerate(order):
            running = max(running, min(1,(len(raw)-rank)*raw[i]))
            adjusted[i]=running
        np.testing.assert_allclose(table['Holm p'].to_numpy(),adjusted,rtol=0,atol=1e-12)


if __name__ == '__main__':
    unittest.main(verbosity=2)
