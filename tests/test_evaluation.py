"""Independent resampling reference, score semantics and degenerate cases."""
import unittest
from unittest.mock import Mock

import numpy as np
from sklearn.metrics import roc_auc_score

from pitstore.evaluation import customer_bootstrap_weights, evaluate_holdout


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.y = np.array([0, 1, 0, 1, 1, 0, 1, 0])
        self.customers = np.array(["a", "a", "b", "b", "b", "c", "c", "d"])
        self.predictions = {
            "count_90d": np.array([0, 3, 1, 1, 4, 0, 2, 1]),
            "recency": np.array([-91, -1, -20, -10, -1, -60, -4, -30]),
            "logistic_regression": np.array([.1, .8, .4, .3, .7, .2, .9, .6]),
        }

    def evaluate(self, **kwargs):
        return evaluate_holdout(self.y, self.customers, self.predictions, n_bootstrap=64, seed=21, **kwargs)

    def test_every_row_of_customer_receives_same_multiplicity(self):
        rng = Mock()
        rng.integers.return_value = np.array([0, 2, 0])
        weights = customer_bootstrap_weights(np.array([0, 0, 1, 2, 2]), 3, rng)
        np.testing.assert_array_equal(weights, [2, 2, 0, 1, 1])
        rng.integers.assert_called_once_with(0, 3, size=3)

    def test_bootstrap_matches_explicitly_duplicated_customer_rows(self):
        result = self.evaluate()
        rng = np.random.default_rng(21)
        customers = np.unique(self.customers)
        draws = {name: [] for name in self.predictions}
        for _ in range(64):
            sampled = rng.choice(customers, size=len(customers), replace=True)
            # Independent implementation physically duplicates all customer rows.
            indexes = np.concatenate([np.flatnonzero(self.customers == c) for c in sampled])
            if len(np.unique(self.y[indexes])) < 2:
                continue
            for name, prediction in self.predictions.items():
                draws[name].append(roc_auc_score(self.y[indexes], prediction[indexes]))
        self.assertEqual(result["bootstrap"]["valid_resamples"], len(draws["count_90d"]))
        for name, samples in draws.items():
            actual = result["methods"][name]["roc_auc_ci"]
            np.testing.assert_allclose([actual["lower"], actual["upper"]], np.quantile(samples, [.025, .975]))
        delta = np.array(draws["logistic_regression"]) - np.array(draws["count_90d"])
        actual = result["paired_auc_differences"]["logistic_regression_minus_count_90d"]
        np.testing.assert_allclose([actual["lower"], actual["upper"]], np.quantile(delta, [.025, .975]))

    def test_identical_methods_have_zero_paired_interval(self):
        p = dict(self.predictions)
        p["count_90d"] = p["logistic_regression"].copy()
        result = evaluate_holdout(self.y, self.customers, p, n_bootstrap=32)
        diff = result["paired_auc_differences"]["logistic_regression_minus_count_90d"]
        self.assertEqual([diff["estimate"], diff["lower"], diff["upper"]], [0., 0., 0.])
        self.assertEqual(diff["interpretation"], "no_clear_difference")

    def test_count_and_recency_are_ranking_scores_not_probabilities(self):
        result = self.evaluate()
        for name in ["count_90d", "recency"]:
            self.assertIsNone(result["methods"][name]["brier_score"])
            self.assertIsNone(result["methods"][name]["log_loss"])
            self.assertEqual(result["methods"][name]["score_type"], "ranking_only")
        expected = roc_auc_score(self.y, self.predictions["recency"])
        self.assertAlmostEqual(result["methods"]["recency"]["roc_auc"], expected)
        self.assertIsNotNone(result["methods"]["logistic_regression"]["brier_score"])

    def test_seed_reproduces_entire_evaluation(self):
        self.assertEqual(self.evaluate(), self.evaluate())

    def test_single_class_replicates_are_counted(self):
        result = evaluate_holdout([0, 0, 1, 1], ["a", "a", "b", "b"],
                                  {"count_90d": np.array([0, 1, 2, 3])}, n_bootstrap=64)
        boot = result["bootstrap"]
        self.assertGreater(boot["skipped_single_class_resamples"], 0)
        self.assertGreater(boot["valid_resamples"], 0)
        self.assertEqual(boot["valid_resamples"] + boot["skipped_single_class_resamples"], 64)

    def test_single_class_holdout_has_no_fabricated_auc_or_interval(self):
        result = evaluate_holdout([1, 1, 1], ["a", "b", "c"],
                                  {"count_90d": np.array([1, 2, 3])}, n_bootstrap=16)
        self.assertIsNone(result["methods"]["count_90d"]["roc_auc"])
        self.assertIsNone(result["methods"]["count_90d"]["roc_auc_ci"]["lower"])
        self.assertEqual(result["bootstrap"]["valid_resamples"], 0)

    def test_invalid_options_and_unaligned_scores_fail(self):
        for kwargs in [{"n_bootstrap": 1}, {"n_bootstrap": True}, {"confidence": 1}, {"confidence": 0}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                evaluate_holdout(self.y, self.customers, self.predictions, **kwargs)
        for prediction in [np.array([.2]), np.full(8, np.nan), np.full(8, 1.1)]:
            with self.assertRaises(ValueError):
                evaluate_holdout(self.y, self.customers, {"logistic_regression": prediction}, n_bootstrap=2)
        with self.assertRaisesRegex(ValueError, "two distinct customers"):
            evaluate_holdout(self.y, ["a"]*len(self.y), self.predictions, n_bootstrap=2)


if __name__ == "__main__":
    unittest.main()
