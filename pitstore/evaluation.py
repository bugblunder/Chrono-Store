"""Untuned ranking baselines and paired, customer-clustered AUC uncertainty."""
from collections.abc import Mapping

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


RULES = {
    "count_30d": "purchases_30d (larger score predicts more repeat purchases)",
    "count_90d": "purchases_90d (larger score predicts more repeat purchases)",
    "recency": "-recency_days (more recent purchases predict more repeat purchases)",
}
LEARNERS = ("logistic_regression", "gradient_boosting")


def validate_bootstrap_options(n_bootstrap: int, confidence: float):
    if type(n_bootstrap) is not int or n_bootstrap < 2:
        raise ValueError("bootstrap resamples must be an integer >= 2 (use at least 2000 for reports)")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between 0 and 1")


def point_scores(y, prediction, *, probability: bool = True) -> dict:
    both = len(np.unique(y)) == 2
    return {
        "roc_auc": float(roc_auc_score(y, prediction)) if both else None,
        "average_precision": float(average_precision_score(y, prediction)) if both else None,
        # A count or negative recency is a ranking score, not a probability.
        "brier_score": float(brier_score_loss(y, prediction)) if probability else None,
        "log_loss": float(log_loss(y, prediction, labels=[0, 1])) if probability else None,
    }


def customer_bootstrap_weights(customer_codes, n_customers, rng):
    """Sample C customers with replacement, giving all rows of each the same weight."""
    sampled = rng.integers(0, n_customers, size=n_customers)
    multiplicities = np.bincount(sampled, minlength=n_customers)
    return multiplicities[customer_codes]


def evaluate_holdout(y, customers, predictions: Mapping[str, np.ndarray], *,
                     probability_methods=LEARNERS + ("prior_baseline",),
                     n_bootstrap: int = 2000, seed: int = 42, confidence: float = 0.95) -> dict:
    """Conditional uncertainty of fixed models on one temporal holdout.

    Each replicate draws whole customers, retaining every anchor for each selected
    customer with its multiplicity. AUC remains purchase-anchor weighted. Identical
    draws across methods give paired AUC differences. Single-class replicates are
    skipped for all methods, reported explicitly, and never redrawn silently.
    """
    validate_bootstrap_options(n_bootstrap, confidence)
    y, customers = np.asarray(y), np.asarray(customers)
    if y.ndim != 1 or len(y) == 0 or not np.isin(y, [0, 1]).all():
        raise ValueError("targets must be a nonempty 1D binary array")
    if customers.ndim != 1 or len(customers) != len(y):
        raise ValueError("one customer ID is required per target")
    if any(c is None or str(c).strip() in ("", "nan", "<NA>") for c in customers):
        raise ValueError("customer IDs cannot be missing")
    if not predictions:
        raise ValueError("at least one prediction method is required")
    values = {name: np.asarray(value, dtype=float) for name, value in predictions.items()}
    for name, value in values.items():
        if value.shape != y.shape or not np.isfinite(value).all():
            raise ValueError(f"{name}: scores must be finite and aligned with targets")
        if name in probability_methods and ((value < 0) | (value > 1)).any():
            raise ValueError(f"{name}: probabilities must lie in [0, 1]")
    unique, codes = np.unique(customers, return_inverse=True)
    if len(unique) < 2:
        raise ValueError("customer bootstrap requires at least two distinct customers")
    methods = {
        name: {"score_type": "probability" if name in probability_methods else "ranking_only",
               **point_scores(y, value, probability=name in probability_methods)}
        for name, value in values.items()
    }
    draws = {name: [] for name in values}
    rng = np.random.default_rng(seed)
    for _ in range(n_bootstrap):
        weights = customer_bootstrap_weights(codes, len(unique), rng)
        if weights[y == 1].sum() == 0 or weights[y == 0].sum() == 0:
            continue
        for name, value in values.items():
            draws[name].append(float(roc_auc_score(y, value, sample_weight=weights)))
    valid = len(next(iter(draws.values())))
    alpha = (1 - confidence) / 2

    def interval(samples):
        if len(samples) < 2:
            return {"lower": None, "upper": None}
        low, high = np.quantile(samples, [alpha, 1-alpha])
        return {"lower": float(low), "upper": float(high)}

    for name in values:
        methods[name]["roc_auc_ci"] = interval(draws[name])
    differences = {}
    for learner in LEARNERS:
        if learner not in values:
            continue
        for reference in RULES:
            if reference not in values:
                continue
            diff = np.asarray(draws[learner]) - np.asarray(draws[reference])
            ci = interval(diff)
            auc, baseline_auc = methods[learner]["roc_auc"], methods[reference]["roc_auc"]
            conclusion = "unavailable"
            if ci["lower"] is not None:
                conclusion = ("higher_on_this_holdout" if ci["lower"] > 0 else
                              "lower_on_this_holdout" if ci["upper"] < 0 else "no_clear_difference")
            differences[f"{learner}_minus_{reference}"] = {
                "estimate": auc-baseline_auc if auc is not None and baseline_auc is not None else None,
                **ci, "interpretation": conclusion,
            }
    return {
        "methods": methods,
        "paired_auc_differences": differences,
        "bootstrap": {
            "method": "paired customer-cluster percentile bootstrap",
            "confidence_level": confidence, "seed": seed,
            "requested_resamples": n_bootstrap, "valid_resamples": valid,
            "skipped_single_class_resamples": n_bootstrap-valid, "test_customers": len(unique),
            "unit_of_resampling": "customer_id",
            "estimand": "purchase-anchor-weighted ROC AUC in this fixed holdout period",
            "primary_reference": "count_90d",
            "multiplicity_adjustment": "none; all comparisons are exploratory",
            "scope": "Conditional on fitted models and one chronological holdout. No model refitting; "
                     "does not measure training-sample, temporal-regime or cross-customer shared-shock uncertainty.",
        },
    }


def render_evaluation(metrics: dict) -> str:
    """Generate the reviewable report directly from the saved evaluation numbers."""
    def number(value):
        return "n/a" if value is None else f"{value:.4f}"

    boot = metrics["bootstrap"]
    lines = [
        "# Repeat-purchase evaluation", "",
        f"Data: `{metrics['data_kind']}`. Feature version: `{metrics['feature_version']}`. "
        f"Snapshot: `{metrics['snapshot_seq']}`.", "",
        f"**Target:** given a customer completes a positive purchase now at `t`, predict another "
        f"positive purchase in `(t, t + {metrics['horizon_days']} days]` using only invoices strictly before `t`. "
        "This is conditional repeat purchase, not general customer churn.", "",
        f"Train: {metrics['train_rows']:,} anchors. Purged: {metrics['purged_rows']:,}. "
        f"Test: {metrics['test_rows']:,} anchors from {boot['test_customers']:,} customers.", "",
        f"Test starts `{metrics['test_start']}`. Latest training label ends "
        f"`{metrics['train_last_label_end']}`. Test positive rate: {metrics['test_positive_rate']:.2%}.", "",
        "All methods use identical rows, served features and split boundaries. "
        "The learners use fixed parameters; the holdout is not used for tuning, early stopping "
        "or choosing the default serving model. Logistic regression remains the default.", "",
        f"| Method | ROC AUC | {boot['confidence_level']:.0%} AUC interval | Average precision | Brier | Log loss |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for name, result in metrics["methods"].items():
        ci = result["roc_auc_ci"]
        lines.append(f"| {name} | {number(result['roc_auc'])} | "
                     f"[{number(ci['lower'])}, {number(ci['upper'])}] | "
                     f"{number(result['average_precision'])} | {number(result['brier_score'])} | "
                     f"{number(result['log_loss'])} |")
    lines += ["", "Count rules use raw prior purchase counts in the stated window. "
              "Recency uses negative days since the last purchase. These are ranking scores; "
              "probability metrics are intentionally undefined. The prior baseline uses only training prevalence.", "",
              "## Paired AUC differences", "",
              "Positive differences favor the learner. The primary reference was fixed as the 90-day count rule. "
              "The 30-day count and recency comparisons are also reported; no winner is selected on the holdout.", "",
              "| Comparison | AUC difference | Interval | Interpretation |", "|---|---:|---|---|"]
    for name, result in metrics["paired_auc_differences"].items():
        lines.append(f"| {name} | {number(result['estimate'])} | "
                     f"[{number(result['lower'])}, {number(result['upper'])}] | "
                     f"{result['interpretation'].replace('_', ' ')} |")
    lines += ["", "## Uncertainty and limits", "",
              f"{boot['requested_resamples']:,} customer bootstrap resamples, seed {boot['seed']}; "
              f"{boot['valid_resamples']:,} valid and {boot['skipped_single_class_resamples']} "
              "single-class resamples skipped. Each draw samples as many customers as observed, "
              "with replacement, and includes every held-out row for each selected customer, with multiplicity. "
              "All methods share each draw. Percentile intervals use the configured confidence level; "
              "paired differences are calculated within each draw, not by subtracting interval endpoints.", "",
              boot["scope"], "These are exploratory comparisons without a multiple-comparison correction. "
              "An interval excluding zero describes this holdout; it does not establish superiority across "
              "seasons, future datasets or unseen customers. Rolling temporal holdouts are a further validation step.", "",
              metrics["availability_note"], "",
              "The same customer can appear in training and test. More active customers contribute more "
              "purchase anchors; this is not a customer-equal-weighted metric. Tail anchors lacking a full "
              "outcome window are excluded. Returns/cancellations are excluded, so the target concerns "
              "gross positive purchase activity.", "",
              "Reproduce with `python -m pitstore train --bootstrap-resamples 2000 --bootstrap-seed 42`. "
              "See `metrics.json`, `training_features.csv` and `holdout_predictions.csv` alongside this report "
              "for exact scores, labels and split membership.", ""]
    return "\n".join(lines)
