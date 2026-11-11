# Repeat-purchase evaluation

Data: `imported`. Feature version: `retail-v1`. Snapshot: `18532`.

**Target:** given a customer completes a positive purchase now at `t`, predict another positive purchase in `(t, t + 30 days]` using only invoices strictly before `t`. This is conditional repeat purchase, not general customer churn.

Train: 7,710 anchors. Purged: 1,501. Test: 3,112 anchors from 1,964 customers.

Test starts `2011-09-23T09:14:00Z`. Latest training label ends `2011-09-23T09:13:00Z`. Test positive rate: 49.42%.

All methods use identical rows, served features and split boundaries. The learners use fixed parameters; the holdout is not used for tuning, early stopping or choosing the default serving model. Logistic regression remains the default.

| Method | ROC AUC | 95% AUC interval | Average precision | Brier | Log loss |
|---|---:|---|---:|---:|---:|
| prior_baseline | 0.5000 | [0.5000, 0.5000] | 0.4942 | 0.2523 | 0.6978 |
| count_30d | 0.7238 | [0.6948, 0.7506] | 0.7155 | n/a | n/a |
| count_90d | 0.7813 | [0.7552, 0.8047] | 0.7836 | n/a | n/a |
| recency | 0.7367 | [0.7109, 0.7602] | 0.6738 | n/a | n/a |
| logistic_regression | 0.7845 | [0.7587, 0.8072] | 0.8061 | 0.1898 | 0.5580 |
| gradient_boosting | 0.7851 | [0.7597, 0.8079] | 0.8050 | 0.1885 | 0.5535 |

Count rules use raw prior purchase counts in the stated window. Recency uses negative days since the last purchase. These are ranking scores; probability metrics are intentionally undefined. The prior baseline uses only training prevalence.

## Paired AUC differences

Positive differences favor the learner. The primary reference was fixed as the 90-day count rule. The 30-day count and recency comparisons are also reported; no winner is selected on the holdout.

| Comparison | AUC difference | Interval | Interpretation |
|---|---:|---|---|
| logistic_regression_minus_count_30d | 0.0607 | [0.0479, 0.0735] | higher on this holdout |
| logistic_regression_minus_count_90d | 0.0032 | [-0.0011, 0.0069] | no clear difference |
| logistic_regression_minus_recency | 0.0477 | [0.0366, 0.0580] | higher on this holdout |
| gradient_boosting_minus_count_30d | 0.0614 | [0.0492, 0.0737] | higher on this holdout |
| gradient_boosting_minus_count_90d | 0.0038 | [-0.0007, 0.0083] | no clear difference |
| gradient_boosting_minus_recency | 0.0484 | [0.0378, 0.0586] | higher on this holdout |

## Uncertainty and limits

2,000 customer bootstrap resamples, seed 42; 2,000 valid and 0 single-class resamples skipped. Each draw samples as many customers as observed, with replacement, and includes every held-out row for each selected customer, with multiplicity. All methods share each draw. Percentile intervals use the configured confidence level; paired differences are calculated within each draw, not by subtracting interval endpoints.

Conditional on fitted models and one chronological holdout. No model refitting; does not measure training-sample, temporal-regime or cross-customer shared-shock uncertainty.
These are exploratory comparisons without a multiple-comparison correction. An interval excluding zero describes this holdout; it does not establish superiority across seasons, future datasets or unseen customers. Rolling temporal holdouts are a further validation step.

All events in the recorded snapshot are assumed available at event time. UCI has no original arrival history; this is not a replay of historical live visibility.

The same customer can appear in training and test. More active customers contribute more purchase anchors; this is not a customer-equal-weighted metric. Tail anchors lacking a full outcome window are excluded. Returns/cancellations are excluded, so the target concerns gross positive purchase activity.

Reproduce with `python -m pitstore train --bootstrap-resamples 2000 --bootstrap-seed 42`. See `metrics.json`, `training_features.csv` and `holdout_predictions.csv` alongside this report for exact scores, labels and split membership.
