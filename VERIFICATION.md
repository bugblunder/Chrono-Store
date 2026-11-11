# Verified run

Evaluation revision 0.2.0 verified on 11 November 2026 with Python 3.12.14,
scikit-learn 1.8.0 and NumPy 2.3.5. Direct package versions are in
`requirements-tested.txt`. The existing UCI database was reused; no fresh data
download or new model tuning was performed in this revision.

## Executed checks

- **34 automated tests passed** (`python -m unittest discover -s tests -v`).
- Built the version 0.2.0 Python wheel and verified installation in an isolated target directory.
- Re-ran the synthetic training workflow on the existing 4,197-event demo store with all six methods and 2,000 bootstrap resamples. These results are smoke tests, not evidence of real-world ML quality.
- Ran the leakage demonstration: original spend 10 GBP; adding future/equal-time events kept it 10 GBP; a late past event changed a new snapshot to 30 GBP; replaying the original snapshot returned 10 GBP.
- Reused the UCI import from the initial project, which was ingested in deliberately shuffled order. Checked the available source workbook checksum and event count against its import report.
- Trained logistic regression and gradient boosting on the same real-data split, evaluated four baselines and saved all features, labels, split assignments and holdout scores.
- Computed 2,000 paired customer-cluster bootstrap samples (all valid) for six AUCs and six learner-minus-rule differences.
- Verified the bootstrap against an independent implementation that physically duplicates whole customers' rows. Changing held-out labels leaves every method's fitted predictions unchanged.
- Started uvicorn separately for both model artifacts and made real localhost HTTP calls to `/health`, `/features`, `/predict`, and `/openapi.json`. Feature and prediction provenance matched exactly; target caveat and model names were exposed, and graceful shutdown succeeded.
- Tested package construction on this Python runtime. The included CI matrix is configured for Python 3.11–3.13; those other runtimes were not executed here.

## UCI preprocessing

| Measurement | Result |
|---|---:|
| Source line items | 541,909 |
| Rows excluded for incomplete identity | 135,080 |
| Cancellation rows excluded after identity filtering | 8,905 |
| Nonpositive/missing value rows excluded next | 40 |
| Eligible positive line items | 397,884 |
| Aggregated purchase events | 18,532 |
| Invoices with multiple line timestamps | 30 |
| Longest invoice timestamp span | 60 seconds |

All multi-timestamp invoices use their latest eligible line time. This avoids making later line items available early. Naive timestamps are explicitly interpreted as Europe/London and converted to UTC. Repeated identical lines are retained; returns are excluded. This is positive purchase activity, not net revenue.

Source workbook SHA-256: `43465a06f2ccf7c8b5bd2892bc7defb52f97487934fe93b16ae4c3936424676d`.

## Real-data holdout

| Split | Examples |
|---|---:|
| Train | 7,710 |
| Purged around holdout boundary | 1,501 |
| Test | 3,112 |

First test timestamp: `2011-09-23T09:14:00Z`. Latest training label end: `2011-09-23T09:13:00Z`. Every training label ends strictly before the first test prediction. The outcome observation endpoint is `2011-12-09T12:50:00Z`. Test positive rate: 49.42%.

| Method | ROC AUC | 95% customer-bootstrap interval | Average precision | Brier | Log loss |
|---|---:|---|---:|---:|---:|
| Training prevalence | 0.5000 | [0.5000, 0.5000] | 0.4942 | 0.2523 | 0.6978 |
| 30-day count | 0.7238 | [0.6948, 0.7506] | 0.7155 | n/a | n/a |
| 90-day count | 0.7813 | [0.7552, 0.8047] | 0.7836 | n/a | n/a |
| Negative recency | 0.7367 | [0.7109, 0.7602] | 0.6738 | n/a | n/a |
| Logistic regression | 0.7845 | [0.7587, 0.8072] | 0.8061 | 0.1898 | 0.5580 |
| Gradient boosting | 0.7851 | [0.7597, 0.8079] | 0.8050 | 0.1885 | 0.5535 |

The holdout contains 1,964 customers. Bootstrap unit: whole customer, sampled with
replacement, retaining all their anchors with multiplicity. Seed: 42. Resamples:
2,000. Percentile confidence level: 95%. Predictions are fixed during resampling;
models are not refit. The same resamples are used for all methods and paired
differences. Counts and negative recency are ranking scores, so their probability
metrics are undefined.

Against the predeclared 90-day count reference, logistic regression's AUC difference
is **+0.0032 (95% interval [-0.0011, +0.0069])**; gradient boosting's is
**+0.0038 ([-0.0007, +0.0083])**. Both include zero. **Neither model demonstrates a
clear AUC improvement over that rule on this holdout.** The full exploratory paired
comparisons are in [the generated report](artifacts/uci/evaluation.md).

The target is another purchase in the next 30 days **given a customer purchases
now**, not general churn. These are one chronological holdout's results, without
tuning on the holdout. The intervals are conditional on this period and fitted
models; they do not measure variation across seasons, training samples or future
deployments, and have no multiple-comparison correction. UCI has no original
arrival history: evaluation reconstructs event-time availability from the imported
snapshot. The model is evaluated on later purchases, including returning customers,
rather than unseen customers only.

## Local performance sample

Measurements retained from the initial project; not rerun for this evaluation-only
revision. Command: `python scripts/benchmark.py`; 200 feature reads per pass and
1,000 single-event writes from eight concurrent clients.

| Measurement | Result |
|---|---:|
| First-pass feature read + persistence, median | 2.635 ms |
| First-pass feature read + persistence, p95 | 7.855 ms |
| Cached feature read + persistence, median | 0.711 ms |
| Cached feature read + persistence, p95 | 0.888 ms |
| Concurrent ingestion | 560.7 events/sec |

Local library calls include SQLite connection setup and persistence. They exclude network latency. These are measurements on this host, not production service-level guarantees; rerunning the first pass can hit existing cache rows.

## Included results

- `data/store.sqlite3`: processed UCI events, committed snapshot and materialized feature rows.
- `artifacts/uci/model.joblib`: trained model; load only trusted local artifacts.
- `artifacts/uci/gradient_boosting.joblib`: fixed-parameter nonlinear alternative, also serveable.
- `artifacts/uci/metrics.json`: machine-readable evaluation and feature metadata.
- `artifacts/uci/evaluation.md`: generated comparison, paired intervals and scope caveats.
- `artifacts/uci/holdout_predictions.csv`: shared holdout identities, labels and all six method scores.
- `artifacts/uci/training_features.csv`: all eligible examples with targets and split assignments.
- `artifacts/import_report.json`: normalization decisions, counts and workbook checksum.
- `artifacts/benchmark.json`: measured local performance and scope.
- `artifacts/api_smoke.json`: actual HTTP feature/prediction response.
- `artifacts/offline_features.csv`: output of the example offline query file.
- `data/raw/source.json`: UCI download provenance; raw workbook omitted from this bundle.

The report, metrics, feature CSV, prediction CSV and import report are committed.
SQLite databases and serialized models are bundled but excluded from Git. The
archive includes actual Git history; `AI_ASSISTANCE.md` explains its scope and the
distinction between automated checkpoints and personal contributions.
