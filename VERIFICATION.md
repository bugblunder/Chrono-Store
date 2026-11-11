# Verified run

Verified on 11 November 2026 with Python 3.12.14 and scikit-learn 1.8.0. Direct package versions are in `requirements-tested.txt`.

## Executed checks

- **23 automated tests passed** (`python -m unittest discover -s tests -v`).
- Built the Python wheel and installed it into an isolated target directory successfully.
- Ran the synthetic end-to-end demo: 4,197 events; model and training CSV generated.
- Ran the leakage demonstration: original spend 10 GBP; adding future/equal-time events kept it 10 GBP; a late past event changed a new snapshot to 30 GBP; replaying the original snapshot returned 10 GBP.
- Downloaded the UCI workbook, normalized it and imported in deliberately shuffled order.
- Trained the real-data model and saved all features, labels and split assignments.
- Exported offline features from the example JSONL requests.
- Started uvicorn and made real localhost HTTP calls to `/health`, `/features`, `/predict`, and `/openapi.json`. Feature and prediction provenance matched exactly; model loading and graceful shutdown succeeded.
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

| Metric | Logistic regression | Prior baseline |
|---|---:|---:|
| ROC AUC (higher is better) | 0.7845 | 0.5000 |
| Average precision (higher is better) | 0.8061 | 0.4942 |
| Brier score (lower is better) | 0.1898 | 0.2523 |
| Log loss (lower is better) | 0.5580 | 0.6978 |

These are one chronological holdout's results, without tuning on the holdout. They are not confidence intervals or claims about production performance. UCI has no original arrival history: evaluation reconstructs event-time availability from the imported snapshot. The model is evaluated on later purchases, including returning customers, rather than unseen customers only.

## Local performance sample

Measured by `python scripts/benchmark.py`; 200 feature reads per pass and 1,000 single-event writes from eight concurrent clients.

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
- `artifacts/uci/metrics.json`: machine-readable evaluation and feature metadata.
- `artifacts/uci/training_features.csv`: all eligible examples with targets and split assignments.
- `artifacts/import_report.json`: normalization decisions, counts and workbook checksum.
- `artifacts/benchmark.json`: measured local performance and scope.
- `artifacts/api_smoke.json`: actual HTTP feature/prediction response.
- `artifacts/offline_features.csv`: output of the example offline query file.
- `data/raw/source.json`: UCI download provenance; raw workbook omitted from this bundle.
