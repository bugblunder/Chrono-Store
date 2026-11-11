"""Shared served features, mature labels and a calendar-time purged holdout."""
from bisect import bisect_right
from datetime import datetime
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .contracts import DAY_US, FEATURE_NAMES, FEATURE_VERSION, FeatureQuery, from_us, iso_us, to_us
from .evaluation import RULES, evaluate_holdout, point_scores, render_evaluation, validate_bootstrap_options
from .store import FeatureStore


def build_training_frame(store: FeatureStore, *, horizon_days: int = 30, warmup_days: int = 90,
                         observation_end: datetime | None = None, snapshot_seq: int | None = None) -> tuple[pd.DataFrame, dict]:
    if horizon_days < 1 or warmup_days < 0:
        raise ValueError("horizon must be positive and warmup nonnegative")
    rows, snapshot = store.export_events(snapshot_seq)
    if not rows:
        raise ValueError("ingest events before training")
    start = rows[0]["event_time_us"]
    end = to_us(observation_end) if observation_end else rows[-1]["event_time_us"]
    if end < start:
        raise ValueError("observation_end precedes all events")
    rows = [r for r in rows if r["event_time_us"] <= end]
    histories = {}
    for r in rows:
        histories.setdefault(r["customer_id"], []).append(r["event_time_us"])
    anchors = []
    seen = set()
    for r in rows:
        t, customer = r["event_time_us"], r["customer_id"]
        if t < start+warmup_days*DAY_US or t+horizon_days*DAY_US > end or (customer, t) in seen:
            continue
        seen.add((customer, t))
        times = histories[customer]
        next_index = bisect_right(times, t)  # Ignore all purchases at the anchor timestamp.
        target = int(next_index < len(times) and times[next_index] <= t+horizon_days*DAY_US)
        anchors.append({"customer_id": customer, "as_of_us": t, "label_end_us": t+horizon_days*DAY_US,
                        "target": target, "snapshot_seq": snapshot})
    if not anchors:
        raise ValueError("no fully observed anchors after warmup; provide a longer history")
    # Every historical example passes through exactly the same store API as serving.
    for offset in range(0, len(anchors), 512):
        chunk = anchors[offset:offset+512]
        vectors = store.features_many([FeatureQuery(customer_id=a["customer_id"], as_of=from_us(a["as_of_us"]),
                                                     snapshot_seq=snapshot) for a in chunk])
        for anchor, vector in zip(chunk, vectors):
            anchor.update(vector.values)
    frame = pd.DataFrame(anchors)
    metadata = {"feature_version": FEATURE_VERSION, "feature_names": FEATURE_NAMES,
                "horizon_days": horizon_days, "warmup_days": warmup_days, "snapshot_seq": snapshot,
                "observation_start": iso_us(start), "observation_end": iso_us(end),
                "availability_mode": "reconstructed_event_time",
                "availability_note": "All events in the recorded snapshot are assumed available at event time. "
                                     "UCI has no original arrival history; this is not a replay of historical live visibility."}
    return frame, metadata


def purged_split(frame: pd.DataFrame, test_fraction: float = 0.25):
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    times = sorted(frame.as_of_us.unique())
    if len(times) < 2:
        raise ValueError("at least two distinct prediction times are required")
    boundary = int(times[min(len(times)-1, max(1, int(len(times)*(1-test_fraction))))])
    # Training labels must already be observable BEFORE the first test prediction.
    train = frame.loc[frame.label_end_us < boundary].copy()
    test = frame.loc[frame.as_of_us >= boundary].copy()
    if len(train) < 20 or len(test) < 10 or train.target.nunique() != 2:
        raise ValueError("not enough mature examples / target classes for a purged temporal holdout")
    return train, test, boundary


def scores(y, probability):
    """Compatibility wrapper for probability-only metrics."""
    return point_scores(y, probability)


def train_model(store: FeatureStore, output: Path, *, horizon_days: int = 30,
                warmup_days: int = 90, observation_end: datetime | None = None,
                test_fraction: float = 0.25, data_kind: str = "imported",
                bootstrap_resamples: int = 2000, bootstrap_seed: int = 42,
                confidence: float = 0.95) -> dict:
    validate_bootstrap_options(bootstrap_resamples, confidence)
    frame, metadata = build_training_frame(store, horizon_days=horizon_days,
                                           warmup_days=warmup_days, observation_end=observation_end)
    train, test, boundary = purged_split(frame, test_fraction)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42))
    model.fit(train[FEATURE_NAMES], train.target)
    baseline = DummyClassifier(strategy="prior").fit(train[FEATURE_NAMES], train.target)
    # Fixed before evaluation. No random validation split, early stopping or test-set tuning.
    boosting_parameters = dict(n_estimators=150, learning_rate=0.05, max_depth=3,
                               min_samples_leaf=30, subsample=1.0, random_state=42,
                               n_iter_no_change=None)
    boosting = GradientBoostingClassifier(**boosting_parameters)
    boosting.fit(train[FEATURE_NAMES], train.target)
    predictions = {
        "prior_baseline": baseline.predict_proba(test[FEATURE_NAMES])[:, 1],
        "count_30d": test.purchases_30d.to_numpy(dtype=float),
        "count_90d": test.purchases_90d.to_numpy(dtype=float),
        "recency": -test.recency_days.to_numpy(dtype=float),
        "logistic_regression": model.predict_proba(test[FEATURE_NAMES])[:, 1],
        "gradient_boosting": boosting.predict_proba(test[FEATURE_NAMES])[:, 1],
    }
    evaluation = evaluate_holdout(test.target, test.customer_id, predictions,
                                  n_bootstrap=bootstrap_resamples, seed=bootstrap_seed, confidence=confidence)
    metrics = {"data_kind": data_kind,
               "train_rows": len(train), "test_rows": len(test), "purged_rows": len(frame)-len(train)-len(test),
               "test_start": iso_us(boundary), "train_last_label_end": iso_us(int(train.label_end_us.max())),
               "train_positive_rate": float(train.target.mean()), "test_positive_rate": float(test.target.mean()),
               "target_definition": f"Given a positive purchase now at t, another positive purchase in "
                                    f"(t, t + {horizon_days} days]; not general churn",
               "served_model": "logistic_regression",
               "model_selection": "Default retained a priori; holdout not used for selection or tuning",
               "baseline_definitions": RULES,
               "model_parameters": {
                   "logistic_regression": {"scaler": "StandardScaler fitted on training only",
                                           **model[-1].get_params()},
                   "gradient_boosting": boosting.get_params(),
               },
               "test_fraction": test_fraction,
               "model": evaluation["methods"]["logistic_regression"],
               "prior_baseline": evaluation["methods"]["prior_baseline"],
               "sklearn_version": sklearn.__version__, "numpy_version": np.__version__,
               **metadata, **evaluation}
    output.mkdir(parents=True, exist_ok=True)
    frame["split"] = np.where(frame.as_of_us >= boundary, "test",
                               np.where(frame.label_end_us < boundary, "train", "purged"))
    frame["as_of"] = frame.as_of_us.map(iso_us)
    frame["label_end"] = frame.label_end_us.map(iso_us)
    frame.to_csv(output / "training_features.csv", index=False)
    holdout = frame.loc[test.index, ["customer_id", "as_of", "as_of_us", "label_end",
                                    "target", "snapshot_seq"]].copy()
    for name, prediction in predictions.items():
        holdout[name] = prediction
    holdout.to_csv(output / "holdout_predictions.csv", index=False)
    metrics["file_sha256"] = {
        name: hashlib.sha256((output/name).read_bytes()).hexdigest()
        for name in ("training_features.csv", "holdout_predictions.csv")
    }
    # Joblib is for trusted, locally produced artifacts only. Both models are serveable.
    joblib.dump({"model": model, "metadata": metrics}, output / "model.joblib")
    boosting_metadata = {**metrics, "served_model": "gradient_boosting",
                         "model": evaluation["methods"]["gradient_boosting"]}
    joblib.dump({"model": boosting, "metadata": boosting_metadata}, output / "gradient_boosting.joblib")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "evaluation.md").write_text(render_evaluation(metrics), encoding="utf-8")
    return metrics


def load_model(path: Path):
    artifact = joblib.load(path)
    metadata = artifact["metadata"]
    if metadata["feature_version"] != FEATURE_VERSION or metadata["feature_names"] != FEATURE_NAMES:
        raise ValueError("model feature schema is incompatible with this feature store")
    return artifact


def predict(artifact, vector):
    values = pd.DataFrame([[vector.values[name] for name in FEATURE_NAMES]], columns=FEATURE_NAMES)
    return float(artifact["model"].predict_proba(values)[0, 1])
