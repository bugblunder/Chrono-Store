"""Shared served features, mature labels and a calendar-time purged holdout."""
from bisect import bisect_right
from datetime import datetime
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .contracts import DAY_US, FEATURE_NAMES, FEATURE_VERSION, FeatureQuery, from_us, iso_us, to_us
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
    both = len(set(y)) == 2
    return {"roc_auc": float(roc_auc_score(y, probability)) if both else None,
            "average_precision": float(average_precision_score(y, probability)) if both else None,
            "brier_score": float(brier_score_loss(y, probability)),
            "log_loss": float(log_loss(y, probability, labels=[0, 1]))}


def train_model(store: FeatureStore, output: Path, *, horizon_days: int = 30,
                warmup_days: int = 90, observation_end: datetime | None = None,
                test_fraction: float = 0.25, data_kind: str = "imported") -> dict:
    frame, metadata = build_training_frame(store, horizon_days=horizon_days,
                                           warmup_days=warmup_days, observation_end=observation_end)
    train, test, boundary = purged_split(frame, test_fraction)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42))
    model.fit(train[FEATURE_NAMES], train.target)
    baseline = DummyClassifier(strategy="prior").fit(train[FEATURE_NAMES], train.target)
    metrics = {"data_kind": data_kind,
               "train_rows": len(train), "test_rows": len(test), "purged_rows": len(frame)-len(train)-len(test),
               "test_start": iso_us(boundary), "train_last_label_end": iso_us(int(train.label_end_us.max())),
               "train_positive_rate": float(train.target.mean()), "test_positive_rate": float(test.target.mean()),
               "model": scores(test.target, model.predict_proba(test[FEATURE_NAMES])[:, 1]),
               "prior_baseline": scores(test.target, baseline.predict_proba(test[FEATURE_NAMES])[:, 1]),
               "sklearn_version": sklearn.__version__, **metadata}
    output.mkdir(parents=True, exist_ok=True)
    # Joblib is for trusted, locally produced artifacts only.
    joblib.dump({"model": model, "metadata": metrics}, output / "model.joblib")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    frame["split"] = np.where(frame.as_of_us >= boundary, "test",
                               np.where(frame.label_end_us < boundary, "train", "purged"))
    frame["as_of"] = frame.as_of_us.map(iso_us)
    frame["label_end"] = frame.label_end_us.map(iso_us)
    frame.to_csv(output / "training_features.csv", index=False)
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
