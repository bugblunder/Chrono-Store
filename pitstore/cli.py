import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import random
import sys
import time

from .contracts import FeatureQuery, Purchase
from .data import download_uci, read_retail, synthetic_events
from .store import FeatureStore
from .training import train_model


def emit(value):
    print(json.dumps(value, indent=2, default=str, allow_nan=False))


def leakage_demo():
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        store = FeatureStore(Path(directory)/"demo.sqlite3")
        cutoff = datetime(2024, 1, 15, tzinfo=timezone.utc)
        def purchase(event_id, day, amount):
            return Purchase(event_id=event_id, customer_id="alice", event_time=cutoff+timedelta(days=day),
                            amount_pence=amount, units=1)
        first = store.ingest([purchase("past", -5, 1000)])
        query = FeatureQuery(customer_id="alice", as_of=cutoff)
        before = store.features(query)
        store.ingest([purchase("future", 5, 99900), purchase("exact-cutoff", 0, 88800)])
        after_future = store.features(query)
        store.ingest([purchase("late-past", -2, 2000)])
        corrected = store.features(query)
        replay = store.features(query.model_copy(update={"snapshot_seq": first.snapshot_seq}))
        assert before.values == after_future.values == replay.values
        assert corrected.values["spend_30d"] == 30.0
        emit({"past_only_spend": before.values["spend_30d"],
              "after_future_and_equal_time_events": after_future.values["spend_30d"],
              "after_late_past_event": corrected.values["spend_30d"],
              "replayed_original_snapshot": replay.values["spend_30d"], "checks_passed": True})


def main(argv=None):
    parser = argparse.ArgumentParser(prog="chronostore")
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download", help="Download the public UCI workbook")
    download.add_argument("--output", type=Path, default=Path("data/raw"))
    importer = sub.add_parser("import-uci", help="Normalize invoice rows and ingest purchases")
    importer.add_argument("file", type=Path)
    importer.add_argument("--db", type=Path, default=Path("data/store.sqlite3"))
    importer.add_argument("--source-timezone", default="Europe/London")
    importer.add_argument("--shuffle", action="store_true", help="Deliberately ingest out of event-time order")
    importer.add_argument("--report", type=Path, default=Path("artifacts/import_report.json"))
    demo = sub.add_parser("demo", help="Seed synthetic data and train a smoke-test model")
    demo.add_argument("--db", type=Path, default=Path("data/demo.sqlite3"))
    demo.add_argument("--output", type=Path, default=Path("artifacts/demo"))
    demo.add_argument("--bootstrap-resamples", type=int, default=2000)
    demo.add_argument("--bootstrap-seed", type=int, default=42)
    train = sub.add_parser("train")
    train.add_argument("--db", type=Path, default=Path("data/store.sqlite3"))
    train.add_argument("--output", type=Path, default=Path("artifacts/uci"))
    train.add_argument("--horizon-days", type=int, default=30)
    train.add_argument("--warmup-days", type=int, default=90)
    train.add_argument("--observation-end", type=datetime.fromisoformat)
    train.add_argument("--test-fraction", type=float, default=0.25)
    train.add_argument("--bootstrap-resamples", type=int, default=2000)
    train.add_argument("--bootstrap-seed", type=int, default=42)
    train.add_argument("--confidence", type=float, default=0.95)
    features = sub.add_parser("features", help="Retrieve one reproducible feature vector")
    features.add_argument("--db", type=Path, default=Path("data/store.sqlite3"))
    features.add_argument("--customer", required=True)
    features.add_argument("--as-of", type=datetime.fromisoformat, required=True)
    features.add_argument("--snapshot", type=int)
    offline = sub.add_parser("offline", help="Materialize JSONL query rows into a feature CSV")
    offline.add_argument("queries", type=Path)
    offline.add_argument("--db", type=Path, default=Path("data/store.sqlite3"))
    offline.add_argument("--output", type=Path, default=Path("artifacts/offline_features.csv"))
    serve = sub.add_parser("serve")
    serve.add_argument("--db", type=Path, default=Path("data/store.sqlite3"))
    serve.add_argument("--model", type=Path)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    sub.add_parser("leakage-demo", help="Demonstrate future-event exclusion and late-event replay")
    args = parser.parse_args(argv)
    try:
        if args.command == "download":
            emit({"workbook": str(download_uci(args.output))})
        elif args.command == "import-uci":
            events, report = read_retail(args.file, args.source_timezone)
            if not events:
                raise ValueError("no eligible purchases in input")
            if args.shuffle:
                random.Random(42).shuffle(events)
            result = FeatureStore(args.db).ingest(events)
            report.update(result.model_dump())
            report["shuffled"] = args.shuffle
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2))
            emit(report)
        elif args.command == "demo":
            store = FeatureStore(args.db)
            if store.snapshot():
                raise ValueError("demo requires an empty database; choose a new --db path")
            events = synthetic_events()
            random.Random(42).shuffle(events)
            result = store.ingest(events)
            print(f"Ingested {result.inserted} SYNTHETIC purchases in shuffled order.", file=sys.stderr)
            emit(train_model(store, args.output, data_kind="synthetic_demo",
                             bootstrap_resamples=args.bootstrap_resamples, bootstrap_seed=args.bootstrap_seed))
        elif args.command == "train":
            started = time.perf_counter()
            metrics = train_model(FeatureStore(args.db), args.output, horizon_days=args.horizon_days,
                                  warmup_days=args.warmup_days, observation_end=args.observation_end,
                                  test_fraction=args.test_fraction, bootstrap_resamples=args.bootstrap_resamples,
                                  bootstrap_seed=args.bootstrap_seed, confidence=args.confidence)
            emit({**metrics, "elapsed_seconds": round(time.perf_counter()-started, 3)})
        elif args.command == "features":
            result = FeatureStore(args.db).features(FeatureQuery(customer_id=args.customer, as_of=args.as_of,
                                                                snapshot_seq=args.snapshot))
            emit(result.model_dump())
        elif args.command == "offline":
            import pandas as pd
            queries = [FeatureQuery.model_validate_json(line) for line in args.queries.read_text().splitlines() if line.strip()]
            vectors = FeatureStore(args.db).features_many(queries)
            records = [{"customer_id": v.customer_id, "as_of": v.as_of, "snapshot_seq": v.snapshot_seq,
                        "feature_version": v.feature_version, **v.values} for v in vectors]
            args.output.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(records).to_csv(args.output, index=False)
            emit({"rows": len(records), "output": str(args.output)})
        elif args.command == "serve":
            import uvicorn
            from .api import create_app
            uvicorn.run(create_app(args.db, args.model), host=args.host, port=args.port)
        else:
            leakage_demo()
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
