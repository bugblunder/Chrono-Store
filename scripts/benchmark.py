"""Small local benchmark; run from the project root, after importing data.

python scripts/benchmark.py --db data/store.sqlite3 --output artifacts/benchmark.json
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import platform
import statistics
import sys
import tempfile
import time

# Also supports running from a source checkout before editable installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pitstore.contracts import FeatureQuery, Purchase, from_us
from pitstore.ingestion import IngestionWorker
from pitstore.store import FeatureStore


def percentiles(values):
    ordered = sorted(values)
    return {"p50_ms": statistics.median(ordered)*1000,
            "p95_ms": ordered[int((len(ordered)-1)*0.95)]*1000,
            "samples": len(ordered)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db",type=Path,default=Path("data/store.sqlite3"))
    parser.add_argument("--output",type=Path,default=Path("artifacts/benchmark.json"))
    args = parser.parse_args()
    store = FeatureStore(args.db)
    rows, snapshot = store.export_events()
    if not rows:
        parser.error("ingest a dataset first")
    selected = rows[::max(1,len(rows)//200)][:200]
    queries = [FeatureQuery(customer_id=r["customer_id"], as_of=from_us(r["event_time_us"]+1),
                            snapshot_seq=snapshot) for r in selected]
    cold = []
    # Different cutoffs from training; read+compute+persist, then cache hits.
    for query in queries:
        start = time.perf_counter()
        store.features(query)
        cold.append(time.perf_counter()-start)
    warm = []
    for query in queries:
        start = time.perf_counter()
        store.features(query)
        warm.append(time.perf_counter()-start)
    with tempfile.TemporaryDirectory() as directory:
        writes = FeatureStore(Path(directory)/"writes.sqlite3")
        now = datetime(2024,1,1,tzinfo=timezone.utc)
        events = [Purchase(event_id=str(i),customer_id=str(i%50),event_time=now+timedelta(seconds=i),
                            amount_pence=100,units=1) for i in range(1000)]
        with IngestionWorker(writes) as worker, ThreadPoolExecutor(max_workers=8) as pool:
            start = time.perf_counter()
            results = list(pool.map(lambda event: worker.submit([event]).result(), events))
            elapsed = time.perf_counter()-start
        assert sum(r.inserted for r in results)==1000
    result = {"scope":"local store calls; includes SQLite connections and persistence; not HTTP/network latency",
              "python":platform.python_version(),"platform":platform.platform(),"dataset_events":len(rows),
              "first_pass":percentiles(cold),"cached_pass":percentiles(warm),
              "concurrent_single_event_writes":{"clients":8,"events":1000,"elapsed_seconds":elapsed,
                                                 "events_per_second":1000/elapsed},
              "caveat":"single run on this host; not a production capacity estimate; first pass may be cached on rerun"}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
