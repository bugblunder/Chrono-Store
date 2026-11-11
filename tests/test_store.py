from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
import random
import sqlite3
import tempfile
from threading import Event
import unittest

from pydantic import ValidationError

from pitstore.contracts import FeatureQuery, Purchase, to_us, from_us
from pitstore.ingestion import IngestionBusy, IngestionWorker
from pitstore.store import EventConflict, FeatureStore

T = datetime(2024, 5, 1, tzinfo=timezone.utc)


def event(i, days=-1, amount=1000, customer="alice", **kwargs):
    return Purchase(event_id=str(i), customer_id=customer, event_time=T+timedelta(days=days),
                    amount_pence=amount, units=1, **kwargs)


def query(snapshot=None, customer="alice", at=T):
    return FeatureQuery(customer_id=customer, as_of=at, snapshot_seq=snapshot)


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/"store.sqlite3"
        self.store = FeatureStore(self.path)

    def test_half_open_boundaries_and_entity_isolation(self):
        self.store.ingest([event("a", -7), event("b", -30, 2000), event("c", -90, 3000),
                           event("old", -91), event("now", 0, 99999), event("future", 1, 99999),
                           event("other", -1, 99999, customer="bob")])
        values = self.store.features(query()).values
        self.assertEqual([values[f"purchases_{d}d"] for d in (7,30,90)], [1,2,3])
        self.assertEqual([values[f"spend_{d}d"] for d in (7,30,90)], [10,30,60])
        self.assertEqual(values["recency_days"], 7)
        self.assertEqual(values["avg_basket_90d"], 20)

    def test_microseconds_and_timezone_normalization(self):
        before = event("before").model_copy(update={"event_time": T-timedelta(microseconds=1)})
        self.store.ingest([before, event("at", 0)])
        offset_t = T.astimezone(timezone(timedelta(hours=5, minutes=30)))
        self.assertEqual(self.store.features(query(at=offset_t)).values["purchases_7d"], 1)
        self.assertEqual(from_us(to_us(before.event_time)), before.event_time)
        with self.assertRaises(ValidationError):
            FeatureQuery(customer_id="a", as_of="2024-01-01T00:00:00")
        with self.assertRaises(ValidationError):
            event("float", amount=1.5)

    def test_future_events_never_change_the_past(self):
        self.store.ingest([event("past")])
        before = self.store.features(query()).values
        self.store.ingest([event("future", 10, 999999), event("same-time", 0, 999999)])
        self.assertEqual(before, self.store.features(query()).values)

    def test_late_event_recomputation_and_snapshot_replay(self):
        first = self.store.ingest([event("past")])
        original = self.store.features(query(first.snapshot_seq))
        self.store.ingest([event("late", -2, 2000)])
        self.assertEqual(self.store.features(query()).values["spend_30d"], 30)
        self.assertEqual(self.store.features(query(first.snapshot_seq)), original)
        # Replay also works from raw events, without relying on a cached vector.
        with self.store.connection() as db:
            db.execute("DELETE FROM feature_vectors")
        self.assertEqual(self.store.features(query(first.snapshot_seq)), original)

    def test_duplicates_conflict_and_batch_rollback(self):
        purchase = event("a")
        first = self.store.ingest([purchase, purchase])
        self.assertEqual((first.inserted, first.duplicates), (1,1))
        again = self.store.ingest([purchase])
        self.assertEqual(again.snapshot_seq, first.snapshot_seq)
        with self.assertRaises(EventConflict):
            self.store.ingest([event("new"), event("a", amount=777)])
        self.assertEqual(len(self.store.export_events()[0]), 1)

    def test_unknown_customer_and_snapshot_validation(self):
        empty = self.store.features(query()).values
        self.assertEqual(empty["recency_days"], 91)
        self.assertEqual(empty["has_history_90d"], 0)
        self.assertEqual(empty["spend_90d"], 0)
        self.store.ingest([event(1),event(2)])
        for snapshot in (1, 3):  # Middle of an atomic batch, and an uncommitted future.
            with self.assertRaises(ValueError):
                self.store.features(query(snapshot))
        self.assertEqual(self.store.features(query(0)).values, empty)

    def test_restart_and_append_only_storage(self):
        result = self.store.ingest([event("a")])
        before = self.store.features(query())
        restarted = FeatureStore(self.path)
        self.assertEqual(restarted.snapshot(), result.snapshot_seq)
        self.assertEqual(restarted.features(query()), before)
        with restarted.connection() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM feature_vectors").fetchone()[0], 1)
            for command in ("DELETE FROM events", "UPDATE events SET amount_pence=1"):
                with self.assertRaises(sqlite3.IntegrityError):
                    db.execute(command)

    def test_random_out_of_order_against_independent_reference(self):
        rng = random.Random(987)
        events = [event(i, rng.uniform(-120, 20), rng.randrange(1,100000), rng.choice(["alice","bob"])) for i in range(300)]
        rng.shuffle(events)
        self.store.ingest(events)
        for delta in (-20, 0, 10):
            cutoff = T+timedelta(days=delta)
            values = self.store.features(query(at=cutoff)).values
            for days in (7,30,90):
                eligible = [e for e in events if e.customer_id=="alice" and cutoff-timedelta(days=days)<=e.event_time<cutoff]
                self.assertEqual(values[f"purchases_{days}d"], len(eligible))
                self.assertEqual(values[f"spend_{days}d"], sum(e.amount_pence for e in eligible)/100)
        other = FeatureStore(Path(self.temp.name)/"ordered.sqlite3")
        other.ingest(sorted(events, key=lambda e:e.event_time))
        self.assertEqual(self.store.features(query()).values, other.features(query()).values)

    def test_concurrent_duplicate_writes_equal_serial_reference(self):
        events = [event(i, -(i%80+1), i+100) for i in range(80)]
        stores = [FeatureStore(self.path) for _ in range(4)]
        with ThreadPoolExecutor(max_workers=12) as pool:
            results = list(pool.map(lambda pair: stores[pair[0]%4].ingest([pair[1]]), enumerate(events*3)))
        self.assertEqual(sum(r.inserted for r in results), 80)
        serial = FeatureStore(Path(self.temp.name)/"serial.sqlite3")
        serial.ingest(events)
        self.assertEqual(self.store.features(query()).values, serial.features(query()).values)
        self.assertEqual(len(self.store.export_events()[0]), 80)

    def test_read_batch_has_one_snapshot_during_concurrent_ingestion(self):
        self.store.ingest([event("old")])
        first_read, writer_done = Event(), Event()
        original = self.store._compute
        calls = 0
        def compute(db, q, snapshot):
            nonlocal calls
            value = original(db, q, snapshot)
            calls += 1
            if calls == 1:
                first_read.set()
                self.assertTrue(writer_done.wait(10))
            return value
        self.store._compute = compute
        def write():
            self.assertTrue(first_read.wait(10))
            self.store.ingest([event("new")])
            writer_done.set()
        with ThreadPoolExecutor() as pool:
            future = pool.submit(write)
            vectors = self.store.features_many([query(), query()])
            future.result()
        self.assertEqual(vectors[0], vectors[1])
        self.assertEqual(vectors[0].values["purchases_7d"], 1)
        self.assertEqual(self.store.features(query()).values["purchases_7d"], 2)

    def test_worker_backpressure_and_shutdown_drain(self):
        entered, release = Event(), Event()
        actual = self.store.ingest
        def blocked(batch):
            entered.set()
            if not release.wait(10):
                raise RuntimeError("test timed out")
            return actual(batch)
        self.store.ingest = blocked
        worker = IngestionWorker(self.store, capacity=1)
        try:
            first = worker.submit([event("a")])
            self.assertTrue(entered.wait(10))
            second = worker.submit([event("b")])
            self.assertFalse(first.cancel())
            with self.assertRaises(IngestionBusy):
                worker.submit([event("c")])
            release.set()
            worker.close()
            self.assertEqual(first.result().inserted+second.result().inserted, 2)
            with self.assertRaises(IngestionBusy):
                worker.submit([event("d")])
        finally:
            release.set()
            worker.close()


if __name__ == "__main__":
    unittest.main()
