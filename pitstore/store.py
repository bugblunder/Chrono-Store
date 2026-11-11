"""Append-only events, transactionally consistent reads and versioned feature rows."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Iterable

from .contracts import (
    DAY_US, FEATURE_VERSION, FeatureQuery, FeatureVector, IngestResult,
    Purchase, iso_us, to_us,
)


class EventConflict(ValueError):
    """An event ID was reused with a different immutable payload."""


class FeatureStore:
    def __init__(self, path: str | Path):
        if str(path) == ":memory:":
            raise ValueError("use a file database: each operation owns its connection")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise ValueError(f"unsupported schema version: {version}")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    customer_id TEXT NOT NULL,
                    event_time_us INTEGER NOT NULL,
                    amount_pence INTEGER NOT NULL CHECK(amount_pence > 0),
                    units INTEGER NOT NULL CHECK(units > 0),
                    ingested_at_us INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS events_customer_time
                    ON events(customer_id, event_time_us, seq);
                CREATE TABLE IF NOT EXISTS snapshots (seq INTEGER PRIMARY KEY);
                INSERT OR IGNORE INTO snapshots VALUES(0);
                CREATE TABLE IF NOT EXISTS feature_vectors (
                    customer_id TEXT NOT NULL,
                    as_of_us INTEGER NOT NULL,
                    snapshot_seq INTEGER NOT NULL,
                    feature_version TEXT NOT NULL,
                    values_json TEXT NOT NULL,
                    PRIMARY KEY(customer_id, as_of_us, snapshot_seq, feature_version)
                );
                CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events
                    BEGIN SELECT RAISE(ABORT, 'events are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events
                    BEGIN SELECT RAISE(ABORT, 'events are immutable'); END;
                PRAGMA user_version=1;
            """)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=30000")
        db.execute("PRAGMA synchronous=FULL")
        try:
            yield db
        finally:
            db.close()

    @staticmethod
    def _head(db) -> int:
        return db.execute("SELECT COALESCE(MAX(seq),0) FROM events").fetchone()[0]

    def ingest(self, events: Iterable[Purchase]) -> IngestResult:
        batch = list(events)
        if any(not isinstance(e, Purchase) for e in batch):
            raise TypeError("ingest expects validated Purchase objects")
        inserted = 0
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                # Assigned inside the writer transaction; snapshot_seq, not this
                # wall clock timestamp, is the exact visibility/replay contract.
                received = to_us(datetime.now(timezone.utc))
                for event in batch:
                    old = db.execute(
                        "SELECT customer_id,event_time_us,amount_pence,units FROM events WHERE event_id=?",
                        (event.event_id,),
                    ).fetchone()
                    if old is not None:
                        if tuple(old) != event.payload():
                            raise EventConflict(f"event {event.event_id!r} already has a different payload")
                        continue
                    db.execute(
                        "INSERT INTO events(event_id,customer_id,event_time_us,amount_pence,units,ingested_at_us) "
                        "VALUES(?,?,?,?,?,?)", (event.event_id, *event.payload(), received),
                    )
                    inserted += 1
                head = self._head(db)
                db.execute("INSERT OR IGNORE INTO snapshots VALUES(?)", (head,))
                db.commit()
            except BaseException:
                db.rollback()
                raise
        return IngestResult(inserted=inserted, duplicates=len(batch)-inserted, snapshot_seq=head)

    def snapshot(self) -> int:
        with self.connection() as db:
            return self._head(db)

    @staticmethod
    def _snapshot(requested: int | None, head: int, db) -> int:
        if requested is None:
            return head
        if type(requested) is not int or not 0 <= requested <= head:
            raise ValueError(f"snapshot_seq must be between 0 and current head {head}")
        if db.execute("SELECT 1 FROM snapshots WHERE seq=?", (requested,)).fetchone() is None:
            raise ValueError("snapshot_seq must be a committed batch boundary returned by this database")
        return requested

    @staticmethod
    def _compute(db, query: FeatureQuery, snapshot: int) -> FeatureVector:
        cutoff = to_us(query.as_of)
        cached = db.execute(
            "SELECT values_json FROM feature_vectors WHERE customer_id=? AND as_of_us=? "
            "AND snapshot_seq=? AND feature_version=?",
            (query.customer_id, cutoff, snapshot, FEATURE_VERSION),
        ).fetchone()
        if cached:
            values = json.loads(cached[0])
        else:
            # Half-open [t-window, t). Equal timestamps are ALWAYS excluded.
            rows = db.execute(
                "SELECT event_time_us,amount_pence FROM events "
                "WHERE customer_id=? AND event_time_us>=? AND event_time_us<? AND seq<=? "
                "ORDER BY event_time_us,seq",
                (query.customer_id, cutoff-90*DAY_US, cutoff, snapshot),
            ).fetchall()
            values = {}
            for days in (7, 30, 90):
                relevant = [r for r in rows if r[0] >= cutoff-days*DAY_US]
                values[f"purchases_{days}d"] = len(relevant)
                values[f"spend_{days}d"] = sum(r[1] for r in relevant) / 100
            count = values["purchases_90d"]
            values["avg_basket_90d"] = values["spend_90d"] / count if count else 0.0
            values["recency_days"] = (cutoff-rows[-1][0])/DAY_US if rows else 91.0
            values["has_history_90d"] = int(bool(rows))
        return FeatureVector(customer_id=query.customer_id, as_of=iso_us(cutoff),
                             snapshot_seq=snapshot, values=values)

    def features(self, query: FeatureQuery, *, persist: bool = True) -> FeatureVector:
        return self.features_many([query], persist=persist)[0]

    def features_many(self, queries: Iterable[FeatureQuery], *, persist: bool = True) -> list[FeatureVector]:
        queries = list(queries)
        with self.connection() as db:
            db.execute("BEGIN")
            try:
                head = self._head(db)  # Pins the SQLite read snapshot for the entire batch.
                vectors = [self._compute(db, q, self._snapshot(q.snapshot_seq, head, db)) for q in queries]
                db.commit()
            except BaseException:
                db.rollback()
                raise
        if persist and vectors:
            # Do not upgrade an old read snapshot to a write transaction. The
            # seq predicate makes these values immutable even if ingestion raced.
            with self.connection() as db:
                db.execute("BEGIN IMMEDIATE")
                try:
                    db.executemany(
                        "INSERT OR IGNORE INTO feature_vectors VALUES(?,?,?,?,?)",
                        [(v.customer_id, to_us(queries[i].as_of), v.snapshot_seq,
                          v.feature_version, json.dumps(v.values, sort_keys=True)) for i, v in enumerate(vectors)],
                    )
                    db.commit()
                except BaseException:
                    db.rollback()
                    raise
        return vectors

    def export_events(self, snapshot_seq: int | None = None) -> tuple[list[dict], int]:
        with self.connection() as db:
            db.execute("BEGIN")
            snapshot = self._snapshot(snapshot_seq, self._head(db), db)
            rows = db.execute("SELECT * FROM events WHERE seq<=? ORDER BY event_time_us,seq", (snapshot,)).fetchall()
            db.commit()
        return [dict(r) for r in rows], snapshot
