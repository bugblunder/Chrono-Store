"""Bounded admission and a single durable writer for concurrent API clients."""
from concurrent.futures import Future
from queue import Full, Queue
from threading import Lock, Thread

from .contracts import Purchase
from .store import FeatureStore


class IngestionBusy(RuntimeError):
    pass


class IngestionWorker:
    def __init__(self, store: FeatureStore, capacity: int = 128):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.store = store
        self.queue = Queue(maxsize=capacity)
        self.lock = Lock()
        self.closed = False
        self.thread = Thread(target=self._run, name="feature-store-writer", daemon=True)
        self.thread.start()

    def submit(self, events: list[Purchase]) -> Future:
        future = Future()
        # Once admitted, a request is durable work even if its client disconnects.
        future.set_running_or_notify_cancel()
        with self.lock:
            if self.closed:
                raise IngestionBusy("ingestion is shutting down")
            try:
                self.queue.put_nowait((tuple(events), future))
            except Full as error:
                raise IngestionBusy("ingestion queue is full; retry with the same event IDs") from error
        return future

    def _run(self):
        while True:
            job = self.queue.get()
            try:
                if job is None:
                    return
                events, future = job
                # Accepted writes survive caller cancellation; success means
                # committed, and retrying an unacknowledged write is idempotent.
                try:
                    result = self.store.ingest(events)
                except Exception as error:
                    if not future.cancelled():
                        future.set_exception(error)
                else:
                    if not future.cancelled():
                        future.set_result(result)
            finally:
                self.queue.task_done()

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
        self.queue.put(None)
        self.thread.join()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
