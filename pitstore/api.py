"""Run with: python -m pitstore serve --db data/store.sqlite3."""
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import Field

from . import __version__
from .contracts import Contract, FEATURE_NAMES, FEATURE_VERSION, FeatureQuery, FeatureVector, IngestResult, Purchase
from .ingestion import IngestionBusy, IngestionWorker
from .store import EventConflict, FeatureStore
from .training import load_model, predict


class EventBatch(Contract):
    events: list[Purchase] = Field(min_length=1, max_length=1000)


class QueryBatch(Contract):
    queries: list[FeatureQuery] = Field(min_length=1, max_length=1000)


def create_app(db_path: str | Path = "data/store.sqlite3", model_path: str | Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        app.state.store = FeatureStore(db_path)
        app.state.model = load_model(Path(model_path)) if model_path else None
        app.state.writer = IngestionWorker(app.state.store)
        try:
            yield
        finally:
            await asyncio.to_thread(app.state.writer.close)

    app = FastAPI(title="Point-in-time Feature Store", version=__version__, lifespan=lifespan,
                  description="Strict past-only features with immutable, replayable database snapshots.")

    @app.get("/health")
    def health():
        return {"status": "ok", "snapshot_seq": app.state.store.snapshot(),
                "model_loaded": app.state.model is not None}

    @app.get("/schema")
    def schema():
        return {"feature_version": FEATURE_VERSION, "feature_names": FEATURE_NAMES,
                "window": "[as_of - window, as_of)", "currency": "GBP",
                "unknown_customer": "zero counts/spend, recency_days=91, has_history_90d=0"}

    @app.post("/events", response_model=IngestResult)
    async def events(batch: EventBatch):
        try:
            future = app.state.writer.submit(batch.events)
            return await asyncio.shield(asyncio.wrap_future(future))
        except IngestionBusy as error:
            raise HTTPException(503, str(error), headers={"Retry-After": "1"}) from error
        except EventConflict as error:
            raise HTTPException(409, str(error)) from error

    def read_features(queries):
        try:
            return app.state.store.features_many(queries)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @app.post("/features", response_model=FeatureVector)
    def features(query: FeatureQuery):
        return read_features([query])[0]

    @app.post("/features/batch", response_model=list[FeatureVector])
    def features_batch(batch: QueryBatch):
        return read_features(batch.queries)

    @app.post("/predict")
    def prediction(query: FeatureQuery):
        if app.state.model is None:
            raise HTTPException(503, "Train a model, then start the server with --model PATH")
        vector = read_features([query])[0]
        return {"probability": predict(app.state.model, vector), "target": "repeat_purchase",
                "target_condition": "given a positive purchase now; not general churn",
                "model_name": app.state.model["metadata"].get("served_model", "logistic_regression"),
                "horizon_days": app.state.model["metadata"]["horizon_days"],
                "model_trained_snapshot_seq": app.state.model["metadata"]["snapshot_seq"],
                "features": vector.model_dump()}

    return app
