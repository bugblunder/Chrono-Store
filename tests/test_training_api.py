from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient
import pandas as pd

from pitstore.api import create_app
from pitstore.contracts import FEATURE_NAMES, FeatureQuery, Purchase, from_us
from pitstore.data import synthetic_events
from pitstore.store import FeatureStore
from pitstore.training import build_training_frame, load_model, predict, train_model


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.temp.name)/"store.sqlite3"
        cls.store = FeatureStore(cls.path)
        cls.store.ingest(synthetic_events(customers=65))
        cls.output = Path(cls.temp.name)/"artifacts"
        cls.metrics = train_model(cls.store, cls.output, data_kind="synthetic_demo")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_training_label_maturity_purge_and_artifacts(self):
        frame = pd.read_csv(self.output/"training_features.csv")
        train, test = frame[frame.split=="train"], frame[frame.split=="test"]
        self.assertLess(train.label_end_us.max(), test.as_of_us.min())
        self.assertLessEqual(pd.to_datetime(frame.label_end,utc=True).max(), pd.Timestamp(self.metrics["observation_end"]))
        self.assertGreater(self.metrics["purged_rows"],0)
        self.assertEqual(self.metrics["availability_mode"], "reconstructed_event_time")
        self.assertGreaterEqual(self.metrics["model"]["roc_auc"], 0)

    def test_offline_features_equal_served_features_and_prediction(self):
        frame = pd.read_csv(self.output/"training_features.csv")
        artifact = load_model(self.output/"model.joblib")
        with TestClient(create_app(self.path, self.output/"model.joblib")) as client:
            for _, row in frame.iloc[::max(1,len(frame)//8)].iterrows():
                request = {"customer_id":row.customer_id, "as_of":row.as_of, "snapshot_seq":int(row.snapshot_seq)}
                response = client.post("/features",json=request)
                self.assertEqual(response.status_code,200)
                for name in FEATURE_NAMES:
                    self.assertAlmostEqual(response.json()["values"][name], row[name])
                vector = self.store.features(FeatureQuery(**request))
                prediction = client.post("/predict",json=request)
                self.assertEqual(prediction.status_code,200)
                self.assertAlmostEqual(prediction.json()["probability"], predict(artifact,vector))

    def test_labels_exclude_same_timestamp_include_horizon_and_censor_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            store = FeatureStore(Path(directory)/"labels.sqlite3")
            start = datetime(2024,1,1,tzinfo=timezone.utc)
            events = [Purchase(event_id=str(i), customer_id=c, event_time=start+timedelta(days=d),
                               amount_pence=100,units=1)
                      for i,(c,d) in enumerate([("a",0),("a",0),("a",30),("b",0),("b",31),("c",100)])]
            store.ingest(events)
            frame,_ = build_training_frame(store,warmup_days=0)
            self.assertEqual(len(frame[(frame.customer_id=="a") & (frame.as_of_us==frame.as_of_us.min())]),1)
            a = frame[frame.customer_id=="a"].sort_values("as_of_us")
            b = frame[frame.customer_id=="b"].sort_values("as_of_us")
            self.assertEqual(a.target.tolist(),[1,0])
            self.assertEqual(b.target.tolist(),[0,0])
            self.assertNotIn("c", frame.customer_id.tolist())

    def test_empty_history_fails_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError,"ingest events"):
                build_training_frame(FeatureStore(Path(directory)/"empty.sqlite3"))


class APITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/"api.sqlite3"
        self.client = self.enterContext(TestClient(create_app(self.path)))

    def purchase(self,i=1):
        return {"event_id":str(i),"customer_id":"alice","event_time":"2024-01-01T12:00:00Z",
                "amount_pence":1000,"units":2}

    def test_validation_conflict_duplicate_and_model_unavailable(self):
        event = self.purchase()
        result = self.client.post("/events",json={"events":[event]})
        self.assertEqual(result.status_code,200)
        self.assertEqual(self.client.post("/events",json={"events":[event]}).json()["duplicates"],1)
        self.assertEqual(self.client.post("/events",json={"events":[{**event,"amount_pence":222}]}).status_code,409)
        self.assertEqual(self.client.post("/events",json={"events":[{**event,"ingested_at":"2020-01-01"}]}).status_code,422)
        self.assertEqual(self.client.post("/events",json={"events":[]}).status_code,422)
        q = {"customer_id":"alice","as_of":"2024-01-02T00:00:00Z"}
        self.assertEqual(self.client.post("/predict",json=q).status_code,503)
        self.assertEqual(self.client.post("/features",json={**q,"snapshot_seq":999}).status_code,422)
        self.assertEqual(self.client.post("/features",json={**q,"as_of":"2024-01-02"}).status_code,422)
        values = self.client.post("/features/batch",json={"queries":[q,q]}).json()
        self.assertEqual(values[0],values[1])
        self.assertEqual(values[0]["values"]["spend_7d"],10)
        self.assertEqual(self.client.get("/schema").json()["feature_names"], FEATURE_NAMES)

    def test_concurrent_http_writes_commit_before_ack(self):
        def post(i):
            return self.client.post("/events",json={"events":[self.purchase(i)]})
        with ThreadPoolExecutor(max_workers=12) as pool:
            responses = list(pool.map(post,list(range(30))*2))
        self.assertTrue(all(r.status_code==200 for r in responses))
        self.assertEqual(sum(r.json()["inserted"] for r in responses),30)
        response = self.client.post("/features",json={"customer_id":"alice","as_of":"2024-01-02T00:00:00Z"})
        self.assertEqual(response.json()["values"]["purchases_7d"],30)
        self.assertEqual(self.client.get("/health").json()["snapshot_seq"],30)


if __name__ == "__main__":
    unittest.main()
