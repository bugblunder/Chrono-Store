from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from pitstore.data import normalize_retail, read_retail


def row(invoice="1", customer="12345", time="2011-06-01 12:00:00", quantity=2, price=1.25):
    return {"InvoiceNo": invoice, "CustomerID": customer, "InvoiceDate": time,
            "Quantity": quantity, "UnitPrice": price}


class DataTests(unittest.TestCase):
    def test_invoice_aggregation_rounding_and_summer_timezone(self):
        events, report = normalize_retail(pd.DataFrame([row(), row(quantity=1, price=0.333)]))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].amount_pence, 283)
        self.assertEqual(events[0].units, 3)
        self.assertEqual(events[0].event_time, datetime(2011,6,1,11,tzinfo=timezone.utc))
        self.assertEqual(report["eligible_line_rows"], 2)

    def test_cleaning_policy_excludes_cancellations_and_incomplete_invoices(self):
        events, report = normalize_retail(pd.DataFrame([
            row(), row(invoice="C1"), row(invoice="2", quantity=-2),
            row(invoice="3", price=0), row(invoice="4", customer=None), row(invoice="4"),
        ]))
        self.assertEqual([e.event_id for e in events], ["uci:1"])
        self.assertEqual(report["rows_without_complete_identity"], 2)
        self.assertEqual(report["cancellation_rows"], 1)
        self.assertEqual(report["nonpositive_or_missing_value_rows"], 2)

    def test_inconsistent_invoice_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            normalize_retail(pd.DataFrame([row(), row(customer="999")]))

    def test_multi_timestamp_invoice_uses_completion_time(self):
        from pitstore.store import FeatureStore
        from pitstore.contracts import FeatureQuery
        events, report = normalize_retail(pd.DataFrame([row(), row(time="2011-06-01 13:00:00")]))
        self.assertEqual(events[0].event_time, datetime(2011,6,1,12,tzinfo=timezone.utc))
        self.assertEqual(report["multi_timestamp_invoices"], 1)
        self.assertEqual(report["max_invoice_span_seconds"], 3600)
        with tempfile.TemporaryDirectory() as directory:
            store = FeatureStore(Path(directory)/"invoice.sqlite3")
            store.ingest(events)
            vector = store.features(FeatureQuery(customer_id="12345", as_of="2011-06-01T11:30:00Z"))
            self.assertEqual(vector.values["spend_7d"], 0)

    def test_csv_and_excel_import_are_equivalent(self):
        with tempfile.TemporaryDirectory() as directory:
            csv, xlsx = Path(directory)/"source.csv", Path(directory)/"source.xlsx"
            frame = pd.DataFrame([row(customer=12345.0),row(invoice="2", quantity=4)])
            frame.to_csv(csv,index=False)
            frame.to_excel(xlsx,index=False)
            self.assertEqual(read_retail(csv)[0], read_retail(xlsx)[0])

    def test_repeated_lines_preserved_and_missing_columns_rejected(self):
        events, _ = normalize_retail(pd.DataFrame([row(),row()]))
        self.assertEqual(events[0].amount_pence, 500)
        with self.assertRaisesRegex(ValueError,"missing UCI columns"):
            normalize_retail(pd.DataFrame({"wrong": [1]}))


if __name__ == "__main__":
    unittest.main()
