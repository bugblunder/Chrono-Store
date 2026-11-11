"""UCI line-item normalization and an explicitly synthetic, deterministic demo."""
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path
import random
import shutil
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd

from .contracts import Purchase

UCI_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
UCI_PAGE = "https://archive.ics.uci.edu/dataset/352/online+retail"


def download_uci(destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "online-retail.zip"
    temporary = destination / "online-retail.zip.part"
    try:
        with urlopen(UCI_URL, timeout=60) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        with ZipFile(temporary) as bundle:
            name = "Online Retail.xlsx"
            with bundle.open(name) as source, (destination / name).open("wb") as output:
                shutil.copyfileobj(source, output)
        temporary.replace(archive)
    finally:
        temporary.unlink(missing_ok=True)
    (destination / "source.json").write_text(json.dumps({
        "source_url": UCI_URL, "dataset_page": UCI_PAGE,
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2))
    return destination / "Online Retail.xlsx"


def normalize_retail(frame: pd.DataFrame, source_timezone: str = "Europe/London") -> tuple[list[Purchase], dict]:
    required = {"InvoiceNo", "CustomerID", "InvoiceDate", "Quantity", "UnitPrice"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing UCI columns: {sorted(missing)}")
    df = frame.copy()
    report = {"input_rows": len(df), "source_timezone_assumption": source_timezone}
    df["InvoiceNo"] = df.InvoiceNo.astype("string").str.strip()
    df["CustomerID"] = df.CustomerID.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    invalid_identity = df.InvoiceNo.isna() | df.InvoiceNo.eq("") | df.CustomerID.isna() | df.CustomerID.eq("")
    # Drop the entire invoice if any of its lines lacks a customer. Never silently
    # turn an incomplete basket into a supposedly complete purchase.
    bad_invoices = set(df.loc[invalid_identity, "InvoiceNo"].dropna())
    identity_drop = invalid_identity | df.InvoiceNo.isin(bad_invoices)
    report["rows_without_complete_identity"] = int(identity_drop.sum())
    df = df.loc[~identity_drop].copy()
    cancelled = df.InvoiceNo.str.upper().str.startswith("C")
    report["cancellation_rows"] = int(cancelled.sum())
    df = df.loc[~cancelled].copy()
    df["Quantity"] = pd.to_numeric(df.Quantity, errors="raise")
    df["UnitPrice"] = pd.to_numeric(df.UnitPrice, errors="raise")
    positive = df.Quantity.gt(0) & df.UnitPrice.gt(0)
    report["nonpositive_or_missing_value_rows"] = int((~positive).sum())
    df = df.loc[positive].copy()
    if not (df.Quantity % 1 == 0).all():
        raise ValueError("purchase quantities must be integers")
    times = pd.to_datetime(df.InvoiceDate, errors="raise")
    if times.isna().any():
        raise ValueError("InvoiceDate cannot be missing")
    if times.dt.tz is None:
        times = times.dt.tz_localize(source_timezone, ambiguous="raise", nonexistent="raise")
    df["time"] = times.dt.tz_convert("UTC")
    df["pence"] = [int((Decimal(str(q))*Decimal(str(p))*100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
                   for q, p in zip(df.Quantity, df.UnitPrice)]
    events = []
    report["multi_timestamp_invoices"] = 0
    report["max_invoice_span_seconds"] = 0.0
    for invoice, group in df.groupby("InvoiceNo", sort=True):
        if group.CustomerID.nunique() != 1:
            raise ValueError(f"invoice {invoice} has inconsistent customer IDs")
        if group.time.nunique() > 1:
            report["multi_timestamp_invoices"] += 1
            span = (group.time.max()-group.time.min()).total_seconds()
            report["max_invoice_span_seconds"] = max(report["max_invoice_span_seconds"], span)
        amount = int(group.pence.sum())
        if amount <= 0:
            continue
        events.append(Purchase(event_id=f"uci:{invoice}", customer_id=str(group.CustomerID.iloc[0]),
                               # A basket cannot be available before its last line.
                               event_time=group.time.max().to_pydatetime(),
                               amount_pence=amount, units=int(group.Quantity.sum())))
    events.sort(key=lambda event: (event.event_time, event.event_id))
    report.update({"eligible_line_rows": len(df), "purchase_events": len(events),
                   "duplicate_line_policy": "keep; repeated identical lines may be legitimate",
                   "amount_policy": "positive purchase lines only; returns are excluded",
                   "invoice_time_policy": "latest eligible line timestamp (invoice completion)",
                   "ingestion_history": "not provided by UCI; assigned locally during import"})
    return events, report


def read_retail(path: Path, source_timezone: str = "Europe/London") -> tuple[list[Purchase], dict]:
    dtypes = {"InvoiceNo": "string", "CustomerID": "string"}
    if path.suffix.lower() == ".xlsx":
        frame = pd.read_excel(path, dtype=dtypes)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, dtype=dtypes)
    else:
        raise ValueError("input must be a UCI .xlsx or .csv file")
    events, report = normalize_retail(frame, source_timezone)
    report["input_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return events, report


def synthetic_events(seed: int = 42, customers: int = 180, days: int = 420) -> list[Purchase]:
    """Artificial customer propensities; metrics on this are smoke tests only."""
    rng = random.Random(seed)
    start = datetime(2023, 1, 1, tzinfo=timezone.utc)
    events = []
    for customer in range(customers):
        probability = rng.choice([0.012, 0.025, 0.06, 0.13])
        for day in range(days):
            if rng.random() < probability:
                events.append(Purchase(
                    event_id=f"demo:{customer}:{day}", customer_id=f"C{customer:04d}",
                    event_time=start+timedelta(days=day, hours=12, minutes=rng.randrange(60)),
                    amount_pence=rng.randrange(500, 25000), units=rng.randrange(1, 20),
                ))
    return sorted(events, key=lambda e: (e.event_time, e.event_id))
