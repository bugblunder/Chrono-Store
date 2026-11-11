"""The versioned public data contract shared by every entry point."""
from datetime import datetime, timedelta, timezone

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator

EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
DAY_US = 86_400_000_000
FEATURE_VERSION = "retail-v1"
FEATURE_NAMES = [
    "purchases_7d", "spend_7d", "purchases_30d", "spend_30d",
    "purchases_90d", "spend_90d", "avg_basket_90d", "recency_days", "has_history_90d",
]


def to_us(value: datetime) -> int:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must include an explicit timezone")
    delta = value.astimezone(timezone.utc) - EPOCH
    return (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds


def from_us(value: int) -> datetime:
    return EPOCH + timedelta(microseconds=value)


def iso_us(value: int) -> str:
    return from_us(value).isoformat().replace("+00:00", "Z")


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Purchase(Contract):
    """One complete positive purchase, never one invoice line."""
    event_id: str = Field(min_length=1, max_length=128)
    customer_id: str = Field(min_length=1, max_length=128)
    event_time: AwareDatetime
    amount_pence: int = Field(strict=True, gt=0, le=10**12)
    units: int = Field(strict=True, gt=0, le=10**9)

    @field_validator("event_id", "customer_id")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip() or value != value.strip():
            raise ValueError("IDs must be nonblank and have no surrounding whitespace")
        return value

    def payload(self) -> tuple:
        return self.customer_id, to_us(self.event_time), self.amount_pence, self.units


class FeatureQuery(Contract):
    customer_id: str = Field(min_length=1, max_length=128)
    as_of: AwareDatetime
    snapshot_seq: int | None = Field(default=None, strict=True, ge=0)


class FeatureVector(Contract):
    customer_id: str
    as_of: str
    snapshot_seq: int
    feature_version: str = FEATURE_VERSION
    values: dict[str, float | int]


class IngestResult(Contract):
    inserted: int
    duplicates: int
    snapshot_seq: int
