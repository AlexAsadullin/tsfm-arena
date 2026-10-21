"""UTC time helpers. Storage is int64 nanoseconds since the epoch; the wire format is ISO-8601."""

from __future__ import annotations

from datetime import UTC, datetime

NS_PER_SECOND = 1_000_000_000
NS_PER_MICROSECOND = 1_000


def ensure_utc(value: datetime) -> datetime:
    """Return ``value`` in UTC; naive datetimes are rejected, never guessed."""
    if value.tzinfo is None:
        raise ValueError(f"naive datetime {value!r}: timestamps must be timezone-aware")
    return value.astimezone(UTC)


def to_ns(value: datetime) -> int:
    """Convert an aware datetime to int64 nanoseconds since the epoch (microsecond precision)."""
    delta = ensure_utc(value) - datetime(1970, 1, 1, tzinfo=UTC)
    micros = (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds
    return micros * NS_PER_MICROSECOND


def from_ns(ns: int) -> datetime:
    """Convert int64 nanoseconds since the epoch to an aware UTC datetime."""
    seconds, rem = divmod(int(ns), NS_PER_SECOND)
    return datetime.fromtimestamp(seconds, tz=UTC).replace(microsecond=rem // NS_PER_MICROSECOND)


def iso(ns: int) -> str:
    """ISO-8601 string for a nanosecond timestamp."""
    return from_ns(ns).isoformat().replace("+00:00", "Z")


def utc_now() -> datetime:
    return datetime.now(tz=UTC)
