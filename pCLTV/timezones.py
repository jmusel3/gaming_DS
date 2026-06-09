"""IANA timezone maps and UTC/local conversion helpers for pCLTV generators."""

from __future__ import annotations

import datetime
from zoneinfo import ZoneInfo

import numpy as np

COUNTRY_HOME_TIMEZONES: dict[str, list[tuple[str, float]]] = {
    "US": [
        ("America/New_York", 0.35),
        ("America/Chicago", 0.25),
        ("America/Denver", 0.10),
        ("America/Los_Angeles", 0.20),
        ("America/Phoenix", 0.04),
        ("America/Anchorage", 0.04),
        ("Pacific/Honolulu", 0.02),
    ],
    "CA": [
        ("America/Toronto", 0.45),
        ("America/Vancouver", 0.25),
        ("America/Edmonton", 0.12),
        ("America/Winnipeg", 0.08),
        ("America/Halifax", 0.07),
        ("America/St_Johns", 0.03),
    ],
    "BR": [
        ("America/Sao_Paulo", 0.55),
        ("America/Manaus", 0.10),
        ("America/Fortaleza", 0.10),
        ("America/Cuiaba", 0.08),
        ("America/Porto_Velho", 0.05),
        ("America/Rio_Branco", 0.05),
        ("America/Noronha", 0.07),
    ],
    "MX": [
        ("America/Mexico_City", 0.55),
        ("America/Cancun", 0.15),
        ("America/Tijuana", 0.12),
        ("America/Mazatlan", 0.10),
        ("America/Hermosillo", 0.08),
    ],
    "JP": [("Asia/Tokyo", 1.0)],
    "KR": [("Asia/Seoul", 1.0)],
    "GB": [("Europe/London", 1.0)],
    "DE": [("Europe/Berlin", 1.0)],
    "FR": [("Europe/Paris", 1.0)],
    "IN": [("Asia/Kolkata", 1.0)],
}

EVENING_HOUR_WEIGHTS = [0.10, 0.12, 0.15, 0.18, 0.25, 0.20]


def draw_home_timezone(country: str, rng: np.random.Generator) -> str:
    zones = COUNTRY_HOME_TIMEZONES[country]
    names = [name for name, _ in zones]
    weights = np.array([weight for _, weight in zones], dtype=float)
    weights /= weights.sum()
    return str(rng.choice(names, p=weights))


def _ensure_utc(dt: datetime.datetime) -> datetime.datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=datetime.timezone.utc)
    return dt.astimezone(datetime.timezone.utc)


def utc_to_local_naive(utc: datetime.datetime, tz_name: str) -> datetime.datetime:
    utc_aware = _ensure_utc(utc)
    local = utc_aware.astimezone(ZoneInfo(tz_name))
    return local.replace(tzinfo=None)


def local_naive_to_utc(local: datetime.datetime, tz_name: str) -> datetime.datetime:
    aware = local.replace(tzinfo=ZoneInfo(tz_name))
    return aware.astimezone(datetime.timezone.utc)


def assert_utc_local_invariant(
    utc: datetime.datetime,
    local: datetime.datetime,
    tz_name: str,
) -> None:
    expected = utc_to_local_naive(utc, tz_name)
    if local != expected:
        raise ValueError(
            f"UTC/local invariant failed for {tz_name}: "
            f"expected {expected}, got {local}"
        )


def draw_evening_local_time(rng: np.random.Generator) -> tuple[int, int, int]:
    hour = int(rng.choice(np.arange(18, 24), p=EVENING_HOUR_WEIGHTS))
    minute = int(rng.integers(0, 60))
    second = int(rng.integers(0, 60))
    return hour, minute, second
