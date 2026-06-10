"""
Synthetic Sessions fact data for a live-service collectible card game CLTV project.

v1 schema
---------
player_id, session_id, session_start_ts_utc, session_end_ts_utc,
session_timezone, session_start_ts_local, session_end_ts_local,
duration_minutes, device, login_region, local_day_of_week, session_end_reason

"""

from __future__ import annotations

import datetime
import math

import numpy as np
import pandas as pd

from pCLTV.player_latents import draw_player_latents
from pCLTV.timezones import (
    assert_utc_local_invariant,
    draw_evening_local_time,
    local_naive_to_utc,
    utc_to_local_naive,
)

REQUIRED_PLAYER_COLUMNS = [
    "player_id",
    "registration_timestamp_utc",
    "home_timezone",
    "install_platform",
    "account_status",
    "banned_datetime_utc",
    "data_region",
    "country",
]

DATA_REGIONS = ["NA", "EU", "APAC", "LATAM"]

BASE_REACTIVATION_PROB = {"loyal": 0.50, "fade": 0.20, "early": 0.08}
REACTIVATION_DECAY = 0.85

SESSION_END_REASONS = ["normal", "app_crash", "disconnected"]
SESSION_END_REASON_WEIGHTS = [0.88, 0.08, 0.04]

MAX_DURATION_MINUTES = 300.0

COLUMN_ORDER = [
    "player_id",
    "session_id",
    "session_start_ts_utc",
    "session_end_ts_utc",
    "session_timezone",
    "session_start_ts_local",
    "session_end_ts_local",
    "duration_minutes",
    "device",
    "login_region",
    "local_day_of_week",
    "session_end_reason",
]


def _validate_random_seed(random_seed: int) -> None:
    if not isinstance(random_seed, int) or isinstance(random_seed, bool):
        raise ValueError("random_seed must be an integer")


def _validate_players(players: pd.DataFrame) -> None:
    if not isinstance(players, pd.DataFrame):
        raise ValueError("players must be a pandas DataFrame")
    if players.empty:
        raise ValueError("players must not be empty")

    missing = [col for col in REQUIRED_PLAYER_COLUMNS if col not in players.columns]
    if missing:
        raise ValueError(f"players is missing required columns: {missing}")

    registration_ts = pd.to_datetime(
        players["registration_timestamp_utc"], utc=True, errors="coerce"
    )
    if registration_ts.isna().any():
        raise ValueError("registration_timestamp_utc must be parseable as UTC datetimes")

    is_banned = players["account_status"] == "banned"
    banned_ts = pd.to_datetime(players["banned_datetime_utc"], utc=True, errors="coerce")
    if is_banned.any() and banned_ts.loc[is_banned].isna().any():
        raise ValueError("banned players must have non-null banned_datetime_utc")
    if (~is_banned).any() and banned_ts.loc[~is_banned].notna().any():
        raise ValueError("non-banned players must have null banned_datetime_utc")


def _ensure_utc(dt: datetime.datetime) -> datetime.datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=datetime.timezone.utc)
    return dt.astimezone(datetime.timezone.utc)


def _to_utc_timestamp(value: pd.Timestamp | datetime.datetime) -> datetime.datetime:
    if isinstance(value, pd.Timestamp):
        return _ensure_utc(value.to_pydatetime())
    return _ensure_utc(value)


def _observation_end_datetime(observation_end: datetime.date) -> datetime.datetime:
    return datetime.datetime.combine(
        observation_end, datetime.time(23, 59, 59), tzinfo=datetime.timezone.utc
    )


def _day_of_week_multiplier(dow: int) -> float:
    if dow == 4:
        return 1.15
    if dow >= 5:
        return 1.25
    return 1.0


def _phase_end_multiplier(days_to_end: float) -> float:
    if days_to_end > 30:
        return 1.0
    if days_to_end > 7:
        return 1.0 - ((30.0 - days_to_end) / 23.0) * 0.6
    return max(0.05, 0.4 * math.exp(-(7.0 - days_to_end) / 3.0))


def _draw_active_period(
    churn_type: str, cycle: int, tenure: int, rng: np.random.Generator
) -> int:
    if cycle == 0:
        if churn_type == "loyal":
            low, high = (60, tenure) if tenure >= 60 else (7, max(7, tenure))
        elif churn_type == "fade":
            low, high = 21, min(60, max(21, tenure))
        else:
            low, high = 7, min(21, max(7, tenure))
    elif churn_type == "loyal":
        low, high = 30, 90
    elif churn_type == "fade":
        low, high = 14, 45
    else:
        low, high = 5, 14

    if low > high:
        high = low
    return int(rng.integers(low, high + 1))


def _draw_dormant_period(churn_type: str, rng: np.random.Generator) -> int:
    ranges = {"loyal": (14, 45), "fade": (21, 60), "early": (30, 90)}
    low, high = ranges[churn_type]
    return int(rng.integers(low, high + 1))


def _draw_activity_cycles(
    latents: dict,
    row: pd.Series,
    rng: np.random.Generator,
    observation_end: datetime.date,
) -> list[tuple[datetime.datetime, datetime.datetime]]:
    obs_end_dt = _observation_end_datetime(observation_end)
    registration_ts_utc = latents["registration_ts_utc"]

    if latents["banned"]:
        banned_ts = _to_utc_timestamp(row["banned_datetime_utc"])
        active_start = registration_ts_utc + datetime.timedelta(
            hours=float(rng.uniform(0, 48))
        )
        active_end = min(banned_ts, obs_end_dt)
        if active_start < active_end:
            return [(active_start, active_end)]
        return []

    cycles: list[tuple[datetime.datetime, datetime.datetime]] = []
    current = registration_ts_utc + datetime.timedelta(
        hours=float(rng.uniform(0, 48))
    )
    cycle = 0

    while current < obs_end_dt:
        active_days = _draw_active_period(
            latents["churn_type"], cycle, latents["tenure"], rng
        )
        active_end = min(
            current + datetime.timedelta(days=active_days), obs_end_dt
        )

        if current < active_end:
            cycles.append((current, active_end))

        current = active_end
        if current >= obs_end_dt:
            break

        dormant_days = _draw_dormant_period(latents["churn_type"], rng)
        dormant_end = current + datetime.timedelta(days=dormant_days)
        if dormant_end > obs_end_dt:
            break

        p_reactivate = (
            latents["base_reactivation_prob"]
            * (latents["engagement_score"] ** 0.3)
            * (REACTIVATION_DECAY**cycle)
        )
        if rng.random() >= p_reactivate:
            break

        current = dormant_end
        cycle += 1

    return cycles


def _apply_evening_hour(
    dt: datetime.datetime, rng: np.random.Generator, session_timezone: str
) -> datetime.datetime:
    local_dt = utc_to_local_naive(_ensure_utc(dt), session_timezone)
    hour, minute, second = draw_evening_local_time(rng)
    local_evening = local_dt.replace(
        hour=hour, minute=minute, second=second, microsecond=0
    )
    return local_naive_to_utc(local_evening, session_timezone)


def _draw_session_times(
    active_start: datetime.datetime,
    active_end: datetime.datetime,
    weekly_session_rate: float,
    session_timezone: str,
    rng: np.random.Generator,
    cycle: int,
) -> list[datetime.datetime]:
    if cycle == 0:
        current = active_start + datetime.timedelta(
            hours=float(rng.uniform(0, 48))
        )
    else:
        current = active_start + datetime.timedelta(
            hours=float(rng.uniform(0, 4))
        )

    current = min(current, active_end)
    if current >= active_end:
        return []

    times: list[datetime.datetime] = []
    rate_per_day = weekly_session_rate / 7.0
    mean_gap_hours = 24.0 / rate_per_day

    while current < active_end:
        session_dt = _apply_evening_hour(current, rng, session_timezone)
        if session_dt >= active_end:
            break
        times.append(session_dt)

        base_gap = float(rng.exponential(mean_gap_hours))
        tentative = current + datetime.timedelta(hours=base_gap)
        local_tentative = utc_to_local_naive(_ensure_utc(tentative), session_timezone)
        dow = local_tentative.weekday()
        w = _day_of_week_multiplier(dow)
        days_to_end = (active_end - tentative).total_seconds() / 86400.0
        m = _phase_end_multiplier(days_to_end)
        effective_gap = base_gap / (w * m)
        current = current + datetime.timedelta(hours=effective_gap)

    return times


def _draw_device(install_platform: str, rng: np.random.Generator) -> str:
    roll = float(rng.random())
    if roll < 0.80:
        return install_platform
    if roll < 0.99:
        if install_platform == "pc":
            return str(rng.choice(["ios", "android"]))
        return "pc"
    if install_platform == "ios":
        return "android"
    if install_platform == "android":
        return "ios"
    return str(rng.choice(["ios", "android"]))


def _draw_login_region(player_region: str, rng: np.random.Generator) -> str:
    if rng.random() < 0.92:
        return player_region
    others = [region for region in DATA_REGIONS if region != player_region]
    return str(rng.choice(others))


def _draw_session_end_reason(rng: np.random.Generator) -> str:
    return str(
        rng.choice(SESSION_END_REASONS, p=SESSION_END_REASON_WEIGHTS)
    )


def _draw_duration_minutes(
    duration_scale: float,
    dow: int,
    end_reason: str,
    rng: np.random.Generator,
) -> float:
    w = _day_of_week_multiplier(dow)
    mean_minutes = 12.0 * duration_scale * w
    duration = float(rng.lognormal(math.log(mean_minutes), 0.5))
    if end_reason in ("app_crash", "disconnected"):
        duration *= float(rng.uniform(0.2, 0.6))
    duration = max(2.0, min(MAX_DURATION_MINUTES, duration))
    return round(duration, 1)


def _player_id_suffix(player_id: str) -> str:
    return player_id[1:] if player_id.startswith("P") else player_id


def _format_session_id(player_id: str, seq: int) -> str:
    return f"S{_player_id_suffix(player_id)}_{seq:04d}"


def _draw_session_row(
    player_row: pd.Series,
    latents: dict,
    start_dt: datetime.datetime,
    seq: int,
    rng: np.random.Generator,
) -> dict:
    session_timezone = player_row["home_timezone"]
    start_ts_utc = _ensure_utc(start_dt)
    start_ts_local = utc_to_local_naive(start_ts_utc, session_timezone)
    local_dow = start_ts_local.weekday()

    end_reason = _draw_session_end_reason(rng)
    duration_minutes = _draw_duration_minutes(
        latents["duration_scale"], local_dow, end_reason, rng
    )
    end_ts_utc = start_ts_utc + datetime.timedelta(minutes=duration_minutes)
    end_ts_local = utc_to_local_naive(end_ts_utc, session_timezone)

    return {
        "player_id": player_row["player_id"],
        "session_id": _format_session_id(player_row["player_id"], seq),
        "session_start_ts_utc": start_ts_utc,
        "session_end_ts_utc": end_ts_utc,
        "session_timezone": session_timezone,
        "session_start_ts_local": start_ts_local,
        "session_end_ts_local": end_ts_local,
        "duration_minutes": duration_minutes,
        "device": _draw_device(player_row["install_platform"], rng),
        "login_region": _draw_login_region(player_row["data_region"], rng),
        "local_day_of_week": local_dow,
        "session_end_reason": end_reason,
    }


def _generate_player_sessions(
    player_row: pd.Series,
    rng: np.random.Generator,
    observation_end: datetime.date,
) -> list[dict]:
    latents = draw_player_latents(player_row, rng, observation_end)
    activity_cycles = _draw_activity_cycles(latents, player_row, rng, observation_end)
    session_timezone = player_row["home_timezone"]

    rows: list[dict] = []
    seq = 1
    for cycle_idx, (active_start, active_end) in enumerate(activity_cycles):
        start_times = _draw_session_times(
            active_start,
            active_end,
            latents["weekly_session_rate"],
            session_timezone,
            rng,
            cycle_idx,
        )
        for start_dt in start_times:
            rows.append(_draw_session_row(player_row, latents, start_dt, seq, rng))
            seq += 1

    return rows


def _validate_session_invariants(df: pd.DataFrame) -> None:
    for _, row in df.iterrows():
        tz_name = row["session_timezone"]
        start_utc = row["session_start_ts_utc"].to_pydatetime()
        end_utc = row["session_end_ts_utc"].to_pydatetime()
        start_local = row["session_start_ts_local"].to_pydatetime()
        end_local = row["session_end_ts_local"].to_pydatetime()
        assert_utc_local_invariant(start_utc, start_local, tz_name)
        assert_utc_local_invariant(end_utc, end_local, tz_name)
        if start_local.weekday() != row["local_day_of_week"]:
            raise ValueError("local_day_of_week does not match session_start_ts_local")


def generate_sessions(
    players: pd.DataFrame,
    random_seed: int,
    observation_end: datetime.date | None = None,
) -> pd.DataFrame:
    """
    Generate synthetic session rows for each player with reproducible randomness.

    Parameters
    ----------
    players : pd.DataFrame
        Player dimension table from ``generate_players``.
    random_seed : int
        Seed for all random draws — guarantees identical output given the same inputs.
    observation_end : date, optional
        Last calendar day sessions can occur. Defaults to today.

    Returns
    -------
    pd.DataFrame
        Sessions fact table with columns in fixed order.
    """
    _validate_random_seed(random_seed)
    _validate_players(players)

    if observation_end is None:
        observation_end = datetime.date.today()
    elif isinstance(observation_end, pd.Timestamp):
        observation_end = observation_end.date()
    elif isinstance(observation_end, datetime.datetime):
        observation_end = observation_end.date()

    if not isinstance(observation_end, datetime.date):
        raise ValueError("observation_end must be a date")

    rng = np.random.default_rng(random_seed)
    sorted_players = players.sort_values("player_id", kind="mergesort")

    all_rows: list[dict] = []
    for _, player_row in sorted_players.iterrows():
        all_rows.extend(_generate_player_sessions(player_row, rng, observation_end))

    if not all_rows:
        return pd.DataFrame(columns=COLUMN_ORDER)

    df = pd.DataFrame(all_rows)
    df["session_start_ts_utc"] = pd.to_datetime(df["session_start_ts_utc"], utc=True)
    df["session_end_ts_utc"] = pd.to_datetime(df["session_end_ts_utc"], utc=True)
    df["session_start_ts_local"] = pd.to_datetime(df["session_start_ts_local"])
    df["session_end_ts_local"] = pd.to_datetime(df["session_end_ts_local"])

    _validate_session_invariants(df)

    df = df.sort_values(
        ["player_id", "session_start_ts_utc"], kind="mergesort"
    ).reset_index(drop=True)
    return df[COLUMN_ORDER]
