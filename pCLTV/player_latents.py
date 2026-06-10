"""Shared per-player latent trait draws for sessions and purchases."""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd

CHURN_TYPES = ["loyal", "fade", "early"]
CHURN_TYPE_WEIGHTS = [0.55, 0.35, 0.10]

BASE_REACTIVATION_PROB = {"loyal": 0.50, "fade": 0.20, "early": 0.08}
BASE_SESSION_RATE = 4.0

PLATFORM_DURATION_FACTOR = {"ios": 1.0, "android": 1.0, "pc": 1.1}


def _ensure_utc(dt: datetime.datetime) -> datetime.datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=datetime.timezone.utc)
    return dt.astimezone(datetime.timezone.utc)


def _to_utc_timestamp(value: pd.Timestamp | datetime.datetime) -> datetime.datetime:
    if isinstance(value, pd.Timestamp):
        return _ensure_utc(value.to_pydatetime())
    return _ensure_utc(value)


def draw_player_latents(
    row: pd.Series, rng: np.random.Generator, observation_end: datetime.date
) -> dict:
    registration_ts_utc = _to_utc_timestamp(row["registration_timestamp_utc"])
    registration_date = registration_ts_utc.date()
    tenure = max(1, (observation_end - registration_date).days)
    account_status = row["account_status"]
    install_platform = row["install_platform"]

    engagement_score = float(rng.lognormal(0.0, 0.6))
    weekly_session_rate = float(rng.gamma(2.0, engagement_score * BASE_SESSION_RATE))
    platform_factor = PLATFORM_DURATION_FACTOR.get(install_platform, 1.0)
    duration_scale = (engagement_score**0.7) * platform_factor

    churn_type = str(rng.choice(CHURN_TYPES, p=CHURN_TYPE_WEIGHTS))
    base_reactivation_prob = BASE_REACTIVATION_PROB[churn_type]
    banned = account_status == "banned"

    if account_status == "inactive":
        churn_type = "early"
        base_reactivation_prob *= 0.5

    return {
        "engagement_score": engagement_score,
        "weekly_session_rate": max(weekly_session_rate, 0.5),
        "duration_scale": duration_scale,
        "churn_type": churn_type,
        "base_reactivation_prob": base_reactivation_prob,
        "banned": banned,
        "tenure": tenure,
        "registration_ts_utc": registration_ts_utc,
    }


def draw_player_latents_map(
    players: pd.DataFrame,
    random_seed: int,
    observation_end: datetime.date,
) -> dict[str, dict]:
    rng = np.random.default_rng(random_seed)
    sorted_players = players.sort_values("player_id", kind="mergesort")
    return {
        row["player_id"]: draw_player_latents(row, rng, observation_end)
        for _, row in sorted_players.iterrows()
    }
