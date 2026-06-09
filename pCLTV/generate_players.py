"""
Synthetic Player dimension data for a live-service collectible card game CLTV project.

v1 schema
---------
player_id, home_timezone, registration_timestamp_utc, registration_timestamp_local,
country, guild_id, acquisition_channel, referrer_player_id,
install_platform, account_status, data_region

"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd

from pCLTV.timezones import (
    assert_utc_local_invariant,
    draw_home_timezone,
    local_naive_to_utc,
    utc_to_local_naive,
)

COUNTRIES = ["US", "JP", "KR", "GB", "DE", "CA", "FR", "BR", "MX", "IN"]
COUNTRY_WEIGHTS = [0.30, 0.15, 0.10, 0.08, 0.07, 0.07, 0.06, 0.07, 0.05, 0.05]

COUNTRY_TO_REGION = {
    "US": "NA",
    "CA": "NA",
    "GB": "EU",
    "DE": "EU",
    "FR": "EU",
    "JP": "APAC",
    "KR": "APAC",
    "IN": "APAC",
    "MX": "LATAM",
    "BR": "LATAM",
}

ACQUISITION_CHANNELS = [
    "organic",
    "paid_social-TikTok",
    "paid_social-Meta",
    "paid_social-YouTube",
    "paid_search-Google",
    "paid_search-Apple_Search_Ads",
    "influencer",
    "referral_code",
    "brand_partnership-podcast",
    "brand_partnership-twitch",
]
ACQUISITION_WEIGHTS = [0.25, 0.15, 0.15, 0.08, 0.08, 0.05, 0.08, 0.06, 0.05, 0.05]

INSTALL_PLATFORMS = ["ios", "android", "pc"]
INSTALL_PLATFORM_WEIGHTS = [0.45, 0.30, 0.25]

ACCOUNT_STATUSES = ["active", "banned", "inactive"]
ACCOUNT_STATUS_WEIGHTS = [0.92, 0.03, 0.05]

COLUMN_ORDER = [
    "player_id",
    "home_timezone",
    "registration_timestamp_utc",
    "registration_timestamp_local",
    "country",
    "guild_id",
    "acquisition_channel",
    "referrer_player_id",
    "install_platform",
    "account_status",
    "data_region",
]


def _validate_inputs(n: int, random_seed: int, game_age: int) -> None:
    if not isinstance(n, int) or isinstance(n, bool):
        raise ValueError("n must be an integer")
    if n < 1:
        raise ValueError("n must be >= 1")
    if not isinstance(random_seed, int) or isinstance(random_seed, bool):
        raise ValueError("random_seed must be an integer")
    if not isinstance(game_age, int) or isinstance(game_age, bool):
        raise ValueError("game_age must be an integer")
    if game_age < 1:
        raise ValueError("game_age must be >= 1")


def _format_player_id(index: int, width: int) -> str:
    return f"P{index:0{width}d}"


def _format_guild_id(index: int, width: int) -> str:
    return f"G{index:0{width}d}"


def _draw_registration_timestamps(
    countries: np.ndarray,
    home_timezones: list[str],
    start_date: datetime.date,
    game_age: int,
    rng: np.random.Generator,
) -> tuple[list[datetime.datetime], list[datetime.datetime]]:
    n = len(countries)
    day_offsets = rng.integers(0, game_age + 1, size=n)
    utc_timestamps: list[datetime.datetime] = []
    local_timestamps: list[datetime.datetime] = []

    for i in range(n):
        reg_date = start_date + datetime.timedelta(days=int(day_offsets[i]))
        hour = int(rng.integers(0, 24))
        minute = int(rng.integers(0, 60))
        second = int(rng.integers(0, 60))
        local_dt = datetime.datetime(reg_date.year, reg_date.month, reg_date.day, hour, minute, second)
        utc_dt = local_naive_to_utc(local_dt, home_timezones[i])
        utc_timestamps.append(utc_dt)
        local_timestamps.append(utc_to_local_naive(utc_dt, home_timezones[i]))

    return utc_timestamps, local_timestamps


def _apply_country_channel_correlation(
    df: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """Subtle post-draw boost: twitch in US/CA, influencer in JP/KR."""
    n_adjust = max(1, int(len(df) * 0.10))
    adjust_idx = rng.choice(len(df), size=n_adjust, replace=False)

    for idx in adjust_idx:
        country = df.at[idx, "country"]
        if country in ("US", "CA") and rng.random() < 0.3:
            df.at[idx, "acquisition_channel"] = "brand_partnership-twitch"
        elif country in ("JP", "KR") and rng.random() < 0.3:
            df.at[idx, "acquisition_channel"] = "influencer"

    return df


def _assign_referrer_player_ids(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    df["referrer_player_id"] = pd.NA

    sorted_idx = df.sort_values(
        "registration_timestamp_utc", kind="mergesort"
    ).index.tolist()
    player_ids_by_sorted_pos = [df.at[i, "player_id"] for i in sorted_idx]

    for pos, row_idx in enumerate(sorted_idx):
        if df.at[row_idx, "acquisition_channel"] != "referral_code":
            continue

        if pos < 2:
            df.at[row_idx, "acquisition_channel"] = "organic"
            continue

        prior_ids = player_ids_by_sorted_pos[:pos]
        df.at[row_idx, "referrer_player_id"] = rng.choice(prior_ids)

    return df


def _validate_registration_invariants(df: pd.DataFrame) -> None:
    for _, row in df.iterrows():
        utc_ts = row["registration_timestamp_utc"].to_pydatetime()
        local_ts = row["registration_timestamp_local"].to_pydatetime()
        assert_utc_local_invariant(utc_ts, local_ts, row["home_timezone"])


def generate_players(n: int, random_seed: int, game_age: int = 180) -> pd.DataFrame:
    """
    Generate n synthetic Player rows with reproducible randomness.

    Parameters
    ----------
    n : int
        Number of players to generate.
    random_seed : int
        Seed for all random draws — guarantees identical output given the same inputs.
    game_age : int, optional
        How many days the game has been live. Registration timestamps are drawn uniformly
        from [today - game_age days, today] inclusive. Default 180 (~0.5 years).

    Returns
    -------
    pd.DataFrame
        Player dimension table with columns in fixed order.
    """
    _validate_inputs(n, random_seed, game_age)
    rng = np.random.default_rng(random_seed)

    id_width = max(8, len(str(n)))
    player_ids = [_format_player_id(i + 1, id_width) for i in range(n)]

    today = datetime.date.today()
    start_date = today - datetime.timedelta(days=game_age)

    countries = rng.choice(COUNTRIES, size=n, p=COUNTRY_WEIGHTS)
    home_timezones = [draw_home_timezone(c, rng) for c in countries]
    registration_utc, registration_local = _draw_registration_timestamps(
        countries, home_timezones, start_date, game_age, rng
    )
    data_regions = [COUNTRY_TO_REGION[c] for c in countries]

    acquisition_channels = rng.choice(
        ACQUISITION_CHANNELS, size=n, p=ACQUISITION_WEIGHTS
    )
    install_platforms = rng.choice(
        INSTALL_PLATFORMS, size=n, p=INSTALL_PLATFORM_WEIGHTS
    )
    account_statuses = rng.choice(
        ACCOUNT_STATUSES, size=n, p=ACCOUNT_STATUS_WEIGHTS
    )

    guild_pool_size = max(10, n // 50)
    guild_width = max(3, len(str(guild_pool_size)))
    guild_ids_pool = [
        _format_guild_id(i + 1, guild_width) for i in range(guild_pool_size)
    ]
    null_rate = rng.uniform(0.30, 0.60)
    has_guild = rng.random(n) >= null_rate
    assigned_guilds = rng.choice(guild_ids_pool, size=n)
    guild_ids = [
        guild if in_guild else None for guild, in_guild in zip(assigned_guilds, has_guild)
    ]

    df = pd.DataFrame(
        {
            "player_id": player_ids,
            "home_timezone": home_timezones,
            "registration_timestamp_utc": registration_utc,
            "registration_timestamp_local": registration_local,
            "country": countries,
            "guild_id": guild_ids,
            "acquisition_channel": acquisition_channels,
            "install_platform": install_platforms,
            "account_status": account_statuses,
            "data_region": data_regions,
        }
    )

    df = _apply_country_channel_correlation(df, rng)
    df = _assign_referrer_player_ids(df, rng)

    df["guild_id"] = df["guild_id"].astype("string")
    df["referrer_player_id"] = df["referrer_player_id"].astype("string")
    df["registration_timestamp_utc"] = pd.to_datetime(
        df["registration_timestamp_utc"], utc=True
    )
    df["registration_timestamp_local"] = pd.to_datetime(
        df["registration_timestamp_local"]
    )

    _validate_registration_invariants(df)

    return df[COLUMN_ORDER]
