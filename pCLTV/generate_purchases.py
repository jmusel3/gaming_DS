"""
Synthetic fiat purchase line items for a live-service collectible card game CLTV project.

v1 schema
---------
player_id, purchase_id, line_item_id, store_platform, promo_code,
purchase_datetime_utc, Timezone, purchase_datetime_local,
product_id, price_tier_id, gross_amount_usd, net_amount_usd,
currency_code, fx_rate_to_usd, gross_amount_local, net_amount_local,
tax_included_in_price, transaction_type, payment_method
"""

from __future__ import annotations

import datetime
import math
from typing import Any

import numpy as np
import pandas as pd

from pCLTV.generate_prices import FIAT_CURRENCIES, FIAT_FX_VS_USD, fiat_eligible
from pCLTV.player_latents import draw_player_latents_map
from pCLTV.timezones import assert_utc_local_invariant, utc_to_local_naive

PURCHASE_RNG_SALT = 9_001_001

REQUIRED_PLAYER_COLUMNS = [
    "player_id",
    "home_timezone",
    "registration_timestamp_utc",
    "country",
    "install_platform",
    "account_status",
    "banned_datetime_utc",
]

REQUIRED_SESSION_COLUMNS = [
    "player_id",
    "session_id",
    "session_start_ts_utc",
    "session_end_ts_utc",
    "device",
    "duration_minutes",
    "local_day_of_week",
]

REQUIRED_PRODUCT_COLUMNS = [
    "product_id",
    "product_type",
    "price_tier_id",
    "effective_from_utc",
    "effective_to_utc",
    "is_active",
]

REQUIRED_PRICE_COLUMNS = [
    "product_id",
    "price_tier_id",
    "price_type",
    "promo_code",
    "store_platform",
    "currency_code",
    "is_soft_currency",
    "amount",
    "effective_from_utc",
    "effective_to_utc",
]

COUNTRY_TO_CURRENCY = {
    "US": "USD",
    "CA": "CAD",
    "GB": "GBP",
    "DE": "EUR",
    "FR": "EUR",
    "JP": "JPY",
    "KR": "KRW",
    "BR": "BRL",
    "MX": "MXN",
    "IN": "INR",
}

TAX_INCLUDED_COUNTRIES = {"DE", "FR", "GB", "JP", "KR", "IN", "BR"}

TAX_RATE_BY_COUNTRY = {
    "DE": 0.20,
    "FR": 0.20,
    "GB": 0.20,
    "JP": 0.10,
    "KR": 0.10,
    "IN": 0.18,
    "BR": 0.17,
}

PLATFORM_FEE_RATE = {"iap": 0.30, "web_store": 0.03}

PAYMENT_METHODS = ["visa", "mastercard", "paypal", "apple_pay"]

PAYMENT_WEIGHTS = {
    ("iap", "ios"): [0.25, 0.20, 0.10, 0.45],
    ("iap", "android"): [0.40, 0.35, 0.25, 0.00],
    ("web_store", "web"): [0.45, 0.35, 0.20, 0.00],
}

COLUMN_ORDER = [
    "player_id",
    "purchase_id",
    "line_item_id",
    "store_platform",
    "promo_code",
    "purchase_datetime_utc",
    "Timezone",
    "purchase_datetime_local",
    "product_id",
    "price_tier_id",
    "gross_amount_usd",
    "net_amount_usd",
    "currency_code",
    "fx_rate_to_usd",
    "gross_amount_local",
    "net_amount_local",
    "tax_included_in_price",
    "transaction_type",
    "payment_method",
]


def _validate_random_seed(random_seed: int) -> None:
    if not isinstance(random_seed, int) or isinstance(random_seed, bool):
        raise ValueError("random_seed must be an integer")


def _validate_dataframe(df: pd.DataFrame, name: str, required: list[str]) -> None:
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"{name} must be a pandas DataFrame")
    if df.empty:
        raise ValueError(f"{name} must not be empty")
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _ensure_utc(dt: datetime.datetime) -> datetime.datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=datetime.timezone.utc)
    return dt.astimezone(datetime.timezone.utc)


def _to_utc_timestamp(value: pd.Timestamp | datetime.datetime) -> datetime.datetime:
    if isinstance(value, pd.Timestamp):
        return _ensure_utc(value.to_pydatetime())
    return _ensure_utc(value)


def _observation_end_date(observation_end: datetime.date | None) -> datetime.date:
    if observation_end is None:
        return datetime.date.today()
    if isinstance(observation_end, pd.Timestamp):
        return observation_end.date()
    if isinstance(observation_end, datetime.datetime):
        return observation_end.date()
    if not isinstance(observation_end, datetime.date):
        raise ValueError("observation_end must be a date")
    return observation_end


def _player_id_suffix(player_id: str) -> str:
    return player_id[1:] if player_id.startswith("P") else player_id


def _format_purchase_id(player_id: str, seq: int) -> str:
    return f"PU{_player_id_suffix(player_id)}_{seq:04d}"


def _format_line_item_id(purchase_id: str, item: int) -> str:
    suffix = purchase_id[2:] if purchase_id.startswith("PU") else purchase_id
    return f"LI{suffix}_{item:02d}"


def _device_to_store_platform(device: str) -> str:
    if device == "ios":
        return "ios"
    if device == "android":
        return "android"
    return "web"


def _store_platform_to_transaction_type(store_platform: str) -> str:
    if store_platform in ("ios", "android"):
        return "iap"
    return "web_store"


def _payer_probability(engagement_score: float, account_status: str) -> float:
    p = min(0.12, max(0.01, 0.015 + 0.035 * math.log1p(engagement_score)))
    if account_status == "inactive":
        p *= 0.15
    return p


def _spend_tier(engagement_score: float) -> str:
    if engagement_score < 1.5:
        return "low"
    if engagement_score < 3.0:
        return "mid"
    return "high"


def _product_weight(row: pd.Series, tier: str) -> float:
    ptype = row["product_type"]
    price_tier = int(row["price_tier_id"])

    if tier == "low":
        if ptype in ("card_pack_gacha", "card_pack_fixed"):
            return 3.0
        if ptype == "card" and price_tier <= 2:
            return 2.0
        if ptype == "player_cosmetic" and price_tier <= 2:
            return 2.0
        return 0.1

    if tier == "mid":
        if ptype == "gem_pack":
            return 2.5
        if ptype in ("cosmetic_pack_gacha", "cosmetic_pack_fixed"):
            return 2.0
        if ptype == "battle_pass":
            return 2.5
        if ptype in ("card_pack_gacha", "card_pack_fixed"):
            return 1.5
        if ptype == "limited_time_event":
            return 1.0
        return 0.1

    if ptype == "gem_pack" and price_tier >= 8:
        return 4.0
    if ptype == "battle_pass":
        return 3.0
    if ptype == "limited_time_event":
        return 2.5
    if ptype == "gem_pack":
        return 2.0
    return 0.1


def _is_product_available(row: pd.Series, purchase_ts: pd.Timestamp) -> bool:
    if not fiat_eligible(row["product_type"], int(row["price_tier_id"])):
        return False
    if purchase_ts < row["effective_from_utc"]:
        return False
    if pd.notna(row["effective_to_utc"]) and purchase_ts >= row["effective_to_utc"]:
        return False
    return True


def _draw_product(
    catalog: pd.DataFrame,
    purchase_ts: pd.Timestamp,
    spend_tier: str,
    rng: np.random.Generator,
) -> pd.Series | None:
    mask = catalog.apply(lambda r: _is_product_available(r, purchase_ts), axis=1)
    pool = catalog.loc[mask]
    if pool.empty:
        return None

    weights = pool.apply(lambda r: _product_weight(r, spend_tier), axis=1).to_numpy(dtype=float)
    if weights.sum() <= 0:
        return None
    weights /= weights.sum()
    idx = int(rng.choice(pool.index.to_numpy(), p=weights))
    return catalog.loc[idx]


def _build_fiat_price_index(prices: pd.DataFrame) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    fiat = prices[~prices["is_soft_currency"]].copy()
    index: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for _, row in fiat.iterrows():
        key = (str(row["product_id"]), str(row["store_platform"]), str(row["currency_code"]))
        index.setdefault(key, []).append(row.to_dict())
    return index


def _resolve_price_row(
    price_index: dict[tuple[str, str, str], list[dict[str, Any]]],
    product_id: str,
    store_platform: str,
    currency_code: str,
    purchase_ts: pd.Timestamp,
    rng: np.random.Generator,
) -> dict[str, Any] | None:
    candidates = price_index.get((product_id, store_platform, currency_code), [])
    active = [
        row
        for row in candidates
        if row["effective_from_utc"] <= purchase_ts
        and (pd.isna(row["effective_to_utc"]) or purchase_ts < row["effective_to_utc"])
    ]
    if not active:
        return None

    promos = [r for r in active if r["price_type"] == "promo"]
    standards = [r for r in active if r["price_type"] == "standard"]

    if promos and (not standards or rng.random() < 0.65):
        return promos[int(rng.integers(0, len(promos)))]
    if standards:
        return standards[0]
    return None


def _compute_net_amounts(
    gross_local: float,
    country: str,
    transaction_type: str,
) -> tuple[float, float, bool]:
    tax_included = country in TAX_INCLUDED_COUNTRIES
    tax_rate = TAX_RATE_BY_COUNTRY.get(country, 0.0)
    platform_fee = PLATFORM_FEE_RATE[transaction_type]
    after_platform = gross_local * (1.0 - platform_fee)
    if tax_included:
        net_local = after_platform / (1.0 + tax_rate)
    else:
        net_local = after_platform
    return round(net_local, 2), tax_rate, tax_included


def _draw_payment_method(
    transaction_type: str,
    store_platform: str,
    rng: np.random.Generator,
) -> str:
    key = (transaction_type, store_platform)
    weights = PAYMENT_WEIGHTS.get(key, PAYMENT_WEIGHTS[("web_store", "web")])
    return str(rng.choice(PAYMENT_METHODS, p=weights))


def _draw_purchase_timestamp(
    session_start: pd.Timestamp,
    session_end: pd.Timestamp,
    banned_ts: pd.Timestamp | pd._libs.nattype.NaTType | None,
    rng: np.random.Generator,
) -> pd.Timestamp | None:
    start = _to_utc_timestamp(session_start) + datetime.timedelta(minutes=2)
    end = _to_utc_timestamp(session_end) - datetime.timedelta(minutes=1)
    if banned_ts is not None and pd.notna(banned_ts):
        end = min(end, _to_utc_timestamp(banned_ts) - datetime.timedelta(seconds=1))
    if end <= start:
        return None
    if (end - start).total_seconds() < 180:
        return None

    offset_seconds = float(rng.uniform(0, (end - start).total_seconds()))
    return pd.Timestamp(start + datetime.timedelta(seconds=offset_seconds))


def _session_purchase_probability(
    engagement_score: float,
    local_day_of_week: int,
    duration_minutes: float,
    session_end_reason: str | None,
) -> float:
    base = 0.012
    weekend_multiplier = 1.25 if local_day_of_week >= 5 else 1.0
    duration_factor = min(2.0, max(0.5, duration_minutes / 20.0))
    p = base * (engagement_score**0.4) * weekend_multiplier * duration_factor
    if session_end_reason in ("app_crash", "disconnected"):
        p *= 0.5
    return p


def _draw_cart_size(rng: np.random.Generator) -> int:
    if rng.random() < 0.85:
        return 1
    return int(rng.integers(2, 4))


def _generate_player_purchases(
    player_row: pd.Series,
    player_sessions: pd.DataFrame,
    latents: dict,
    catalog: pd.DataFrame,
    price_index: dict[tuple[str, str, str], list[dict[str, Any]]],
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    engagement = latents["engagement_score"]
    if rng.random() >= _payer_probability(engagement, str(player_row["account_status"])):
        return []

    is_banned = player_row["account_status"] == "banned"
    banned_ts = player_row["banned_datetime_utc"] if is_banned else pd.NaT

    if is_banned and pd.isna(banned_ts):
        return []

    sessions = player_sessions.sort_values("session_start_ts_utc", kind="mergesort")
    if is_banned:
        banned_dt = _to_utc_timestamp(banned_ts)
        sessions = sessions[
            pd.to_datetime(sessions["session_end_ts_utc"], utc=True) <= banned_dt
        ]

    if sessions.empty:
        return []

    country = str(player_row["country"])
    currency_code = COUNTRY_TO_CURRENCY[country]
    home_timezone = str(player_row["home_timezone"])
    spend_tier = _spend_tier(engagement)

    rows: list[dict[str, Any]] = []
    purchase_seq = 1
    purchase_dates: set[datetime.date] = set()

    def _append_purchase_from_session(session: pd.Series, force: bool = False) -> bool:
        nonlocal purchase_seq

        end_reason = (
            str(session["session_end_reason"])
            if "session_end_reason" in session.index
            else None
        )
        if not force:
            p = _session_purchase_probability(
                engagement,
                int(session["local_day_of_week"]),
                float(session["duration_minutes"]),
                end_reason,
            )
            if rng.random() >= p:
                return False

        purchase_ts = _draw_purchase_timestamp(
            session["session_start_ts_utc"],
            session["session_end_ts_utc"],
            banned_ts if is_banned else None,
            rng,
        )
        if purchase_ts is None:
            return False

        purchase_date = purchase_ts.date()
        if purchase_date in purchase_dates:
            return False
        purchase_dates.add(purchase_date)

        store_platform = _device_to_store_platform(str(session["device"]))
        transaction_type = _store_platform_to_transaction_type(store_platform)
        cart_size = _draw_cart_size(rng)
        purchase_id = _format_purchase_id(str(player_row["player_id"]), purchase_seq)
        purchase_seq += 1
        purchase_local = utc_to_local_naive(purchase_ts.to_pydatetime(), home_timezone)

        added = False
        for item_idx in range(1, cart_size + 1):
            product = _draw_product(catalog, purchase_ts, spend_tier, rng)
            if product is None:
                break

            price_row = _resolve_price_row(
                price_index,
                str(product["product_id"]),
                store_platform,
                currency_code,
                purchase_ts,
                rng,
            )
            if price_row is None:
                continue

            gross_local = float(price_row["amount"])
            fx_rate = FIAT_FX_VS_USD[currency_code]
            gross_usd = round(gross_local / fx_rate, 2)
            net_local, _, tax_included = _compute_net_amounts(
                gross_local, country, transaction_type
            )
            net_usd = round(net_local / fx_rate, 2)

            rows.append(
                {
                    "player_id": player_row["player_id"],
                    "purchase_id": purchase_id,
                    "line_item_id": _format_line_item_id(purchase_id, item_idx),
                    "store_platform": store_platform,
                    "promo_code": price_row["promo_code"],
                    "purchase_datetime_utc": purchase_ts,
                    "Timezone": home_timezone,
                    "purchase_datetime_local": purchase_local,
                    "product_id": product["product_id"],
                    "price_tier_id": int(product["price_tier_id"]),
                    "gross_amount_usd": gross_usd,
                    "net_amount_usd": net_usd,
                    "currency_code": currency_code,
                    "fx_rate_to_usd": fx_rate,
                    "gross_amount_local": gross_local,
                    "net_amount_local": net_local,
                    "tax_included_in_price": tax_included,
                    "transaction_type": transaction_type,
                    "payment_method": _draw_payment_method(
                        transaction_type, store_platform, rng
                    ),
                }
            )
            added = True

        if not added:
            purchase_dates.discard(purchase_date)
            purchase_seq -= 1
        return added

    for _, session in sessions.iterrows():
        _append_purchase_from_session(session)

    if not rows:
        shuffled_idx = rng.permutation(len(sessions))
        for idx in shuffled_idx:
            if _append_purchase_from_session(sessions.iloc[int(idx)], force=True):
                break

    return rows


def _validate_purchase_invariants(
    purchases: pd.DataFrame,
    players: pd.DataFrame,
    sessions: pd.DataFrame,
    products: pd.DataFrame,
    prices: pd.DataFrame,
) -> None:
    if purchases.empty:
        return

    player_ids = set(players["player_id"])
    product_ids = set(products["product_id"])

    for _, row in purchases.iterrows():
        if row["player_id"] not in player_ids:
            raise ValueError(f"unknown player_id {row['player_id']}")
        if row["product_id"] not in product_ids:
            raise ValueError(f"unknown product_id {row['product_id']}")
        if row["currency_code"] not in FIAT_CURRENCIES:
            raise ValueError(f"non-fiat currency {row['currency_code']}")

        purchase_utc = row["purchase_datetime_utc"].to_pydatetime()
        purchase_local = row["purchase_datetime_local"].to_pydatetime()
        tz_name = row["Timezone"]
        assert_utc_local_invariant(purchase_utc, purchase_local, tz_name)

        player = players.loc[players["player_id"] == row["player_id"]].iloc[0]
        if player["account_status"] == "banned":
            ban_ts = player["banned_datetime_utc"].to_pydatetime()
            if purchase_utc >= ban_ts:
                raise ValueError(
                    f"{row['purchase_id']}: purchase at or after banned_datetime_utc"
                )

        player_sessions = sessions.loc[sessions["player_id"] == row["player_id"]]
        matching = player_sessions[
            (pd.to_datetime(player_sessions["session_start_ts_utc"], utc=True) <= row["purchase_datetime_utc"])
            & (pd.to_datetime(player_sessions["session_end_ts_utc"], utc=True) >= row["purchase_datetime_utc"])
        ]
        if matching.empty:
            raise ValueError(f"{row['purchase_id']}: purchase outside all session windows")
        matching_platforms = {
            _device_to_store_platform(str(d)) for d in matching["device"]
        }
        if row["store_platform"] not in matching_platforms:
            raise ValueError(f"{row['purchase_id']}: store_platform inconsistent with session device")

    if purchases.duplicated(subset=["purchase_id", "line_item_id"]).any():
        raise ValueError("duplicate line_item_id within purchase_id")

    fiat_prices = prices[~prices["is_soft_currency"]]
    for _, row in purchases.iterrows():
        match = fiat_prices[
            (fiat_prices["product_id"] == row["product_id"])
            & (fiat_prices["store_platform"] == row["store_platform"])
            & (fiat_prices["currency_code"] == row["currency_code"])
            & (fiat_prices["amount"] == row["gross_amount_local"])
        ]
        promo_code = row["promo_code"]
        if pd.isna(promo_code):
            match = match[match["promo_code"].isna()]
        else:
            match = match[match["promo_code"] == promo_code]
        ts = row["purchase_datetime_utc"]
        match = match[
            (match["effective_from_utc"] <= ts)
            & (match["effective_to_utc"].isna() | (match["effective_to_utc"] > ts))
        ]
        if match.empty:
            raise ValueError(
                f"{row['line_item_id']}: gross_amount_local does not match any price row"
            )


def generate_purchases(
    players: pd.DataFrame,
    sessions: pd.DataFrame,
    products: pd.DataFrame,
    prices: pd.DataFrame,
    random_seed: int,
    observation_end: datetime.date | None = None,
) -> pd.DataFrame:
    """
    Generate synthetic fiat purchase line items anchored to player sessions.

    Parameters
    ----------
    players : pd.DataFrame
        Player dimension table from ``generate_players``.
    sessions : pd.DataFrame
        Sessions fact table from ``generate_sessions``.
    products : pd.DataFrame
        Product catalog from ``generate_products``.
    prices : pd.DataFrame
        Product prices from ``generate_prices``.
    random_seed : int
        Seed for purchase draws. Engagement latents use the same seed as sessions.
    observation_end : date, optional
        Observation window end for latent alignment. Defaults to today.

    Returns
    -------
    pd.DataFrame
        Purchases fact table with one row per line item.
    """
    _validate_random_seed(random_seed)
    _validate_dataframe(players, "players", REQUIRED_PLAYER_COLUMNS)
    _validate_dataframe(sessions, "sessions", REQUIRED_SESSION_COLUMNS)
    _validate_dataframe(products, "products", REQUIRED_PRODUCT_COLUMNS)
    _validate_dataframe(prices, "prices", REQUIRED_PRICE_COLUMNS)

    obs_end = _observation_end_date(observation_end)
    latents_map = draw_player_latents_map(players, random_seed, obs_end)
    rng = np.random.default_rng(random_seed + PURCHASE_RNG_SALT)

    catalog = products.copy()
    catalog["effective_from_utc"] = pd.to_datetime(catalog["effective_from_utc"], utc=True)
    catalog["effective_to_utc"] = pd.to_datetime(catalog["effective_to_utc"], utc=True)
    catalog = catalog.sort_values("product_id", kind="mergesort")

    fiat_prices = prices[~prices["is_soft_currency"]].copy()
    fiat_prices["effective_from_utc"] = pd.to_datetime(
        fiat_prices["effective_from_utc"], utc=True
    )
    fiat_prices["effective_to_utc"] = pd.to_datetime(
        fiat_prices["effective_to_utc"], utc=True
    )
    price_index = _build_fiat_price_index(fiat_prices)

    sorted_players = players.sort_values("player_id", kind="mergesort")
    sessions_by_player = {
        pid: grp for pid, grp in sessions.groupby("player_id", sort=False)
    }

    all_rows: list[dict[str, Any]] = []
    for _, player_row in sorted_players.iterrows():
        player_sessions = sessions_by_player.get(player_row["player_id"])
        if player_sessions is None or player_sessions.empty:
            continue
        latents = latents_map[player_row["player_id"]]
        all_rows.extend(
            _generate_player_purchases(
                player_row,
                player_sessions,
                latents,
                catalog,
                price_index,
                rng,
            )
        )

    if not all_rows:
        return pd.DataFrame(columns=COLUMN_ORDER)

    df = pd.DataFrame(all_rows)
    df["purchase_datetime_utc"] = pd.to_datetime(df["purchase_datetime_utc"], utc=True)
    df["purchase_datetime_local"] = pd.to_datetime(df["purchase_datetime_local"])
    df["promo_code"] = df["promo_code"].astype("string")
    df["store_platform"] = df["store_platform"].astype("string")
    df["currency_code"] = df["currency_code"].astype("string")
    df["transaction_type"] = df["transaction_type"].astype("string")
    df["payment_method"] = df["payment_method"].astype("string")
    df["Timezone"] = df["Timezone"].astype("string")
    df["tax_included_in_price"] = df["tax_included_in_price"].astype("bool")
    df["price_tier_id"] = df["price_tier_id"].astype("int64")

    _validate_purchase_invariants(df, players, sessions, products, fiat_prices)

    df = df.sort_values(
        ["player_id", "purchase_datetime_utc", "purchase_id", "line_item_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    return df[COLUMN_ORDER]
