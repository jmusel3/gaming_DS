"""
Synthetic product prices for a live-service collectible card game CLTV project.

Resolves ``price_tier_id`` from the product catalog into localized fiat amounts,
soft-currency (GOLD/GEMS) costs, platform markups, and time-bounded promos.

v1 schema
---------
product_id, price_tier_id, price_type, promo_code, store_platform, currency_code,
is_soft_currency, amount, effective_from_utc, effective_to_utc
"""

from __future__ import annotations

import datetime
import math
from typing import Any

import numpy as np
import pandas as pd

EXPANSION_DAY = 90
SEASON_LENGTH_DAYS = 30

REQUIRED_PRODUCT_COLUMNS = [
    "product_id",
    "product_type",
    "price_tier_id",
    "effective_from_utc",
    "effective_to_utc",
]

FIAT_CURRENCIES = ["USD", "CAD", "GBP", "EUR", "JPY", "KRW", "BRL", "MXN", "INR"]
FIAT_FX_VS_USD = {
    "USD": 1.00,
    "CAD": 1.35,
    "GBP": 0.79,
    "EUR": 0.92,
    "JPY": 150.0,
    "KRW": 1350.0,
    "BRL": 5.0,
    "MXN": 17.0,
    "INR": 83.0,
}

STORE_PLATFORMS = ["web", "ios", "android"]
PLATFORM_MARKUP = {"web": 1.0, "ios": 1.2, "android": 1.2}

TIER_USD_WEB = {
    1: 0.99,
    2: 1.99,
    3: 4.99,
    4: 6.99,
    5: 9.99,
    6: 14.99,
    7: 19.99,
    8: 29.99,
    9: 49.99,
    10: 99.99,
}

TIER_GOLD = {
    1: 500,
    2: 1000,
    3: 2500,
    4: 4000,
    5: 6000,
    6: 9000,
    7: 12000,
    8: 18000,
    9: 30000,
    10: 60000,
}

TIER_GEMS = {
    1: 50,
    2: 100,
    3: 250,
    4: 400,
    5: 600,
    6: 900,
    7: 1200,
    8: 1800,
    9: 3000,
    10: 6000,
}

INTEGER_FIAT_CURRENCIES = {"JPY", "KRW", "INR"}

FIAT_ONLY_TYPES = {
    "card_pack_gacha",
    "cosmetic_pack_gacha",
    "gem_pack",
    "battle_pass",
}

FIAT_AND_GOLD_TYPES = {
    "card_pack_fixed",
    "cosmetic_pack_fixed",
    "limited_time_event",
}

PACK_PROMO_TYPES = {
    "card_pack_gacha",
    "card_pack_fixed",
    "cosmetic_pack_gacha",
    "cosmetic_pack_fixed",
}

COLUMN_ORDER = [
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


def _validate_inputs(random_seed: int, game_age: int) -> None:
    if not isinstance(random_seed, int) or isinstance(random_seed, bool):
        raise ValueError("random_seed must be an integer")
    if not isinstance(game_age, int) or isinstance(game_age, bool):
        raise ValueError("game_age must be an integer")
    if game_age < 1:
        raise ValueError("game_age must be >= 1")


def _validate_products(products: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_PRODUCT_COLUMNS if c not in products.columns]
    if missing:
        raise ValueError(f"products missing required columns: {missing}")
    if products["product_id"].duplicated().any():
        raise ValueError("products contains duplicate product_id values")
    tiers = products["price_tier_id"]
    if not tiers.between(1, 10).all():
        raise ValueError("price_tier_id must be between 1 and 10")


def _midnight_utc(d: datetime.date) -> datetime.datetime:
    return datetime.datetime(d.year, d.month, d.day, tzinfo=datetime.timezone.utc)


def gems_eligible(product_type: str, tier: int) -> bool:
    if product_type in ("deck", "card_pack_flex"):
        return True
    if product_type == "card" and tier in (3, 10):
        return True
    if product_type == "player_cosmetic" and tier in (3, 4):
        return True
    return False


def fiat_eligible(product_type: str, tier: int) -> bool:
    if gems_eligible(product_type, tier):
        return False
    if product_type in FIAT_ONLY_TYPES or product_type in FIAT_AND_GOLD_TYPES:
        return True
    if product_type == "card" and tier in (1, 2):
        return True
    if product_type == "player_cosmetic" and tier in (1, 2):
        return True
    return False


def gold_eligible(product_type: str, tier: int) -> bool:
    if gems_eligible(product_type, tier):
        return False
    if product_type in FIAT_AND_GOLD_TYPES:
        return True
    if product_type == "card" and tier in (1, 2):
        return True
    if product_type == "player_cosmetic" and tier in (1, 2):
        return True
    return False


def _round_fiat(amount: float, currency_code: str) -> float:
    if currency_code in INTEGER_FIAT_CURRENCIES:
        return float(round(amount))
    return round(amount, 2)


def _fiat_amount(tier: int, currency_code: str, store_platform: str) -> float:
    base_usd = TIER_USD_WEB[tier]
    raw = base_usd * FIAT_FX_VS_USD[currency_code] * PLATFORM_MARKUP[store_platform]
    return _round_fiat(raw, currency_code)


def _clip_promo_window(
    promo_from: pd.Timestamp,
    promo_to: pd.Timestamp,
    product_from: pd.Timestamp,
    product_to: pd.Timestamp | pd._libs.nattype.NaTType,
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    start = max(promo_from, product_from)
    if pd.isna(product_to):
        end = promo_to
    else:
        end = min(promo_to, product_to)
    if start >= end:
        return None
    return start, end


def _price_row(
    product_id: str,
    price_tier_id: int,
    price_type: str,
    promo_code: str | None,
    store_platform: str,
    currency_code: str,
    is_soft_currency: bool,
    amount: float,
    effective_from: pd.Timestamp,
    effective_to: pd.Timestamp | pd._libs.nattype.NaTType,
) -> dict[str, Any]:
    return {
        "product_id": product_id,
        "price_tier_id": price_tier_id,
        "price_type": price_type,
        "promo_code": promo_code,
        "store_platform": store_platform,
        "currency_code": currency_code,
        "is_soft_currency": is_soft_currency,
        "amount": amount,
        "effective_from_utc": effective_from,
        "effective_to_utc": effective_to,
    }


def _build_promo_calendar(
    products: pd.DataFrame,
    launch: pd.Timestamp,
    expansion: pd.Timestamp,
    game_age: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    promos: list[dict[str, Any]] = []

    promos.append(
        {
            "promo_code": "LAUNCH20",
            "discount": 0.20,
            "from": launch,
            "to": launch + pd.Timedelta(days=7),
            "matcher": lambda row: row["product_type"] in PACK_PROMO_TYPES,
        }
    )

    promos.append(
        {
            "promo_code": "EXPANSION15",
            "discount": 0.15,
            "from": expansion,
            "to": expansion + pd.Timedelta(days=14),
            "matcher": lambda row: (
                row["effective_from_utc"] == expansion
                and fiat_eligible(row["product_type"], int(row["price_tier_id"]))
            ),
        }
    )

    n_seasons = math.ceil(game_age / SEASON_LENGTH_DAYS)
    for k in range(n_seasons):
        season = k + 1
        season_start = launch + pd.Timedelta(days=SEASON_LENGTH_DAYS * k)
        season_end = launch + pd.Timedelta(days=SEASON_LENGTH_DAYS * (k + 1))
        prefix = f"BP_S{season:02d}_"

        def _pass_matcher(row: pd.Series, p: str = prefix) -> bool:
            return row["product_type"] == "battle_pass" and str(row["product_id"]).startswith(p)

        promos.append(
            {
                "promo_code": f"S{season:02d}_PASS10",
                "discount": 0.10,
                "from": season_start,
                "to": season_end,
                "matcher": _pass_matcher,
            }
        )

    flash_pool = products[
        (products["product_type"] == "player_cosmetic") & (products["price_tier_id"] <= 2)
    ]
    if len(flash_pool) > 0 and game_age > 37:
        n_flash = min(30, len(flash_pool))
        chosen_ids = set(
            rng.choice(flash_pool["product_id"].to_numpy(), size=n_flash, replace=False)
        )
        max_offset = max(31, game_age - 7)
        offset = int(rng.integers(30, max_offset))
        flash_start = launch + pd.Timedelta(days=offset)
        flash_end = flash_start + pd.Timedelta(days=3)

        promos.append(
            {
                "promo_code": "FLASH_COS",
                "discount": 0.25,
                "from": flash_start,
                "to": flash_end,
                "matcher": lambda row, ids=chosen_ids: row["product_id"] in ids,
            }
        )

    return promos


def _matching_promos(row: pd.Series, promos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [p for p in promos if p["matcher"](row)]


def _emit_gems_row(row: pd.Series) -> dict[str, Any]:
    tier = int(row["price_tier_id"])
    return _price_row(
        product_id=str(row["product_id"]),
        price_tier_id=tier,
        price_type="standard",
        promo_code=None,
        store_platform="web",
        currency_code="GEMS",
        is_soft_currency=True,
        amount=float(TIER_GEMS[tier]),
        effective_from=row["effective_from_utc"],
        effective_to=row["effective_to_utc"],
    )


def _emit_gold_row(row: pd.Series) -> dict[str, Any]:
    tier = int(row["price_tier_id"])
    return _price_row(
        product_id=str(row["product_id"]),
        price_tier_id=tier,
        price_type="standard",
        promo_code=None,
        store_platform="web",
        currency_code="GOLD",
        is_soft_currency=True,
        amount=float(TIER_GOLD[tier]),
        effective_from=row["effective_from_utc"],
        effective_to=row["effective_to_utc"],
    )


def _emit_fiat_standard_rows(row: pd.Series) -> list[dict[str, Any]]:
    tier = int(row["price_tier_id"])
    rows: list[dict[str, Any]] = []
    for platform in STORE_PLATFORMS:
        for currency in FIAT_CURRENCIES:
            rows.append(
                _price_row(
                    product_id=str(row["product_id"]),
                    price_tier_id=tier,
                    price_type="standard",
                    promo_code=None,
                    store_platform=platform,
                    currency_code=currency,
                    is_soft_currency=False,
                    amount=_fiat_amount(tier, currency, platform),
                    effective_from=row["effective_from_utc"],
                    effective_to=row["effective_to_utc"],
                )
            )
    return rows


def _emit_fiat_promo_rows(
    row: pd.Series,
    promos: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tier = int(row["price_tier_id"])
    rows: list[dict[str, Any]] = []
    for promo in _matching_promos(row, promos):
        clipped = _clip_promo_window(
            promo["from"],
            promo["to"],
            row["effective_from_utc"],
            row["effective_to_utc"],
        )
        if clipped is None:
            continue
        promo_from, promo_to = clipped
        discount = promo["discount"]
        for platform in STORE_PLATFORMS:
            for currency in FIAT_CURRENCIES:
                standard = _fiat_amount(tier, currency, platform)
                promo_amount = _round_fiat(standard * (1.0 - discount), currency)
                rows.append(
                    _price_row(
                        product_id=str(row["product_id"]),
                        price_tier_id=tier,
                        price_type="promo",
                        promo_code=promo["promo_code"],
                        store_platform=platform,
                        currency_code=currency,
                        is_soft_currency=False,
                        amount=promo_amount,
                        effective_from=promo_from,
                        effective_to=promo_to,
                    )
                )
    return rows


def _validate_prices_invariants(prices: pd.DataFrame, products: pd.DataFrame) -> None:
    catalog_ids = set(products["product_id"])
    if not set(prices["product_id"]).issubset(catalog_ids):
        raise ValueError("prices contains unknown product_id values")

    tier_by_product = dict(zip(products["product_id"], products["price_tier_id"]))
    for _, row in prices.iterrows():
        expected_tier = tier_by_product[row["product_id"]]
        if row["price_tier_id"] != expected_tier:
            raise ValueError(
                f"{row['product_id']}: price_tier_id {row['price_tier_id']} != catalog {expected_tier}"
            )

    standard = prices["price_type"] == "standard"
    promo = prices["price_type"] == "promo"
    if prices.loc[standard, "promo_code"].notna().any():
        raise ValueError("standard rows must have null promo_code")
    if prices.loc[promo, "promo_code"].isna().any():
        raise ValueError("promo rows must have non-null promo_code")

    soft = prices[prices["is_soft_currency"]]
    if not soft["is_soft_currency"].all():
        pass
    if (soft["price_type"] == "promo").any():
        raise ValueError("soft-currency rows cannot be promos")
    if soft["store_platform"].nunique() > 0 and not (soft["store_platform"] == "web").all():
        raise ValueError("soft-currency rows must use store_platform=web")

    if (prices["amount"] <= 0).any():
        raise ValueError("amount must be positive")
    if not prices["price_tier_id"].between(1, 10).all():
        raise ValueError("price_tier_id must be between 1 and 10")

    key_cols = [
        "product_id",
        "price_type",
        "promo_code",
        "store_platform",
        "currency_code",
        "effective_from_utc",
    ]
    if prices.duplicated(subset=key_cols).any():
        raise ValueError("duplicate price rows for uniqueness key")

    fiat_set = set(FIAT_CURRENCIES)
    gems_product_ids = set(prices.loc[prices["currency_code"] == "GEMS", "product_id"])
    if gems_product_ids:
        gems_with_fiat = prices[
            prices["product_id"].isin(gems_product_ids)
            & prices["currency_code"].isin(fiat_set)
        ]
        if not gems_with_fiat.empty:
            raise ValueError("GEMS products must not have fiat price rows")
        gems_row_counts = prices[prices["currency_code"] == "GEMS"].groupby("product_id").size()
        if not (gems_row_counts == 1).all():
            raise ValueError("GEMS-only products must have exactly one price row")

    for _, prod in products.iterrows():
        product_type = prod["product_type"]
        tier = int(prod["price_tier_id"])
        pid = prod["product_id"]
        prod_prices = prices[prices["product_id"] == pid]
        if prod_prices.empty:
            raise ValueError(f"{pid}: no price rows generated")

        if gems_eligible(product_type, tier):
            if not (
                len(prod_prices) == 1
                and prod_prices.iloc[0]["currency_code"] == "GEMS"
            ):
                raise ValueError(f"{pid}: GEMS-eligible product must have one GEMS row only")
            continue

        if fiat_eligible(product_type, tier):
            if not prod_prices[~prod_prices["currency_code"].isin({"GOLD", "GEMS"})].empty:
                pass
            else:
                raise ValueError(f"{pid}: fiat-eligible product missing fiat rows")

        if gold_eligible(product_type, tier):
            if not (prod_prices["currency_code"] == "GOLD").any():
                raise ValueError(f"{pid}: GOLD-eligible product missing GOLD row")

        if not fiat_eligible(product_type, tier) and not gold_eligible(product_type, tier):
            raise ValueError(f"{pid}: product has no eligible pricing bucket")

    for _, row in prices.iterrows():
        prod = products.loc[products["product_id"] == row["product_id"]].iloc[0]
        product_from = prod["effective_from_utc"]
        product_to = prod["effective_to_utc"]
        if row["effective_from_utc"] < product_from:
            raise ValueError(f"{row['product_id']}: price effective_from before product launch")
        if pd.notna(product_to) and pd.notna(row["effective_to_utc"]):
            if row["effective_to_utc"] > product_to:
                raise ValueError(f"{row['product_id']}: price effective_to after product end")
        if pd.notna(row["effective_to_utc"]) and row["effective_from_utc"] >= row["effective_to_utc"]:
            raise ValueError(f"{row['product_id']}: price window invalid")


def generate_prices(
    products: pd.DataFrame,
    random_seed: int,
    game_age: int = 180,
) -> pd.DataFrame:
    """
    Generate synthetic price rows for each product in the catalog.

    Parameters
    ----------
    products : pd.DataFrame
        Product catalog from ``generate_products``.
    random_seed : int
        Seed for promo subset sampling — guarantees identical output given the same inputs.
    game_age : int, optional
        How many days the game has been live; used to build season and promo calendars.
        Default 180 (~0.5 years).

    Returns
    -------
    pd.DataFrame
        Product prices table with columns in fixed order.
    """
    _validate_inputs(random_seed, game_age)
    _validate_products(products)

    rng = np.random.default_rng(random_seed)

    today = datetime.date.today()
    launch = pd.Timestamp(_midnight_utc(today - datetime.timedelta(days=game_age)))
    expansion = launch + pd.Timedelta(days=EXPANSION_DAY)

    catalog = products.copy()
    catalog["effective_from_utc"] = pd.to_datetime(catalog["effective_from_utc"], utc=True)
    catalog["effective_to_utc"] = pd.to_datetime(catalog["effective_to_utc"], utc=True)
    catalog = catalog.sort_values("product_id", kind="mergesort")

    promos = _build_promo_calendar(catalog, launch, expansion, game_age, rng)

    rows: list[dict[str, Any]] = []
    for _, product_row in catalog.iterrows():
        product_type = product_row["product_type"]
        tier = int(product_row["price_tier_id"])

        if gems_eligible(product_type, tier):
            rows.append(_emit_gems_row(product_row))
            continue

        if fiat_eligible(product_type, tier):
            rows.extend(_emit_fiat_standard_rows(product_row))
            rows.extend(_emit_fiat_promo_rows(product_row, promos))

        if gold_eligible(product_type, tier):
            rows.append(_emit_gold_row(product_row))

    df = pd.DataFrame(rows)
    df["effective_from_utc"] = pd.to_datetime(df["effective_from_utc"], utc=True)
    df["effective_to_utc"] = pd.to_datetime(df["effective_to_utc"], utc=True)
    df["price_tier_id"] = df["price_tier_id"].astype("int64")
    df["is_soft_currency"] = df["is_soft_currency"].astype("bool")
    df["amount"] = df["amount"].astype("float64")
    df["promo_code"] = df["promo_code"].astype("string")
    df["store_platform"] = df["store_platform"].astype("string")
    df["currency_code"] = df["currency_code"].astype("string")
    df["price_type"] = df["price_type"].astype("string")

    df = df.sort_values(
        ["product_id", "currency_code", "store_platform", "price_type", "effective_from_utc"],
        kind="mergesort",
    ).reset_index(drop=True)

    _validate_prices_invariants(df, catalog)

    return df[COLUMN_ORDER]
