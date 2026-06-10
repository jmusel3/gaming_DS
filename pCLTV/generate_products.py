"""
Synthetic product catalog for a live-service collectible card game CLTV project.

v1 schema
---------
product_id, product_name, product_type, items_amount, items_array, loot_table_id,
effective_from_utc, effective_to_utc, is_active, price_tier_id

``price_tier_id`` is a semantic value scale (1–10); localized amounts, platform
markups, and promos are resolved by ``generate_prices``.

Inventory (game_age=180)
------------------------
750 cards (500 launch + 250 expansion at day 90), 150 cosmetics (100 + 50),
20 decks, 20 card packs (7 gacha / 7 fixed / 6 flex), 10 cosmetic packs
(5 gacha / 5 fixed), 4 gem packs, 3 battle passes per 30-day season,
4 limited-time events per season.
"""

from __future__ import annotations

import datetime
import math

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Inventory constants
# ---------------------------------------------------------------------------

N_LAUNCH_CARDS = 500
N_EXPANSION_CARDS = 250
N_LAUNCH_COSMETICS = 100
N_EXPANSION_COSMETICS = 50
N_DECKS = 20
N_EXPANSION_DECKS = 5
N_CARD_PACKS_GACHA = 7
N_CARD_PACKS_FIXED = 7
N_CARD_PACKS_FLEX = 6
N_EXPANSION_FIXED_CARD_PACKS = 2
N_COSMETIC_PACKS_GACHA = 5
N_COSMETIC_PACKS_FIXED = 5
N_EXPANSION_FIXED_COSMETIC_PACKS = 1

EXPANSION_DAY = 90
SEASON_LENGTH_DAYS = 30
EVENTS_PER_SEASON = 4
EVENT_LENGTH_DAYS = 7

DECK_SIZE = 30
PACK_SIZES = [5, 10, 15]

# ---------------------------------------------------------------------------
# Card naming
# ---------------------------------------------------------------------------

ELEMENTS = ["Fire", "Water", "Earth", "Wind", "Lightning", "Shadow", "Light"]

ELEMENT_FLAVOR_WORDS: dict[str, list[str]] = {
    "Fire": [
        "Emberfang", "Cinderveil", "Blazeheart", "Ashen", "Pyroclast",
        "Scorchwind", "Flarecrest", "Magmaforged", "Infernal", "Smoldering",
        "Charbrand", "Sunscorched", "Volcanic", "Kindled",
        "Hellfire", "Blazeborn", "Ashwrought", "Pyreheart", "Searstone",
        "Flameborn", "Burnished", "Cinderfall", "Sparkwrought", "Molten",
        "Heatwave", "Firebrand", "Ashfall", "Hotspark", "Coalborn",
        "Flareborn", "Wildfire", "Smokeveil", "Blistering", "Furnaceborn",
    ],
    "Water": [
        "Tidal", "Deepcurrent", "Frostbrine", "Wavecaller", "Abyssal",
        "Mistveil", "Coralbound", "Stormtide", "Glacial", "Riptide",
        "Pearlescent", "Brinewrought", "Undertow", "Drowned",
        "Brineborn", "Seaborn", "Icebound", "Foamcrest", "Depthcaller",
        "Saltwrought", "Moontide", "Rainveil", "Sprayborn", "Hydroforge",
        "Currentborn", "Reefbound", "Tidewrought", "Saltcrest", "Iceveil",
        "Floodborn", "Streamborn", "Bubblecrest", "Cryosea", "Maelstrom",
    ],
    "Earth": [
        "Stoneheart", "Verdant", "Mossbound", "Granite", "Rootwoven",
        "Quaking", "Ironvein", "Thornclad", "Loamborn", "Boulderfist",
        "Petrified", "Wildgrove", "Earthen", "Bramble",
        "Bedrock", "Rootbound", "Clayborn", "Mountainheart", "Fungal",
        "Sandwrought", "Cragborn", "Sapbound", "Terracotta", "Groveheart",
        "Mudborn", "Slatelock", "Vinebound", "Hillborn", "Dustwrought",
        "Ferngrove", "Pebbleborn", "Thicket", "Orevein", "Mossheart",
    ],
    "Wind": [
        "Galeborn", "Zephyr", "Skydancer", "Tempest", "Whirlwind",
        "Cloudpiercer", "Aether", "Breezewoven", "Cyclonic", "Featherlight",
        "Stratos", "Windswept", "Soaring", "Squall",
        "Skyborn", "Draftwrought", "Gustveil", "Jetstream", "Cloudborn",
        "Airweaver", "Stormwing", "Highgale", "Windrider", "Vortexborn",
        "Skystorm", "Galewrought", "Updraft", "Mistral", "Aircrest",
        "Gustborn", "Tornado", "Breezeborn", "Stratoclimb", "Windshaper",
    ],
    "Lightning": [
        "Stormpiercer", "Voltaic", "Thunderborn", "Arcflash", "Crackling",
        "Ionbound", "Skysplitter", "Surgewrought", "Galvanic", "Boltforged",
        "Stormcalled", "Fulminant", "Chargebound", "Thunderhead",
        "Boltborn", "Staticveil", "Overcharge", "Stormforge", "Sparkborn",
        "Voltcrest", "Thunderwrought", "Plasmaborn", "Shockwave", "Arclight",
        "Livewire", "Sparkveil", "Ioncrest", "Zapborn", "Thunderclap",
        "Voltwrought", "Chainbolt", "Stormlash", "Flashpoint", "Surgebirth",
    ],
    "Shadow": [
        "Umbral", "Nightveil", "Duskborn", "Gloomwrought", "Shadewoven",
        "Eclipsed", "Murkbound", "Veilpiercer", "Darkwhisper", "Voidtouched",
        "Twilight", "Nocturnal", "Grimshade", "Phantasmal",
        "Nightborn", "Voidveil", "Gloomcrest", "Dreadwrought", "Shadeborn",
        "Blackened", "Wraithbound", "Abyssborn", "Spectral", "Duskfall",
        "Shadowmere", "Deathveil", "Gloomfang", "Nightshroud", "Hollowborn",
        "Dimming", "Pitborn", "Shadeclad", "Mirken", "Obsidianveil",
    ],
    "Light": [
        "Dawnforged", "Radiant", "Sunblessed", "Luminous", "Halcyon",
        "Gleaming", "Aurelian", "Brightwoven", "Solarflare", "Hallowed",
        "Prismatic", "Lightbringer", "Glorybound", "Celestine",
        "Starborn", "Sunforge", "Beaconwrought", "Purelight", "Dawnbreak",
        "Haloborn", "Crystalveil", "Goldwoven", "Seraphic", "Daybreak",
        "Sunbeam", "Lucent", "Brightstar", "Goldflare", "Pearlcrest",
        "Shimmering", "Lightveil", "Saintborn", "Solstice", "Haloheart",
    ],
}

CARD_CLASSES = ["Creature", "Spell", "Equipment"]

CLASS_NOUNS: dict[str, list[str]] = {
    "Creature": [
        "Drake", "Golem", "Wyrm", "Sentinel", "Stalker", "Behemoth",
        "Sprite", "Warden", "Colossus", "Serpent", "Gryphon", "Revenant",
        "Hydra", "Chimera", "Basilisk", "Direwolf",
        "Leviathan", "Phoenix", "Kraken", "Titan", "Avatar", "Primordial",
        "Sovereign", "Guardian", "Manticore", "Roc", "Elemental", "Oracle",
        "Lich", "Gargoyle", "Giant", "Ogre", "Siren", "Cerberus", "Imp",
        "Nymph",
    ],
    "Spell": [
        "Sigil", "Burst", "Ritual", "Invocation", "Torrent", "Hex",
        "Benediction", "Cascade", "Rupture", "Ward", "Conflux", "Decree",
        "Surge", "Omen", "Litany", "Requiem",
        "Evocation", "Cantrip", "Conjuration", "Eclipse", "Resonance",
        "Annihilation", "Blessing", "Incantation", "Glyph", "Pulse",
        "Channel", "Manifest", "Dissipation", "Binding", "Unravel", "Flux",
        "Echo", "Purge", "Mirage", "Nova",
    ],
    "Equipment": [
        "Lance", "Blade", "Aegis", "Gauntlet", "Talisman", "Warplate",
        "Crown", "Scepter", "Longbow", "Dagger", "Bulwark", "Pendant",
        "Greaves", "Halberd", "Quiver", "Signet",
        "Katana", "Chakram", "Orb", "Hammer", "Shield", "Amulet", "Corslet",
        "Saber", "Staff", "Buckler", "Claymore", "Trident", "Spear", "Mace",
        "Censer", "Codex", "Bracelet", "Pauldron", "Ankh", "Grimoire",
    ],
}

CARD_RARITIES = ["common", "rare", "epic", "legendary"]
CARD_RARITY_WEIGHTS = [0.60, 0.27, 0.10, 0.03]
# Legendary cards jump to the top of the 10-tier price scale.
CARD_RARITY_PRICE_TIER = {"common": 1, "rare": 2, "epic": 3, "legendary": 10}

# ---------------------------------------------------------------------------
# Cosmetic naming
# ---------------------------------------------------------------------------

ACCESSORY_TYPES = [
    "Headwear", "Jacket", "Pants", "Boots", "Gloves", "Cape", "Mask", "Banner",
    "Shoulders", "Belt", "Emote", "CardBack",
]

# Launch cosmetics draw from the first 6 families; the expansion introduces 2 more.
FLAVOR_FAMILIES = [
    "Emperor", "Swordsman", "Dragonkin", "Arcanist", "Shogun", "Corsair",
    "Frostborn", "Sylvan", "Ronin", "Paladin", "Necromancer", "Ranger",
]
N_LAUNCH_FAMILIES = 6

COSMETIC_ADJECTIVES = [
    "Crimson", "Gilded", "Obsidian", "Ivory", "Azure", "Verdigris",
    "Scarlet", "Onyx", "Pearl", "Cobalt", "Amber", "Violet", "Jade", "Silver",
    "Golden", "Midnight", "Frosted", "Ember", "Shadow", "Celestial", "Copper",
    "Bronze",
]

# ---------------------------------------------------------------------------
# Bundle / seasonal naming
# ---------------------------------------------------------------------------

DECK_STYLES = [
    "Starter", "Champion's", "Vanguard", "Ascendant", "Master's",
    "Initiate's", "Elite", "Grandmaster's", "Legendary", "Mythic",
]

GACHA_CARD_PACK_NAMES = [
    "Standard Booster", "Premium Booster", "Elemental Surge Booster",
    "Mythic Booster", "Shadowfall Booster", "Dawnlight Booster",
    "Wildcard Booster",
]

# (name, theme_type, theme_value) — pools for fixed card packs.
FIXED_CARD_PACK_THEMES = [
    ("Swordsman's Arsenal", "class", "Equipment"),
    ("Beastmaster's Menagerie", "class", "Creature"),
    ("Archmage's Grimoire", "class", "Spell"),
    ("Inferno Cache", "element", "Fire"),
    ("Tidecaller's Trove", "element", "Water"),
    ("Stormbringer's Cache", "element", "Lightning"),
    ("Umbral Reliquary", "element", "Shadow"),
]

FLEX_CARD_PACK_NAMES = [
    "Novice Flex Pack", "Adept Flex Pack", "Expert Flex Pack",
    "Master Flex Pack", "Grandmaster Flex Pack", "Legend Flex Pack",
]

GACHA_COSMETIC_PACK_NAMES = [
    "Wardrobe Capsule", "Couture Capsule", "Regalia Capsule",
    "Masquerade Capsule", "Heritage Capsule",
]

BATTLE_PASS_TIERS = [
    ("PREMIUM", "Premium Battle Pass", 5),
    ("PREMPLUS", "Premium Plus Battle Pass", 7),
    ("VIP", "VIP Battle Pass", 9),
]

EVENT_THEMES = [
    "Dragon's Hoard", "Mirror Arena", "Elemental Clash", "Guild Gauntlet",
    "Twilight Trials", "Treasure Tides", "Champion's Crucible",
    "Runebound Rumble", "Festival of Embers", "Frostfall Skirmish",
    "Skyward Tournament", "Relic Hunt",
    "Crown Clash", "Arcane Auction", "Siege of Spires", "Phantom Pursuit",
    "Solar Solstice", "Void Vanguard", "Crystal Carnival", "Warlord's Wake",
    "Mystic Milestone", "Iron Invitational", "Bloom Blitz", "Obsidian Onslaught",
]

GEM_PACKS = [
    ("GEM1", "Handful of Gems (500)", 2),
    ("GEM2", "Pouch of Gems (1200)", 5),
    ("GEM3", "Chest of Gems (2600)", 8),
    ("GEM4", "Vault of Gems (6500)", 10),
]

PACK_SIZE_PRICE_TIER = {5: 3, 10: 5, 15: 6}
DECK_PRICE_TIER = 6
EVENT_PRICE_TIER = 4

COLUMN_ORDER = [
    "product_id",
    "product_name",
    "product_type",
    "items_amount",
    "items_array",
    "loot_table_id",
    "effective_from_utc",
    "effective_to_utc",
    "is_active",
    "price_tier_id",
]


def _validate_inputs(random_seed: int, game_age: int) -> None:
    if not isinstance(random_seed, int) or isinstance(random_seed, bool):
        raise ValueError("random_seed must be an integer")
    if not isinstance(game_age, int) or isinstance(game_age, bool):
        raise ValueError("game_age must be an integer")
    if game_age < 1:
        raise ValueError("game_age must be >= 1")


def _midnight_utc(d: datetime.date) -> datetime.datetime:
    return datetime.datetime(d.year, d.month, d.day, tzinfo=datetime.timezone.utc)


def _balanced_assignment(values: list[str], n: int, rng: np.random.Generator) -> list[str]:
    """Near-even split of n slots across values, order shuffled."""
    reps = math.ceil(n / len(values))
    assigned = np.tile(np.array(values, dtype=object), reps)[:n]
    rng.shuffle(assigned)
    return [str(v) for v in assigned]


def _row(
    product_id: str,
    product_name: str,
    product_type: str,
    items_amount: int,
    items_array: list[str] | None,
    loot_table_id: str | None,
    effective_from: datetime.datetime,
    effective_to: datetime.datetime | None,
    price_tier_id: int,
) -> dict:
    return {
        "product_id": product_id,
        "product_name": product_name,
        "product_type": product_type,
        "items_amount": items_amount,
        "items_array": items_array,
        "loot_table_id": loot_table_id,
        "effective_from_utc": effective_from,
        "effective_to_utc": effective_to,
        "price_tier_id": price_tier_id,
    }


# ---------------------------------------------------------------------------
# Cards and cosmetics
# ---------------------------------------------------------------------------

def _build_cards(
    rng: np.random.Generator,
    launch: datetime.datetime,
    expansion: datetime.datetime,
) -> list[dict]:
    n_total = N_LAUNCH_CARDS + N_EXPANSION_CARDS
    elements = _balanced_assignment(ELEMENTS, n_total, rng)
    classes = _balanced_assignment(CARD_CLASSES, n_total, rng)
    rarities = rng.choice(CARD_RARITIES, size=n_total, p=CARD_RARITY_WEIGHTS)

    used_names: set[str] = set()
    rows: list[dict] = []
    for i in range(n_total):
        element = elements[i]
        card_class = classes[i]
        while True:
            flavor = str(rng.choice(ELEMENT_FLAVOR_WORDS[element]))
            noun = str(rng.choice(CLASS_NOUNS[card_class]))
            name = f"{flavor} {noun} ({element} {card_class})"
            if name not in used_names:
                break
        used_names.add(name)

        is_expansion = i >= N_LAUNCH_CARDS
        row = _row(
            product_id=f"CARD{i + 1:04d}",
            product_name=name,
            product_type="card",
            items_amount=1,
            items_array=None,
            loot_table_id=None,
            effective_from=expansion if is_expansion else launch,
            effective_to=None,
            price_tier_id=CARD_RARITY_PRICE_TIER[str(rarities[i])],
        )
        row["_element"] = element
        row["_class"] = card_class
        row["_expansion"] = is_expansion
        rows.append(row)
    return rows


def _build_cosmetics(
    rng: np.random.Generator,
    launch: datetime.datetime,
    expansion: datetime.datetime,
) -> list[dict]:
    launch_families = FLAVOR_FAMILIES[:N_LAUNCH_FAMILIES]
    family_slots = (
        _balanced_assignment(launch_families, N_LAUNCH_COSMETICS, rng)
        + _balanced_assignment(FLAVOR_FAMILIES, N_EXPANSION_COSMETICS, rng)
    )

    used_names: set[str] = set()
    rows: list[dict] = []
    for i, family in enumerate(family_slots):
        while True:
            adjective = str(rng.choice(COSMETIC_ADJECTIVES))
            accessory = str(rng.choice(ACCESSORY_TYPES))
            name = f"{adjective} {family} {accessory}"
            if name not in used_names:
                break
        used_names.add(name)

        is_expansion = i >= N_LAUNCH_COSMETICS
        row = _row(
            product_id=f"COS{i + 1:03d}",
            product_name=name,
            product_type="player_cosmetic",
            items_amount=1,
            items_array=None,
            loot_table_id=None,
            effective_from=expansion if is_expansion else launch,
            effective_to=None,
            price_tier_id=int(rng.integers(1, 5)),
        )
        row["_family"] = family
        row["_expansion"] = is_expansion
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Decks and packs
# ---------------------------------------------------------------------------

def _sample_ids(pool: list[str], size: int, rng: np.random.Generator) -> list[str]:
    if len(pool) < size:
        raise ValueError(f"Pool of {len(pool)} too small to sample {size} items")
    return [str(x) for x in rng.choice(np.array(pool, dtype=object), size=size, replace=False)]


def _build_decks(
    rng: np.random.Generator,
    cards: list[dict],
    launch: datetime.datetime,
    expansion: datetime.datetime,
) -> list[dict]:
    deck_elements = _balanced_assignment(ELEMENTS, N_DECKS, rng)
    expansion_idx = set(rng.choice(N_DECKS, size=N_EXPANSION_DECKS, replace=False).tolist())

    style_counter: dict[str, int] = {}
    rows: list[dict] = []
    for i in range(N_DECKS):
        element = deck_elements[i]
        is_expansion = i in expansion_idx
        style = DECK_STYLES[style_counter.get(element, 0)]
        style_counter[element] = style_counter.get(element, 0) + 1

        pool = [
            c["product_id"]
            for c in cards
            if c["_element"] == element and (is_expansion or not c["_expansion"])
        ]
        rows.append(
            _row(
                product_id=f"DECK{i + 1:02d}",
                product_name=f"{element} {style} Deck",
                product_type="deck",
                items_amount=DECK_SIZE,
                items_array=_sample_ids(pool, DECK_SIZE, rng),
                loot_table_id=None,
                effective_from=expansion if is_expansion else launch,
                effective_to=None,
                price_tier_id=DECK_PRICE_TIER,
            )
        )
    return rows


def _build_card_packs(
    rng: np.random.Generator,
    cards: list[dict],
    launch: datetime.datetime,
    expansion: datetime.datetime,
) -> list[dict]:
    rows: list[dict] = []
    pack_seq = 0

    for name in GACHA_CARD_PACK_NAMES:
        pack_seq += 1
        product_id = f"CPK{pack_seq:02d}"
        amount = int(rng.choice(PACK_SIZES))
        rows.append(
            _row(
                product_id=product_id,
                product_name=name,
                product_type="card_pack_gacha",
                items_amount=amount,
                items_array=None,
                loot_table_id=f"LT_{product_id}",
                effective_from=launch,
                effective_to=None,
                price_tier_id=PACK_SIZE_PRICE_TIER[amount],
            )
        )

    expansion_fixed = set(
        rng.choice(
            N_CARD_PACKS_FIXED, size=N_EXPANSION_FIXED_CARD_PACKS, replace=False
        ).tolist()
    )
    for j, (name, theme_type, theme_value) in enumerate(FIXED_CARD_PACK_THEMES):
        pack_seq += 1
        is_expansion = j in expansion_fixed
        amount = int(rng.choice(PACK_SIZES))
        meta_key = "_element" if theme_type == "element" else "_class"
        pool = [
            c["product_id"]
            for c in cards
            if c[meta_key] == theme_value and (is_expansion or not c["_expansion"])
        ]
        rows.append(
            _row(
                product_id=f"CPK{pack_seq:02d}",
                product_name=f"{name} Pack",
                product_type="card_pack_fixed",
                items_amount=amount,
                items_array=_sample_ids(pool, amount, rng),
                loot_table_id=None,
                effective_from=expansion if is_expansion else launch,
                effective_to=None,
                price_tier_id=PACK_SIZE_PRICE_TIER[amount],
            )
        )

    for name in FLEX_CARD_PACK_NAMES:
        pack_seq += 1
        amount = int(rng.choice(PACK_SIZES))
        rows.append(
            _row(
                product_id=f"CPK{pack_seq:02d}",
                product_name=name,
                product_type="card_pack_flex",
                items_amount=amount,
                items_array=None,
                loot_table_id=None,
                effective_from=launch,
                effective_to=None,
                price_tier_id=PACK_SIZE_PRICE_TIER[amount],
            )
        )
    return rows


def _build_cosmetic_packs(
    rng: np.random.Generator,
    cosmetics: list[dict],
    launch: datetime.datetime,
    expansion: datetime.datetime,
) -> list[dict]:
    rows: list[dict] = []
    pack_seq = 0

    for name in GACHA_COSMETIC_PACK_NAMES:
        pack_seq += 1
        product_id = f"KPK{pack_seq:02d}"
        amount = int(rng.choice(PACK_SIZES))
        rows.append(
            _row(
                product_id=product_id,
                product_name=name,
                product_type="cosmetic_pack_gacha",
                items_amount=amount,
                items_array=None,
                loot_table_id=f"LT_{product_id}",
                effective_from=launch,
                effective_to=None,
                price_tier_id=PACK_SIZE_PRICE_TIER[amount],
            )
        )

    launch_families = FLAVOR_FAMILIES[:N_LAUNCH_FAMILIES]
    set_families = [
        str(f)
        for f in rng.choice(
            np.array(launch_families, dtype=object),
            size=N_COSMETIC_PACKS_FIXED,
            replace=False,
        )
    ]
    expansion_fixed = set(
        rng.choice(
            N_COSMETIC_PACKS_FIXED, size=N_EXPANSION_FIXED_COSMETIC_PACKS, replace=False
        ).tolist()
    )
    for j, family in enumerate(set_families):
        pack_seq += 1
        is_expansion = j in expansion_fixed
        amount = int(rng.choice(PACK_SIZES))
        pool = [
            c["product_id"]
            for c in cosmetics
            if c["_family"] == family and (is_expansion or not c["_expansion"])
        ]
        rows.append(
            _row(
                product_id=f"KPK{pack_seq:02d}",
                product_name=f"{family} Regalia Set",
                product_type="cosmetic_pack_fixed",
                items_amount=amount,
                items_array=_sample_ids(pool, amount, rng),
                loot_table_id=None,
                effective_from=expansion if is_expansion else launch,
                effective_to=None,
                price_tier_id=PACK_SIZE_PRICE_TIER[amount],
            )
        )
    return rows


def _build_gem_packs(launch: datetime.datetime) -> list[dict]:
    return [
        _row(
            product_id=product_id,
            product_name=name,
            product_type="gem_pack",
            items_amount=1,
            items_array=None,
            loot_table_id=None,
            effective_from=launch,
            effective_to=None,
            price_tier_id=tier,
        )
        for product_id, name, tier in GEM_PACKS
    ]


# ---------------------------------------------------------------------------
# Seasonal products
# ---------------------------------------------------------------------------

def _season_windows(
    launch: datetime.datetime, game_age: int
) -> list[tuple[int, datetime.datetime, datetime.datetime]]:
    n_seasons = math.ceil(game_age / SEASON_LENGTH_DAYS)
    windows = []
    for k in range(n_seasons):
        start = launch + datetime.timedelta(days=SEASON_LENGTH_DAYS * k)
        end = launch + datetime.timedelta(days=SEASON_LENGTH_DAYS * (k + 1))
        windows.append((k + 1, start, end))
    return windows


def _build_battle_passes(launch: datetime.datetime, game_age: int) -> list[dict]:
    rows: list[dict] = []
    for season, start, end in _season_windows(launch, game_age):
        for suffix, label, tier in BATTLE_PASS_TIERS:
            rows.append(
                _row(
                    product_id=f"BP_S{season:02d}_{suffix}",
                    product_name=f"Season {season:02d} {label}",
                    product_type="battle_pass",
                    items_amount=1,
                    items_array=None,
                    loot_table_id=None,
                    effective_from=start,
                    effective_to=end,
                    price_tier_id=tier,
                )
            )
    return rows


def _build_limited_time_events(launch: datetime.datetime, game_age: int) -> list[dict]:
    rows: list[dict] = []
    event_counter = 0
    for season, start, _end in _season_windows(launch, game_age):
        for j in range(EVENTS_PER_SEASON):
            theme = EVENT_THEMES[event_counter % len(EVENT_THEMES)]
            event_counter += 1
            event_start = start + datetime.timedelta(days=EVENT_LENGTH_DAYS * j)
            rows.append(
                _row(
                    product_id=f"LTE_S{season:02d}_E{j + 1}",
                    product_name=f"{theme} Event (S{season:02d}E{j + 1})",
                    product_type="limited_time_event",
                    items_amount=1,
                    items_array=None,
                    loot_table_id=None,
                    effective_from=event_start,
                    effective_to=event_start + datetime.timedelta(days=EVENT_LENGTH_DAYS),
                    price_tier_id=EVENT_PRICE_TIER,
                )
            )
    return rows


# ---------------------------------------------------------------------------
# Invariant checks
# ---------------------------------------------------------------------------

ARRAY_TYPES = {"deck", "card_pack_fixed", "cosmetic_pack_fixed"}
GACHA_TYPES = {"card_pack_gacha", "cosmetic_pack_gacha"}


def _validate_catalog_invariants(df: pd.DataFrame, now: datetime.datetime) -> None:
    if df["product_id"].duplicated().any():
        raise ValueError("Duplicate product_id values")
    if df["product_name"].duplicated().any():
        raise ValueError("Duplicate product_name values")

    effective_from_by_id = dict(zip(df["product_id"], df["effective_from_utc"]))

    for _, row in df.iterrows():
        ptype = row["product_type"]
        items = row["items_array"]

        if ptype in ARRAY_TYPES:
            if not isinstance(items, list):
                raise ValueError(f"{row['product_id']}: items_array missing")
            if len(items) != row["items_amount"]:
                raise ValueError(f"{row['product_id']}: items_array length mismatch")
            if len(set(items)) != len(items):
                raise ValueError(f"{row['product_id']}: duplicate items in items_array")
            for item_id in items:
                if item_id not in effective_from_by_id:
                    raise ValueError(f"{row['product_id']}: unknown item {item_id}")
                if effective_from_by_id[item_id] > row["effective_from_utc"]:
                    raise ValueError(
                        f"{row['product_id']}: item {item_id} effective after bundle"
                    )
        elif items is not None:
            raise ValueError(f"{row['product_id']}: unexpected items_array")

        has_loot = pd.notna(row["loot_table_id"])
        if (ptype in GACHA_TYPES) != has_loot:
            raise ValueError(f"{row['product_id']}: loot_table_id inconsistent")

        starts_past = row["effective_from_utc"] <= now
        not_ended = pd.isna(row["effective_to_utc"]) or now < row["effective_to_utc"]
        if bool(row["is_active"]) != (starts_past and not_ended):
            raise ValueError(f"{row['product_id']}: is_active inconsistent with window")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_products(random_seed: int, game_age: int = 180) -> pd.DataFrame:
    """
    Generate the synthetic product catalog with reproducible randomness.

    Parameters
    ----------
    random_seed : int
        Seed for all random draws — guarantees identical output given the same inputs.
    game_age : int, optional
        How many days the game has been live. Launch is `today - game_age` days;
        the expansion wave activates 90 days after launch (future-dated and inactive
        if game_age < 90). Battle passes and limited-time events scale with game_age.
        Default 180 (~0.5 years).

    Returns
    -------
    pd.DataFrame
        Product catalog table with columns in fixed order.
    """
    _validate_inputs(random_seed, game_age)
    rng = np.random.default_rng(random_seed)

    today = datetime.date.today()
    launch = _midnight_utc(today - datetime.timedelta(days=game_age))
    expansion = launch + datetime.timedelta(days=EXPANSION_DAY)
    now = datetime.datetime.now(datetime.timezone.utc)

    cards = _build_cards(rng, launch, expansion)
    cosmetics = _build_cosmetics(rng, launch, expansion)

    rows = (
        cards
        + cosmetics
        + _build_decks(rng, cards, launch, expansion)
        + _build_card_packs(rng, cards, launch, expansion)
        + _build_cosmetic_packs(rng, cosmetics, launch, expansion)
        + _build_gem_packs(launch)
        + _build_battle_passes(launch, game_age)
        + _build_limited_time_events(launch, game_age)
    )

    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])

    df["effective_from_utc"] = pd.to_datetime(df["effective_from_utc"], utc=True)
    df["effective_to_utc"] = pd.to_datetime(df["effective_to_utc"], utc=True)
    now_ts = pd.Timestamp(now)
    df["is_active"] = (df["effective_from_utc"] <= now_ts) & (
        df["effective_to_utc"].isna() | (now_ts < df["effective_to_utc"])
    )
    df["loot_table_id"] = df["loot_table_id"].astype("string")
    df["items_amount"] = df["items_amount"].astype("int64")
    df["price_tier_id"] = df["price_tier_id"].astype("int64")

    _validate_catalog_invariants(df, now)

    return df[COLUMN_ORDER]
