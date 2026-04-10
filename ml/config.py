"""
Centralised configuration for the PoB ML suite.

All tunables — API endpoints, league info, model hyper-parameters,
feature toggles, and training settings — live here so that every module
imports a single source of truth.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"

for _d in (DATA_RAW_DIR, DATA_PROCESSED_DIR, CHECKPOINT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── League / API ───────────────────────────────────────────────────────

LEAGUE = os.getenv("POE_LEAGUE", "Mirage")  # change each league

# poe.ninja API base URLs (community-documented, see:
#   https://github.com/Davenads/poeninjaAPI-2025
#   https://github.com/JakubRak-gamedev/poeninja_API_guide)
#
# PoE 1 ONLY — all builds and economy endpoints target Path of Exile 1.
# PoE 2 leagues (e.g. phrecia2.0, keepers) are intentionally ignored.
#
# Economy endpoints (currency, items) live under /api/data/ (no poe1 prefix).
# Builds use a versioned 2-step flow under /poe1/api/data/ and /poe1/api/builds/.
# Index-state lives under /poe1/api/data/ for PoE 1 league/snapshot metadata.
POE_NINJA_API_BASE = "https://poe.ninja/api/data"
POE_NINJA_POE1_API_BASE = "https://poe.ninja/poe1/api/data"

# Index-state endpoint — returns PoE 1 league info and snapshot versions
# Used for discovering leagues and resolving snapshot versions for builds.
#   GET /poe1/api/data/index-state
#   → { buildLeagues, economyLeagues, snapshotVersions }
#   buildLeagues[].url is the lowercase league slug (e.g. "mirage").
#   snapshotVersions[].version is used in builds API path (e.g. "1056-20260410-17302").
POE_NINJA_INDEX_STATE_ENDPOINT = f"{POE_NINJA_POE1_API_BASE}/index-state"

# Versioned builds base URL — PoE 1 only.
# Step 1: resolve snapshot version from index-state (snapshotVersions[].version).
# Step 2 (bulk overview):
#   GET /poe1/api/builds/{version}/overview
#       ?overview={league_url}&type=0
# Step 2 (individual character):
#   GET /poe1/api/builds/{version}/character
#       ?account={acct}&name={name}&overview={league_url}&type=0&timeMachine=
#
# version   — from snapshotVersions[].version (e.g. "1056-20260410-17302")
# league_url — lowercase slug from buildLeagues[].url (e.g. "mirage")
# type      — numeric: 0=exp, 1=depthsolo
POE_NINJA_BUILDS_BASE = "https://poe.ninja/poe1/api/builds"
POE_NINJA_BUILDS_CHARACTER_BASE = POE_NINJA_BUILDS_BASE  # alias kept for back-compat

# Economy endpoints — returns item/currency prices
POE_NINJA_CURRENCY_ENDPOINT = f"{POE_NINJA_API_BASE}/currencyoverview"
POE_NINJA_ITEM_ENDPOINT = f"{POE_NINJA_API_BASE}/itemoverview"

# Supported economy item types for itemoverview endpoint
POE_NINJA_ITEM_TYPES = [
    "Currency", "Fragment",          # currencyoverview
    "UniqueWeapon", "UniqueArmour", "UniqueAccessory",
    "UniqueFlask", "UniqueJewel", "UniqueRelic",
    "SkillGem", "ClusterJewel",
    "Oil", "Incubator", "Scarab", "Fossil", "Resonator",
    "Essence", "DivinationCard", "Beast", "BaseType",
    "HelmetEnchant", "UniqueMap", "Map",
    "BlightedMap", "BlightRavagedMap",
    "DeliriumOrb", "Invitation", "Memory",
    "Coffin", "AllflameEmber", "Omen",
]

GGG_PASSIVE_TREE_URL = (
    "https://www.pathofexile.com/passive-skill-tree"  # HTML; JSON via JS
)
GGG_TREE_DATA_VERSION = "3.28"  # maps to src/TreeData/3_28

POE_TRADE_API = "https://www.pathofexile.com/api/trade"

# Atlas strategy reference sources
POE_ATLAS_COM_URL = "https://poe-atlas.com"
MAXROLL_ATLAS_PLANNER_URL = "https://maxroll.gg/poe/poe-atlas-tree"
POE_VAULT_ATLAS_URL = "https://www.poe-vault.com/guides/atlas-passive-skill-tree-strategies"
POE_NINJA_ATLAS_URL = "https://poe.ninja/builds/atlas"

# Rate-limit guards
# poe.ninja enforces 12 requests per 5 minutes (= 0.04 rps sustained)
# We use conservative defaults that stay well under the limit.
POE_NINJA_MAX_REQUESTS = 12
POE_NINJA_WINDOW_SECONDS = 300        # 5 minutes
POE_NINJA_RPS = 2                      # burst rate; overall limited by window
POE_TRADE_RPS = 1                      # GGG enforces strict limits

# ── Feature engineering ────────────────────────────────────────────────

# Damage types recognised by CalcOffence.lua
DAMAGE_TYPES = ["Physical", "Lightning", "Cold", "Fire", "Chaos"]

# Mod types from Global.lua
MOD_TYPES = ["BASE", "INC", "MORE", "OVERRIDE", "FLAG"]

# Maximum number of passive points (normal + quest + book of regrets)
MAX_PASSIVE_POINTS = 123

# Maximum atlas passive points
MAX_ATLAS_POINTS = 132

# Atlas farming strategies (sourced from poe-atlas.com categories)
# Each strategy maps to a set of mechanic weights the model uses.
ATLAS_STRATEGIES: dict[str, dict[str, float]] = {
    # Phase-based
    "league_start": {
        "MapSustain": 1.0, "Kirac": 0.8, "Generic": 0.6,
    },
    "early_mapping": {
        "MapSustain": 0.8, "Scarab": 0.6, "Shrine": 0.5, "Strongbox": 0.5,
    },
    # Mechanic-specific (from poe-atlas.com guide categories)
    "scarab_shrine_strongbox": {
        "Scarab": 1.0, "Shrine": 0.9, "Strongbox": 0.8, "Generic": 0.3,
    },
    "harvest": {
        "Harvest": 1.0, "Generic": 0.2,
    },
    "expedition": {
        "Expedition": 1.0, "Shrine": 0.4, "Generic": 0.2,
    },
    "heist": {
        "Heist": 1.0, "Generic": 0.2,
    },
    "blight": {
        "Blight": 1.0, "Generic": 0.2,
    },
    "delirium": {
        "Delirium": 1.0, "Beyond": 0.5, "Generic": 0.3,
    },
    "legion": {
        "Legion": 1.0, "Generic": 0.3,
    },
    "breach": {
        "Breach": 1.0, "Delirium": 0.4, "Generic": 0.2,
    },
    "abyss": {
        "Abyss": 1.0, "Delirium": 0.4, "Generic": 0.2,
    },
    "bossing": {
        "Boss": 1.0, "Maven": 0.9, "MapSustain": 0.3,
    },
    "ritual_beyond": {
        "Ritual": 1.0, "Beyond": 0.9, "Generic": 0.3,
    },
    "fortress_farming": {
        "Fortress": 1.0, "Boss": 0.5, "Generic": 0.3,
    },
    # Combo strategies
    "breach_delirium": {
        "Breach": 1.0, "Delirium": 1.0, "Generic": 0.2,
    },
    "expedition_shrines_altars": {
        "Expedition": 1.0, "Shrine": 0.8, "Altar": 0.7, "Generic": 0.2,
    },
}


# ── Model hyper-parameters ─────────────────────────────────────────────

@dataclass
class BuildTreeModelConfig:
    """GAT-based passive-tree optimiser."""

    node_feature_dim: int = 64
    hidden_dim: int = 128
    num_gat_heads: int = 4
    num_gat_layers: int = 3
    dropout: float = 0.15
    context_embed_dim: int = 32
    # RL (PPO) alternative
    rl_gamma: float = 0.99
    rl_clip_eps: float = 0.2
    rl_entropy_coef: float = 0.01


@dataclass
class AtlasModelConfig:
    """ML + ILP atlas-tree optimiser."""

    n_estimators: int = 500
    max_depth: int = 8
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    # ILP solver
    solver_time_limit_s: int = 30


@dataclass
class GearSuggesterConfig:
    """Slot ranker + mod recommender."""

    # Slot impact predictor (XGBoost)
    slot_n_estimators: int = 300
    slot_max_depth: int = 6
    slot_learning_rate: float = 0.05
    # Mod recommender (LambdaMART via LightGBM)
    mod_n_estimators: int = 400
    mod_num_leaves: int = 63
    mod_learning_rate: float = 0.05


# ── Training ───────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    """Shared training settings."""

    seed: int = 42
    batch_size: int = 64
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-5
    early_stop_patience: int = 10
    val_split: float = 0.1
    test_split: float = 0.1
    num_workers: int = 4
    checkpoint_every: int = 5
    # poe.ninja data collection
    max_builds_per_class: int = 10_000
    classes: list[str] = field(
        default_factory=lambda: [
            "Marauder", "Ranger", "Witch",
            "Duelist", "Templar", "Shadow", "Scion",
        ]
    )
