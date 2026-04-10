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

LEAGUE = os.getenv("POE_LEAGUE", "Settlers")  # change each league

POE_NINJA_API = "https://poe.ninja/api/data"
POE_NINJA_BUILDS_ENDPOINT = f"{POE_NINJA_API}/builds"

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

# Rate-limit guards (requests per second)
POE_NINJA_RPS = 2
POE_TRADE_RPS = 1  # GGG enforces strict limits

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
