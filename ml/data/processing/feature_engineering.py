"""
Feature engineering pipeline.

Converts raw collected data (poe.ninja builds, parsed PoB exports,
tree JSON) into feature vectors suitable for model training.

Three output feature sets:
1. **Build Tree features** — per-build: (graph, context_vector, labels)
2. **Atlas Tree features** — per-atlas-strategy: (node_returns, economy_context)
3. **Gear Suggester features** — per-item-slot: (slot_features, build_context, target)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from ml.config import (
    DAMAGE_TYPES,
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    MAX_PASSIVE_POINTS,
    TrainingConfig,
)


# ── Stat-text → numeric vector ─────────────────────────────────────────

# Common stat patterns and their feature indices
STAT_PATTERNS: dict[str, int] = {
    "maximum life": 0,
    "fire resistance": 1,
    "cold resistance": 2,
    "lightning resistance": 3,
    "chaos resistance": 4,
    "maximum energy shield": 5,
    "armour": 6,
    "evasion": 7,
    "attack speed": 8,
    "cast speed": 9,
    "critical strike chance": 10,
    "critical strike multiplier": 11,
    "physical damage": 12,
    "fire damage": 13,
    "cold damage": 14,
    "lightning damage": 15,
    "chaos damage": 16,
    "spell damage": 17,
    "damage over time": 18,
    "movement speed": 19,
    "accuracy": 20,
    "mana": 21,
    "strength": 22,
    "dexterity": 23,
    "intelligence": 24,
    "minion": 25,
    "totem": 26,
    "aura": 27,
    "curse": 28,
    "block": 29,
    "spell suppression": 30,
}
NUM_STAT_FEATURES = len(STAT_PATTERNS)

import re

def _parse_stat_value(text: str) -> float:
    """Extract the first numeric value from a stat text line."""
    m = re.search(r"[+-]?(\d+\.?\d*)", text)
    return float(m.group(1)) if m else 0.0


def stat_text_to_vector(stat_lines: list[str]) -> np.ndarray:
    """
    Convert a list of stat text lines into a fixed-length numeric vector.

    Parameters
    ----------
    stat_lines : list[str]
        e.g. ["+10 to maximum Life", "+4% to Fire Resistance"]

    Returns
    -------
    np.ndarray of shape (NUM_STAT_FEATURES,)
    """
    vec = np.zeros(NUM_STAT_FEATURES, dtype=np.float32)
    for line in stat_lines:
        lower = line.lower()
        val = _parse_stat_value(line)
        for pattern, idx in STAT_PATTERNS.items():
            if pattern in lower:
                vec[idx] += val
                break
    return vec


# ── Build Tree feature engineering ─────────────────────────────────────

@dataclass
class BuildTreeSample:
    """One training sample for the Build Tree Optimizer."""
    # Build context
    class_id: int         # 0-6
    ascendancy_id: int    # 0-18 (3 per class)
    level: int
    main_skill_tags: np.ndarray  # one-hot or embedding of skill tags

    # Target: which nodes are allocated (binary mask over tree)
    node_labels: np.ndarray  # shape (num_tree_nodes,) binary

    # Quality label: DPS + EHP score for this build
    dps: float
    ehp: float
    combined_score: float


CLASS_NAME_TO_ID = {
    "Scion": 0, "Marauder": 1, "Ranger": 2, "Witch": 3,
    "Duelist": 4, "Templar": 5, "Shadow": 6,
}

ASCENDANCY_TO_ID: dict[str, int] = {}
_asc_counter = 0
for cls, ascs in {
    "Scion": ["Ascendant"],
    "Marauder": ["Juggernaut", "Berserker", "Chieftain"],
    "Ranger": ["Warden", "Deadeye", "Pathfinder"],
    "Witch": ["Occultist", "Elementalist", "Necromancer"],
    "Duelist": ["Slayer", "Gladiator", "Champion"],
    "Templar": ["Inquisitor", "Hierophant", "Guardian"],
    "Shadow": ["Assassin", "Trickster", "Saboteur"],
}.items():
    for a in ascs:
        ASCENDANCY_TO_ID[a] = _asc_counter
        _asc_counter += 1


def build_tree_features(
    builds_dir: Path | None = None,
    tree_json_path: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    """
    Process raw poe.ninja builds into Build Tree training samples.

    Steps:
    1. Load tree JSON → get node_id → index mapping
    2. For each build, create binary label vector over tree nodes
    3. Extract build context features
    4. Compute combined score: w1*norm_dps + w2*norm_ehp

    Returns path to saved .npz file.
    """
    builds_dir = builds_dir or (DATA_RAW_DIR / "poe_ninja")
    output_dir = output_dir or DATA_PROCESSED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tree node mapping
    if tree_json_path is None:
        tree_json_path = DATA_RAW_DIR / "passive_tree_3_28.json"

    if not tree_json_path.exists():
        logger.warning(f"Tree JSON not found at {tree_json_path}. Run tree_data.py first.")
        # Create placeholder mapping
        node_ids = list(range(1300))  # placeholder
    else:
        tree_data = json.loads(tree_json_path.read_text())
        node_ids = [int(nid) for nid in tree_data["nodes"].keys()]

    node_id_to_idx = {nid: i for i, nid in enumerate(sorted(node_ids))}
    num_nodes = len(node_id_to_idx)

    logger.info(f"Tree has {num_nodes} nodes")

    # Collect all builds
    all_contexts = []
    all_labels = []
    all_scores = []

    build_files = list(builds_dir.glob("*_builds.json"))
    if not build_files:
        logger.warning(f"No build files found in {builds_dir}")
        return output_dir

    for bf in build_files:
        builds = json.loads(bf.read_text())
        logger.info(f"Processing {len(builds)} builds from {bf.name}")

        for build in builds:
            # Context vector
            class_id = CLASS_NAME_TO_ID.get(build.get("class_name", ""), 0)
            level = build.get("level", 0)
            dps = build.get("dps", 0)
            life = build.get("life", 0)
            es = build.get("es", 0)
            ehp = life + es  # simplified

            context = np.array([class_id, level, dps, ehp], dtype=np.float32)
            all_contexts.append(context)

            # Node allocation labels
            labels = np.zeros(num_nodes, dtype=np.float32)
            for nid in build.get("tree_node_ids", []):
                if nid in node_id_to_idx:
                    labels[node_id_to_idx[nid]] = 1.0
            all_labels.append(labels)

            # Score
            all_scores.append(np.array([dps, ehp], dtype=np.float32))

    if not all_contexts:
        logger.warning("No builds processed")
        return output_dir

    # Stack into arrays
    contexts = np.stack(all_contexts)
    labels = np.stack(all_labels)
    scores = np.stack(all_scores)

    # Normalise scores for combined objective
    dps_norm = scores[:, 0] / max(scores[:, 0].max(), 1)
    ehp_norm = scores[:, 1] / max(scores[:, 1].max(), 1)
    combined = 0.6 * dps_norm + 0.4 * ehp_norm  # tuneable weights

    out_path = output_dir / "build_tree_dataset.npz"
    np.savez_compressed(
        out_path,
        contexts=contexts,
        labels=labels,
        scores=scores,
        combined_scores=combined,
    )
    logger.info(f"Saved {len(contexts)} samples → {out_path}")
    return out_path


# ── Gear Suggester feature engineering ─────────────────────────────────

SLOT_NAMES = [
    "Weapon 1", "Weapon 2", "Helmet", "Body Armour",
    "Gloves", "Boots", "Belt", "Amulet",
    "Ring 1", "Ring 2", "Flask 1", "Flask 2",
    "Flask 3", "Flask 4", "Flask 5",
]
SLOT_TO_ID = {s: i for i, s in enumerate(SLOT_NAMES)}

RARITY_TO_ID = {"NORMAL": 0, "MAGIC": 1, "RARE": 2, "UNIQUE": 3}


@dataclass
class GearSlotFeatures:
    """Features for one equipped gear slot."""
    slot_id: int
    rarity_id: int
    item_level: int
    quality: int
    num_prefixes: int
    num_suffixes: int
    stat_vector: np.ndarray  # shape (NUM_STAT_FEATURES,)
    has_influence: bool
    num_links: int


def extract_gear_features(
    items: list[dict[str, Any]],
) -> list[GearSlotFeatures]:
    """
    Extract per-slot features from parsed PoB item data.

    Parameters
    ----------
    items : list[dict]
        From ``ParsedBuild.to_dict()["items"]``.

    Returns
    -------
    list[GearSlotFeatures] — one per equipped slot
    """
    results = []
    for item in items:
        slot_name = item.get("slot", "")
        if slot_name not in SLOT_TO_ID:
            continue

        mods = item.get("mods", [])
        mod_texts = [m["text"] for m in mods if m.get("text")]
        stat_vec = stat_text_to_vector(mod_texts)

        # Count prefixes/suffixes (approximation based on mod_type)
        n_prefix = sum(1 for m in mods if m.get("mod_type") in ("EXPLICIT", "FRACTURED"))
        n_suffix = max(0, len([m for m in mods if m.get("mod_type") in ("EXPLICIT", "FRACTURED")]) - n_prefix)

        # Count links from socket string
        sockets_str = item.get("sockets", "")
        num_links = sockets_str.count("-") + 1 if sockets_str else 0

        results.append(GearSlotFeatures(
            slot_id=SLOT_TO_ID[slot_name],
            rarity_id=RARITY_TO_ID.get(item.get("rarity", "RARE"), 2),
            item_level=item.get("item_level", 0),
            quality=item.get("quality", 0),
            num_prefixes=min(n_prefix, 3),
            num_suffixes=min(n_suffix, 3),
            stat_vector=stat_vec,
            has_influence=bool(item.get("influences")),
            num_links=num_links,
        ))

    return results


def gear_features_to_array(slots: list[GearSlotFeatures]) -> np.ndarray:
    """
    Flatten all gear slot features into a single feature vector.

    Returns shape (len(SLOT_NAMES), 6 + NUM_STAT_FEATURES) padded with zeros
    for empty slots.
    """
    per_slot = 6 + NUM_STAT_FEATURES  # scalar features + stat vector
    result = np.zeros((len(SLOT_NAMES), per_slot), dtype=np.float32)

    for sf in slots:
        row = np.concatenate([
            np.array([
                sf.rarity_id, sf.item_level, sf.quality,
                sf.num_prefixes, sf.num_suffixes, sf.num_links,
            ], dtype=np.float32),
            sf.stat_vector,
        ])
        result[sf.slot_id] = row

    return result


# ── CLI ────────────────────────────────────────────────────────────────

def main() -> None:
    import typer

    app = typer.Typer(help="Build feature engineering pipeline")

    @app.command("build-tree")
    def cmd_build_tree() -> None:
        build_tree_features()

    @app.command("all")
    def cmd_all() -> None:
        build_tree_features()
        logger.info("Feature engineering complete.")

    app()


if __name__ == "__main__":
    main()
