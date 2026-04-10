"""
PyTorch Dataset classes for the three ML models.

Each dataset loads pre-processed .npz files (from feature_engineering.py)
and yields training samples in the format expected by its model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ml.config import DATA_PROCESSED_DIR

# ── Lazy torch imports (only needed if actually used) ──────────────────

def _import_torch():
    import torch
    return torch

def _import_pyg():
    from torch_geometric.data import Data, Dataset
    return Data, Dataset


# ── Build Tree Dataset ─────────────────────────────────────────────────

class BuildTreeDataset:
    """
    Dataset for the Build Tree Optimizer.

    Each sample is:
    - ``context``: [class_id, level, dps, ehp]
    - ``labels``: binary vector of allocated nodes (num_tree_nodes,)
    - ``score``: combined quality score (scalar)

    The graph (edge_index, node features) is shared across all samples
    and passed separately to the model.
    """

    def __init__(
        self,
        data_path: Path | None = None,
        split: str = "train",
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> None:
        data_path = data_path or (DATA_PROCESSED_DIR / "build_tree_dataset.npz")
        data = np.load(data_path)

        contexts = data["contexts"]
        labels = data["labels"]
        scores = data["combined_scores"]

        # Split
        n = len(contexts)
        rng = np.random.RandomState(seed)
        indices = rng.permutation(n)

        n_test = int(n * test_ratio)
        n_val = int(n * val_ratio)

        if split == "test":
            idx = indices[:n_test]
        elif split == "val":
            idx = indices[n_test : n_test + n_val]
        else:
            idx = indices[n_test + n_val :]

        self.contexts = contexts[idx]
        self.labels = labels[idx]
        self.scores = scores[idx]

    def __len__(self) -> int:
        return len(self.contexts)

    def __getitem__(self, i: int) -> dict[str, Any]:
        torch = _import_torch()
        return {
            "context": torch.from_numpy(self.contexts[i]),
            "labels": torch.from_numpy(self.labels[i]),
            "score": torch.tensor(self.scores[i], dtype=torch.float32),
        }


# ── Gear Suggester Dataset ─────────────────────────────────────────────

class GearSuggesterDataset:
    """
    Dataset for the Gear Improvement Suggester.

    Each sample is:
    - ``gear_features``: (num_slots, features_per_slot)
    - ``context``: [class_id, level, dps, ehp]
    - ``target_slot``: int — which slot was upgraded
    - ``dps_delta``: float — DPS improvement from the upgrade

    Requires a specialised .npz created by a separate upgrade-simulation
    script (see training/train_gear.py for the generation logic).
    """

    def __init__(
        self,
        data_path: Path | None = None,
        split: str = "train",
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> None:
        data_path = data_path or (DATA_PROCESSED_DIR / "gear_suggester_dataset.npz")

        if not data_path.exists():
            # Create a placeholder for development
            _create_placeholder_gear_dataset(data_path)

        data = np.load(data_path)
        gear = data["gear_features"]
        contexts = data["contexts"]
        target_slots = data["target_slots"]
        dps_deltas = data["dps_deltas"]

        n = len(gear)
        rng = np.random.RandomState(seed)
        idx = rng.permutation(n)

        n_test = int(n * test_ratio)
        n_val = int(n * val_ratio)

        if split == "test":
            sel = idx[:n_test]
        elif split == "val":
            sel = idx[n_test : n_test + n_val]
        else:
            sel = idx[n_test + n_val :]

        self.gear = gear[sel]
        self.contexts = contexts[sel]
        self.target_slots = target_slots[sel]
        self.dps_deltas = dps_deltas[sel]

    def __len__(self) -> int:
        return len(self.gear)

    def __getitem__(self, i: int) -> dict[str, Any]:
        torch = _import_torch()
        return {
            "gear_features": torch.from_numpy(self.gear[i]),
            "context": torch.from_numpy(self.contexts[i]),
            "target_slot": torch.tensor(self.target_slots[i], dtype=torch.long),
            "dps_delta": torch.tensor(self.dps_deltas[i], dtype=torch.float32),
        }


def _create_placeholder_gear_dataset(path: Path) -> None:
    """Create a small random placeholder for development/testing."""
    from ml.data.processing.feature_engineering import NUM_STAT_FEATURES, SLOT_NAMES

    n = 100  # placeholder samples
    n_slots = len(SLOT_NAMES)
    feat_per_slot = 6 + NUM_STAT_FEATURES

    np.savez_compressed(
        path,
        gear_features=np.random.randn(n, n_slots, feat_per_slot).astype(np.float32),
        contexts=np.random.randn(n, 4).astype(np.float32),
        target_slots=np.random.randint(0, n_slots, size=n),
        dps_deltas=np.random.randn(n).astype(np.float32) * 100_000,
    )
