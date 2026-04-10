"""
Training loop for the Gear Improvement Suggester.

Two sub-models are trained:
1. **Slot Impact Predictor** (XGBoost) — predicts DPS/EHP delta per slot
2. **Mod Recommender** (LightGBM LambdaMART) — ranks mod candidates

Training data generation:
1. Load builds from poe.ninja
2. For each build, simulate "upgrading" each slot by replacing
   with higher-tier mods
3. Compute DPS/EHP delta using PoB engine (or approximation)
4. Use (gear_features, context) → delta as training pairs

Usage:
    python -m ml.training.train_gear
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from loguru import logger

from ml.config import CHECKPOINT_DIR, DATA_PROCESSED_DIR, GearSuggesterConfig


def generate_slot_impact_data(
    n_builds: int = 5000,
    n_slots: int = 15,
    n_features: int = 37,  # 6 scalar + 31 stat features
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic slot impact training data.

    In production, this would:
    1. Load real builds from poe.ninja
    2. For each build + each slot:
       a. Get current item's contribution to DPS/EHP
       b. Replace with "next tier" item (T1 mods)
       c. Compute DPS/EHP delta via PoB engine
    3. Create (X, y_dps, y_ehp) training tuples

    For now, generates synthetic data to validate the pipeline.

    Returns:
        X: (n_builds * n_slots, features) — gear features + build context
        y_dps: (n_builds * n_slots,) — DPS delta
        y_ehp: (n_builds * n_slots,) — EHP delta
    """
    rng = np.random.RandomState(42)

    # Each row = [slot_features(37) + build_context(4)]
    total_features = n_features + 4  # context: class, level, dps, ehp
    n_total = n_builds * n_slots

    X = np.zeros((n_total, total_features), dtype=np.float32)
    y_dps = np.zeros(n_total, dtype=np.float32)
    y_ehp = np.zeros(n_total, dtype=np.float32)

    for build_idx in range(n_builds):
        class_id = rng.randint(0, 7)
        level = rng.randint(70, 100)
        current_dps = rng.uniform(100_000, 5_000_000)
        current_ehp = rng.uniform(3000, 8000)

        for slot_idx in range(n_slots):
            row_idx = build_idx * n_slots + slot_idx

            # Slot features (synthetic)
            rarity = rng.randint(0, 4)
            ilvl = rng.randint(60, 87)
            quality = rng.randint(0, 21)
            n_prefix = rng.randint(0, 4)
            n_suffix = rng.randint(0, 4)
            n_links = rng.choice([1, 2, 3, 4, 5, 6])
            stat_features = rng.randn(n_features - 6) * 10

            slot_feats = np.concatenate([
                [rarity, ilvl, quality, n_prefix, n_suffix, n_links],
                stat_features,
            ])

            context = np.array([class_id, level, current_dps, current_ehp])
            X[row_idx] = np.concatenate([slot_feats, context])

            # Synthetic target: weapon and body armour have highest impact
            # Slot indices: 0=Weapon1, 3=BodyArmour (see SLOT_NAMES)
            base_impact = {0: 3.0, 1: 1.5, 2: 1.0, 3: 2.5, 4: 0.8, 5: 0.8,
                          6: 0.7, 7: 1.2, 8: 0.9, 9: 0.9}.get(slot_idx, 0.5)
            gap_factor = (86 - ilvl) / 26  # higher gap = more room to improve
            quality_factor = (20 - quality) / 20

            dps_delta = base_impact * gap_factor * current_dps * 0.1 + rng.normal(0, 10000)
            ehp_delta = base_impact * quality_factor * current_ehp * 0.05 + rng.normal(0, 200)

            y_dps[row_idx] = max(0, dps_delta)
            y_ehp[row_idx] = max(0, ehp_delta)

    return X, y_dps, y_ehp


def generate_mod_ranking_data(
    n_queries: int = 3000,
    candidates_per_query: int = 10,
    n_features: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic mod ranking training data.

    Each "query" is a slot + build context.
    Each "candidate" is a possible mod improvement for that slot.
    The label is the DPS gain from that mod change.

    Returns:
        X: (n_queries * candidates_per_query, features)
        y: (n_queries * candidates_per_query,) — relevance score
        groups: (n_queries,) — group sizes for LambdaMART
    """
    rng = np.random.RandomState(42)

    n_total = n_queries * candidates_per_query
    X = rng.randn(n_total, n_features).astype(np.float32)
    y = np.abs(rng.randn(n_total).astype(np.float32)) * 50000

    groups = np.full(n_queries, candidates_per_query, dtype=np.int32)

    return X, y, groups


def train(checkpoint_dir: Path | None = None) -> None:
    """Train both gear suggester sub-models."""
    checkpoint_dir = checkpoint_dir or CHECKPOINT_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    cfg = GearSuggesterConfig()

    # ═══════════════════════════════════════════════════════════════════
    # Model 1: Slot Impact Predictor
    # ═══════════════════════════════════════════════════════════════════
    logger.info("=== Training Slot Impact Predictor ===")

    X, y_dps, y_ehp = generate_slot_impact_data()
    n = len(X)
    idx = np.random.RandomState(42).permutation(n)
    n_test = int(n * 0.1)
    n_val = int(n * 0.1)

    X_train = X[idx[n_test + n_val:]]
    y_dps_train = y_dps[idx[n_test + n_val:]]
    y_ehp_train = y_ehp[idx[n_test + n_val:]]
    X_test = X[idx[:n_test]]
    y_dps_test = y_dps[idx[:n_test]]
    y_ehp_test = y_ehp[idx[:n_test]]

    logger.info(f"Slot predictor: {len(X_train)} train, {n_test} test samples")

    from ml.models.gear_suggester import SlotImpactPredictor

    slot_predictor = SlotImpactPredictor(cfg)

    try:
        slot_predictor.fit(X_train, y_dps_train, y_ehp_train)

        # Evaluate
        dps_pred, ehp_pred = slot_predictor.predict(X_test)
        dps_rmse = np.sqrt(np.mean((dps_pred - y_dps_test) ** 2))
        ehp_rmse = np.sqrt(np.mean((ehp_pred - y_ehp_test) ** 2))
        logger.info(f"Slot predictor — DPS RMSE: {dps_rmse:.0f}, EHP RMSE: {ehp_rmse:.0f}")

        # Save
        slot_predictor.save(
            str(checkpoint_dir / "slot_dps.xgb"),
            str(checkpoint_dir / "slot_ehp.xgb"),
        )
        logger.success("Slot impact predictor saved.")

    except ImportError:
        logger.error("xgboost not installed. Run: pip install xgboost")

    # ═══════════════════════════════════════════════════════════════════
    # Model 2: Mod Recommender (LambdaMART)
    # ═══════════════════════════════════════════════════════════════════
    logger.info("=== Training Mod Recommender ===")

    X_mod, y_mod, groups = generate_mod_ranking_data()

    from ml.models.gear_suggester import ModRecommender

    mod_recommender = ModRecommender(cfg)

    try:
        mod_recommender.fit(X_mod, y_mod, groups)
        mod_recommender.save(str(checkpoint_dir / "mod_recommender.lgb"))
        logger.success("Mod recommender saved.")
    except ImportError:
        logger.error("lightgbm not installed. Run: pip install lightgbm")

    logger.success("Gear suggester training complete.")


# ── CLI ────────────────────────────────────────────────────────────────

def main() -> None:
    import typer
    app = typer.Typer(help="Train the Gear Improvement Suggester")

    @app.command()
    def run() -> None:
        train()

    app()


if __name__ == "__main__":
    main()
