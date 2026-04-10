"""
Training loop for the Atlas Tree Optimizer.

Since the atlas tree model is an XGBoost regressor + ILP solver,
training only applies to the ML return-prediction stage.

Training data is generated from:
1. Community-shared atlas trees (labeled with reported returns)
2. Simulated mapping runs using datamined drop tables
3. Economy snapshots from poe.ninja

Usage:
    python -m ml.training.train_atlas
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from loguru import logger

from ml.config import CHECKPOINT_DIR, DATA_PROCESSED_DIR, AtlasModelConfig


def generate_training_data(
    n_samples: int = 5000,
    n_nodes: int = 132,
    n_mechanics: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic atlas training data.

    In production, this would be replaced with:
    1. Real atlas allocations from community datasets
    2. Simulated mapping returns using drop tables
    3. Economy-weighted reward calculations

    For now, creates synthetic data to validate the pipeline.

    Returns (X, y) where:
    - X: (n_samples * n_nodes, features_per_node)
    - y: (n_samples * n_nodes,) — chaos/hr return for each node
    """
    rng = np.random.RandomState(42)

    # Node features per node:
    # [spawn_bonus, reward_mult, difficulty_mod, is_notable, is_keystone,
    #  league_day, divine_rate, mechanic_value, strategy_weight, mech_oh...]
    features_per_node = 9 + n_mechanics

    X_all = []
    y_all = []

    for _ in range(n_samples):
        # Economy context (same for all nodes in this sample)
        league_day = rng.randint(1, 90)
        divine_rate = rng.uniform(100, 250)
        mechanic_values = rng.uniform(5, 50, size=n_mechanics)
        strategy_weights = rng.dirichlet(np.ones(n_mechanics))

        for _ in range(n_nodes):
            spawn_bonus = rng.uniform(0, 0.5)
            reward_mult = rng.uniform(1.0, 3.0)
            difficulty_mod = rng.uniform(0, 0.3)
            is_notable = float(rng.random() < 0.15)
            is_keystone = float(rng.random() < 0.03)
            mechanic_idx = rng.randint(0, n_mechanics)
            mech_oh = np.zeros(n_mechanics)
            mech_oh[mechanic_idx] = 1.0

            node_features = np.concatenate([
                [spawn_bonus, reward_mult, difficulty_mod,
                 is_notable, is_keystone,
                 league_day, divine_rate,
                 mechanic_values[mechanic_idx],
                 strategy_weights[mechanic_idx]],
                mech_oh,
            ])
            X_all.append(node_features)

            # Synthetic target: return correlates with reward_mult * mechanic_value * strategy_weight
            base_return = reward_mult * mechanic_values[mechanic_idx] * strategy_weights[mechanic_idx]
            notable_bonus = 2.0 if is_notable else 0.0
            keystone_bonus = 5.0 if is_keystone else 0.0
            noise = rng.normal(0, 1)
            y_all.append(base_return + notable_bonus + keystone_bonus + noise)

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.float32)
    y = np.clip(y, 0, None)  # no negative returns

    return X, y


def train(
    checkpoint_dir: Path | None = None,
) -> None:
    """Train the atlas return predictor."""
    checkpoint_dir = checkpoint_dir or CHECKPOINT_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    cfg = AtlasModelConfig()

    # ── Generate / load data ───────────────────────────────────────────
    logger.info("Generating atlas training data...")
    X, y = generate_training_data()

    # Split
    n = len(X)
    idx = np.random.RandomState(42).permutation(n)
    n_test = int(n * 0.1)
    n_val = int(n * 0.1)

    X_test, y_test = X[idx[:n_test]], y[idx[:n_test]]
    X_val, y_val = X[idx[n_test:n_test + n_val]], y[idx[n_test:n_test + n_val]]
    X_train, y_train = X[idx[n_test + n_val:]], y[idx[n_test + n_val:]]

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # ── Train ──────────────────────────────────────────────────────────
    from ml.models.atlas_tree_optimizer import AtlasReturnPredictor

    predictor = AtlasReturnPredictor(cfg)

    try:
        import xgboost as xgb

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_test, label=y_test)

        params = {
            "objective": "reg:squarederror",
            "max_depth": cfg.max_depth,
            "learning_rate": cfg.learning_rate,
            "subsample": cfg.subsample,
            "colsample_bytree": cfg.colsample_bytree,
            "eval_metric": "rmse",
            "verbosity": 1,
        }

        evals = [(dtrain, "train"), (dval, "val")]
        predictor.model = xgb.train(
            params, dtrain,
            num_boost_round=cfg.n_estimators,
            evals=evals,
            early_stopping_rounds=20,
            verbose_eval=50,
        )

        # ── Evaluate ───────────────────────────────────────────────────
        preds = predictor.model.predict(dtest)
        rmse = np.sqrt(np.mean((preds - y_test) ** 2))
        mae = np.mean(np.abs(preds - y_test))
        logger.info(f"Test RMSE: {rmse:.3f}, MAE: {mae:.3f}")

        # ── Save ───────────────────────────────────────────────────────
        model_path = str(checkpoint_dir / "atlas_return_predictor.xgb")
        predictor.save(model_path)
        logger.success(f"Atlas model saved → {model_path}")

    except ImportError:
        logger.error("xgboost not installed. Run: pip install xgboost")
        return


# ── CLI ────────────────────────────────────────────────────────────────

def main() -> None:
    import typer
    app = typer.Typer(help="Train the Atlas Tree Optimizer")

    @app.command()
    def run() -> None:
        train()

    app()


if __name__ == "__main__":
    main()
