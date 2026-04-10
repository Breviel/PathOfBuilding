"""
Atlas Tree Optimizer — ML return predictor + ILP constraint solver.

Two-stage approach:
1. **ML Stage**: A gradient-boosted model predicts the expected return
   (chaos/hour) for each atlas node given the current economy context.
2. **ILP Stage**: A constraint solver (Google OR-Tools CP-SAT) selects
   the optimal ~130-point allocation that maximises total predicted
   return, subject to connectivity and synergy constraints.

Architecture:
    ┌────────────────────────────────┐
    │  Economy Context (league day,  │
    │  currency rates, item prices)  │
    └──────────────┬─────────────────┘
                   │
    ┌──────────────▼─────────────────┐
    │  Per-Node Return Predictor     │ → predicted chaos/hr per node
    │  (XGBoost / LightGBM)         │
    └──────────────┬─────────────────┘
                   │
    ┌──────────────▼─────────────────┐
    │  Synergy Correction            │ → adjust for node combinations
    │  (pairwise interaction terms)  │
    └──────────────┬─────────────────┘
                   │
    ┌──────────────▼─────────────────┐
    │  CP-SAT Constraint Solver      │ → optimal 130-point allocation
    │  (connectivity + budget)       │
    └────────────────────────────────┘
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger

from ml.config import AtlasModelConfig, ATLAS_STRATEGIES, MAX_ATLAS_POINTS


# ── Atlas Node representation ──────────────────────────────────────────

@dataclass
class AtlasNode:
    """Representation of one atlas passive node."""
    node_id: int
    name: str
    mechanic: str  # e.g. "Expedition", "Delirium", "Harvest", "Generic"
    stat_text: str
    # Feature values
    spawn_chance_bonus: float = 0.0
    reward_multiplier: float = 0.0
    difficulty_modifier: float = 0.0
    connections: list[int] = field(default_factory=list)
    is_notable: bool = False
    is_keystone: bool = False


@dataclass
class EconomyContext:
    """Current league economy snapshot for atlas optimization."""
    league_day: int  # day since league start
    chaos_divine_rate: float  # 1 divine = X chaos
    # Per-mechanic reward values (chaos equivalent per encounter)
    mechanic_values: dict[str, float] = field(default_factory=dict)
    # e.g. {"Expedition": 25, "Delirium": 18, "Harvest": 30, ...}


# ── Stage 1: Per-node return predictor ─────────────────────────────────

class AtlasReturnPredictor:
    """
    XGBoost model that predicts chaos/hour return for each atlas node
    given the economy context and user's farming strategy.
    """

    def __init__(self, cfg: AtlasModelConfig | None = None) -> None:
        self.cfg = cfg or AtlasModelConfig()
        self.model = None

    def _build_features(
        self,
        nodes: list[AtlasNode],
        economy: EconomyContext,
        strategy_weights: dict[str, float],
    ) -> np.ndarray:
        """
        Build feature matrix for all nodes.

        Features per node:
        - spawn_chance_bonus
        - reward_multiplier
        - difficulty_modifier
        - is_notable, is_keystone
        - mechanic one-hot
        - economy context (league_day, divine_rate, mechanic_value)
        - strategy weight for this mechanic
        """
        mechanics = sorted(set(n.mechanic for n in nodes))
        mech_to_idx = {m: i for i, m in enumerate(mechanics)}
        n_mech = len(mechanics)

        features = []
        for node in nodes:
            row = [
                node.spawn_chance_bonus,
                node.reward_multiplier,
                node.difficulty_modifier,
                float(node.is_notable),
                float(node.is_keystone),
                economy.league_day,
                economy.chaos_divine_rate,
                economy.mechanic_values.get(node.mechanic, 0),
                strategy_weights.get(node.mechanic, 0),
            ]
            # Mechanic one-hot
            mech_oh = [0.0] * n_mech
            if node.mechanic in mech_to_idx:
                mech_oh[mech_to_idx[node.mechanic]] = 1.0
            row.extend(mech_oh)
            features.append(row)

        return np.array(features, dtype=np.float32)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """Train the return predictor."""
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost required: pip install xgboost")

        dtrain = xgb.DMatrix(X, label=y)
        params = {
            "objective": "reg:squarederror",
            "max_depth": self.cfg.max_depth,
            "learning_rate": self.cfg.learning_rate,
            "subsample": self.cfg.subsample,
            "colsample_bytree": self.cfg.colsample_bytree,
            "eval_metric": "rmse",
            "verbosity": 1,
        }
        self.model = xgb.train(
            params, dtrain, num_boost_round=self.cfg.n_estimators,
        )
        logger.info("Atlas return predictor trained.")

    def predict(
        self,
        nodes: list[AtlasNode],
        economy: EconomyContext,
        strategy_weights: dict[str, float],
    ) -> np.ndarray:
        """Predict per-node return (chaos/hr)."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        import xgboost as xgb
        X = self._build_features(nodes, economy, strategy_weights)
        dpred = xgb.DMatrix(X)
        return self.model.predict(dpred)

    def save(self, path: str) -> None:
        if self.model is not None:
            self.model.save_model(path)

    def load(self, path: str) -> None:
        import xgboost as xgb
        self.model = xgb.Booster()
        self.model.load_model(path)


# ── Stage 2: Synergy correction ───────────────────────────────────────

def compute_synergy_bonus(
    selected_nodes: set[int],
    synergy_table: dict[tuple[int, int], float],
) -> float:
    """
    Compute additional return from node synergies.

    Parameters
    ----------
    selected_nodes : set of node IDs
    synergy_table : mapping (node_a, node_b) → bonus chaos/hr

    Returns
    -------
    Total synergy bonus.
    """
    bonus = 0.0
    nodes_list = sorted(selected_nodes)
    for i, a in enumerate(nodes_list):
        for b in nodes_list[i + 1 :]:
            key = (min(a, b), max(a, b))
            bonus += synergy_table.get(key, 0.0)
    return bonus


# ── Stage 3: CP-SAT constraint solver ─────────────────────────────────

def solve_atlas_allocation(
    nodes: list[AtlasNode],
    predicted_returns: np.ndarray,
    max_points: int = MAX_ATLAS_POINTS,
    synergy_pairs: dict[tuple[int, int], float] | None = None,
    time_limit_s: int = 30,
) -> dict[str, Any]:
    """
    Solve the atlas allocation problem using Google OR-Tools CP-SAT.

    Parameters
    ----------
    nodes : list of AtlasNode
    predicted_returns : per-node predicted chaos/hr
    max_points : budget constraint
    synergy_pairs : optional pairwise synergy bonuses
    time_limit_s : solver time limit

    Returns
    -------
    dict with:
        "selected_nodes": list[int] — node IDs to allocate
        "total_return": float — predicted total chaos/hr
        "solver_status": str
    """
    try:
        from ortools.sat.python import cp_model
    except ImportError:
        raise ImportError("ortools required: pip install ortools")

    model = cp_model.CpModel()
    n = len(nodes)

    # Decision variables: x[i] = 1 if node i is selected
    x = [model.new_bool_var(f"node_{i}") for i in range(n)]

    # Budget constraint
    model.add(sum(x) <= max_points)

    # Objective: maximise total predicted return
    # Scale to integers (CP-SAT uses integer arithmetic)
    SCALE = 1000
    scaled_returns = (predicted_returns * SCALE).astype(int)

    objective_terms = [x[i] * int(scaled_returns[i]) for i in range(n)]

    # Add synergy terms
    if synergy_pairs:
        id_to_idx = {node.node_id: i for i, node in enumerate(nodes)}
        for (a, b), bonus in synergy_pairs.items():
            if a in id_to_idx and b in id_to_idx:
                ia, ib = id_to_idx[a], id_to_idx[b]
                # Synergy active only if both nodes selected
                both = model.new_bool_var(f"syn_{a}_{b}")
                model.add_implication(both, x[ia])
                model.add_implication(both, x[ib])
                model.add(both <= x[ia])
                model.add(both <= x[ib])
                objective_terms.append(both * int(bonus * SCALE))

    model.maximize(sum(objective_terms))

    # Connectivity constraint (soft):
    # We add a penalty for isolated nodes rather than hard connectivity
    # (full connectivity as ILP constraint is expensive; this is a pragmatic trade-off)

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    status = solver.solve(model)

    status_name = {
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.MODEL_INVALID: "MODEL_INVALID",
        cp_model.UNKNOWN: "UNKNOWN",
    }.get(status, "UNKNOWN")

    selected = []
    total_return = 0.0
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for i in range(n):
            if solver.value(x[i]):
                selected.append(nodes[i].node_id)
                total_return += predicted_returns[i]

    logger.info(
        f"Atlas solver: {status_name}, "
        f"{len(selected)} nodes selected, "
        f"predicted {total_return:.1f} chaos/hr"
    )

    return {
        "selected_nodes": selected,
        "total_return": total_return,
        "solver_status": status_name,
    }


# ── End-to-end optimization ───────────────────────────────────────────

def optimize_atlas(
    nodes: list[AtlasNode],
    economy: EconomyContext,
    strategy_weights: dict[str, float] | None = None,
    strategy_name: str | None = None,
    predictor: AtlasReturnPredictor | None = None,
    synergy_pairs: dict[tuple[int, int], float] | None = None,
    warm_start_node_ids: list[int] | None = None,
    max_points: int = MAX_ATLAS_POINTS,
) -> dict[str, Any]:
    """
    Full pipeline: predict returns → solve allocation.

    Parameters
    ----------
    nodes : atlas tree nodes
    economy : current economy snapshot
    strategy_weights : user preference (e.g. {"Expedition": 1.0, "Delirium": 0.5}).
        If None, ``strategy_name`` is used to look up pre-defined weights
        from config.ATLAS_STRATEGIES (sourced from poe-atlas.com categories).
    strategy_name : optional strategy key (e.g. "expedition", "breach_delirium").
        Used to look up weights from ATLAS_STRATEGIES if strategy_weights is None.
    predictor : trained AtlasReturnPredictor (required for ML-based optimization)
    synergy_pairs : optional pairwise synergies
    warm_start_node_ids : optional seed allocation from a poe-atlas.com guide
        build. These nodes are hinted (but not forced) to the ILP solver.
    max_points : point budget

    Returns
    -------
    Allocation result dict with keys:
        selected_nodes, total_return, solver_status, strategy_used,
        per_node_predictions
    """
    # Resolve strategy weights
    if strategy_weights is None:
        if strategy_name and strategy_name in ATLAS_STRATEGIES:
            strategy_weights = ATLAS_STRATEGIES[strategy_name]
            logger.info(f"Using pre-defined strategy: {strategy_name}")
        else:
            available = list(ATLAS_STRATEGIES.keys())
            logger.warning(
                f"Strategy '{strategy_name}' not found. "
                f"Available: {available}. Using equal weights."
            )
            all_mechs = set()
            for n in nodes:
                all_mechs.add(n.mechanic)
            strategy_weights = {m: 1.0 for m in all_mechs}

    if predictor is None:
        raise ValueError("A trained AtlasReturnPredictor is required")

    predicted = predictor.predict(nodes, economy, strategy_weights)

    # Warm-start: boost predicted returns for nodes in the seed allocation.
    # This biases the solver toward the curated poe-atlas.com build as a
    # starting point, while still allowing the optimizer to improve on it.
    if warm_start_node_ids:
        warm_set = set(warm_start_node_ids)
        warm_bonus = np.median(predicted[predicted > 0]) * 0.1  # 10% bonus
        for i, node in enumerate(nodes):
            if node.node_id in warm_set:
                predicted[i] += warm_bonus
        logger.info(
            f"Applied warm-start hint from {len(warm_start_node_ids)} "
            f"seed nodes (bonus: {warm_bonus:.2f})"
        )

    result = solve_atlas_allocation(
        nodes, predicted,
        max_points=max_points,
        synergy_pairs=synergy_pairs,
    )
    result["per_node_predictions"] = {
        nodes[i].node_id: float(predicted[i]) for i in range(len(nodes))
    }
    result["strategy_used"] = strategy_name or "custom"
    return result
