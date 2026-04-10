"""
Build Tree Optimizer — Graph Attention Network.

Predicts the optimal passive-tree allocation for a given build context
(class, ascendancy, skill tags, stat-weight preferences).

Architecture:
    ┌──────────────────────────────┐
    │  Context Encoder (MLP)       │ → context embedding
    └──────────────┬───────────────┘
                   │
    ┌──────────────▼───────────────┐
    │  GAT Layers × N              │ → per-node embeddings
    │  (node features + context    │    conditioned on build context
    │   injected via concat)       │
    └──────────────┬───────────────┘
                   │
    ┌──────────────▼───────────────┐
    │  Node Classifier (MLP)       │ → per-node σ(p) ∈ [0,1]
    └──────────────────────────────┘

Loss:
    BCE(predicted, actual_allocation) + λ_conn * connectivity_penalty
    + λ_budget * budget_penalty

The connectivity penalty encourages the predicted nodes to form a
connected subgraph rooted at the class start node.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.config import BuildTreeModelConfig

# Try importing PyG; fail gracefully if not installed
try:
    from torch_geometric.nn import GATConv, global_mean_pool
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class ContextEncoder(nn.Module):
    """Encodes build context (class, ascendancy, level, stat prefs) into a vector."""

    def __init__(
        self,
        num_classes: int = 7,
        num_ascendancies: int = 19,
        class_embed_dim: int = 8,
        asc_embed_dim: int = 8,
        extra_features: int = 2,  # level + stat pref dim placeholder
        output_dim: int = 32,
    ) -> None:
        super().__init__()
        self.class_embed = nn.Embedding(num_classes, class_embed_dim)
        self.asc_embed = nn.Embedding(num_ascendancies, asc_embed_dim)
        in_dim = class_embed_dim + asc_embed_dim + extra_features
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(
        self,
        class_ids: torch.Tensor,
        asc_ids: torch.Tensor,
        extra: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        class_ids : (B,) long
        asc_ids : (B,) long
        extra : (B, extra_features) float

        Returns
        -------
        (B, output_dim)
        """
        c = self.class_embed(class_ids)
        a = self.asc_embed(asc_ids)
        x = torch.cat([c, a, extra], dim=-1)
        return self.mlp(x)


class BuildTreeGAT(nn.Module):
    """
    GAT-based passive-tree node classifier.

    Inputs
    ------
    - Node feature matrix ``x`` (N, F)
    - Edge index ``edge_index`` (2, E)
    - Build context per sample ``context`` (B, context_dim)
    - Batch vector mapping each node to its sample index

    Output
    ------
    - Per-node allocation probability ``(N, 1)``
    """

    def __init__(self, cfg: BuildTreeModelConfig | None = None) -> None:
        super().__init__()
        if not HAS_PYG:
            raise ImportError(
                "torch-geometric is required for BuildTreeGAT. "
                "Install with: pip install torch-geometric"
            )
        cfg = cfg or BuildTreeModelConfig()

        self.context_encoder = ContextEncoder(output_dim=cfg.context_embed_dim)

        # Input projection: node features + injected context
        self.input_proj = nn.Linear(
            cfg.node_feature_dim + cfg.context_embed_dim,
            cfg.hidden_dim,
        )

        # GAT layers
        self.gat_layers = nn.ModuleList()
        for i in range(cfg.num_gat_layers):
            in_channels = cfg.hidden_dim
            out_channels = cfg.hidden_dim // cfg.num_gat_heads
            self.gat_layers.append(
                GATConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    heads=cfg.num_gat_heads,
                    dropout=cfg.dropout,
                    concat=True,  # output: out_channels * heads = hidden_dim
                )
            )

        # Node classifier head
        self.classifier = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim // 2, 1),
        )

        self.cfg = cfg

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        context: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (N, node_feature_dim)
        edge_index : (2, E)
        context : (B, context_embed_dim) — one per graph in the batch
        batch : (N,) — maps each node to its sample index in the batch

        Returns
        -------
        (N, 1) — allocation probabilities
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Broadcast context to each node
        node_context = context[batch]  # (N, context_embed_dim)
        h = torch.cat([x, node_context], dim=-1)  # (N, F + context_dim)
        h = self.input_proj(h)  # (N, hidden_dim)

        # GAT message passing
        for gat in self.gat_layers:
            h = gat(h, edge_index)
            h = F.elu(h)

        # Classify each node
        logits = self.classifier(h)  # (N, 1)
        return torch.sigmoid(logits)


# ── Loss functions ─────────────────────────────────────────────────────

def allocation_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    budget: int = 123,
    lambda_budget: float = 0.1,
) -> torch.Tensor:
    """
    Binary cross-entropy + point-budget penalty.

    Parameters
    ----------
    pred : (N,) predicted allocation probabilities
    target : (N,) ground-truth binary allocations
    budget : int — max passive points
    lambda_budget : float — penalty weight for over-allocating
    """
    bce = F.binary_cross_entropy(pred, target, reduction="mean")

    # Budget penalty: penalise if predicted allocation count exceeds budget
    pred_count = pred.sum()
    over_budget = F.relu(pred_count - budget)
    budget_penalty = lambda_budget * over_budget

    return bce + budget_penalty


def connectivity_penalty(
    pred: torch.Tensor,
    edge_index: torch.Tensor,
    start_node_idx: int,
    threshold: float = 0.5,
    lambda_conn: float = 0.05,
) -> torch.Tensor:
    """
    Soft penalty for disconnected allocations.

    Encourages allocated nodes to be reachable from the class start node
    through other allocated nodes.

    This is a differentiable approximation: for each predicted-active node,
    we check that at least one neighbor is also predicted-active.
    """
    active = (pred > threshold).float()
    src, dst = edge_index

    # For each active node, sum of active neighbors
    neighbor_active = torch.zeros_like(pred)
    neighbor_active.scatter_add_(0, dst, active[src])

    # Nodes that are active but have no active neighbors (except start)
    isolated = active * (neighbor_active == 0).float()
    isolated[start_node_idx] = 0  # start node is always OK

    return lambda_conn * isolated.sum()


# ── Reinforcement Learning alternative ─────────────────────────────────

class TreeAllocEnv:
    """
    Gym-like environment for sequential passive-tree allocation.

    State:  current allocation mask + remaining points + current stats
    Action: allocate next point to an adjacent unallocated node
    Reward: delta in combined score (DPS + EHP) per point

    This is a skeleton — full implementation requires wrapping
    the PoB Lua calculation engine.
    """

    def __init__(
        self,
        tree_data: dict,
        class_start_node: int,
        total_points: int = 123,
    ) -> None:
        self.tree_data = tree_data
        self.start_node = class_start_node
        self.total_points = total_points
        self.allocated: set[int] = set()
        self.remaining_points = total_points

    def reset(self) -> dict:
        """Reset to class start."""
        self.allocated = {self.start_node}
        self.remaining_points = self.total_points - 1
        return self._get_state()

    def _get_state(self) -> dict:
        return {
            "allocated": list(self.allocated),
            "remaining_points": self.remaining_points,
        }

    def get_valid_actions(self) -> list[int]:
        """Return node IDs adjacent to current allocation but not yet allocated."""
        valid = set()
        nodes = self.tree_data.get("nodes", {})
        for nid in self.allocated:
            node = nodes.get(str(nid), {})
            for neighbor in node.get("connections", []):
                if neighbor not in self.allocated:
                    valid.add(neighbor)
        return list(valid)

    def step(self, node_id: int) -> tuple[dict, float, bool]:
        """
        Allocate a point to ``node_id``.

        Returns (state, reward, done).
        Reward is a placeholder — real implementation would call PoB engine.
        """
        if node_id in self.allocated or self.remaining_points <= 0:
            return self._get_state(), -1.0, True

        self.allocated.add(node_id)
        self.remaining_points -= 1

        # Placeholder reward — in production, run PoB calc engine here
        reward = 1.0  # TODO: compute delta DPS/EHP via PoB

        done = self.remaining_points <= 0
        return self._get_state(), reward, done
