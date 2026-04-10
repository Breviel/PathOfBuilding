"""
Training loop for the Build Tree Optimizer (GAT model).

Steps:
1. Load processed dataset (build_tree_dataset.npz)
2. Load tree graph (for shared edge_index)
3. Train GAT with BCE + budget + connectivity penalties
4. Evaluate on validation set
5. Checkpoint best model

Usage:
    python -m ml.training.train_build_tree
    python -m ml.training.train_build_tree --epochs 200 --lr 0.001
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from loguru import logger

from ml.config import (
    CHECKPOINT_DIR,
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    BuildTreeModelConfig,
    TrainingConfig,
)


def train(
    epochs: int | None = None,
    lr: float | None = None,
    batch_size: int | None = None,
    checkpoint_dir: Path | None = None,
) -> None:
    """
    Full training loop for the Build Tree GNN.
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    cfg = TrainingConfig()
    model_cfg = BuildTreeModelConfig()
    epochs = epochs or cfg.epochs
    lr = lr or cfg.lr
    batch_size = batch_size or cfg.batch_size
    checkpoint_dir = checkpoint_dir or CHECKPOINT_DIR

    # ── 1. Load data ───────────────────────────────────────────────────
    logger.info("Loading datasets...")

    from ml.data.processing.dataset import BuildTreeDataset

    train_ds = BuildTreeDataset(split="train")
    val_ds = BuildTreeDataset(split="val")

    logger.info(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # ── 2. Load tree graph ─────────────────────────────────────────────
    tree_json = DATA_RAW_DIR / "passive_tree_3_28.json"
    if tree_json.exists():
        from ml.data.collectors.tree_data import load_passive_tree, tree_to_pyg
        tree = load_passive_tree()
        graph_data = tree_to_pyg(tree)
        edge_index = graph_data.edge_index
        node_features = graph_data.x
        num_tree_nodes = graph_data.num_nodes
    else:
        logger.warning("Tree JSON not found. Using placeholder graph.")
        num_tree_nodes = 1300
        node_features = torch.randn(num_tree_nodes, 9)
        # Simple chain graph as placeholder
        src = list(range(num_tree_nodes - 1))
        dst = list(range(1, num_tree_nodes))
        edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)

    # Pad node features to match model input dim
    if node_features.size(1) < model_cfg.node_feature_dim:
        padding = torch.zeros(
            num_tree_nodes,
            model_cfg.node_feature_dim - node_features.size(1),
        )
        node_features = torch.cat([node_features, padding], dim=1)

    # ── 3. Build model ─────────────────────────────────────────────────
    from ml.models.build_tree_optimizer import (
        BuildTreeGAT,
        allocation_loss,
        connectivity_penalty,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BuildTreeGAT(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5,
    )

    node_features = node_features.to(device)
    edge_index = edge_index.to(device)

    # ── 4. Training loop ───────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for batch in train_loader:
            context = batch["context"].to(device)
            labels = batch["labels"].to(device)
            B = context.size(0)

            # For each sample in batch, we need to expand the shared graph
            # In practice, use PyG Batch; here we simplify with loop
            total_loss = torch.tensor(0.0, device=device)

            for i in range(B):
                # Extract context components
                class_id = context[i, 0].long().unsqueeze(0)
                level = context[i, 1:2]
                extra = context[i, 1:3].unsqueeze(0)  # level + dps placeholder

                # Encode context
                asc_id = torch.zeros(1, dtype=torch.long, device=device)
                ctx_embed = model.context_encoder(class_id, asc_id, extra)

                # Forward pass
                pred = model(node_features, edge_index, ctx_embed).squeeze(-1)

                # Compute loss (only up to num_tree_nodes from labels)
                target = labels[i, :num_tree_nodes]
                loss = allocation_loss(pred, target)
                total_loss = total_loss + loss

            total_loss = total_loss / B
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(total_loss.item())

        avg_train = np.mean(train_losses)

        # ── Validation ─────────────────────────────────────────────────
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                context = batch["context"].to(device)
                labels = batch["labels"].to(device)
                B = context.size(0)

                batch_loss = 0.0
                for i in range(B):
                    class_id = context[i, 0].long().unsqueeze(0)
                    extra = context[i, 1:3].unsqueeze(0)
                    asc_id = torch.zeros(1, dtype=torch.long, device=device)
                    ctx_embed = model.context_encoder(class_id, asc_id, extra)
                    pred = model(node_features, edge_index, ctx_embed).squeeze(-1)
                    target = labels[i, :num_tree_nodes]
                    loss = allocation_loss(pred, target)
                    batch_loss += loss.item()

                val_losses.append(batch_loss / max(B, 1))

        avg_val = np.mean(val_losses) if val_losses else 0
        scheduler.step(avg_val)

        logger.info(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {avg_train:.4f} | "
            f"Val Loss: {avg_val:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        # Checkpointing
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            ckpt_path = checkpoint_dir / "build_tree_gat_best.pt"
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"  → Saved best model ({avg_val:.4f}) → {ckpt_path}")
        else:
            patience_counter += 1

        if patience_counter >= cfg.early_stop_patience:
            logger.info(f"Early stopping after {epoch} epochs")
            break

        if epoch % cfg.checkpoint_every == 0:
            ckpt_path = checkpoint_dir / f"build_tree_gat_epoch{epoch}.pt"
            torch.save(model.state_dict(), ckpt_path)

    logger.success(f"Training complete. Best val loss: {best_val_loss:.4f}")


# ── CLI ────────────────────────────────────────────────────────────────

def main() -> None:
    import typer

    app = typer.Typer(help="Train the Build Tree Optimizer")

    @app.command()
    def run(
        epochs: int = typer.Option(100, help="Number of epochs"),
        lr: float = typer.Option(1e-3, help="Learning rate"),
        batch_size: int = typer.Option(64, help="Batch size"),
    ) -> None:
        train(epochs=epochs, lr=lr, batch_size=batch_size)

    app()


if __name__ == "__main__":
    main()
