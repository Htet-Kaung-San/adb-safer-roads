"""
Stage 3: Graph Attention Network (GAT) for spatially-aware Speed Safety Score refinement.

The GNN takes Stage-1 tabular scores (+ optional VLM features) as node features
and propagates context through the road network graph. Adjacent segments influence
each other — a dangerous arterial raises the perceived risk of connected local roads.

Architecture: 3-layer GAT with skip connections → regression head → refined score [0,100]

Uncertainty: Monte Carlo Dropout — 50 stochastic forward passes at inference time
             produce per-segment 95% confidence intervals.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm
from torch_geometric.data import Data
from typing import Tuple

from src.config import (
    GNN_HIDDEN_DIM, GNN_NUM_LAYERS, GNN_HEADS, GNN_DROPOUT,
    GNN_EPOCHS, GNN_LR, GNN_MC_SAMPLES, SCORE_BANDS,
)


class SpeedSafetyGAT(nn.Module):
    """
    Graph Attention Network that refines the Stage-1 Speed Safety Score
    by incorporating spatial context from the road network topology.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = GNN_HIDDEN_DIM,
        num_layers: int = GNN_NUM_LAYERS,
        heads: int = GNN_HEADS,
        dropout: float = GNN_DROPOUT,
    ):
        super().__init__()
        self.dropout = dropout

        self.input_proj = nn.Linear(in_features, hidden_dim)

        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim * heads
            out_heads = 1 if i == num_layers - 1 else heads
            self.gat_layers.append(
                GATConv(in_dim, hidden_dim, heads=out_heads, dropout=dropout, concat=(out_heads > 1))
            )
            out_dim = hidden_dim * out_heads
            self.norms.append(BatchNorm(out_dim))

        self.skip = nn.Linear(hidden_dim, hidden_dim)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.input_proj(x))
        skip_h = self.skip(h)

        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.norms)):
            h = gat(h, edge_index)
            h = norm(h)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        h = h + skip_h
        score = self.head(h).squeeze(-1) * 100.0
        return score


class GNNTrainer:
    """Thin wrapper for training SpeedSafetyGAT on a single graph."""

    def __init__(self, model: SpeedSafetyGAT, device: str = "cuda"):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=GNN_LR, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=GNN_EPOCHS
        )

    def train(self, data: Data, epochs: int = GNN_EPOCHS, log_every: int = 10):
        """Train on the full graph (transductive setting). Returns loss history."""
        data = data.to(self.device)
        losses = []

        for epoch in range(1, epochs + 1):
            self.model.train()
            self.optimizer.zero_grad()
            pred = self.model(data.x, data.edge_index)
            loss = F.mse_loss(pred, data.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            losses.append(loss.item())

            if epoch % log_every == 0:
                rmse = loss.item() ** 0.5
                print(f"Epoch {epoch:4d} | Loss {loss.item():.4f} | RMSE {rmse:.2f}")

        return losses

    @torch.no_grad()
    def predict(self, data: Data) -> torch.Tensor:
        self.model.eval()
        data = data.to(self.device)
        return self.model(data.x, data.edge_index).cpu()

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        data: Data,
        n_samples: int = GNN_MC_SAMPLES,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Monte Carlo Dropout uncertainty estimation.

        Keeps dropout active during inference (model.train() mode) and runs
        n_samples stochastic forward passes. Returns per-node statistics.

        Returns:
            mean      — mean prediction across samples (N,)
            std       — standard deviation across samples (N,)
            ci_low    — lower bound of 95% confidence interval (N,)
            ci_high   — upper bound of 95% confidence interval (N,)

        The 95% CI uses the normal approximation: mean ± 1.96 × std.
        Segments where the CI crosses a grade boundary (every 20 points)
        are flagged as uncertain — the true grade could differ.
        """
        # Keep dropout active by staying in train() mode
        self.model.train()
        data = data.to(self.device)

        samples = []
        for _ in range(n_samples):
            pred = self.model(data.x, data.edge_index).cpu()
            samples.append(pred)

        samples = torch.stack(samples, dim=0)   # (n_samples, N)

        mean   = samples.mean(dim=0)
        std    = samples.std(dim=0)
        ci_low  = (mean - 1.96 * std).clamp(0, 100)
        ci_high = (mean + 1.96 * std).clamp(0, 100)

        return mean, std, ci_low, ci_high

    def grade_uncertainty_flag(
        self,
        mean: torch.Tensor,
        ci_low: torch.Tensor,
        ci_high: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns a boolean tensor: True where the 95% CI spans a grade boundary.
        Grade boundaries are at 20, 40, 60, 80.
        A flagged segment's true grade is ambiguous — could be one grade up or down.
        """
        boundaries = torch.tensor([20.0, 40.0, 60.0, 80.0])
        uncertain = torch.zeros(mean.shape[0], dtype=torch.bool)
        for boundary in boundaries:
            crosses = (ci_low < boundary) & (ci_high > boundary)
            uncertain = uncertain | crosses
        return uncertain


def build_and_train_gnn(
    gdf,
    x,
    edge_index,
    y,
    device: str = "cuda",
    epochs: int = GNN_EPOCHS,
    compute_uncertainty: bool = True,
):
    """
    Build model, train, return (model, refined_scores, uncertainty_dict).

    uncertainty_dict contains: mean, std, ci_low, ci_high, grade_uncertain
    All tensors of shape (N,). If compute_uncertainty=False, returns empty dict.
    """
    import numpy as np
    from src.gnn.graph_builder import to_pyg_data

    data = to_pyg_data(x, edge_index, y)
    in_features = x.shape[1]

    model = SpeedSafetyGAT(in_features=in_features)
    trainer = GNNTrainer(model, device=device)
    trainer.train(data, epochs=epochs)

    # Point estimate
    refined = trainer.predict(data)

    uncertainty = {}
    if compute_uncertainty:
        print(f"Computing uncertainty via MC Dropout ({GNN_MC_SAMPLES} samples) …")
        mean, std, ci_low, ci_high = trainer.predict_with_uncertainty(data)
        grade_uncertain = trainer.grade_uncertainty_flag(mean, ci_low, ci_high)
        uncertainty = {
            "score_mean":      mean,
            "score_std":       std,
            "score_ci_low":    ci_low,
            "score_ci_high":   ci_high,
            "grade_uncertain": grade_uncertain,
        }
        n_uncertain = grade_uncertain.sum().item()
        n_total = len(mean)
        print(f"Uncertainty: {n_uncertain}/{n_total} segments ({n_uncertain/n_total*100:.1f}%) "
              f"have grade-ambiguous 95% CI")

    return model, refined, uncertainty
