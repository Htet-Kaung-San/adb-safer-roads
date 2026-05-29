"""
Stage 3: Graph Attention Network (GAT) for spatially-aware Speed Safety Score refinement.

The GNN takes Stage-1 tabular scores (+ optional VLM features) as node features
and propagates context through the road network graph. Adjacent segments influence
each other — a dangerous arterial raises the perceived risk of connected local roads.

Architecture: 3-layer GAT with skip connections → regression head → refined score [0,100]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm
from torch_geometric.data import Data

from src.config import GNN_HIDDEN_DIM, GNN_NUM_LAYERS, GNN_HEADS, GNN_DROPOUT, GNN_EPOCHS, GNN_LR


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
            # Last layer: single head for clean output
            out_heads = 1 if i == num_layers - 1 else heads
            self.gat_layers.append(
                GATConv(in_dim, hidden_dim, heads=out_heads, dropout=dropout, concat=(out_heads > 1))
            )
            out_dim = hidden_dim * out_heads
            self.norms.append(BatchNorm(out_dim))

        # Skip connection from input projection to final layer
        self.skip = nn.Linear(hidden_dim, hidden_dim)

        # Regression head: outputs a refined score in [0, 100]
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),   # → [0, 1], multiply by 100 at inference
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.input_proj(x))
        skip_h = self.skip(h)

        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.norms)):
            h = gat(h, edge_index)
            h = norm(h)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        # Add skip connection (after last GAT layer has hidden_dim output)
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

    def train(self, data: Data, epochs: int = GNN_EPOCHS, log_every: int = 10) -> list[float]:
        """
        Train on the full graph (transductive setting).
        Target y = Stage-1 Speed Safety Score.
        Returns loss history.
        """
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


def build_and_train_gnn(
    gdf,
    x: "np.ndarray",
    edge_index: "np.ndarray",
    y: "np.ndarray",
    device: str = "cuda",
    epochs: int = GNN_EPOCHS,
) -> tuple["SpeedSafetyGAT", torch.Tensor]:
    """
    Convenience function: build model, train, return (model, refined_scores).
    refined_scores is a tensor of shape (N,) with GNN-refined Speed Safety Scores.
    """
    import numpy as np
    from src.gnn.graph_builder import to_pyg_data

    data = to_pyg_data(x, edge_index, y)
    in_features = x.shape[1]

    model = SpeedSafetyGAT(in_features=in_features)
    trainer = GNNTrainer(model, device=device)
    trainer.train(data, epochs=epochs)
    refined = trainer.predict(data)
    return model, refined
