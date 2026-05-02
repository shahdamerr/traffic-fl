import torch
import torch.nn as nn


class GRUForecaster(nn.Module):
    """
    Optimized single-sensor GRU forecaster.

    Architecture:
        - Multi-layer GRU with inter-layer dropout
        - 2-layer MLP head (Linear -> ReLU -> Dropout -> Linear)

    Input:  x [B, L, 1]
    Output: y [B, H]
    """

    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 2,
        horizon: int = 12,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 2-layer MLP decoder
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon),
        )

    def forward(self, x):
        # x: [B, L, 1]
        out, _ = self.gru(x)              # [B, L, hidden]
        last = out[:, -1, :]              # [B, hidden]
        y = self.head(last)               # [B, H]
        return y
