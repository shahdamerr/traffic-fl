import torch
import torch.nn as nn

class GRUForecaster(nn.Module):
    """
    Input:  x [B, L, 1]
    Output: y [B, H]
    """
    def __init__(self, hidden_size: int = 64, num_layers: int = 1, horizon: int = 12):
        super().__init__()
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        # x: [B, L, 1]
        out, _ = self.gru(x)          # out: [B, L, hidden]
        last = out[:, -1, :]          # [B, hidden]
        y = self.head(last)           # [B, H]
        return y
