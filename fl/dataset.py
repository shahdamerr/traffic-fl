import numpy as np
import torch
from torch.utils.data import Dataset

# 288 = 24h * 12 five-minute intervals per day
_INTERVALS_PER_DAY = 288


class NodeTrafficDataset(Dataset):
    """Dataset for a single node (sensor i).

    Takes:
        X:           [N_samples, seq_len, num_nodes]  — normalized speed
        Y:           [N_samples, horizon, num_nodes]  — normalized speed targets
        node_idx:    which sensor column to extract
        tod_indices: [N_samples, seq_len] int32 — time-of-day index (0..287)
                     for each step in each window.  If None, time features
                     are omitted and x has shape [seq_len, 1] (legacy mode).

    Returns per-sample:
        x: [seq_len, n_features]  where n_features = 3 if tod given, else 1
               feature 0: normalized speed
               feature 1: sin(2π * tod / 288)   — time-of-day sine
               feature 2: cos(2π * tod / 288)   — time-of-day cosine
        y: [horizon]  — normalized speed targets
    """

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        node_idx: int,
        tod_indices: np.ndarray = None,
    ):
        self.Xn  = X[:, :, node_idx].astype(np.float32)   # [N, L]
        self.Yn  = Y[:, :, node_idx].astype(np.float32)   # [N, H]
        self.tod = tod_indices  # [N, L] int32 or None

    def __len__(self):
        return self.Xn.shape[0]

    def __getitem__(self, idx):
        speed = torch.from_numpy(self.Xn[idx]).unsqueeze(-1)  # [L, 1]

        if self.tod is not None:
            # Cyclical encoding: maps integer 0..287 to a point on the unit circle
            angle = 2.0 * np.pi * self.tod[idx].astype(np.float32) / _INTERVALS_PER_DAY
            sin_tod = torch.from_numpy(np.sin(angle)).unsqueeze(-1)   # [L, 1]
            cos_tod = torch.from_numpy(np.cos(angle)).unsqueeze(-1)   # [L, 1]
            x = torch.cat([speed, sin_tod, cos_tod], dim=-1)          # [L, 3]
        else:
            x = speed  # [L, 1]  — legacy / backward-compatible mode

        y = torch.from_numpy(self.Yn[idx])  # [H]
        return x, y
