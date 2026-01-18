import numpy as np
import torch
from torch.utils.data import Dataset

class NodeTrafficDataset(Dataset):
    """
    Dataset for a single node (sensor i).
    Takes X: [N_samples, seq_len, num_nodes]
          Y: [N_samples, horizon, num_nodes]
    Returns per-sample:
      x: [seq_len, 1]
      y: [horizon]   (or [horizon, 1] if you prefer)
    """
    def __init__(self, X: np.ndarray, Y: np.ndarray, node_idx: int):
        self.Xn = X[:, :, node_idx].astype(np.float32)  # [N, L]
        self.Yn = Y[:, :, node_idx].astype(np.float32)  # [N, H]

    def __len__(self):
        return self.Xn.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.Xn[idx]).unsqueeze(-1)  # [L, 1]
        y = torch.from_numpy(self.Yn[idx])                # [H]
        return x, y
