"""Compare TSMixer vs GRU standalone (same node, same epochs)."""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.gru_forecaster import GRUForecaster
from models.tsmixer import TSMixer
from fl.dataset import NodeTrafficDataset

proc   = np.load("data/processed/metr_la_processed.npz")
scaler = np.load("data/processed/scaler_stats.npz")
X_train, Y_train = proc["X_train"], proc["Y_train"]
X_test,  Y_test  = proc["X_test"],  proc["Y_test"]
mean_all, std_all = scaler["mean"], scaler["std"]
nid = 18
device = "cuda" if torch.cuda.is_available() else "cpu"

def run(model, name, epochs=10):
    model = model.to(device)
    loader = DataLoader(NodeTrafficDataset(X_train, Y_train, nid), batch_size=64, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    for ep in range(epochs):
        tot, n = 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * x.size(0); n += x.size(0)
        if ep < 3 or ep == epochs-1:
            print(f"  {name} ep {ep+1:2d}: MSE={tot/n:.5f}")
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in DataLoader(NodeTrafficDataset(X_test, Y_test, nid), batch_size=256):
            preds.append(model(x.to(device)).cpu().numpy())
            trues.append(y.numpy())
    p = np.concatenate(preds) * std_all[nid] + mean_all[nid]
    t = np.concatenate(trues) * std_all[nid] + mean_all[nid]
    print(f"  {name} MAE={np.mean(np.abs(p-t)):.3f}")

seq_len = X_train.shape[1]
horizon = Y_train.shape[1]
print(f"Device: {device}\n")
run(GRUForecaster(hidden_size=128, num_layers=2, horizon=horizon, dropout=0.2), "GRU    ")
print()
run(TSMixer(seq_len=seq_len, horizon=horizon, n_blocks=6, ff_dim=256, dropout=0.1), "TSMixer")
