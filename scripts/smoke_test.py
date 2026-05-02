"""Quick smoke test: run all 5 models on 3 nodes to verify everything works."""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from torch.utils.data import DataLoader
from fl.dataset import NodeTrafficDataset
from fl.local_train import train_one_node, evaluate_model
from models.gru_forecaster import GRUForecaster
from models.lstm_forecaster import LSTMForecaster
from models.seq2seq_gru import Seq2SeqGRU
from models.baselines import naive_forecast, moving_average_forecast
from utils.metrics import mae, rmse, mape, evaluate_per_horizon

TEST_NODES = [0, 100, 200]
EPOCHS = 15  # quick


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    proc = np.load("data/processed/metr_la_processed.npz")
    s = np.load("data/processed/scaler_stats.npz")
    horizon = proc["Y_train"].shape[1]
    mean_all, std_all = s["mean"], s["std"]

    results = {}

    # 1. Statistical baselines
    print("\n=== Naive & Moving Average ===")
    for name, fn in [("Naive", naive_forecast), ("MovingAvg", moving_average_forecast)]:
        maes_60 = []
        for n in TEST_NODES:
            x = proc["X_test"][:, :, n]
            y = proc["Y_test"][:, :, n]
            p = fn(x, horizon)
            p_d = p * std_all[n] + mean_all[n]
            t_d = y * std_all[n] + mean_all[n]
            maes_60.append(mae(t_d[:, 11], p_d[:, 11]))
        avg = np.mean(maes_60)
        print(f"  {name}: avg 60min MAE = {avg:.3f}")
        results[name] = avg

    # 2. Neural models
    for name, cls, s2s in [("LSTM", LSTMForecaster, False),
                           ("GRU", GRUForecaster, False),
                           ("Seq2Seq", Seq2SeqGRU, True)]:
        print(f"\n=== {name} ===")
        maes_60 = []
        for n in TEST_NODES:
            model = cls(hidden_size=128, num_layers=2, horizon=horizon, dropout=0.2)
            tl = DataLoader(NodeTrafficDataset(proc["X_train"], proc["Y_train"], n),
                            batch_size=64, shuffle=True)
            vl = DataLoader(NodeTrafficDataset(proc["X_val"], proc["Y_val"], n),
                            batch_size=256, shuffle=False)
            tel = DataLoader(NodeTrafficDataset(proc["X_test"], proc["Y_test"], n),
                             batch_size=256, shuffle=False)

            model = train_one_node(model, tl, vl, epochs=EPOCHS, lr=1e-3,
                                   weight_decay=1e-5, patience=10,
                                   max_grad_norm=5.0, device=device,
                                   seq2seq=s2s, tf_start=0.5, tf_end=0.0)

            preds, trues = evaluate_model(model, tel, mean_all[n], std_all[n],
                                          device=device, seq2seq=s2s)
            h = evaluate_per_horizon(trues, preds)
            m60 = h["60min"]["mae"]
            maes_60.append(m60)
            print(f"  Node {n}: 60min MAE={m60:.3f}, range=[{preds.min():.0f},{preds.max():.0f}]")

        avg = np.mean(maes_60)
        print(f"  {name} avg 60min MAE = {avg:.3f}")
        results[name] = avg

    print("\n" + "=" * 40)
    print("SMOKE TEST SUMMARY (60min MAE)")
    print("=" * 40)
    for name, v in results.items():
        print(f"  {name:<12s}: {v:.3f}")
    print("=" * 40)
    print("DONE")


if __name__ == "__main__":
    main()
