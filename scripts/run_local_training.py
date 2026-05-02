"""
Local baseline training — optimized per-node GRU on METR-LA.

Trains all 207 nodes, evaluates on the held-out TEST set, and reports
per-horizon MAE / RMSE / MAPE at 15 min, 30 min, and 60 min.

Published baselines for comparison (METR-LA, horizon 12 = 60 min):
    ┌──────────────────┬───────┬───────┬──────────┬────────────────────────┐
    │ Model            │  MAE  │ RMSE  │ MAPE (%) │ Citation               │
    ├──────────────────┼───────┼───────┼──────────┼────────────────────────┤
    │ HA               │ 4.16  │  7.80 │  13.0    │ —                      │
    │ SVR              │ 5.15  │ 11.30 │  12.7    │ —                      │
    │ FC-LSTM          │ 4.37  │  8.69 │  13.2    │ Li et al., ICLR 2018   │
    │ DCRNN            │ 3.60  │  7.59 │  10.5    │ Li et al., ICLR 2018   │
    │ STGCN            │ 3.90  │  7.80 │   —      │ Yu et al., IJCAI 2018  │
    │ Graph WaveNet    │ 3.53  │  7.37 │  10.0    │ Wu et al., IJCAI 2019  │
    │ AGCRN            │ 3.49  │  7.31 │   9.9    │ Bai et al., NeurIPS 20 │
    └──────────────────┴───────┴───────┴──────────┴────────────────────────┘

Target for optimized local GRU: MAE ≈ 3.9–4.2  (at horizon 12)
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from torch.utils.data import DataLoader

from fl.dataset import NodeTrafficDataset
from fl.local_train import train_one_node, evaluate_model
from models.gru_forecaster import GRUForecaster
from utils.metrics import mae, rmse, mape, evaluate_per_horizon


# ──────────────────────────── configuration ──────────────────────────── #
EPOCHS = 100            # early stopping will cut this short (~30-50 typically)
LR = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 15
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2
BATCH_TRAIN = 64
BATCH_EVAL = 256
MAX_GRAD_NORM = 5.0


def main():
    # ── load data ──
    proc = np.load("data/processed/metr_la_processed.npz")
    X_train = proc["X_train"]   # [N, L, 207]
    Y_train = proc["Y_train"]   # [N, H, 207]
    X_val   = proc["X_val"]
    Y_val   = proc["Y_val"]
    X_test  = proc["X_test"]
    Y_test  = proc["Y_test"]

    scaler_stats = np.load("data/processed/scaler_stats.npz")
    mean_all = scaler_stats["mean"]   # [207]
    std_all  = scaler_stats["std"]    # [207]

    clusters_npz = np.load("data/processed/graph_clusters.npz")
    cluster_labels = clusters_npz["cluster_labels"]
    num_clusters = int(clusters_npz["num_clusters"])

    num_nodes = X_train.shape[2]
    seq_len   = X_train.shape[1]
    horizon   = Y_train.shape[1]

    print(f"Nodes: {num_nodes}  |  Seq: {seq_len}  |  Horizon: {horizon}")
    print(f"Clusters: {num_clusters}  |  Labels: {cluster_labels.shape}")
    print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    os.makedirs("data/models_local", exist_ok=True)

    # ── train all nodes ──
    nodes_to_train = list(range(num_nodes))

    # Accumulate per-node test predictions for aggregate metrics
    all_preds = []   # each entry: [N_test, H]
    all_trues = []

    for i, node_idx in enumerate(nodes_to_train):
        print(f"\n{'='*60}")
        print(f"  Node {node_idx:3d}/{num_nodes}  "
              f"(cluster {cluster_labels[node_idx]})  "
              f"[{i+1}/{len(nodes_to_train)}]")
        print(f"{'='*60}")

        # datasets
        train_ds = NodeTrafficDataset(X_train, Y_train, node_idx)
        val_ds   = NodeTrafficDataset(X_val,   Y_val,   node_idx)
        test_ds  = NodeTrafficDataset(X_test,  Y_test,  node_idx)

        train_loader = DataLoader(train_ds, batch_size=BATCH_TRAIN, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_EVAL,  shuffle=False)
        test_loader  = DataLoader(test_ds,  batch_size=BATCH_EVAL,  shuffle=False)

        # model
        model = GRUForecaster(
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            horizon=horizon,
            dropout=DROPOUT,
        )

        # train (with early stopping)
        model = train_one_node(
            model,
            train_loader,
            val_loader,
            epochs=EPOCHS,
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            patience=PATIENCE,
            max_grad_norm=MAX_GRAD_NORM,
            device=device,
        )

        # ── evaluate on TEST set ──
        node_mean = mean_all[node_idx]
        node_std  = std_all[node_idx]

        preds, trues = evaluate_model(
            model, test_loader,
            mean=node_mean, std=node_std,
            device=device,
        )

        all_preds.append(preds)
        all_trues.append(trues)

        node_mae  = mae(trues, preds)
        node_rmse = rmse(trues, preds)
        node_mape = mape(trues, preds)

        print(f"  TEST  Node {node_idx:3d} | "
              f"MAE={node_mae:.3f}  RMSE={node_rmse:.3f}  MAPE={node_mape:.2f}%")

        # per-horizon
        h_metrics = evaluate_per_horizon(trues, preds)
        for label, m in h_metrics.items():
            print(f"         {label:>5s} | "
                  f"MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}  MAPE={m['mape']:.2f}%")

        # save checkpoint
        out_path = f"data/models_local/node_{node_idx}.pt"
        torch.save(model.state_dict(), out_path)

    # ══════════════════════════════════════════════════════════════════
    #  AGGREGATE METRICS (what papers report)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  AGGREGATE TEST METRICS ACROSS ALL NODES")
    print("=" * 70)

    # Stack: [num_nodes, N_test, H] → concat along node axis for overall
    # But per-horizon metrics should be averaged across nodes
    all_preds_np = np.array(all_preds)  # [num_nodes, N_test, H]
    all_trues_np = np.array(all_trues)  # [num_nodes, N_test, H]

    # Overall (flatten all nodes, all samples, all horizons)
    overall_mae  = mae(all_trues_np, all_preds_np)
    overall_rmse = rmse(all_trues_np, all_preds_np)
    overall_mape = mape(all_trues_np, all_preds_np)

    print(f"\n  Overall:  MAE={overall_mae:.3f}  "
          f"RMSE={overall_rmse:.3f}  MAPE={overall_mape:.2f}%")

    # Per-horizon aggregate
    horizons = {"15min": 2, "30min": 5, "60min": 11}
    print(f"\n  {'Horizon':>8s} | {'MAE':>7s} | {'RMSE':>7s} | {'MAPE(%)':>8s}")
    print(f"  {'-'*8}-+-{'-'*7}-+-{'-'*7}-+-{'-'*8}")

    for label, h_idx in horizons.items():
        if h_idx >= all_preds_np.shape[2]:
            continue
        yt = all_trues_np[:, :, h_idx].flatten()
        yp = all_preds_np[:, :, h_idx].flatten()
        h_mae  = mae(yt, yp)
        h_rmse = rmse(yt, yp)
        h_mape = mape(yt, yp)
        print(f"  {label:>8s} | {h_mae:7.3f} | {h_rmse:7.3f} | {h_mape:8.2f}")

    print("\n" + "=" * 70)
    print("  Done. Models saved to data/models_local/")

    # save aggregate results for later comparison
    np.savez(
        "data/models_local/aggregate_results.npz",
        all_preds=all_preds_np,
        all_trues=all_trues_np,
    )
    print("  Aggregate predictions saved to data/models_local/aggregate_results.npz")


if __name__ == "__main__":
    main()
