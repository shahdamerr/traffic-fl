import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from torch.utils.data import DataLoader

from fl.dataset import NodeTrafficDataset
from fl.local_train import train_one_node, evaluate_model
from models.gru_forecaster import GRUForecaster
from utils.metrics import mae, rmse


def main():
    # Load processed data
    proc = np.load("data/processed/metr_la_processed.npz")
    X_train = proc["X_train"]   # [N, L, 207]
    Y_train = proc["Y_train"]   # [N, H, 207]
    X_val = proc["X_val"]
    Y_val = proc["Y_val"]

    # Load scaler stats
    scaler_stats = np.load("data/processed/scaler_stats.npz")
    mean_all = scaler_stats["mean"]   # [207]
    std_all = scaler_stats["std"]     # [207]

    # Load cluster labels
    clusters_npz = np.load("data/processed/graph_clusters.npz")
    cluster_labels = clusters_npz["cluster_labels"]
    num_clusters = int(clusters_npz["num_clusters"])
    print("Loaded clusters:", num_clusters, "labels shape:", cluster_labels.shape)

    num_nodes = X_train.shape[2]
    seq_len = X_train.shape[1]
    horizon = Y_train.shape[1]
    print("Train shapes:", X_train.shape, Y_train.shape)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    os.makedirs("data/models_local", exist_ok=True)

    # Start small
    #nodes_to_train = list(range(num_nodes))

    nodes_to_train = [0, 1, 2, 3, 4]

    for node_idx in nodes_to_train:
        print(f"\n=== Training node {node_idx} (cluster {cluster_labels[node_idx]}) ===")

        train_ds = NodeTrafficDataset(X_train, Y_train, node_idx)
        val_ds = NodeTrafficDataset(X_val, Y_val, node_idx)

        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

        model = GRUForecaster(hidden_size=64, num_layers=1, horizon=horizon)

        model = train_one_node(
            model,
            train_loader,
            val_loader,
            epochs=2,
            lr=1e-3,
            device=device
        )

        # ==========================
        # 🔹 Evaluation (NEW PART)
        # ==========================
        node_mean = mean_all[node_idx]
        node_std = std_all[node_idx]

        preds, trues = evaluate_model(
            model,
            val_loader,
            mean=node_mean,
            std=node_std,
            device=device
        )

        node_mae = mae(trues, preds)
        node_rmse = rmse(trues, preds)

        print(
            f"Node {node_idx} | "
            f"MAE={node_mae:.3f}, RMSE={node_rmse:.3f}"
        )

        # Save model weights
        out_path = f"data/models_local/node_{node_idx}.pt"
        torch.save(model.state_dict(), out_path)
        print("Saved:", out_path)


if __name__ == "__main__":
    main()
