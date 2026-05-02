"""Sanity checks on the data pipeline and representative node selection."""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np


def main():
    proc = np.load("data/processed/metr_la_processed.npz")
    s = np.load("data/processed/scaler_stats.npz")
    X_train, Y_train = proc["X_train"], proc["Y_train"]
    X_val, Y_val = proc["X_val"], proc["Y_val"]
    X_test, Y_test = proc["X_test"], proc["Y_test"]
    mean_all, std_all = s["mean"], s["std"]

    # == Check 1: Denormalization ==
    print("=== Check 1: Denormalization ===")
    node = 0
    raw_y = Y_test[:3, :, node]
    denorm_y = raw_y * std_all[node] + mean_all[node]
    print(f"  Node {node}: mean={mean_all[node]:.2f}, std={std_all[node]:.2f}")
    print(f"  Normalized Y_test[0, :3]: {raw_y[0, :3]}")
    print(f"  Denormed   Y_test[0, :3]: {denorm_y[0, :3]}")
    print(f"  Range: [{denorm_y.min():.1f}, {denorm_y.max():.1f}] (expect ~0-70 mph)")
    ok1 = denorm_y.min() >= -5 and denorm_y.max() <= 80
    print(f"  PASS: {ok1}")

    # == Check 2: No data leakage ==
    print("\n=== Check 2: No Data Leakage (time-based split) ===")
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Val:   {X_val.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")
    last_train = X_train[-1, :, 0]
    first_val = X_val[0, :, 0]
    overlap = np.array_equal(last_train, first_val)
    print(f"  Last train == first val? {overlap} (should be False)")
    ok2 = not overlap
    print(f"  PASS: {ok2}")

    # == Check 3: Y alignment (Y follows X in time) ==
    print("\n=== Check 3: Y alignment ===")
    x_last = X_train[0, -1, 0]
    y_first = Y_train[0, 0, 0]
    diff = abs(y_first - x_last)
    print(f"  X_train[0, -1, node0] = {x_last:.4f}")
    print(f"  Y_train[0,  0, node0] = {y_first:.4f}")
    print(f"  Difference: {diff:.4f} (consecutive timesteps, should be small)")
    # Sliding window check
    x0_second = X_train[0, 1, 0]
    x1_first = X_train[1, 0, 0]
    same = abs(x0_second - x1_first) < 1e-6
    print(f"  X[0,1] == X[1,0]? {same} (sliding window shift=1)")
    ok3 = diff < 2.0 and same
    print(f"  PASS: {ok3}")

    # == Select representative nodes ==
    print("\n=== Representative Node Selection ===")
    clusters = np.load("data/processed/graph_clusters.npz")
    labels = clusters["cluster_labels"]
    num_clusters = int(clusters["num_clusters"])

    res = np.load("data/models_local/aggregate_results.npz")
    preds, trues = res["all_preds"], res["all_trues"]
    node_mae_60 = np.mean(np.abs(trues[:, :, 11] - preds[:, :, 11]), axis=1)

    selected = []
    for c in range(num_clusters):
        nodes_in_c = np.where(labels == c)[0]
        maes = node_mae_60[nodes_in_c]
        sorted_idx = np.argsort(maes)
        median_node = nodes_in_c[sorted_idx[len(sorted_idx) // 2]]
        hard_node = nodes_in_c[sorted_idx[-1]]
        selected.extend([int(median_node), int(hard_node)])
        print(f"  Cluster {c}: median=node {median_node} (MAE={node_mae_60[median_node]:.2f}), "
              f"hard=node {hard_node} (MAE={node_mae_60[hard_node]:.2f})")

    selected = sorted(selected)
    print(f"\nDEV_NODES = {selected}")
    print(f"  Count: {len(selected)}")
    print(f"  Overall 60min MAE: all={node_mae_60.mean():.3f}, subset={node_mae_60[selected].mean():.3f}")

    all_ok = ok1 and ok2 and ok3
    print(f"\n{'='*40}")
    print(f"  ALL SANITY CHECKS: {'PASSED' if all_ok else 'FAILED'}")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
