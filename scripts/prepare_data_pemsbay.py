"""Prepare PEMS-BAY dataset for Federated Learning experiments.

Same preprocessing as METR-LA (prepare_data.py):
  1. Load raw .h5 → mask zeros → per-node mean imputation
  2. Z-score normalization (per-sensor)
  3. Sliding windows (12 in, 12 out)
  4. Time-of-day indices for sin/cos features

Output:
  data-PemsBay/processed/pemsbay_processed.npz
  data-PemsBay/processed/scaler_stats.npz

Usage:
  python scripts/prepare_data_pemsbay.py
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RAW_FILE = "data-PemsBay/pems-bay.h5"
OUT_DIR  = "data-PemsBay/processed"

SEQ_LEN = 12    # past 1 hour (12 x 5min)
HORIZON = 12    # predict next hour

os.makedirs(OUT_DIR, exist_ok=True)


def load_pemsbay():
    """Load PEMS-BAY speed data from HDF5."""
    df = pd.read_hdf(RAW_FILE)
    data = df.values.astype(float)  # shape (T, 325)
    timestamps = df.index           # DatetimeIndex
    print(f"  Raw shape: {data.shape}  ({data.shape[0]} timesteps x {data.shape[1]} sensors)")
    print(f"  Time range: {timestamps[0]} to {timestamps[-1]}")
    return data, timestamps


def create_sliding_windows(data, seq_len, horizon, timestamps):
    """Slide a window to build supervised samples.

    Returns:
        X:     [N, seq_len, num_nodes]
        Y:     [N, horizon, num_nodes]
        T_idx: [N] — index of first timestep in each window
    """
    X, Y, T_idx = [], [], []
    T = data.shape[0]

    for t in range(T - seq_len - horizon):
        X.append(data[t : t + seq_len])
        Y.append(data[t + seq_len : t + seq_len + horizon])
        T_idx.append(t)

    return np.array(X), np.array(Y), np.array(T_idx)


def main():
    print("=" * 60)
    print("  PEMS-BAY Data Preprocessing")
    print("=" * 60)

    data, timestamps = load_pemsbay()

    # ── Fix zero masking ──────────────────────────────────────────────────
    zeros_mask = (data == 0.0)
    n_zeros = zeros_mask.sum()
    print(f"\n  Zero values (sensor failures): {n_zeros:,}  "
          f"({100.0 * n_zeros / data.size:.2f}% of all readings)")

    data[zeros_mask] = np.nan

    # Per-node (column) mean imputation
    col_means = np.nanmean(data, axis=0)
    nan_rows, nan_cols = np.where(np.isnan(data))
    data[nan_rows, nan_cols] = col_means[nan_cols]
    print(f"  Imputed {len(nan_rows):,} missing values with per-node means.")

    # ── Normalize: per-node z-score ───────────────────────────────────────
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    print(f"  Normalized. Global mean: {data.mean():.4f}, std: {data.std():.4f}")

    # ── Build supervised windows ──────────────────────────────────────────
    X, Y, T_idx = create_sliding_windows(data, SEQ_LEN, HORIZON, timestamps)
    print(f"\n  X shape: {X.shape}")
    print(f"  Y shape: {Y.shape}")

    # ── Time-of-day feature indices ───────────────────────────────────────
    tod_indices = np.zeros((len(T_idx), SEQ_LEN), dtype=np.int32)
    for i, t in enumerate(T_idx):
        for s in range(SEQ_LEN):
            ts = timestamps[t + s]
            tod_indices[i, s] = (ts.hour * 60 + ts.minute) // 5  # 0..287

    print(f"  Time-of-day index shape: {tod_indices.shape}")

    # ── Train / val / test split (time-ordered) ──────────────────────────
    n = X.shape[0]
    train_end = int(n * 0.7)
    val_end   = int(n * 0.8)

    dataset = {
        "X_train":   X[:train_end],
        "Y_train":   Y[:train_end],
        "X_val":     X[train_end:val_end],
        "Y_val":     Y[train_end:val_end],
        "X_test":    X[val_end:],
        "Y_test":    Y[val_end:],
        "tod_train": tod_indices[:train_end],
        "tod_val":   tod_indices[train_end:val_end],
        "tod_test":  tod_indices[val_end:],
    }

    # Save scaler stats
    np.savez(
        os.path.join(OUT_DIR, "scaler_stats.npz"),
        mean=scaler.mean_,
        std=scaler.scale_,
    )

    np.savez(os.path.join(OUT_DIR, "pemsbay_processed.npz"), **dataset)

    print(f"\n  Saved to {OUT_DIR}/")
    print(f"    Train: {X[:train_end].shape[0]:,} samples")
    print(f"    Val:   {X[train_end:val_end].shape[0]:,} samples")
    print(f"    Test:  {X[val_end:].shape[0]:,} samples")
    print(f"    Sensors: {X.shape[2]}")


if __name__ == "__main__":
    main()
