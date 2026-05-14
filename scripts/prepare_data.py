import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RAW_DIR = "data/raw"
OUT_DIR = "data/processed"

SEQ_LEN = 12    # past 1 hour (12 x 5min)
HORIZON = 12    # predict next hour

os.makedirs(OUT_DIR, exist_ok=True)

def load_metr_la():
    # Load traffic data
    df = pd.read_hdf("data/raw/metr-la.h5", key="df")
    data = df.values       # shape (T, N) — float64 speed values
    timestamps = df.index  # DatetimeIndex — needed for time-of-day features

    # Load adjacency matrix
    with open("data/raw/adj_mx.pkl", "rb") as f:
        _, _, adj = pickle.load(f, encoding="latin1")

    adj = np.array(adj)
    return data, adj, timestamps


def create_sliding_windows(data, seq_len, horizon, timestamps):
    """Slide a window over T to build supervised samples.

    Returns:
        X:       [N, seq_len, num_nodes]
        Y:       [N, horizon, num_nodes]
        T_idx:   [N] — index of first timestep in each input window
    """
    X, Y, T_idx = [], [], []
    T = data.shape[0]

    for t in range(T - seq_len - horizon):
        X.append(data[t : t + seq_len])
        Y.append(data[t + seq_len : t + seq_len + horizon])
        T_idx.append(t)  # store raw timestep index for time features

    return np.array(X), np.array(Y), np.array(T_idx)


def main():
    data, adj, timestamps = load_metr_la()
    print("Raw data shape:", data.shape)

    # ── Fix zero masking ──────────────────────────────────────────────────
    # In METR-LA, 0.0 means sensor failure / missing data (not real speed).
    # Treating zeros as valid speed values corrupts normalization because
    # sensors record highway speeds of 30-70 mph — zero is physically impossible.
    # Correct approach: mask zeros as NaN and impute with per-node means.
    data = data.astype(float)
    zeros_mask = (data == 0.0)
    n_zeros = zeros_mask.sum()
    print(f"Zero values (sensor failures): {n_zeros:,}  "
          f"({100.0 * n_zeros / data.size:.2f}% of all readings)")

    data[zeros_mask] = np.nan

    # Per-node (column) mean imputation
    col_means = np.nanmean(data, axis=0)          # shape (207,)
    nan_rows, nan_cols = np.where(np.isnan(data))
    data[nan_rows, nan_cols] = col_means[nan_cols]
    print(f"Imputed {len(nan_rows):,} missing values with per-node means.")

    # ── Normalize: per-node z-score ───────────────────────────────────────
    # StandardScaler on a 2-D array normalizes per column → per sensor node.
    scaler = StandardScaler()
    data = scaler.fit_transform(data)  # shape (T, 207), each col ~ N(0,1)
    print(f"Normalized. Global mean after: {data.mean():.4f}, std: {data.std():.4f}")

    # ── Build supervised windows ──────────────────────────────────────────
    X, Y, T_idx = create_sliding_windows(data, SEQ_LEN, HORIZON, timestamps)
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

    # ── Time-of-day feature indices ───────────────────────────────────────
    # For each sample window, record the 5-minute interval index (0..287)
    # of every input timestep.  288 = 24h * 12 five-minute slots.
    # dataset.py converts these to sin/cos cyclical features at load time.
    intervals_per_day = 24 * 12  # 288

    tod_indices = np.zeros((len(T_idx), SEQ_LEN), dtype=np.int32)
    for i, t in enumerate(T_idx):
        for s in range(SEQ_LEN):
            ts = timestamps[t + s]
            tod_indices[i, s] = (ts.hour * 60 + ts.minute) // 5  # 0..287

    print("Time-of-day index shape:", tod_indices.shape)
    print("Sample window 0 TOD values:", tod_indices[0].tolist())

    # ── Train / val / test split (time-ordered, no shuffling) ────────────
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
        "adj":       adj,
        # Time-of-day indices (0..287) per sample × per timestep
        "tod_train": tod_indices[:train_end],
        "tod_val":   tod_indices[train_end:val_end],
        "tod_test":  tod_indices[val_end:],
    }

    # Save scaler statistics (used by FLClient to denormalise predictions)
    np.savez(
        os.path.join(OUT_DIR, "scaler_stats.npz"),
        mean=scaler.mean_,
        std=scaler.scale_,
    )

    np.savez(os.path.join(OUT_DIR, "metr_la_processed.npz"), **dataset)

    print("\nSaved processed dataset to", OUT_DIR)
    print(f"  Train: {X[:train_end].shape[0]:,} samples")
    print(f"  Val:   {X[train_end:val_end].shape[0]:,} samples")
    print(f"  Test:  {X[val_end:].shape[0]:,} samples")


if __name__ == "__main__":
    main()
