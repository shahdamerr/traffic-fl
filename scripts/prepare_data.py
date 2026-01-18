import os
import pickle
import h5py
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
    data = df.values  # shape (T, N)

    # Load adjacency (CORRECT extraction)
    with open("data/raw/adj_mx.pkl", "rb") as f:
        _, _, adj = pickle.load(f, encoding="latin1")

    adj = np.array(adj)
    return data, adj


def create_sliding_windows(data, seq_len, horizon):
    X, Y = [], []
    T = data.shape[0]
    
    for t in range(T - seq_len - horizon):
        X.append(data[t:t+seq_len])
        Y.append(data[t+seq_len:t+seq_len+horizon])
    
    return np.array(X), np.array(Y)

def main():
    data, adj = load_metr_la()
    print("Raw data shape:", data.shape)

    # Handle missing values
    data = np.nan_to_num(data, nan=np.nanmean(data))

    # Normalize (global z-score)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    # Create supervised samples
    X, Y = create_sliding_windows(data, SEQ_LEN, HORIZON)
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

    # Train/val/test split (time-based)
    n = X.shape[0]
    train_end = int(n * 0.7)
    val_end = int(n * 0.8)

    dataset = {
        "X_train": X[:train_end],
        "Y_train": Y[:train_end],
        "X_val": X[train_end:val_end],
        "Y_val": Y[train_end:val_end],
        "X_test": X[val_end:],
        "Y_test": Y[val_end:],
        "adj": adj,
    }

    np.savez(
    os.path.join(OUT_DIR, "scaler_stats.npz"),
    mean=scaler.mean_,
    std=scaler.scale_
    )

    np.savez(os.path.join(OUT_DIR, "metr_la_processed.npz"), **dataset)
    print("Saved processed dataset")

if __name__ == "__main__":
    main()
