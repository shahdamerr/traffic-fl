"""
Statistical Baselines for METR-LA
=================================
Evaluates three simple baselines on the same test set used by the FL experiments:

  1. Persistence (Last Value) - predict y_hat[t] = x[-1] for all horizons
  2. Historical Average      - predict the per-sensor training-set mean
  3. Ridge Regression         - linear model from 12 input steps to 12 output steps

All evaluation is on DENORMALIZED (mph) values using the same metrics as
the FL experiments (MAE, RMSE, MAPE with |y|>5 masking).

Usage:
    python scripts/run_statistical_baselines.py

No GPU required. Runs in under 1 minute.
"""

import os
import sys
import time
import numpy as np

# Project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.metrics import mae, rmse, mape, evaluate_per_horizon

# ── Paths ────────────────────────────────────────────────────────────────────
PROC_NPZ = "data/processed/metr_la_processed.npz"
SCALER   = "data/processed/scaler_stats.npz"


def load_data():
    """Load processed data and scaler stats."""
    print("[1/5] Loading data...")
    data = np.load(PROC_NPZ)
    scaler = np.load(SCALER)

    X_train = data["X_train"]  # (N_train, 12, 207) - normalized
    Y_train = data["Y_train"]  # (N_train, 12, 207) - normalized
    X_test  = data["X_test"]   # (N_test, 12, 207)  - normalized
    Y_test  = data["Y_test"]   # (N_test, 12, 207)  - normalized

    mean = scaler["mean"]  # (207,)
    std  = scaler["std"]   # (207,)

    print(f"    X_train: {X_train.shape}")
    print(f"    X_test:  {X_test.shape}")
    print(f"    Y_test:  {Y_test.shape}")
    print(f"    Sensors: {X_test.shape[2]}")

    return X_train, Y_train, X_test, Y_test, mean, std


def denormalize(arr, mean, std):
    """Reverse z-score normalization: x_original = x_normalized * std + mean.

    arr shape: (N, 12, 207) or (N, 12) for a single sensor.
    mean/std shape: (207,) - broadcast over first two dims.
    """
    return arr * std + mean


def evaluate_baseline(name, Y_true, Y_pred):
    """Compute and print all metrics for a baseline."""
    # Y_true, Y_pred: (N, 12, 207) in original mph units
    N, H, S = Y_true.shape

    # Flatten for overall metrics: (N*12*207,)
    yt_flat = Y_true.reshape(-1)
    yp_flat = Y_pred.reshape(-1)

    overall_mae  = mae(yt_flat, yp_flat)
    overall_rmse = rmse(yt_flat, yp_flat)
    overall_mape = mape(yt_flat, yp_flat)

    # Per-horizon metrics: average across all sensors
    # Reshape to (N*S, H) for evaluate_per_horizon
    yt_h = Y_true.transpose(0, 2, 1).reshape(-1, H)  # (N*207, 12)
    yp_h = Y_pred.transpose(0, 2, 1).reshape(-1, H)   # (N*207, 12)
    horizon_results = evaluate_per_horizon(yt_h, yp_h)

    print(f"\n  {name}")
    print(f"  {'='*60}")
    print(f"    Overall:  MAE={overall_mae:.3f}  RMSE={overall_rmse:.3f}  MAPE={overall_mape:.2f}%")
    print(f"    15 min:   MAE={horizon_results['15min']['mae']:.3f}")
    print(f"    30 min:   MAE={horizon_results['30min']['mae']:.3f}")
    print(f"    60 min:   MAE={horizon_results['60min']['mae']:.3f}")

    return {
        "mae": overall_mae,
        "rmse": overall_rmse,
        "mape": overall_mape,
        "h15": horizon_results["15min"]["mae"],
        "h30": horizon_results["30min"]["mae"],
        "h60": horizon_results["60min"]["mae"],
    }


# ═══════════════════════════════════════════════════════════════════════════
# BASELINE 1: Persistence (Last Value)
# ═══════════════════════════════════════════════════════════════════════════

def persistence_baseline(X_test, Y_test, mean, std):
    """Predict y_hat[t] = last observed value for all future horizons.

    This is the simplest possible baseline: assume traffic speed
    stays exactly the same as the last observed value.
    """
    print("\n[2/5] Persistence baseline...")

    # Last observed value: X_test[:, -1, :] -> shape (N, 207)
    last_val = X_test[:, -1, :]  # normalized

    # Repeat across all 12 horizons: (N, 12, 207)
    Y_pred_norm = np.repeat(last_val[:, np.newaxis, :], 12, axis=1)

    # Denormalize both
    Y_true_mph = denormalize(Y_test, mean, std)
    Y_pred_mph = denormalize(Y_pred_norm, mean, std)

    return evaluate_baseline("PERSISTENCE (Last Value)", Y_true_mph, Y_pred_mph)


# ═══════════════════════════════════════════════════════════════════════════
# BASELINE 2: Historical Average
# ═══════════════════════════════════════════════════════════════════════════

def historical_average_baseline(X_train, Y_train, Y_test, mean, std):
    """Predict the per-sensor mean of all training targets.

    For each sensor, the prediction is the average speed across all
    training samples and all horizons. This tests whether the model
    learns anything beyond the unconditional mean.
    """
    print("\n[3/5] Historical Average baseline...")

    # Compute per-sensor mean from training targets (normalized)
    # Y_train: (N_train, 12, 207)
    sensor_means = Y_train.mean(axis=(0, 1))  # (207,) - normalized

    # Broadcast to test shape: (N_test, 12, 207)
    N_test = Y_test.shape[0]
    Y_pred_norm = np.broadcast_to(
        sensor_means[np.newaxis, np.newaxis, :], (N_test, 12, 207)
    ).copy()

    # Denormalize both
    Y_true_mph = denormalize(Y_test, mean, std)
    Y_pred_mph = denormalize(Y_pred_norm, mean, std)

    return evaluate_baseline("HISTORICAL AVERAGE (Per-Sensor Mean)", Y_true_mph, Y_pred_mph)


# ═══════════════════════════════════════════════════════════════════════════
# BASELINE 3: Ridge Regression (Linear)
# ═══════════════════════════════════════════════════════════════════════════

def ridge_baseline(X_train, Y_train, X_test, Y_test, mean, std, alpha=1.0):
    """Fit a per-sensor Ridge regression from 12 input steps to 12 output steps.

    This directly addresses the examiner question: "Why not use a linear model?"
    If Ridge achieves comparable performance, the GRU is not justified.
    If Ridge is substantially worse, it proves nonlinear modeling is needed.

    For each sensor independently:
        Input:  x_i = [x(t-11), x(t-10), ..., x(t)]  (12 features)
        Output: y_i = [y(t+1), y(t+2), ..., y(t+12)]  (12 targets)
        Model:  y_i = W @ x_i + b  (Ridge regression, alpha=1.0)
    """
    print(f"\n[4/5] Ridge Regression baseline (alpha={alpha})...")
    from sklearn.linear_model import Ridge

    N_sensors = X_train.shape[2]
    N_test = X_test.shape[0]

    Y_pred_norm = np.zeros_like(Y_test)  # (N_test, 12, 207)

    for s in range(N_sensors):
        # Extract per-sensor data
        x_tr = X_train[:, :, s]  # (N_train, 12)
        y_tr = Y_train[:, :, s]  # (N_train, 12)
        x_te = X_test[:, :, s]   # (N_test, 12)

        # Fit Ridge: 12 inputs -> 12 outputs
        model = Ridge(alpha=alpha)
        model.fit(x_tr, y_tr)

        # Predict
        Y_pred_norm[:, :, s] = model.predict(x_te)

    # Denormalize
    Y_true_mph = denormalize(Y_test, mean, std)
    Y_pred_mph = denormalize(Y_pred_norm, mean, std)

    return evaluate_baseline(f"RIDGE REGRESSION (alpha={alpha})", Y_true_mph, Y_pred_mph)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  STATISTICAL BASELINES FOR METR-LA")
    print("  (No GPU required)")
    print("=" * 65)

    t0 = time.time()

    # Load data
    X_train, Y_train, X_test, Y_test, mean, std = load_data()

    # Run baselines
    results = {}
    results["Persistence"]    = persistence_baseline(X_test, Y_test, mean, std)
    results["Hist. Average"]  = historical_average_baseline(X_train, Y_train, Y_test, mean, std)
    results["Ridge (a=1.0)"]  = ridge_baseline(X_train, Y_train, X_test, Y_test, mean, std, alpha=1.0)

    elapsed = time.time() - t0

    # ── Summary table ────────────────────────────────────────────────────
    print("\n\n" + "=" * 65)
    print("[5/5] SUMMARY TABLE")
    print("=" * 65)

    header = (f"  {'Method':<25s} | {'MAE':>6s} | {'RMSE':>6s} | {'MAPE%':>6s} | "
              f"{'15min':>6s} | {'30min':>6s} | {'60min':>6s}")
    print(header)
    print("  " + "-" * 75)

    for name, r in results.items():
        print(f"  {name:<25s} | {r['mae']:6.3f} | {r['rmse']:6.3f} | {r['mape']:6.2f} | "
              f"{r['h15']:6.3f} | {r['h30']:6.3f} | {r['h60']:6.3f}")

    # Add FL results for comparison
    print("  " + "-" * 75)
    print(f"  {'Run A (Global FedAvg)':<25s} | {3.832:6.3f} | {7.171:6.3f} | {11.11:6.2f} | "
          f"{3.078:6.3f} | {3.868:6.3f} | {4.896:6.3f}")
    print(f"  {'Run G (Proposed)':<25s} | {3.676:6.3f} | {6.759:6.3f} | {10.46:6.2f} | "
          f"{3.051:6.3f} | {3.705:6.3f} | {4.559:6.3f}")

    print(f"\n  Total time: {elapsed:.1f}s")

    # ── LaTeX table ──────────────────────────────────────────────────────
    print("\n\n  LaTeX table (copy-paste):")
    print("  " + "-" * 65)
    for name, r in results.items():
        latex_name = name.replace("_", r"\_")
        print(f"  {latex_name} & {r['mae']:.3f} & {r['rmse']:.3f} & {r['mape']:.2f} "
              f"& {r['h15']:.3f} & {r['h30']:.3f} & {r['h60']:.3f} \\\\")

    # ── Save results ─────────────────────────────────────────────────────
    np.savez(
        "results/statistical_baselines.npz",
        **{name: np.array([r['mae'], r['rmse'], r['mape'],
                           r['h15'], r['h30'], r['h60']])
           for name, r in results.items()}
    )
    print("\n  Results saved to: results/statistical_baselines.npz")


if __name__ == "__main__":
    main()
