"""Run all baseline models on the dev subset and produce a comparison table.

Models:
    1. Naive (last value)
    2. Moving Average (12-step window)
    3. LSTM (direct multi-step)
    4. GRU (direct multi-step)
    5. Seq2Seq GRU (autoregressive)

Usage:
    python scripts/run_baseline_comparison.py
"""
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import DEV_NODES, EPOCHS, LR, WEIGHT_DECAY, PATIENCE, HIDDEN_SIZE
from config import NUM_LAYERS, DROPOUT, BATCH_TRAIN, BATCH_EVAL, MAX_GRAD_NORM
from config import TEACHER_FORCING_START, TEACHER_FORCING_END
from fl.dataset import NodeTrafficDataset
from fl.local_train import train_one_node, evaluate_model
from models.gru_forecaster import GRUForecaster
from models.lstm_forecaster import LSTMForecaster
from models.seq2seq_gru import Seq2SeqGRU
from models.baselines import naive_forecast, moving_average_forecast
from utils.metrics import mae, rmse, mape, evaluate_per_horizon


def run_statistical_baselines(X_test, Y_test, mean_all, std_all, nodes, horizon):
    """Run Naive and Moving Average baselines (no training needed)."""
    results = {}

    for name, forecast_fn in [("Naive", naive_forecast), ("MovingAvg", moving_average_forecast)]:
        all_preds, all_trues = [], []

        for n in nodes:
            x_node = X_test[:, :, n]  # [N_test, L]
            y_node = Y_test[:, :, n]  # [N_test, H]

            # Predict (in normalized space)
            pred = forecast_fn(x_node, horizon)  # [N_test, H]

            # Denormalize
            pred_denorm = pred * std_all[n] + mean_all[n]
            true_denorm = y_node * std_all[n] + mean_all[n]

            all_preds.append(pred_denorm)
            all_trues.append(true_denorm)

        # Stack: [N_nodes, N_test, H] -> flatten to [N_nodes*N_test, H]
        p = np.concatenate(all_preds, axis=0)
        t = np.concatenate(all_trues, axis=0)

        h = evaluate_per_horizon(
            np.stack(all_trues),   # need shape for per-sample evaluation
            np.stack(all_preds),
        )
        # Compute aggregate across all samples from all nodes
        results[name] = {
            "mae": mae(t, p),
            "rmse": rmse(t, p),
            "mape": mape(t, p),
            "h15_mae": mae(
                np.concatenate([tr[:, 2:3] for tr in all_trues]),
                np.concatenate([pr[:, 2:3] for pr in all_preds]),
            ),
            "h30_mae": mae(
                np.concatenate([tr[:, 5:6] for tr in all_trues]),
                np.concatenate([pr[:, 5:6] for pr in all_preds]),
            ),
            "h60_mae": mae(
                np.concatenate([tr[:, 11:12] for tr in all_trues]),
                np.concatenate([pr[:, 11:12] for pr in all_preds]),
            ),
        }
        print(f"  {name}: MAE={results[name]['mae']:.3f}  RMSE={results[name]['rmse']:.3f}")

    return results


def run_neural_model(model_name, model_class, proc, scaler, nodes, device,
                     seq2seq=False):
    """Train and evaluate a neural model on all dev nodes."""
    X_train, Y_train = proc["X_train"], proc["Y_train"]
    X_val, Y_val = proc["X_val"], proc["Y_val"]
    X_test, Y_test = proc["X_test"], proc["Y_test"]
    mean_all, std_all = scaler["mean"], scaler["std"]
    horizon = Y_train.shape[1]

    all_preds, all_trues = [], []

    for i, n in enumerate(nodes):
        print(f"\n  [{model_name}] Node {n} ({i+1}/{len(nodes)})")

        model = model_class(
            hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
            horizon=horizon, dropout=DROPOUT
        )
        tl = DataLoader(
            NodeTrafficDataset(X_train, Y_train, n),
            batch_size=BATCH_TRAIN, shuffle=True,
        )
        vl = DataLoader(
            NodeTrafficDataset(X_val, Y_val, n),
            batch_size=BATCH_EVAL, shuffle=False,
        )
        tel = DataLoader(
            NodeTrafficDataset(X_test, Y_test, n),
            batch_size=BATCH_EVAL, shuffle=False,
        )

        model = train_one_node(
            model, tl, vl,
            epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY,
            patience=PATIENCE, max_grad_norm=MAX_GRAD_NORM,
            device=device, seq2seq=seq2seq,
            tf_start=TEACHER_FORCING_START, tf_end=TEACHER_FORCING_END,
        )

        preds, trues = evaluate_model(model, tel, mean_all[n], std_all[n],
                                       device=device, seq2seq=seq2seq)
        all_preds.append(preds)
        all_trues.append(trues)

        node_mae = mae(trues, preds)
        h = evaluate_per_horizon(trues, preds)
        print(f"    TEST: MAE={node_mae:.3f}  "
              f"15m={h['15min']['mae']:.3f}  "
              f"30m={h['30min']['mae']:.3f}  "
              f"60m={h['60min']['mae']:.3f}")

    # Aggregate
    p = np.concatenate(all_preds, axis=0)
    t = np.concatenate(all_trues, axis=0)

    result = {
        "mae": mae(t, p),
        "rmse": rmse(t, p),
        "mape": mape(t, p),
        "h15_mae": mae(
            np.concatenate([tr[:, 2] for tr in all_trues]),
            np.concatenate([pr[:, 2] for pr in all_preds]),
        ),
        "h30_mae": mae(
            np.concatenate([tr[:, 5] for tr in all_trues]),
            np.concatenate([pr[:, 5] for pr in all_preds]),
        ),
        "h60_mae": mae(
            np.concatenate([tr[:, 11] for tr in all_trues]),
            np.concatenate([pr[:, 11] for pr in all_preds]),
        ),
    }
    return result


def print_results_table(results):
    """Print formatted comparison table."""
    print("\n")
    print("=" * 80)
    print("  BASELINE COMPARISON RESULTS (Dev Subset: {} nodes)".format(len(DEV_NODES)))
    print("=" * 80)

    header = (f"  {'Model':<16s} | {'Overall MAE':>11s} | {'RMSE':>7s} | {'MAPE%':>7s} | "
              f"{'15min':>7s} | {'30min':>7s} | {'60min':>7s}")
    print(header)
    print("  " + "-" * 76)

    # Published baselines for context
    published = {
        "DCRNN*":     {"mae": 3.17, "rmse": 6.45, "mape": 8.8,
                       "h15_mae": 2.77, "h30_mae": 3.15, "h60_mae": 3.60},
        "STGCN*":     {"mae": 3.38, "rmse": 7.10, "mape": None,
                       "h15_mae": 2.88, "h30_mae": 3.47, "h60_mae": 3.90},
        "GWaveNet*":  {"mae": 3.07, "rmse": 6.22, "mape": 8.4,
                       "h15_mae": 2.69, "h30_mae": 3.07, "h60_mae": 3.53},
    }

    for name, r in results.items():
        mape_str = f"{r['mape']:7.2f}" if r['mape'] is not None else "    N/A"
        print(f"  {name:<16s} | {r['mae']:11.3f} | {r['rmse']:7.3f} | {mape_str} | "
              f"{r['h15_mae']:7.3f} | {r['h30_mae']:7.3f} | {r['h60_mae']:7.3f}")

    print("  " + "-" * 76)
    print("  Published centralized baselines (for context, all 207 nodes):")
    print("  " + "-" * 76)
    for name, r in published.items():
        mape_str = f"{r['mape']:7.1f}" if r['mape'] is not None else "    N/A"
        print(f"  {name:<16s} | {r['mae']:11.3f} | {r['rmse']:7.3f} | {mape_str} | "
              f"{r['h15_mae']:7.3f} | {r['h30_mae']:7.3f} | {r['h60_mae']:7.3f}")

    print("  " + "=" * 76)
    print("  * = Published numbers from original papers (cite, not reimplemented)")
    print()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Dev nodes: {len(DEV_NODES)} ({len(DEV_NODES)/207*100:.0f}% of network)")

    proc = np.load("data/processed/metr_la_processed.npz")
    scaler = np.load("data/processed/scaler_stats.npz")
    horizon = proc["Y_train"].shape[1]

    results = {}
    start = time.time()

    # 1. Statistical baselines (instant)
    print("\n--- Statistical Baselines ---")
    stat_results = run_statistical_baselines(
        proc["X_test"], proc["Y_test"],
        scaler["mean"], scaler["std"],
        DEV_NODES, horizon,
    )
    results.update(stat_results)

    # 2. LSTM
    print("\n--- LSTM (direct) ---")
    t0 = time.time()
    results["LSTM"] = run_neural_model(
        "LSTM", LSTMForecaster, proc, scaler, DEV_NODES, device
    )
    print(f"  LSTM total time: {(time.time()-t0)/60:.1f} min")

    # 3. GRU (direct)
    print("\n--- GRU (direct) ---")
    t0 = time.time()
    results["GRU"] = run_neural_model(
        "GRU", GRUForecaster, proc, scaler, DEV_NODES, device
    )
    print(f"  GRU total time: {(time.time()-t0)/60:.1f} min")

    # 4. Seq2Seq GRU
    print("\n--- Seq2Seq GRU ---")
    t0 = time.time()
    results["Seq2Seq GRU"] = run_neural_model(
        "Seq2Seq GRU", Seq2SeqGRU, proc, scaler, DEV_NODES, device,
        seq2seq=True,
    )
    print(f"  Seq2Seq total time: {(time.time()-t0)/60:.1f} min")

    total_time = (time.time() - start) / 60
    print(f"\nTotal comparison time: {total_time:.1f} min")

    # Print comparison table
    print_results_table(results)

    # Save results
    save_path = "data/models_local/baseline_comparison.npz"
    np.savez(
        save_path,
        **{f"{k}_mae": v["mae"] for k, v in results.items()},
        **{f"{k}_rmse": v["rmse"] for k, v in results.items()},
        **{f"{k}_h60": v["h60_mae"] for k, v in results.items()},
    )
    print(f"Results saved to {save_path}")


if __name__ == "__main__":
    main()
