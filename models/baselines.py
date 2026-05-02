"""Statistical baselines that require no training."""

import numpy as np


def naive_forecast(X, horizon):
    """Naive (Last Value) baseline.

    Repeats the last observed value for all horizon steps.

    Args:
        X: [N_samples, seq_len] input sequences for one node (normalized)
        horizon: int, number of steps to predict

    Returns:
        preds: [N_samples, horizon]
    """
    last_value = X[:, -1]  # [N]
    return np.tile(last_value[:, np.newaxis], (1, horizon))


def moving_average_forecast(X, horizon, window=12):
    """Moving Average baseline.

    Predicts the average of the last `window` observations for all horizons.

    Args:
        X: [N_samples, seq_len] input sequences for one node (normalized)
        horizon: int
        window: int, number of past steps to average (default=12 = 1 hour)

    Returns:
        preds: [N_samples, horizon]
    """
    avg = np.mean(X[:, -window:], axis=1)  # [N]
    return np.tile(avg[:, np.newaxis], (1, horizon))
