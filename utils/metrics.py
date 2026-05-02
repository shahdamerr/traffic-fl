import numpy as np


def mae(y_true, y_pred):
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    """Root Mean Square Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mape(y_true, y_pred, threshold=5.0):
    """Mean Absolute Percentage Error (%).

    Follows the standard METR-LA convention: zero and near-zero speed
    readings are masked out because percentage error is undefined there.
    The DCRNN paper and most traffic benchmarks use this masking approach.

    Args:
        y_true: ground truth (denormalized)
        y_pred: predictions  (denormalized)
        threshold: minimum absolute value to include in MAPE calculation.
                   Values below this are masked. Default 5.0 mph matches
                   the convention in Li et al. (ICLR 2018).
    """
    mask = np.abs(y_true) > threshold
    if mask.sum() == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_per_horizon(y_true, y_pred, horizons=None):
    """Compute MAE, RMSE, MAPE at specific forecast horizons.

    Args:
        y_true: [N, H] ground truth (denormalized)
        y_pred: [N, H] predictions  (denormalized)
        horizons: dict mapping label -> 0-based horizon index
                  Defaults to {15min: 2, 30min: 5, 60min: 11}

    Returns:
        dict of {label: {mae, rmse, mape}}
    """
    if horizons is None:
        horizons = {"15min": 2, "30min": 5, "60min": 11}

    results = {}
    for label, h_idx in horizons.items():
        if h_idx >= y_true.shape[1]:
            continue
        yt = y_true[:, h_idx]
        yp = y_pred[:, h_idx]
        results[label] = {
            "mae": mae(yt, yp),
            "rmse": rmse(yt, yp),
            "mape": mape(yt, yp),
        }
    return results
