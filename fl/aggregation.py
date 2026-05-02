"""FedAvg aggregation utilities.

Pure functions for model weight manipulation and communication tracking.

Aggregation strategies:
    fedavg()                 — standard sample-weighted average (McMahan et al. 2017)
    quality_weighted_fedavg() — sample × loss-quality weighting (Dai et al. 2024, Paper 1)
                               Nodes that achieve lower training loss under equal budget
                               contribute more to the cluster model.
"""
import copy
import math
import torch


def fedavg(state_dicts, weights=None):
    """Weighted average of model state_dicts (FedAvg).

    Standard sample-count weighted aggregation from McMahan et al. 2017.

    Args:
        state_dicts: list of OrderedDicts (model.state_dict())
        weights: list of floats summing to 1. If None, uses uniform.

    Returns:
        Averaged state_dict.
    """
    if len(state_dicts) == 0:
        raise ValueError("Need at least one state_dict to aggregate.")

    if weights is None:
        weights = [1.0 / len(state_dicts)] * len(state_dicts)

    assert abs(sum(weights) - 1.0) < 1e-6, f"Weights must sum to 1, got {sum(weights)}"

    avg_sd = copy.deepcopy(state_dicts[0])
    for key in avg_sd:
        avg_sd[key] = torch.zeros_like(avg_sd[key], dtype=torch.float32)
        for sd, w in zip(state_dicts, weights):
            avg_sd[key] += sd[key].float() * w

    return avg_sd


def quality_weighted_fedavg(state_dicts, n_samples_list, losses, temperature=0.5):
    """Quality-weighted FedAvg aggregation.

    Implements the reputation/contribution-score-based weighting from:
        Dai et al. 2024 — "Personalized FL based on Client Reputation"
        Feng et al. 2024 — "Multi-factor weighted aggregation"

    Weight for client i:
        base_i   = n_samples_i / total_samples      (data quantity)
        quality_i = exp(-loss_i / temperature)       (training quality)
        w_i      = base_i * quality_i / Z            (normalized)

    Clients with lower training loss (better convergence) contribute more.
    Temperature T controls how strongly quality is emphasized:
        T → ∞: degrades to plain FedAvg
        T → 0: winner-takes-all (only best client contributes)
        T = 0.5: recommended balance

    Args:
        state_dicts:   list of model state_dicts from each client
        n_samples_list: list of int — training samples per client
        losses:        list of float — avg training loss per client this round
        temperature:   float — quality emphasis (default 0.5)

    Returns:
        quality-weighted averaged state_dict
    """
    if len(state_dicts) == 0:
        raise ValueError("Need at least one state_dict to aggregate.")

    total_samples = sum(n_samples_list)
    # Composite weights: sample-proportion × quality-score
    raw_weights = [
        (n / total_samples) * math.exp(-l / temperature)
        for n, l in zip(n_samples_list, losses)
    ]
    Z = sum(raw_weights)
    if Z < 1e-9:
        # Degenerate case: fall back to uniform
        raw_weights = [1.0 / len(state_dicts)] * len(state_dicts)
        Z = 1.0
    norm_weights = [w / Z for w in raw_weights]

    return fedavg(state_dicts, norm_weights)


def count_bytes(state_dict, top_k_fraction=1.0):
    """Count total bytes transmitted for a state_dict.

    Args:
        state_dict:      model state dict
        top_k_fraction:  fraction of params transmitted (1.0 = all, 0.1 = 10%)
                         Models Top-K gradient sparsification (Gao et al. 2024)
    """
    total = sum(v.numel() for v in state_dict.values())
    return int(total * top_k_fraction) * 4  # float32 = 4 bytes


def count_parameters(state_dict):
    """Count total number of parameters in a state_dict."""
    return sum(v.numel() for v in state_dict.values())


def sparsify_delta(initial_sd, final_sd, top_k_fraction=0.1):
    """Compute Top-K sparsified weight delta (gradient in weight space).

    Implements Top-K sparsification from:
        Gao et al. 2024 — FedCE: 80% bandwidth reduction at 27% accuracy gain

    Instead of transmitting the full model weights, transmits only the
    top-K% of weight changes (by absolute value). This directly reduces
    upload communication cost by (1 - K) × 100%.

    Args:
        initial_sd:      state_dict before local training (cluster model)
        final_sd:        state_dict after local training
        top_k_fraction:  fraction of params to keep (e.g., 0.1 = top 10%)

    Returns:
        sparse_delta: dict[key → sparse tensor]  (zeros for pruned entries)
        actual_fraction: actual fraction transmitted (may differ slightly)
    """
    sparse_delta = {}
    all_vals = []

    # 1. Compute per-layer deltas (always on CPU to avoid device mismatch)
    deltas = {}
    for k in final_sd:
        d = (final_sd[k].float().cpu() - initial_sd[k].float().cpu())
        deltas[k] = d
        if d.is_floating_point():
            all_vals.append(d.abs().flatten())

    if not all_vals:
        return deltas, 1.0

    # 2. Find global threshold across ALL layers combined
    all_flat = torch.cat(all_vals)
    k = max(1, int(top_k_fraction * all_flat.numel()))
    threshold = all_flat.topk(k).values.min()

    # 3. Apply mask — zero out sub-threshold entries
    for key, d in deltas.items():
        if d.is_floating_point():
            mask = d.abs() >= threshold
            sparse_delta[key] = d * mask
        else:
            sparse_delta[key] = d  # int/bool layers: always send

    actual = k / all_flat.numel()
    return sparse_delta, actual
