"""Adaptive Client Selection for Federated Learning.

Implements the dynamic network conditions simulation and two client selection
strategies for the thesis ablation study (Rows 4 vs 5 in the comparison table):

    Row 3: Full participation        — all clients in cluster train each round
    Row 4: Random selection          — clients drop out randomly; survivors chosen randomly
    Row 5: Adaptive selection        — clients drop out stochastically; survivors ranked
                                       by reliability score, top-K selected

Reliability Score (composite):
    s_i = p_i * exp(-l_i / L_MAX)
    where p_i = availability probability, l_i = latency (ms), L_MAX = 500 ms

Node stability tiers (assigned ONCE at initialisation, fixed for all rounds):
    - Stable     (60% of nodes): p ~ Uniform(0.85, 0.95), l ~ Uniform(50, 150) ms
    - Moderate   (25% of nodes): p ~ Uniform(0.55, 0.70), l ~ Uniform(150, 350) ms
    - Unreliable (15% of nodes): p ~ Uniform(0.20, 0.40), l ~ Uniform(350, 500) ms

Reference:
    Nishio & Yonetani, "Client Selection for Federated Learning with Heterogeneous
    Resources in Mobile Edge" (ICC 2019): https://arxiv.org/abs/1804.08333
"""
import math
import numpy as np


LATENCY_MAX_MS = 500.0          # normalisation ceiling (eq. to worst-case latency)
MIN_FRACTION    = 0.5           # guarantee at least 50% of cluster participates


# ── Reliability score ──────────────────────────────────────────────────────────

def reliability_score(availability_prob: float, latency_ms: float) -> float:
    """Composite reliability score ∈ (0, 1].

    Higher → client is more reliable and should be preferred.

    Args:
        availability_prob: Bernoulli probability of being available (0–1)
        latency_ms: expected upload/download latency in milliseconds
    """
    return availability_prob * math.exp(-latency_ms / LATENCY_MAX_MS)


# ── Node profile assignment ────────────────────────────────────────────────────

def assign_node_profiles(node_ids: list, seed: int = 42) -> dict:
    """Assign stable / moderate / unreliable profile to each node.

    Profiles are fixed for the duration of an experiment (assigned once).
    Uses a seeded RNG for reproducibility.

    Args:
        node_ids: list of integer node IDs
        seed: random seed

    Returns:
        dict mapping node_id → {"availability_prob": float, "latency_ms": float,
                                 "reliability_score": float, "tier": str}
    """
    rng = np.random.default_rng(seed)
    n = len(node_ids)

    # Tier thresholds
    n_stable     = int(round(n * 0.60))
    n_moderate   = int(round(n * 0.25))
    # n_unreliable = remainder

    indices = rng.permutation(n)
    stable_idx     = indices[:n_stable]
    moderate_idx   = indices[n_stable:n_stable + n_moderate]
    unreliable_idx = indices[n_stable + n_moderate:]

    profiles = {}
    for idx in stable_idx:
        nid = node_ids[idx]
        p = float(rng.uniform(0.85, 0.95))
        l = float(rng.uniform(50.0, 150.0))
        profiles[nid] = {"availability_prob": p, "latency_ms": l,
                          "reliability_score": reliability_score(p, l), "tier": "stable"}

    for idx in moderate_idx:
        nid = node_ids[idx]
        p = float(rng.uniform(0.55, 0.70))
        l = float(rng.uniform(150.0, 350.0))
        profiles[nid] = {"availability_prob": p, "latency_ms": l,
                          "reliability_score": reliability_score(p, l), "tier": "moderate"}

    for idx in unreliable_idx:
        nid = node_ids[idx]
        p = float(rng.uniform(0.20, 0.40))
        l = float(rng.uniform(350.0, 500.0))
        profiles[nid] = {"availability_prob": p, "latency_ms": l,
                          "reliability_score": reliability_score(p, l), "tier": "unreliable"}

    return profiles


# ── Client selectors ───────────────────────────────────────────────────────────

class FullParticipationSelector:
    """Row 3 — All clients participate every round. No dropout."""

    def select(self, clients: dict, cluster_node_ids: list, rng, round_num: int) -> list:
        """Return all node IDs in the cluster."""
        return list(cluster_node_ids)

    def __repr__(self):
        return "FullParticipation"


class RandomDropSelector:
    """Row 4 — Bernoulli dropout; surviving clients chosen uniformly at random.

    This is the *uncontrolled* dynamic condition baseline: nodes drop out but
    we do NOT use reliability information to choose who trains.

    Args:
        min_fraction: minimum fraction of cluster guaranteed to participate
    """

    def __init__(self, min_fraction: float = MIN_FRACTION):
        self.min_fraction = min_fraction

    def select(self, clients: dict, cluster_node_ids: list, rng, round_num: int) -> list:
        """Sample availability then pick randomly among available nodes."""
        available = [
            nid for nid in cluster_node_ids
            if clients[nid].sample_availability(rng)
        ]

        # Enforce minimum participation floor
        min_k = max(1, int(math.ceil(len(cluster_node_ids) * self.min_fraction)))
        if len(available) < min_k:
            # Not enough available → supplement with random unavailable nodes
            unavailable = [n for n in cluster_node_ids if n not in available]
            shortfall = min_k - len(available)
            extra = list(rng.choice(unavailable, size=min(shortfall, len(unavailable)),
                                    replace=False))
            available = available + extra

        # Random ordering — no reliability ranking
        rng.shuffle(available)
        return available

    def __repr__(self):
        return f"RandomDrop(min_frac={self.min_fraction})"


class AdaptiveReliabilitySelector:
    """Row 5 — Bernoulli dropout; surviving clients ranked by reliability score.

    This is the thesis contribution: under the SAME dropout conditions as Row 4,
    we prioritise the most reliable available nodes — leading to better convergence
    stability and lower effective communication cost.

    Args:
        min_fraction: minimum fraction of cluster guaranteed to participate
    """

    def __init__(self, min_fraction: float = MIN_FRACTION):
        self.min_fraction = min_fraction

    def select(self, clients: dict, cluster_node_ids: list, rng, round_num: int) -> list:
        """Sample availability then rank available nodes by reliability score."""
        available = [
            nid for nid in cluster_node_ids
            if clients[nid].sample_availability(rng)
        ]

        # Enforce minimum participation floor
        min_k = max(1, int(math.ceil(len(cluster_node_ids) * self.min_fraction)))
        if len(available) < min_k:
            unavailable = [n for n in cluster_node_ids if n not in available]
            shortfall = min_k - len(available)
            # When forced to add nodes, prefer highest-reliability ones
            extras = sorted(unavailable,
                            key=lambda n: clients[n].reliability_score,
                            reverse=True)
            available = available + extras[:shortfall]

        # Sort by reliability score (descending) — THIS is the adaptive intelligence
        selected = sorted(available,
                          key=lambda n: clients[n].reliability_score,
                          reverse=True)
        return selected

    def __repr__(self):
        return f"AdaptiveReliability(min_frac={self.min_fraction})"


# ── Factory ────────────────────────────────────────────────────────────────────

def build_selector(mode: str, min_fraction: float = MIN_FRACTION):
    """Instantiate the correct selector by name.

    Args:
        mode: one of "full", "random_drop", "adaptive"
        min_fraction: minimum participation floor for dropout modes

    Returns:
        Selector instance
    """
    modes = {
        "full":         FullParticipationSelector,
        "random_drop":  RandomDropSelector,
        "adaptive":     AdaptiveReliabilitySelector,
    }
    if mode not in modes:
        raise ValueError(f"Unknown selection mode '{mode}'. Choose from: {list(modes)}")
    cls = modes[mode]
    if mode == "full":
        return cls()
    return cls(min_fraction=min_fraction)
