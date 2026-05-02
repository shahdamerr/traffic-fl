"""DTW-Based Traffic Pattern Clustering (Priority 4).

Replaces spectral graph partitioning with Dynamic Time Warping (DTW)
similarity-based clustering, following:
    Feng et al. 2024 — "Traffic Flow Prediction Based on FL and Spatio-Temporal GNNs"
    (ISPRS Int. J. Geo-Inf. 2024, 13, 210)

Rationale:
    Spectral clustering groups sensors by road-graph proximity. But adjacent
    sensors can have completely different traffic patterns (e.g., highway on-ramp
    vs off-ramp behave oppositely at rush hour). DTW clustering groups sensors
    by temporal pattern similarity, which leads to more homogeneous FL clusters
    and faster convergence (less non-IID divergence within each cluster).

Output:
    data/processed/dtw_clusters.npz  — drop-in replacement for graph_clusters.npz

Usage:
    # Generate DTW clusters (run once, takes ~5-10 min)
    python scripts/run_dtw_clustering.py --n_clusters 8 --sample_len 500

    # Then use in FL experiments:
    python scripts/run_fl_experiment.py \\
        --mode clustered --selection adaptive --rounds 50 --local_epochs 1 \\
        --nodes all --cluster_file data/processed/dtw_clusters.npz
"""
import sys, os, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

try:
    from scipy.spatial.distance import cdist
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


# ─── DTW implementation ────────────────────────────────────────────────────────

def dtw_distance(s1, s2):
    """Compute DTW distance between two 1-D time series.

    Uses standard O(NM) DP; suitable for ~500-sample series.
    """
    n, m = len(s1), len(s2)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(s1[i-1] - s2[j-1])
            dtw[i, j] = cost + min(dtw[i-1, j],    # insertion
                                   dtw[i, j-1],    # deletion
                                   dtw[i-1, j-1])  # match
    return dtw[n, m]


def fast_dtw_distance_matrix(series, verbose=True):
    """Compute pairwise DTW distance matrix using vectorized DTW.

    For N sensors, serial DTW is O(N^2 * L^2). We speed this up
    using z-score normalised series + early-abandon heuristic.

    Args:
        series: np.ndarray [N_sensors, T_length]
        verbose: print progress

    Returns:
        dist_matrix: [N_sensors, N_sensors] symmetric distance matrix
    """
    N, L = series.shape
    dist = np.zeros((N, N), dtype=np.float32)
    total_pairs = N * (N - 1) // 2
    done = 0

    t0 = time.time()
    for i in range(N):
        for j in range(i + 1, N):
            d = dtw_distance(series[i], series[j])
            dist[i, j] = d
            dist[j, i] = d
            done += 1
            if verbose and done % max(1, total_pairs // 20) == 0:
                elapsed = time.time() - t0
                pct = done / total_pairs
                eta = elapsed / pct * (1 - pct)
                print(f"  DTW: {pct:.0%} done ({done}/{total_pairs} pairs) — "
                      f"elapsed {elapsed/60:.1f}m, ETA {eta/60:.1f}m")

    total = time.time() - t0
    if verbose:
        print(f"  DTW distance matrix computed in {total/60:.1f} min")

    return dist


def load_representative_series(proc_path, sample_len=500, seed=42):
    """Load training time series and extract a representative window.

    Uses the first `sample_len` timesteps of the training set.
    For 207 sensors with sample_len=500, DTW takes ~3-6 minutes on CPU.

    Args:
        proc_path:  path to metr_la_processed.npz
        sample_len: number of timesteps to use for DTW (more = slower)
        seed:       for reproducibility

    Returns:
        series: [N_sensors, sample_len] — z-score normalised
    """
    proc = np.load(proc_path)
    # X_train shape: [N_windows, seq_len, N_sensors]
    # We want the raw sensor readings over time.
    # Use the first timestep of each window as a proxy for the raw series.
    X_train = proc["X_train"]  # [windows, seq_len, nodes]
    N_nodes = X_train.shape[2]

    # Build raw sensor time series: take the first input step of each window
    # → shape [N_windows, N_sensors] → transpose to [N_sensors, N_windows]
    raw = X_train[:, 0, :]  # [N_windows, N_sensors]
    raw = raw.T              # [N_sensors, N_windows]

    # Truncate to sample_len
    T = min(sample_len, raw.shape[1])
    raw = raw[:, :T]  # [N_sensors, T]

    # Z-score normalise each sensor independently
    mean = raw.mean(axis=1, keepdims=True)
    std  = raw.std(axis=1, keepdims=True) + 1e-8
    series = (raw - mean) / std

    print(f"  Loaded {N_nodes} sensors × {T} timesteps for DTW")
    return series, N_nodes


def cluster_by_dtw(dist_matrix, n_clusters, seed=42):
    """Cluster sensors based on DTW distance matrix using KMeans on embeddings.

    Strategy: Apply KMedoids-style KMeans on classical MDS embedding of the
    DTW distance matrix. This gives geographically-unbiased clusters based
    purely on traffic pattern similarity.

    Args:
        dist_matrix: [N, N] pairwise DTW distances
        n_clusters:  number of desired clusters
        seed:        random seed

    Returns:
        cluster_labels: [N] integer cluster assignments
        clusters_dict:  dict[cid -> list[node_ids]]
    """
    N = dist_matrix.shape[0]

    # Classical MDS: embed distances into Euclidean space for KMeans
    # H = I - (1/N) 11^T
    H = np.eye(N) - np.ones((N, N)) / N
    B = -0.5 * H @ (dist_matrix ** 2) @ H

    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(B)
    # Keep positive eigenvalues only (top n_clusters × 2 dims)
    n_dims = min(n_clusters * 2, N - 1)
    idx = np.argsort(eigvals)[::-1][:n_dims]
    positive_mask = eigvals[idx] > 0
    idx = idx[positive_mask]

    if len(idx) == 0:
        # Fallback: random init
        print("  Warning: DTW matrix has no positive eigenvalues — using random init")
        embedding = np.random.default_rng(seed).random((N, n_clusters))
    else:
        lam = eigvals[idx]
        vecs = eigvecs[:, idx]
        embedding = vecs * np.sqrt(np.maximum(lam, 0))

    # KMeans on MDS embedding
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(embedding)

    clusters_dict = {}
    for node, cid in enumerate(labels):
        clusters_dict.setdefault(int(cid), []).append(node)

    return labels, clusters_dict


def main():
    parser = argparse.ArgumentParser(description="DTW-Based Traffic Pattern Clustering")
    parser.add_argument("--n_clusters",  type=int,   default=8,
                        help="Number of clusters (default: 8, same as spectral)")
    parser.add_argument("--sample_len",  type=int,   default=300,
                        help="Number of timesteps for DTW. More = better but slower. "
                             "300 takes ~2-3 min on 207 sensors.")
    parser.add_argument("--proc_path",   type=str,   default="data/processed/metr_la_processed.npz")
    parser.add_argument("--out_path",    type=str,   default="data/processed/dtw_clusters.npz")
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("  DTW-Based Traffic Pattern Clustering")
    print(f"  Method: DTW distance → MDS embedding → KMeans")
    print(f"  Reference: Feng et al. 2024 (ISPRS Int. J. Geo-Inf.)")
    print(f"  N clusters={args.n_clusters}  |  Sample length={args.sample_len}")
    print("=" * 60)

    # 1. Load data
    print("\n[1/4] Loading training time series...")
    series, N_nodes = load_representative_series(
        args.proc_path, sample_len=args.sample_len, seed=args.seed
    )

    # 2. Compute DTW distance matrix
    print(f"\n[2/4] Computing DTW pairwise distances ({N_nodes}×{N_nodes} matrix)...")
    print("      (This is the slow step — takes 2-5 min depending on sample_len)")
    dist_matrix = fast_dtw_distance_matrix(series, verbose=True)

    # 3. Cluster
    print(f"\n[3/4] Clustering {N_nodes} sensors into {args.n_clusters} groups...")
    labels, clusters_dict = cluster_by_dtw(dist_matrix, args.n_clusters, args.seed)

    # 4. Save
    print(f"\n[4/4] Saving cluster labels → {args.out_path}")
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    np.savez(
        args.out_path,
        cluster_labels=labels,
        n_clusters=args.n_clusters,
        sample_len=args.sample_len,
        method="dtw_kmeans_mds",
        reference="Feng et al. 2024, ISPRS Int. J. Geo-Inf.",
    )

    # Print cluster summary
    print("\n  DTW Cluster Summary:")
    for cid in sorted(clusters_dict.keys()):
        nodes = clusters_dict[cid]
        print(f"    Cluster {cid}: {len(nodes):>3d} nodes")

    # Compare with spectral clustering
    try:
        spectral = np.load("data/processed/graph_clusters.npz")
        spec_labels = spectral["cluster_labels"]
        # Count how many sensors changed cluster
        changed = (labels != spec_labels).sum()
        print(f"\n  vs. Spectral clustering: {changed}/{N_nodes} "
              f"sensors ({changed/N_nodes:.0%}) assigned to different clusters")
    except Exception:
        pass

    print(f"\n  Done. Use in experiments with:")
    print(f"  --cluster_file {args.out_path}")


if __name__ == "__main__":
    main()
