"""Justify the choice of K=8 clusters using Silhouette and Elbow analysis.

Computes:
  1. Elbow plot (within-cluster inertia vs K)
  2. Silhouette score plot (cluster quality vs K)
  3. Prints the optimal K from both methods

Uses the pre-computed DTW distance matrix embedded via MDS,
matching the exact pipeline from run_dtw_clustering.py.

Usage:
    python scripts/justify_k_clusters.py
    python scripts/justify_k_clusters.py --k_max 15
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ── Reuse DTW + MDS pipeline from run_dtw_clustering.py ─────────────────────

def load_and_compute_dtw_matrix(proc_path="data/processed/metr_la_processed.npz",
                                 sample_len=300):
    """Load data + compute pairwise DTW distance matrix."""
    from scripts.run_dtw_clustering import load_representative_series, fast_dtw_distance_matrix

    print("[1/3] Loading time series...")
    series, N_nodes = load_representative_series(proc_path, sample_len=sample_len)

    print(f"[2/3] Computing DTW distance matrix ({N_nodes}x{N_nodes})...")
    dist_matrix = fast_dtw_distance_matrix(series, verbose=True)

    return dist_matrix, N_nodes


def mds_embedding(dist_matrix, max_dims=30):
    """Classical MDS embedding of the DTW distance matrix."""
    N = dist_matrix.shape[0]
    H = np.eye(N) - np.ones((N, N)) / N
    B = -0.5 * H @ (dist_matrix ** 2) @ H

    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1][:max_dims]
    positive_mask = eigvals[idx] > 0
    idx = idx[positive_mask]

    lam = eigvals[idx]
    vecs = eigvecs[:, idx]
    embedding = vecs * np.sqrt(np.maximum(lam, 0))

    print(f"  MDS embedding: {N} sensors -> {embedding.shape[1]} dimensions")
    return embedding


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--k_min", type=int, default=2)
    parser.add_argument("--k_max", type=int, default=15)
    parser.add_argument("--sample_len", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    k_range = range(args.k_min, args.k_max + 1)

    # Step 1: Get DTW distance matrix
    dist_matrix, N = load_and_compute_dtw_matrix(sample_len=args.sample_len)

    # Step 2: MDS embedding (use enough dims for the largest K)
    embedding = mds_embedding(dist_matrix, max_dims=args.k_max * 2)

    # Step 3: Run KMeans for each K and collect metrics
    print(f"\n[3/3] Running KMeans for K={args.k_min} to {args.k_max}...")
    inertias = []
    silhouettes = []
    cluster_sizes = {}

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=args.seed, n_init=10)
        labels = km.fit_predict(embedding)
        inertias.append(km.inertia_)
        sil = silhouette_score(embedding, labels)
        silhouettes.append(sil)

        sizes = [int((labels == c).sum()) for c in range(k)]
        cluster_sizes[k] = sizes

        print(f"  K={k:>2d}:  Inertia={km.inertia_:>10.1f}  "
              f"Silhouette={sil:.4f}  "
              f"Sizes={sizes}")

    # Step 4: Find optimal K
    best_sil_k = list(k_range)[np.argmax(silhouettes)]
    print(f"\n  Best Silhouette score: K={best_sil_k} "
          f"(score={max(silhouettes):.4f})")

    # Elbow: find the K with largest second derivative of inertia
    inertias_arr = np.array(inertias)
    if len(inertias_arr) > 2:
        diffs = np.diff(inertias_arr)
        diffs2 = np.diff(diffs)
        elbow_k = list(k_range)[np.argmax(np.abs(diffs2)) + 1]
        print(f"  Elbow method suggests: K={elbow_k}")

    # Step 5: Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Elbow plot
    ax1.plot(list(k_range), inertias, 'b-o', linewidth=2, markersize=6)
    ax1.axvline(x=8, color='red', linestyle='--', alpha=0.7, label='K=8 (chosen)')
    ax1.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax1.set_ylabel('Within-Cluster Inertia', fontsize=12)
    ax1.set_title('Elbow Method', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(list(k_range))

    # Silhouette plot
    ax2.plot(list(k_range), silhouettes, 'g-o', linewidth=2, markersize=6)
    ax2.axvline(x=8, color='red', linestyle='--', alpha=0.7, label='K=8 (chosen)')
    ax2.axvline(x=best_sil_k, color='blue', linestyle=':', alpha=0.7,
                label=f'K={best_sil_k} (best silhouette)')
    ax2.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax2.set_ylabel('Silhouette Score', fontsize=12)
    ax2.set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(list(k_range))

    fig.suptitle('Justification for K=8 Clusters (DTW + MDS + KMeans)',
                 fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()

    out = "figures/k_selection_analysis.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"\n  Saved {out}")
    plt.close()

    # Step 6: Print cluster size table for K=8
    print(f"\n  K=8 cluster sizes: {cluster_sizes[8]}")
    print(f"  Min cluster: {min(cluster_sizes[8])} sensors")
    print(f"  Max cluster: {max(cluster_sizes[8])} sensors")


if __name__ == "__main__":
    main()
