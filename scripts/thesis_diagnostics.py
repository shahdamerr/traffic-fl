"""
Non-IID Statistical Proof + Autocorrelation Analysis + Communication Estimates
===============================================================================
This script produces all three Phase 2 diagnostic analyses for the thesis:

1. Non-IID statistical proof (per-sensor distributions, within/between DTW)
2. Autocorrelation analysis (ACF/PACF for representative sensors)
3. Communication estimates (raw data size, DTW init overhead)

Outputs saved to: figures/ directory (thesis-ready)

Usage:
    python scripts/thesis_diagnostics.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from statsmodels.tsa.stattools import acf, pacf

# Project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ── Paths ────────────────────────────────────────────────────────────────────
RAW_H5   = "data/raw/metr-la.h5"
PROC_NPZ = "data/processed/metr_la_processed.npz"
# Use graph_clusters.npz — this is the cluster file used in ALL thesis experiments
# (Runs A through G). dtw_clusters.npz exists but was NOT used in the final runs.
CLUSTER_NPZ  = "data/processed/graph_clusters.npz"
SCALER   = "data/processed/scaler_stats.npz"
FIG_DIR  = "figures"

os.makedirs(FIG_DIR, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def load_raw_data():
    """Load raw METR-LA data (before normalization) with zero masking."""
    print("[1/6] Loading raw METR-LA data...")
    df = pd.read_hdf(RAW_H5, key="df")
    data = df.values.astype(float)  # (T, 207)

    # Mask zeros as NaN (sensor failures)
    data[data == 0.0] = np.nan

    # Per-sensor mean imputation (same as prepare_data.py)
    col_means = np.nanmean(data, axis=0)
    nan_rows, nan_cols = np.where(np.isnan(data))
    data[nan_rows, nan_cols] = col_means[nan_cols]

    print(f"    Raw data shape: {data.shape}")
    print(f"    Sensors: {data.shape[1]}, Timesteps: {data.shape[0]}")
    print(f"    Time range: {df.index[0]} to {df.index[-1]}")
    return data, df.index


def load_clusters():
    """Load cluster labels used in the actual thesis experiments."""
    clust = np.load(CLUSTER_NPZ)
    labels = clust["cluster_labels"]
    # graph_clusters.npz uses 'num_clusters', dtw uses 'n_clusters'
    if "n_clusters" in clust:
        n_clusters = int(clust["n_clusters"])
    elif "num_clusters" in clust:
        n_clusters = int(clust["num_clusters"])
    else:
        n_clusters = len(np.unique(labels))
    print(f"    Loaded {n_clusters} clusters from {CLUSTER_NPZ}")
    return labels, n_clusters


# =============================================================================
# ANALYSIS 1: Non-IID Statistical Proof
# =============================================================================

def noniid_analysis(data, cluster_labels, n_clusters):
    """Generate non-IID evidence: per-sensor stats, distributions, DTW ratios."""
    print("\n" + "="*60)
    print("[2/6] NON-IID STATISTICAL PROOF")
    print("="*60)

    N_sensors = data.shape[1]

    # ── Per-sensor statistics ───────────────────────────────────────────────
    stats = {}
    for i in range(N_sensors):
        s = data[:, i]
        stats[i] = {
            'mean': np.mean(s),
            'std': np.std(s),
            'cv': np.std(s) / (np.mean(s) + 1e-8),
            'median': np.median(s),
            'p5': np.percentile(s, 5),
            'p95': np.percentile(s, 95),
            'cluster': cluster_labels[i],
        }

    means = np.array([stats[i]['mean'] for i in range(N_sensors)])
    stds  = np.array([stats[i]['std'] for i in range(N_sensors)])
    cvs   = np.array([stats[i]['cv'] for i in range(N_sensors)])

    print(f"\n  Per-sensor mean speed:")
    print(f"    Range: {means.min():.2f} to {means.max():.2f} mph")
    print(f"    Mean ± Std: {means.mean():.2f} ± {means.std():.2f} mph")
    print(f"  Per-sensor std:")
    print(f"    Range: {stds.min():.2f} to {stds.max():.2f} mph")
    print(f"  Per-sensor coefficient of variation:")
    print(f"    Range: {cvs.min():.3f} to {cvs.max():.3f}")

    # ── Figure 1: Histograms of per-sensor mean and std ─────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].hist(means, bins=25, edgecolor='black', alpha=0.8, color='#2196F3')
    axes[0].set_xlabel('Mean Speed (mph)')
    axes[0].set_ylabel('Number of Sensors')
    axes[0].set_title('Distribution of Per-Sensor Mean Speeds')
    axes[0].axvline(means.mean(), color='red', linestyle='--', label=f'Overall mean = {means.mean():.1f}')
    axes[0].legend()

    axes[1].hist(stds, bins=25, edgecolor='black', alpha=0.8, color='#FF9800')
    axes[1].set_xlabel('Standard Deviation (mph)')
    axes[1].set_ylabel('Number of Sensors')
    axes[1].set_title('Distribution of Per-Sensor Speed Variability')
    axes[1].axvline(stds.mean(), color='red', linestyle='--', label=f'Overall mean = {stds.mean():.1f}')
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'noniid_sensor_distributions.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")

    # ── Figure 2: Boxplots by cluster ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    cluster_data = []
    cluster_labels_sorted = sorted(set(cluster_labels))
    for cid in cluster_labels_sorted:
        sensor_ids = np.where(cluster_labels == cid)[0]
        # Use mean speed of each sensor in this cluster
        cluster_means = means[sensor_ids]
        cluster_data.append(cluster_means)

    bp = ax.boxplot(cluster_data, labels=[f'C{c}\n(n={len(d)})' for c, d in zip(cluster_labels_sorted, cluster_data)],
                    patch_artist=True, widths=0.6)

    colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_xlabel('DTW Cluster')
    ax.set_ylabel('Mean Speed (mph)')
    ax.set_title('Per-Sensor Mean Speed Distribution by DTW Cluster')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'noniid_cluster_boxplots.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")

    # ── Within-cluster vs Between-cluster distance ──────────────────────
    print("\n  Computing within/between cluster distance ratio...")
    T_sample = min(300, data.shape[0])
    # Z-score normalize per sensor
    sample = data[:T_sample, :].T  # (207, T_sample)
    sample_mean = sample.mean(axis=1, keepdims=True)
    sample_std  = sample.std(axis=1, keepdims=True) + 1e-8
    sample_norm = (sample - sample_mean) / sample_std

    # Euclidean distance matrix on normalized series
    from scipy.spatial.distance import pdist, squareform
    print("    Computing Euclidean distance on normalized representative series...")
    euc_dist = squareform(pdist(sample_norm, metric='euclidean'))

    # Compute within vs between using Euclidean distance
    within_dists = []
    between_dists = []
    for i in range(N_sensors):
        for j in range(i+1, N_sensors):
            if cluster_labels[i] == cluster_labels[j]:
                within_dists.append(euc_dist[i, j])
            else:
                between_dists.append(euc_dist[i, j])

    within_mean = np.mean(within_dists)
    between_mean = np.mean(between_dists)
    ratio = within_mean / between_mean

    print(f"\n  ┌──────────────────────────────────────────────────────┐")
    print(f"  │  Temporal-Distance Within/Between Ratio              │")
    print(f"  │  (Euclidean on normalized 300-step representative    │")
    print(f"  │   series, NOT DTW — see wording note below)          │")
    print(f"  ├──────────────────────────────────────────────────────┤")
    print(f"  │  Within-cluster mean distance:  {within_mean:.4f}             │")
    print(f"  │  Between-cluster mean distance: {between_mean:.4f}             │")
    print(f"  │  Ratio (within/between):        {ratio:.4f}              │")
    print(f"  │  Interpretation: {'Clustering reduces heterogeneity ✓' if ratio < 1 else 'Clustering does NOT reduce heterogeneity ✗'}│")
    print(f"  └──────────────────────────────────────────────────────┘")
    print(f"  NOTE: This is Euclidean distance, not DTW. In the thesis,")
    print(f"  write 'normalized temporal-pattern distance' not 'DTW distance'.")

    # ── Permutation test ────────────────────────────────────────────────────
    print("\n  Running permutation test (1000 shuffles)...")
    n_perms = 1000
    rng = np.random.default_rng(42)
    perm_ratios = []
    for _ in range(n_perms):
        shuffled = rng.permutation(cluster_labels)
        w, b = [], []
        for i in range(N_sensors):
            for j in range(i+1, N_sensors):
                if shuffled[i] == shuffled[j]:
                    w.append(euc_dist[i, j])
                else:
                    b.append(euc_dist[i, j])
        perm_ratios.append(np.mean(w) / np.mean(b))

    perm_ratios = np.array(perm_ratios)
    p_value = (perm_ratios <= ratio).sum() / n_perms

    print(f"    DTW clustering ratio: {ratio:.4f}")
    print(f"    Random clustering ratio (mean ± std): {perm_ratios.mean():.4f} ± {perm_ratios.std():.4f}")
    print(f"    p-value: {p_value:.4f} ({'significant' if p_value < 0.05 else 'not significant'})")

    # ── Figure 3: Permutation test histogram ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(perm_ratios, bins=50, edgecolor='black', alpha=0.7, color='#9E9E9E', label='Random clustering')
    ax.axvline(ratio, color='red', linewidth=2, linestyle='--',
               label=f'DTW clustering = {ratio:.4f} (p = {p_value:.4f})')
    ax.set_xlabel('Within/Between Distance Ratio')
    ax.set_ylabel('Count')
    ax.set_title('Permutation Test: DTW Clustering vs. Random Assignment')
    ax.legend()
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'noniid_permutation_test.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")

    # ── Print per-cluster summary table ──────────────────────────────────────
    print("\n  Per-Cluster Summary Table:")
    print(f"  {'Cluster':>8} {'Sensors':>8} {'Mean Speed':>12} {'Std Speed':>10} {'CV':>8} {'P5':>8} {'P95':>8}")
    print("  " + "-"*68)
    for cid in cluster_labels_sorted:
        sensor_ids = np.where(cluster_labels == cid)[0]
        c_means = means[sensor_ids]
        c_stds  = stds[sensor_ids]
        c_cvs   = cvs[sensor_ids]
        c_p5    = np.array([stats[i]['p5'] for i in sensor_ids])
        c_p95   = np.array([stats[i]['p95'] for i in sensor_ids])
        print(f"  {cid:>8d} {len(sensor_ids):>8d} {c_means.mean():>12.2f} {c_stds.mean():>10.2f} {c_cvs.mean():>8.3f} {c_p5.mean():>8.2f} {c_p95.mean():>8.2f}")

    return ratio, p_value


# =============================================================================
# ANALYSIS 2: Autocorrelation Analysis
# =============================================================================

def autocorrelation_analysis(data, cluster_labels, n_clusters):
    """ACF/PACF for representative sensors, one per cluster."""
    print("\n" + "="*60)
    print("[3/6] AUTOCORRELATION ANALYSIS")
    print("="*60)

    N_sensors = data.shape[1]
    lags_of_interest = {
        1:   '5 min',
        3:   '15 min',
        6:   '30 min',
        12:  '1 hour',
        288: '1 day',
    }
    max_lag = 300  # Compute up to 300 lags

    # Pick one representative sensor per cluster (median-speed sensor)
    cluster_labels_sorted = sorted(set(cluster_labels))
    rep_sensors = {}
    means = np.array([np.mean(data[:, i]) for i in range(N_sensors)])

    for cid in cluster_labels_sorted:
        sensor_ids = np.where(cluster_labels == cid)[0]
        c_means = means[sensor_ids]
        median_idx = sensor_ids[np.argmin(np.abs(c_means - np.median(c_means)))]
        rep_sensors[cid] = median_idx

    print(f"  Representative sensors (1 per cluster, median speed):")
    for cid, sid in rep_sensors.items():
        print(f"    Cluster {cid}: Sensor {sid} (mean = {means[sid]:.2f} mph)")

    # ── Compute ACF for all sensors ─────────────────────────────────────────
    print("\n  Computing ACF at key lags for all 207 sensors...")
    all_acfs = {lag: [] for lag in lags_of_interest}

    for i in range(N_sensors):
        a = acf(data[:, i], nlags=max_lag, fft=True)
        for lag in lags_of_interest:
            if lag < len(a):
                all_acfs[lag].append(a[lag])
            else:
                all_acfs[lag].append(np.nan)

    # ── Print ACF summary table ─────────────────────────────────────────────
    print(f"\n  ACF Summary across all 207 sensors:")
    print(f"  {'Lag':>6} {'Time':>8} {'Mean ACF':>10} {'Std ACF':>10} {'Min':>8} {'Max':>8}")
    print("  " + "-"*56)
    for lag, name in lags_of_interest.items():
        vals = np.array(all_acfs[lag])
        vals = vals[~np.isnan(vals)]
        print(f"  {lag:>6d} {name:>8} {vals.mean():>10.4f} {vals.std():>10.4f} {vals.min():>8.4f} {vals.max():>8.4f}")

    # ── Per-cluster ACF table ───────────────────────────────────────────────
    print(f"\n  ACF by DTW Cluster (lag=12 = 1 hour):")
    print(f"  {'Cluster':>8} {'Sensors':>8} {'ACF lag=1':>10} {'ACF lag=12':>11} {'ACF lag=288':>12}")
    print("  " + "-"*54)
    for cid in cluster_labels_sorted:
        sensor_ids = np.where(cluster_labels == cid)[0]
        acf_1   = np.mean([all_acfs[1][i] for i in sensor_ids])
        acf_12  = np.mean([all_acfs[12][i] for i in sensor_ids])
        acf_288 = np.mean([all_acfs[288][i] for i in sensor_ids])
        print(f"  {cid:>8d} {len(sensor_ids):>8d} {acf_1:>10.4f} {acf_12:>11.4f} {acf_288:>12.4f}")

    # ── Figure 4: ACF/PACF for representative sensors ───────────────────────
    n_reps = len(rep_sensors)
    fig, axes = plt.subplots(n_reps, 2, figsize=(14, 2.5 * n_reps))
    if n_reps == 1:
        axes = axes.reshape(1, -1)

    for idx, (cid, sid) in enumerate(sorted(rep_sensors.items())):
        a = acf(data[:, sid], nlags=max_lag, fft=True)
        p = pacf(data[:, sid], nlags=min(50, max_lag), method='ywm')

        # ACF plot
        axes[idx, 0].bar(range(len(a)), a, width=1.0, color='#2196F3', alpha=0.7)
        axes[idx, 0].set_ylabel(f'C{cid} (S{sid})')
        if idx == 0:
            axes[idx, 0].set_title('Autocorrelation Function (ACF)')
        if idx == n_reps - 1:
            axes[idx, 0].set_xlabel('Lag (5-min intervals)')
        # Mark key lags
        for lag in [12, 288]:
            if lag < len(a):
                axes[idx, 0].axvline(lag, color='red', alpha=0.4, linestyle='--', linewidth=0.8)

        # PACF plot
        axes[idx, 1].bar(range(len(p)), p, width=1.0, color='#FF9800', alpha=0.7)
        if idx == 0:
            axes[idx, 1].set_title('Partial Autocorrelation Function (PACF)')
        if idx == n_reps - 1:
            axes[idx, 1].set_xlabel('Lag (5-min intervals)')

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'acf_pacf_by_cluster.png')
    plt.savefig(path)
    plt.close()
    print(f"\n  Saved: {path}")

    # ── Figure 5: Heatmap of ACF at lag=12 by cluster ────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))
    acf12_vals = np.array(all_acfs[12])
    cluster_acf12 = []
    for cid in cluster_labels_sorted:
        sensor_ids = np.where(cluster_labels == cid)[0]
        cluster_acf12.append(acf12_vals[sensor_ids].mean())
    
    bars = ax.bar([f'C{c}' for c in cluster_labels_sorted], cluster_acf12,
                  color=plt.cm.Set3(np.linspace(0, 1, n_clusters)), edgecolor='black')
    ax.set_ylabel('Mean ACF at lag=12 (1 hour)')
    ax.set_xlabel('DTW Cluster')
    ax.set_title('Temporal Memory (1-hour ACF) by DTW Cluster')
    ax.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, cluster_acf12):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'acf_lag12_by_cluster.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# ANALYSIS 3: Communication Estimates
# =============================================================================

def communication_estimates(data):
    """Compute raw data size vs FL communication."""
    print("\n" + "="*60)
    print("[4/6] COMMUNICATION ESTIMATES")
    print("="*60)

    T, N = data.shape
    raw_bytes = T * N * 4  # float32
    raw_mb = raw_bytes / (1000**2)  # decimal MB (SI), not MiB

    # DTW init overhead
    dtw_init_bytes = N * 300 * 4  # 207 sensors x 300 timesteps x 4 bytes
    dtw_init_mb = dtw_init_bytes / (1000**2)

    # Model size
    model_params = 316161
    model_bytes = model_params * 4
    model_mb = model_bytes / (1000**2)  # decimal MB

    # FL communication (from thesis results)
    run_f_gb = 21.5
    run_g_gb = 25.8
    cmba_gb  = 36.4

    print(f"\n  ┌───────────────────────────────────────────────────────────┐")
    print(f"  │  RAW DATA vs FL COMMUNICATION                            │")
    print(f"  ├───────────────────────────────────────────────────────────┤")
    print(f"  │  METR-LA speed matrix ({T} × {N} × 4 bytes)        │")
    print(f"  │    Raw data size:              {raw_mb:>10.2f} MB              │")
    print(f"  │    DTW init overhead:          {dtw_init_mb:>10.4f} MB              │")
    print(f"  │    Model size (per transfer):  {model_mb:>10.2f} MB              │")
    print(f"  ├───────────────────────────────────────────────────────────┤")
    print(f"  │  FL MODEL COMMUNICATION (C_FL)                           │")
    print(f"  │    Run F (50 rounds, 75%):     {run_f_gb:>10.1f} GB              │")
    print(f"  │    Run G (60 rounds, 75%):     {run_g_gb:>10.1f} GB              │")
    print(f"  │    CMBA-FL (~109 rounds):      {cmba_gb:>10.1f} GB              │")
    print(f"  ├───────────────────────────────────────────────────────────┤")
    print(f"  │  FL/Raw ratio:                                           │")
    print(f"  │    Run F: {run_f_gb*1024/raw_mb:>8.0f}× larger than raw data upload    │")
    print(f"  │    Run G: {run_g_gb*1024/raw_mb:>8.0f}× larger than raw data upload    │")
    print(f"  └───────────────────────────────────────────────────────────┘")

    print(f"\n  Thesis Table (copy-paste for LaTeX):")
    print(f"  ───────────────────────────────────────────────────────────")
    print(f"  Quantity                            | Approx Size | Interpretation")
    print(f"  One-time METR-LA speed matrix (f32) | {raw_mb:.1f} MB     | centralized upload")
    print(f"  DTW initialization overhead         | {dtw_init_mb:.2f} MB    | one-time setup")
    print(f"  Single model transfer               | {model_mb:.2f} MB     | per-client per-round")
    print(f"  Run F FL communication              | {run_f_gb:.1f} GB     | comm-efficient point")
    print(f"  Run G FL communication              | {run_g_gb:.1f} GB     | best-accuracy point")
    print(f"  CMBA-FL FL communication            | {cmba_gb:.1f} GB     | published baseline")

    # Per-sensor storage estimate for RSU
    train_frac = 0.7
    n_windows = int((T * train_frac) - 12 - 12)
    per_sensor_bytes = n_windows * (12 + 12) * 4  # input + output windows, float32
    per_sensor_mb = per_sensor_bytes / (1024**2)
    print(f"\n  Per-sensor local storage at RSU:")
    print(f"    Training windows: {n_windows:,}")
    print(f"    Per-sensor storage: {per_sensor_mb:.2f} MB")
    print(f"    Total across 207 sensors: {per_sensor_mb * N:.1f} MB")


# =============================================================================
# ANALYSIS 4: Non-IID evidence from per-cluster MAE (using existing results)
# =============================================================================

def existing_evidence_summary():
    """Print the indirect non-IID evidence from existing thesis results."""
    print("\n" + "="*60)
    print("[5/6] EXISTING NON-IID EVIDENCE (from thesis results)")
    print("="*60)

    cluster_maes = {
        0: (31, 3.779), 1: (32, 3.835), 2: (8, 2.414), 3: (36, 4.281),
        4: (8, 2.588), 5: (25, 4.161), 6: (39, 2.000), 7: (28, 5.177),
    }

    print(f"\n  {'Cluster':>8} {'Sensors':>8} {'MAE':>8} {'Relative to Best':>18}")
    print("  " + "-"*46)
    best = min(v[1] for v in cluster_maes.values())
    for cid, (n, mae) in sorted(cluster_maes.items()):
        rel = mae / best
        print(f"  {cid:>8d} {n:>8d} {mae:>8.3f} {rel:>18.2f}×")

    print(f"\n  MAE range: {best:.3f} to {max(v[1] for v in cluster_maes.values()):.3f}")
    print(f"  Ratio: {max(v[1] for v in cluster_maes.values()) / best:.1f}×")
    print(f"  → This 2.6× range is indirect evidence of heterogeneous sensor behavior.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("  THESIS DIAGNOSTIC ANALYSES")
    print("  Phase 2: Quick Analysis Scripts")
    print("=" * 60)

    # Load data
    data, timestamps = load_raw_data()
    cluster_labels, n_clusters = load_clusters()

    # Run analyses
    ratio, p_value = noniid_analysis(data, cluster_labels, n_clusters)
    autocorrelation_analysis(data, cluster_labels, n_clusters)
    communication_estimates(data)
    existing_evidence_summary()

    # Final summary
    print("\n" + "=" * 60)
    print("[6/6] SUMMARY")
    print("=" * 60)
    print(f"""
  Non-IID Evidence:
    Within/between distance ratio: {ratio:.4f} (p = {p_value:.4f})
    {'✓ Clustering significantly reduces heterogeneity' if p_value < 0.05 else '✗ Not statistically significant'}

  Figures saved to {FIG_DIR}/:
    - noniid_sensor_distributions.png  (histogram of sensor means/stds)
    - noniid_cluster_boxplots.png      (boxplots by cluster)
    - noniid_permutation_test.png      (permutation test)
    - acf_pacf_by_cluster.png          (ACF/PACF for 8 representative sensors)
    - acf_lag12_by_cluster.png         (1-hour ACF by cluster)

  Copy the printed tables into your thesis LaTeX files.
  The figures go into the figures/ directory for \\includegraphics.
""")


if __name__ == "__main__":
    main()
