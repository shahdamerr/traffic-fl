"""Run FL experiments on PEMS-BAY dataset.

Thin wrapper around run_fl_experiment.py that overrides data paths
to use the PEMS-BAY preprocessed data instead of METR-LA.

Usage:
  # Step 1: Preprocess
  python scripts/prepare_data_pemsbay.py

  # Step 2: DTW clustering
  python scripts/run_dtw_clustering.py \
    --proc_path data-PemsBay/processed/pemsbay_processed.npz \
    --out_path data-PemsBay/processed/dtw_clusters.npz \
    --n_clusters 8 --sample_len 300

  # Step 3: Run best experiment
  python scripts/run_fl_pemsbay.py \
    --rounds 60 --local_steps 20 --lr 0.0005 \
    --quality_agg --hier_alpha 0.8 --hier_every 5 --hier_mode dtw \
    --selection adaptive --lr_decay "40:0.00025,50:0.0001"
"""
import sys
import os

# Monkey-patch the data paths before importing run_fl_experiment
# This avoids modifying the original script and breaking METR-LA runs

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import argparse

# Override config for 325 sensors
import config
config.ALL_NODES = list(range(325))
# Keep DEV_NODES as-is (won't be used for --nodes all)

def main():
    """Patch data paths and delegate to run_fl_experiment.main()."""
    import scripts.run_fl_experiment as runner

    # Monkey-patch the data loading in main() by overriding np.load paths
    # Save original main and replace data paths
    original_main = runner.main

    # We need to patch argparse defaults and file paths
    # Simplest: just run with modified sys.argv adding proper defaults
    # and patch the hardcoded paths

    # Patch: override the hardcoded data paths in runner.main
    import types

    _orig_np_load = np.load

    PEMSBAY_DATA = "data-PemsBay/processed/pemsbay_processed.npz"
    PEMSBAY_SCALER = "data-PemsBay/processed/scaler_stats.npz"
    PEMSBAY_CLUSTERS_DEFAULT = "data-PemsBay/processed/dtw_clusters.npz"

    def patched_load(path, *args, **kwargs):
        path_str = str(path)
        if "metr_la_processed" in path_str:
            print(f"[PEMS-BAY] Redirecting data load: {path_str} → {PEMSBAY_DATA}")
            return _orig_np_load(PEMSBAY_DATA, *args, **kwargs)
        elif path_str == "data/processed/scaler_stats.npz":
            print(f"[PEMS-BAY] Redirecting scaler load → {PEMSBAY_SCALER}")
            return _orig_np_load(PEMSBAY_SCALER, *args, **kwargs)
        elif path_str == "data/processed/graph_clusters.npz":
            # Default cluster file → use PEMS-BAY DTW clusters
            print(f"[PEMS-BAY] Redirecting cluster load → {PEMSBAY_CLUSTERS_DEFAULT}")
            return _orig_np_load(PEMSBAY_CLUSTERS_DEFAULT, *args, **kwargs)
        else:
            return _orig_np_load(path, *args, **kwargs)

    # Apply patch
    np.load = patched_load

    # Also need to handle the results directory
    os.makedirs("results-pemsbay", exist_ok=True)

    try:
        original_main()
    finally:
        # Restore
        np.load = _orig_np_load


if __name__ == "__main__":
    main()
