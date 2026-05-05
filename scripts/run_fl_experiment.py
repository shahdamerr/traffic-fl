"""Run Federated Learning experiments.

Supports Clustered FL and Global FedAvg, with three client selection modes
for the ablation study (Rows 3, 4, 5 in the thesis comparison table).

Usage:
    # Row 3 — Clustered FL, full participation (no dropout)
    python scripts/run_fl_experiment.py --mode clustered --selection full --rounds 30 --local_epochs 1 --nodes dev

    # Row 4 — Clustered FL, random dropout (no adaptive selection)
    python scripts/run_fl_experiment.py --mode clustered --selection random_drop --rounds 30 --local_epochs 1 --nodes dev

    # Row 5 — Clustered FL, adaptive reliability-based selection
    python scripts/run_fl_experiment.py --mode clustered --selection adaptive --rounds 30 --local_epochs 1 --nodes dev

    # Global FedAvg baseline
    python scripts/run_fl_experiment.py --mode global --selection full --rounds 30 --local_epochs 1 --nodes dev

    # Smoke test (~2 min)
    python scripts/run_fl_experiment.py --mode clustered --selection adaptive --rounds 5 --local_epochs 1 --nodes smoke
"""
import sys
import os
import argparse
from math import radians, sin, cos, sqrt, atan2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import DEV_NODES, ALL_NODES, BATCH_TRAIN, BATCH_EVAL, MAX_GRAD_NORM, LR
from models.factory import get_model_kwargs
from fl.dataset import NodeTrafficDataset
from fl.client import FLClient
from fl.fl_trainer import FLTrainer
from fl.client_selector import build_selector, assign_node_profiles


def build_clients(proc, scaler, node_list, cluster_labels, model_kwargs, profiles=None):
    """Create FLClient instances with optional network profiles."""
    X_train, Y_train = proc["X_train"], proc["Y_train"]
    X_val, Y_val     = proc["X_val"],   proc["Y_val"]
    X_test, Y_test   = proc["X_test"],  proc["Y_test"]
    mean_all, std_all = scaler["mean"], scaler["std"]

    clients = {}
    for nid in node_list:
        profile = profiles.get(nid, {}) if profiles else {}

        tl  = DataLoader(NodeTrafficDataset(X_train, Y_train, nid),
                         batch_size=BATCH_TRAIN, shuffle=True,  pin_memory=True)
        vl  = DataLoader(NodeTrafficDataset(X_val,   Y_val,   nid),
                         batch_size=BATCH_EVAL,  shuffle=False, pin_memory=True)
        tel = DataLoader(NodeTrafficDataset(X_test,  Y_test,  nid),
                         batch_size=BATCH_EVAL,  shuffle=False, pin_memory=True)

        clients[nid] = FLClient(
            node_id=nid,
            cluster_id=int(cluster_labels[nid]),
            train_loader=tl,
            val_loader=vl,
            test_loader=tel,
            scaler_mean=mean_all[nid],
            scaler_std=std_all[nid],
            n_samples=X_train.shape[0],
            model_kwargs=model_kwargs,
            availability_prob=profile.get("availability_prob", 1.0),
            latency_ms=profile.get("latency_ms", 50.0),
        )

    return clients


def build_cluster_assignments(clients, mode="clustered"):
    """Build FL cluster assignments.

    - "clustered": one group per spectral cluster
    - "global": all nodes in one group (standard FedAvg)
    """
    if mode == "global":
        return {0: list(clients.keys())}

    assignments = {}
    for nid, client in clients.items():
        cid = client.cluster_id
        assignments.setdefault(cid, []).append(nid)
    return assignments


def main():
    parser = argparse.ArgumentParser(description="Run FL experiment (AdaptiveCFL)")
    parser.add_argument("--mode",      choices=["clustered", "global"], default="clustered")
    parser.add_argument("--selection", choices=["full", "random_drop", "adaptive"],
                        default="full",
                        help="Client selection strategy (full | random_drop | adaptive)")
    parser.add_argument("--rounds",       type=int,   default=30)
    parser.add_argument("--local_epochs", type=int,   default=1)
    parser.add_argument("--local_steps",  type=int,   default=None,
                        help="If set, do exactly K gradient steps per round instead of "
                             "full epochs. Faster for large node counts. "
                             "Equivalent to McMahan et al. 2017 evaluation protocol.")
    parser.add_argument("--lr",           type=float, default=LR)
    parser.add_argument("--nodes",    choices=["smoke", "dev", "all"], default="dev")
    parser.add_argument("--eval_every",   type=int,   default=5)
    parser.add_argument("--seed",         type=int,   default=42,
                        help="RNG seed for reproducible dropout simulation")
    # ── New: thesis contribution improvements ─────────────────────────────────
    parser.add_argument("--quality_agg",  action="store_true", default=False,
                        help="Use quality-weighted FedAvg (Dai et al. 2024). "
                             "Weights each client by n_samples * exp(-loss/T). "
                             "Nodes converging to lower loss get higher aggregation weight.")
    parser.add_argument("--top_k",        type=float, default=None,
                        help="Top-K gradient sparsification fraction (Gao et al. 2024 FedCE). "
                             "E.g. 0.1 transmits only the top 10%% of weight changes, "
                             "reducing upload bandwidth by 90%%. Default: None (disabled).")
    parser.add_argument("--cluster_file", type=str,   default=None,
                        help="Path to alternative cluster labels .npz file. "
                             "If not set, uses data/processed/graph_clusters.npz (spectral). "
                             "Use data/processed/dtw_clusters.npz for DTW-based clustering.")
    parser.add_argument("--tf_start",     type=float, default=0.0,
                        help="Initial teacher forcing ratio for GRUSeq2Seq models. "
                             "Decays linearly to 0 across FL rounds. "
                             "E.g. 0.5 = 50%% teacher forcing in round 1, 0%% in final round. "
                             "Ignored for non-Seq2Seq models. Default: 0.0 (disabled).")
    parser.add_argument("--hier_alpha",   type=float, default=1.0,
                        help="Hierarchical mixing coefficient (Level 2 aggregation). "
                             "1.0 = no cross-cluster sharing (flat CFL, default). "
                             "0.8 = 80%% own cluster + 20%% neighbor-weighted global.")
    parser.add_argument("--hier_every",   type=int,   default=5,
                        help="How often to perform Level 2 hierarchical aggregation. "
                             "5 = every 5 rounds (default). Lower = more sharing.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Mode: {args.mode}  |  Selection: {args.selection}  |  Seed: {args.seed}")

    # Load data
    proc     = np.load("data/processed/metr_la_processed.npz")
    scaler   = np.load("data/processed/scaler_stats.npz")
    # Support alternative cluster files (e.g. DTW-based)
    cluster_file = args.cluster_file or "data/processed/graph_clusters.npz"
    clusters = np.load(cluster_file)
    cluster_labels = clusters["cluster_labels"]
    clustering_method = "dtw" if (args.cluster_file and "dtw" in args.cluster_file) else "spectral"
    horizon  = proc["Y_train"].shape[1]
    seq_len  = proc["X_train"].shape[1]

    # Select nodes
    if args.nodes == "smoke":
        node_list = [int(n) for n in np.where(cluster_labels == 2)[0]]
        print(f"Smoke: cluster 2, {len(node_list)} nodes")
    elif args.nodes == "dev":
        node_list = DEV_NODES
        print(f"Dev subset: {len(node_list)} nodes")
    else:
        node_list = ALL_NODES
        print(f"Full dataset: {len(node_list)} nodes")

    # Assign network profiles (only used when selection != "full")
    if args.selection != "full":
        profiles = assign_node_profiles(node_list, seed=args.seed)
        tiers = {"stable": 0, "moderate": 0, "unreliable": 0}
        for p in profiles.values():
            tiers[p["tier"]] += 1
        print(f"Node profiles: {tiers}")
    else:
        profiles = None

    # Model kwargs (factory picks TSMixer or GRU based on config.MODEL_TYPE)
    model_kwargs = get_model_kwargs(horizon=horizon, seq_len=seq_len)
    print(f"Model: {model_kwargs['model_type']}")

    # Build clients
    clients = build_clients(proc, scaler, node_list, cluster_labels, model_kwargs, profiles)

    # Cluster assignments
    assignments = build_cluster_assignments(clients, mode=args.mode)
    print(f"Clusters ({args.mode}):")
    for cid, nids in sorted(assignments.items()):
        print(f"  Cluster {cid}: {len(nids)} nodes")

    # Build selector
    selector = build_selector(args.selection)

    # Compute cluster centroid distances for hierarchical aggregation
    cluster_distances = None
    if args.hier_alpha < 1.0 and args.mode == "clustered":
        try:
            locs = pd.read_csv("data/raw/graph_sensor_locations.csv")
            # Compute cluster centroids
            centroids = {}
            for cid, nids in assignments.items():
                # nids are node indices — look up their GPS coordinates
                node_indices = [clients[nid].node_id for nid in nids]
                lats = [locs.iloc[ni]["latitude"] for ni in node_indices]
                lons = [locs.iloc[ni]["longitude"] for ni in node_indices]
                centroids[cid] = (np.mean(lats), np.mean(lons))

            # Haversine pairwise distances
            def _haversine(lat1, lon1, lat2, lon2):
                R = 6371  # km
                dlat = radians(lat2 - lat1)
                dlon = radians(lon2 - lon1)
                a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
                return R * 2 * atan2(sqrt(a), sqrt(1 - a))

            cluster_distances = {}
            cids = sorted(centroids.keys())
            for i, ci in enumerate(cids):
                for cj in cids[i+1:]:
                    d = _haversine(*centroids[ci], *centroids[cj])
                    cluster_distances[(ci, cj)] = d
                    cluster_distances[(cj, ci)] = d

            print(f"Hierarchical aggregation: alpha={args.hier_alpha}, "
                  f"every={args.hier_every} rounds [dist-weighted]")
        except Exception as e:
            print(f"Warning: could not compute cluster distances ({e}), using uniform weights")

    # Run FL
    trainer = FLTrainer(
        clients=clients,
        cluster_assignments=assignments,
        model_kwargs=model_kwargs,
        device=device,
        lr=args.lr,
        max_grad_norm=MAX_GRAD_NORM,
        selector=selector,
        seed=args.seed,
        cluster_distances=cluster_distances,
    )

    results = trainer.run(
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        eval_every=args.eval_every,
        local_steps=args.local_steps,
        quality_agg=args.quality_agg,
        top_k=args.top_k,
        tf_start=args.tf_start,
        hier_alpha=args.hier_alpha,
        hier_every=args.hier_every,
    )

    # Save results
    os.makedirs("results", exist_ok=True)
    step_tag   = f"S{args.local_steps}" if args.local_steps else f"E{args.local_epochs}"
    qual_tag   = "_QW" if args.quality_agg else ""
    topk_tag   = f"_K{int(args.top_k*100)}" if args.top_k else ""
    tf_tag     = f"_TF{int(args.tf_start*100)}" if args.tf_start > 0 else ""
    hier_tag   = f"_H{int(args.hier_alpha*100)}e{args.hier_every}" if args.hier_alpha < 1.0 else ""
    clust_tag  = f"_dtw" if (args.cluster_file and "dtw" in args.cluster_file) else ""
    save_name  = (f"fl_{args.mode}_{args.selection}"
                  f"_R{args.rounds}_{step_tag}_{args.nodes}"
                  f"{qual_tag}{topk_tag}{tf_tag}{hier_tag}{clust_tag}")
    save_path  = f"results/{save_name}.npz"

    overall = results["overall"]
    par_rates = [h.get("participation_rate", 1.0) for h in results["history"]]
    np.savez(
        save_path,
        mode=args.mode,
        selection=args.selection,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        overall_mae=overall["mae"],
        overall_rmse=overall["rmse"],
        overall_mape=overall["mape"],
        h15_mae=overall["h15_mae"],
        h30_mae=overall["h30_mae"],
        h60_mae=overall["h60_mae"],
        total_time_min=results["total_time_min"],
        total_bytes_mb=results["total_bytes_mb"],
        avg_participation=np.mean(par_rates),
        round_losses=np.array([h["avg_loss"] for h in results["history"]]),
    )
    print(f"\nResults saved to {save_path}")

    # Comparison table
    print(f"\n{'='*65}")
    print(f"  ABLATION TABLE")
    print(f"{'='*65}")
    print(f"  {'Method':<28s} | {'15min':>6} | {'30min':>6} | {'60min':>6} | {'Overall':>7}")
    print(f"  {'-'*61}")
    if args.mode == "global":
        print(f"  {'Local TSMixer (baseline)':<28s} | {'TBD':>6} | {'TBD':>6} | {'TBD':>6} | {'TBD':>7}")
        print(f"  {'Global FedAvg':<28s} | {overall['h15_mae']:6.3f} | {overall['h30_mae']:6.3f} | "
              f"{overall['h60_mae']:6.3f} | {overall['mae']:7.3f}")
        print(f"  {'CFL (TBD)':<28s} | {'TBD':>6} | {'TBD':>6} | {'TBD':>6} | {'TBD':>7}")
    else:
        print(f"  {'Local TSMixer (baseline)':<28s} | {'TBD':>6} | {'TBD':>6} | {'TBD':>6} | {'TBD':>7}")
        print(f"  {'Global FedAvg':<28s} | {'TBD':>6} | {'TBD':>6} | {'TBD':>6} | {'TBD':>7}")
        row_label = f"CFL ({args.selection})"
        print(f"  {row_label:<28s} | {overall['h15_mae']:6.3f} | {overall['h30_mae']:6.3f} | "
              f"{overall['h60_mae']:6.3f} | {overall['mae']:7.3f}")

    print(f"  {'DCRNN (cited, centralized)':<28s} | {'2.770':>6} | {'3.150':>6} | {'3.600':>6} | {'3.170':>7}")
    print(f"  {'='*61}")
    print(f"\n  Avg participation rate: {np.mean(par_rates):.1%}")
    print(f"  Communication: {results['total_bytes_mb']:.1f} MB total")


if __name__ == "__main__":
    main()
