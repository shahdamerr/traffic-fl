"""Federated Learning Trainer.

Orchestrates synchronous FedAvg across clusters.
Supports both Clustered FL and Global FedAvg (single cluster = all nodes).

Hierarchical aggregation (Level 2):
    After intra-cluster FedAvg, each cluster model is blended with a
    distance-weighted global average every N rounds. Nearby clusters
    share more knowledge, matching real-world congestion propagation.
    Controlled by hier_alpha (blend ratio) and hier_every (frequency).
"""
import time
import copy
import numpy as np
import torch

from models.factory import build_model
from fl.aggregation import fedavg, quality_weighted_fedavg, sparsify_delta, count_bytes, count_parameters
from fl.client_selector import FullParticipationSelector, build_selector
from utils.metrics import mae, rmse, mape


class FLTrainer:
    """Synchronous Federated Averaging trainer.

    Handles both:
        - Clustered FL: multiple clusters, each with its own model
        - Global FedAvg: one cluster containing all nodes

    Args:
        clients: dict[node_id -> FLClient]
        cluster_assignments: dict[cluster_id -> list[node_id]]
        model_kwargs: dict passed to GRUForecaster() for initialization
        device: torch device string
        lr: learning rate for local updates
        max_grad_norm: gradient clipping threshold
    """

    def __init__(self, clients, cluster_assignments, model_kwargs, device, lr,
                 max_grad_norm=5.0, selector=None, seed=42,
                 cluster_distances=None):
        self.clients = clients
        self.cluster_assignments = cluster_assignments
        self.model_kwargs = model_kwargs
        self.device = device
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        # Client selector (defaults to full participation if not provided)
        self.selector = selector if selector is not None else FullParticipationSelector()
        # Seeded RNG for reproducible dropout / availability sampling
        self._rng = np.random.default_rng(seed)

        # Initialize ALL clusters from the SAME random model (standard FL).
        # This ensures divergence comes only from data heterogeneity,
        # not from random initialization luck.
        shared_init = build_model(model_kwargs).state_dict()
        self.cluster_models = {}
        for cid in cluster_assignments:
            self.cluster_models[cid] = copy.deepcopy(shared_init)

        # Communication tracking
        self._param_bytes = count_bytes(self.cluster_models[list(cluster_assignments.keys())[0]])
        self._n_params = count_parameters(self.cluster_models[list(cluster_assignments.keys())[0]])
        self.total_bytes_communicated = 0

        # Hierarchical aggregation: inter-cluster distance weights
        # cluster_distances: dict[(cid_i, cid_j)] -> distance_km
        self.cluster_distances = cluster_distances
        self._hier_weights = None  # computed lazily on first use

        # History
        self.round_history = []

    def _compute_hier_weights(self):
        """Compute per-cluster blending weights from geographic distances.

        For each cluster i, compute how much influence every other cluster j
        should have, proportional to 1/distance(i, j). Closer clusters share
        more knowledge, matching real-world congestion propagation.

        Returns:
            dict[cid -> dict[cid -> float]]: normalized weights per cluster
        """
        cids = sorted(self.cluster_assignments.keys())
        weights = {}
        for ci in cids:
            raw = {}
            for cj in cids:
                if ci == cj:
                    continue
                key = (ci, cj) if (ci, cj) in self.cluster_distances else (cj, ci)
                dist = self.cluster_distances.get(key, 1.0)
                raw[cj] = 1.0 / max(dist, 0.1)  # avoid div by zero
            total = sum(raw.values())
            weights[ci] = {cj: w / total for cj, w in raw.items()} if total > 0 else {}
        return weights

    def _hierarchical_aggregate(self, hier_alpha, verbose=False):
        """Level 2: Inter-cluster hierarchical aggregation.

        Blends each cluster model with a distance-weighted average of the
        other cluster models. Nearby clusters contribute more.

        cluster_i = alpha * cluster_i + (1-alpha) * weighted_avg(other clusters)

        Args:
            hier_alpha: float in [0, 1]. 1.0 = no sharing (flat CFL).
            verbose: print debug info
        """
        if self._hier_weights is None:
            if self.cluster_distances is not None:
                self._hier_weights = self._compute_hier_weights()
            else:
                # Uniform weights (no distance info)
                cids = sorted(self.cluster_assignments.keys())
                self._hier_weights = {
                    ci: {cj: 1.0 / (len(cids) - 1) for cj in cids if cj != ci}
                    for ci in cids
                }

        # Blend each cluster with its distance-weighted neighbor average
        blended_models = {}
        for cid in self.cluster_models:
            neighbor_weights = self._hier_weights[cid]
            if not neighbor_weights:
                blended_models[cid] = self.cluster_models[cid]
                continue

            # Compute distance-weighted average of neighbor cluster models
            neighbor_sd = {}
            for key in self.cluster_models[cid]:
                neighbor_sd[key] = torch.zeros_like(
                    self.cluster_models[cid][key], dtype=torch.float32
                )
                for other_cid, w in neighbor_weights.items():
                    neighbor_sd[key] += self.cluster_models[other_cid][key].float() * w

            # Blend: alpha * own + (1-alpha) * neighbor-weighted global
            # IMPORTANT: cast back to original dtype to avoid corrupting
            # non-float buffers (e.g. BatchNorm num_batches_tracked is int64)
            blended = {}
            orig_sd = self.cluster_models[cid]
            for key in orig_sd:
                orig_dtype = orig_sd[key].dtype
                if orig_dtype.is_floating_point:
                    blended[key] = (
                        hier_alpha * orig_sd[key].float()
                        + (1 - hier_alpha) * neighbor_sd[key]
                    ).to(orig_dtype)
                else:
                    # Non-float buffers (int64 counters etc.) — keep own value
                    blended[key] = orig_sd[key].clone()
            blended_models[cid] = blended

        self.cluster_models = blended_models

        # Track inter-cluster communication cost
        # Each cluster uploads its model + downloads the blended result
        self.total_bytes_communicated += (
            2 * len(self.cluster_models) * self._param_bytes
        )

    def run(self, rounds, local_epochs, eval_every=5, verbose=True,
            local_steps=None, quality_agg=False, top_k=None, tf_start=0.0,
            hier_alpha=1.0, hier_every=5, lr_schedule=None):
        """Run the FL training loop.

        Args:
            rounds:      number of communication rounds (R)
            local_epochs: local epochs per round (E)
            eval_every:  evaluate every N rounds (0 = only at end)
            verbose:     print progress
            local_steps: if set, do exactly K gradient steps instead of epochs
            quality_agg: if True, use quality-weighted FedAvg (Dai et al. 2024)
                         Nodes achieving lower training loss get higher weight.
            top_k:       if set (0<top_k<=1.0), apply Top-K gradient sparsification
                         before upload (Gao et al. 2024 -- FedCE). Reduces
                         upload comm cost by (1-top_k)*100%.
            tf_start:    initial teacher forcing ratio for Seq2Seq models.
                         Decays linearly to 0 over all rounds. Only affects
                         GRUSeq2Seq; ignored for GRU/TSMixer. (default: 0.0)
            hier_alpha:  hierarchical mixing coefficient (Level 2).
                         1.0 = no cross-cluster sharing (flat CFL, default).
                         0.8 = 80% own cluster + 20% neighbor-weighted global.
                         Applied every hier_every rounds.
            hier_every:  how often to perform Level 2 aggregation.
                         5 = every 5 rounds (default). Lower = more sharing
                         but higher communication cost.
            lr_schedule: optional list of (round, lr) tuples for step-wise
                         LR decay. E.g. [(40, 0.00025), (60, 0.0001)] means
                         lr drops to 0.00025 at round 40 and 0.0001 at round 60.
                         If None, lr stays constant throughout training.

        Returns:
            dict with final metrics and history
        """
        n_clients = sum(len(nids) for nids in self.cluster_assignments.values())
        n_clusters = len(self.cluster_assignments)
        use_hier = hier_alpha < 1.0 and n_clusters > 1

        if verbose:
            print(f"\n{'='*60}")
            print(f"  FL Training: {n_clients} clients, {n_clusters} clusters")
            if local_steps is not None:
                print(f"  Rounds={rounds}, Local Steps={local_steps} (fast mode), LR={self.lr}")
            else:
                print(f"  Rounds={rounds}, Local Epochs={local_epochs}, LR={self.lr}")
            print(f"  Selection: {self.selector}")
            agg_label = f"QualityWeighted(T=0.5)" if quality_agg else "FedAvg"
            top_k_label = f", Top-K={top_k:.0%}" if top_k else ""
            tf_label = f", TF={tf_start:.0%}->0%" if tf_start > 0 else ""
            hier_label = (f", Hier(a={hier_alpha}, every={hier_every})"
                          if use_hier else "")
            dist_label = " [dist-weighted]" if (use_hier and self.cluster_distances) else ""
            print(f"  Aggregation: {agg_label}{top_k_label}{tf_label}{hier_label}{dist_label}")
            print(f"  Model params: {self._n_params:,} ({self._param_bytes:,} bytes)")
            print(f"{'='*60}\n")

        start_time = time.time()

        # Pre-sort LR schedule for efficient lookup
        _lr_milestones = sorted(lr_schedule, key=lambda x: x[0]) if lr_schedule else []

        for r in range(1, rounds + 1):
            round_start = time.time()
            round_losses = []
            total_selected = 0

            # ── Step-wise LR decay ────────────────────────────────────────
            for milestone_round, milestone_lr in _lr_milestones:
                if r == milestone_round:
                    old_lr = self.lr
                    self.lr = milestone_lr
                    if verbose:
                        print(f"    >> LR decay: {old_lr} -> {self.lr} at round {r}")
                    break

            # Compute teacher forcing ratio for this round (decays linearly)
            tf_ratio = tf_start * (1.0 - (r - 1) / max(rounds - 1, 1)) if tf_start > 0 else 0.0

            # Process each cluster
            for cid, node_ids in self.cluster_assignments.items():
                # 1. Adaptive selection: determine which clients participate this round
                selected_ids = self.selector.select(
                    clients=self.clients,
                    cluster_node_ids=node_ids,
                    rng=self._rng,
                    round_num=r,
                )
                total_selected += len(selected_ids)

                # 2. Broadcast current cluster model to ALL clients in cluster
                #    (even non-selected ones receive the latest model)
                for nid in node_ids:
                    self.clients[nid].set_weights(self.cluster_models[cid])

                # 3. Local training — only selected clients train
                updated_sds = []
                client_weights = []
                client_losses  = []   # for quality-weighted aggregation

                for nid in selected_ids:
                    # Save initial weights before training (needed for Top-K delta)
                    initial_sd = self.clients[nid].get_weights() if top_k else None

                    sd, loss = self.clients[nid].local_update(
                        local_epochs=local_epochs,
                        lr=self.lr,
                        max_grad_norm=self.max_grad_norm,
                        device=self.device,
                        local_steps=local_steps,
                        teacher_forcing_ratio=tf_ratio,
                    )

                    # Top-K gradient sparsification (Gao et al. 2024 — FedCE)
                    if top_k is not None and initial_sd is not None:
                        sparse_delta, actual_frac = sparsify_delta(initial_sd, sd, top_k)
                        # Reconstruct full model: apply sparse delta to cluster base
                        reconstructed = {
                            k: self.cluster_models[cid][k].float() + sparse_delta[k]
                            for k in sparse_delta
                        }
                        updated_sds.append(reconstructed)
                    else:
                        updated_sds.append(sd)

                    client_weights.append(self.clients[nid].n_samples)
                    client_losses.append(loss)
                    round_losses.append(loss)

                # 4. Aggregation
                if quality_agg:
                    # Quality-weighted FedAvg (Dai et al. 2024)
                    self.cluster_models[cid] = quality_weighted_fedavg(
                        updated_sds, client_weights, client_losses, temperature=0.5
                    )
                else:
                    # Standard FedAvg (McMahan et al. 2017)
                    total_samples = sum(client_weights)
                    norm_weights = [w / total_samples for w in client_weights]
                    self.cluster_models[cid] = fedavg(updated_sds, norm_weights)

                # 5. Track communication cost
                upload_frac = top_k if top_k is not None else 1.0
                self.total_bytes_communicated += (
                    len(selected_ids) * int(self._param_bytes * upload_frac) +  # sparse uploads
                    len(node_ids) * self._param_bytes                            # full downloads
                )

            # Log round
            avg_loss = np.mean(round_losses) if round_losses else float('nan')
            elapsed = time.time() - round_start
            total_elapsed = time.time() - start_time
            participation_rate = total_selected / n_clients

            self.round_history.append({
                "round": r,
                "avg_loss": avg_loss,
                "round_time": elapsed,
                "participation_rate": participation_rate,
            })

            if verbose:
                remaining = (total_elapsed / r) * (rounds - r)
                print(f"  Round {r:3d}/{rounds}: loss={avg_loss:.6f}  "
                      f"par={participation_rate:.0%}  "
                      f"time={elapsed:.1f}s  "
                      f"ETA={remaining/60:.1f}min")

            # Periodic evaluation — runs BEFORE hier sync so it matches
            # the same model state that the final evaluation will see
            # (clients are set_weights'd at top of each round from cluster_models)
            if eval_every > 0 and r % eval_every == 0:
                # Temporarily push current cluster models to clients for eval
                for cid, node_ids in self.cluster_assignments.items():
                    for nid in node_ids:
                        self.clients[nid].set_weights(self.cluster_models[cid])
                metrics = self.evaluate_all()
                if verbose:
                    print(f"    >> Eval: MAE={metrics['overall']['mae']:.3f}  "
                          f"60min={metrics['overall']['h60_mae']:.3f}")

            # ── Level 2: Hierarchical inter-cluster aggregation ──────────
            # Runs AFTER eval so eval reflects the pre-sync model state.
            # Skipped on the last round: a sync only benefits future training
            # rounds — firing on round R with no rounds remaining corrupts the
            # final model before evaluation with no opportunity to recover.
            if use_hier and r % hier_every == 0 and r < rounds:
                self._hierarchical_aggregate(hier_alpha, verbose=verbose)
                if verbose:
                    print(f"    >> Hier sync (a={hier_alpha}, "
                          f"{'dist-weighted' if self.cluster_distances else 'uniform'})")

        # Final: set all clients to their cluster's final model
        for cid, node_ids in self.cluster_assignments.items():
            for nid in node_ids:
                self.clients[nid].set_weights(self.cluster_models[cid])

        total_time = time.time() - start_time

        # Final evaluation
        final_metrics = self.evaluate_all()
        final_metrics["total_time_min"] = total_time / 60
        final_metrics["total_bytes"] = self.total_bytes_communicated
        final_metrics["total_bytes_mb"] = self.total_bytes_communicated / (1024 * 1024)
        final_metrics["history"] = self.round_history

        if verbose:
            print(f"\n  Training complete in {total_time/60:.1f} min")
            print(f"  Communication: {final_metrics['total_bytes_mb']:.1f} MB total")
            self._print_results(final_metrics)

        return final_metrics

    def evaluate_all(self):
        """Evaluate all clients on test set.

        Returns dict with per-node, per-cluster, and overall metrics.
        """
        all_preds, all_trues = [], []
        per_node = {}
        per_cluster = {cid: [] for cid in self.cluster_assignments}

        # Build reverse map: node_id -> FL cluster (NOT the original spectral cluster)
        node_to_fl_cluster = {}
        for cid, nids in self.cluster_assignments.items():
            for nid in nids:
                node_to_fl_cluster[nid] = cid

        for nid, client in self.clients.items():
            preds, trues = client.evaluate(self.device)
            all_preds.append(preds)
            all_trues.append(trues)

            node_metrics = {
                "mae": mae(trues, preds),
                "rmse": rmse(trues, preds),
                "mape": mape(trues, preds),
                "h15_mae": mae(trues[:, 2], preds[:, 2]),
                "h30_mae": mae(trues[:, 5], preds[:, 5]),
                "h60_mae": mae(trues[:, 11], preds[:, 11]),
            }
            per_node[nid] = node_metrics
            fl_cid = node_to_fl_cluster[nid]
            per_cluster[fl_cid].append(node_metrics)

        # Aggregate per-cluster
        cluster_avg = {}
        for cid, metrics_list in per_cluster.items():
            if len(metrics_list) == 0:
                continue
            cluster_avg[cid] = {
                k: np.mean([m[k] for m in metrics_list])
                for k in metrics_list[0]
            }

        # Overall aggregate
        p = np.concatenate(all_preds, axis=0)
        t = np.concatenate(all_trues, axis=0)
        overall = {
            "mae": mae(t, p),
            "rmse": rmse(t, p),
            "mape": mape(t, p),
            "h15_mae": mae(t[:, 2], p[:, 2]),
            "h30_mae": mae(t[:, 5], p[:, 5]),
            "h60_mae": mae(t[:, 11], p[:, 11]),
        }

        return {
            "per_node": per_node,
            "per_cluster": cluster_avg,
            "overall": overall,
        }

    def _print_results(self, metrics):
        """Print formatted results."""
        print(f"\n{'='*70}")
        print(f"  FINAL RESULTS")
        print(f"{'='*70}")

        overall = metrics["overall"]
        print(f"\n  Overall: MAE={overall['mae']:.3f}  RMSE={overall['rmse']:.3f}  "
              f"MAPE={overall['mape']:.2f}%")
        print(f"  Per-horizon:  15min={overall['h15_mae']:.3f}  "
              f"30min={overall['h30_mae']:.3f}  60min={overall['h60_mae']:.3f}")

        print(f"\n  Per-cluster:")
        for cid in sorted(metrics["per_cluster"].keys()):
            cm = metrics["per_cluster"][cid]
            n_nodes = len(self.cluster_assignments[cid])
            print(f"    Cluster {cid} ({n_nodes:>2d} nodes): "
                  f"MAE={cm['mae']:.3f}  60min={cm['h60_mae']:.3f}")

        print(f"\n{'='*70}")
