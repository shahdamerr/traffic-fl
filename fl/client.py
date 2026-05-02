"""FL Client abstraction.

Each FLClient represents one sensor node in the federated learning system.
It holds its own model, data loaders, and provides methods for local training
and evaluation.

Also models realistic dynamic network conditions for ablation experiments:
    - availability_prob: Bernoulli(p) availability each round
    - latency_ms: Uniform(50, 500) ms communication latency
    - reliability_score: composite score used by adaptive client selection
"""
import copy
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from models.factory import build_model
from utils.metrics import mae, rmse, mape, evaluate_per_horizon


class FLClient:
    """A single federated learning client (one sensor node).

    Attributes:
        node_id: sensor index (0-206)
        cluster_id: cluster assignment
        n_samples: number of training samples (for weighted aggregation)

        # Network simulation (for dynamic conditions ablation)
        availability_prob: probability of being online each round (Bernoulli)
        latency_ms: communication latency in ms (Uniform 50-500)
        reliability_score: composite score = availability * exp(-latency/500)
    """

    # Latency ceiling for normalising reliability scores
    _LATENCY_MAX_MS = 500.0

    def __init__(
        self,
        node_id,
        cluster_id,
        train_loader,
        val_loader,
        test_loader,
        scaler_mean,
        scaler_std,
        n_samples,
        model_kwargs,
        availability_prob: float = 1.0,
        latency_ms: float = 50.0,
    ):
        self.node_id = node_id
        self.cluster_id = cluster_id
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler_mean = scaler_mean
        self.scaler_std = scaler_std
        self.n_samples = n_samples
        self.model_kwargs = model_kwargs

        # Network simulation fields
        self.availability_prob = availability_prob
        self.latency_ms = latency_ms
        self.reliability_score = self._compute_score(availability_prob, latency_ms)

        # Build model via factory (honours config.MODEL_TYPE)
        self.model = build_model(model_kwargs)

    @staticmethod
    def _compute_score(availability_prob: float, latency_ms: float) -> float:
        """Composite reliability score ∈ [0, 1].

        s = p * exp(-latency / latency_max)
        Higher is better. Used by adaptive client selection.
        """
        import math
        return availability_prob * math.exp(-latency_ms / FLClient._LATENCY_MAX_MS)

    def sample_availability(self, rng=None) -> bool:
        """Sample whether this client is available this round.

        Args:
            rng: np.random.Generator (pass for reproducibility)
        Returns:
            True if the client participates this round.
        """
        if rng is None:
            return np.random.random() < self.availability_prob
        return rng.random() < self.availability_prob

    def set_weights(self, state_dict):
        """Replace model weights with the aggregated cluster model."""
        self.model.load_state_dict(copy.deepcopy(state_dict))

    def get_weights(self):
        """Return a copy of the current model weights."""
        return copy.deepcopy(self.model.state_dict())

    def local_update(self, local_epochs, lr, max_grad_norm, device,
                      local_steps=None, teacher_forcing_ratio=0.0):
        """Perform local training. Returns updated state_dict.

        Two modes (mutually exclusive):

        1. local_epochs (default):  train for E complete passes over all data.
           Standard FL evaluation. Slow for large node counts.

        2. local_steps (fast mode): train for exactly K gradient steps,
           cycling through the DataLoader as needed. This is how FL is
           evaluated in the original FedAvg paper (McMahan et al., 2017)
           and is 20-30x faster per round on large node sets.

        Args:
            local_epochs:  number of full epochs (used if local_steps is None)
            lr:            fixed learning rate
            max_grad_norm: gradient clipping threshold
            device:        torch device
            local_steps:   if set, do exactly this many gradient steps per round
            teacher_forcing_ratio: float in [0,1]. For Seq2Seq models, probability
                of feeding ground truth instead of own prediction at each decoder
                step. Ignored for non-Seq2Seq models. (Zhu et al. 2025)

        Returns:
            state_dict: updated model weights
            avg_loss:   average training loss
        """
        self.model.to(device)
        self.model.train()

        # Detect if this is a Seq2Seq model
        use_tf = getattr(self.model, 'is_seq2seq', False) and teacher_forcing_ratio > 0

        # Fresh optimizer each round — standard FedAvg
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        total_loss = 0.0
        total_samples = 0

        if local_steps is not None:
            # ── Fast mode: exactly K gradient steps ───────────────────────
            # Cycle through the DataLoader indefinitely until K steps done.
            loader_iter = iter(self.train_loader)
            for _ in range(local_steps):
                try:
                    x, y = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(self.train_loader)
                    x, y = next(loader_iter)

                x, y = x.to(device), y.to(device)
                if use_tf:
                    pred = self.model(x, y=y, teacher_forcing_ratio=teacher_forcing_ratio)
                else:
                    pred = self.model(x)
                loss = loss_fn(pred, y)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                optimizer.step()

                total_loss += loss.item() * x.size(0)
                total_samples += x.size(0)
        else:
            # ── Standard mode: full epochs ────────────────────────────────
            for ep in range(local_epochs):
                for x, y in self.train_loader:
                    x = x.to(device)
                    y = y.to(device)

                    if use_tf:
                        pred = self.model(x, y=y, teacher_forcing_ratio=teacher_forcing_ratio)
                    else:
                        pred = self.model(x)
                    loss = loss_fn(pred, y)

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()

                    total_loss += loss.item() * x.size(0)
                    total_samples += x.size(0)

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        return self.get_weights(), avg_loss

    def evaluate(self, device):
        """Evaluate model on test set.

        Returns:
            preds: [N_test, H] denormalized predictions
            trues: [N_test, H] denormalized ground truth
        """
        self.model.to(device)
        self.model.eval()

        preds_list, trues_list = [], []
        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(device)
                pred = self.model(x)
                preds_list.append(pred.cpu().numpy())
                trues_list.append(y.numpy())

        preds = np.concatenate(preds_list, axis=0)
        trues = np.concatenate(trues_list, axis=0)

        # Denormalize
        preds = preds * self.scaler_std + self.scaler_mean
        trues = trues * self.scaler_std + self.scaler_mean

        return preds, trues

    def get_metrics(self, device):
        """Evaluate and return metrics dict."""
        preds, trues = self.evaluate(device)
        h = evaluate_per_horizon(trues, preds)
        return {
            "mae": mae(trues, preds),
            "rmse": rmse(trues, preds),
            "mape": mape(trues, preds),
            "h15_mae": h["15min"]["mae"],
            "h30_mae": h["30min"]["mae"],
            "h60_mae": h["60min"]["mae"],
        }
