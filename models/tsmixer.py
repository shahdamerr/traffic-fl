"""TSMixer: An All-MLP Architecture for Time Series Forecasting.

Adapted from: Chen et al., "TSMixer: An All-MLP Architecture for Time Series
Forecasting" (Google Research, 2023).
ArXiv: https://arxiv.org/abs/2303.06053

IMPLEMENTATION NOTE FOR UNIVARIATE TRAFFIC FL:
    The original TSMixer uses Feature Mixing across C channels. With n_features=1
    (one speed value per sensor — strict FL data isolation), LayerNorm on a single
    scalar collapses the signal similarly to the GRU-LayerNorm issue we diagnosed.

    Fix: For univariate input, we drop the Feature Mixing block entirely and use
    a pure Temporal Mixing (MLP-over-time) architecture, which is still fully
    MLP-based, lightweight, and training-efficient. This is consistent with
    the N-BEATS and DLinear literature where purely temporal MLPs are SOTA.

    If you extend this to multivariate (n_features > 1), Feature Mixing is
    re-enabled automatically.

Input:  x [B, L, 1]   (batch, seq_len, 1 feature per sensor)
Output: y [B, H]      (horizon predictions)
"""
import torch
import torch.nn as nn


class ResidualTemporalBlock(nn.Module):
    """One residual temporal-mixing block.

    Applies a 2-layer MLP across the time dimension with a skip connection.
    No LayerNorm on the feature axis (avoids single-scalar collapse for C=1).
    """

    def __init__(self, seq_len: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        # Pre-norm on time axis only
        self.norm = nn.LayerNorm(seq_len)
        self.mlp  = nn.Sequential(
            nn.Linear(seq_len, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, seq_len),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """x: [B, C, L]  (feature-first for time-axis mixing)"""
        residual = x
        x = self.norm(x)          # LayerNorm over L (safe — L=12, not 1)
        x = self.mlp(x)           # [B, C, L]
        return x + residual


class FeatureMixingBlock(nn.Module):
    """Feature-axis mixing block (only used when n_features > 1)."""

    def __init__(self, n_features: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(n_features)
        self.mlp  = nn.Sequential(
            nn.Linear(n_features, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, n_features),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """x: [B, L, C]"""
        residual = x
        x = self.norm(x)
        x = self.mlp(x)
        return x + residual


class TSMixer(nn.Module):
    """TSMixer forecaster — drop-in replacement for GRUForecaster.

    For n_features=1 (univariate FL): pure temporal MLP-Mixer, no feature mixing.
    For n_features>1 (multivariate): temporal + feature mixing (full TSMixer).

    Args:
        seq_len:    Input sequence length (12 → 1 hour at 5-min resolution)
        horizon:    Forecast horizon (12 steps)
        n_blocks:   Number of residual temporal blocks
        ff_dim:     MLP hidden dimension
        dropout:    Dropout rate
        n_features: Number of input channels (1 for per-sensor univariate FL)

    Input:  x  [B, L, n_features]
    Output: y  [B, H]
    """

    def __init__(
        self,
        seq_len: int    = 12,
        horizon: int    = 12,
        n_blocks: int   = 4,
        ff_dim: int     = 256,
        dropout: float  = 0.1,
        n_features: int = 1,
    ):
        super().__init__()
        self.seq_len    = seq_len
        self.horizon    = horizon
        self.n_features = n_features

        # Temporal mixing blocks (always present)
        self.temporal_blocks = nn.ModuleList([
            ResidualTemporalBlock(seq_len, ff_dim, dropout)
            for _ in range(n_blocks)
        ])

        # Feature mixing (only for multivariate)
        self.feature_blocks = nn.ModuleList([
            FeatureMixingBlock(n_features, max(ff_dim // 4, 4), dropout)
            for _ in range(n_blocks)
        ]) if n_features > 1 else None

        # Projection head: [B, L, C] → [B, H]
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),                       # [B, L*C]
            nn.Linear(seq_len * n_features, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, horizon),
        )

    def forward(self, x):
        # x: [B, L, C]
        # Permute to feature-first for temporal mixing
        x = x.permute(0, 2, 1)      # [B, C, L]

        for i, tb in enumerate(self.temporal_blocks):
            x = tb(x)               # [B, C, L]  — mixing over L
            if self.feature_blocks is not None:
                x = x.permute(0, 2, 1)                    # [B, L, C]
                x = self.feature_blocks[i](x)              # mix over C
                x = x.permute(0, 2, 1)                    # [B, C, L]

        x = x.permute(0, 2, 1)      # [B, L, C]
        y = self.head(x)             # [B, H]
        return y
