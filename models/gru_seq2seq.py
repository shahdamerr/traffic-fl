"""GRU Encoder-Decoder (Seq2Seq) for traffic flow prediction.

Reproduces the client model architecture from:
    Zhu et al. 2025 — "CMBA-FL: Communication-mitigated and blockchain-assisted
    federated learning for traffic flow predictions"
    Digital Communications and Networks 11 (2025) 724–733

Architecture (Section 3.2, Equations 3–4):
    Encoder:  Multi-layer GRU reads input sequence X = {x1..x12}
              → produces context hidden state hc
    Decoder:  GRU + Linear (FNN), autoregressive step-by-step decoding
              Step 1: Decoder(x_last, hc)   → h1, ŷ1
              Step 2: Decoder(ŷ1, h1)       → h2, ŷ2
              ...
              Step H: Decoder(ŷ_{H-1}, h_{H-1}) → hH, ŷH

Teacher forcing (training aid):
    During training, with probability `teacher_forcing_ratio`, feed the
    ground-truth value instead of the model's own prediction at each step.
    This prevents error accumulation during early training and improves
    convergence. The ratio should decay from ~0.5 to 0.0 over training.

Input:  x  [B, L, 1]   (batch, seq_len=12, 1 speed feature)
Output: y  [B, H]      (batch, horizon=12 predicted speed values)
"""
import torch
import torch.nn as nn


class GRUSeq2Seq(nn.Module):
    """GRU-based Encoder-Decoder for univariate traffic speed forecasting.

    Drop-in replacement for GRUForecaster / TSMixer — same input/output shapes.
    """

    # Flag for the FL pipeline to detect seq2seq models
    is_seq2seq = True

    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 2,
        horizon: int = 12,
        dropout: float = 0.2,
        input_size: int = 1,
    ):
        """Args:
            input_size: number of input features per time step.
                1 = speed only (default, backward compatible).
                3 = speed + sin(tod) + cos(tod)  (multivariate mode).
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.horizon = horizon
        self.input_size = input_size

        # ── Encoder ──────────────────────────────────────────────────────
        # Multi-layer GRU that reads the full input sequence and produces
        # a context hidden state hc capturing temporal patterns.
        self.encoder = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # ── Decoder ──────────────────────────────────────────────────────
        # GRU that processes one time step at a time (autoregressive).
        # The decoder always receives only the predicted speed scalar (1 feature)
        # since we only forecast speed (not time features).
        self.decoder_gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # FNN projection: hidden state → 1 predicted speed value
        # (the paper calls this the "Fully Connected Network" in the decoder)
        self.decoder_fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x, y=None, teacher_forcing_ratio=0.0):
        """Forward pass: encode input sequence, decode future predictions.

        Args:
            x: [B, L, 1] — input speed sequence (12 past time steps)
            y: [B, H] — ground truth future values (only used during training
                         for teacher forcing). None during evaluation.
            teacher_forcing_ratio: float in [0, 1]. Probability of using
                ground truth instead of own prediction at each decoder step.
                0.0 = pure autoregressive (paper's algorithm, used at eval).
                0.5 = 50% chance of using real value (training aid).

        Returns:
            predictions: [B, H] — predicted speed values for H future steps
        """
        # ── Encode ───────────────────────────────────────────────────────
        # hc shape: [num_layers, B, hidden_size]
        _, hc = self.encoder(x)

        # ── Decode (autoregressive, one step at a time) ──────────────────
        # First decoder input: last observed SPEED value from input sequence.
        # We always take only feature 0 (speed) because the decoder GRU has
        # input_size=1 — it predicts speed, not time features.
        decoder_input = x[:, -1:, 0:1]  # [B, 1, 1] — speed only
        decoder_hidden = hc              # [num_layers, B, H]

        outputs = []
        for t in range(self.horizon):
            # One GRU step
            decoder_out, decoder_hidden = self.decoder_gru(
                decoder_input, decoder_hidden
            )
            # Project hidden → predicted speed
            prediction = self.decoder_fc(decoder_out.squeeze(1))  # [B, 1]
            outputs.append(prediction)

            # Next input: teacher forcing or own prediction
            if self.training and y is not None and torch.rand(1).item() < teacher_forcing_ratio:
                # Use ground truth value as next input
                decoder_input = y[:, t].unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
            else:
                # Use own prediction as next input (pure autoregressive)
                decoder_input = prediction.unsqueeze(1)  # [B, 1, 1]

        # Concatenate all step predictions: [B, H]
        return torch.cat(outputs, dim=1)
