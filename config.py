"""Central configuration for the Traffic-FL project."""

# 25% representative subset (54 nodes, 3 per cluster: easy/median/hard)
# Selected based on 60min MAE distribution across 8 spectral clusters
DEV_NODES = [
    0, 3, 4, 8, 16, 17, 18, 22, 34, 37, 38, 43, 45, 49, 53, 55, 57, 59,
    61, 62, 68, 71, 76, 77, 78, 83, 87, 88, 90, 91, 99, 112, 113, 117,
    124, 126, 127, 132, 136, 137, 139, 142, 144, 148, 152, 160, 161, 167,
    171, 178, 198, 199, 200, 202,
]

ALL_NODES = list(range(207))

# ── Model selection ────────────────────────────────────────────────────────────
# Set to "tsmixer" or "gru" to switch the base model across the entire pipeline.
MODEL_TYPE = "gru_seq2seq"   # Options: "tsmixer", "gru", "gru_seq2seq"

# Training hyperparameters (shared)
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 15
BATCH_TRAIN = 512
BATCH_EVAL = 256
MAX_GRAD_NORM = 5.0

# ── GRU hyperparameters ────────────────────────────────────────────────────────
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2

# ── TSMixer hyperparameters ────────────────────────────────────────────────────
# Paper: Chen et al., "TSMixer: An All-MLP Architecture for Time Series
# Forecasting" (Google Research, 2023) — https://arxiv.org/abs/2303.06053
# NOTE: ff_dim=256 gives ~55K params (3x lighter than GRU but enough capacity)
TSMIXER_N_BLOCKS = 6      # number of Mixer blocks stacked
TSMIXER_FF_DIM   = 256    # hidden dim inside each MLP
TSMIXER_DROPOUT  = 0.1

# Seq2Seq specific (legacy — only used with seq2seq GRU)
TEACHER_FORCING_START = 0.5
TEACHER_FORCING_END   = 0.0

# ── Multivariate / time features ───────────────────────────────────────────────
# Set USE_TIME_FEATURES = True to add sin/cos time-of-day as extra input channels.
# This changes input_size from 1 → 3 (speed + sin_tod + cos_tod).
# Requires prepare_data.py to have been run with the updated version
# (which saves tod_train / tod_val / tod_test alongside X/Y).
USE_TIME_FEATURES = True           # ← change to True to enable
N_INPUT_FEATURES  = 3 if USE_TIME_FEATURES else 1
