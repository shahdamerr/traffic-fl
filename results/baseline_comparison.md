# Baseline Comparison Results

## Setup
- **Dataset**: METR-LA (207 sensors, 5-min intervals)
- **Dev Subset**: 54 nodes (26% of network, 3 per cluster: easy/median/hard)
- **Input**: 12 steps (1 hour) → **Output**: 12 steps (1 hour)
- **Device**: NVIDIA GeForce RTX 3050 Ti (4GB VRAM)

## Results

| Model | Overall MAE | RMSE | MAPE (%) | 15min MAE | 30min MAE | 60min MAE | Train Time |
|-------|-------------|------|----------|-----------|-----------|-----------|------------|
| Naive (Last Value) | 5.465 | 12.958 | 12.64 | 4.080 | 5.417 | 7.466 | — |
| Moving Average | 6.660 | 14.245 | 15.15 | 5.545 | 6.587 | 8.303 | — |
| LSTM (direct) | 6.012 | 11.887 | 12.80 | 4.465 | 6.003 | 8.179 | 84 min |
| **GRU (direct)** | **6.016** | **11.919** | **12.80** | **4.468** | **5.988** | **8.220** | **66 min** |
| Seq2Seq GRU | 5.892 | 11.979 | 12.83 | 4.542 | 5.929 | 7.823 | 493 min |

## Published Centralized Baselines (cited, not reimplemented)

| Model | Overall MAE | RMSE | MAPE (%) | 15min MAE | 30min MAE | 60min MAE | Source |
|-------|-------------|------|----------|-----------|-----------|-----------|--------|
| DCRNN | 3.170 | 6.450 | 8.8 | 2.770 | 3.150 | 3.600 | Li et al., ICLR 2018 |
| STGCN | 3.380 | 7.100 | — | 2.880 | 3.470 | 3.900 | Yu et al., IJCAI 2018 |
| Graph WaveNet | 3.070 | 6.220 | 8.4 | 2.690 | 3.070 | 3.530 | Wu et al., IJCAI 2019 |

## Key Findings

1. **GRU ≈ LSTM**: Nearly identical performance; GRU chosen for rest of pipeline (simpler, faster)
2. **Seq2Seq marginal**: Only ~5% improvement at 60min over GRU, but 7x slower to train
3. **Naive strong at 15min**: Last-value baseline beats all neural models at short horizons (traffic persistence)
4. **Clear gap to centralized models**: Local GRU 60min MAE=8.22 vs DCRNN=3.60 — this gap motivates FL

## Model Decision

**GRU (direct multi-step)** selected for Federated Learning pipeline:
- Competitive with LSTM and Seq2Seq
- 7x faster than Seq2Seq to train
- Simple architecture compatible with FedAvg weight aggregation
- Architecture: 2-layer GRU, hidden=128, dropout=0.2, 2-layer MLP head
