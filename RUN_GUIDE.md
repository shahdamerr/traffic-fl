# How to Run the FL Experiments

## Prerequisites
- All code is already written and ready
- PyTorch with CUDA is installed (verified: RTX 3050 Ti, `torch 2.11.0+cu128`)
- Data is at `data/processed/metr_la_processed.npz`

---

## GPU vs CPU

The scripts **auto-detect GPU**. If CUDA is available, it uses GPU. No flags needed.

To **force CPU** (if GPU is busy or you want to compare):
```powershell
$env:CUDA_VISIBLE_DEVICES="-1"
# then run any command below
```

To **reset back to GPU**:
```powershell
Remove-Item Env:CUDA_VISIBLE_DEVICES
```

---

## Step-by-Step Experiments

### Step 1: Smoke Test (verify FL works — ~2 min)

```powershell
cd C:\Users\hp\Desktop\Masters\traffic-fl
.venv\Scripts\python.exe scripts\run_fl_experiment.py --mode clustered --rounds 5 --local_epochs 1 --nodes smoke --eval_every 2
```

**What it does**: Runs clustered FL on 1 cluster (8 nodes), 5 rounds. Verifies weights are shared and loss decreases.

**Expected output**: MAE should be ~5.0 (better than local GRU's 6.0).

---

### Step 2: Clustered FL — Dev Subset (~70 min on GPU)

```powershell
.venv\Scripts\python.exe scripts\run_fl_experiment.py --mode clustered --rounds 30 --local_epochs 1 --nodes dev --eval_every 5
```

**What it does**: Runs intra-cluster FedAvg on all 8 clusters using 54 dev nodes, 30 communication rounds.

**Results saved to**: `results/fl_clustered_R30_E1_dev.npz`

---

### Step 3: Global FedAvg — Dev Subset (~70 min on GPU)

```powershell
.venv\Scripts\python.exe scripts\run_fl_experiment.py --mode global --rounds 30 --local_epochs 1 --nodes dev --eval_every 5
```

**What it does**: Standard FedAvg with ONE global model across all 54 nodes (no clustering).

**Results saved to**: `results/fl_global_R30_E1_dev.npz`

---

### Step 4: Compare All Results

After Steps 2 and 3 complete, the comparison table prints automatically at the end of each run:

```
  Method               |   15min |   30min |   60min |  Overall
  --------------------------------------------------------
  Local GRU            |   4.468 |   5.988 |   8.220 |    6.016
  Clustered FL         |   ?.??? |   ?.??? |   ?.??? |    ?.???
  Global FedAvg        |   ?.??? |   ?.??? |   ?.??? |    ?.???
  DCRNN (cited)        |   2.770 |   3.150 |   3.600 |    3.170
```

---

## Optional: Ablation Experiments

### Vary local epochs (E)
```powershell
# E=2
.venv\Scripts\python.exe scripts\run_fl_experiment.py --mode clustered --rounds 20 --local_epochs 2 --nodes dev

# E=5
.venv\Scripts\python.exe scripts\run_fl_experiment.py --mode clustered --rounds 10 --local_epochs 5 --nodes dev
```

### Vary communication rounds (R)
```powershell
# R=10
.venv\Scripts\python.exe scripts\run_fl_experiment.py --mode clustered --rounds 10 --local_epochs 1 --nodes dev

# R=20
.venv\Scripts\python.exe scripts\run_fl_experiment.py --mode clustered --rounds 20 --local_epochs 1 --nodes dev
```

### Vary learning rate
```powershell
.venv\Scripts\python.exe scripts\run_fl_experiment.py --mode clustered --rounds 30 --local_epochs 1 --nodes dev --lr 0.0005
```

---

## Final Thesis Run (all 207 nodes — ~4-6 hours on GPU)

Only run this ONCE after you're happy with dev results:

```powershell
# Clustered FL - all nodes
.venv\Scripts\python.exe scripts\run_fl_experiment.py --mode clustered --rounds 30 --local_epochs 1 --nodes all --eval_every 10

# Global FedAvg - all nodes
.venv\Scripts\python.exe scripts\run_fl_experiment.py --mode global --rounds 30 --local_epochs 1 --nodes all --eval_every 10
```

---

## Where Results Are Saved

| File | Contents |
|------|----------|
| `results/fl_clustered_R30_E1_dev.npz` | Clustered FL dev results |
| `results/fl_global_R30_E1_dev.npz` | Global FedAvg dev results |
| `results/baseline_comparison.md` | Local baseline comparison table |
| `data/models_local/aggregate_results.npz` | Full 207-node local GRU results |

---

## Troubleshooting

**"CUDA out of memory"**: Reduce batch size in `config.py`:
```python
BATCH_TRAIN = 32  # was 64
```

**Very slow on GPU**: Check GPU is being used:
```powershell
.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

**Want to resume a killed run**: Not supported — just re-run from scratch. Each experiment takes ~70 min.
