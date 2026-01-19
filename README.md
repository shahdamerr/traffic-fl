# Traffic-fl

Developing a Federated Learning pipeline for traffic forecasting in ITS for my masters thesis titled 'Adaptive and Communication-Efficient Federated Learning for Intelligent Transportation Systems Under Data Heterogeneity and Dynamic Network Conditions '.

# Traffic-FL: Adaptive Federated Learning for Intelligent Transportation Systems

## Overview

This repository contains the implementation of an **adaptive and communication-efficient Federated Learning (FL) framework** for **traffic forecasting in Intelligent Transportation Systems (ITS)**.  
The work focuses on handling **data heterogeneity**, **dynamic network conditions**, and **scalability challenges** by leveraging **graph-based clustering** and **hierarchical decentralized learning**.

This project is developed as part of a **Master’s thesis**, with experiments conducted on the **METR-LA traffic dataset**.

---

## Research Motivation

Traffic sensor networks exhibit several real-world challenges:

- Strong **spatial heterogeneity** across regions
- **Non-IID data** distribution among sensors
- **Temporal dynamics** and non-stationarity
- **Communication constraints** in large-scale edge networks

Traditional centralized learning and standard federated learning methods (e.g., FedAvg) often perform poorly under these conditions.

This project addresses these issues by:

- Modeling sensors as a **graph**
- Partitioning the graph into **spatially coherent subgraphs**
- Enabling **localized peer-to-peer (P2P) learning**
- Reducing global communication overhead

---

## Key Contributions

- Graph-based partitioning of traffic sensors into spatial subgraphs
- Local per-sensor forecasting models (non-IID baseline)
- Hierarchical FL design:
  - node-level local models
  - subgraph-level aggregation
  - inter-subgraph coordination (planned)
- Communication-efficient design for edge environments
- Evaluation using interpretable metrics (**MAE**, **RMSE**)

---

## Dataset

The project uses the **METR-LA dataset**, which contains:

- Traffic speed measurements from **207 sensors**
- Collected every **5 minutes**
- Road network adjacency information

Dataset source (not included in repository):
https://www.kaggle.com/datasets/annnnguyen/metr-la-dataset

**Note:**  
Raw and processed data files are intentionally excluded from the repository.

---

## Project Structure

traffic-fl/
├── fl/
│ ├── dataset.py # Per-node dataset loader
│ ├── local_train.py # Local client training & evaluation
│
├── graph/
│ ├── build_graph.py # Graph construction from adjacency matrix
│ ├── partition.py # Spectral clustering into subgraphs
│
├── models/
│ └── gru_forecaster.py # GRU-based traffic forecasting model
│
├── scripts/
│ ├── prepare_data.py # Data preprocessing & windowing
│ ├── run_graph_partition.py # Graph clustering pipeline
│ ├── run_local_training.py # Local-only FL baseline
│
├── utils/
│ └── metrics.py # MAE / RMSE evaluation
│
├── .gitignore
└── README.md

---

## Methodology Overview

### Step 1 — Data Preprocessing

- Missing value handling
- Global z-score normalization
- Sliding window generation:
  - Input: past 12 time steps
  - Output: next 12 time steps
- Time-based train / validation / test split

### Step 2 — Graph Construction & Clustering

- Traffic sensors are modeled as nodes in a graph
- Edges defined using road network adjacency
- Spectral clustering applied to obtain spatial subgraphs
- Each subgraph represents a localized FL neighborhood

### Step 3 — Local Training Baseline (Current)

- Each sensor is treated as an independent FL client
- Clients train **only on their own local time series**
- GRU-based forecasting model
- Evaluation using **MAE and RMSE** on denormalized predictions

### Step 4 — Intra-cluster P2P Federated Learning (Upcoming)

- Peer-to-peer parameter exchange within each subgraph
- Subgraph-level aggregation
- Reduced communication cost compared to global FL

---

## Model

### GRU Forecaster

- Input: `[batch_size, sequence_length, 1]`
- Output: multi-step traffic speed prediction
- Lightweight and stable for edge-based federated learning

GRU is chosen due to:

- Robustness under limited local data
- Faster convergence compared to LSTM
- Suitability for decentralized learning settings

---

## Evaluation Metrics

All evaluations are performed on **denormalized predictions**:

- **MAE (Mean Absolute Error)**
- **RMSE (Root Mean Squared Error)**

These metrics are standard in traffic forecasting literature and allow comparison with existing baselines.

---

## How to Run

### 1. Data Preprocessing

````bash
python scripts/prepare_data.py

```bash
python scripts/run_local_training.py

````

## Current Status

✔ Data preprocessing implemented

✔ Graph-based clustering completed

✔ Local non-IID baseline established

⏳ Intra-cluster P2P FL (in progress)

⏳ Hierarchical aggregation and analysis (planned)
