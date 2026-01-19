import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from graph.build_graph import build_graph_from_adj
from graph.partition import spectral_partition
from graph.select_clusters_size import select_k_eigengap


# Load processed data
data = np.load("data/processed/metr_la_processed.npz")
adj = data["adj"]

# Symmetrize once (recommended)
adj = (adj + adj.T) / 2

print("Adjacency shape:", adj.shape)

G = build_graph_from_adj(adj)
print("Graph nodes:", G.number_of_nodes())
print("Graph edges:", G.number_of_edges())

k_opt, eigvals = select_k_eigengap(adj, k_max=20)
print("Selected number of clusters:", k_opt)

labels, clusters = spectral_partition(adj, k_opt)

for k, nodes in clusters.items():
    print(f"Cluster {k}: {len(nodes)} nodes")

# Save clusters + eigvals for plotting / experiments
np.savez(
    "data/processed/graph_clusters.npz",
    cluster_labels=labels,
    num_clusters=k_opt,
    eigvals=eigvals
)

print("Saved graph clusters")
