import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from graph.build_graph import build_graph_from_adj
from graph.partition import spectral_partition

# Load processed data
data = np.load("data/processed/metr_la_processed.npz")
adj = data["adj"]

print("Adjacency shape:", adj.shape)

# Build graph
G = build_graph_from_adj(adj)
print("Graph nodes:", G.number_of_nodes())
print("Graph edges:", G.number_of_edges())

NUM_SUBGRAPHS = 10

labels, clusters = spectral_partition(adj, NUM_SUBGRAPHS)

for k, nodes in clusters.items():
    print(f"Cluster {k}: {len(nodes)} nodes")

np.savez(
    "data/processed/graph_clusters.npz",
    cluster_labels=labels,
    num_clusters=NUM_SUBGRAPHS
)

print("Saved graph clusters")
