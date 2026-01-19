import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from graph.build_graph import build_graph_from_adj

# Load adjacency
data = np.load("data/processed/metr_la_processed.npz")
adj = data["adj"]

# Build graph
G = build_graph_from_adj(adj)

plt.figure(figsize=(10, 10))

# Spring layout for visualization (topology-based)
pos = nx.spring_layout(G, seed=42)

nx.draw(
    G,
    pos,
    node_size=20,
    alpha=0.6,
    edge_color="gray",
    width=0.5
)

plt.title("METR-LA Sensor Graph (207 Nodes)")
plt.savefig("data/processed/metr_la_graph.png", dpi=200, bbox_inches="tight")
plt.show()
