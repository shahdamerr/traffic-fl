import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm

from graph.build_graph import build_graph_from_adj

# Load data
proc = np.load("data/processed/metr_la_processed.npz")
adj = proc["adj"]

clusters_npz = np.load("data/processed/graph_clusters.npz")
labels = clusters_npz["cluster_labels"]
num_clusters = int(clusters_npz["num_clusters"])

# Build graph
G = build_graph_from_adj(adj)

# Layout
pos = nx.spring_layout(G, seed=42)

# Color map
cmap = cm.get_cmap("tab10", num_clusters)
node_colors = [cmap(labels[i]) for i in G.nodes()]

plt.figure(figsize=(10, 10))

nx.draw(
    G,
    pos,
    node_color=node_colors,
    node_size=25,
    alpha=0.8,
    edge_color="lightgray",
    width=0.4
)

plt.title("METR-LA Graph Clustered into Subgraphs")
plt.show()
