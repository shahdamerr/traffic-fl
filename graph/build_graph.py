import networkx as nx
import numpy as np

def build_graph_from_adj(adj, threshold=0.0):
    """
    Build an undirected graph from adjacency matrix
    """
    G = nx.Graph()  # Initialize an undirected graph Because spatial proximity / road topology is usually treated as bidirectional connectivity for message exchange.

    num_nodes = adj.shape[0]
    G.add_nodes_from(range(num_nodes))

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adj[i, j] > threshold:
                G.add_edge(i, j, weight=adj[i, j])

    return G
