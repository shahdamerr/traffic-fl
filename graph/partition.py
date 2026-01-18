import numpy as np
from sklearn.cluster import SpectralClustering

def spectral_partition(adj, num_clusters):
    """
    Partition graph using spectral clustering
    """
    clustering = SpectralClustering(
        n_clusters=num_clusters,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=42
    )

    cluster_labels = clustering.fit_predict(adj)

    clusters = {}
    for node, cid in enumerate(cluster_labels):
        clusters.setdefault(cid, []).append(node)

    return cluster_labels, clusters
