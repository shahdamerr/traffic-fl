import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

# Load cluster labels
clusters_npz = np.load("data/processed/graph_clusters.npz")
labels = clusters_npz["cluster_labels"]
num_clusters = int(clusters_npz["num_clusters"])

# Load sensor locations
locs = pd.read_csv("data/raw/graph_sensor_locations.csv")

# Sanity check
assert len(locs) == len(labels), "Mismatch between sensors and cluster labels"

# Assign cluster to each sensor
locs["cluster"] = labels

# Plot
plt.figure(figsize=(10, 8))
cmap = cm.get_cmap("tab10", num_clusters)

for k in range(num_clusters):
    sub = locs[locs["cluster"] == k]
    plt.scatter(
        sub["longitude"],
        sub["latitude"],
        s=30,
        color=cmap(k),
        label=f"Cluster {k}",
        alpha=0.8
    )

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Geographical Visualization of METR-LA Sensor Clusters")
plt.legend(markerscale=1.5, fontsize=8)
plt.grid(True)
plt.savefig("data/processed/metr_la_geo_clusters.png", dpi=200, bbox_inches="tight")
plt.show()
