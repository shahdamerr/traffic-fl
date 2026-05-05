"""Check OSM road types for METR-LA sensors — per-sensor nearest edge lookup."""
import pandas as pd
import osmnx as ox
from collections import Counter

df = pd.read_csv("data/raw/graph_sensor_locations.csv")

# Query a small area around each sensor (much faster than full bbox)
road_types = []
print("Looking up nearest road for each sensor...")

for i, row in df.iterrows():
    lat, lon = row["latitude"], row["longitude"]
    try:
        # Small graph around each sensor (200m radius)
        G = ox.graph_from_point((lat, lon), dist=200, network_type="drive")
        nearest = ox.nearest_edges(G, lon, lat)
        u, v, key = nearest
        edge_data = G[u][v][key]
        hw = edge_data.get("highway", "unknown")
        if isinstance(hw, list):
            hw = hw[0]
        road_types.append(hw)
        if i < 15 or i % 50 == 0:
            name = edge_data.get("name", "unnamed")
            lanes = edge_data.get("lanes", "?")
            print(f"  Sensor {i:3d}: {hw:20s} | lanes={str(lanes):4s} | {name}")
    except Exception as e:
        road_types.append("unknown")
        if i < 15:
            print(f"  Sensor {i:3d}: ERROR - {e}")

print(f"\n{'='*50}")
print(f"Road Type Distribution (all {len(road_types)} sensors)")
print(f"{'='*50}")
for rtype, count in Counter(road_types).most_common():
    print(f"  {rtype:20s}: {count:3d} sensors ({count/len(road_types)*100:.1f}%)")
