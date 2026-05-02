"""Check OSM road types for METR-LA sensors using osmnx."""
import pandas as pd
import osmnx as ox
from collections import Counter

df = pd.read_csv("data/raw/graph_sensor_locations.csv")

# Get the road network around METR-LA area
min_lat = df["latitude"].min() - 0.01
max_lat = df["latitude"].max() + 0.01
min_lon = df["longitude"].min() - 0.01
max_lon = df["longitude"].max() + 0.01

print(f"Bounding box: ({min_lat:.3f}, {min_lon:.3f}) to ({max_lat:.3f}, {max_lon:.3f})")
print("Downloading road network (this may take a moment)...")

G = ox.graph_from_bbox(bbox=(max_lat, min_lat, max_lon, min_lon), network_type="drive")
edges = ox.graph_to_gdfs(G, nodes=False)

print(f"Downloaded {len(edges)} road segments\n")

# For each sensor, find the nearest road and get its type
road_types = []
for i, row in df.iterrows():
    lat, lon = row["latitude"], row["longitude"]
    try:
        nearest_edge = ox.nearest_edges(G, lon, lat)
        u, v, key = nearest_edge
        edge_data = G[u][v][key]
        hw = edge_data.get("highway", "unknown")
        # highway can be a list
        if isinstance(hw, list):
            hw = hw[0]
        road_types.append(hw)
        if i < 15:
            name = edge_data.get("name", "unnamed")
            print(f"Sensor {i:3d}: {hw:20s} | {name}")
    except Exception as e:
        road_types.append("error")
        if i < 15:
            print(f"Sensor {i:3d}: ERROR - {e}")

print(f"\n--- Road Type Distribution (all 207 sensors) ---")
for rtype, count in Counter(road_types).most_common():
    print(f"  {rtype:20s}: {count:3d} sensors ({count/len(road_types)*100:.1f}%)")
