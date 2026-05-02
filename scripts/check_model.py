import numpy as np, os

files = {
    "Global FedAvg (30R)":  "results/fl_global_full_R30_E1_dev.npz",
    "CFL Full   (30R)":     "results/fl_clustered_full_R30_E1_dev.npz",
}
for name, f in files.items():
    if os.path.exists(f):
        d = np.load(f, allow_pickle=True)
        h15  = float(d["h15_mae"])
        h30  = float(d["h30_mae"])
        h60  = float(d["h60_mae"])
        mae  = float(d["overall_mae"])
        comm = float(d["total_bytes_mb"])
        mins = float(d["total_time_min"])
        print(f"{name}:")
        print(f"  15min={h15:.3f}  30min={h30:.3f}  60min={h60:.3f}  overall={mae:.3f}")
        print(f"  comm={comm:.1f} MB  time={mins:.1f} min")
    else:
        print(f"{name}: NOT FOUND")
