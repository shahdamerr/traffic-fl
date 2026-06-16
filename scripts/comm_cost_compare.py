import numpy as np

files = {
    'Global FedAvg':        'results/fl_global_full_R50_S20_all.npz',
    'CFL flat (full)':      'results/fl_clustered_full_R50_S20_all.npz',
    'CFL + QW + Hier':      'results/fl_clustered_full_R50_S20_all_QW_H80e5.npz',
    'CFL adaptive':         'results/fl_clustered_adaptive_R50_S20_all.npz',
    'CFL random-drop':      'results/fl_clustered_random_drop_R50_S20_all.npz',
    'CFL+QW+K10+Hier(75R)': 'results/fl_clustered_full_R75_S20_all_QW_K10_H80e5.npz',
}

print(f"{'Method':<28} | {'MAE':>6} | {'Comm_MB':>9} | {'vs FedAvg':>10} | {'comm_saving':>12}")
print('-' * 78)

fedavg_comm = None
fedavg_mae = None
for name, f in files.items():
    d = np.load(f, allow_pickle=True)
    mae = float(d['overall_mae'])
    comm = float(d['total_bytes_mb'])
    if fedavg_comm is None:
        fedavg_comm = comm
        fedavg_mae = mae
    ratio = comm / fedavg_comm
    saving = (1 - comm / fedavg_comm) * 100
    print(f'{name:<28} | {mae:>6.3f} | {comm:>9.1f} | {ratio:>9.1f}x | {saving:>11.1f}%')
