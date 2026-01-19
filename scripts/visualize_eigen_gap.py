import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt

from graph.select_clusters_size import select_k_eigengap

def main():
    data = np.load("data/processed/metr_la_processed.npz")
    adj = data["adj"]
    adj = (adj + adj.T) / 2

    k_opt, eigvals = select_k_eigengap(adj, k_max=20)

    gaps = np.diff(eigvals)

    plt.figure(figsize=(9, 4))
    plt.plot(range(1, len(eigvals) + 1), eigvals, marker="o")
    plt.axvline(k_opt, linestyle="--")
    plt.title(f"Normalized Laplacian Eigenvalues (k_opt = {k_opt})")
    plt.xlabel("Index k")
    plt.ylabel("Eigenvalue λ_k")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("data/processed/eigvals_plot.png", dpi=200)
    plt.show()

    plt.figure(figsize=(9, 4))
    plt.plot(range(1, len(gaps) + 1), gaps, marker="o")
    plt.axvline(k_opt, linestyle="--")
    plt.title("Eigengap Δλ_k = λ_(k+1) - λ_k")
    plt.xlabel("k")
    plt.ylabel("Eigengap")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("data/processed/eigengap_plot.png", dpi=200)
    plt.show()

    print("Saved:")
    print(" - data/processed/eigvals_plot.png")
    print(" - data/processed/eigengap_plot.png")

if __name__ == "__main__":
    main()
