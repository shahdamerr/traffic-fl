import numpy as np
from scipy.sparse.csgraph import laplacian

def select_k_eigengap(adj, k_max=20):
    # Symmetrize
    adj = (adj + adj.T) / 2

    # Normalized Laplacian
    L = laplacian(adj, normed=True)

    # Eigenvalues
    eigvals = np.linalg.eigvalsh(L)

    # Compute eigengaps
    gaps = np.diff(eigvals[:k_max])
    k_opt = np.argmax(gaps) + 1

    return k_opt, eigvals[:k_max]
