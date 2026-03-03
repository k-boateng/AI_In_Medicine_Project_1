# src/llc.py
from __future__ import annotations
import numpy as np
from typing import Any
from sklearn.neighbors import NearestNeighbors

def llc_encode(
    descriptors: np.ndarray,
    codebook: Any,          # fitted kmeans or object with cluster_centers_
    knn: int = 5,
    beta: float = 1e-4,
    nonneg: bool = True,
) -> np.ndarray:
    """
    Encode descriptors with LLC relative to codebook centers.
    Returns codes shape (N, K) float32 (dense for simplicity; you can optimize later).
    """
    X = np.asarray(descriptors, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != 128:
        raise ValueError(f"Expected descriptors (N,128), got {X.shape}")

    centers = np.asarray(codebook.cluster_centers_, dtype=np.float32)
    K = centers.shape[0]
    if knn <= 0 or knn > K:
        raise ValueError("knn must be in [1, K].")

    # nearest neighbors in codebook space
    nn = NearestNeighbors(n_neighbors=knn, algorithm="auto").fit(centers)
    _, idx = nn.kneighbors(X, return_distance=True)  # (N, knn)

    N = X.shape[0]
    codes = np.zeros((N, K), dtype=np.float32)

    ones = np.ones((knn,), dtype=np.float32)

    for i in range(N):
        Ii = idx[i]                # knn indices
        B = centers[Ii]            # (knn, 128)
        x = X[i]                   # (128,)

        # local covariance
        Z = (B - x)                # (knn, 128)
        C = Z @ Z.T                # (knn, knn)
        C = C + (beta * np.trace(C) + 1e-12) * np.eye(knn, dtype=np.float32)

        w = np.linalg.solve(C, ones)   # (knn,)
        w = w / (np.sum(w) + 1e-12)

        if nonneg:
            w = np.maximum(w, 0.0)
            w = w / (np.sum(w) + 1e-12)

        codes[i, Ii] = w.astype(np.float32, copy=False)

    return codes