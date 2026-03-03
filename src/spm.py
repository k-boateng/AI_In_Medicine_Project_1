# src/spm.py
from __future__ import annotations
import numpy as np
from typing import Sequence, Tuple

def spm_max_pool(
    codes: np.ndarray,         # (N, K)
    xy: np.ndarray,            # (N, 2) x,y
    image_hw: Tuple[int, int], # (H, W)
    levels: Sequence[int] = (1, 2, 4),
) -> np.ndarray:
    """
    Return pooled feature vector of shape (K * sum(L^2),).
    """
    C = np.asarray(codes, dtype=np.float32)
    P = np.asarray(xy, dtype=np.float32)
    H, W = image_hw

    if C.ndim != 2:
        raise ValueError("codes must be (N,K).")
    if P.shape[0] != C.shape[0] or P.shape[1] != 2:
        raise ValueError("xy must be (N,2) and match codes rows.")
    if H <= 0 or W <= 0:
        return np.zeros((C.shape[1] * sum(l*l for l in levels),), dtype=np.float32)

    K = C.shape[1]
    feats = []

    x = P[:, 0]
    y = P[:, 1]

    for L in levels:
        # bin indices in [0, L-1]
        bx = np.clip((x / W * L).astype(int), 0, L - 1)
        by = np.clip((y / H * L).astype(int), 0, L - 1)

        for j in range(L):
            for i in range(L):
                mask = (bx == i) & (by == j)
                if not np.any(mask):
                    feats.append(np.zeros((K,), dtype=np.float32))
                else:
                    feats.append(np.max(C[mask], axis=0))

    return np.concatenate(feats, axis=0).astype(np.float32, copy=False)