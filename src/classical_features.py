from __future__ import annotations

from typing import Any, List, Sequence, Tuple
import numpy as np


"""HELPER FUNCTIONS"""

def spm_feature_dim(K: int, levels: Sequence[int]) -> int:
    if K <= 0:
        raise ValueError("K must be positive.")
    if not levels or any(int(L) <= 0 for L in levels):
        raise ValueError("levels must be a non-empty sequence of positive ints.")
    return int(K) * sum(int(L) * int(L) for L in levels)


def l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def _validate_desc_array(d: np.ndarray, name: str = "descriptors") -> np.ndarray:
    d = np.asarray(d)
    if d.ndim != 2 or d.shape[1] != 128:
        raise ValueError(f"{name} must have shape (N, 128); got {d.shape}.")
    if d.dtype != np.float32:
        d = d.astype(np.float32, copy=False)
    return d


def sample_descriptors(
    descriptor_list: List[np.ndarray],
    sample_size: int = 100_000,
    seed: int = 42
) -> np.ndarray:
    """
    Randomly sample up to sample_size descriptors from the pooled descriptor_list.
    Memory-safe: does not concatenate unless total <= sample_size.
    Returns shape (N_sample, 128) float32.
    """
    if sample_size is None or sample_size <= 0:
        raise ValueError("sample_size must be a positive integer.")

    valid: List[np.ndarray] = []
    for d in descriptor_list:
        if d is None:
            continue
        d = np.asarray(d)
        if d.size == 0:
            continue
        d = _validate_desc_array(d, name="descriptors")
        valid.append(d)

    if not valid:
        return np.empty((0, 128), dtype=np.float32)

    total = sum(d.shape[0] for d in valid)
    if total <= sample_size:
        return np.vstack(valid).astype(np.float32, copy=False)

    rng = np.random.default_rng(seed)
    global_idx = rng.choice(total, size=sample_size, replace=False)
    global_idx.sort()

    out = np.empty((sample_size, 128), dtype=np.float32)
    write_pos = 0
    cursor = 0

    for d in valid:
        n = d.shape[0]
        start, end = cursor, cursor + n

        lo = np.searchsorted(global_idx, start, side="left")
        hi = np.searchsorted(global_idx, end, side="left")
        if hi > lo:
            local = global_idx[lo:hi] - start
            k = local.size
            out[write_pos:write_pos + k] = d[local]
            write_pos += k

        cursor = end
        if write_pos == sample_size:
            break

    return out[:write_pos]


"""MAIN APIS"""

def fit_codebook_from_descriptors(
    train_desc_list: List[np.ndarray],
    K: int = 2048,
    sample_size: int = 100_000,
    batch_size: int = 4096,
    random_state: int = 42,
) -> Any:
    """
    Fit a visual dictionary (codebook) using MiniBatchKMeans on descriptors.

    Parameters
    ----------
    train_desc_list:
        List of per-image descriptor arrays, each (N_i, 128) float-like.
        (Typically descriptors from Dense SIFT for the *fold-train* split.)
    K:
        Codebook size (#clusters).
    sample_size:
        Max number of descriptors to sample from train_desc_list for KMeans fitting.
    batch_size:
        MiniBatchKMeans batch size.
    random_state:
        Seed for KMeans.

    Returns
    -------
    kmeans:
        Fitted MiniBatchKMeans object with .cluster_centers_ and .predict().
    """
    from sklearn.cluster import MiniBatchKMeans

    if K <= 1:
        raise ValueError("K must be >= 2.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    X = sample_descriptors(train_desc_list, sample_size=sample_size, seed=random_state)
    if X.size == 0:
        raise ValueError("No descriptors available to fit codebook.")
    if X.shape[0] < K:
        raise ValueError(f"Need at least K descriptors to fit KMeans. Got N={X.shape[0]}, K={K}.")

    kmeans = MiniBatchKMeans(
        n_clusters=K,
        batch_size=batch_size,
        random_state=random_state,
        n_init="auto",
        reassignment_ratio=0.01,
    )
    kmeans.fit(X)
    return kmeans


def transform_with_codebook(
    desc_xy_hw_list: List[Tuple[np.ndarray, np.ndarray, Tuple[int, int]]],
    codebook: Any,
    levels: Sequence[int] = (1, 2, 4),
    knn: int = 5,
    beta: float = 1e-4,
    nonneg: bool = True,
    l2_normalize: bool = True,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Transform images into fixed-length feature vectors using:
      LLC encoding + Spatial Pyramid Matching (SPM) + Max pooling (+ optional L2 norm)

    Parameters
    ----------
    desc_xy_hw_list:
        List where each item is (descriptors, xy, (H, W)):
          - descriptors: (N, 128)
          - xy: (N, 2) with x,y coordinates in pixels (same resized image used for descriptors)
          - (H, W): height/width of that resized image
        This is typically output from your Dense SIFT extraction step (cached is fine).
    codebook:
        Fitted MiniBatchKMeans (or any object with .cluster_centers_).
    levels:
        Pyramid levels, e.g. (1,2,4).
    knn:
        Number of nearest codewords used per descriptor in LLC.
    beta:
        LLC regularization strength (small positive).
    nonneg:
        Whether to clamp LLC weights to non-negative and renormalize.
    l2_normalize:
        Whether to L2-normalize each final feature vector row.
    show_progress:
        If True, uses tqdm if available.

    Returns
    -------
    X:
        Array shape (n_images, K * sum(level^2)) float32.
    """
    if codebook is None or not hasattr(codebook, "cluster_centers_"):
        raise ValueError("codebook must be fitted and have .cluster_centers_.")
    centers = np.asarray(codebook.cluster_centers_, dtype=np.float32)
    if centers.ndim != 2:
        raise ValueError("codebook.cluster_centers_ must be 2D.")
    K = int(centers.shape[0])
    D = spm_feature_dim(K, levels)

    from llc import llc_encode
    from spm import spm_max_pool

    iterator = enumerate(desc_xy_hw_list)
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, total=len(desc_xy_hw_list), desc="LLC+SPM")
        except Exception:
            pass

    X_out = np.zeros((len(desc_xy_hw_list), D), dtype=np.float32)

    for i, (desc, xy, hw) in iterator:
        if desc is None:
            continue

        desc = np.asarray(desc)
        if desc.size == 0:
            continue  # keep zero row

        desc = _validate_desc_array(desc, name="descriptors")

        xy = np.asarray(xy, dtype=np.float32)
        if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] != desc.shape[0]:
            raise ValueError(f"xy must be (N,2) and match descriptors rows; got xy {xy.shape}, desc {desc.shape}.")

        H, W = int(hw[0]), int(hw[1])
        if H <= 0 or W <= 0:
            continue  # keep zero row

        codes = llc_encode(
            descriptors=desc,
            codebook=codebook,
            knn=knn,
            beta=beta,
            nonneg=nonneg,
        )  # (N, K)

        feat = spm_max_pool(
            codes=codes,
            xy=xy,
            image_hw=(H, W),
            levels=levels,
        )  # (D,)

        if feat.shape[0] != D:
            raise RuntimeError(f"SPM feature length mismatch: expected {D}, got {feat.shape[0]}.")

        X_out[i] = feat.astype(np.float32, copy=False)

    if l2_normalize:
        X_out = l2_normalize_rows(X_out)

    return X_out