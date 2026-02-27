# src/sift_bovw.py
from __future__ import annotations

from pathlib import Path
import numpy as np
from typing import List, Tuple, Any
import cv2
import json
import pickle


def extract_sift_descriptors(
    image_paths: List[str],
    max_keypoints: int = 500,
    show_progress: bool = False,
    max_image_side: int | None = 300,
    resize_interpolation: int = cv2.INTER_AREA,
) -> List[np.ndarray]:
    """
    For each path in image_paths: detect SIFT keypoints and compute descriptors.
    Return a list of descriptor arrays (one per image). If an image has no descriptors,
    return an empty array for that image.

    Notes:
      - Reads grayscale.
      - Optionally resizes so max(H, W) <= max_image_side (keeps aspect ratio).
      - Optionally keeps top `max_keypoints` by keypoint response.
    """
    if max_keypoints is not None and max_keypoints <= 0:
        raise ValueError("max_keypoints must be a positive integer or None.")
    if max_image_side is not None and max_image_side <= 0:
        raise ValueError("max_image_side must be a positive integer or None.")

    # Create SIFT once (reuse across images)
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError(
            "cv2.SIFT_create is not available. Install opencv-contrib-python."
        )
    sift = cv2.SIFT_create()

    results: List[np.ndarray] = []

    iterator = enumerate(image_paths)
    if show_progress:
        try:
            from tqdm import tqdm  # optional dependency
            iterator = tqdm(iterator, total=len(image_paths), desc="SIFT")
        except Exception:
            pass

    for _, path in iterator:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            results.append(np.empty((0, 128), dtype=np.float32))
            continue

        # Optional resize to control runtime / descriptor counts
        if max_image_side is not None:
            h, w = img.shape[:2]
            max_side = max(h, w)
            if max_side > max_image_side:
                scale = max_image_side / float(max_side)
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
                img = cv2.resize(img, (new_w, new_h), interpolation=resize_interpolation)

        keypoints, descriptors = sift.detectAndCompute(img, None)

        if descriptors is None or len(keypoints) == 0:
            results.append(np.empty((0, 128), dtype=np.float32))
            continue

        if max_keypoints is not None and len(keypoints) > max_keypoints:
            idx = np.argsort([-kp.response for kp in keypoints])[:max_keypoints]
            descriptors = descriptors[idx]

        if descriptors.dtype != np.float32:
            descriptors = descriptors.astype(np.float32, copy=False)

        results.append(descriptors)

    return results


def sample_descriptors(
    descriptor_list: List[np.ndarray],
    sample_size: int = 100000,
    seed: int = 42
) -> np.ndarray:
    """
    Randomly sample up to sample_size descriptors from the pooled descriptor_list.
    Return shape (N_sample, 128).

    Implementation notes:
      - Concatenate descriptors from all images into one big array (or sample per-image
        to avoid memory spike).
      - If total descriptors <= sample_size, return all.
    """
    if sample_size is None or sample_size <= 0:
        raise ValueError("sample_size must be a positive integer.")

    # Filter out empties / None
    valid = []
    for d in descriptor_list:
        if d is None:
            continue
        d = np.asarray(d)
        if d.size == 0:
            continue
        if d.ndim != 2 or d.shape[1] != 128:
            raise ValueError(f"Expected descriptors of shape (N, 128); got {d.shape}.")
        if d.dtype != np.float32:
            d = d.astype(np.float32, copy=False)
        valid.append(d)

    if not valid:
        return np.empty((0, 128), dtype=np.float32)

    # Total number of descriptors available
    total = sum(d.shape[0] for d in valid)

    # If we don't have more than sample_size, just return everything (concatenated)
    if total <= sample_size:
        return np.vstack(valid).astype(np.float32, copy=False)

    rng = np.random.default_rng(seed)

    # Memory-safe global sampling without concatenating:
    # pick sample_size global indices in [0, total)
    global_idx = rng.choice(total, size=sample_size, replace=False)
    global_idx.sort()  # makes the scan deterministic and efficient

    out = np.empty((sample_size, 128), dtype=np.float32)

    write_pos = 0
    cursor = 0  # start offset of current image in the pooled index space

    # Fill `out` by scanning images and taking indices that fall in each image's range
    for d in valid:
        n = d.shape[0]
        start = cursor
        end = cursor + n

        # Find which sampled indices lie within [start, end)
        lo = np.searchsorted(global_idx, start, side="left")
        hi = np.searchsorted(global_idx, end, side="left")
        if hi > lo:
            local = global_idx[lo:hi] - start  # convert to per-image indices
            k = local.size
            out[write_pos:write_pos + k] = d[local]
            write_pos += k

        cursor = end

        if write_pos == sample_size:
            break

    # Should be exact, but keep a safety slice
    return out[:write_pos]


def build_vocabulary(
    sampled_descriptors: np.ndarray,
    K: int = 256,
    batch_size: int = 4096,
    random_state: int = 42
) -> Any:
    """
    Run MiniBatchKMeans on sampled_descriptors and return the fitted kmeans object.
    Implementation notes:
      - Use sklearn.cluster.MiniBatchKMeans
      - Save cluster centers or full kmeans for later `predict` calls.
    """
    from sklearn.cluster import MiniBatchKMeans

    if sampled_descriptors is None:
        raise ValueError("sampled_descriptors is None.")
    X = np.asarray(sampled_descriptors)

    if X.size == 0:
        raise ValueError("sampled_descriptors is empty; cannot build a vocabulary.")
    if X.ndim != 2 or X.shape[1] != 128:
        raise ValueError(f"Expected sampled_descriptors of shape (N, 128); got {X.shape}.")
    if K <= 1:
        raise ValueError("K must be >= 2.")
    if X.shape[0] < K:
        raise ValueError(f"Need at least K descriptors to fit KMeans. Got N={X.shape[0]}, K={K}.")
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    # Ensure float32 (faster / less memory; sklearn can handle float32)
    if X.dtype != np.float32:
        X = X.astype(np.float32, copy=False)

    kmeans = MiniBatchKMeans(
        n_clusters=K,
        batch_size=batch_size,
        random_state=random_state,
        n_init="auto",          # modern sklearn; falls back appropriately depending on version
        reassignment_ratio=0.01 # common default-ish; helps stability for minibatches
    )
    kmeans.fit(X)
    return kmeans



def compute_bovw_histograms(
    descriptor_list: List[np.ndarray],
    kmeans: Any,
    K: int = 256
) -> np.ndarray:
    """
    For each image's descriptors, assign each descriptor to nearest cluster center
    (kmeans.predict or nearest neighbor) and build a histogram of length K.
    Return histograms array shape (n_images, K).

    Implementation notes:
      - If an image has zero descriptors, use a zero-vector histogram.
      - Normalize rows (L2 or L1) before returning or save unnormalized and store scaler params.
    """
    if K <= 0:
        raise ValueError("K must be a positive integer.")
    if kmeans is None or not hasattr(kmeans, "predict"):
        raise ValueError("kmeans must be a fitted object with a .predict method.")

    n_images = len(descriptor_list)
    hists = np.zeros((n_images, K), dtype=np.float32)

    for i, desc in enumerate(descriptor_list):
        if desc is None:
            continue
        desc = np.asarray(desc)

        # Empty / no descriptors => keep zero histogram
        if desc.size == 0:
            continue

        if desc.ndim != 2 or desc.shape[1] != 128:
            raise ValueError(f"Expected descriptors of shape (N, 128); got {desc.shape} at index {i}.")

        if desc.dtype != np.float32:
            desc = desc.astype(np.float32, copy=False)

        # Assign descriptors to visual words
        labels = kmeans.predict(desc)

        # Build histogram
        # minlength ensures length K even if some bins are missing
        hist = np.bincount(labels, minlength=K).astype(np.float32, copy=False)
        hists[i] = hist

    # L2-normalize each histogram (common for BoVW)
    norms = np.linalg.norm(hists, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)  # avoid divide-by-zero for all-zero rows
    hists = hists / norms

    return hists



def save_vocabulary_and_histograms(
    out_dir: str | Path,
    kmeans: Any,
    histograms: np.ndarray,
    labels: np.ndarray,
    meta: dict
) -> None:
    """
    Save kmeans (pickle), histograms (npy), labels (npy), and meta (pkl/json).
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Basic validation / normalization
    H = np.asarray(histograms)
    y = np.asarray(labels)

    if H.ndim != 2:
        raise ValueError(f"histograms must be 2D (n_images, K). Got {H.shape}.")
    if y.ndim != 1:
        raise ValueError(f"labels must be 1D (n_images,). Got {y.shape}.")
    if H.shape[0] != y.shape[0]:
        raise ValueError(f"histograms and labels must have same n_images. Got {H.shape[0]} vs {y.shape[0]}.")
    if not isinstance(meta, dict):
        raise ValueError("meta must be a dict.")

    # Filenames
    kmeans_fp = out_path / "kmeans.pkl"
    hist_fp = out_path / "histograms.npy"
    labels_fp = out_path / "labels.npy"
    meta_pkl_fp = out_path / "meta.pkl"
    meta_json_fp = out_path / "meta.json"

    # Save kmeans
    with open(kmeans_fp, "wb") as f:
        pickle.dump(kmeans, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save arrays
    np.save(hist_fp, H.astype(np.float32, copy=False))
    np.save(labels_fp, y)

    # Save meta as pickle (handles arbitrary objects)
    with open(meta_pkl_fp, "wb") as f:
        pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Also attempt meta as JSON (nice for inspection)
    # If meta isn't JSON-serializable, fallback to a sanitized version.
    def _jsonable(obj):
        try:
            json.dumps(obj)
            return obj
        except TypeError:
            # best-effort conversion for common types
            if isinstance(obj, (Path,)):
                return str(obj)
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            if isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            if isinstance(obj, (set, tuple)):
                return list(obj)
            return str(obj)

    meta_for_json = {str(k): _jsonable(v) for k, v in meta.items()}
    with open(meta_json_fp, "w", encoding="utf-8") as f:
        json.dump(meta_for_json, f, indent=2, sort_keys=True)