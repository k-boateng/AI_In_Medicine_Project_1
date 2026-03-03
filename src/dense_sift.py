from __future__ import annotations
from typing import List, Tuple, Sequence
import numpy as np
import cv2

def extract_dense_sift(
    image_paths: List[str],
    step: int = 4,
    sizes: Sequence[int] = (16,),
    max_image_side: int | None = 300,
    resize_interpolation: int = cv2.INTER_AREA,
    show_progress: bool = False,
) -> List[Tuple[np.ndarray, np.ndarray, Tuple[int, int]]]:
    """
    Returns list of (descriptors, xy, (H, W)) per image.
      - descriptors: (N, 128) float32
      - xy: (N, 2) float32 in pixel coords (x, y) in resized image
      - (H, W): resized image shape for SPM binning
    """
    if step <= 0:
        raise ValueError("step must be positive.")
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError("cv2.SIFT_create not available. Install opencv-contrib-python.")
    sift = cv2.SIFT_create()

    results = []

    iterator = enumerate(image_paths)
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, total=len(image_paths), desc="Dense SIFT")
        except Exception:
            pass

    for _, path in iterator:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            results.append((np.empty((0,128), np.float32), np.empty((0,2), np.float32), (0,0)))
            continue

        # optional resize
        if max_image_side is not None:
            h, w = img.shape[:2]
            max_side = max(h, w)
            if max_side > max_image_side:
                scale = max_image_side / float(max_side)
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
                img = cv2.resize(img, (new_w, new_h), interpolation=resize_interpolation)

        H, W = img.shape[:2]

        # build dense grid keypoints at one or multiple sizes
        keypoints = []
        for s in sizes:
            for y in range(0, H, step):
                for x in range(0, W, step):
                    keypoints.append(cv2.KeyPoint(float(x), float(y), float(s)))

        if len(keypoints) == 0:
            results.append((np.empty((0,128), np.float32), np.empty((0,2), np.float32), (H,W)))
            continue

        keypoints, descriptors = sift.compute(img, keypoints)
        if descriptors is None or len(keypoints) == 0:
            results.append((np.empty((0,128), np.float32), np.empty((0,2), np.float32), (H,W)))
            continue

        xy = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints], dtype=np.float32)
        descriptors = descriptors.astype(np.float32, copy=False)

        results.append((descriptors, xy, (H, W)))

    return results