from pathlib import Path
from typing import Sequence, Callable, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset


class ImagePathDataset(Dataset):
    """
    PyTorch Dataset that loads images from a list of file paths and returns
    (image_tensor, label_int). Images are opened with PIL and converted to RGB;
    torchvision transforms should be passed in via `transform`.
    """

    def __init__(
        self,
        paths: Sequence[str],
        labels: Sequence[int],
        transform: Optional[Callable] = None,
    ):
        if len(paths) != len(labels):
            raise ValueError("paths and labels must have the same length")
        self.paths = [str(Path(p)) for p in paths]
        self.labels = [int(l) for l in labels]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        label = self.labels[idx]

        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to open image at {path}: {e}")

        if self.transform is not None:
            img = self.transform(img)

        # Return: (image_tensor, label_int)
        return img, label

    def __repr__(self) -> str:
        return f"ImagePathDataset(num_samples={len(self)}, transform={self.transform is not None})"