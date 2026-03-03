from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from torch_dataset import ImagePathDataset


@dataclass(frozen=True)
class EffNetConfig:
    img_size: int = 224
    batch_size: int = 64
    num_workers: int = 4

    unfreeze_last_n: int = 2  # unfreeze last N modules in model.features
    lr_head: float = 3e-4
    lr_ft: float = 1e-4
    weight_decay: float = 1e-4

    use_weighted_sampler: bool = True
    seed: int = 42


def set_torch_perf_flags() -> None:
    """
    Enable common speed settings for fixed-size CNN inputs.
    Safe for 224x224 image classification.
    """
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights

def _get_mean_std(weights: EfficientNet_B0_Weights):
    #weights.meta
    meta = getattr(weights, "meta", None)
    if isinstance(meta, dict) and ("mean" in meta) and ("std" in meta):
        return meta["mean"], meta["std"]

    #Try to find Normalize inside weights.transforms()
    try:
        pre = weights.transforms()
        for t in getattr(pre, "transforms", []):
            if isinstance(t, transforms.Normalize):
                return t.mean, t.std
    except Exception:
        pass

    # Fallback
    return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def make_transforms(
    img_size: int = 224,
    weights: EfficientNet_B0_Weights = EfficientNet_B0_Weights.DEFAULT,
):
    mean, std = _get_mean_std(weights)

    train_tf = transforms.Compose([
        transforms.Resize(256),  # Keep aspect ratio
        transforms.RandomCrop(img_size), # Gentler crop
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_tf, eval_tf


def build_efficientnet_b0(
    num_classes: int,
    unfreeze_last_n: int = 2,
    weights: EfficientNet_B0_Weights = EfficientNet_B0_Weights.DEFAULT,
    device: str | torch.device = "cuda",
) -> nn.Module:
    """
    Build EfficientNet-B0 with pretrained weights, replace classifier head,
    freeze everything, then unfreeze classifier + last N feature modules.
    """
    if num_classes <= 1:
        raise ValueError("num_classes must be >= 2.")
    if unfreeze_last_n < 0:
        raise ValueError("unfreeze_last_n must be >= 0.")

    model = efficientnet_b0(weights=weights)

    # Replace head
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    # Freeze all
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze classifier
    for p in model.classifier.parameters():
        p.requires_grad = True

    # Unfreeze last N blocks in features
    if unfreeze_last_n > 0:
        for block in model.features[-unfreeze_last_n:]:
            for p in block.parameters():
                p.requires_grad = True

    return model.to(device)


def make_optimizer(
    model: nn.Module,
    unfreeze_last_n: int,
    lr_head: float,
    lr_ft: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """
    Two LR groups: classifier and last N feature blocks.
    """
    params = [
        {"params": model.classifier.parameters(), "lr": lr_head},
    ]
    if unfreeze_last_n > 0:
        params.append({"params": model.features[-unfreeze_last_n:].parameters(), "lr": lr_ft})

    return torch.optim.AdamW(params, weight_decay=weight_decay)


def make_weighted_sampler(
    y: Sequence[int],
    num_classes: Optional[int] = None,
) -> WeightedRandomSampler:
    """
    WeightedRandomSampler to upsample minority classes within a training fold.
    Pass fold-train labels y.
    """
    y = np.asarray(y, dtype=np.int64)
    if num_classes is None:
        num_classes = int(y.max()) + 1

    counts = np.bincount(y, minlength=num_classes)
    class_w = 1.0 / np.maximum(counts, 1)
    sample_w = class_w[y]

    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_w, dtype=torch.double),
        num_samples=len(sample_w),
        replacement=True,
    )


def make_loaders_for_fold(
    train_paths: Sequence[str],
    y_train: Sequence[int],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    train_tf,
    eval_tf,
    batch_size: int,
    num_workers: int,
    use_weighted_sampler: bool,
    num_classes: int,
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates (train_loader, val_loader) for a fold using indices into train_paths/y_train.
    """
    tr_paths = [str(train_paths[i]) for i in train_idx]
    tr_y = [int(y_train[i]) for i in train_idx]
    va_paths = [str(train_paths[i]) for i in val_idx]
    va_y = [int(y_train[i]) for i in val_idx]

    train_ds = ImagePathDataset(tr_paths, tr_y, transform=train_tf)
    val_ds = ImagePathDataset(va_paths, va_y, transform=eval_tf)

    sampler = None
    shuffle = True
    if use_weighted_sampler:
        sampler = make_weighted_sampler(tr_y, num_classes=num_classes)
        shuffle = False  # sampler and shuffle are mutually exclusive

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader


@torch.no_grad()
def predict_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: str | torch.device = "cuda",
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_y = []
    all_pred = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1).detach().cpu().numpy()
        all_pred.append(pred)
        all_y.append(np.asarray(y))

    return np.concatenate(all_y), np.concatenate(all_pred)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    'Straight' F1 for multiclass single-label = micro F1.
    """
    from sklearn.metrics import accuracy_score, f1_score

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    micro_f1 = float(f1_score(y_true, y_pred, average="micro"))
    return {"accuracy": acc, "micro_f1": micro_f1}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: str | torch.device = "cuda",
) -> float:
    model.train()

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    total_loss = 0.0
    n = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = torch.as_tensor(y, device=device)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(str(device).startswith("cuda"))):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = x.size(0)
        total_loss += float(loss.detach().cpu()) * bs
        n += bs

    return total_loss / max(n, 1)