from typing import Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from torchvision import models


def build_resnet18(num_classes: int,
                   pretrained: bool = True,
                   freeze_backbone: bool = True
                   ) -> nn.Module:
    """
    Build a ResNet-18 model with final fc replaced for `num_classes`.
    If freeze_backbone is True, all parameters except the final fc will have requires_grad=False.
    """
    model = models.resnet18(pretrained=pretrained)
    in_feats = model.fc.in_features  # typically 512 for ResNet-18
    model.fc = nn.Linear(in_feats, num_classes)

    if freeze_backbone:
        # Freeze everything except final FC
        for name, param in model.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False
            else:
                param.requires_grad = True

    return model


class ResNet18FeatureExtractor(nn.Module):
    """
    ResNet-18 that returns the 512-d feature vector after global avg pool (before FC).
    Useful for extracting features with a frozen backbone.
    """
    def __init__(self, pretrained: bool = True, freeze_backbone: bool = True):
        super().__init__()
        base = models.resnet18(pretrained=pretrained)
        # Keep all layers except the final fc
        # We'll call base up to avgpool, then flatten
        self.backbone = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
            base.avgpool
        )
        # Optionally freeze
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: x shape (B,3,H,W)
        Output: features shape (B, 512) -- flattened
        """
        x = self.backbone(x)              # (B, 512, 1, 1)
        x = torch.flatten(x, 1)          # (B, 512)
        return x


def build_resnet18_feature_extractor(pretrained: bool = True, freeze_backbone: bool = True) -> ResNet18FeatureExtractor:
    """
    Convenience constructor for the feature extractor module.
    """
    return ResNet18FeatureExtractor(pretrained=pretrained, freeze_backbone=freeze_backbone)


def extract_features(model: torch.nn.Module,
                     dataloader: DataLoader,
                     device: Optional[torch.device] = None,
                     return_labels: bool = True
                     ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Run `model` over `dataloader` and collect features and labels.

    - model should return a feature tensor of shape (B, D)
      (i.e., use build_resnet18_feature_extractor or similar).
    - device: torch.device (if None, uses CUDA if available)
    - returns: (features, labels) where features is (N, D) np.float32 and
      labels is (N,) np.int64 (or None if return_labels=False)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    feats_list = []
    labels_list = []

    with torch.no_grad():
        for batch in dataloader:
            # Accept two common batch formats:
            # (inputs, labels) or dict-like.
            if isinstance(batch, (list, tuple)):
                inputs, labels = batch[0], batch[1]
            else:
                inputs = batch["image"]
                labels = batch["label"]

            inputs = inputs.to(device)
            features = model(inputs)  # (B, D)
            feats_list.append(features.cpu().numpy())
            if return_labels:
                labels_list.append(labels.numpy() if isinstance(labels, np.ndarray) else labels.cpu().numpy())

    features = np.vstack(feats_list).astype(np.float32)
    labels_arr = None
    if return_labels:
        labels_arr = np.concatenate(labels_list).astype(np.int64)

    return features, labels_arr