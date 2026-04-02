"""
Dataset classes for anomaly detection training and inference.
- NormalFrameDataset: loads training frames (all normal) for CAE training.
- TestFrameDataset:   loads test frames with frame paths for inference.
"""

import os
import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


def get_train_transforms(resolution: list) -> transforms.Compose:
    h, w = resolution
    return transforms.Compose([
        transforms.Resize((h, w)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),               # [0, 255] → [0.0, 1.0]
        transforms.Normalize([0.5], [0.5]),  # → [-1.0, 1.0]
    ])


def get_inference_transforms(resolution: list) -> transforms.Compose:
    h, w = resolution
    return transforms.Compose([
        transforms.Resize((h, w)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])


class NormalFrameDataset(Dataset):
    """
    Loads all .jpg frames from training directory (normal frames only).
    Used for CAE training on Colab and validation split.

    Args:
        train_dir (str): path to training/frames/ — contains sequence subfolders.
        resolution (list): [H, W] target resolution.
        num_workers_hint (int): informational only — set in DataLoader, not here.
    """

    def __init__(self, train_dir: str, resolution: list):
        self.transform = get_train_transforms(resolution)
        self.frame_paths = sorted(
            glob.glob(os.path.join(train_dir, "**", "*.jpg"), recursive=True)
        )
        if len(self.frame_paths) == 0:
            raise FileNotFoundError(f"No .jpg frames found in {train_dir}")
        print(f"[NormalFrameDataset] {len(self.frame_paths):,} frames loaded from {train_dir}")

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        path = self.frame_paths[idx]
        img = Image.open(path).convert("RGB")
        tensor = self.transform(img)
        return tensor, path


class TestFrameDataset(Dataset):
    """
    Loads test frames in sorted order for frame-by-frame inference.
    Returns (tensor, path) so anomaly scores can be aligned to frame indices.

    Args:
        test_dir (str): path to testing/frames/ — contains sequence subfolders.
        resolution (list): [H, W] target resolution.
    """

    def __init__(self, test_dir: str, resolution: list):
        self.transform = get_inference_transforms(resolution)
        self.frame_paths = sorted(
            glob.glob(os.path.join(test_dir, "**", "*.jpg"), recursive=True)
        )
        if len(self.frame_paths) == 0:
            raise FileNotFoundError(f"No .jpg frames found in {test_dir}")
        print(f"[TestFrameDataset] {len(self.frame_paths):,} frames loaded from {test_dir}")

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        path = self.frame_paths[idx]
        img = Image.open(path).convert("RGB")
        tensor = self.transform(img)
        return tensor, path


if __name__ == "__main__":
    import yaml

    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    ds = NormalFrameDataset(
        train_dir=config["dataset"]["train_dir"],
        resolution=config["dataset"]["resolution"],
    )
    t, p = ds[0]
    print(f"Sample tensor shape : {t.shape}")   # expect [1, 256, 256]
    print(f"Sample tensor range : [{t.min():.3f}, {t.max():.3f}]")  # expect [-1, 1]
    print(f"Sample path         : {p}")