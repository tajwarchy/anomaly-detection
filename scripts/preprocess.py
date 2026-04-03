"""
Dataset classes for anomaly detection training and inference.
- NormalFrameDataset: loads training frames (normal only) for CAE training.
  Supports single-channel (grayscale) and dual-channel (grayscale + flow).
- TestFrameDataset:   loads test frames with frame paths for inference.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


def get_transform(resolution: list) -> transforms.Compose:
    h, w = resolution
    return transforms.Compose([
        transforms.Resize((h, w)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),               # [0,255] → [0.0,1.0]
        transforms.Normalize([0.5], [0.5]),  # → [-1.0,1.0]
    ])


def flow_npy_to_tensor(flow_mag: np.ndarray) -> torch.Tensor:
    """HxW float32 [0,1] flow map → [1,H,W] normalized tensor [-1,1]."""
    t = torch.from_numpy(flow_mag).unsqueeze(0)
    return (t - 0.5) / 0.5


class NormalFrameDataset(Dataset):
    """
    Loads all .jpg frames from training directory (normal frames only).

    When flow_enabled=True, also loads pre-extracted .npy flow maps
    from a sibling flow/ directory and returns 2-channel tensors.

    Args:
        train_dir    : path to combined_normal/frames/ — sequence subfolders
        resolution   : [H, W]
        flow_enabled : if True, load dual-channel (grayscale + flow) inputs
    """

    def __init__(self, train_dir: str, resolution: list, flow_enabled: bool = False):
        self.transform    = get_transform(resolution)
        self.flow_enabled = flow_enabled
        self.flow_dir     = train_dir.replace("frames", "flow") if flow_enabled else None

        self.frame_paths = sorted(
            glob.glob(os.path.join(train_dir, "**", "*.jpg"), recursive=True)
        )
        if len(self.frame_paths) == 0:
            raise FileNotFoundError(f"No .jpg frames found in {train_dir}")

        print(f"[NormalFrameDataset] {len(self.frame_paths):,} frames | "
              f"flow={'ON' if flow_enabled else 'OFF'}")

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        path   = self.frame_paths[idx]
        img    = Image.open(path).convert("RGB")
        gray_t = self.transform(img)           # [1, H, W]

        if self.flow_enabled:
            # Load pre-extracted flow map — same relative path, .npy extension
            flow_path = path.replace(
                os.sep + "frames" + os.sep,
                os.sep + "flow" + os.sep
            ).replace(".jpg", ".npy")

            if os.path.exists(flow_path):
                flow_mag = np.load(flow_path).astype(np.float32)
            else:
                h, w     = gray_t.shape[1], gray_t.shape[2]
                flow_mag = np.zeros((h, w), dtype=np.float32)

            flow_t = flow_npy_to_tensor(flow_mag)        # [1, H, W]
            tensor = torch.cat([gray_t, flow_t], dim=0)  # [2, H, W]
        else:
            tensor = gray_t                              # [1, H, W]

        return tensor, path


class TestFrameDataset(Dataset):
    """
    Loads test frames in sorted order for frame-by-frame inference.
    Returns (tensor, path) for alignment with anomaly score indices.
    """

    def __init__(self, test_dir: str, resolution: list):
        self.transform   = get_transform(resolution)
        self.frame_paths = sorted(
            glob.glob(os.path.join(test_dir, "**", "*.jpg"), recursive=True)
        )
        if len(self.frame_paths) == 0:
            raise FileNotFoundError(f"No .jpg frames found in {test_dir}")
        print(f"[TestFrameDataset] {len(self.frame_paths):,} frames loaded.")

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        path   = self.frame_paths[idx]
        img    = Image.open(path).convert("RGB")
        tensor = self.transform(img)
        return tensor, path


if __name__ == "__main__":
    import yaml
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    flow_enabled = config["flow"]["enabled"]

    ds = NormalFrameDataset(
        train_dir=config["dataset"]["train_dir"],
        resolution=config["dataset"]["resolution"],
        flow_enabled=flow_enabled,
    )
    t, p = ds[0]
    print(f"Tensor shape : {t.shape}")   # [1,256,256] or [2,256,256]
    print(f"Tensor range : [{t.min():.3f}, {t.max():.3f}]")
    print(f"Path         : {p}")