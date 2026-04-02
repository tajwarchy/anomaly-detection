"""
Optical flow extraction utilities.
Supports Farnebäck (CPU, OpenCV) and optionally RAFT (MPS).
Flow magnitude is normalized to [0, 1] and used as a second input channel.
Toggled via config: flow.enabled and flow.method.
"""

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
#  Farnebäck Flow (CPU — fast, no model weights required)
# ─────────────────────────────────────────────────────────────────────────────

def compute_farneback_flow(prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
    """
    Compute dense optical flow between two grayscale frames using Farnebäck.

    Args:
        prev_gray: HxW uint8 grayscale frame (previous)
        curr_gray: HxW uint8 grayscale frame (current)

    Returns:
        flow_mag: HxW float32 flow magnitude, normalized to [0, 1]
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    # flow shape: HxWx2 (dx, dy per pixel)
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)  # magnitude

    # Normalize to [0, 1] — clip at 99th percentile to avoid outlier domination
    p99 = np.percentile(mag, 99)
    if p99 > 0:
        mag = np.clip(mag / p99, 0.0, 1.0)
    else:
        mag = np.zeros_like(mag)

    return mag.astype(np.float32)


def flow_to_tensor(flow_mag: np.ndarray) -> torch.Tensor:
    """
    Convert HxW float32 flow magnitude to [1, H, W] normalized tensor.
    Applies the same normalization as grayscale frames: mean=0.5, std=0.5 → [-1, 1].

    Args:
        flow_mag: HxW float32, values in [0, 1]

    Returns:
        torch.Tensor [1, H, W], values in [-1, 1]
    """
    t = torch.from_numpy(flow_mag).unsqueeze(0)  # [1, H, W]
    t = (t - 0.5) / 0.5                          # normalize to [-1, 1]
    return t


# ─────────────────────────────────────────────────────────────────────────────
#  Dual-channel input builder
# ─────────────────────────────────────────────────────────────────────────────

def build_dual_channel_input(
    frame_tensor: torch.Tensor,
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
) -> torch.Tensor:
    """
    Concatenate grayscale frame tensor with optical flow magnitude tensor.

    Args:
        frame_tensor: [1, H, W] grayscale frame tensor (normalized)
        prev_gray:    HxW uint8 previous frame for flow computation
        curr_gray:    HxW uint8 current frame for flow computation

    Returns:
        dual: [2, H, W] tensor — channel 0: appearance, channel 1: flow
    """
    flow_mag = compute_farneback_flow(prev_gray, curr_gray)
    flow_t   = flow_to_tensor(flow_mag)               # [1, H, W]
    dual     = torch.cat([frame_tensor, flow_t], dim=0)  # [2, H, W]
    return dual


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers: load a frame as both tensor and uint8 numpy (for flow)
# ─────────────────────────────────────────────────────────────────────────────

_transform_cache = {}

def get_frame_transform(resolution: list) -> transforms.Compose:
    key = tuple(resolution)
    if key not in _transform_cache:
        h, w = resolution
        _transform_cache[key] = transforms.Compose([
            transforms.Resize((h, w)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    return _transform_cache[key]


def load_frame(path: str, resolution: list):
    """
    Load a frame from disk.

    Returns:
        tensor:    [1, H, W] normalized tensor for model input
        gray_np:   HxW uint8 numpy array for optical flow computation
    """
    h, w   = resolution
    img    = Image.open(path).convert("RGB")
    transform = get_frame_transform(resolution)
    tensor = transform(img)

    # Also produce uint8 grayscale numpy for Farnebäck
    gray_np = np.array(img.convert("L").resize((w, h), Image.BILINEAR))
    return tensor, gray_np


# ─────────────────────────────────────────────────────────────────────────────
#  Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import glob, yaml, os

    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    res   = config["dataset"]["resolution"]
    paths = sorted(glob.glob(
        os.path.join(config["dataset"]["train_dir"], "**", "*.jpg"), recursive=True
    ))[:2]

    assert len(paths) >= 2, "Need at least 2 frames for flow test"

    t0, g0 = load_frame(paths[0], res)
    t1, g1 = load_frame(paths[1], res)

    flow_mag = compute_farneback_flow(g0, g1)
    print(f"Flow magnitude — min: {flow_mag.min():.4f}  max: {flow_mag.max():.4f}  "
          f"mean: {flow_mag.mean():.4f}")

    dual = build_dual_channel_input(t1, g0, g1)
    print(f"Single-channel tensor : {t1.shape}")    # [1, 256, 256]
    print(f"Dual-channel tensor   : {dual.shape}")  # [2, 256, 256]
    print(f"Dual channel range    : [{dual.min():.3f}, {dual.max():.3f}]")
    print("flow_utils.py smoke test passed.")