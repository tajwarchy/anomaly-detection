"""
Anomaly score computation for CAE-based reconstruction error.

Frame-level score  = mean per-pixel MSE between input and reconstruction.
Pixel-level map    = per-pixel MSE resized to original frame resolution.

Supports:
  - Single-channel (grayscale only): config flow.enabled = false
  - Dual-channel (grayscale + flow): config flow.enabled = true
"""

import os
import glob
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.cae import build_model
from utils.flow_utils import load_frame, build_dual_channel_input


# ─────────────────────────────────────────────────────────────────────────────
#  Core scoring functions
# ─────────────────────────────────────────────────────────────────────────────

def compute_reconstruction_error(
    model: torch.nn.Module,
    frame_tensor: torch.Tensor,
    device: torch.device,
) -> tuple[float, np.ndarray]:
    """
    Forward pass through CAE, compute per-pixel MSE.

    Args:
        model:        trained CAE in eval mode
        frame_tensor: [C, H, W] input tensor (normalized)
        device:       torch device

    Returns:
        score:    float — mean reconstruction error (frame-level anomaly score)
        error_map: HxW float32 numpy — per-pixel MSE (for heatmap generation)
    """
    x = frame_tensor.unsqueeze(0).to(device)        # [1, C, H, W]
    with torch.no_grad():
        recon = model(x)                             # [1, C, H, W]

    # Per-pixel MSE across channels → [1, 1, H, W]
    error = F.mse_loss(recon, x, reduction='none')  # [1, C, H, W]
    error_map = error.mean(dim=1).squeeze()          # [H, W]

    score     = error_map.mean().item()
    error_np  = error_map.cpu().numpy()

    return score, error_np


# ─────────────────────────────────────────────────────────────────────────────
#  Full test-set scoring pipeline
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_scores(config: dict) -> tuple[np.ndarray, list[str]]:
    """
    Run CAE inference over all test frames, compute per-frame anomaly scores.

    Args:
        config: loaded config.yaml dict

    Returns:
        scores:      np.ndarray [N] — frame-level anomaly scores
        frame_paths: list[str] [N]  — corresponding frame file paths (sorted)
    """
    device_str = config["inference"]["device"]
    device     = torch.device(device_str if torch.backends.mps.is_available()
                              else "cpu") if device_str == "mps" else torch.device(device_str)

    # Load model
    ckpt_path = os.path.join(
        config["output"]["checkpoint_dir"],
        config["output"]["checkpoint_name"],
    )
    ckpt  = torch.load(ckpt_path, map_location=device)
    model = build_model(config).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"[anomaly_score] Loaded checkpoint from {ckpt_path}  "
          f"(epoch {ckpt.get('epoch', '?')}, val_loss {ckpt.get('val_loss', '?'):.6f})")

    res          = config["dataset"]["resolution"]
    flow_enabled = config["flow"]["enabled"]
    test_dir     = config["dataset"]["test_dir"]

    frame_paths = sorted(
        glob.glob(os.path.join(test_dir, "**", "*.jpg"), recursive=True)
    )
    print(f"[anomaly_score] Scoring {len(frame_paths):,} test frames  "
          f"| flow={'ON' if flow_enabled else 'OFF'}  | device={device}")

    scores    = []
    prev_gray = None  # needed for flow

    for i, path in enumerate(tqdm(frame_paths, desc="Scoring frames")):
        frame_tensor, curr_gray = load_frame(path, res)

        if flow_enabled:
            if prev_gray is None:
                # First frame: no previous → use same frame (zero flow)
                prev_gray = curr_gray
            inp = build_dual_channel_input(frame_tensor, prev_gray, curr_gray)
        else:
            inp = frame_tensor  # [1, H, W]

        score, _ = compute_reconstruction_error(model, inp, device)
        scores.append(score)
        prev_gray = curr_gray

    scores = np.array(scores, dtype=np.float32)

    # Save to disk
    os.makedirs("outputs", exist_ok=True)
    np.save("outputs/anomaly_scores.npy", scores)
    np.save("outputs/frame_paths.npy", np.array(frame_paths))
    print(f"[anomaly_score] Scores saved → outputs/anomaly_scores.npy")
    print(f"  min={scores.min():.6f}  max={scores.max():.6f}  mean={scores.mean():.6f}")

    return scores, frame_paths


# ─────────────────────────────────────────────────────────────────────────────
#  Training-set score distribution (for threshold calibration)
# ─────────────────────────────────────────────────────────────────────────────

def compute_train_score_distribution(config: dict) -> np.ndarray:
    """
    Compute anomaly scores on training frames to calibrate the detection threshold.
    The threshold is set at the Nth percentile of this distribution.

    Returns:
        train_scores: np.ndarray [N] — scores on normal training frames
    """
    device_str = config["inference"]["device"]
    device     = torch.device(device_str if torch.backends.mps.is_available()
                              else "cpu") if device_str == "mps" else torch.device(device_str)

    ckpt_path = os.path.join(
        config["output"]["checkpoint_dir"],
        config["output"]["checkpoint_name"],
    )
    ckpt  = torch.load(ckpt_path, map_location=device)
    model = build_model(config).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    res        = config["dataset"]["resolution"]
    train_dir  = config["dataset"]["train_dir"]
    train_paths = sorted(
        glob.glob(os.path.join(train_dir, "**", "*.jpg"), recursive=True)
    )
    print(f"[anomaly_score] Computing train score distribution on {len(train_paths):,} frames...")

    train_scores = []
    for path in tqdm(train_paths, desc="Train scores"):
        frame_tensor, _ = load_frame(path, res)
        score, _        = compute_reconstruction_error(model, frame_tensor, device)
        train_scores.append(score)

    train_scores = np.array(train_scores, dtype=np.float32)
    np.save("outputs/train_scores.npy", train_scores)

    pct = config["inference"]["threshold_percentile"]
    threshold = np.percentile(train_scores, pct)
    print(f"[anomaly_score] Train scores saved → outputs/train_scores.npy")
    print(f"  {pct}th percentile threshold = {threshold:.6f}")

    return train_scores


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Step 1: calibrate threshold on train set
    train_scores = compute_train_score_distribution(config)

    # Step 2: score all test frames
    scores, paths = compute_all_scores(config)

    print("\n[DONE] Anomaly scoring complete.")
    print(f"  Train score range : [{train_scores.min():.6f}, {train_scores.max():.6f}]")
    print(f"  Test  score range : [{scores.min():.6f}, {scores.max():.6f}]")