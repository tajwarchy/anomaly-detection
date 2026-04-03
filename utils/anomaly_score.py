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
    x = frame_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        recon = model(x)
    error     = F.mse_loss(recon, x, reduction='none')
    error_map = error.mean(dim=1).squeeze()
    score     = error_map.mean().item()
    error_np  = error_map.cpu().numpy()
    return score, error_np


# ─────────────────────────────────────────────────────────────────────────────
#  Full test-set scoring pipeline
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_scores(config: dict) -> tuple[np.ndarray, list[str], np.ndarray]:
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
    print(f"[anomaly_score] Loaded checkpoint from {ckpt_path}  "
          f"(epoch {ckpt.get('epoch', '?')}, val_loss {ckpt.get('val_loss', '?'):.6f})")

    res          = config["dataset"]["resolution"]
    flow_enabled = config["flow"]["enabled"]
    test_dir     = config["dataset"]["test_dir"]

    # Collect frames grouped by sequence — preserves sequence boundaries
    seq_dirs = sorted([
        d for d in glob.glob(os.path.join(test_dir, "*"))
        if os.path.isdir(d)
    ])

    frame_paths   = []
    sequence_ids  = []   # integer sequence index per frame — for per-seq normalization

    for seq_idx, seq_dir in enumerate(seq_dirs):
        paths = sorted(glob.glob(os.path.join(seq_dir, "*.jpg")))
        frame_paths.extend(paths)
        sequence_ids.extend([seq_idx] * len(paths))

    sequence_ids = np.array(sequence_ids, dtype=np.int32)
    print(f"[anomaly_score] Scoring {len(frame_paths):,} test frames across "
          f"{len(seq_dirs)} sequences | flow={'ON' if flow_enabled else 'OFF'} | device={device}")

    scores    = []
    prev_gray = None
    prev_seq  = None

    for i, path in enumerate(tqdm(frame_paths, desc="Scoring frames")):
        curr_seq = sequence_ids[i]

        frame_tensor, curr_gray = load_frame(path, res)

        if flow_enabled:
            # Reset flow at sequence boundaries
            if prev_gray is None or curr_seq != prev_seq:
                prev_gray = curr_gray
            inp = build_dual_channel_input(frame_tensor, prev_gray, curr_gray)
        else:
            inp = frame_tensor

        score, _ = compute_reconstruction_error(model, inp, device)
        scores.append(score)
        prev_gray = curr_gray
        prev_seq  = curr_seq

    scores = np.array(scores, dtype=np.float32)

    os.makedirs("outputs", exist_ok=True)
    np.save("outputs/anomaly_scores.npy",  scores)
    np.save("outputs/frame_paths.npy",     np.array(frame_paths))
    np.save("outputs/sequence_ids.npy",    sequence_ids)
    print(f"[anomaly_score] Scores saved → outputs/anomaly_scores.npy")
    print(f"  min={scores.min():.6f}  max={scores.max():.6f}  mean={scores.mean():.6f}")

    return scores, frame_paths, sequence_ids


# ─────────────────────────────────────────────────────────────────────────────
#  Training-set score distribution (for threshold calibration)
# ─────────────────────────────────────────────────────────────────────────────

def compute_train_score_distribution(config: dict) -> np.ndarray:
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

    res         = config["dataset"]["resolution"]
    train_dir   = config["dataset"]["train_dir"]
    train_paths = sorted(
        glob.glob(os.path.join(train_dir, "**", "*.jpg"), recursive=True)
    )
    print(f"[anomaly_score] Computing train score distribution on {len(train_paths):,} frames...")

    flow_enabled = config["flow"]["enabled"]
    train_scores = []
    prev_gray    = None
    prev_seq     = None

    for path in tqdm(train_paths, desc="Train scores"):
        # Detect sequence boundary by parent folder name
        curr_seq = os.path.basename(os.path.dirname(path))

        frame_tensor, curr_gray = load_frame(path, res)

        if flow_enabled:
            if prev_gray is None or curr_seq != prev_seq:
                prev_gray = curr_gray
            inp = build_dual_channel_input(frame_tensor, prev_gray, curr_gray)
        else:
            inp = frame_tensor

        score, _  = compute_reconstruction_error(model, inp, device)
        train_scores.append(score)
        prev_gray = curr_gray
        prev_seq  = curr_seq

    train_scores = np.array(train_scores, dtype=np.float32)
    np.save("outputs/train_scores.npy", train_scores)

    pct       = config["inference"]["threshold_percentile"]
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

    train_scores              = compute_train_score_distribution(config)
    scores, paths, seq_ids    = compute_all_scores(config)

    print("\n[DONE] Anomaly scoring complete.")
    print(f"  Train score range : [{train_scores.min():.6f}, {train_scores.max():.6f}]")
    print(f"  Test  score range : [{scores.min():.6f}, {scores.max():.6f}]")