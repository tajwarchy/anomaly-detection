"""
Full inference pipeline for Project V2.2 — Anomaly Detection.

Processes test frames end-to-end on M1:
  1. Load CAE checkpoint
  2. Per-frame reconstruction + anomaly score (MPS)
  3. Per-sequence score normalization
  4. Heatmap overlay + score graph composite frame (CPU)
  5. Export full annotated video as MP4
  6. Auto-extract and save flagged anomalous clips as separate MP4s

Usage:
    python scripts/inference_pipeline.py --config configs/config.yaml
    python scripts/inference_pipeline.py --config configs/config.yaml --source data/ucsd_ped2/testing/frames/Test001
"""

import os
import glob
import argparse
import yaml
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.cae import build_model
from utils.flow_utils import load_frame, build_dual_channel_input
from utils.temporal_smoothing import gaussian_smooth
from utils.visualization import (
    ScoreGraphStrip,
    build_composite_frame,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(config: dict, device: torch.device):
    ckpt_path = os.path.join(
        config["output"]["checkpoint_dir"],
        config["output"]["checkpoint_name"],
    )
    ckpt  = torch.load(ckpt_path, map_location=device)
    model = build_model(config).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"[pipeline] Loaded checkpoint: {ckpt_path} "
          f"(epoch {ckpt.get('epoch','?')}, val_loss {ckpt.get('val_loss',0):.6f})")
    return model


def get_device(config: dict) -> torch.device:
    d = config["inference"]["device"]
    if d == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def collect_frame_paths(source_dir: str) -> tuple[list[str], list[int]]:
    """
    Collect all .jpg frame paths from source_dir.
    If source_dir contains subdirectories (sequences), collect per-sequence.
    Returns (frame_paths, sequence_ids).
    """
    subdirs = sorted([
        d for d in glob.glob(os.path.join(source_dir, "*"))
        if os.path.isdir(d)
    ])

    if subdirs:
        frame_paths, seq_ids = [], []
        for seq_idx, seq_dir in enumerate(subdirs):
            paths = sorted(glob.glob(os.path.join(seq_dir, "*.jpg")))
            frame_paths.extend(paths)
            seq_ids.extend([seq_idx] * len(paths))
        return frame_paths, seq_ids
    else:
        # Single sequence directory
        paths = sorted(glob.glob(os.path.join(source_dir, "*.jpg")))
        return paths, [0] * len(paths)


def normalize_per_sequence(
    scores: np.ndarray,
    seq_ids: np.ndarray,
) -> np.ndarray:
    normalized = np.zeros_like(scores)
    for sid in np.unique(seq_ids):
        mask = seq_ids == sid
        s    = scores[mask]
        lo, hi = s.min(), s.max()
        normalized[mask] = (s - lo) / (hi - lo + 1e-8)
    return normalized


def load_train_threshold(config: dict) -> float:
    train_scores = np.load("outputs/train_scores.npy")
    pct = config["inference"]["threshold_percentile"]
    return float(np.percentile(train_scores, pct))


# ─────────────────────────────────────────────────────────────────────────────
#  Pass 1 — Score all frames
# ─────────────────────────────────────────────────────────────────────────────

def score_all_frames(
    model: torch.nn.Module,
    frame_paths: list[str],
    seq_ids: list[int],
    config: dict,
    device: torch.device,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Forward pass through CAE for every frame.

    Returns:
        raw_scores: np.ndarray [N] — per-frame MSE scores
        error_maps: list of HxW float32 numpy arrays
    """
    res          = config["dataset"]["resolution"]
    flow_enabled = config["flow"]["enabled"]
    raw_scores   = []
    error_maps   = []
    prev_gray    = None
    prev_seq     = None

    for i, path in enumerate(tqdm(frame_paths, desc="Pass 1 — Scoring")):
        curr_seq = seq_ids[i]
        frame_tensor, curr_gray = load_frame(path, res)

        if flow_enabled:
            if prev_gray is None or curr_seq != prev_seq:
                prev_gray = curr_gray
            inp = build_dual_channel_input(frame_tensor, prev_gray, curr_gray)
        else:
            inp = frame_tensor

        x = inp.unsqueeze(0).to(device)
        with torch.no_grad():
            recon = model(x)
        err      = F.mse_loss(recon, x, reduction='none').mean(dim=1).squeeze()
        score    = err.mean().item()
        err_np   = err.cpu().numpy()

        raw_scores.append(score)
        error_maps.append(err_np)
        prev_gray = curr_gray
        prev_seq  = curr_seq

    return np.array(raw_scores, dtype=np.float32), error_maps


# ─────────────────────────────────────────────────────────────────────────────
#  Pass 2 — Render & write video
# ─────────────────────────────────────────────────────────────────────────────

def render_annotated_video(
    frame_paths: list[str],
    error_maps: list[np.ndarray],
    norm_scores: np.ndarray,
    raw_scores: np.ndarray,
    config: dict,
    output_path: str,
) -> list[dict]:
    """
    Render composite frames and write annotated video.
    Returns list of flagged segment dicts for clip extraction.
    """
    h, w        = config["dataset"]["resolution"]
    fps         = config["output"]["video_fps"]
    alpha       = config["visualization"]["heatmap_alpha"]
    threshold   = config["inference"].get("detection_threshold", 0.5)
    window_size = config["inference"]["score_graph_frames"]
    strip_h     = 80
    out_h       = h + strip_h

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, out_h))

    strip = ScoreGraphStrip(
        frame_width=w,
        strip_height=strip_h,
        window_size=window_size,
        threshold=threshold,
    )

    flagged_segments = []
    in_anomaly       = False
    seg_start        = 0

    print(f"\n[pipeline] Pass 2 — Rendering {len(frame_paths)} frames → {output_path}")

    for i, path in enumerate(tqdm(frame_paths, desc="Pass 2 — Rendering")):
        frame_bgr = cv2.imread(path)
        frame_bgr = cv2.resize(frame_bgr, (w, h))

        norm_score = float(norm_scores[i])
        raw_score  = float(raw_scores[i])
        is_anomalous = norm_score >= threshold

        strip.update(norm_score)

        composite = build_composite_frame(
            frame_bgr=frame_bgr,
            error_map=error_maps[i],
            norm_score=norm_score,
            raw_score=raw_score,
            frame_idx=i,
            threshold=threshold,
            score_strip=strip,
            heatmap_alpha=alpha,
        )

        writer.write(composite)

        # Track flagged segments
        if is_anomalous and not in_anomaly:
            seg_start  = i
            in_anomaly = True
        elif not is_anomalous and in_anomaly:
            flagged_segments.append({
                "start": seg_start,
                "end":   i - 1,
                "peak":  float(norm_scores[seg_start:i].max()),
            })
            in_anomaly = False

    if in_anomaly:
        flagged_segments.append({
            "start": seg_start,
            "end":   len(frame_paths) - 1,
            "peak":  float(norm_scores[seg_start:].max()),
        })

    writer.release()
    print(f"  → Annotated video saved: {output_path}")
    return flagged_segments


# ─────────────────────────────────────────────────────────────────────────────
#  Flagged Clip Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_flagged_clips(
    frame_paths: list[str],
    error_maps: list[np.ndarray],
    norm_scores: np.ndarray,
    raw_scores: np.ndarray,
    flagged_segments: list[dict],
    config: dict,
):
    if not flagged_segments:
        print("[pipeline] No flagged segments — no clips to extract.")
        return

    h, w       = config["dataset"]["resolution"]
    fps        = config["output"]["video_fps"]
    alpha      = config["visualization"]["heatmap_alpha"]
    threshold  = config["inference"].get("detection_threshold", 0.5)
    padding    = config["output"]["clip_padding_frames"]
    clip_dir   = config["output"]["flagged_clip_dir"]
    window_size = config["inference"]["score_graph_frames"]
    strip_h    = 80
    out_h      = h + strip_h
    n          = len(frame_paths)

    os.makedirs(clip_dir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    print(f"\n[pipeline] Extracting {len(flagged_segments)} flagged clips...")

    for clip_idx, seg in enumerate(flagged_segments):
        start = max(0, seg["start"] - padding)
        end   = min(n - 1, seg["end"] + padding)
        clip_path = os.path.join(clip_dir, f"clip_{clip_idx+1:03d}.mp4")

        writer = cv2.VideoWriter(clip_path, fourcc, fps, (w, out_h))

        strip = ScoreGraphStrip(
            frame_width=w,
            strip_height=strip_h,
            window_size=window_size,
            threshold=threshold,
        )

        for i in range(start, end + 1):
            frame_bgr = cv2.imread(frame_paths[i])
            frame_bgr = cv2.resize(frame_bgr, (w, h))

            strip.update(float(norm_scores[i]))

            composite = build_composite_frame(
                frame_bgr=frame_bgr,
                error_map=error_maps[i],
                norm_score=float(norm_scores[i]),
                raw_score=float(raw_scores[i]),
                frame_idx=i,
                threshold=threshold,
                score_strip=strip,
                heatmap_alpha=alpha,
            )
            writer.write(composite)

        writer.release()
        print(f"  Clip {clip_idx+1:03d}: frames {start}–{end} "
              f"(peak score={seg['peak']:.3f}) → {clip_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(config: dict, source_dir: str):
    device = get_device(config)
    print(f"[pipeline] Device: {device}")

    # ── Load model ─────────────────────────────────────────────────────────
    model = load_checkpoint(config, device)

    # ── Collect frames ─────────────────────────────────────────────────────
    frame_paths, seq_ids_list = collect_frame_paths(source_dir)
    seq_ids = np.array(seq_ids_list, dtype=np.int32)
    print(f"[pipeline] {len(frame_paths):,} frames from {source_dir}")

    if len(frame_paths) == 0:
        print("[pipeline] No frames found. Check --source path.")
        return

    # ── Pass 1: score all frames ───────────────────────────────────────────
    raw_scores, error_maps = score_all_frames(
        model, frame_paths, seq_ids_list, config, device
    )

    # ── Temporal smoothing ─────────────────────────────────────────────────
    smoothed = gaussian_smooth(
        raw_scores,
        window=config["inference"]["temporal_window"],
        sigma=config["inference"]["temporal_sigma"],
    )

    # ── Per-sequence normalization ─────────────────────────────────────────
    norm_scores = normalize_per_sequence(smoothed, seq_ids)

    # ── Detection threshold (fixed at 0.5 after per-seq normalization) ─────
    config["inference"]["detection_threshold"] = 0.5

    # ── Save scores ────────────────────────────────────────────────────────
    os.makedirs("outputs", exist_ok=True)
    np.save("outputs/pipeline_raw_scores.npy",  raw_scores)
    np.save("outputs/pipeline_norm_scores.npy", norm_scores)

    n_anomalous = (norm_scores >= 0.5).sum()
    print(f"[pipeline] Anomalous frames: {n_anomalous}/{len(frame_paths)} "
          f"({100*n_anomalous/len(frame_paths):.1f}%)")

    # ── Pass 2: render annotated video ─────────────────────────────────────
    annotated_path    = config["output"]["annotated_video_path"]
    flagged_segments  = render_annotated_video(
        frame_paths, error_maps, norm_scores, raw_scores, config, annotated_path
    )

    # ── Extract flagged clips ──────────────────────────────────────────────
    extract_flagged_clips(
        frame_paths, error_maps, norm_scores, raw_scores,
        flagged_segments, config
    )

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"""
╔══════════════════════════════════════════════╗
║         Pipeline Complete — V2.2             ║
╠══════════════════════════════════════════════╣
║  Frames processed : {len(frame_paths):<6}                    ║
║  Anomalous frames : {n_anomalous:<6}                    ║
║  Flagged clips    : {len(flagged_segments):<6}                    ║
║  Annotated video  : {os.path.basename(annotated_path):<25} ║
╚══════════════════════════════════════════════╝
""")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V2.2 Anomaly Detection Inference Pipeline")
    parser.add_argument("--config", default="configs/config.yaml",
                        help="Path to config YAML")
    parser.add_argument("--source", default=None,
                        help="Override source directory (default: config test_dir)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    source_dir = args.source or config["dataset"]["test_dir"]
    run_pipeline(config, source_dir)