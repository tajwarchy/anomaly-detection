"""
Evaluation script for Project V2.2 — Anomaly Detection.

Computes:
  - Frame-level AUC-ROC
  - Equal Error Rate (EER)
  - Optimal threshold (Youden's J)

Generates and saves:
  - outputs/roc_curve.png
  - outputs/score_timeline.png
  - outputs/error_distribution.png
"""

import os
import glob
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.ndimage import gaussian_filter1d

from utils.temporal_smoothing import smooth_and_save, normalize_scores


# ─────────────────────────────────────────────────────────────────────────────
#  Ground Truth Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_frame_level_gt(config: dict, frame_paths: list[str]) -> np.ndarray:
    """
    Derive frame-level binary GT labels from pixel-level mask files.

    A frame is labelled anomalous (1) if its corresponding GT mask contains
    any non-zero pixel. Normal frames have no GT mask file → labelled 0.

    The GT mask filenames follow the same sequence/index structure as test frames.
    frame_paths are sorted the same way as masks, so index alignment is direct.

    Args:
        config:      loaded config.yaml dict
        frame_paths: sorted list of test frame .jpg paths (length N)

    Returns:
        gt_labels: np.ndarray [N] int — 0=normal, 1=anomalous
    """
    gt_mask_dir = config["dataset"]["gt_mask_dir"]
    gt_labels   = np.zeros(len(frame_paths), dtype=np.int32)

    # Build sorted list of all GT mask paths — same sort order as frame_paths
    mask_paths = sorted(
        glob.glob(os.path.join(gt_mask_dir, "**", "*.png"), recursive=True)
    )

    if len(mask_paths) == 0:
        raise FileNotFoundError(f"No GT masks found in {gt_mask_dir}")

    if len(mask_paths) != len(frame_paths):
        print(f"[WARNING] GT masks ({len(mask_paths)}) != test frames ({len(frame_paths)}). "
              f"Aligning by minimum length.")

    n = min(len(mask_paths), len(frame_paths))

    import cv2
    for i in range(n):
        mask = cv2.imread(mask_paths[i], cv2.IMREAD_GRAYSCALE)
        if mask is not None and mask.max() > 0:
            gt_labels[i] = 1

    n_anomalous = gt_labels.sum()
    n_normal    = (gt_labels == 0).sum()
    print(f"[GT] Loaded {n} frame labels — Normal: {n_normal}, Anomalous: {n_anomalous}")
    return gt_labels


# ─────────────────────────────────────────────────────────────────────────────
#  EER Computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_eer(fpr: np.ndarray, tpr: np.ndarray) -> tuple[float, float]:
    """
    Compute Equal Error Rate (EER) — point where FAR = FRR.

    FAR (False Acceptance Rate) = FPR
    FRR (False Rejection Rate)  = 1 - TPR

    Returns:
        eer:       float — EER value in [0, 1]
        threshold: float — FPR value at EER point
    """
    fnr = 1.0 - tpr
    # Find index where |FPR - FNR| is minimized
    idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return float(eer), float(fpr[idx])


# ─────────────────────────────────────────────────────────────────────────────
#  Plot: AUC-ROC Curve
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curve(fpr, tpr, auc, eer, save_path: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#2196F3", linewidth=2.5,
            label=f"CAE (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    # Mark EER point
    eer_idx = np.argmin(np.abs(fpr - (1 - tpr)))
    ax.scatter(fpr[eer_idx], tpr[eer_idx], color="#F44336", zorder=5,
               s=80, label=f"EER = {eer*100:.2f}%")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Frame-Level AUC-ROC — UCSD Ped2", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Plot: Score Timeline
# ─────────────────────────────────────────────────────────────────────────────

def plot_score_timeline(
    raw_scores: np.ndarray,
    smoothed_scores: np.ndarray,
    gt_labels: np.ndarray,
    threshold: float,
    save_path: str,
):
    frames = np.arange(len(raw_scores))

    # Normalize all to [0,1] for overlay clarity
    raw_n  = normalize_scores(raw_scores)
    smo_n  = normalize_scores(smoothed_scores)
    thr_n  = (threshold - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-8)

    fig, ax = plt.subplots(figsize=(14, 4))

    # Shade anomalous ground-truth regions
    in_anomaly = False
    start_idx  = 0
    for i, label in enumerate(gt_labels):
        if label == 1 and not in_anomaly:
            start_idx = i
            in_anomaly = True
        elif label == 0 and in_anomaly:
            ax.axvspan(start_idx, i, color="#FFCDD2", alpha=0.5, label="GT Anomaly" if start_idx == next(
                (j for j, l in enumerate(gt_labels) if l == 1), None) else "")
            in_anomaly = False
    if in_anomaly:
        ax.axvspan(start_idx, len(gt_labels), color="#FFCDD2", alpha=0.5)

    ax.plot(frames, raw_n,  color="#90CAF9", linewidth=0.8, alpha=0.7, label="Raw score")
    ax.plot(frames, smo_n,  color="#1565C0", linewidth=1.8, label="Smoothed score")
    ax.axhline(thr_n, color="#F44336", linewidth=1.5, linestyle="--", label=f"Threshold ({threshold:.4f})")

    ax.set_xlabel("Frame Index", fontsize=11)
    ax.set_ylabel("Anomaly Score (normalized)", fontsize=11)
    ax.set_title("Anomaly Score Timeline — UCSD Ped2 Test Set", fontsize=13)
    ax.legend(fontsize=10, loc="upper left")
    ax.set_xlim(0, len(frames))
    ax.set_ylim(-0.05, 1.15)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Plot: Error Distribution (Normal vs Anomalous)
# ─────────────────────────────────────────────────────────────────────────────

def plot_error_distribution(
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    gt_labels: np.ndarray,
    threshold: float,
    save_path: str,
):
    normal_scores   = test_scores[gt_labels == 0]
    anomalous_scores = test_scores[gt_labels == 1]

    fig, ax = plt.subplots(figsize=(9, 5))

    bins = 60
    ax.hist(train_scores,      bins=bins, alpha=0.6, color="#4CAF50",
            label="Train (normal)", density=True)
    ax.hist(normal_scores,     bins=bins, alpha=0.6, color="#2196F3",
            label="Test — Normal",  density=True)
    ax.hist(anomalous_scores,  bins=bins, alpha=0.6, color="#F44336",
            label="Test — Anomalous", density=True)
    ax.axvline(threshold, color="black", linewidth=2, linestyle="--",
               label=f"Threshold = {threshold:.4f}")

    ax.set_xlabel("Reconstruction Error (MSE)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Reconstruction Error Distribution", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(config: dict):
    os.makedirs("outputs", exist_ok=True)

    # ── Load scores ────────────────────────────────────────────────────────
    raw_scores = np.load("outputs/anomaly_scores.npy")
    frame_paths = list(np.load("outputs/frame_paths.npy", allow_pickle=True))

    # ── Temporal smoothing ─────────────────────────────────────────────────
    raw_scores, smoothed_scores = smooth_and_save(config)

    # ── Ground truth ───────────────────────────────────────────────────────
    gt_labels = load_frame_level_gt(config, frame_paths)

    # Align lengths (safety)
    n = min(len(smoothed_scores), len(gt_labels))
    smoothed_scores = smoothed_scores[:n]
    raw_scores      = raw_scores[:n]
    gt_labels       = gt_labels[:n]

    # ── Threshold from training distribution ───────────────────────────────
    train_scores = np.load("outputs/train_scores.npy")
    pct          = config["inference"]["threshold_percentile"]
    threshold    = float(np.percentile(train_scores, pct))

    # ── AUC-ROC ────────────────────────────────────────────────────────────
    auc = roc_auc_score(gt_labels, smoothed_scores)
    fpr, tpr, roc_thresholds = roc_curve(gt_labels, smoothed_scores)

    # ── EER ────────────────────────────────────────────────────────────────
    eer, _ = compute_eer(fpr, tpr)

    # ── Youden's J optimal threshold ───────────────────────────────────────
    j_scores    = tpr - fpr
    best_idx    = np.argmax(j_scores)
    best_thresh = roc_thresholds[best_idx]

    # ── Plots ──────────────────────────────────────────────────────────────
    print("\n[evaluate] Generating plots...")
    plot_roc_curve(fpr, tpr, auc, eer,
                   save_path="outputs/roc_curve.png")
    plot_score_timeline(raw_scores, smoothed_scores, gt_labels, threshold,
                        save_path="outputs/score_timeline.png")
    plot_error_distribution(train_scores, smoothed_scores, gt_labels, threshold,
                            save_path="outputs/error_distribution.png")

    # ── Summary ────────────────────────────────────────────────────────────
    summary = f"""
╔══════════════════════════════════════════════╗
║         Evaluation Summary — V2.2            ║
╠══════════════════════════════════════════════╣
║  Dataset        : UCSD Ped2 (test set)       ║
║  Model          : CAE (appearance only)      ║
║  Flow           : {'ON ' if config['flow']['enabled'] else 'OFF'}                            ║
╠══════════════════════════════════════════════╣
║  AUC-ROC (frame): {auc:.4f}                      ║
║  EER            : {eer*100:.2f}%                      ║
║  Threshold ({pct}th pct): {threshold:.6f}          ║
║  Youden thresh  : {best_thresh:.6f}              ║
╠══════════════════════════════════════════════╣
║  GT  — Normal frames   : {(gt_labels==0).sum():<6}               ║
║  GT  — Anomalous frames: {(gt_labels==1).sum():<6}               ║
╚══════════════════════════════════════════════╝
"""
    print(summary)

    # Save summary as text
    with open("outputs/eval_summary.txt", "w") as f:
        f.write(summary)
    print("  Saved → outputs/eval_summary.txt")

    return auc, eer, threshold


if __name__ == "__main__":
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    auc, eer, threshold = run_evaluation(config)

    # Flow decision gate
    print("\n[FLOW DECISION GATE]")
    if auc >= 0.85:
        print(f"  AUC={auc:.4f} ≥ 0.85 → Appearance-only is strong. "
              f"Optical flow is a bonus — proceed to Phase 5.")
    elif auc >= 0.80:
        print(f"  AUC={auc:.4f} — Decent. Flow may push above 0.85. "
              f"Consider retraining with flow.enabled=true.")
    else:
        print(f"  AUC={auc:.4f} < 0.80 → Optical flow retraining recommended. "
              f"Set flow.enabled=true in config and retrain on Colab.")