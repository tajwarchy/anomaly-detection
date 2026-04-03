"""
Evaluation script for Project V2.2 — Anomaly Detection.

Computes:
  - Frame-level AUC-ROC (with per-sequence score normalization)
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

from utils.temporal_smoothing import smooth_and_save, normalize_scores


# ─────────────────────────────────────────────────────────────────────────────
#  Ground Truth Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_frame_level_gt(config: dict, frame_paths: list) -> np.ndarray:
    gt_mask_dir = config["dataset"]["gt_mask_dir"]
    gt_labels   = np.zeros(len(frame_paths), dtype=np.int32)

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
#  Per-Sequence Normalization
# ─────────────────────────────────────────────────────────────────────────────

def normalize_scores_per_sequence(
    scores: np.ndarray,
    sequence_ids: np.ndarray,
) -> np.ndarray:
    """
    Normalize anomaly scores independently within each test sequence.

    Within-sequence normalization stretches each sequence's score range to [0,1],
    making local anomaly spikes visible even when absolute MSE values are similar
    across all frames — which is the dominant failure mode for UCSD Ped2.

    Args:
        scores:       np.ndarray [N] — raw or smoothed anomaly scores
        sequence_ids: np.ndarray [N] — integer sequence index per frame

    Returns:
        normalized: np.ndarray [N] — per-sequence min-max normalized scores
    """
    normalized = np.zeros_like(scores)
    for seq_id in np.unique(sequence_ids):
        mask    = sequence_ids == seq_id
        s       = scores[mask]
        s_min, s_max = s.min(), s.max()
        if s_max - s_min < 1e-8:
            normalized[mask] = 0.0
        else:
            normalized[mask] = (s - s_min) / (s_max - s_min)
    return normalized


# ─────────────────────────────────────────────────────────────────────────────
#  EER Computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_eer(fpr: np.ndarray, tpr: np.ndarray) -> tuple[float, float]:
    fnr = 1.0 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return float(eer), float(fpr[idx])


# ─────────────────────────────────────────────────────────────────────────────
#  Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curve(fpr, tpr, auc, eer, save_path: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#2196F3", linewidth=2.5,
            label=f"CAE (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
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


def plot_score_timeline(
    raw_scores: np.ndarray,
    smoothed_scores: np.ndarray,
    normalized_scores: np.ndarray,
    gt_labels: np.ndarray,
    sequence_ids: np.ndarray,
    save_path: str,
):
    frames = np.arange(len(raw_scores))
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    # ── Top: raw + smoothed scores ──────────────────────────────────────────
    ax = axes[0]
    _shade_anomalies(ax, gt_labels)
    ax.plot(frames, raw_scores,      color="#90CAF9", linewidth=0.8, alpha=0.7, label="Raw score")
    ax.plot(frames, smoothed_scores, color="#1565C0", linewidth=1.8, label="Smoothed score")
    _draw_seq_boundaries(ax, sequence_ids)
    ax.set_ylabel("MSE Score", fontsize=10)
    ax.set_title("Raw Anomaly Scores", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.2)

    # ── Bottom: per-sequence normalized scores ──────────────────────────────
    ax = axes[1]
    _shade_anomalies(ax, gt_labels)
    ax.plot(frames, normalized_scores, color="#1B5E20", linewidth=1.5,
            label="Per-sequence normalized")
    ax.axhline(0.5, color="#F44336", linewidth=1.2, linestyle="--", label="Threshold=0.5")
    _draw_seq_boundaries(ax, sequence_ids)
    ax.set_xlabel("Frame Index", fontsize=11)
    ax.set_ylabel("Normalized Score [0,1]", fontsize=10)
    ax.set_title("Per-Sequence Normalized Scores (used for AUC)", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(0, len(frames))
    ax.set_ylim(-0.05, 1.15)
    ax.grid(True, alpha=0.2)

    plt.suptitle("Anomaly Score Timeline — UCSD Ped2 Test Set", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def _shade_anomalies(ax, gt_labels):
    in_anomaly, start_idx = False, 0
    labeled = False
    for i, label in enumerate(gt_labels):
        if label == 1 and not in_anomaly:
            start_idx = i
            in_anomaly = True
        elif label == 0 and in_anomaly:
            ax.axvspan(start_idx, i, color="#FFCDD2", alpha=0.4,
                       label="GT Anomaly" if not labeled else "")
            labeled    = True
            in_anomaly = False
    if in_anomaly:
        ax.axvspan(start_idx, len(gt_labels), color="#FFCDD2", alpha=0.4)


def _draw_seq_boundaries(ax, sequence_ids):
    boundaries = np.where(np.diff(sequence_ids))[0] + 1
    for b in boundaries:
        ax.axvline(b, color="gray", linewidth=0.6, linestyle=":", alpha=0.6)


def plot_error_distribution(
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    gt_labels: np.ndarray,
    save_path: str,
):
    normal_scores    = test_scores[gt_labels == 0]
    anomalous_scores = test_scores[gt_labels == 1]

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = 60
    ax.hist(train_scores,      bins=bins, alpha=0.6, color="#4CAF50",
            label="Train (normal)", density=True)
    ax.hist(normal_scores,     bins=bins, alpha=0.6, color="#2196F3",
            label="Test — Normal",  density=True)
    ax.hist(anomalous_scores,  bins=bins, alpha=0.6, color="#F44336",
            label="Test — Anomalous", density=True)
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

    # ── Load scores & metadata ─────────────────────────────────────────────
    raw_scores   = np.load("outputs/anomaly_scores.npy")
    frame_paths  = list(np.load("outputs/frame_paths.npy", allow_pickle=True))
    sequence_ids = np.load("outputs/sequence_ids.npy")

    # ── Temporal smoothing ─────────────────────────────────────────────────
    raw_scores, smoothed_scores = smooth_and_save(config)

    # ── Per-sequence normalization ─────────────────────────────────────────
    normalized_scores = normalize_scores_per_sequence(smoothed_scores, sequence_ids)
    np.save("outputs/anomaly_scores_normalized.npy", normalized_scores)
    print(f"[evaluate] Per-sequence normalized scores saved.")

    # ── Ground truth ───────────────────────────────────────────────────────
    gt_labels = load_frame_level_gt(config, frame_paths)

    # Align lengths
    n                 = min(len(normalized_scores), len(gt_labels))
    normalized_scores = normalized_scores[:n]
    smoothed_scores   = smoothed_scores[:n]
    raw_scores        = raw_scores[:n]
    gt_labels         = gt_labels[:n]
    sequence_ids      = sequence_ids[:n]

    # ── AUC-ROC on normalized scores ───────────────────────────────────────
    auc = roc_auc_score(gt_labels, normalized_scores)
    fpr, tpr, roc_thresholds = roc_curve(gt_labels, normalized_scores)

    # ── EER ────────────────────────────────────────────────────────────────
    eer, _ = compute_eer(fpr, tpr)

    # ── Youden's J optimal threshold ───────────────────────────────────────
    j_scores    = tpr - fpr
    best_idx    = np.argmax(j_scores)
    best_thresh = roc_thresholds[best_idx]

    # ── Train scores ───────────────────────────────────────────────────────
    train_scores = np.load("outputs/train_scores.npy")
    pct          = config["inference"]["threshold_percentile"]

    # ── Plots ──────────────────────────────────────────────────────────────
    print("\n[evaluate] Generating plots...")
    plot_roc_curve(fpr, tpr, auc, eer,
                   save_path="outputs/roc_curve.png")
    plot_score_timeline(raw_scores, smoothed_scores, normalized_scores,
                        gt_labels, sequence_ids,
                        save_path="outputs/score_timeline.png")
    plot_error_distribution(train_scores, smoothed_scores, gt_labels,
                            save_path="outputs/error_distribution.png")

    # ── Summary ────────────────────────────────────────────────────────────
    summary = f"""
╔══════════════════════════════════════════════╗
║         Evaluation Summary — V2.2            ║
╠══════════════════════════════════════════════╣
║  Dataset        : UCSD Ped2 (test set)       ║
║  Model          : CAE (appearance only)      ║
║  Flow           : {'ON ' if config['flow']['enabled'] else 'OFF'}                            ║
║  Normalization  : Per-sequence               ║
╠══════════════════════════════════════════════╣
║  AUC-ROC (frame): {auc:.4f}                      ║
║  EER            : {eer*100:.2f}%                      ║
║  Youden thresh  : {best_thresh:.6f}              ║
╠══════════════════════════════════════════════╣
║  GT  — Normal frames   : {(gt_labels==0).sum():<6}               ║
║  GT  — Anomalous frames: {(gt_labels==1).sum():<6}               ║
╚══════════════════════════════════════════════╝
"""
    print(summary)
    with open("outputs/eval_summary.txt", "w") as f:
        f.write(summary)
    print("  Saved → outputs/eval_summary.txt")

    return auc, eer, best_thresh


if __name__ == "__main__":
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    auc, eer, threshold = run_evaluation(config)

    print("\n[FLOW DECISION GATE]")
    if auc >= 0.85:
        print(f"  AUC={auc:.4f} ≥ 0.85 → Strong result. Proceed to Phase 5.")
    elif auc >= 0.75:
        print(f"  AUC={auc:.4f} — Good. Flow may push above 0.85. "
              f"Consider enabling flow.enabled=true.")
    else:
        print(f"  AUC={auc:.4f} < 0.75 → Enable flow.enabled=true and retrain.")