"""
Temporal smoothing for anomaly score sequences.
Applies a Gaussian-weighted sliding window to reduce single-frame false positive spikes
while preserving genuine anomaly bursts.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d


def gaussian_smooth(scores: np.ndarray, window: int, sigma: float) -> np.ndarray:
    """
    Apply Gaussian smoothing to a 1D anomaly score sequence.

    Args:
        scores: np.ndarray [N] — raw per-frame anomaly scores
        window: int            — context window size (informational; scipy uses sigma)
        sigma:  float          — Gaussian standard deviation in frames

    Returns:
        smoothed: np.ndarray [N] — smoothed scores, same length as input
    """
    smoothed = gaussian_filter1d(scores.astype(np.float64), sigma=sigma)
    return smoothed.astype(np.float32)


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """
    Min-max normalize scores to [0, 1] for threshold-agnostic comparison.
    """
    s_min, s_max = scores.min(), scores.max()
    if s_max - s_min < 1e-8:
        return np.zeros_like(scores)
    return (scores - s_min) / (s_max - s_min)


def smooth_and_save(config: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Load raw scores, apply temporal smoothing, save smoothed scores.

    Returns:
        raw_scores:      np.ndarray [N]
        smoothed_scores: np.ndarray [N]
    """
    import os

    raw_scores = np.load("outputs/anomaly_scores.npy")

    window = config["inference"]["temporal_window"]
    sigma  = config["inference"]["temporal_sigma"]

    smoothed = gaussian_smooth(raw_scores, window, sigma)

    os.makedirs("outputs", exist_ok=True)
    np.save("outputs/anomaly_scores_smoothed.npy", smoothed)

    print(f"[temporal_smoothing] Window={window}, Sigma={sigma}")
    print(f"  Raw      — min={raw_scores.min():.6f}  max={raw_scores.max():.6f}")
    print(f"  Smoothed — min={smoothed.min():.6f}  max={smoothed.max():.6f}")
    print(f"  Saved → outputs/anomaly_scores_smoothed.npy")

    return raw_scores, smoothed


if __name__ == "__main__":
    import yaml
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    smooth_and_save(config)