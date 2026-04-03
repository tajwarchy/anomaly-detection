"""
Visualization utilities for Project V2.2 — Anomaly Detection.

Provides:
  - Per-frame heatmap overlay (JET colormap blended onto original frame)
  - Binary anomaly mask with contour overlay
  - Running anomaly score graph strip (bottom of frame)
  - Anomaly banner text overlay
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import deque


# ─────────────────────────────────────────────────────────────────────────────
#  Heatmap Overlay
# ─────────────────────────────────────────────────────────────────────────────

def make_heatmap_overlay(
    frame_bgr: np.ndarray,
    error_map: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Blend per-pixel reconstruction error map onto the original frame as a
    JET colormap heatmap.

    Args:
        frame_bgr: HxWx3 uint8 BGR frame (original resolution)
        error_map: HxW float32 reconstruction error (any scale)
        alpha:     blend weight for heatmap (0=frame only, 1=heatmap only)

    Returns:
        overlay: HxWx3 uint8 BGR — blended frame+heatmap
    """
    h, w = frame_bgr.shape[:2]

    # Resize error map to frame resolution
    err_resized = cv2.resize(error_map, (w, h), interpolation=cv2.INTER_LINEAR)

    # Normalize error map to [0, 255]
    e_min, e_max = err_resized.min(), err_resized.max()
    if e_max - e_min < 1e-8:
        err_norm = np.zeros_like(err_resized, dtype=np.uint8)
    else:
        err_norm = ((err_resized - e_min) / (e_max - e_min) * 255).astype(np.uint8)

    # Apply JET colormap
    heatmap = cv2.applyColorMap(err_norm, cv2.COLORMAP_JET)

    # Blend with original frame
    overlay = cv2.addWeighted(frame_bgr, 1 - alpha, heatmap, alpha, 0)
    return overlay


# ─────────────────────────────────────────────────────────────────────────────
#  Binary Anomaly Mask
# ─────────────────────────────────────────────────────────────────────────────

def make_binary_mask_overlay(
    frame_bgr: np.ndarray,
    error_map: np.ndarray,
    threshold_percentile: float = 90.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Threshold the error map to produce a binary anomaly mask,
    then draw contours of the largest anomalous regions onto the frame.

    Args:
        frame_bgr:            HxWx3 uint8 BGR frame
        error_map:            HxW float32 reconstruction error
        threshold_percentile: percentile of error_map used as binary threshold

    Returns:
        contour_frame: HxWx3 uint8 — frame with contours drawn
        binary_mask:   HxW uint8   — binary mask (0 or 255)
    """
    h, w = frame_bgr.shape[:2]
    err_resized = cv2.resize(error_map, (w, h), interpolation=cv2.INTER_LINEAR)

    thresh = np.percentile(err_resized, threshold_percentile)
    binary = (err_resized > thresh).astype(np.uint8) * 255

    # Morphological cleanup — remove tiny noise blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Draw contours on frame copy
    contour_frame = frame_bgr.copy()
    contours, _   = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Only draw contours for regions larger than 1% of frame area
    min_area = 0.01 * h * w
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(contour_frame, [cnt], -1, (0, 255, 255), 2)  # cyan

    return contour_frame, binary


# ─────────────────────────────────────────────────────────────────────────────
#  Running Score Graph Strip
# ─────────────────────────────────────────────────────────────────────────────

class ScoreGraphStrip:
    """
    Maintains a rolling window of anomaly scores and renders a score graph
    as a fixed-height BGR image strip to be pasted at the bottom of each frame.

    Args:
        frame_width:   width of the output video frame (pixels)
        strip_height:  height of the score strip (pixels)
        window_size:   number of frames shown in the rolling graph
        threshold:     normalized detection threshold (drawn as red dashed line)
    """

    def __init__(
        self,
        frame_width: int,
        strip_height: int = 80,
        window_size: int = 100,
        threshold: float = 0.5,
    ):
        self.frame_width  = frame_width
        self.strip_height = strip_height
        self.window_size  = window_size
        self.threshold    = threshold
        self.scores       = deque(maxlen=window_size)

    def update(self, score: float):
        self.scores.append(float(score))

    def render(self, is_anomalous: bool) -> np.ndarray:
        """
        Render the current score history as a BGR strip image.

        Returns:
            strip: strip_height x frame_width x 3 uint8 BGR image
        """
        dpi    = 100
        fig_w  = self.frame_width / dpi
        fig_h  = self.strip_height / dpi

        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        fig.patch.set_facecolor("#111111")
        ax.set_facecolor("#111111")

        scores_list = list(self.scores)
        xs          = list(range(len(scores_list)))

        if len(scores_list) > 1:
            # Color line segments: red if anomalous region, green otherwise
            line_color = "#FF4444" if is_anomalous else "#44FF88"
            ax.plot(xs, scores_list, color=line_color, linewidth=1.2)
            ax.fill_between(xs, scores_list, alpha=0.25, color=line_color)

        # Threshold line
        ax.axhline(self.threshold, color="#FF0000", linewidth=0.8,
                   linestyle="--", alpha=0.8)

        # Current score marker
        if scores_list:
            marker_color = "#FF4444" if is_anomalous else "#44FF88"
            ax.scatter([xs[-1]], [scores_list[-1]],
                       color=marker_color, s=18, zorder=5)

        ax.set_xlim(0, self.window_size)
        ax.set_ylim(-0.05, 1.15)
        ax.set_ylabel("Score", color="white", fontsize=5, labelpad=2)
        ax.tick_params(colors="white", labelsize=4, pad=1)
        ax.spines[:].set_color("#444444")
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

        plt.tight_layout(pad=0.2)

        # Render figure to numpy array
        fig.canvas.draw()
        buf   = fig.canvas.buffer_rgba()
        strip = np.asarray(buf, dtype=np.uint8).copy()
        plt.close(fig)

        # RGBA → BGR, resize to exact frame width
        strip_bgr = cv2.cvtColor(strip, cv2.COLOR_RGBA2BGR)
        strip_bgr = cv2.resize(strip_bgr, (self.frame_width, self.strip_height))
        return strip_bgr


# ─────────────────────────────────────────────────────────────────────────────
#  Text & Banner Overlays
# ─────────────────────────────────────────────────────────────────────────────

def draw_info_text(
    frame_bgr: np.ndarray,
    frame_idx: int,
    raw_score: float,
    norm_score: float,
) -> np.ndarray:
    """
    Draw frame index and anomaly score in the top-left corner.
    """
    out = frame_bgr.copy()
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness  = 1
    color      = (255, 255, 255)
    pad        = 6

    lines = [
        f"Frame : {frame_idx:04d}",
        f"Score : {norm_score:.3f}",
        f"MSE   : {raw_score:.5f}",
    ]
    y = pad + 14
    for line in lines:
        # Dark background for readability
        (tw, th), _ = cv2.getTextSize(line, font, font_scale, thickness)
        cv2.rectangle(out, (pad - 2, y - th - 2), (pad + tw + 2, y + 2),
                      (0, 0, 0), -1)
        cv2.putText(out, line, (pad, y), font, font_scale, color, thickness,
                    cv2.LINE_AA)
        y += th + 6

    return out


def draw_anomaly_banner(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Draw a red [ANOMALY DETECTED] banner across the top of the frame.
    """
    out   = frame_bgr.copy()
    h, w  = out.shape[:2]
    bh    = 26  # banner height px

    # Semi-transparent red bar
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, bh), (0, 0, 200), -1)
    cv2.addWeighted(overlay, 0.6, out, 0.4, 0, out)

    text  = "[ ANOMALY DETECTED ]"
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thick = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    tx = (w - tw) // 2
    ty = bh // 2 + th // 2
    cv2.putText(out, text, (tx, ty), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Composite Frame Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_composite_frame(
    frame_bgr: np.ndarray,
    error_map: np.ndarray,
    norm_score: float,
    raw_score: float,
    frame_idx: int,
    threshold: float,
    score_strip: ScoreGraphStrip,
    heatmap_alpha: float = 0.5,
    mask_percentile: float = 90.0,
) -> np.ndarray:
    """
    Assemble the full annotated output frame:
      1. Heatmap overlay
      2. Binary mask contours
      3. Info text (top-left)
      4. Anomaly banner (if anomalous)
      5. Score graph strip (bottom)

    Args:
        frame_bgr:       original HxWx3 BGR frame
        error_map:       HxW float32 reconstruction error
        norm_score:      per-sequence normalized score [0,1]
        raw_score:       raw MSE score
        frame_idx:       frame index for display
        threshold:       detection threshold on normalized score
        score_strip:     ScoreGraphStrip instance (already updated this frame)
        heatmap_alpha:   blend weight for heatmap
        mask_percentile: percentile for binary mask threshold

    Returns:
        composite: (H + strip_height) x W x 3 uint8 BGR
    """
    is_anomalous = norm_score >= threshold

    # 1. Heatmap overlay
    out = make_heatmap_overlay(frame_bgr, error_map, alpha=heatmap_alpha)

    # 2. Binary mask contours
    out, _ = make_binary_mask_overlay(out, error_map,
                                      threshold_percentile=mask_percentile)

    # 3. Info text
    out = draw_info_text(out, frame_idx, raw_score, norm_score)

    # 4. Anomaly banner
    if is_anomalous:
        out = draw_anomaly_banner(out)

    # 5. Score strip
    strip = score_strip.render(is_anomalous)
    composite = np.vstack([out, strip])

    return composite


# ─────────────────────────────────────────────────────────────────────────────
#  Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yaml, os

    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    h, w        = config["dataset"]["resolution"]
    alpha       = config["visualization"]["heatmap_alpha"]
    strip_h     = 80
    window_size = config["inference"]["score_graph_frames"]

    # Dummy frame and error map
    dummy_frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    dummy_error = np.random.rand(h, w).astype(np.float32)

    strip = ScoreGraphStrip(
        frame_width=w,
        strip_height=strip_h,
        window_size=window_size,
        threshold=0.5,
    )

    # Simulate 20 frames
    for i in range(20):
        score = float(np.random.rand())
        strip.update(score)

    composite = build_composite_frame(
        frame_bgr=dummy_frame,
        error_map=dummy_error,
        norm_score=0.75,
        raw_score=0.42,
        frame_idx=42,
        threshold=0.5,
        score_strip=strip,
        heatmap_alpha=alpha,
    )

    os.makedirs("outputs/heatmaps", exist_ok=True)
    out_path = "outputs/heatmaps/smoke_test_frame.png"
    cv2.imwrite(out_path, composite)
    print(f"Composite frame shape : {composite.shape}")
    print(f"Saved smoke test frame → {out_path}")
    print("visualization.py smoke test passed.")