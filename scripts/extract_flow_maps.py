"""
Pre-extract Farnebäck optical flow magnitude maps for all training frames.
Saves flow maps as .npy files alongside the .jpg frames so Colab can load
dual-channel inputs without recomputing flow during training.

Output structure mirrors the frames/ folder:
  data/combined_normal/flow/Train001/0000.npy
  data/combined_normal/flow/Train001/0001.npy
  ...
  data/combined_normal/flow/P2_Test001/0000.npy
  ...

First frame of each sequence gets a zero flow map (no previous frame).
"""

import os
import glob
import yaml
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image


def extract_flow_for_split(train_dir: str, flow_out_dir: str, resolution: list):
    h, w = resolution
    seq_dirs = sorted([
        d for d in glob.glob(os.path.join(train_dir, "*"))
        if os.path.isdir(d)
    ])

    print(f"[extract_flow] Processing {len(seq_dirs)} sequences from {train_dir}")
    total = 0

    for seq_dir in tqdm(seq_dirs, desc="Sequences"):
        seq_name = os.path.basename(seq_dir)
        out_seq_dir = os.path.join(flow_out_dir, seq_name)
        os.makedirs(out_seq_dir, exist_ok=True)

        frame_paths = sorted(glob.glob(os.path.join(seq_dir, "*.jpg")))
        prev_gray   = None

        for idx, path in enumerate(frame_paths):
            img      = Image.open(path).convert("L").resize((w, h), Image.BILINEAR)
            curr_gray = np.array(img, dtype=np.uint8)

            if prev_gray is None:
                # First frame — zero flow
                flow_mag = np.zeros((h, w), dtype=np.float32)
            else:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, curr_gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
                )
                mag   = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                p99   = np.percentile(mag, 99)
                flow_mag = np.clip(mag / p99, 0.0, 1.0).astype(np.float32) if p99 > 0 \
                           else np.zeros_like(mag, dtype=np.float32)

            out_path = os.path.join(out_seq_dir, f"{idx:04d}.npy")
            np.save(out_path, flow_mag)
            prev_gray = curr_gray
            total += 1

    print(f"  → Saved {total} flow maps to {flow_out_dir}")
    return total


if __name__ == "__main__":
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    train_dir    = config["dataset"]["train_dir"]   # data/combined_normal/frames
    flow_out_dir = train_dir.replace("frames", "flow")
    resolution   = config["dataset"]["resolution"]

    os.makedirs(flow_out_dir, exist_ok=True)
    n = extract_flow_for_split(train_dir, flow_out_dir, resolution)

    print(f"\n[DONE] {n} flow maps saved to {flow_out_dir}")
    print(f"Next: zip both frames/ and flow/ together and upload to Colab.")