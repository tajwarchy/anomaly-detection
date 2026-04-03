"""
Extract normal frames from UCSD Ped2 test sequences for in-domain training.

A frame is considered normal if its GT mask is all-zero.
These frames are saved to data/ucsd_ped2/training/frames/ and used
as training data for the retrained CAE — eliminating the Ped1→Ped2 domain gap.
"""

import os
import glob
import shutil
import cv2
import yaml
import numpy as np
from tqdm import tqdm


def extract_ped2_normal_frames(config: dict):
    test_frames_dir = config["dataset"]["test_dir"]         # testing/frames/
    gt_mask_dir     = config["dataset"]["gt_mask_dir"]      # testing/gt_masks/
    out_dir         = "data/ucsd_ped2/training/frames"      # new training source

    os.makedirs(out_dir, exist_ok=True)

    # Get all test sequences
    seq_dirs = sorted([
        d for d in glob.glob(os.path.join(test_frames_dir, "*"))
        if os.path.isdir(d)
    ])

    total_normal = 0
    total_skipped = 0

    print(f"[extract_ped2_normal] Processing {len(seq_dirs)} test sequences...")

    for seq_dir in tqdm(seq_dirs, desc="Sequences"):
        seq_name = os.path.basename(seq_dir)
        gt_seq_dir = os.path.join(gt_mask_dir, seq_name)
        out_seq_dir = os.path.join(out_dir, seq_name)
        os.makedirs(out_seq_dir, exist_ok=True)

        frame_paths = sorted(glob.glob(os.path.join(seq_dir, "*.jpg")))

        for frame_path in frame_paths:
            frame_name = os.path.basename(frame_path)          # e.g. 0000.jpg
            mask_name  = frame_name.replace(".jpg", ".png")
            mask_path  = os.path.join(gt_seq_dir, mask_name)

            # Check if GT mask exists and is all-zero (normal frame)
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None and mask.max() > 0:
                    total_skipped += 1
                    continue  # anomalous frame — skip

            # Normal frame — copy to training dir
            out_path = os.path.join(out_seq_dir, frame_name)
            shutil.copy2(frame_path, out_path)
            total_normal += 1

    print(f"\n[extract_ped2_normal] Done.")
    print(f"  Normal frames saved : {total_normal}")
    print(f"  Anomalous skipped   : {total_skipped}")
    print(f"  Output dir          : {out_dir}")
    return total_normal


if __name__ == "__main__":
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    n = extract_ped2_normal_frames(config)
    if n < 200:
        print("\n[WARNING] Very few normal frames extracted. "
              "Check GT mask alignment before retraining.")
    else:
        print(f"\n[OK] {n} normal frames ready for in-domain retraining.")