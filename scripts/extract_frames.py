import os
import cv2
import glob
import yaml
import numpy as np
from tqdm import tqdm
from PIL import Image


def extract_frames(split: str, config: dict):
    # split is "train" or "test" — matches config keys train_dir / test_dir
    base_dir = config["dataset"][f"{split}_dir"]
    raw_dir = os.path.dirname(base_dir)
    h, w = config["dataset"]["resolution"]

    seq_dirs = sorted([
        d for d in glob.glob(os.path.join(raw_dir, "*"))
        if os.path.isdir(d) and not d.endswith("_gt") and not d.endswith("frames")
    ])

    print(f"\n[{split.upper()}] Found {len(seq_dirs)} sequences in {raw_dir}")
    total = 0

    for seq_dir in tqdm(seq_dirs, desc=f"Extracting {split} frames"):
        seq_name = os.path.basename(seq_dir)
        out_dir = os.path.join(base_dir, seq_name)
        os.makedirs(out_dir, exist_ok=True)

        tif_files = sorted(glob.glob(os.path.join(seq_dir, "*.tif")))
        for idx, tif_path in enumerate(tif_files):
            img = Image.open(tif_path).convert("L")
            img = img.resize((w, h), Image.BILINEAR)
            out_path = os.path.join(out_dir, f"{idx:04d}.jpg")
            img.save(out_path, quality=95)
            total += 1

    print(f"  → Saved {total} frames to {base_dir}")


def extract_gt_masks(config: dict):
    test_raw_dir = os.path.dirname(config["dataset"]["test_dir"])
    gt_out_dir = os.path.join(test_raw_dir, "gt_masks")
    os.makedirs(gt_out_dir, exist_ok=True)
    h, w = config["dataset"]["resolution"]

    gt_dirs = sorted(glob.glob(os.path.join(test_raw_dir, "*_gt")))
    print(f"\n[GT MASKS] Found {len(gt_dirs)} ground truth folders")

    for gt_dir in tqdm(gt_dirs, desc="Extracting GT masks"):
        seq_name = os.path.basename(gt_dir).replace("_gt", "")
        out_seq_dir = os.path.join(gt_out_dir, seq_name)
        os.makedirs(out_seq_dir, exist_ok=True)

        bmp_files = sorted(glob.glob(os.path.join(gt_dir, "*.bmp")))
        for idx, bmp_path in enumerate(bmp_files):
            mask = cv2.imread(bmp_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.uint8) * 255
            out_path = os.path.join(out_seq_dir, f"{idx:04d}.png")
            cv2.imwrite(out_path, mask)

    print(f"  → GT masks saved to {gt_out_dir}")


if __name__ == "__main__":
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    extract_frames("train", config)   # uses config["dataset"]["train_dir"]
    extract_frames("test", config)    # uses config["dataset"]["test_dir"]
    extract_gt_masks(config)
    print("\n[DONE] Frame extraction complete.")