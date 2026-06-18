"""
preprocess.py  —  Run once before challenge.py

Produces two cache files:
  dataset/preprocessed/train.pt   — training set (images + labels + masks + ground_truths)
  dataset/preprocessed/test.pt    — test set     (images only, no ground truth)

Expected folder layout:
  dataset/
    train/
      images/         *.jpg
      scribbles/      *.png   (0=bg, 1=fg, 255=unlabeled)
      ground_truth/   *.png
    test/
      images/         *.jpg
      scribbles/      *.png
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import slic
from sklearn.semi_supervised import LabelSpreading

# ── Paths ─────────────────────────────────────────────────────────────────────
TRAIN_IMG_DIR  = "dataset/train/images"
TRAIN_SCR_DIR  = "dataset/train/scribbles"
TRAIN_GT_DIR   = "dataset/train/ground_truth"
TRAIN_OUT      = "dataset/preprocessed/train.pt"

TEST_IMG_DIR   = "dataset/test/images"
TEST_SCR_DIR   = "dataset/test/scribbles"
TEST_OUT       = "dataset/preprocessed/test.pt"


# ── File helpers ──────────────────────────────────────────────────────────────

def stems_in(folder: str, exts: tuple) -> dict:
    return {
        os.path.splitext(f)[0]: os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if f.lower().endswith(exts) and not f.startswith(".")
    }


# ── Label propagation ─────────────────────────────────────────────────────────

def propagate_labels(
    image: np.ndarray,
    scribble: np.ndarray,
    n_segments: int = 2500,
    gamma: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Use SLIC superpixels and LabelSpreading to propagate scribble labels
    across the full image.

    Returns:
        prob_fg  (H, W) float32 — foreground probability per pixel
        conf     (H, W) float32 — confidence (distance from the 0.5 boundary)
    """
    H, W, _ = image.shape
    yy, xx  = np.mgrid[0:H, 0:W]

    segs     = slic(image, n_segments=n_segments, compactness=12, start_label=0)
    n_segs   = int(segs.max()) + 1

    feats, labels = [], []
    for sid in range(n_segs):
        px  = segs == sid
        rgb = image[px].mean(axis=0) / 255.0
        xc  = xx[px].mean() / W
        yc  = yy[px].mean() / H
        feats.append([*rgb, xc, yc])

        scr_vals = scribble[px]
        scr_vals = scr_vals[scr_vals != 255]
        labels.append(int(scr_vals[0]) if len(scr_vals) else -1)

    X = np.array(feats, dtype=np.float32)
    y = np.array(labels, dtype=int)

    labeled = y[y != -1]
    n_classes = len(np.unique(labeled)) if labeled.size else 0

    if n_classes < 2:
        # Not enough labelled classes — seed directly from scribbles
        p_fg = np.where(y == 1, 1.0, np.where(y == 0, 0.0, 0.5)).astype(np.float32)
        conf = np.where(y != -1, 1.0, 0.0).astype(np.float32)
    else:
        lp = LabelSpreading(kernel="rbf", gamma=gamma, max_iter=50)
        lp.fit(X, y)
        probs   = lp.label_distributions_
        classes = lp.classes_
        fg_col  = int(np.where(classes == 1)[0][0]) if (classes == 1).any() else -1
        p_fg    = probs[:, fg_col].astype(np.float32) if fg_col >= 0 else np.full(n_segs, 0.0, np.float32)
        conf    = (np.abs(p_fg - 0.5) * 2.0).astype(np.float32)

    # Map superpixel values back to pixel grid
    prob_map = np.empty((H, W), np.float32)
    conf_map = np.empty((H, W), np.float32)
    for sid in range(n_segs):
        px = segs == sid
        prob_map[px] = p_fg[sid]
        conf_map[px] = conf[sid]

    return prob_map, conf_map


# ── Feature building ──────────────────────────────────────────────────────────

def scribble_distance_channels(scribble: np.ndarray, sigma: float = 100.0) -> np.ndarray:
    """
    Two channels encoding proximity to fg / bg scribble pixels.
    Values decay exponentially with distance — peak 1 at scribble, ~0 far away.
    """
    d_fg = distance_transform_edt(scribble != 1)
    d_bg = distance_transform_edt(scribble != 0)
    return np.stack([
        np.exp(-d_fg / sigma).astype(np.float32),
        np.exp(-d_bg / sigma).astype(np.float32),
    ])   # (2, H, W)


def build_weight_map(conf: np.ndarray, scribble: np.ndarray, tau: float = 150.0) -> np.ndarray:
    """
    Per-pixel training weight: maximum of
      - 1.0 at any scribble pixel (always trust the annotation)
      - confidence × proximity-to-fg-scribble (trust propagated labels nearby)
    """
    seeded    = (scribble != 255).astype(np.float32)
    d_fg      = distance_transform_edt(scribble != 1)
    proximity = np.exp(-d_fg / tau)
    return np.maximum(seeded, conf * proximity).astype(np.float32)


def build_tensor(image_pil, scribble_pil):
    """Build (img_t, lbl_t, mask_t) tensors for one image."""
    img = np.array(image_pil)          # (H, W, 3) uint8
    scr = np.array(scribble_pil)       # (H, W)

    prob_fg, conf = propagate_labels(img, scr)
    weight        = build_weight_map(conf, scr)
    scr_channels  = scribble_distance_channels(scr)

    rgb  = img.transpose(2, 0, 1).astype(np.float32) / 255.0   # (3, H, W)
    x5   = np.concatenate([rgb, scr_channels], axis=0)          # (5, H, W)

    img_t  = torch.from_numpy(x5)
    lbl_t  = torch.from_numpy(prob_fg[None])                    # (1, H, W)
    mask_t = torch.from_numpy(weight[None])                     # (1, H, W)
    return img_t, lbl_t, mask_t


def process_training(img_dir, scr_dir, gt_dir, out_path):
    """Preprocess training split — has images, scribbles, and ground truth."""
    if os.path.exists(out_path):
        print(f"Training cache exists: {out_path}")
        return

    img_map = stems_in(img_dir, (".jpg", ".jpeg", ".png"))
    scr_map = stems_in(scr_dir, (".png",))
    gt_map  = stems_in(gt_dir,  (".png",))

    stems = sorted(set(img_map) & set(scr_map) & set(gt_map))
    if not stems:
        raise RuntimeError(f"No matching triplets in {img_dir}, {scr_dir}, {gt_dir}")
    print(f"Training: {len(stems)} images")

    all_imgs, all_lbls, all_msks, all_gts, all_names = [], [], [], [], []

    for stem in tqdm(stems, desc="Training"):
        img_t, lbl_t, msk_t = build_tensor(
            Image.open(img_map[stem]).convert("RGB"),
            Image.open(scr_map[stem]).convert("L"),
        )
        gt_arr = np.array(Image.open(gt_map[stem]))
        gt_t   = torch.from_numpy((gt_arr == 1).astype(np.float32)[None])

        all_imgs.append(img_t)
        all_lbls.append(lbl_t)
        all_msks.append(msk_t)
        all_gts.append(gt_t)
        all_names.append(stem)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({
        "images":        torch.stack(all_imgs),
        "labels":        torch.stack(all_lbls),
        "masks":         torch.stack(all_msks),
        "ground_truths": torch.stack(all_gts),
        "filenames":     all_names,
    }, out_path)
    print(f"Saved training cache → {out_path}")


def process_test(img_dir, scr_dir, out_path):
    """Preprocess test split — has images and scribbles only (no ground truth)."""
    if os.path.exists(out_path):
        print(f"Test cache exists: {out_path}")
        return

    img_map = stems_in(img_dir, (".jpg", ".jpeg", ".png"))
    scr_map = stems_in(scr_dir, (".png",))

    stems = sorted(set(img_map) & set(scr_map))
    if not stems:
        raise RuntimeError(f"No matching pairs in {img_dir}, {scr_dir}")
    print(f"Test: {len(stems)} images")

    all_imgs, all_lbls, all_msks, all_names = [], [], [], []

    for stem in tqdm(stems, desc="Test"):
        img_t, lbl_t, msk_t = build_tensor(
            Image.open(img_map[stem]).convert("RGB"),
            Image.open(scr_map[stem]).convert("L"),
        )
        all_imgs.append(img_t)
        all_lbls.append(lbl_t)
        all_msks.append(msk_t)
        all_names.append(stem)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({
        "images":    torch.stack(all_imgs),
        "labels":    torch.stack(all_lbls),
        "masks":     torch.stack(all_msks),
        "filenames": all_names,
    }, out_path)
    print(f"Saved test cache → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    process_training(TRAIN_IMG_DIR, TRAIN_SCR_DIR, TRAIN_GT_DIR, TRAIN_OUT)
    process_test(TEST_IMG_DIR, TEST_SCR_DIR, TEST_OUT)
