"""
challenge.py  —  Training, evaluation, and prediction pipeline

Flow:
  1. Load preprocessed training pack — split 80/20 train/val.
  2. Train SegmentationNet with hybrid loss + early stopping on val mIoU.
  3. Inference on validation split → refine → evaluate → visualise.
  4. Load preprocessed test pack → predict → save (no ground truth needed).
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from PIL import Image

from src.data      import make_loader, save_masks
from src.model     import SegmentationNet
from src.loss      import total_loss
from src.metrics   import batch_iou, dataset_iou
from src.inference import run_inference, refine_mask
from src.visualise import show_result


# ── Config ────────────────────────────────────────────────────────────────────

BATCH_SIZE  = 2
EPOCHS      = 20
LR          = 1e-4
LAM         = 0.3
PATIENCE    = 5
CHECKPOINT  = "best_model.pth"

TRAIN_PACK  = "dataset/preprocessed/train.pt"
TEST_PACK   = "dataset/preprocessed/test.pt"

TRAIN_IMG_DIR = "dataset/train/images"
TRAIN_SCR_DIR = "dataset/train/scribbles"
TRAIN_GT_DIR  = "dataset/train/ground_truth"
TRAIN_PRED_DIR = "dataset/train/predictions"

TEST_IMG_DIR  = "dataset/test/images"
TEST_PRED_DIR = "dataset/test/predictions"


# ════════════════════════════════════════════════════════════════════════════════
# PART 1 — Training
# ════════════════════════════════════════════════════════════════════════════════

pack  = torch.load(TRAIN_PACK, map_location="cpu")
imgs  = pack["images"].float()
lbls  = pack["labels"].float()
msks  = pack["masks"].float()
gtrs  = pack["ground_truths"].float()
names = pack["filenames"]

# 80 / 20 split
all_idx = np.arange(len(imgs))
tr_idx, va_idx = train_test_split(all_idx, test_size=0.2, random_state=42, shuffle=True)

tr_loader = make_loader(
    (imgs[tr_idx], lbls[tr_idx], msks[tr_idx], gtrs[tr_idx]),
    batch_size=BATCH_SIZE, shuffle=True,
)
va_loader = make_loader(
    (imgs[va_idx], lbls[va_idx], msks[va_idx], gtrs[va_idx]),
    batch_size=BATCH_SIZE, shuffle=False,
)

net       = SegmentationNet(in_channels=5, pretrained=True)
optimiser = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1e-4)

best_val_miou   = 0.0
epochs_stagnant = 0

for epoch in range(EPOCHS):
    net.train()
    tr_loss_sum = 0.0
    for x, lbl, msk, gt in tr_loader:
        optimiser.zero_grad()
        loss = total_loss(net(x), lbl, msk, gt, lam=LAM)
        loss.backward()
        optimiser.step()
        tr_loss_sum += loss.item()

    net.eval()
    va_loss_sum = va_iou_sum = va_batches = 0.0
    with torch.no_grad():
        for x, lbl, msk, gt in va_loader:
            logits       = net(x)
            va_loss_sum += total_loss(logits, lbl, msk, gt, lam=LAM).item()
            va_iou_sum  += batch_iou(logits, gt)["mean"]
            va_batches  += 1

    tr_loss = tr_loss_sum / max(1, len(tr_loader))
    va_loss = va_loss_sum / max(1, va_batches)
    va_miou = va_iou_sum  / max(1, va_batches)

    print(
        f"Epoch {epoch:02d}  "
        f"train_loss={tr_loss:.4f}  "
        f"val_loss={va_loss:.4f}  "
        f"val_mIoU={va_miou:.4f}"
    )

    if va_miou > best_val_miou:
        best_val_miou, epochs_stagnant = va_miou, 0
        torch.save(net.state_dict(), CHECKPOINT)
        print(f"  ✓ checkpoint saved (val_mIoU={best_val_miou:.4f})")
    else:
        epochs_stagnant += 1
        if epochs_stagnant >= PATIENCE:
            print(f"Early stopping — no improvement for {PATIENCE} epochs.")
            break

net.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
net.eval()


# ════════════════════════════════════════════════════════════════════════════════
# PART 2 — Validation evaluation
# ════════════════════════════════════════════════════════════════════════════════

va_idx_list = va_idx.tolist()
va_names    = [names[i] for i in va_idx_list]
va_gtrs     = gtrs[va_idx_list]

raw_probs = run_inference(net, imgs[va_idx_list])

refined_val = np.stack([
    refine_mask(
        np.array(Image.open(os.path.join(TRAIN_IMG_DIR, n + ".jpg")).convert("RGB")),
        raw_probs[i],
    )
    for i, n in enumerate(va_names)
])

print("\n── Validation set evaluation ──")
dataset_iou(va_gtrs, refined_val)

# Visualise one random example
pick = np.random.randint(len(va_names))
show_result(
    image=        np.array(Image.open(os.path.join(TRAIN_IMG_DIR, va_names[pick] + ".jpg")).convert("RGB")),
    scribble=     np.array(Image.open(os.path.join(TRAIN_SCR_DIR, va_names[pick] + ".png")).convert("L")),
    ground_truth= va_gtrs[pick].squeeze(0).numpy().astype(np.uint8),
    prediction=   refined_val[pick],
)

# Load the colour palette once from a known training GT file — reused
# for saving both training and test predictions.
_palette_file = sorted(
    f for f in os.listdir(TRAIN_GT_DIR) if f.lower().endswith(".png")
)[0]
palette = Image.open(os.path.join(TRAIN_GT_DIR, _palette_file)).getpalette()

# Save training predictions (full set)
all_probs  = run_inference(net, imgs)
all_binary = (all_probs > 0.5).astype(np.uint8)
save_masks(all_binary, names, out_dir=TRAIN_PRED_DIR, palette=palette)


# ════════════════════════════════════════════════════════════════════════════════
# PART 3 — Test set prediction (no ground truth)
# ════════════════════════════════════════════════════════════════════════════════

print("\n── Test set prediction ──")

test_pack   = torch.load(TEST_PACK, map_location="cpu")
test_imgs   = test_pack["images"].float()
test_names  = test_pack["filenames"]

test_probs = run_inference(net, test_imgs)

refined_test = np.stack([
    refine_mask(
        np.array(Image.open(os.path.join(TEST_IMG_DIR, n + ".jpg")).convert("RGB")),
        test_probs[i],
    )
    for i, n in enumerate(test_names)
])

# Reuse the same training GT palette for consistent palette PNGs
save_masks(refined_test, test_names, out_dir=TEST_PRED_DIR, palette=palette)

print("Done.")