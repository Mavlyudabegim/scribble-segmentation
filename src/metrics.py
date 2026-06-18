"""
src/metrics.py  —  Evaluation metrics

Two interfaces for the same concept (mean IoU over fg and bg classes):

  batch_iou  : operates on torch tensors — used inside the training loop
               so no numpy conversion overhead per batch
  dataset_iou: operates on numpy arrays  — used after inference for the
               final printed summary
"""

import numpy as np
import torch


def batch_iou(logits: torch.Tensor, gt: torch.Tensor, eps: float = 1e-7) -> dict:
    """
    Compute fg IoU, bg IoU, and their mean for a batch.

    Args:
        logits: (B, 1, H, W) raw logits.
        gt:     (B, 1, H, W) float binary ground truth.

    Returns:
        dict with keys 'fg', 'bg', 'mean'.
    """
    pred = (torch.sigmoid(logits) > 0.5).float()

    inter_fg = (pred * gt).sum(dim=(1, 2, 3))
    union_fg = ((pred + gt) > 0).float().sum(dim=(1, 2, 3))
    iou_fg   = (inter_fg / (union_fg + eps)).mean().item()

    inv_pred = 1.0 - pred
    inv_gt   = 1.0 - gt
    inter_bg = (inv_pred * inv_gt).sum(dim=(1, 2, 3))
    union_bg = ((inv_pred + inv_gt) > 0).float().sum(dim=(1, 2, 3))
    iou_bg   = (inter_bg / (union_bg + eps)).mean().item()

    return {"fg": iou_fg, "bg": iou_bg, "mean": (iou_fg + iou_bg) / 2}


def dataset_iou(ground_truths, predictions: np.ndarray) -> None:
    """
    Print per-class and mean IoU across the full dataset.

    Args:
        ground_truths: Tensor (N, 1, H, W) or list of (H, W) arrays.
        predictions:   (N, H, W) uint8 binary masks.
    """
    fg_scores, bg_scores = [], []

    for gt, pred in zip(ground_truths, predictions):
        if hasattr(gt, "numpy"):
            gt = gt.squeeze().numpy()
        gt   = np.asarray(gt, dtype=np.uint8)
        pred = np.asarray(pred, dtype=np.uint8)

        for label, fg_scores_list, bg_scores_list in [
            (1, fg_scores, None),
            (0, None, bg_scores),
        ]:
            p_mask = pred == label
            g_mask = gt   == label
            inter  = np.logical_and(p_mask, g_mask).sum()
            union  = np.logical_or( p_mask, g_mask).sum()
            score  = 1.0 if union == 0 else inter / union
            if label == 1:
                fg_scores.append(score)
            else:
                bg_scores.append(score)

    mean_fg = np.mean(fg_scores)
    mean_bg = np.mean(bg_scores)
    print(
        f"FG IoU: {mean_fg:.4f}  "
        f"BG IoU: {mean_bg:.4f}  "
        f"Mean IoU: {(mean_fg + mean_bg) / 2:.4f}"
    )
