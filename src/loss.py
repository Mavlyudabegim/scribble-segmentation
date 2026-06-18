"""
src/loss.py  —  Loss functions

Three building blocks:
  soft_dice    : overlap-based; stable when fg/bg ratio is extreme
  focal        : down-weights easy pixels, focuses training on boundaries
  supervised   : focal + dice combined, with a per-pixel weight map

The training objective is a weighted sum of two supervised terms:
  - one computed against the soft scribble-propagated labels (weak signal)
  - one computed against the hard ground-truth labels      (strong signal)

The mixing ratio lam controls how much we trust the scribble branch.
"""

import torch
import torch.nn.functional as F


def soft_dice(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    smooth: float = 1e-6,
) -> torch.Tensor:
    prob = torch.sigmoid(logits) * weight
    tgt  = target * weight
    num  = 2.0 * (prob * tgt).sum(dim=(1, 2, 3)) + smooth
    den  = prob.sum(dim=(1, 2, 3)) + tgt.sum(dim=(1, 2, 3)) + smooth
    return (1.0 - num / den).mean()


def focal(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    prob   = torch.clamp(torch.sigmoid(logits), eps, 1 - eps) * weight
    tgt    = target * weight
    p_t    = torch.where(tgt == 1, prob, 1 - prob)
    a_t    = torch.where(tgt == 1,
                         torch.full_like(prob, alpha),
                         torch.full_like(prob, 1 - alpha))
    return (-a_t * (1 - p_t) ** gamma * torch.log(p_t)).mean()


def supervised(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Combined focal + dice, weighted by a per-pixel map."""
    return focal(logits, target, weight) + soft_dice(logits, target, weight)


def total_loss(
    logits: torch.Tensor,
    soft_labels: torch.Tensor,
    weight_map: torch.Tensor,
    ground_truth: torch.Tensor,
    lam: float = 0.3,
) -> torch.Tensor:
    """
    Weighted combination of scribble-supervised and GT-supervised losses.

    Args:
        logits:       (B, 1, H, W) raw model output.
        soft_labels:  (B, 1, H, W) propagated scribble probabilities.
        weight_map:   (B, 1, H, W) per-pixel reliability weights.
        ground_truth: (B, 1, H, W) binary ground-truth masks.
        lam:          Weight for scribble branch (1-lam goes to GT branch).
    """
    scribble_loss = supervised(logits, soft_labels, weight_map)
    gt_loss       = supervised(logits, ground_truth, torch.ones_like(ground_truth))
    return lam * scribble_loss + (1.0 - lam) * gt_loss
