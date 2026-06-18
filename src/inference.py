"""
src/inference.py  —  Inference and post-processing

run_inference : forward pass over a tensor stack → probability maps
refine_mask   : edge-aware binary refinement using bilateral filter +
                morphological ops (replaces pydensecrf; Python 3.13 safe)
"""

import numpy as np
import torch
import cv2
from scipy.ndimage import binary_closing, binary_opening


def _best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_inference(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device | None = None,
) -> np.ndarray:
    """
    Run the model over a stack of images and return probability maps.

    Args:
        model:  Trained SegmentationNet.
        images: (N, 5, H, W) tensor.
        device: Inference device — auto-detected if None.

    Returns:
        (N, H, W) float32 array of foreground probabilities in [0, 1].
    """
    device = device or _best_device()
    model  = model.to(device).eval()

    probs = []
    with torch.no_grad():
        for i in range(len(images)):
            x      = images[i].unsqueeze(0).to(device)
            prob   = torch.sigmoid(model(x))[0, 0].cpu().numpy()
            probs.append(prob)

    return np.stack(probs)   # (N, H, W)


def refine_mask(
    image_rgb: np.ndarray,
    prob_map: np.ndarray,
    morph_radius: int = 5,
) -> np.ndarray:
    """
    Edge-aware refinement of a probability map.

    Steps:
      1. Bilateral filter — smooths the probability map while preserving
         colour edges (equivalent to the CRF bilateral pairwise term).
      2. Threshold at 0.5.
      3. Morphological closing — fills small holes inside the foreground.
      4. Morphological opening  — removes isolated specks of noise.

    Args:
        image_rgb:   (H, W, 3) uint8 RGB image — used for edge cues.
        prob_map:    (H, W) float32 probability map in [0, 1].
        morph_radius:Radius of the morphological structuring element.

    Returns:
        (H, W) uint8 binary mask — 0 (background) or 1 (foreground).
    """
    # Bilateral filter on the probability map
    prob_u8   = (prob_map * 255).clip(0, 255).astype(np.uint8)
    smoothed  = cv2.bilateralFilter(prob_u8, d=-1, sigmaColor=30, sigmaSpace=15)
    prob_smooth = smoothed.astype(np.float32) / 255.0

    binary = prob_smooth >= 0.5

    # Morphological cleanup
    r      = max(1, morph_radius)
    disk   = np.ones((2 * r + 1, 2 * r + 1), dtype=bool)
    binary = binary_closing(binary, structure=disk)
    binary = binary_opening(binary, structure=np.ones((3, 3), dtype=bool))

    return binary.astype(np.uint8)
