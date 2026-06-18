"""
src/visualise.py  —  Visualisation

show_result : displays image+scribbles | ground truth | prediction
              in a 1×3 matplotlib figure (blue = background, red = foreground)
"""

import numpy as np
import matplotlib.pyplot as plt


def _blend_scribbles(
    image: np.ndarray,
    scribble: np.ndarray,
    alpha: float,
) -> np.ndarray:
    out = image.astype(np.float32)
    for label, colour in [(1, (255, 0, 0)), (0, (0, 0, 255))]:
        mask = scribble == label
        for ch in range(3):
            out[..., ch][mask] = alpha * colour[ch] + (1 - alpha) * out[..., ch][mask]
    return out.clip(0, 255).astype(np.uint8)


def show_result(
    image: np.ndarray,
    scribble: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    alpha: float = 0.6,
) -> None:
    """
    Plot three panels side by side.

    Args:
        image:        (H, W, 3) uint8 RGB.
        scribble:     (H, W) with values 0 / 1 / 255.
        ground_truth: (H, W) binary mask.
        prediction:   (H, W) binary mask.
        alpha:        Scribble overlay strength.
    """
    cmap = plt.get_cmap("bwr")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(_blend_scribbles(image, scribble, alpha))
    axes[0].set_title("Image + scribbles")

    axes[1].imshow(ground_truth, cmap=cmap, vmin=0, vmax=1)
    axes[1].set_title("Ground truth")

    axes[2].imshow(prediction, cmap=cmap, vmin=0, vmax=1)
    axes[2].set_title("Prediction")

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()
