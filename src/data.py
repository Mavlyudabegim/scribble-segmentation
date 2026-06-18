"""
src/data.py  —  I/O utilities

Provides two things:
  - make_loader : wraps stacked tensors in a DataLoader
  - save_masks  : writes predicted binary masks as palette PNGs
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import TensorDataset, DataLoader


def make_loader(
    tensor_tuple: tuple[torch.Tensor, ...],
    batch_size: int = 2,
    shuffle: bool = False,
) -> DataLoader:
    """
    Wrap a tuple of equal-length tensors in a DataLoader.

    Every element of the tuple becomes one output per batch iteration,
    so (images, labels, masks, gts) yields four tensors per step.
    """
    return DataLoader(
        TensorDataset(*tensor_tuple),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
    )


def save_masks(
    masks: np.ndarray,
    stems: list[str],
    out_dir: str = "dataset/predictions",
    palette: list[int] | None = None,
    palette_source_dir: str | None = None,
) -> None:
    """
    Save binary segmentation masks as palette PNG files.

    The colour palette can be supplied directly (preferred — avoids
    assuming any particular filename exists in palette_source_dir), or
    loaded from the first file found in palette_source_dir as a fallback.

    Args:
        masks:               (N, H, W) uint8 array — values 0 or 1.
        stems:                Filename stems (no extension), one per mask.
        out_dir:              Destination folder.
        palette:              Explicit palette list (e.g. from a known
                               training ground-truth image). Preferred.
        palette_source_dir:   Fallback — folder to scan for any PNG to
                               read a palette from, used only if `palette`
                               is not provided.
    """
    os.makedirs(out_dir, exist_ok=True)

    if palette is None:
        if palette_source_dir is None:
            raise ValueError("Provide either `palette` or `palette_source_dir`.")
        # Use the first PNG actually present in the folder — not stems[0],
        # which may belong to a different split (e.g. test stems don't
        # exist in the training ground-truth folder).
        any_png = next(
            f for f in sorted(os.listdir(palette_source_dir)) if f.lower().endswith(".png")
        )
        palette = Image.open(os.path.join(palette_source_dir, any_png)).getpalette()

    for stem, mask in zip(stems, masks):
        img = Image.fromarray(mask.astype(np.uint8), mode="P")
        img.putpalette(palette)
        img.save(os.path.join(out_dir, stem + ".png"))

    print(f"Saved {len(masks)} masks → {out_dir}/")