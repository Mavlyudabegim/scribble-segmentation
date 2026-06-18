"""
src/model.py  —  Segmentation model

UNet with a pretrained ResNet-34 encoder.

Design decisions:
  - 5 input channels: RGB (3) + distance-to-fg-scribble (1) + distance-to-bg-scribble (1)
    The extra channels give the network spatial scribble context without any
    architectural change — the encoder handles variable input channels natively.
  - Single output channel: raw logit for binary foreground/background.
  - Optional bilinear upsampling at inference time so the output always
    matches the input spatial resolution regardless of encoder downsampling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class SegmentationNet(nn.Module):

    def __init__(self, in_channels: int = 5, pretrained: bool = True) -> None:
        super().__init__()
        self.unet = smp.Unet(
            encoder_name=   "resnet34",
            encoder_weights="imagenet" if pretrained else None,
            in_channels=    in_channels,
            classes=        1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 5, H, W)
        Returns:
            logits: (B, 1, H, W) — same spatial size as input
        """
        logits = self.unet(x)
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits
