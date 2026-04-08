"""
model.py
--------
Standard U-Net architecture for binary Brain MRI segmentation.

Architecture:
    Encoder: 4 double-conv blocks with MaxPool downsampling
    Bottleneck: double-conv block
    Decoder: 4 up-conv blocks with skip connections
    Head: 1x1 conv → sigmoid

Input:  [B, 1, 128, 128]
Output: [B, 1, 128, 128]  — probability map in [0, 1]
"""

import torch
import torch.nn as nn


# ── Building Blocks ──────────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    """
    Two consecutive: Conv2d → BatchNorm → ReLU blocks.
    Optional dropout after the second activation (used in bottleneck).
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout) if dropout > 0.0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """MaxPool2d → DoubleConv"""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class Up(nn.Module):
    """
    Bilinear upsample → concatenate skip → DoubleConv.
    Bilinear upsampling is preferred over transposed conv for stability.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle odd spatial dimensions gracefully
        if x.shape != skip.shape:
            x = nn.functional.pad(x, [0, skip.shape[3] - x.shape[3],
                                       0, skip.shape[2] - x.shape[2]])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ── U-Net ────────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    Standard U-Net.

    Args:
        in_channels  : number of input channels (1 for grayscale)
        out_channels : number of output channels (1 for binary segmentation)
        features     : list of feature map sizes for encoder levels
        dropout      : dropout probability (applied in bottleneck and decoder)
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: list = [64, 128, 256, 512],
        dropout: float = 0.0,
    ):
        super().__init__()

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc1 = DoubleConv(in_channels, features[0])           # 128 → 128
        self.enc2 = Down(features[0], features[1])                  # 128 → 64
        self.enc3 = Down(features[1], features[2])                  # 64  → 32
        self.enc4 = Down(features[2], features[3])                  # 32  → 16

        # ── Bottleneck ───────────────────────────────────────────────────────
        self.bottleneck = Down(features[3], features[3] * 2, dropout)  # 16 → 8

        # ── Decoder ──────────────────────────────────────────────────────────
        self.dec4 = Up(features[3] * 2 + features[3], features[3], dropout)
        self.dec3 = Up(features[3] + features[2],     features[2], dropout)
        self.dec2 = Up(features[2] + features[1],     features[1], dropout)
        self.dec1 = Up(features[1] + features[0],     features[0])

        # ── Output head ──────────────────────────────────────────────────────
        self.head = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder — save skip connections
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        # Bottleneck
        b = self.bottleneck(s4)

        # Decoder — fuse with skips
        d4 = self.dec4(b,  s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        # Return raw logits — sigmoid is applied by the loss (BCEWithLogitsLoss)
        # and explicitly at inference time via torch.sigmoid().
        # This is required for AMP (autocast) compatibility.
        return self.head(d1)


def build_model(dropout: float = 0.0, device: str = "cpu") -> UNet:
    """Convenience factory used by training scripts."""
    model = UNet(dropout=dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] U-Net | Parameters: {n_params:,} | dropout={dropout:.3f}")
    return model