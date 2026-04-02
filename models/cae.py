"""
Convolutional Autoencoder (CAE) for unsupervised anomaly detection.
Trained on normal frames only. Anomaly score = per-pixel reconstruction error.
"""

import torch
import torch.nn as nn


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),  # downsample
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, final=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),  # upsample
            nn.BatchNorm2d(out_ch) if not final else nn.Identity(),
            nn.ReLU(inplace=True) if not final else nn.Sigmoid(),
        )

    def forward(self, x):
        return self.block(x)


class CAE(nn.Module):
    """
    Convolutional Autoencoder.

    Input:  [B, C, H, W]  — C=1 (grayscale) or C=2 (grayscale + flow)
    Output: [B, C, H, W]  — reconstructed input, values in [0, 1]

    Encoder: 256 → 128 → 64 → 32 → 16
    Decoder: 16  → 32  → 64 → 128 → 256
    """

    def __init__(self, input_channels: int = 1, base_channels: int = 32):
        super().__init__()

        c = base_channels  # 32

        # Encoder: progressively halve spatial dims, double channels
        self.enc1 = EncoderBlock(input_channels, c)       # 256 → 128
        self.enc2 = EncoderBlock(c,              c * 2)   # 128 → 64
        self.enc3 = EncoderBlock(c * 2,          c * 4)   # 64  → 32
        self.enc4 = EncoderBlock(c * 4,          c * 8)   # 32  → 16

        # Bottleneck (keeps spatial size, compresses channels)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(c * 8, c * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(c * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Decoder: progressively double spatial dims, halve channels
        self.dec4 = DecoderBlock(c * 8, c * 4)            # 16  → 32
        self.dec3 = DecoderBlock(c * 4, c * 2)            # 32  → 64
        self.dec2 = DecoderBlock(c * 2, c)                # 64  → 128
        self.dec1 = DecoderBlock(c,     input_channels, final=True)  # 128 → 256

    def forward(self, x):
        # Encode
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Bottleneck
        b = self.bottleneck(e4)

        # Decode
        d4 = self.dec4(b)
        d3 = self.dec3(d4)
        d2 = self.dec2(d3)
        out = self.dec1(d2)

        return out

    def encode(self, x):
        """Return bottleneck representation (for future use)."""
        return self.bottleneck(self.enc4(self.enc3(self.enc2(self.enc1(x)))))


def build_model(config: dict) -> CAE:
    return CAE(
        input_channels=config["model"]["input_channels"],
        base_channels=config["model"]["base_channels"],
    )


if __name__ == "__main__":
    # Quick shape check
    model = CAE(input_channels=1, base_channels=32)
    x = torch.randn(4, 1, 256, 256)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    assert x.shape == out.shape, "Input/output shape mismatch!"

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {total_params:,}")
    print("CAE shape check passed.")