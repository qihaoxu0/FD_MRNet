from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ResidualBlock3D, WindowSelfAttention3D


class ResidualUpsampler3D(nn.Module):
    def coarse(self, lr: torch.Tensor, target_shape_dhw):
        return F.interpolate(lr, size=target_shape_dhw, mode="trilinear", align_corners=False)


class EDSR3D(ResidualUpsampler3D):
    def __init__(self, channels=64, num_blocks=16):
        super().__init__()
        self.head = nn.Conv3d(1, channels, 3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock3D(channels) for _ in range(num_blocks)])
        self.tail = nn.Conv3d(channels, 1, 3, padding=1)

    def forward(self, lr, target_shape_dhw):
        coarse = self.coarse(lr, target_shape_dhw)
        return {"pred": coarse + self.tail(self.body(self.head(coarse))), "coarse": coarse}


class DenseLayer3D(nn.Module):
    def __init__(self, in_channels, growth):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, growth, 3, padding=1)

    def forward(self, x):
        return torch.cat((x, F.leaky_relu(self.conv(x), 0.1)), dim=1)


class RDB3D(nn.Module):
    def __init__(self, channels, growth, layers):
        super().__init__()
        modules, current = [], channels
        for _ in range(layers):
            modules.append(DenseLayer3D(current, growth))
            current += growth
        self.layers = nn.Sequential(*modules)
        self.compress = nn.Conv3d(current, channels, 1)

    def forward(self, x):
        return x + self.compress(self.layers(x))


class RDN3D(ResidualUpsampler3D):
    def __init__(self, channels=48, growth=24, num_blocks=6, layers=4):
        super().__init__()
        self.head = nn.Conv3d(1, channels, 3, padding=1)
        self.blocks = nn.ModuleList([RDB3D(channels, growth, layers) for _ in range(num_blocks)])
        self.fuse = nn.Conv3d(channels * num_blocks, channels, 1)
        self.tail = nn.Conv3d(channels, 1, 3, padding=1)

    def forward(self, lr, target_shape_dhw):
        coarse = self.coarse(lr, target_shape_dhw)
        x = self.head(coarse)
        outputs = []
        for block in self.blocks:
            x = block(x)
            outputs.append(x)
        return {"pred": coarse + self.tail(self.fuse(torch.cat(outputs, 1))), "coarse": coarse}


class SwinIR3D(ResidualUpsampler3D):
    """Compact window-transformer baseline under the shared 3D protocol."""

    def __init__(self, channels=48, num_blocks=8, heads=4, window_dhw=(4, 4, 4)):
        super().__init__()
        self.head = nn.Conv3d(1, channels, 3, padding=1)
        self.blocks = nn.ModuleList([WindowSelfAttention3D(channels, heads, window_dhw) for _ in range(num_blocks)])
        self.tail = nn.Conv3d(channels, 1, 3, padding=1)

    def forward(self, lr, target_shape_dhw):
        coarse = self.coarse(lr, target_shape_dhw)
        x = self.head(coarse)
        for block in self.blocks:
            x = block(x)
        return {"pred": coarse + self.tail(x), "coarse": coarse}


class FourierUnit3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.mix = nn.Conv3d(channels * 2, channels * 2, 1)

    def forward(self, x):
        shape = x.shape[-3:]
        freq = torch.fft.rfftn(x, dim=(-3, -2, -1), norm="ortho")
        merged = torch.cat((freq.real, freq.imag), dim=1)
        merged = self.mix(merged)
        real, imag = merged.chunk(2, dim=1)
        return torch.fft.irfftn(torch.complex(real, imag), s=shape, dim=(-3, -2, -1), norm="ortho")


class SpectralSR3D(ResidualUpsampler3D):
    def __init__(self, channels=48, num_blocks=6):
        super().__init__()
        self.head = nn.Conv3d(1, channels, 3, padding=1)
        self.spatial = nn.ModuleList([ResidualBlock3D(channels) for _ in range(num_blocks)])
        self.spectral = nn.ModuleList([FourierUnit3D(channels) for _ in range(num_blocks)])
        self.tail = nn.Conv3d(channels, 1, 3, padding=1)

    def forward(self, lr, target_shape_dhw):
        coarse = self.coarse(lr, target_shape_dhw)
        x = self.head(coarse)
        for spatial, spectral in zip(self.spatial, self.spectral):
            x = spatial(x) + spectral(x)
        return {"pred": coarse + self.tail(x), "coarse": coarse}


class MatchedResidual3D(EDSR3D):
    """Capacity control; choose channels/blocks using scripts/match_capacity.py."""


class InterpolationBaseline(nn.Module):
    def forward(self, lr, target_shape_dhw):
        coarse = F.interpolate(lr, size=target_shape_dhw, mode="trilinear", align_corners=False)
        return {"pred": coarse, "coarse": coarse}

