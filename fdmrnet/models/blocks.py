from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fdmrnet.degradation.gaussian import blur_depth
from fdmrnet.geometry import warp_volume


class ResidualBlock3D(nn.Module):
    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = channels * expansion
        self.net = nn.Sequential(
            nn.Conv3d(channels, hidden, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(hidden, channels, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class FrequencyDecomposition3D(nn.Module):
    """Fixed low-pass plus residual high-pass decomposition."""

    def __init__(self, sigma_depth: float = 1.0, spatial_kernel: int = 3) -> None:
        super().__init__()
        self.sigma_depth = float(sigma_depth)
        self.spatial = nn.AvgPool3d((1, spatial_kernel, spatial_kernel), stride=1, padding=(0, spatial_kernel // 2, spatial_kernel // 2))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        low = self.spatial(blur_depth(x, self.sigma_depth, "replicate"))
        return low, x - low


class WindowSelfAttention3D(nn.Module):
    def __init__(self, channels: int, heads: int, window_dhw: tuple[int, int, int]) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("channels must be divisible by attention heads")
        self.channels = channels
        self.heads = heads
        self.dim_head = channels // heads
        self.window = tuple(int(v) for v in window_dhw)
        self.norm1 = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(nn.Linear(channels, channels * 2), nn.GELU(), nn.Linear(channels * 2, channels))

    def _partition(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        b, c, d, h, w = x.shape
        wd, wh, ww = self.window
        pd, ph, pw = (-d) % wd, (-h) % wh, (-w) % ww
        x = F.pad(x, (0, pw, 0, ph, 0, pd))
        dp, hp, wp = x.shape[-3:]
        x = x.view(b, c, dp // wd, wd, hp // wh, wh, wp // ww, ww)
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
        windows = x.view(-1, wd * wh * ww, c)
        return windows, (b, c, d, h, w, dp, hp, wp)

    def _reverse(self, windows: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
        b, c, d, h, w, dp, hp, wp = shape
        wd, wh, ww = self.window
        x = windows.view(b, dp // wd, hp // wh, wp // ww, wd, wh, ww, c)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous().view(b, c, dp, hp, wp)
        return x[:, :, :d, :h, :w]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        windows, shape = self._partition(x)
        residual = windows
        qkv = self.qkv(self.norm1(windows)).view(windows.shape[0], windows.shape[1], 3, self.heads, self.dim_head)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = (q @ k.transpose(-2, -1)) / math.sqrt(self.dim_head)
        attention = attention.softmax(dim=-1)
        out = (attention @ v).transpose(1, 2).reshape(windows.shape[0], windows.shape[1], self.channels)
        windows = residual + self.proj(out)
        windows = windows + self.mlp(self.norm2(windows))
        return self._reverse(windows, shape)


class OffsetHead3D(nn.Module):
    def __init__(self, in_channels: int, hidden: int, max_displacement: float) -> None:
        super().__init__()
        self.max_displacement = float(max_displacement)
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, hidden, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(hidden, hidden, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(hidden, 3, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(x)) * self.max_displacement


class FAFB3D(nn.Module):
    def __init__(
        self,
        channels: int,
        heads: int,
        window_dhw: tuple[int, int, int],
        frequency_sigma: float,
        max_coarse_disp: float,
        max_residual_disp: float,
        use_frequency: bool = True,
        use_attention: bool = True,
        use_cross_gate: bool = True,
        use_adaptive_fusion: bool = True,
        use_hierarchical_offsets: bool = True,
        use_residual_propagation: bool = True,
    ) -> None:
        super().__init__()
        self.use_frequency = use_frequency
        self.use_attention = use_attention
        self.use_cross_gate = use_cross_gate
        self.use_adaptive_fusion = use_adaptive_fusion
        self.use_hierarchical_offsets = use_hierarchical_offsets
        self.use_residual_propagation = use_residual_propagation
        self.frequency = FrequencyDecomposition3D(frequency_sigma)
        self.low_refine = ResidualBlock3D(channels)
        self.high_refine = ResidualBlock3D(channels)
        self.low_attn = WindowSelfAttention3D(channels, heads, window_dhw) if use_attention else nn.Identity()
        self.high_attn = WindowSelfAttention3D(channels, heads, window_dhw) if use_attention else nn.Identity()
        self.cross_gate = nn.Sequential(nn.Conv3d(channels, channels, 1), nn.Sigmoid()) if use_cross_gate else None
        self.coarse_offset = OffsetHead3D(channels * 2, channels, max_coarse_disp)
        self.residual_offset = (
            OffsetHead3D(channels * 2, channels, max_residual_disp) if use_hierarchical_offsets else None
        )
        squeeze = max(channels // 8, 4)
        self.fusion_gate = (
            nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Conv3d(channels * 2, squeeze, 1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv3d(squeeze, channels, 1),
                nn.Sigmoid(),
            )
            if use_adaptive_fusion
            else None
        )
        self.post = nn.Conv3d(channels, channels, 3, padding=1)
        self.gamma = nn.Parameter(torch.full((1, channels, 1, 1, 1), 0.1)) if use_residual_propagation else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.use_frequency:
            low, high = self.frequency(x)
        else:
            low = high = x
        low, high = self.low_refine(low), self.high_refine(high)
        if self.use_attention:
            low, high = self.low_attn(low), self.high_attn(high)
        if self.use_cross_gate:
            assert self.cross_gate is not None
            high = high * self.cross_gate(low)
        d_low = self.coarse_offset(torch.cat((low, high), dim=1))
        low_warped = warp_volume(low, d_low)
        if self.use_hierarchical_offsets:
            assert self.residual_offset is not None
            d_high = self.residual_offset(torch.cat((low_warped, high), dim=1))
        else:
            d_high = torch.zeros_like(d_low)
        high_warped = warp_volume(high, d_low + d_high)
        if self.use_adaptive_fusion:
            assert self.fusion_gate is not None
            alpha = self.fusion_gate(torch.cat((low_warped, high_warped), dim=1))
        else:
            alpha = torch.full_like(low_warped, 0.5)
        fused = alpha * high_warped + (1 - alpha) * low_warped
        fused = self.post(fused)
        out = x + self.gamma * fused if self.use_residual_propagation else fused
        return out, {
            "low": low,
            "high": high,
            "low_warped": low_warped,
            "high_warped": high_warped,
            "d_low": d_low,
            "d_high": d_high,
            "alpha": alpha,
        }
