from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from fdmrnet.degradation.gaussian import blur_depth

from .blocks import FAFB3D, ResidualBlock3D


class FDMRNet(nn.Module):
    def __init__(
        self,
        channels: int = 48,
        num_blocks: int = 6,
        heads: int = 4,
        window_dhw: tuple[int, int, int] = (4, 4, 4),
        frequency_sigma: float = 1.0,
        max_coarse_disp: float = 2.0,
        max_residual_disp: float = 1.0,
        gradient_checkpointing: bool = True,
        use_global_residual: bool = True,
        **ablation,
    ) -> None:
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.use_global_residual = bool(use_global_residual)
        self.stem = nn.Sequential(
            nn.Conv3d(1, channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock3D(channels),
        )
        self.blocks = nn.ModuleList(
            [
                FAFB3D(
                    channels,
                    heads,
                    window_dhw,
                    frequency_sigma,
                    max_coarse_disp,
                    max_residual_disp,
                    **ablation,
                )
                for _ in range(num_blocks)
            ]
        )
        self.decoder = nn.Sequential(
            ResidualBlock3D(channels),
            nn.Conv3d(channels, channels // 2, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(channels // 2, 1, 3, padding=1),
        )
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)
        self.frequency_sigma = float(frequency_sigma)

    @staticmethod
    def _run_block(block: nn.Module, x: torch.Tensor):
        out, diagnostics = block(x)
        return out, diagnostics["d_low"], diagnostics["d_high"], diagnostics["alpha"]

    def forward(self, lr: torch.Tensor, target_shape_dhw: tuple[int, int, int] | None = None) -> dict:
        if target_shape_dhw is None:
            raise ValueError("target_shape_dhw is required because formal runs support both x2 and x4")
        coarse = F.interpolate(lr, size=target_shape_dhw, mode="trilinear", align_corners=False)
        features = self.stem(coarse)
        d_low_all, d_high_all, alpha_all = [], [], []
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and features.requires_grad:
                # Keep the module in the closure: only tensors should be checkpoint
                # arguments, otherwise parameter-gradient discovery can differ across
                # PyTorch versions.
                def run(z, current_block=block):
                    return self._run_block(current_block, z)

                features, d_low, d_high, alpha = checkpoint(run, features, use_reentrant=False)
            else:
                features, diag = block(features)
                d_low, d_high, alpha = diag["d_low"], diag["d_high"], diag["alpha"]
            d_low_all.append(d_low)
            d_high_all.append(d_high)
            alpha_all.append(alpha)
        residual = self.decoder(features)
        pred = coarse + residual if self.use_global_residual else residual
        pred_low = blur_depth(pred, self.frequency_sigma, "replicate")
        target_shape = pred.shape[-3:]
        return {
            "pred": pred,
            "coarse": coarse,
            "pred_low": pred_low,
            "pred_high": pred - pred_low,
            "d_low": torch.stack(d_low_all, dim=1),
            "d_high": torch.stack(d_high_all, dim=1),
            "d_composite": d_low_all[-1] + d_high_all[-1],
            "alpha": torch.stack(alpha_all, dim=1),
            "target_shape_dhw": target_shape,
        }
