"""Temporal UNet backbone for Diffusion Policy (Chi et al., 2023).

Input  : noisy action sequence  (B, T_pred, action_dim)
Cond   : flattened observations (B, T_obs * obs_dim)  +  diffusion timestep
Output : predicted noise        (B, T_pred, action_dim)

Sequence dimension is treated as the spatial axis for 1-D convolutions.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    """Fixed sinusoidal timestep embedding."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10_000) * torch.arange(half, device=t.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        x = t.float()[:, None] * freqs[None]
        return torch.cat([x.sin(), x.cos()], dim=-1)


class ResidualBlock1D(nn.Module):
    """Conv1D residual block with GroupNorm and FiLM conditioning.

    FiLM: scale and shift derived from the conditioning vector.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        cond_dim: int,
        groups: int = 8,
    ) -> None:
        super().__init__()
        # Reduce groups if channel count is small
        g = min(groups, in_ch, out_ch)
        while out_ch % g != 0 or in_ch % g != 0:
            g -= 1

        self.conv1  = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.conv2  = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.norm1  = nn.GroupNorm(g, out_ch)
        self.norm2  = nn.GroupNorm(g, out_ch)
        self.act    = nn.Mish()
        self.film   = nn.Linear(cond_dim, out_ch * 2)   # scale + bias
        self.skip   = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, in_ch, T)   cond: (B, cond_dim)
        scale, bias = self.film(cond).unsqueeze(-1).chunk(2, dim=1)
        h = self.act(self.norm1(self.conv1(x)))
        h = h * (1.0 + scale) + bias          # FiLM modulation
        h = self.act(self.norm2(self.conv2(h)))
        return h + self.skip(x)


# ---------------------------------------------------------------------------
# UNet
# ---------------------------------------------------------------------------

class TemporalUNet(nn.Module):
    """Temporal UNet for joint-space Diffusion Policy.

    Args:
        action_dim   : dimensionality of one action step (e.g. 7 for FR3)
        obs_dim      : flattened observation size (T_obs × per_step_obs_dim)
        down_channels: channel sizes at each encoder level, e.g. (256, 512, 1024)
        time_emb_dim : sinusoidal embedding dimension for diffusion timestep
    """

    def __init__(
        self,
        action_dim: int,
        obs_dim: int,
        down_channels: tuple[int, ...] = (256, 512, 1024),
        time_emb_dim: int = 256,
    ) -> None:
        if time_emb_dim % 2 != 0:
            raise ValueError(f"time_emb_dim must be even, got {time_emb_dim}")
        super().__init__()
        n = len(down_channels)
        cond_dim = time_emb_dim * 2

        # --- conditioning projections ---
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.Mish(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )
        self.obs_proj = nn.Sequential(
            nn.Linear(obs_dim, time_emb_dim),
            nn.Mish(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # --- encoder ---
        self.input_conv = nn.Conv1d(action_dim, down_channels[0], 3, padding=1)
        self.down_res = nn.ModuleList([
            ResidualBlock1D(down_channels[i], down_channels[i], cond_dim)
            for i in range(n - 1)
        ])
        self.downsamplers = nn.ModuleList([
            nn.Conv1d(down_channels[i], down_channels[i + 1], 3, stride=2, padding=1)
            for i in range(n - 1)
        ])

        # --- bottleneck ---
        mid = down_channels[-1]
        self.mid1 = ResidualBlock1D(mid, mid, cond_dim)
        self.mid2 = ResidualBlock1D(mid, mid, cond_dim)

        # --- decoder ---
        # Up block i receives:  down_channels[n-1-i] from below + down_channels[n-2-i] as skip
        self.up_res = nn.ModuleList([
            ResidualBlock1D(
                down_channels[n - 1 - i] + down_channels[n - 2 - i],
                down_channels[n - 2 - i],
                cond_dim,
            )
            for i in range(n - 1)
        ])

        g0 = 8
        while down_channels[0] % g0 != 0:
            g0 -= 1
        self.output_proj = nn.Sequential(
            nn.GroupNorm(g0, down_channels[0]),
            nn.Mish(),
            nn.Conv1d(down_channels[0], action_dim, 1),
        )

    def forward(
        self,
        noisy_actions: torch.Tensor,   # (B, T_pred, action_dim)
        timestep: torch.Tensor,        # (B,)
        obs: torch.Tensor,             # (B, obs_dim)
    ) -> torch.Tensor:
        cond = torch.cat([self.time_emb(timestep), self.obs_proj(obs)], dim=-1)

        # (B, action_dim, T_pred) for Conv1D
        x = self.input_conv(noisy_actions.transpose(1, 2))

        skips: list[torch.Tensor] = []
        for res, ds in zip(self.down_res, self.downsamplers):
            x = res(x, cond)
            skips.append(x)
            x = ds(x)

        x = self.mid2(self.mid1(x, cond), cond)

        for up_res, skip in zip(self.up_res, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
            x = up_res(torch.cat([x, skip], dim=1), cond)

        return self.output_proj(x).transpose(1, 2)   # (B, T_pred, action_dim)
