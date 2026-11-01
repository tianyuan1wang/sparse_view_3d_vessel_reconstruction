"""Inference-only 3-D diffusion U-Net and its trained linear noise schedule.

The module names, parameter keys, and float32 scheduler calculations match the
reconstruction pipeline and the released diffusion prior checkpoint.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    device = timesteps.device
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half, device=device, dtype=torch.float32) / max(half - 1, 1)
    )
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=min(8, in_ch), num_channels=in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Down3D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv3d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Up3D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.ConvTranspose3d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class UNet3DDiffusion(nn.Module):
    def __init__(self, in_ch: int = 1, base_ch: int = 32, time_dim: int = 128):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.in_conv = nn.Conv3d(in_ch, base_ch, 3, padding=1)
        self.rb1 = ResBlock3D(base_ch, base_ch, time_dim)
        self.down1 = Down3D(base_ch)
        self.rb2 = ResBlock3D(base_ch, base_ch * 2, time_dim)
        self.down2 = Down3D(base_ch * 2)
        self.rb3 = ResBlock3D(base_ch * 2, base_ch * 4, time_dim)
        self.mid1 = ResBlock3D(base_ch * 4, base_ch * 4, time_dim)
        self.mid2 = ResBlock3D(base_ch * 4, base_ch * 4, time_dim)
        self.up2 = Up3D(base_ch * 4)
        self.rb4 = ResBlock3D(base_ch * 4 + base_ch * 2, base_ch * 2, time_dim)
        self.up1 = Up3D(base_ch * 2)
        self.rb5 = ResBlock3D(base_ch * 2 + base_ch, base_ch, time_dim)
        self.out_norm = nn.GroupNorm(num_groups=min(8, base_ch), num_channels=base_ch)
        self.out_conv = nn.Conv3d(base_ch, 1, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(timestep_embedding(t, self.time_dim))
        x0 = self.in_conv(x)
        x1 = self.rb1(x0, t_emb)
        x2 = self.rb2(self.down1(x1), t_emb)
        x3 = self.rb3(self.down2(x2), t_emb)
        h = self.mid1(x3, t_emb)
        h = self.mid2(h, t_emb)
        h = self.up2(h)
        h = torch.cat([h, x2], dim=1)
        h = self.rb4(h, t_emb)
        h = self.up1(h)
        h = torch.cat([h, x1], dim=1)
        h = self.rb5(h, t_emb)
        return self.out_conv(F.silu(self.out_norm(h)))


class DiffusionConfig:
    def __init__(self, num_steps=200, beta_start=1e-4, beta_end=2e-2):
        self.num_steps = int(num_steps)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)


class GaussianDiffusion(nn.Module):
    def __init__(self, cfg: DiffusionConfig):
        super().__init__()
        self.cfg = cfg

        betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.num_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat([torch.tensor([1.0], dtype=torch.float32), alpha_bars[:-1]], dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alpha_bars_prev", alpha_bars_prev)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))
        self.register_buffer(
            "posterior_var",
            betas * (1.0 - alpha_bars_prev) / torch.clamp(1.0 - alpha_bars, min=1e-8)
        )

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            self.sqrt_alpha_bars[t][:, None, None, None, None] * x0
            + self.sqrt_one_minus_alpha_bars[t][:, None, None, None, None] * noise
        )
