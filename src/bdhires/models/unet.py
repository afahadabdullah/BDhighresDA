"""ADM-style U-Net used as the velocity network u_theta(x_t, t, cond).

Deliberately self-contained (no diffusers / physicsnemo dependency) so it runs
on an air-gapped HPC node.  ~20-40 M parameters at the default settings, which
fits comfortably on a 32 GB V100 at 128x128 with batch size 16 and AMP.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def fourier_time_embedding(t: torch.Tensor, dim: int, max_period: float = 1e4) -> torch.Tensor:
    """Sinusoidal embedding of a continuous time in [0, 1]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float().reshape(-1, 1) * freqs.reshape(1, -1) * 1000.0
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_ch: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(32 if in_ch % 32 == 0 else 8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(emb_ch, 2 * out_ch)
        self.norm2 = nn.GroupNorm(32 if out_ch % 32 == 0 else 8, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb_proj(F.silu(emb))[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    def __init__(self, ch: int, heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(32 if ch % 32 == 0 else 8, ch)
        self.qkv = nn.Conv2d(ch, 3 * ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        self.heads = heads
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.reshape(b, 3, self.heads, c // self.heads, h * w).unbind(1)
        q, k, v = (t.transpose(-2, -1) for t in (q, k, v))  # b, heads, hw, dim
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(-2, -1).reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    """U-Net returning a field of shape ``(B, out_channels, H, W)``.

    Conditioning is by channel concatenation: ``forward(x, t, cond)`` where
    ``cond`` holds the upsampled ERA5 predictors, the static fields
    (orography, land-sea mask, positional encoding) and the seasonal encoding
    broadcast to maps. IMERG is an assimilation-time observation and never a
    network predictor.
    """

    def __init__(
        self,
        in_channels: int = 1,
        cond_channels: int = 0,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_mult: tuple[int, ...] = (1, 2, 3, 4),
        num_res_blocks: int = 2,
        attn_resolutions: tuple[int, ...] = (16,),
        dropout: float = 0.1,
        image_size: int = 128,
        num_heads: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        # Retained so the model can describe itself for the startup summary.
        self.base_channels_arg = base_channels
        self.channel_mult = tuple(channel_mult)
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = tuple(attn_resolutions)
        self.dropout = dropout
        self.image_size = image_size
        self.num_heads = num_heads
        emb_ch = base_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(base_channels, emb_ch), nn.SiLU(), nn.Linear(emb_ch, emb_ch)
        )
        self.base_channels = base_channels

        self.in_conv = nn.Conv2d(in_channels + cond_channels, base_channels, 3, padding=1)

        self.down = nn.ModuleList()
        chans = [base_channels]
        ch = base_channels
        res = image_size
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                block = nn.ModuleList([ResBlock(ch, base_channels * mult, emb_ch, dropout)])
                ch = base_channels * mult
                if res in attn_resolutions:
                    block.append(AttnBlock(ch, num_heads))
                self.down.append(block)
                chans.append(ch)
            if level != len(channel_mult) - 1:
                self.down.append(nn.ModuleList([Downsample(ch)]))
                chans.append(ch)
                res //= 2

        self.mid = nn.ModuleList(
            [ResBlock(ch, ch, emb_ch, dropout), AttnBlock(ch, num_heads), ResBlock(ch, ch, emb_ch, dropout)]
        )

        self.up = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mult))):
            for i in range(num_res_blocks + 1):
                block = nn.ModuleList(
                    [ResBlock(ch + chans.pop(), base_channels * mult, emb_ch, dropout)]
                )
                ch = base_channels * mult
                if res in attn_resolutions:
                    block.append(AttnBlock(ch, num_heads))
                if level and i == num_res_blocks:
                    block.append(Upsample(ch))
                    res *= 2
                self.up.append(block)

        self.out_norm = nn.GroupNorm(32 if ch % 32 == 0 else 8, ch)
        self.out_conv = nn.Conv2d(ch, out_channels, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor | None = None):
        emb = self.time_embed(fourier_time_embedding(t, self.base_channels))
        h = torch.cat([x, cond], dim=1) if cond is not None else x
        h = self.in_conv(h)
        hs = [h]
        for block in self.down:
            for layer in block:
                h = layer(h, emb) if isinstance(layer, ResBlock) else layer(h)
            hs.append(h)
        for layer in self.mid:
            h = layer(h, emb) if isinstance(layer, ResBlock) else layer(h)
        for block in self.up:
            h = torch.cat([h, hs.pop()], dim=1)
            for layer in block:
                h = layer(h, emb) if isinstance(layer, ResBlock) else layer(h)
        return self.out_conv(F.silu(self.out_norm(h)))

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameters_by_module(self) -> "dict[str, int]":
        """Parameter count per top-level child, for the startup summary."""
        return {name: sum(p.numel() for p in child.parameters())
                for name, child in self.named_children()}

    def levels(self) -> "list[dict]":
        """Resolution, channel width and attention flag at each U-Net level.

        Mirrors the loop in ``__init__`` rather than inspecting the built
        modules, so it stays readable and matches what the config asked for.
        """
        out = []
        resolution = self.image_size
        for level, mult in enumerate(self.channel_mult):
            channels = self.base_channels_arg * mult
            out.append(
                {
                    "level": level,
                    "resolution": resolution,
                    "channels": channels,
                    "attention": resolution in self.attn_resolutions,
                    "res_blocks": self.num_res_blocks,
                }
            )
            if level != len(self.channel_mult) - 1:
                resolution //= 2
        out.append(
            {
                "level": "mid",
                "resolution": resolution,
                "channels": self.base_channels_arg * self.channel_mult[-1],
                "attention": True,          # the middle block always has attention
                "res_blocks": 2,
            }
        )
        return out
