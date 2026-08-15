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


def _zero_projection(in_channels: int, out_channels: int) -> nn.Conv2d:
    """A stable residual injection that initially leaves the original U-Net unchanged."""
    projection = nn.Conv2d(in_channels, out_channels, 1)
    nn.init.zeros_(projection.weight)
    nn.init.zeros_(projection.bias)
    return projection


class MultiscaleConditionEncoder(nn.Module):
    """Encode the full conditioning stack at every U-Net resolution."""

    def __init__(self, in_channels: int, widths: list[int]):
        super().__init__()
        blocks = []
        previous = in_channels
        for level, width in enumerate(widths):
            stride = 1 if level == 0 else 2
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(previous, width, 3, stride=stride, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(width, width, 3, padding=1),
                    nn.SiLU(),
                )
            )
            previous = width
        self.blocks = nn.ModuleList(blocks)

    def forward(self, condition: torch.Tensor) -> list[torch.Tensor]:
        features = []
        h = condition
        for block in self.blocks:
            h = block(h)
            features.append(h)
        return features


class UNet(nn.Module):
    """U-Net returning a field of shape ``(B, out_channels, H, W)``.

    Conditioning always enters by channel concatenation.  When
    ``multiscale_conditioning`` is enabled, a learned condition pyramid is also
    injected through zero-initialized projections at every down/up resolution.
    ``cond`` holds the dynamic predictors, the static fields
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
        multiscale_conditioning: bool = False,
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
        self.multiscale_conditioning = bool(multiscale_conditioning)
        emb_ch = base_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(base_channels, emb_ch), nn.SiLU(), nn.Linear(emb_ch, emb_ch)
        )
        self.base_channels = base_channels

        self.in_conv = nn.Conv2d(in_channels + cond_channels, base_channels, 3, padding=1)

        condition_widths = [base_channels * mult for mult in channel_mult]
        self.condition_encoder = (
            MultiscaleConditionEncoder(cond_channels, condition_widths)
            if self.multiscale_conditioning and cond_channels > 0
            else None
        )

        self.down = nn.ModuleList()
        self.down_condition_projections = nn.ModuleList()
        self.down_condition_levels: list[int] = []
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
                self.down_condition_projections.append(
                    _zero_projection(condition_widths[level], ch)
                    if self.condition_encoder is not None
                    else nn.Identity()
                )
                self.down_condition_levels.append(level)
                chans.append(ch)
            if level != len(channel_mult) - 1:
                self.down.append(nn.ModuleList([Downsample(ch)]))
                self.down_condition_projections.append(nn.Identity())
                self.down_condition_levels.append(-1)
                chans.append(ch)
                res //= 2

        self.mid = nn.ModuleList(
            [ResBlock(ch, ch, emb_ch, dropout), AttnBlock(ch, num_heads), ResBlock(ch, ch, emb_ch, dropout)]
        )
        self.mid_condition_projection = (
            _zero_projection(condition_widths[-1], ch)
            if self.condition_encoder is not None
            else nn.Identity()
        )

        self.up = nn.ModuleList()
        self.up_condition_projections = nn.ModuleList()
        self.up_condition_levels: list[int] = []
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
                self.up_condition_projections.append(
                    _zero_projection(condition_widths[level], ch)
                    if self.condition_encoder is not None
                    else nn.Identity()
                )
                self.up_condition_levels.append(level)

        self.out_norm = nn.GroupNorm(32 if ch % 32 == 0 else 8, ch)
        self.out_conv = nn.Conv2d(ch, out_channels, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor | None = None):
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"x must have shape (B,{self.in_channels},H,W); got {tuple(x.shape)}"
            )
        divisor = 2 ** (len(self.channel_mult) - 1)
        if x.shape[-2] % divisor or x.shape[-1] % divisor:
            raise ValueError(
                f"input spatial shape {tuple(x.shape[-2:])} must be divisible by "
                f"the U-Net pyramid factor {divisor}; use the aligned production "
                "canvas rather than sampling the raw CPC core"
            )
        if t.ndim != 1 or t.shape[0] != x.shape[0]:
            raise ValueError(f"t must have shape ({x.shape[0]},); got {tuple(t.shape)}")
        if self.cond_channels:
            if cond is None:
                raise ValueError(
                    f"this U-Net requires {self.cond_channels} conditioning channels"
                )
            if (
                cond.ndim != 4
                or cond.shape[0] != x.shape[0]
                or cond.shape[1] != self.cond_channels
                or cond.shape[-2:] != x.shape[-2:]
            ):
                raise ValueError(
                    "cond must have shape "
                    f"(B,{self.cond_channels},H,W) matching x; got {tuple(cond.shape)}"
                )
        elif cond is not None:
            if cond.ndim != 4 or cond.shape[0] != x.shape[0] or cond.shape[-2:] != x.shape[-2:]:
                raise ValueError("zero-channel cond must still match x batch and spatial shape")
            if cond.shape[1] != 0:
                raise ValueError(
                    f"this U-Net expects no conditioning channels; got {cond.shape[1]}"
                )
        emb = self.time_embed(fourier_time_embedding(t, self.base_channels))
        condition_features = (
            self.condition_encoder(cond)
            if self.condition_encoder is not None and cond is not None
            else None
        )
        h = torch.cat([x, cond], dim=1) if cond is not None else x
        h = self.in_conv(h)
        hs = [h]
        for index, block in enumerate(self.down):
            for layer in block:
                if isinstance(layer, ResBlock):
                    h = layer(h, emb)
                    if condition_features is not None:
                        level = self.down_condition_levels[index]
                        h = h + self.down_condition_projections[index](
                            condition_features[level]
                        )
                else:
                    h = layer(h)
            hs.append(h)
        for index, layer in enumerate(self.mid):
            h = layer(h, emb) if isinstance(layer, ResBlock) else layer(h)
            if index == 0 and condition_features is not None:
                h = h + self.mid_condition_projection(condition_features[-1])
        for index, block in enumerate(self.up):
            h = torch.cat([h, hs.pop()], dim=1)
            for layer in block:
                if isinstance(layer, ResBlock):
                    h = layer(h, emb)
                    if condition_features is not None:
                        level = self.up_condition_levels[index]
                        h = h + self.up_condition_projections[index](
                            condition_features[level]
                        )
                else:
                    h = layer(h)
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
