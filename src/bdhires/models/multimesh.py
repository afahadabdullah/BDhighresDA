"""Regional, nonperiodic multimesh on nested subsets of a latent grid.

This is a direct PyTorch implementation, not GraphCast source code. Edges are
latent-grid geometry, not meteorological observations or physical footprints.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from math import hypot

import torch


def validate_strides(mesh_strides) -> tuple[int, ...]:
    strides = tuple(mesh_strides)
    if (not strides or any(type(s) is not int or s < 1 for s in strides)
            or strides != tuple(sorted(set(strides))) or strides[0] != 1):
        raise ValueError("mesh_strides must be increasing unique positive integers starting at 1")
    return strides


@dataclass(frozen=True)
class MeshTopology:
    """Shared edges [E], geometric features [E,4], and incoming degree [N]."""

    src: torch.Tensor
    dst: torch.Tensor
    edge_features: torch.Tensor
    degree: torch.Tensor


def build_multimesh(
    height: int, width: int, mesh_strides=(1, 2, 4, 8),
    diagonal_edges: bool = True, device="cpu",
) -> MeshTopology:
    """Connect stride-aligned nodes in both directions without boundary wrap.

    At stride s, both coordinates of each endpoint are multiples of s. Four
    forward directions (+x, +y and two diagonals) plus their reverses
    generate the union without duplicate edges. Features describe dst - src:
    dx/max_stride, dy/max_stride, distance/max_stride, level/max_level.
    A singleton grid has no edges and uses degree 1 for a safe zero aggregate.
    """
    strides = validate_strides(mesh_strides)
    if type(height) is not int or type(width) is not int or min(height, width) < 1:
        raise ValueError("latent height and width must be positive integers")
    sources, destinations, features = [], [], []
    directions = [(0, 1), (1, 0)]
    if diagonal_edges:
        directions += [(1, 1), (1, -1)]
    for level, stride in enumerate(strides):
        yy, xx = torch.meshgrid(
            torch.arange(0, height, stride), torch.arange(0, width, stride), indexing="ij"
        )
        for dy, dx in directions:
            y2, x2 = yy + dy * stride, xx + dx * stride
            valid = (y2 >= 0) & (y2 < height) & (x2 >= 0) & (x2 < width)
            src = (yy[valid] * width + xx[valid]).flatten()
            dst = (y2[valid] * width + x2[valid]).flatten()
            for a, b, sign in ((src, dst, 1), (dst, src, -1)):
                sources.append(a)
                destinations.append(b)
                raw = torch.tensor([
                    sign * dx * stride / strides[-1],
                    sign * dy * stride / strides[-1],
                    hypot(dx, dy) * stride / strides[-1],
                    level / max(1, len(strides) - 1),
                ], dtype=torch.float32)
                features.append(raw.expand(a.numel(), 4))
    src, dst = torch.cat(sources), torch.cat(destinations)
    degree = torch.bincount(dst, minlength=height * width).float().clamp_min(1)
    return MeshTopology(src.to(device), dst.to(device),
                        torch.cat(features).to(device), degree.to(device))


class MeshCache:
    """Small per-processor LRU; neither buffers nor checkpoint state.

    Build normal tensors even on an unguided inference_mode cache miss: a later
    guided call must be able to save the same indices for autograd backward.
    """

    def __init__(self, max_entries: int = 4):
        self.max_entries = max_entries
        self._entries: OrderedDict[tuple, MeshTopology] = OrderedDict()

    def clear(self) -> None:
        self._entries.clear()

    def get(self, height, width, mesh_strides, diagonal_edges, device) -> MeshTopology:
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        key = (height, width, tuple(mesh_strides), diagonal_edges, device)
        if key not in self._entries:
            with torch.inference_mode(False):
                self._entries[key] = build_multimesh(
                    height, width, mesh_strides, diagonal_edges, device
                )
            if len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
        self._entries.move_to_end(key)
        return self._entries[key]
