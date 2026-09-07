"""Batched residual message passing with shared static multimesh topology."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .multimesh import MeshCache, MeshTopology, validate_strides


class MessageBlock(nn.Module):
    """Update nodes [B,N,D] using geometric edge embeddings [E,D].

    Factoring the first edge linear layer into source/destination/edge terms
    is algebraically equivalent to a linear layer on their concatenation and
    avoids allocating a [B,E,3D] tensor. Edge states do not evolve across blocks.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.node_norm = nn.LayerNorm(dim)
        self.source = nn.Linear(dim, dim, bias=False)
        self.destination = nn.Linear(dim, dim)
        self.geometry = nn.Linear(dim, dim, bias=False)
        self.message = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.update = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.SiLU(), nn.Linear(dim, dim), nn.LayerNorm(dim)
        )

    def forward(self, nodes: torch.Tensor, edges: torch.Tensor, mesh: MeshTopology):
        h = self.node_norm(nodes)
        messages = self.message(
            self.source(h)[:, mesh.src] + self.destination(h)[:, mesh.dst]
            + self.geometry(edges)[None]
        )
        # Accumulate in FP32 under AMP; preserve double precision for gradcheck.
        reduction_dtype = torch.float64 if messages.dtype == torch.float64 else torch.float32
        aggregate = torch.zeros(nodes.shape, device=nodes.device, dtype=reduction_dtype)
        aggregate.index_add_(1, mesh.dst, messages.to(reduction_dtype))
        aggregate = aggregate / mesh.degree[None, :, None]
        return nodes + self.update(torch.cat([h, aggregate.to(h.dtype)], dim=-1))


class MultimeshProcessor(nn.Module):
    """Grid [B,C,H,W] -> nodes -> residual grid update, with no data conversion."""

    def __init__(
        self, channels: int, hidden_dim: int = 256, num_blocks: int = 8,
        mesh_strides=(1, 2, 4, 8), diagonal_edges: bool = True,
        aggregation: str = "mean", zero_init_output: bool = True,
        checkpoint_blocks: bool = False,
    ):
        super().__init__()
        if type(hidden_dim) is not int or hidden_dim < 1 or type(num_blocks) is not int or num_blocks < 1:
            raise ValueError("graph hidden_dim and num_blocks must be positive integers")
        if aggregation != "mean":
            raise ValueError("G0 supports only degree-normalized mean aggregation")
        self.mesh_strides = validate_strides(mesh_strides)
        self.diagonal_edges = bool(diagonal_edges)
        self.checkpoint_blocks = bool(checkpoint_blocks)
        self.cache = MeshCache()
        self.input_projection = nn.Linear(channels, hidden_dim)
        self.edge_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.blocks = nn.ModuleList([MessageBlock(hidden_dim) for _ in range(num_blocks)])
        self.output_projection = nn.Linear(hidden_dim, channels)
        if zero_init_output:
            nn.init.zeros_(self.output_projection.weight)
            nn.init.zeros_(self.output_projection.bias)

    def _apply(self, fn, recurse=True):
        # Moving a model must release cached tensors on its previous device.
        self.cache.clear()
        return super()._apply(fn, recurse=recurse)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        b, c, height, width = field.shape
        mesh = self.cache.get(height, width, self.mesh_strides, self.diagonal_edges, field.device)
        nodes = self.input_projection(field.flatten(2).transpose(1, 2))
        edges = self.edge_encoder(mesh.edge_features.to(dtype=self.input_projection.weight.dtype))
        for block in self.blocks:
            if self.checkpoint_blocks and torch.is_grad_enabled():
                nodes = checkpoint(block, nodes, edges, mesh, use_reentrant=False)
            else:
                nodes = block(nodes, edges, mesh)
        delta = self.output_projection(nodes).transpose(1, 2).reshape(b, c, height, width)
        return field + delta
