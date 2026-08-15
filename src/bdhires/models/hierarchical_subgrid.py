"""Hierarchical rectified-flow networks for CPC-native subgrid downscaling.

V3-SG evolves a structured state rather than one rainfall image:

``coarse``
    two 0.5-degree channels (positive-amount latent, occurrence logit), and
``allocation``
    two 0.05-degree channels (positive allocation logit, occurrence logit).

The operational network predicts both velocities in one call.  A zero-started
fine-to-coarse coupling preserves literal branch pretraining at initialisation,
then learns the cross-scale term that a sequential pair of scores would miss.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .flow import RectifiedFlow
from .unet import UNet


@dataclass
class HierarchicalState:
    coarse: torch.Tensor
    allocation: torch.Tensor

    def detach(self) -> "HierarchicalState":
        return HierarchicalState(self.coarse.detach(), self.allocation.detach())

    def to(self, *args, **kwargs) -> "HierarchicalState":
        return HierarchicalState(
            self.coarse.to(*args, **kwargs), self.allocation.to(*args, **kwargs)
        )


class CoarseHurdleFlow(nn.Module):
    """Shallow flow over native-CPC amount and wetness latents."""

    def __init__(
        self,
        cond_channels: int,
        image_size: int = 26,
        base_channels: int = 64,
        channel_mult: tuple[int, ...] = (1, 2),
        num_res_blocks: int = 2,
        dropout: float = 0.1,
        num_heads: int = 4,
    ):
        super().__init__()
        self.cond_channels = int(cond_channels)
        self.net = UNet(
            in_channels=2,
            cond_channels=self.cond_channels,
            out_channels=2,
            image_size=image_size,
            base_channels=base_channels,
            channel_mult=tuple(channel_mult),
            num_res_blocks=num_res_blocks,
            attn_resolutions=(max(1, image_size // 2),),
            dropout=dropout,
            num_heads=num_heads,
            multiscale_conditioning=self.cond_channels > 0,
        )

    def forward(self, state, t, cond=None):
        if self.cond_channels and cond is None:
            raise ValueError("coarse conditioning is required by this model")
        return self.net(state, t, cond)


class AllocationFlow(nn.Module):
    """Fine allocation velocity with an interface shared by Phase 2 and joint flow.

    The coarse corruption level is always a real input channel.  Phase-2
    conditioning augmentation and Phase-3 noisy coarse state therefore use the
    same parameters; no first-layer channel is discarded during transfer.
    """

    def __init__(
        self,
        fine_cond_channels: int,
        image_size: int = 120,
        base_channels: int = 64,
        channel_mult: tuple[int, ...] = (1, 2, 3, 4),
        num_res_blocks: int = 2,
        attn_resolutions: tuple[int, ...] = (15, 30),
        dropout: float = 0.1,
        num_heads: int = 4,
    ):
        super().__init__()
        self.fine_cond_channels = int(fine_cond_channels)
        # fine predictors + two coarse-state channels + one uncertainty level
        self.total_cond_channels = self.fine_cond_channels + 3
        self.net = UNet(
            in_channels=2,
            cond_channels=self.total_cond_channels,
            out_channels=2,
            image_size=image_size,
            base_channels=base_channels,
            channel_mult=tuple(channel_mult),
            num_res_blocks=num_res_blocks,
            attn_resolutions=tuple(attn_resolutions),
            dropout=dropout,
            num_heads=num_heads,
            multiscale_conditioning=True,
        )

    @staticmethod
    def _level_map(level, reference: torch.Tensor) -> torch.Tensor:
        if level is None:
            level = torch.zeros(reference.shape[0], device=reference.device, dtype=reference.dtype)
        if not torch.is_tensor(level):
            level = torch.full(
                (reference.shape[0],), float(level),
                device=reference.device, dtype=reference.dtype,
            )
        level = level.to(device=reference.device, dtype=reference.dtype)
        if level.ndim == 0:
            level = level.expand(reference.shape[0])
        if level.shape != (reference.shape[0],):
            raise ValueError("coarse uncertainty level must be scalar or shape (B,)")
        return level[:, None, None, None].expand(-1, 1, *reference.shape[-2:])

    def forward(
        self,
        allocation_state: torch.Tensor,
        t: torch.Tensor,
        fine_cond: torch.Tensor | None,
        coarse_context: torch.Tensor,
        coarse_uncertainty=None,
    ) -> torch.Tensor:
        if coarse_context.ndim != 4 or coarse_context.shape[1] != 2:
            raise ValueError("coarse_context must have shape (B,2,Hc,Wc)")
        if coarse_context.shape[0] == 1 and allocation_state.shape[0] > 1:
            coarse_context = coarse_context.expand(allocation_state.shape[0], -1, -1, -1)
        upsampled = F.interpolate(
            coarse_context, size=allocation_state.shape[-2:], mode="bilinear", align_corners=False
        )
        condition = []
        if self.fine_cond_channels:
            if fine_cond is None or fine_cond.shape[1] != self.fine_cond_channels:
                raise ValueError(
                    f"expected {self.fine_cond_channels} fine conditioning channels"
                )
            condition.append(fine_cond)
        condition.extend([upsampled, self._level_map(coarse_uncertainty, allocation_state)])
        return self.net(allocation_state, t, torch.cat(condition, dim=1))


class CoupledSubgridFlow(nn.Module):
    """Joint multiscale velocity over ``(coarse, allocation)``.

    ``clean_coarse_context`` is a trained conditioning mode for oracle/clamped
    diagnostics.  A sampler must not claim conditional sampling by merely
    replacing the noisy coarse trajectory unless this clean context was used in
    joint training; :mod:`bdhires.da.hierarchical_sampler` enforces that flag.
    """

    supports_clean_coarse_context = True

    def __init__(self, coarse_branch: CoarseHurdleFlow, allocation_branch: AllocationFlow):
        super().__init__()
        self.coarse_branch = coarse_branch
        self.allocation_branch = allocation_branch
        self.fine_to_coarse = nn.Conv2d(2, 2, kernel_size=1)
        nn.init.zeros_(self.fine_to_coarse.weight)
        nn.init.zeros_(self.fine_to_coarse.bias)

    def forward(
        self,
        coarse_state: torch.Tensor,
        allocation_state: torch.Tensor,
        t: torch.Tensor,
        coarse_cond: torch.Tensor | None = None,
        fine_cond: torch.Tensor | None = None,
        *,
        coarse_uncertainty=None,
        clean_coarse_context: torch.Tensor | None = None,
    ) -> HierarchicalState:
        coarse_velocity = self.coarse_branch(coarse_state, t, coarse_cond)
        pooled_fine = F.adaptive_avg_pool2d(allocation_state, coarse_state.shape[-2:])
        coarse_velocity = coarse_velocity + self.fine_to_coarse(pooled_fine)

        context = coarse_state if clean_coarse_context is None else clean_coarse_context
        if coarse_uncertainty is None:
            coarse_uncertainty = 1.0 - t if clean_coarse_context is None else 0.0
        allocation_velocity = self.allocation_branch(
            allocation_state,
            t,
            fine_cond,
            context,
            coarse_uncertainty,
        )
        return HierarchicalState(coarse_velocity, allocation_velocity)

    def load_pretrained_branches(
        self, coarse_state_dict: dict | None = None, allocation_state_dict: dict | None = None
    ) -> None:
        """Strict branch transfer; unmatched intended parameters are an error."""
        if coarse_state_dict is not None:
            self.coarse_branch.load_state_dict(coarse_state_dict, strict=True)
        if allocation_state_dict is not None:
            self.allocation_branch.load_state_dict(allocation_state_dict, strict=True)


class HierarchicalRectifiedFlow:
    """Linear interpolant conversions applied to both state resolutions."""

    def __init__(self, t_min: float = 1.0e-3, t_max: float = 1.0 - 1.0e-3):
        self.scalar = RectifiedFlow(t_min=t_min, t_max=t_max)

    def sample_t(self, batch: int, device, logit_normal: bool = True):
        return self.scalar.sample_t(batch, device, logit_normal)

    def interpolate(
        self, state1: HierarchicalState, t: torch.Tensor, noise: HierarchicalState | None = None
    ) -> tuple[HierarchicalState, HierarchicalState, HierarchicalState]:
        noise = noise or HierarchicalState(
            torch.randn_like(state1.coarse), torch.randn_like(state1.allocation)
        )
        coarse_t, coarse_u, _ = self.scalar.interpolate(state1.coarse, t, noise.coarse)
        fine_t, fine_u, _ = self.scalar.interpolate(state1.allocation, t, noise.allocation)
        return (
            HierarchicalState(coarse_t, fine_t),
            HierarchicalState(coarse_u, fine_u),
            noise,
        )

    def x1_hat(
        self, state_t: HierarchicalState, t: torch.Tensor, velocity: HierarchicalState
    ) -> HierarchicalState:
        return HierarchicalState(
            self.scalar.x1_hat(state_t.coarse, t, velocity.coarse),
            self.scalar.x1_hat(state_t.allocation, t, velocity.allocation),
        )

    def score(
        self, state_t: HierarchicalState, t: torch.Tensor, velocity: HierarchicalState
    ) -> HierarchicalState:
        return HierarchicalState(
            self.scalar.score(state_t.coarse, t, velocity.coarse),
            self.scalar.score(state_t.allocation, t, velocity.allocation),
        )

    def score_to_velocity_factor(self, t: torch.Tensor, ref: HierarchicalState):
        return HierarchicalState(
            self.scalar.score_to_velocity_factor(t, ref.coarse),
            self.scalar.score_to_velocity_factor(t, ref.allocation),
        )


def _masked_mse(prediction, target, mask) -> torch.Tensor:
    weight = mask.to(prediction.dtype)
    if weight.shape[1] == 1 and prediction.shape[1] > 1:
        weight = weight.expand(-1, prediction.shape[1], -1, -1)
    return ((prediction - target).square() * weight).sum() / weight.sum().clamp_min(1.0)


def hierarchical_flow_matching_loss(
    model: CoupledSubgridFlow,
    state1: HierarchicalState,
    coarse_cond: torch.Tensor | None,
    fine_cond: torch.Tensor | None,
    coarse_mask: torch.Tensor,
    fine_mask: torch.Tensor,
    flow: HierarchicalRectifiedFlow | None = None,
    *,
    cond_dropout: float = 0.0,
    logit_normal_t: bool = True,
    coarse_weight: float = 1.0,
    allocation_weight: float = 1.0,
    clean_context_probability: float = 0.15,
    return_components: bool = False,
):
    """Joint flow-matching objective with a trained clean-clamp oracle mode."""
    flow = flow or HierarchicalRectifiedFlow()
    batch = state1.coarse.shape[0]
    t = flow.sample_t(batch, state1.coarse.device, logit_normal_t)
    state_t, target, _ = flow.interpolate(state1, t)
    if cond_dropout:
        keep = (
            torch.rand(batch, 1, 1, 1, device=state1.coarse.device) > cond_dropout
        ).to(state1.coarse.dtype)
        if coarse_cond is not None:
            coarse_cond = coarse_cond * keep
        if fine_cond is not None:
            fine_cond = fine_cond * keep

    clean_context = None
    uncertainty = 1.0 - t
    if clean_context_probability:
        if not 0.0 <= clean_context_probability <= 1.0:
            raise ValueError("clean_context_probability must lie in [0, 1]")
        use_clean = (
            torch.rand(batch, device=t.device) < clean_context_probability
        ).view(batch, 1, 1, 1)
        clean_context = torch.where(use_clean, state1.coarse, state_t.coarse)
        uncertainty = torch.where(use_clean.flatten(), torch.zeros_like(t), 1.0 - t)

    prediction = model(
        state_t.coarse,
        state_t.allocation,
        t,
        coarse_cond,
        fine_cond,
        coarse_uncertainty=uncertainty,
        clean_coarse_context=clean_context,
    )
    coarse_loss = _masked_mse(prediction.coarse, target.coarse, coarse_mask)
    allocation_loss = _masked_mse(
        prediction.allocation, target.allocation, fine_mask
    )
    total = float(coarse_weight) * coarse_loss + float(allocation_weight) * allocation_loss
    if return_components:
        return total, coarse_loss, allocation_loss
    return total


def coarse_flow_matching_loss(
    model: CoarseHurdleFlow,
    target: torch.Tensor,
    cond: torch.Tensor | None,
    mask: torch.Tensor,
    flow: RectifiedFlow | None = None,
):
    flow = flow or RectifiedFlow()
    t = flow.sample_t(target.shape[0], target.device)
    state_t, velocity, _ = flow.interpolate(target, t)
    return _masked_mse(model(state_t, t, cond), velocity, mask)


def allocation_flow_matching_loss(
    model: AllocationFlow,
    target: torch.Tensor,
    fine_cond: torch.Tensor | None,
    coarse_truth: torch.Tensor,
    mask: torch.Tensor,
    flow: RectifiedFlow | None = None,
    *,
    max_coarse_noise: float = 0.35,
    clean_probability: float = 0.25,
):
    """Phase-2 objective with an interface identical to the joint fine branch."""
    flow = flow or RectifiedFlow()
    batch = target.shape[0]
    t = flow.sample_t(batch, target.device)
    state_t, velocity, _ = flow.interpolate(target, t)
    level = torch.rand(batch, device=target.device) * float(max_coarse_noise)
    clean = torch.rand(batch, device=target.device) < float(clean_probability)
    level = torch.where(clean, torch.zeros_like(level), level)
    scale = level[:, None, None, None]
    coarse_context = (1.0 - scale) * coarse_truth + scale * torch.randn_like(coarse_truth)
    prediction = model(state_t, t, fine_cond, coarse_context, level)
    return _masked_mse(prediction, velocity, mask)
