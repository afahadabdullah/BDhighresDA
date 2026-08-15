"""Joint V3-SG sampling and observation guidance over coarse/fine state.

This module intentionally does not adapt the legacy single-field sampler.  The
likelihood is differentiated through the joint velocity and the conservative
decoder in one graph, so both the coarse amount and fine allocation receive the
correct posterior nudge.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..data.subgrid_dataset import (
    SubgridEncoding,
    decode_and_reconstruct,
)
from ..models.hierarchical_subgrid import (
    CoupledSubgridFlow,
    HierarchicalRectifiedFlow,
    HierarchicalState,
)
from .guidance import GuidanceConfig, obs_log_likelihood


@dataclass
class HierarchicalSamplerConfig:
    n_steps: int = 50
    heun: bool = True
    schedule_power: float = 1.0
    n_corrections: int = 0
    corrector_tau: float = 0.3
    corrector_max_step: float | None = 0.3
    occurrence_temperature: float = 1.0
    seed: int | None = None
    soft_hard_median_sigma: float = 0.10
    soft_hard_p95_sigma: float = 0.50


@dataclass
class HierarchicalObservations:
    operator: torch.nn.Module
    values: torch.Tensor
    variance: torch.Tensor
    guidance: GuidanceConfig = field(default_factory=GuidanceConfig)


@dataclass
class HierarchicalSample:
    state: HierarchicalState
    precipitation: torch.Tensor
    diagnostics: dict


def _schedule(cfg: HierarchicalSamplerConfig, device) -> torch.Tensor:
    if cfg.n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if cfg.schedule_power <= 0.0:
        raise ValueError("schedule_power must be positive")
    index = torch.arange(cfg.n_steps + 1, device=device, dtype=torch.float32)
    index = index / cfg.n_steps
    return 1.0 - (1.0 - index) ** cfg.schedule_power


def _expand_condition(condition, batch: int):
    if condition is None or condition.shape[0] == batch:
        return condition
    if condition.shape[0] != 1:
        raise ValueError(
            f"cannot broadcast conditioning batch {condition.shape[0]} to {batch}"
        )
    return condition.expand(batch, -1, -1, -1)


def _expand_observations(observations: HierarchicalObservations, batch: int):
    values = observations.values
    if values.shape[0] == 1 and batch > 1:
        values = values.expand(batch, -1, -1)
    if values.shape[0] != batch:
        raise ValueError(f"observation batch {values.shape[0]} does not match {batch}")
    return values


def _model_velocity(
    model,
    state: HierarchicalState,
    t: torch.Tensor,
    coarse_cond,
    fine_cond,
    clean_coarse_context=None,
) -> HierarchicalState:
    output = model(
        state.coarse,
        state.allocation,
        t,
        coarse_cond,
        fine_cond,
        clean_coarse_context=clean_coarse_context,
        coarse_uncertainty=(0.0 if clean_coarse_context is not None else 1.0 - t),
    )
    if not isinstance(output, HierarchicalState):
        raise TypeError(
            "joint V3 sampling requires one model call returning both velocity "
            "components; separate Model-A/Model-B scores omit the cross-scale score"
        )
    return output


def hierarchical_guidance_grad(
    model,
    state_t: HierarchicalState,
    t: torch.Tensor,
    coarse_cond,
    fine_cond,
    observations: HierarchicalObservations,
    flow: HierarchicalRectifiedFlow,
    encoding: SubgridEncoding,
    coarse_valid: torch.Tensor,
    fine_valid: torch.Tensor,
    area: torch.Tensor,
    occurrence_temperature: float,
    clean_coarse_context=None,
) -> tuple[HierarchicalState, HierarchicalState, torch.Tensor]:
    """Return joint velocity, joint likelihood gradient and hard physical field."""
    if torch.is_inference_mode_enabled():
        raise RuntimeError("joint observation guidance cannot run in inference_mode")
    with torch.enable_grad():
        coarse = state_t.coarse.detach().requires_grad_(True)
        allocation = state_t.allocation.detach().requires_grad_(True)
        state = HierarchicalState(coarse, allocation)
        velocity = _model_velocity(
            model, state, t, coarse_cond, fine_cond, clean_coarse_context
        )
        clean = flow.x1_hat(state, t, velocity)
        physical = decode_and_reconstruct(
            clean.coarse,
            clean.allocation,
            coarse_valid,
            fine_valid,
            area,
            encoding,
            temperature=occurrence_temperature,
            hard=True,
        )
        hx = observations.operator(physical)
        values = _expand_observations(observations, state.coarse.shape[0])
        likelihood = obs_log_likelihood(
            values, hx, observations.variance, t, observations.guidance
        ).sum()
        gradient = torch.autograd.grad(likelihood, (coarse, allocation))

    coarse_gradient, allocation_gradient = (part.detach() for part in gradient)
    for label, part in (
        ("coarse", coarse_gradient), ("allocation", allocation_gradient)
    ):
        if not torch.isfinite(part).all():
            count = int((~torch.isfinite(part)).sum().cpu())
            raise FloatingPointError(
                f"non-finite {label} joint guidance gradient: {count}/{part.numel()}"
            )
    clip = observations.guidance.clip_norm
    if clip is not None:
        norm = torch.sqrt(
            coarse_gradient.flatten(1).square().sum(1)
            + allocation_gradient.flatten(1).square().sum(1)
        ).clamp_min(float(clip))
        scale = (float(clip) / norm)
        coarse_gradient = coarse_gradient * scale[:, None, None, None]
        allocation_gradient = allocation_gradient * scale[:, None, None, None]
    return (
        velocity.detach(),
        HierarchicalState(coarse_gradient, allocation_gradient),
        physical.detach(),
    )


def _guided_velocity(
    velocity: HierarchicalState,
    gradient: HierarchicalState,
    state: HierarchicalState,
    t: torch.Tensor,
    flow: HierarchicalRectifiedFlow,
    config: GuidanceConfig,
) -> HierarchicalState:
    factor = flow.score_to_velocity_factor(t, state)
    active = ((t >= config.t_start) & (t <= config.t_end)).to(state.coarse.dtype)
    coarse_active = active.view(-1, 1, 1, 1)
    return HierarchicalState(
        velocity.coarse
        + float(config.scale) * factor.coarse * gradient.coarse * coarse_active,
        velocity.allocation
        + float(config.scale) * factor.allocation * gradient.allocation * coarse_active,
    )


def _add(left: HierarchicalState, right: HierarchicalState, scale: float):
    return HierarchicalState(
        left.coarse + scale * right.coarse,
        left.allocation + scale * right.allocation,
    )


def _oracle_coarse(
    truth: torch.Tensor, noise: torch.Tensor, t: float | torch.Tensor
) -> torch.Tensor:
    if not torch.is_tensor(t):
        t = torch.full((truth.shape[0],), float(t), device=truth.device, dtype=truth.dtype)
    tb = t.view(-1, 1, 1, 1).to(truth.dtype)
    return (1.0 - tb) * noise + tb * truth


def _correct(
    model,
    state,
    t,
    coarse_cond,
    fine_cond,
    observations,
    flow,
    encoding,
    coarse_valid,
    fine_valid,
    area,
    cfg,
    oracle_truth,
    oracle_noise,
):
    for _ in range(cfg.n_corrections):
        time = torch.full(
            (state.coarse.shape[0],), float(t),
            device=state.coarse.device, dtype=state.coarse.dtype,
        )
        clean_context = oracle_truth if oracle_truth is not None else None
        if observations is None:
            with torch.no_grad():
                velocity = _model_velocity(
                    model, state, time, coarse_cond, fine_cond, clean_context
                )
            score = flow.score(state, time, velocity)
        else:
            velocity, likelihood_gradient, _ = hierarchical_guidance_grad(
                model, state, time, coarse_cond, fine_cond, observations, flow,
                encoding, coarse_valid, fine_valid, area,
                cfg.occurrence_temperature, clean_context,
            )
            prior_score = flow.score(state, time, velocity)
            guidance = observations.guidance
            active = float(guidance.t_start <= float(t) <= guidance.t_end)
            likelihood_scale = active * float(guidance.scale)
            score = HierarchicalState(
                prior_score.coarse + likelihood_scale * likelihood_gradient.coarse,
                prior_score.allocation
                + likelihood_scale * likelihood_gradient.allocation,
            )
        norm2 = (
            score.coarse.flatten(1).square().sum(1)
            + score.allocation.flatten(1).square().sum(1)
        ).clamp_min(1.0e-8)
        dimension = score.coarse[0].numel() + score.allocation[0].numel()
        delta = cfg.corrector_tau * dimension / norm2
        if cfg.corrector_max_step is not None:
            delta = delta.clamp_max(float(cfg.corrector_max_step))
        dc = delta.view(-1, 1, 1, 1)
        state = HierarchicalState(
            state.coarse + dc * score.coarse + torch.sqrt(2.0 * dc) * torch.randn_like(state.coarse),
            state.allocation
            + dc * score.allocation
            + torch.sqrt(2.0 * dc) * torch.randn_like(state.allocation),
        )
        if oracle_truth is not None:
            state.coarse = _oracle_coarse(oracle_truth, oracle_noise, t)
    return state


def sample_hierarchical(
    model: CoupledSubgridFlow,
    coarse_cond: torch.Tensor | None,
    fine_cond: torch.Tensor | None,
    coarse_shape: tuple[int, int, int, int],
    allocation_shape: tuple[int, int, int, int],
    coarse_valid: torch.Tensor,
    fine_valid: torch.Tensor,
    area: torch.Tensor,
    encoding: SubgridEncoding,
    *,
    observations: HierarchicalObservations | None = None,
    config: HierarchicalSamplerConfig | None = None,
    flow: HierarchicalRectifiedFlow | None = None,
    initial_noise: HierarchicalState | None = None,
    oracle_coarse_truth: torch.Tensor | None = None,
) -> HierarchicalSample:
    """Draw a joint background or guided analysis ensemble.

    ``oracle_coarse_truth`` activates the clean-context mode seen during joint
    training.  It is deliberately rejected for arbitrary models, preventing a
    naive replacement sampler from being mislabeled as ``p(z | m_truth)``.
    """
    cfg = config or HierarchicalSamplerConfig()
    flow = flow or HierarchicalRectifiedFlow()
    encoding.validate()
    if not isinstance(model, CoupledSubgridFlow):
        raise TypeError("sample_hierarchical requires one CoupledSubgridFlow")
    if oracle_coarse_truth is not None and not getattr(
        model, "supports_clean_coarse_context", False
    ):
        raise ValueError(
            "oracle clamping requires a joint model trained with clean coarse context"
        )
    if oracle_coarse_truth is not None and not model.clean_context_trained:
        raise ValueError(
            "oracle clamping is disabled because this checkpoint records zero "
            "clean-context training probability"
        )
    if coarse_shape[0] != allocation_shape[0] or coarse_shape[1] != 2 or allocation_shape[1] != 2:
        raise ValueError("coarse/allocation shapes must be (same_B,2,H,W)")
    device = next(model.parameters()).device
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
    model.eval()
    batch = coarse_shape[0]
    coarse_cond = _expand_condition(coarse_cond, batch)
    fine_cond = _expand_condition(fine_cond, batch)
    if initial_noise is None:
        state = HierarchicalState(
            torch.randn(coarse_shape, device=device),
            torch.randn(allocation_shape, device=device),
        )
    else:
        state = initial_noise.to(device)
    oracle_noise = state.coarse.detach().clone()
    if oracle_coarse_truth is not None:
        oracle_coarse_truth = oracle_coarse_truth.to(device)
        if oracle_coarse_truth.shape[0] == 1 and batch > 1:
            oracle_coarse_truth = oracle_coarse_truth.expand(batch, -1, -1, -1)
        if oracle_coarse_truth.shape != state.coarse.shape:
            raise ValueError("oracle_coarse_truth shape must equal the coarse state shape")
        state.coarse = _oracle_coarse(oracle_coarse_truth, oracle_noise, 0.0)

    times = _schedule(cfg, device)
    last_likelihood_field = None
    for index in range(cfg.n_steps):
        t0, t1 = float(times[index]), float(times[index + 1])
        dt = t1 - t0
        time0 = torch.full((batch,), t0, device=device)
        clean_context = oracle_coarse_truth
        if observations is None:
            with torch.no_grad():
                velocity = _model_velocity(
                    model, state, time0, coarse_cond, fine_cond, clean_context
                )
            applied = velocity
        else:
            velocity, gradient, last_likelihood_field = hierarchical_guidance_grad(
                model, state, time0, coarse_cond, fine_cond, observations, flow,
                encoding, coarse_valid, fine_valid, area,
                cfg.occurrence_temperature, clean_context,
            )
            applied = _guided_velocity(
                velocity, gradient, state, time0, flow, observations.guidance
            )

        euler = _add(state, applied, dt)
        if oracle_coarse_truth is not None:
            euler.coarse = _oracle_coarse(oracle_coarse_truth, oracle_noise, t1)
        if cfg.heun and index < cfg.n_steps - 1:
            time1 = torch.full((batch,), t1, device=device)
            if observations is None:
                with torch.no_grad():
                    velocity1 = _model_velocity(
                        model, euler, time1, coarse_cond, fine_cond, clean_context
                    )
                applied1 = velocity1
            else:
                velocity1, gradient1, last_likelihood_field = hierarchical_guidance_grad(
                    model, euler, time1, coarse_cond, fine_cond, observations, flow,
                    encoding, coarse_valid, fine_valid, area,
                    cfg.occurrence_temperature, clean_context,
                )
                applied1 = _guided_velocity(
                    velocity1, gradient1, euler, time1, flow, observations.guidance
                )
            state = HierarchicalState(
                state.coarse + 0.5 * dt * (applied.coarse + applied1.coarse),
                state.allocation
                + 0.5 * dt * (applied.allocation + applied1.allocation),
            )
        else:
            state = euler
        if oracle_coarse_truth is not None:
            state.coarse = _oracle_coarse(oracle_coarse_truth, oracle_noise, t1)
        if cfg.n_corrections and 0.0 < t1 < 1.0:
            state = _correct(
                model, state, t1, coarse_cond, fine_cond, observations, flow,
                encoding, coarse_valid, fine_valid, area, cfg,
                oracle_coarse_truth, oracle_noise,
            )

    physical = decode_and_reconstruct(
        state.coarse, state.allocation, coarse_valid, fine_valid, area, encoding,
        temperature=cfg.occurrence_temperature, hard=True,
    )
    soft = decode_and_reconstruct(
        state.coarse, state.allocation, coarse_valid, fine_valid, area, encoding,
        temperature=cfg.occurrence_temperature, hard=False,
    )
    # Re-evaluate the exact hard terminal state used by the likelihood.  This
    # catches any future divergence between the field returned to an archive
    # writer and the decoder/operator path used during DA.
    terminal_likelihood_field = decode_and_reconstruct(
        state.coarse, state.allocation, coarse_valid, fine_valid, area, encoding,
        temperature=cfg.occurrence_temperature, hard=True,
    )
    terminal_decoder_error = float(
        (physical - terminal_likelihood_field).abs().max().detach().cpu()
    )
    if terminal_decoder_error > 1.0e-6:
        raise RuntimeError(
            "terminal hard-decoder mismatch between returned and likelihood fields: "
            f"{terminal_decoder_error:.3g} mm/day"
        )
    diagnostics = {
        "n_steps": cfg.n_steps,
        "heun": cfg.heun,
        "oracle_clean_context": oracle_coarse_truth is not None,
        "terminal_hard_decode_max_abs_mm_day": terminal_decoder_error,
        "terminal_decoder_consistent": True,
    }
    if last_likelihood_field is not None:
        diagnostics["last_guidance_to_final_field_max_abs_mm"] = float(
            (physical - last_likelihood_field).abs().max().cpu()
        )
    if observations is not None:
        values = _expand_observations(observations, batch)
        hard_hx = observations.operator(terminal_likelihood_field)
        soft_hx = observations.operator(soft)
        sigma = torch.sqrt(observations.variance.to(hard_hx).clamp_min(1.0e-12))
        terminal_time = torch.ones(batch, device=hard_hx.device, dtype=hard_hx.dtype)
        terminal_ll = obs_log_likelihood(
            values, hard_hx, observations.variance, terminal_time,
            observations.guidance,
        )
        valid = (
            torch.isfinite(values)
            & torch.isfinite(hard_hx)
            & torch.isfinite(sigma)
        )
        terminal_residual = torch.where(
            valid, (hard_hx - values) / sigma, torch.zeros_like(hard_hx)
        )
        residual_values = terminal_residual[valid]
        diagnostics["terminal_log_likelihood_mean"] = float(
            terminal_ll.mean().detach().cpu()
        )
        diagnostics["terminal_valid_observation_count"] = int(valid.sum().cpu())
        diagnostics["terminal_oa_bias_sigma"] = (
            float(residual_values.mean().detach().cpu())
            if residual_values.numel() else 0.0
        )
        diagnostics["terminal_oa_rmse_sigma"] = (
            float(residual_values.square().mean().sqrt().detach().cpu())
            if residual_values.numel() else 0.0
        )
        diagnostics["terminal_oa_max_abs_sigma"] = (
            float(residual_values.abs().max().detach().cpu())
            if residual_values.numel() else 0.0
        )
        difference = ((hard_hx - soft_hx).abs() / sigma)
        difference = difference[torch.isfinite(values) & torch.isfinite(difference)]
        if difference.numel():
            median = float(torch.quantile(difference, 0.50).cpu())
            p95 = float(torch.quantile(difference, 0.95).cpu())
        else:
            median = p95 = 0.0
        diagnostics["soft_hard_oa_median_sigma"] = median
        diagnostics["soft_hard_oa_p95_sigma"] = p95
        diagnostics["soft_hard_bounds_pass"] = bool(
            median <= cfg.soft_hard_median_sigma and p95 <= cfg.soft_hard_p95_sigma
        )
    return HierarchicalSample(state.detach(), physical.detach(), diagnostics)


def authority_decomposition(
    background: HierarchicalState,
    analysis: HierarchicalState,
    coarse_valid: torch.Tensor,
    fine_valid: torch.Tensor,
    area: torch.Tensor,
    encoding: SubgridEncoding,
    *,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Symmetric amount/allocation attribution in physical mm/day."""
    def reconstruct(coarse, allocation):
        return decode_and_reconstruct(
            coarse, allocation, coarse_valid, fine_valid, area, encoding,
            temperature=temperature, hard=True,
        )

    x_bb = reconstruct(background.coarse, background.allocation)
    x_ab = reconstruct(analysis.coarse, background.allocation)
    x_ba = reconstruct(background.coarse, analysis.allocation)
    x_aa = reconstruct(analysis.coarse, analysis.allocation)
    amount = 0.5 * ((x_ab - x_bb) + (x_aa - x_ba))
    allocation = 0.5 * ((x_ba - x_bb) + (x_aa - x_ab))
    residual = (amount + allocation) - (x_aa - x_bb)
    return amount, allocation, residual


def amount_authority_share(amount: torch.Tensor, allocation: torch.Tensor, mask=None) -> float:
    if mask is not None:
        amount = amount * mask
        allocation = allocation * mask
    numerator = amount.abs().sum()
    denominator = numerator + allocation.abs().sum()
    return float((numerator / denominator.clamp_min(1.0e-12)).detach().cpu())
