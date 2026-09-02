"""Samplers for the rectified flow, with and without observation guidance.

Integration runs over t: 0 (noise) -> 1 (field).

ODE  (``noise_scale = 0``, default)
    dx = u_t(x) dt
    Deterministic given x0; Heun, 2 NFE per step.

SDE  (``noise_scale = eta > 0``)
    dx = [ u_t(x) + (g_t^2/2) * score_t(x) ] dt + g_t dW,   g_t^2 = 2*eta*(1-t)
    With our interpolant the drift correction collapses to ``-eta * x0_hat``,
    so the update is
        x <- x + dt*(u - eta*x0_hat) + sqrt(2*eta*(1-t)*dt) * z.

    IMPORTANT, because it is easy to assume otherwise: this SDE has the *same
    marginals* as the ODE by construction (Albergo et al. 2025).  It is NOT a
    spread-inflation mechanism.  Its purpose is to re-inject entropy so that
    integration error and mode-seeking guidance cannot compound along a
    trajectory.  With an imperfect score it can move spread either way -- in a
    small toy test here it slightly *reduced* it.  Treat eta as something to
    tune, never as the thing that fixes an under-dispersive ensemble.

PRIOR TEMPERATURE  (``prior_temperature = T > 1``)
    This is the knob that genuinely widens the prior.  Sampling the tempered
    density p^(1/T) means using score/T, and converting that to a velocity via
    ``u = (x_t + (1-t)*score)/t`` gives, exactly,

        u_T = u + (1 - 1/T) * x0_hat / t

    one extra term, monotone in T.  Measured on the toy prior in
    ``scripts/smoke_test.py`` (true sd 0.50, ``schedule_power=2.0``): T=1.0 ->
    0.43, T=1.25 -> 0.56, T=1.6 -> 0.71, T=2.0 -> 0.84.  Those figures are
    schedule-dependent, so the smoke test pins the schedule to keep them
    comparable across changes to the sampler defaults.

    Inflating the *prior* rather than the analysis is the right place to do it:
    observations then pull the ensemble back where they exist, so spread grows
    where the field is unconstrained and stays tight where it is observed --
    which is the behaviour you actually want from a reanalysis.  The 1/t factor
    is gated below ``temperature_t_start``.

    CRUCIAL CAVEAT: that argument holds only when observations are present.  For
    the UNGUIDED background nothing pulls members back, and because the prior is
    inflated in *transformed* space, Jensen's inequality biases the mm-space
    ensemble mean high -- T=1.25 produced a +6.4 mm bias on a 1.7 mm day
    (docs/DIAGNOSIS_epoch119.md item 2).  Hence T defaults to 1.0 here and
    configs/da.yaml carries a separate ``background_sampler`` block.

CLASSIFIER-FREE GUIDANCE  (``cfg_scale = w > 1``)
    Training drops the whole conditioning stack to zero with probability
    ``train.cond_dropout`` (models/flow.py), which buys an unconditional branch
    from the same weights.  That branch is only worth paying for if sampling
    actually uses it:

        u_w = u(x_t, 0) + w * ( u(x_t, cond) - u(x_t, 0) )

    w = 1 recovers plain conditional sampling and costs one network evaluation;
    w > 1 sharpens adherence to the ERA5 conditioning at the cost of a second
    evaluation per step and some ensemble spread.  This is the direct remedy when
    samples track the conditioning too weakly -- the epoch-119 background scored a
    *lower* pattern correlation against CHIRPS than the raw ERA5 field it was
    conditioned on (docs/DIAGNOSIS_epoch119.md item 3).  Start around w = 1.5-3
    and tune it against CRPS, not against ensemble-mean RMSE.

Langevin corrector steps (Rozet & Louppe 2024, Alg. 4) run on top of any mode.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..models.flow import RectifiedFlow
from .guidance import GuidanceConfig, guidance_grad, guided_velocity


@dataclass
class SamplerConfig:
    n_steps: int = 50            # ODE/SDE steps (2 NFE each with Heun)
    heun: bool = True            # ignored when noise_scale > 0 (Euler-Maruyama)
    schedule_power: float = 1.0  # t_i = 1 - (1 - i/n)^p ; p = 1 is uniform.
                                 # p > 1 back-loads steps into t -> 1, where the
                                 # field is already decided; prefer p <= 1.
    noise_scale: float = 0.0     # eta: 0 = probability-flow ODE, >0 = SDE
    cfg_scale: float = 1.0       # w: classifier-free guidance (1 = off)
    prior_temperature: float = 1.0   # T > 1 broadens the prior (see below)
    temperature_t_start: float = 0.15
    n_corrections: int = 0       # Langevin corrector steps per level (C in SDA)
    corrector_tau: float = 0.3   # tau~ ; step size delta = tau~ * dim(s)/||s||^2
    corrector_max_step: float | None = 0.3
    # The adaptive delta is singular when prior and observation scores nearly
    # cancel. The state is standardised, so delta <= tau is the conservative
    # Euler stability bound; set None only for exact legacy reproduction.
    t_noise_end: float = 0.98    # stop injecting noise near t = 1 (avoids grain)
    mask_fill: float = 0.0       # value held at masked cells; must be the
                                 # transform of 0 mm, not a literal 0.0
    seed: int | None = None


def apply_mask(x, mask, fill: float = 0.0):
    """Hold masked-out cells at ``fill`` rather than at zero.

    ``PrecipTransform.forward(0 mm)`` is ``-mu/sd``, not 0, so pinning ocean
    cells to a literal 0.0 pins them to a moderate rain rate and lets the global
    attention blocks leak it into the land field.
    """
    if mask is None:
        return x
    # Multiplication does not repair NaNs: NaN * 0 remains NaN. Corrector noise
    # is applied over the whole tensor, so use where() to restore the invariant
    # that every masked cell has the finite training-time fill value.
    return torch.where(
        mask.to(device=x.device, dtype=torch.bool),
        x,
        torch.full_like(x, fill),
    )


def make_schedule(cfg: SamplerConfig, device) -> torch.Tensor:
    i = torch.arange(cfg.n_steps + 1, dtype=torch.float32, device=device) / cfg.n_steps
    return 1.0 - (1.0 - i) ** cfg.schedule_power


@torch.no_grad()
def _langevin_correct(
    x,
    t,
    prior_velocity,
    flow,
    cfg: SamplerConfig,
    guide=None,
    mask=None,
    stats: dict | None = None,
    effective_dim: int | None = None,
):
    """C steps of Langevin MC at fixed t using the (possibly guided) score."""
    for _ in range(cfg.n_corrections):
        # A correction is a full sampler sub-step. Keep the same mask invariant
        # as the outer integrator before every model evaluation, not merely
        # after all C corrections have finished.
        x = apply_mask(x, mask, cfg.mask_fill)
        tb = torch.full((x.shape[0],), float(t), device=x.device)
        if guide is None:
            u = prior_velocity(x, tb)
            s = flow.score(x, tb, u)
        else:
            u, g = guide(x, tb)
            s = flow.score(x, tb, u) + g
        if mask is not None:
            score_mask = mask.to(device=s.device, dtype=torch.bool)
            s = torch.where(score_mask, s, torch.zeros_like(s))
        norm2 = s.flatten(1).pow(2).sum(dim=1).clamp_min(1e-8)
        if effective_dim is None:
            if mask is None:
                dim = s[0].numel()
            else:
                dim = int(
                    score_mask.expand(1, s.shape[1], *s.shape[2:]).sum().item()
                )
        else:
            dim = effective_dim
        raw_delta = cfg.corrector_tau * dim / norm2
        delta = raw_delta
        if cfg.corrector_max_step is not None:
            if cfg.corrector_max_step <= 0.0:
                raise ValueError("corrector_max_step must be positive or None")
            delta = raw_delta.clamp_max(float(cfg.corrector_max_step))
        if stats is not None:
            stats["member_steps"] += x.shape[0]
            stats["capped_member_steps"] = stats["capped_member_steps"] + (
                delta < raw_delta
            ).sum()
            stats["max_raw_step"] = torch.maximum(
                stats["max_raw_step"],
                raw_delta.detach().amax().to(stats["max_raw_step"].dtype),
            )
            stats["max_applied_step"] = torch.maximum(
                stats["max_applied_step"],
                delta.detach().amax().to(stats["max_applied_step"].dtype),
            )
        delta = delta.view(-1, 1, 1, 1)
        x = x + delta * s + (2 * delta).sqrt() * torch.randn_like(x)
        x = apply_mask(x, mask, cfg.mask_fill)
    return x


def sample(
    model,
    cond: torch.Tensor | None,
    shape: tuple[int, ...],
    device,
    cfg: SamplerConfig | None = None,
    flow: RectifiedFlow | None = None,
    mask: torch.Tensor | None = None,
    to_precip=None,
) -> torch.Tensor:
    """Unguided conditional generation (the downscaler / background)."""
    return assimilate(
        model, cond, shape, device, H=None, y=None, R=None, cfg=cfg, flow=flow,
        mask=mask, to_precip=to_precip,
    )


def assimilate(
    model,
    cond: torch.Tensor | None,
    shape: tuple[int, ...],
    device,
    H=None,
    y: torch.Tensor | None = None,
    R: torch.Tensor | None = None,
    cfg: SamplerConfig | None = None,
    gcfg: GuidanceConfig | None = None,
    flow: RectifiedFlow | None = None,
    mask: torch.Tensor | None = None,
    x0: torch.Tensor | None = None,
    to_precip=None,
    diagnostics: dict | None = None,
) -> torch.Tensor:
    """Generate an ensemble, optionally guided by observations.

    ``to_precip`` maps the network's variable to transformed-precipitation
    space: the identity for an absolute target, ``residual * std + mean + base``
    for a residual one.  It is used only where physical units matter -- inside
    the observation likelihood -- so the returned tensor is always in the
    network's own variable and the caller decodes it.

    ``shape`` is ``(B, C, H, W)`` where B is the ensemble size.

    ``y`` may be ``(1, C, S)`` (all members see the same observations) or
    ``(B, C, S)`` (each member sees its own perturbed draw).  The second form
    is strongly preferred: assimilating identical observations into every
    member is the classic cause of an under-dispersive analysis ensemble, in
    exactly the same way that an EnKF with unperturbed observations
    systematically underestimates the analysis covariance.  Use
    ``bdhires.da.observation.perturb_observations`` to build it.
    """
    cfg = cfg or SamplerConfig()
    gcfg = gcfg or GuidanceConfig()
    flow = flow or RectifiedFlow()
    model.eval()

    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)

    n = shape[0]
    x = torch.randn(shape, device=device) if x0 is None else x0.to(device)
    if cond is not None and cond.shape[0] == 1 and n > 1:
        cond = cond.expand(n, -1, -1, -1)

    guided = H is not None and y is not None
    if guided and y.shape[0] == 1 and n > 1:
        y = y.expand(n, -1, -1)

    use_cfg = cond is not None and cfg.cfg_scale != 1.0
    cond_null = torch.zeros_like(cond) if use_cfg else None

    def combine(u_cond, xx, tt):
        """Blend in the unconditional branch -- classifier-free guidance."""
        if not use_cfg:
            return u_cond
        with torch.no_grad():
            u_uncond = model(xx, tt, cond_null)
        return u_uncond + cfg.cfg_scale * (u_cond - u_uncond)

    def prior_velocity(xx, tt):
        with torch.no_grad():
            u = model(xx, tt, cond)
        return combine(u, xx, tt)

    def guide(xx, tt):
        u, g = guidance_grad(
            xx, tt, model, flow, cond, H, y, R, gcfg,
            mask=mask, mask_fill=cfg.mask_fill, to_precip=to_precip,
            diagnostics=diagnostics,
        )
        return combine(u, xx, tt), g

    stochastic = cfg.noise_scale > 0.0
    ts = make_schedule(cfg, device)
    corrector_stats = None
    corrector_effective_dim = None
    if cfg.n_corrections and mask is not None:
        corrector_effective_dim = int(mask[0].to(dtype=torch.bool).sum().item()) * shape[1]
    if diagnostics is not None and cfg.n_corrections:
        corrector_stats = {
            "member_steps": 0,
            "capped_member_steps": torch.zeros((), dtype=torch.long, device=device),
            "max_raw_step": torch.zeros((), dtype=torch.float64, device=device),
            "max_applied_step": torch.zeros((), dtype=torch.float64, device=device),
        }

    for i in range(cfg.n_steps):
        t0, t1 = float(ts[i]), float(ts[i + 1])
        dt = t1 - t0
        tb = torch.full((n,), t0, device=device)

        if guided:
            u, g = guide(x, tb)
            v = guided_velocity(u, g, tb, flow, gcfg, x)
        else:
            u = prior_velocity(x, tb)
            v = u

        if cfg.prior_temperature != 1.0 and t0 >= cfg.temperature_t_start:
            kappa = 1.0 - 1.0 / cfg.prior_temperature
            v = v + kappa * flow.x0_hat(x, tb, u) / max(t0, 1e-3)

        if stochastic:
            # Euler-Maruyama on the SDE with matching marginals.
            eta = cfg.noise_scale
            x0_hat = flow.x0_hat(x, tb, u)
            x = x + dt * (v - eta * x0_hat)
            if t1 < cfg.t_noise_end:
                g_dt = (2.0 * eta * max(1.0 - t0, 0.0) * dt) ** 0.5
                x = x + g_dt * torch.randn_like(x)
        elif cfg.heun and i < cfg.n_steps - 1:
            x_eul = x + dt * v
            tb1 = torch.full((n,), t1, device=device)
            if guided:
                u1, g1 = guide(x_eul, tb1)
                v1 = guided_velocity(u1, g1, tb1, flow, gcfg, x_eul)
            else:
                v1 = prior_velocity(x_eul, tb1)
            x = x + dt * 0.5 * (v + v1)
        else:
            x = x + dt * v

        if cfg.n_corrections and 0.0 < t1 < 1.0:
            x = _langevin_correct(
                x,
                t1,
                prior_velocity,
                flow,
                cfg,
                guide=guide if guided else None,
                mask=mask,
                stats=corrector_stats,
                effective_dim=corrector_effective_dim,
            )

        x = apply_mask(x, mask, cfg.mask_fill)

    if diagnostics is not None:
        if corrector_stats is None:
            diagnostics["corrector"] = {
                "member_steps": 0,
                "capped_member_steps": 0,
                "capped_fraction": 0.0,
                "max_raw_step": None,
                "max_applied_step": None,
                "configured_max_step": cfg.corrector_max_step,
            }
        else:
            member_steps = int(corrector_stats["member_steps"])
            capped = int(corrector_stats["capped_member_steps"].item())
            diagnostics["corrector"] = {
                "member_steps": member_steps,
                "capped_member_steps": capped,
                "capped_fraction": capped / member_steps if member_steps else 0.0,
                "max_raw_step": float(corrector_stats["max_raw_step"].item()),
                "max_applied_step": float(
                    corrector_stats["max_applied_step"].item()
                ),
                "configured_max_step": cfg.corrector_max_step,
            }
    return x


def ensemble(
    model,
    cond,
    n_members: int,
    shape_chw: tuple[int, int, int],
    device,
    chunk: int = 8,
    y=None,
    **kwargs,
) -> torch.Tensor:
    """Generate ``n_members`` samples in memory-safe chunks. Returns (N, C, H, W).

    If ``y`` has one row per member it is sliced consistently with the chunking,
    so per-member observation perturbations survive.
    """
    out = []
    done = 0
    while done < n_members:
        b = min(chunk, n_members - done)
        yb = None
        if y is not None:
            yb = y[done : done + b] if y.shape[0] == n_members else y
        out.append(assimilate(model, cond, (b, *shape_chw), device, y=yb, **kwargs).cpu())
        done += b
    return torch.cat(out, dim=0)
