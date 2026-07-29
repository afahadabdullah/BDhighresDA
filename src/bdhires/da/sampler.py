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
    ``scripts/smoke_test.py`` (true sd 0.50): T=1.0 -> 0.43, T=1.25 -> 0.56,
    T=1.6 -> 0.71, T=2.0 -> 0.84.

    Inflating the *prior* rather than the analysis is the right place to do it:
    observations then pull the ensemble back where they exist, so spread grows
    where the field is unconstrained and stays tight where it is observed --
    which is the behaviour you actually want from a reanalysis.  The 1/t factor
    is gated below ``temperature_t_start``.

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
    schedule_power: float = 2.0  # t_i = 1 - (1 - i/n)^p ; p = 1 is uniform
    noise_scale: float = 0.0     # eta: 0 = probability-flow ODE, >0 = SDE
    prior_temperature: float = 1.0   # T > 1 broadens the prior (see below)
    temperature_t_start: float = 0.15
    n_corrections: int = 2       # Langevin corrector steps per level (C in SDA)
    corrector_tau: float = 0.3   # tau~ ; step size delta = tau~ * dim(s)/||s||^2
    t_noise_end: float = 0.98    # stop injecting noise near t = 1 (avoids grain)
    seed: int | None = None


def make_schedule(cfg: SamplerConfig, device) -> torch.Tensor:
    i = torch.arange(cfg.n_steps + 1, dtype=torch.float32, device=device) / cfg.n_steps
    return 1.0 - (1.0 - i) ** cfg.schedule_power


@torch.no_grad()
def _langevin_correct(x, t, model, flow, cond, cfg: SamplerConfig, guide=None):
    """C steps of Langevin MC at fixed t using the (possibly guided) score."""
    for _ in range(cfg.n_corrections):
        tb = torch.full((x.shape[0],), float(t), device=x.device)
        if guide is None:
            u = model(x, tb, cond)
            s = flow.score(x, tb, u)
        else:
            u, g = guide(x, tb)
            s = flow.score(x, tb, u) + g
        norm2 = s.flatten(1).pow(2).sum(dim=1).clamp_min(1e-8)
        dim = s[0].numel()
        delta = (cfg.corrector_tau * dim / norm2).view(-1, 1, 1, 1)
        x = x + delta * s + (2 * delta).sqrt() * torch.randn_like(x)
    return x


def sample(
    model,
    cond: torch.Tensor | None,
    shape: tuple[int, ...],
    device,
    cfg: SamplerConfig | None = None,
    flow: RectifiedFlow | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Unguided conditional generation (the downscaler / background)."""
    return assimilate(
        model, cond, shape, device, H=None, y=None, R=None, cfg=cfg, flow=flow, mask=mask
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
) -> torch.Tensor:
    """Generate an ensemble, optionally guided by observations.

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

    def guide(xx, tt):
        return guidance_grad(xx, tt, model, flow, cond, H, y, R, gcfg, mask=mask)

    stochastic = cfg.noise_scale > 0.0
    ts = make_schedule(cfg, device)

    for i in range(cfg.n_steps):
        t0, t1 = float(ts[i]), float(ts[i + 1])
        dt = t1 - t0
        tb = torch.full((n,), t0, device=device)

        if guided:
            u, g = guide(x, tb)
            v = guided_velocity(u, g, tb, flow, gcfg, x)
        else:
            with torch.no_grad():
                u = model(x, tb, cond)
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
                with torch.no_grad():
                    v1 = model(x_eul, tb1, cond)
            x = x + dt * 0.5 * (v + v1)
        else:
            x = x + dt * v

        if cfg.n_corrections and 0.0 < t1 < 1.0:
            x = _langevin_correct(
                x, t1, model, flow, cond, cfg, guide=guide if guided else None
            )

        if mask is not None:
            x = x * mask

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
