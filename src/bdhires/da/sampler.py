"""ODE samplers for the rectified flow, with and without observation guidance.

Three modes:

* ``sample``           -- conditional downscaling (ERA5 + IMERG -> 5 km).  No
                          station data.  This is the "background"/first guess.
* ``assimilate``       -- the same trajectory, guided at every step by the
                          station likelihood.  This is the analysis.
* ``assimilate(cond=None)`` -- pure SDA in the Manshausen sense: an
                          unconditional prior guided only by observations.
                          Works because the model is trained with conditioning
                          dropout.

Integration uses Heun (2nd order) with a non-uniform schedule that takes
smaller steps as t -> 1, plus optional Langevin Monte Carlo corrector steps
(Rozet & Louppe 2024, Algorithm 4) to stop guidance errors accumulating.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..models.flow import RectifiedFlow
from .guidance import GuidanceConfig, guidance_grad, guided_velocity


@dataclass
class SamplerConfig:
    n_steps: int = 50            # ODE steps (2 NFE each with Heun)
    heun: bool = True
    schedule_power: float = 2.0  # t_i = 1 - (1 - i/n)^p ; p=1 is uniform
    n_corrections: int = 2       # Langevin corrector steps per ODE step (C in SDA)
    corrector_tau: float = 0.3   # tau~ in SDA; step size delta = tau~ * dim(s)/||s||^2
    churn: float = 0.0           # optional stochasticity, 0 = deterministic ODE
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
    """Unguided conditional generation (the ML downscaler)."""
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
    """Generate an ensemble member, optionally guided by station observations.

    ``shape`` is ``(B, C, H, W)``; B is the ensemble size (all members share
    the same conditioning, which is broadcast).
    """
    cfg = cfg or SamplerConfig()
    gcfg = gcfg or GuidanceConfig()
    flow = flow or RectifiedFlow()
    model.eval()

    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)

    x = torch.randn(shape, device=device) if x0 is None else x0.to(device)
    if cond is not None and cond.shape[0] == 1 and shape[0] > 1:
        cond = cond.expand(shape[0], -1, -1, -1)

    guided = H is not None and y is not None

    def guide(xx, tt):
        return guidance_grad(xx, tt, model, flow, cond, H, y, R, gcfg, mask=mask)

    ts = make_schedule(cfg, device)

    for i in range(cfg.n_steps):
        t0, t1 = ts[i], ts[i + 1]
        dt = t1 - t0
        tb = torch.full((shape[0],), float(t0), device=device)

        if guided:
            u, g = guide(x, tb)
            v = guided_velocity(u, g, tb, flow, gcfg, x)
        else:
            with torch.no_grad():
                v = model(x, tb, cond)

        if cfg.heun and i < cfg.n_steps - 1:
            x_eul = x + dt * v
            tb1 = torch.full((shape[0],), float(t1), device=device)
            if guided:
                u1, g1 = guide(x_eul, tb1)
                v1 = guided_velocity(u1, g1, tb1, flow, gcfg, x_eul)
            else:
                with torch.no_grad():
                    v1 = model(x_eul, tb1, cond)
            x = x + dt * 0.5 * (v + v1)
        else:
            x = x + dt * v

        if cfg.n_corrections and 0.0 < float(t1) < 1.0:
            x = _langevin_correct(
                x, float(t1), model, flow, cond, cfg, guide=guide if guided else None
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
    **kwargs,
) -> torch.Tensor:
    """Generate ``n_members`` samples in memory-safe chunks. Returns (N, C, H, W)."""
    out = []
    remaining = n_members
    while remaining > 0:
        b = min(chunk, remaining)
        out.append(assimilate(model, cond, (b, *shape_chw), device, **kwargs).cpu())
        remaining -= b
    return torch.cat(out, dim=0)
