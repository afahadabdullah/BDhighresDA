"""Observation likelihood and its gradient (the "guidance" term).

Following Rozet & Louppe (2024) and Manshausen et al. (2025) Eq. (3), the
likelihood of the observations given a *noisy* state is approximated by a
Gaussian centred on the observation operator applied to the DENOISED state:

    p(y | x_t) = N( y | H(x1_hat),  R + (sigma_t^2 / mu_t^2) * Gamma )

For the rectified-flow interpolant used here, mu_t = t and sigma_t = 1 - t, so

    V(t) = R + Gamma * (1 - t)^2 / t^2

which -> R as t -> 1 (clean state) and blows up as t -> 0 (pure noise), i.e.
the observations are correctly down-weighted early in the trajectory.  Gamma
is a scalar hyperparameter; Manshausen et al. found 1e-3 better than the 1e-2
of the original SDA paper, particularly for the precipitation channel.

The log-likelihood is then

    log p(y|x_t) = -0.5 * (y - H(x1_hat))^T V^-1 (y - H(x1_hat))

and the guidance term added to the velocity field is (see models/flow.py, Eq. C)

    d(u) = ((1 - t) / t) * grad_{x_t} log p(y | x_t)

The gradient is taken THROUGH the network (diffusion posterior sampling), so a
guided sample costs roughly 2-3x an unguided one.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class GuidanceConfig:
    gamma: float = 1e-3          # Gamma in Eq. (3); SDA default 1e-2
    scale: float = 1.0           # extra multiplier on the guidance term
    t_start: float = 0.10        # do not guide below this t (variance explodes)
    t_end: float = 0.999         # stop guiding very close to t = 1
    clip_norm: float | None = 50.0   # per-sample grad-norm clip, guards blow-ups
    huber_delta: float | None = None  # if set, use a Huber (robust) cost in place of L2
    spread_cells: float = 0.0
    """Gaussian spreading length for the guidance gradient, in grid cells.

    A point gauge enters through a bilinear operator, so the likelihood gradient
    with respect to the reconstructed field is non-zero at only the four cells
    surrounding the station. Everything further out relies on the network's own
    Jacobian to carry the correction, and in this system that carries it barely:
    OSSE analyses show bright spots pinned to individual gauges with the field
    between them essentially unchanged, while sub-footprint correlation with
    truth sits at 0.03. With exact observations R is near zero, so the analysis
    is forced onto each gauge exactly, and a nearly diagonal implied covariance
    then leaves that correction nowhere to go.

    Setting this convolves the gradient with an isotropic Gaussian of this
    standard deviation before it is applied, so one observation informs a
    neighbourhood rather than a pixel. On the 0.05-degree grid one cell is
    ~5.5 km, and the measured variogram range for the BMD network is ~146 km,
    so 6-12 cells (33-66 km) is the physically motivated range.

    THIS IS AN APPROXIMATION AND MUST BE REPORTED AS ONE. Exact diffusion
    posterior guidance uses the unsmoothed gradient; smoothing it asserts a
    broader background covariance than the network implies. It is a statement
    about B, not a numerical trick, and 0.0 (off) remains the default so every
    earlier result is unchanged.
    """


def obs_log_likelihood(
    y: torch.Tensor,          # (B, C, S) transformed observations, NaN where missing
    hx: torch.Tensor,         # (B, C, S) = H(x1_hat)
    R: torch.Tensor,          # (S,) or (B, C, S) observation error variance
    t: torch.Tensor,          # (B,)
    cfg: GuidanceConfig,
) -> torch.Tensor:
    """Return the per-sample log-likelihood (B,).  Missing obs are skipped."""
    tb = t.view(-1, 1, 1).clamp_min(1e-4)
    inflation = cfg.gamma * ((1.0 - tb) ** 2) / (tb**2)
    V = R.to(hx.dtype) + inflation

    valid = torch.isfinite(y)
    resid = torch.where(valid, y - hx, torch.zeros_like(hx))

    if cfg.huber_delta is None:
        cost = resid**2 / V
    else:
        a = resid.abs() / V.sqrt()
        d = cfg.huber_delta
        cost = torch.where(a <= d, a**2, d * (2 * a - d))
    cost = torch.where(valid, cost, torch.zeros_like(cost))
    return -0.5 * cost.flatten(1).sum(dim=1)


def guidance_grad(
    x_t: torch.Tensor,
    t: torch.Tensor,
    model,
    flow,
    cond: torch.Tensor | None,
    H,
    y: torch.Tensor,
    R: torch.Tensor,
    cfg: GuidanceConfig,
    mask: torch.Tensor | None = None,
    mask_fill: float = 0.0,
    to_precip=None,
):
    """Return ``(velocity, grad_x log p(y|x_t))``.

    The unguided velocity is returned alongside so the caller does not pay for
    a second network evaluation.

    ``mask_fill`` is the transformed-space value of 0 mm.  It matters here because
    the block-average observation operator averages over 2x2 cells that may
    straddle the coast: pinning the ocean half to a literal 0.0 would inject a
    spurious rain rate into the modelled satellite observation.

    ``to_precip`` maps the network's variable into transformed-precipitation
    space.  It is the identity for an absolute target and adds the ERA5 base for
    a residual target -- the observation operator compares against measured
    rainfall either way, so the likelihood must always be evaluated on the
    reconstructed field, never on a bare residual.  The gradient still flows back
    through it to ``x_t`` because the mapping is affine and differentiable.
    """
    # torch.inference_mode() is strictly stronger than no_grad(): tensors created
    # inside it are permanently barred from autograd, and enable_grad() below
    # cannot lift that.  The failure surfaces deep in the backward engine as
    # "element 0 of tensors does not require grad", which points nowhere useful,
    # so check for it here where the cause can be named.
    if torch.is_inference_mode_enabled():
        raise RuntimeError(
            "guidance_grad differentiates through the network and cannot run "
            "under torch.inference_mode(). Drop the inference_mode context "
            "around the guided sampler call; use torch.no_grad() only for "
            "UNGUIDED sampling, and .detach() the result instead."
        )
    with torch.enable_grad():
        x = x_t.detach().requires_grad_(True)
        u = model(x, t, cond)
        x1_hat = flow.x1_hat(x, t, u)
        if mask is not None:
            x1_hat = x1_hat * mask + mask_fill * (1.0 - mask)
        if to_precip is not None:
            x1_hat = to_precip(x1_hat)
        hx = H(x1_hat)
        ll = obs_log_likelihood(y, hx, R, t, cfg).sum()
        (grad,) = torch.autograd.grad(ll, x)

    if cfg.spread_cells and cfg.spread_cells > 0.0:
        grad = spread_gradient(grad, cfg.spread_cells, mask=mask)

    # Clip AFTER spreading. Spreading redistributes the gradient without
    # changing its total, but it lowers the peak and hence the norm, so
    # clipping first would rescale against a norm the applied field no longer
    # has.
    if cfg.clip_norm is not None:
        n = grad.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
        grad = grad * (cfg.clip_norm / n.clamp_min(cfg.clip_norm))

    return u.detach(), grad.detach()


def spread_gradient(
    grad: torch.Tensor, sigma_cells: float, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Convolve the guidance gradient with an isotropic Gaussian.

    Separable, so cost is linear rather than quadratic in the kernel width.

    ``mask`` is respected by normalising against the smoothed mask, so a gauge
    near the coast spreads along the land rather than bleeding its increment
    into the sea and coming back diluted. Without that renormalisation a coastal
    station would be systematically under-corrected relative to an inland one.
    """
    if sigma_cells <= 0.0:
        return grad
    radius = int(max(1, round(3.0 * float(sigma_cells))))
    offsets = torch.arange(
        -radius, radius + 1, device=grad.device, dtype=grad.dtype
    )
    kernel = torch.exp(-0.5 * (offsets / float(sigma_cells)) ** 2)
    kernel = kernel / kernel.sum()
    channels = grad.shape[1]
    # .contiguous(): conv2d weights from .expand() are non-contiguous views and
    # some backends reject or silently copy them.
    kx = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1).contiguous()
    ky = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1).contiguous()

    def blur(field: torch.Tensor) -> torch.Tensor:
        field = F.conv2d(field, kx, padding=(0, radius), groups=field.shape[1])
        return F.conv2d(field, ky, padding=(radius, 0), groups=field.shape[1])

    if mask is None:
        return blur(grad)
    weight = mask.to(grad.dtype)
    while weight.dim() < grad.dim():
        weight = weight.unsqueeze(0)
    weight = weight.expand(grad.shape[0], channels, *grad.shape[2:]).contiguous()
    smoothed = blur(grad * weight)
    norm = blur(weight)
    return torch.where(norm > 1e-6, smoothed / norm.clamp_min(1e-6), grad) * weight


def guided_velocity(u, grad, t, flow, cfg: GuidanceConfig, ref):
    """Combine prior velocity and likelihood gradient -- Eq. (C)."""
    factor = flow.score_to_velocity_factor(t, ref)
    active = ((t >= cfg.t_start) & (t <= cfg.t_end)).view(-1, 1, 1, 1).to(u.dtype)
    return u + cfg.scale * factor * grad * active
