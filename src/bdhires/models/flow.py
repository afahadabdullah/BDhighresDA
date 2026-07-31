"""Rectified-flow / stochastic-interpolant machinery.

Convention (Lipman et al. 2023; Albergo et al. 2025), matching Wetherell (2026):

    x_t = alpha_t * x1 + beta_t * x0,   alpha_t = t,  beta_t = 1 - t
    x0 ~ N(0, I)   (noise, t = 0)
    x1 ~ p_data    (high-res precipitation field, t = 1)

The network regresses the conditional velocity  u = x1 - x0.

Everything the data-assimilation code needs follows analytically from that.
With alpha_t = t, beta_t = 1-t and Gaussian x0, the conditional path is
``p(x_t | x1) = N(t*x1, (1-t)^2 I)``, so with
``x1_hat = x_t + (1-t) u`` and ``x0_hat = x_t - t u``:

    score(x_t) := grad_x log p_t(x_t) = -(x_t - t * x1_hat) / (1-t)^2
                                      = -x0_hat / (1 - t)                (A)

and, inverting,

    u(x_t) = (x_t + (1-t) * score) / t                                   (B)

Equation (B) is the key to observation guidance: an additive perturbation
``d(score)`` to the score corresponds to an additive perturbation

    d(u) = ((1 - t) / t) * d(score)                                      (C)

to the velocity.  This is the flow-matching analogue of the score-based data
assimilation update of Rozet & Louppe (2024) / Manshausen et al. (2025), and
lets us keep a *single* trained network for both downscaling (conditional
generation) and assimilation (guided generation).

Classifier-free-style conditioning dropout during training means the same
weights also give an *unconditional* prior p(x1) -- i.e. the exact object
Manshausen et al. train separately -- for pure "DA without a background" runs.
"""

from __future__ import annotations

import torch


class RectifiedFlow:
    """Stateless helper implementing the linear interpolant path."""

    def __init__(self, t_min: float = 1e-3, t_max: float = 1.0 - 1e-3):
        self.t_min = t_min
        self.t_max = t_max

    # -- training ----------------------------------------------------------
    def sample_t(self, batch: int, device, logit_normal: bool = True) -> torch.Tensor:
        """Sample interpolation times.

        ``logit_normal`` uses the SD3 timestep density, which concentrates
        samples near t = 0.5 where the velocity field is hardest to learn and
        is a consistent win over uniform sampling for image-like data.
        """
        if logit_normal:
            t = torch.sigmoid(torch.randn(batch, device=device))
        else:
            t = torch.rand(batch, device=device)
        return t.clamp(self.t_min, self.t_max)

    def interpolate(self, x1: torch.Tensor, t: torch.Tensor, x0: torch.Tensor | None = None):
        """Return ``(x_t, target_velocity, x0)``."""
        if x0 is None:
            x0 = torch.randn_like(x1)
        tb = t.view(-1, *([1] * (x1.ndim - 1)))
        x_t = tb * x1 + (1.0 - tb) * x0
        return x_t, x1 - x0, x0

    # -- conversions -------------------------------------------------------
    @staticmethod
    def _tb(t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return t.view(-1, *([1] * (ref.ndim - 1)))

    def x1_hat(self, x_t: torch.Tensor, t: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """E[x1 | x_t] -- the denoised / clean-field estimate."""
        return x_t + (1.0 - self._tb(t, x_t)) * u

    def x0_hat(self, x_t: torch.Tensor, t: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return x_t - self._tb(t, x_t) * u

    def score(self, x_t: torch.Tensor, t: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """grad_x log p_t(x_t), equation (A)."""
        tb = self._tb(t, x_t)
        return -self.x0_hat(x_t, t, u) / (1.0 - tb).clamp_min(1e-4)

    def velocity_from_score(self, x_t, t, score):
        """Equation (B)."""
        tb = self._tb(t, x_t)
        return (x_t + (1.0 - tb) * score) / tb.clamp_min(1e-4)

    def score_to_velocity_factor(self, t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Equation (C): the (1-t)/t factor converting a score nudge to a velocity nudge."""
        tb = self._tb(t, ref)
        return (1.0 - tb) / tb.clamp_min(1e-4)

    # -- SDA noise-schedule analogues -------------------------------------
    @staticmethod
    def mu_sigma(t: torch.Tensor):
        """(mu_t, sigma_t) in the notation of Manshausen et al. Eq. (1)."""
        return t, 1.0 - t


def flow_matching_loss(
    model,
    x1: torch.Tensor,
    cond: torch.Tensor | None,
    flow: RectifiedFlow,
    mask: torch.Tensor | None = None,
    cond_dropout: float = 0.1,
    logit_normal_t: bool = True,
    clean_loss_fn=None,
    return_components: bool = False,
):
    """Conditional flow-matching loss with optional conditioning dropout.

    ``mask``: 1 where the target is valid (CHIRPS is land-only -- ocean cells
    must be excluded from the loss or the model wastes capacity learning the
    fill value and the DA guidance leaks over the Bay of Bengal).
    """
    b = x1.shape[0]
    t = flow.sample_t(b, x1.device, logit_normal=logit_normal_t)
    x_t, target, _ = flow.interpolate(x1, t)

    if cond is not None and cond_dropout > 0.0:
        keep = (torch.rand(b, 1, 1, 1, device=x1.device) > cond_dropout).float()
        cond = cond * keep

    pred = model(x_t, t, cond)
    err = (pred - target) ** 2
    if mask is not None:
        m = mask.expand_as(err)
        flow_loss = (err * m).sum() / m.sum().clamp_min(1.0)
    else:
        flow_loss = err.mean()

    auxiliary_loss = pred.new_zeros(())
    if clean_loss_fn is not None:
        clean = flow.x1_hat(x_t, t, pred)
        auxiliary_loss = clean_loss_fn(clean)
        if auxiliary_loss.ndim:
            raise ValueError("clean_loss_fn must return a scalar")
    total = flow_loss + auxiliary_loss
    if return_components:
        return total, flow_loss, auxiliary_loss
    return total


def select_weights(checkpoint: dict) -> dict:
    """Return the state dict a checkpoint intends for inference.

    Checkpoints written with ``train.use_ema: true`` carry both ``model`` (online)
    and ``ema`` weights and mean the latter.  With ``use_ema: false`` there is no
    EMA and ``model`` is authoritative.  The ``weights`` key records which, and
    older checkpoints that predate the flag always carry usable EMA weights.
    """
    preferred = checkpoint.get("weights")
    if preferred == "model":
        return checkpoint["model"]
    if preferred == "ema" or checkpoint.get("ema") is not None:
        ema = checkpoint.get("ema")
        if ema is None:
            raise ValueError("checkpoint claims EMA weights but carries none")
        return ema
    if "model" not in checkpoint:
        raise ValueError("checkpoint contains neither 'ema' nor 'model' weights")
    return checkpoint["model"]


class EMA:
    """Exponential moving average of model weights (decay 0.999 by default)."""

    def __init__(self, model, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model):
        model.load_state_dict({k: v.to(dtype=p.dtype) for (k, v), p in
                               zip(self.shadow.items(), model.state_dict().values())})

    def state_dict(self):
        return self.shadow
