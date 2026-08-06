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
import torch.nn.functional as F


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


def apply_dry_mask(
    field_t: torch.Tensor,
    dry_logit: torch.Tensor,
    dry_value: float,
    mode: str = "threshold",
    threshold: float = 0.5,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Impose the atom at zero rainfall on a field in TRANSFORMED space.

    Rectified flow models a continuous density. Daily rainfall is not
    continuous: CHIRPS is exactly zero over roughly half the domain, and no
    amount of training makes a smooth density put finite mass on a single
    point. The consequence measured on the v1 prior was a wet-day frequency of
    0.649 against a target of 0.459, biased worst in dry regions -- which is
    where the BMD gauges are, so it showed up as +5.88 mm/day at stations
    against +2.65 domain-wide.

    The hurdle head predicts P(dry) directly and this applies it, which is the
    only part of the design that can represent the atom exactly.

    ``mode``:
      ``"threshold"``  dry where P(dry) > ``threshold``. Deterministic given the
                       member; ensemble spread survives because each member
                       follows its own trajectory and so gets its own P(dry).
                       Spatially coherent, which matters for a rainfall field.
      ``"sample"``     Bernoulli draw. Correct marginal wet fraction by
                       construction, but speckles the wet/dry boundary.

    ``dry_value`` must be ``transform.forward(0.0)`` -- in transformed space
    zero rainfall is not zero, it is ``-mu/sd``.
    """
    probability = torch.sigmoid(dry_logit)
    if mode == "sample":
        noise = torch.rand(
            probability.shape, device=probability.device,
            dtype=probability.dtype, generator=generator,
        )
        is_dry = noise < probability
    elif mode == "threshold":
        is_dry = probability > threshold
    else:
        raise ValueError(f"unknown dry mask mode {mode!r}")
    return torch.where(is_dry, torch.full_like(field_t, dry_value), field_t)


class VelocityOnly(torch.nn.Module):
    """Present a 2-channel hurdle network through the 1-channel interface.

    ``sampler.assimilate`` and ``guidance.guidance_grad`` both call
    ``model(x, t, cond)`` and treat the result as the velocity. Rather than
    thread a channel index through both, wrap the network once at the call
    site: the flow machinery is then completely unaware the hurdle head exists.

    The dry logit is NOT cached from these calls on purpose. With
    classifier-free guidance the last evaluation is the *unconditional* branch,
    so a cache would silently hand back the wrong logit. Read it explicitly
    with :func:`predict_dry_logit` once sampling has finished.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, t, cond=None):
        out = self.model(x, t, cond)
        return out[:, :1] if out.shape[1] > 1 else out


@torch.no_grad()
def predict_dry_logit(model, x1: torch.Tensor, cond, t_eval: float = 0.99) -> torch.Tensor:
    """Dry-probability logit for a finished sample.

    Evaluated at ``t_eval`` close to 1, where ``x_t`` is within a percent of the
    clean field, so the classifier is used in the regime it is most accurate in.
    ``x1`` is passed directly rather than re-noised: at t = 0.99 the noise term
    contributes 1% and adding it would only make the mask stochastic for no
    modelling gain.
    """
    batch = x1.shape[0]
    tb = torch.full((batch,), float(t_eval), device=x1.device, dtype=x1.dtype)
    out = model(x1, tb, cond)
    if out.shape[1] < 2:
        raise ValueError(
            "model has no hurdle head; train with train.hurdle.enabled to use it"
        )
    return out[:, 1:2]


def split_prediction(pred: torch.Tensor, hurdle: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Separate the flow velocity from the hurdle logit in the model output."""
    if not hurdle:
        return pred, None
    if pred.shape[1] < 2:
        raise ValueError(
            f"hurdle head requires out_channels >= 2, model returned {pred.shape[1]}"
        )
    return pred[:, :1], pred[:, 1:2]


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
    dry_target: torch.Tensor | None = None,
    hurdle_weight: float = 1.0,
    dry_weight: float = 1.0,
):
    """Conditional flow-matching loss with optional conditioning dropout.

    ``mask``: 1 where the target is valid (CHIRPS is land-only -- ocean cells
    must be excluded from the loss or the model wastes capacity learning the
    fill value and the DA guidance leaks over the Bay of Bengal).

    ``dry_target``: 1 where the CLEAN field is dry, same shape as ``x1``.
    Supplying it switches on the hurdle head, which must then be present as a
    second model output channel. The head is trained at every ``t``: near
    ``t=1`` the input is almost clean and the classification is easy, near
    ``t=0`` it is nearly pure noise and the head can only fall back on the
    conditioning. That is the same difficulty gradient x1-prediction already
    has, and it is what makes the head usable at the end of sampling.

    ``dry_weight``: multiplier on dry cells in the flow MSE. The v1 failure was
    concentrated in the dry regime -- an unweighted mean over a field that is
    half dry still lets the wet half dominate the gradient, because wet cells
    carry far larger residuals. Values above 1 buy dry-end accuracy at some
    cost to the extreme tail, so this is a knob, not a default.
    """
    b = x1.shape[0]
    t = flow.sample_t(b, x1.device, logit_normal=logit_normal_t)
    x_t, target, _ = flow.interpolate(x1, t)

    if cond is not None and cond_dropout > 0.0:
        keep = (torch.rand(b, 1, 1, 1, device=x1.device) > cond_dropout).float()
        cond = cond * keep

    raw = model(x_t, t, cond)
    pred, dry_logit = split_prediction(raw, dry_target is not None)

    err = (pred - target) ** 2
    weights = torch.ones_like(err) if mask is None else mask.expand_as(err).clone().float()
    if dry_target is not None and dry_weight != 1.0:
        weights = weights * (1.0 + (dry_weight - 1.0) * dry_target.expand_as(err))
    flow_loss = (err * weights).sum() / weights.sum().clamp_min(1.0)

    hurdle_loss = pred.new_zeros(())
    if dry_target is not None:
        elementwise = F.binary_cross_entropy_with_logits(
            dry_logit, dry_target.to(dry_logit.dtype), reduction="none"
        )
        m = torch.ones_like(elementwise) if mask is None else mask.expand_as(elementwise).float()
        hurdle_loss = hurdle_weight * (elementwise * m).sum() / m.sum().clamp_min(1.0)

    auxiliary_loss = pred.new_zeros(())
    if clean_loss_fn is not None:
        clean = flow.x1_hat(x_t, t, pred)
        auxiliary_loss = clean_loss_fn(clean)
        if auxiliary_loss.ndim:
            raise ValueError("clean_loss_fn must return a scalar")

    total = flow_loss + auxiliary_loss + hurdle_loss
    if return_components:
        return total, flow_loss, auxiliary_loss, hurdle_loss
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
