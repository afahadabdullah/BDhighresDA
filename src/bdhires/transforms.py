"""Precipitation transforms.

Daily rainfall is non-negative, has an atom at zero and a heavy upper tail.
Generative models are trained in a transformed space where the marginal is
roughly Gaussian; the choice of transform materially changes how well the
upper tail is reproduced.

Manshausen et al. (2025) used log/exp and explicitly flagged that the inverse
exponential produced occasional unphysical extremes (their Appendix C).
Wetherell (2026) used sqrt + linear rescale and reported the opposite failure
mode -- a *dry* bias in the far tail.  We therefore implement several and make
the choice a config knob, with ``log1p`` as the default and ``sqrt`` as the
recommended ablation.

All transforms map mm/day -> a roughly standardised real-valued field, and are
exactly invertible on ``[0, inf)``.  IMPORTANT: the same transform must be
applied to station observations before they enter the DA likelihood, because
the observation operator acts in transformed space.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # torch is optional for the pure-numpy preprocessing scripts
    import torch

    _Tensor = torch.Tensor
except Exception:  # pragma: no cover
    torch = None
    _Tensor = object


ArrayLike = "np.ndarray | _Tensor"


def _backend(x):
    if torch is not None and isinstance(x, torch.Tensor):
        return torch
    return np


@dataclass(frozen=True)
class PrecipTransform:
    """Invertible mm/day <-> model-space transform.

    Parameters
    ----------
    kind:
        ``"log1p"``  y = (log(1 + p/eps) - mu) / sd
        ``"sqrt"``   y = (sqrt(p) - mu) / sd
        ``"cbrt"``   y = (p ** (1/3) - mu) / sd
        ``"none"``   y = (p - mu) / sd
    eps:
        Only used by ``log1p``.  0.1 mm is roughly the BMD reporting
        resolution, so values below it are indistinguishable from zero.
    mu, sd:
        Standardisation constants computed on the TRAINING period only
        (see ``scripts/06_compute_stats.py``).
    """

    kind: str = "log1p"
    eps: float = 0.1
    mu: float = 0.0
    sd: float = 1.0

    def forward(self, p):
        xp = _backend(p)
        p = xp.clip(p, 0.0, None) if xp is np else p.clamp_min(0.0)
        if self.kind == "log1p":
            y = xp.log1p(p / self.eps)
        elif self.kind == "sqrt":
            y = xp.sqrt(p)
        elif self.kind == "cbrt":
            y = p ** (1.0 / 3.0)
        elif self.kind == "none":
            y = p
        else:
            raise ValueError(f"unknown transform {self.kind!r}")
        return (y - self.mu) / self.sd

    def inverse(self, y):
        xp = _backend(y)
        z = y * self.sd + self.mu
        if self.kind == "log1p":
            z = xp.clip(z, None, 30.0) if xp is np else z.clamp_max(30.0)
            p = self.eps * xp.expm1(z)
        elif self.kind == "sqrt":
            z = xp.clip(z, 0.0, None) if xp is np else z.clamp_min(0.0)
            p = z**2
        elif self.kind == "cbrt":
            p = z**3
        elif self.kind == "none":
            p = z
        else:
            raise ValueError(f"unknown transform {self.kind!r}")
        return xp.clip(p, 0.0, None) if xp is np else p.clamp_min(0.0)

    # -- convenience -------------------------------------------------------
    def fit(self, p_train) -> "PrecipTransform":
        """Return a copy with mu/sd estimated from a training sample."""
        raw = PrecipTransform(kind=self.kind, eps=self.eps, mu=0.0, sd=1.0)
        y = raw.forward(np.asarray(p_train, dtype=np.float64))
        y = y[np.isfinite(y)]
        return PrecipTransform(
            kind=self.kind, eps=self.eps, mu=float(y.mean()), sd=float(y.std() + 1e-12)
        )

    def to_dict(self) -> dict:
        return dict(kind=self.kind, eps=self.eps, mu=self.mu, sd=self.sd)

    @classmethod
    def from_dict(cls, d: dict) -> "PrecipTransform":
        return cls(**{k: d[k] for k in ("kind", "eps", "mu", "sd") if k in d})


def standardize(x, mu, sd):
    return (x - mu) / sd


def unstandardize(x, mu, sd):
    return x * sd + mu
