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


# ---------------------------------------------------------------------------
# Conditioning-channel transforms
# ---------------------------------------------------------------------------
#
# The ERA5 predictors are stored in the Zarr in raw physical units, which is the
# right thing for a data store.  They must NOT be handed to the network that way.
#
# Daily ERA5 ``tp`` has a skewness of roughly 10: a global z-score puts ~95% of
# days in a sliver just below zero and turns monsoon days into +20 sigma
# outliers.  A single input convolution cannot extract a usable signal from that,
# so the network falls back on climatology -- which is exactly the failure mode
# seen in the epoch-119 diagnostics (docs/DIAGNOSIS_epoch119.md, item 1).
#
# The fix is to give the *predictor* the same treatment as the target: compress
# the tail first, standardise second.  ``cape`` is likewise strongly right-skewed
# and gets a square root.  Everything else is near-Gaussian and passes through.
#
# This is applied at load time rather than at pack time, so changing it costs a
# ``06_compute_stats.py`` rerun rather than a full multi-decade ERA5 repack.  The
# chosen spec is written into ``stats.json`` so training and inference cannot
# silently disagree about it.

DEFAULT_COND_TRANSFORMS: dict[str, str] = {
    "era5_tp": "log1p",
    "era5_cape": "sqrt",
}


@dataclass(frozen=True)
class CondTransform:
    """Per-channel variance-stabilising transform for the ERA5 predictors.

    ``kinds`` is one transform name per conditioning channel, in channel order,
    drawn from the same vocabulary as :class:`PrecipTransform` (``log1p``,
    ``sqrt``, ``cbrt``, ``none``).  Applied *before* standardisation.

    Unlike :class:`PrecipTransform` this carries no ``mu``/``sd``: standardisation
    of the conditioning stack stays where it already lives, in ``cond_mean`` and
    ``cond_std``.  Those constants must therefore be recomputed whenever ``kinds``
    changes -- :func:`from_stats` enforces the pairing by reading both from the
    same ``stats.json``.
    """

    kinds: tuple[str, ...] = ()
    eps: float = 0.1

    @classmethod
    def for_channels(
        cls,
        channels: "list[str] | tuple[str, ...]",
        spec: "dict[str, str] | None" = None,
        eps: float = 0.1,
    ) -> "CondTransform":
        """Build a transform for named channels using ``spec`` (default: module default)."""
        spec = DEFAULT_COND_TRANSFORMS if spec is None else spec
        return cls(kinds=tuple(spec.get(name, "none") for name in channels), eps=eps)

    @classmethod
    def identity(cls, n_channels: int) -> "CondTransform":
        return cls(kinds=("none",) * n_channels)

    def forward(self, cond, channel_axis: int = -3):
        """Transform ``cond`` in place-safe fashion along ``channel_axis``.

        Accepts ``(C, H, W)`` or ``(N, C, H, W)``; the default ``channel_axis``
        of -3 covers both.
        """
        if not self.kinds:
            return cond
        xp = _backend(cond)
        n = cond.shape[channel_axis]
        if n != len(self.kinds):
            raise ValueError(
                f"CondTransform has {len(self.kinds)} channels but the array "
                f"has {n} along axis {channel_axis}"
            )
        if all(kind == "none" for kind in self.kinds):
            return cond

        axis = channel_axis % cond.ndim
        out = xp.moveaxis(cond, axis, 0) if xp is np else cond.movedim(axis, 0)
        out = out.copy() if xp is np else out.clone()
        for i, kind in enumerate(self.kinds):
            if kind == "none":
                continue
            channel = out[i]
            channel = (
                xp.clip(channel, 0.0, None) if xp is np else channel.clamp_min(0.0)
            )
            if kind == "log1p":
                out[i] = xp.log1p(channel / self.eps)
            elif kind == "sqrt":
                out[i] = xp.sqrt(channel)
            elif kind == "cbrt":
                out[i] = channel ** (1.0 / 3.0)
            else:
                raise ValueError(f"unknown conditioning transform {kind!r}")
        return xp.moveaxis(out, 0, axis) if xp is np else out.movedim(0, axis)

    def forward_channel(self, values, channel: int):
        """Transform a single channel, given as a bare array of any shape.

        :meth:`forward` needs a full channel-stacked array; diagnostics and
        per-variable plots hold one channel at a time.
        """
        if not self.kinds:
            return values
        kind = self.kinds[channel]
        if kind == "none":
            return values
        xp = _backend(values)
        z = xp.clip(values, 0.0, None) if xp is np else values.clamp_min(0.0)
        if kind == "log1p":
            return xp.log1p(z / self.eps)
        if kind == "sqrt":
            return xp.sqrt(z)
        if kind == "cbrt":
            return z ** (1.0 / 3.0)
        raise ValueError(f"unknown conditioning transform {kind!r}")

    def inverse_channel(self, values, channel: int):
        """Invert a single channel back to physical units.

        Needed by the diagnostics: once ``era5_tp`` is stored transformed, undoing
        only the standardisation leaves log-space numbers that would be plotted
        and scored as if they were mm.
        """
        if not self.kinds:
            return values
        kind = self.kinds[channel]
        if kind == "none":
            return values
        xp = _backend(values)
        if kind == "log1p":
            z = xp.clip(values, None, 30.0) if xp is np else values.clamp_max(30.0)
            out = self.eps * xp.expm1(z)
        elif kind == "sqrt":
            z = xp.clip(values, 0.0, None) if xp is np else values.clamp_min(0.0)
            out = z**2
        elif kind == "cbrt":
            out = values**3
        else:
            raise ValueError(f"unknown conditioning transform {kind!r}")
        return xp.clip(out, 0.0, None) if xp is np else out.clamp_min(0.0)

    def to_dict(self) -> dict:
        return dict(kinds=list(self.kinds), eps=self.eps)

    @classmethod
    def from_dict(cls, d: dict) -> "CondTransform":
        return cls(kinds=tuple(d["kinds"]), eps=float(d.get("eps", 0.1)))

    @classmethod
    def from_stats(cls, stats: dict) -> "CondTransform":
        """Read the transform recorded in ``stats.json``.

        Falls back to the identity for statistics files written before
        conditioning transforms existed, so old checkpoints stay reproducible.
        """
        if "cond_transform" in stats:
            return cls.from_dict(stats["cond_transform"])
        return cls.identity(len(stats["cond_mean"]))
