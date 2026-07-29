"""Ensemble calibration: diagnostics and post-hoc correction.

Every published generative-DA study we are aware of reports under-dispersive
ensembles -- Manshausen et al. (2025) say so explicitly for their km-scale SDA
system, and attribute it partly to the mode-seeking Gaussian approximation in
the guidance term.  Assume this project will hit the same problem and measure
it from day one.

Order of attack (see docs/METHODOLOGY.md Section 6):

1. Fix it *by design*: perturbed observations, SDE sampling, ERA5-EDA
   conditioning, correctly-sized R.  These change the sampler, not the numbers.
2. Only then, if a residual deficit remains, apply the post-hoc corrections in
   this module -- and report both the raw and calibrated scores.

The spread-skill relation used throughout (Fortin et al. 2014):

    MSE(ensemble mean, obs)  ~=  spread^2 * (R+1)/R  +  sigma_obs^2

so a fair comparison against point gauges must include the observation-error
variance on the spread side.  Forgetting it makes a perfectly calibrated
ensemble look under-dispersive.
"""

from __future__ import annotations

import numpy as np


def spread_skill(ens: np.ndarray, obs: np.ndarray, obs_var: float = 0.0) -> dict:
    """``ens`` is (R, ...) and ``obs`` broadcasts to ``ens.shape[1:]``."""
    R = ens.shape[0]
    mean = ens.mean(axis=0)
    var = ens.var(axis=0, ddof=1)
    obs_b = np.broadcast_to(obs, mean.shape)
    m = np.isfinite(obs_b) & np.isfinite(mean)
    if not m.any():
        return dict(skill=np.nan, spread=np.nan, ratio=np.nan, n=0)
    skill = float(np.sqrt(np.mean((mean[m] - obs_b[m]) ** 2)))
    spread = float(np.sqrt(np.mean(var[m]) * (R + 1) / R + obs_var))
    return dict(skill=skill, spread=spread, ratio=spread / skill if skill else np.nan,
                n=int(m.sum()))


def spread_skill_by_bin(
    ens: np.ndarray,
    obs: np.ndarray,
    bins=(0, 1, 10, 25, 50, 100, 1e9),
    obs_var: float = 0.0,
) -> list[dict]:
    """Spread/skill stratified by observed intensity.

    Under-dispersion is almost never uniform: generative priors are usually
    acceptable for light rain and badly under-dispersive for extremes, which is
    precisely the regime a flood application cares about.  A single
    domain-averaged ratio hides this.
    """
    obs_b = np.broadcast_to(obs, ens.shape[1:])
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = np.isfinite(obs_b) & (obs_b >= lo) & (obs_b < hi)
        if m.sum() < 20:
            continue
        r = spread_skill(ens[:, m], obs_b[m], obs_var=obs_var)
        out.append(dict(lo=float(lo), hi=float(hi), **r))
    return out


def rank_histogram(ens: np.ndarray, obs: np.ndarray, obs_sd: float = 0.0,
                   seed: int = 0) -> np.ndarray:
    """Rank histogram with observation error added to the members.

    Following Manshausen et al. Appendix D: if the observations carry error,
    the members must be perturbed by the same error model before ranking, or
    the histogram is U-shaped by construction.
    """
    rng = np.random.default_rng(seed)
    e = ens + (rng.normal(0, obs_sd, ens.shape) if obs_sd > 0 else 0.0)
    obs_b = np.broadcast_to(obs, ens.shape[1:])
    m = np.isfinite(obs_b) & np.all(np.isfinite(e), axis=0)
    ranks = (e[:, m] < obs_b[m][None]).sum(axis=0)
    return np.bincount(ranks, minlength=ens.shape[0] + 1).astype(float)


def rank_histogram_deviation(hist: np.ndarray) -> float:
    """Scalar summary of rank-histogram flatness. 0 = flat, larger = worse.

    Under-dispersion gives a U shape; over-dispersion an inverted U; bias a
    slope. Reporting one number alongside the plot makes tuning tractable.
    """
    p = hist / hist.sum()
    q = np.full_like(p, 1.0 / len(p))
    return float(np.sum(np.abs(p - q)) / 2.0)


def fit_inflation(ens: np.ndarray, obs: np.ndarray, obs_var: float = 0.0,
                  bounds=(0.5, 6.0)) -> float:
    """Multiplicative inflation factor that makes spread match skill.

    alpha such that  spread^2 * alpha^2 + obs_var = MSE(mean, obs).
    A last-resort correction: it fixes the second moment and nothing else, so
    it will not repair a bad rank histogram shape.  Always report the
    uninflated numbers too.
    """
    R = ens.shape[0]
    mean = ens.mean(axis=0)
    var = ens.var(axis=0, ddof=1) * (R + 1) / R
    obs_b = np.broadcast_to(obs, mean.shape)
    m = np.isfinite(obs_b) & np.isfinite(mean)
    mse = float(np.mean((mean[m] - obs_b[m]) ** 2))
    sp2 = float(np.mean(var[m]))
    if sp2 <= 0:
        return 1.0
    alpha = np.sqrt(max(mse - obs_var, 1e-9) / sp2)
    return float(np.clip(alpha, *bounds))


def apply_inflation(ens: np.ndarray, alpha: float, floor: float = 0.0) -> np.ndarray:
    """x_r <- mean + alpha * (x_r - mean), clipped at ``floor`` (0 for rainfall)."""
    mean = ens.mean(axis=0, keepdims=True)
    return np.clip(mean + alpha * (ens - mean), floor, None)


def fit_quantile_recalibration(ens: np.ndarray, obs: np.ndarray, n_q: int = 21):
    """Rank-based recalibration map (Ben Bouallegue-style EMOS-lite).

    Returns a monotone map from nominal to empirical quantile level, estimated
    on a validation period, that can be applied to future ensembles.  Unlike
    variance inflation this fixes the whole distribution shape, including the
    tails, at the cost of needing a decent validation sample.
    """
    R = ens.shape[0]
    obs_b = np.broadcast_to(obs, ens.shape[1:])
    m = np.isfinite(obs_b) & np.all(np.isfinite(ens), axis=0)
    ranks = (ens[:, m] < obs_b[m][None]).sum(axis=0) / R
    nominal = np.linspace(0, 1, n_q)
    empirical = np.quantile(ranks, nominal)
    return dict(nominal=nominal.tolist(), empirical=empirical.tolist())


def apply_quantile_recalibration(ens: np.ndarray, cal: dict) -> np.ndarray:
    """Re-map ensemble quantile levels using a map from ``fit_quantile_recalibration``."""
    R = ens.shape[0]
    srt = np.sort(ens, axis=0)
    lev = (np.arange(R) + 0.5) / R
    new = np.interp(lev, np.asarray(cal["empirical"]), np.asarray(cal["nominal"]))
    idx = np.clip((new * R - 0.5).round().astype(int), 0, R - 1)
    return srt[idx]


def calibration_report(ens: np.ndarray, obs: np.ndarray, obs_sd: float = 0.0) -> dict:
    """One call that produces everything needed for the calibration figure."""
    ov = obs_sd**2
    hist = rank_histogram(ens, obs, obs_sd=obs_sd)
    return dict(
        overall=spread_skill(ens, obs, obs_var=ov),
        by_intensity=spread_skill_by_bin(ens, obs, obs_var=ov),
        rank_hist=hist.tolist(),
        rank_hist_deviation=rank_histogram_deviation(hist),
        suggested_inflation=fit_inflation(ens, obs, obs_var=ov),
    )
