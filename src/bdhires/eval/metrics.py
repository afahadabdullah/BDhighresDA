"""Verification metrics.

Point metrics follow Manshausen et al. Appendix D (RMSE/MAE/CRPS/spread-skill
/rank histogram, all evaluated at LEFT-OUT stations); spatial metrics follow
the precipitation-downscaling literature (FSS, SAL, categorical scores).
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------- point scores
def rmse(pred: np.ndarray, obs: np.ndarray) -> float:
    m = np.isfinite(pred) & np.isfinite(obs)
    return float(np.sqrt(np.mean((pred[m] - obs[m]) ** 2)))


def mae(pred: np.ndarray, obs: np.ndarray) -> float:
    m = np.isfinite(pred) & np.isfinite(obs)
    return float(np.mean(np.abs(pred[m] - obs[m])))


def bias(pred: np.ndarray, obs: np.ndarray) -> float:
    m = np.isfinite(pred) & np.isfinite(obs)
    return float(np.mean(pred[m] - obs[m]))


def crps_ensemble(ens: np.ndarray, obs: np.ndarray) -> float:
    """Fair (unbiased) CRPS, Zamo & Naveau (2018) eq. eFAIR.

    ``ens``: (R, ...) ensemble members; ``obs``: (...) observations.
    """
    R = ens.shape[0]
    obs_b = np.broadcast_to(obs, ens.shape[1:])
    valid = np.isfinite(obs_b) & np.all(np.isfinite(ens), axis=0)
    if not valid.any():
        return np.nan
    e = ens[:, valid]
    o = obs_b[valid]
    term1 = np.abs(e - o[None]).mean(axis=0)
    term2 = np.abs(e[:, None] - e[None, :]).sum(axis=(0, 1)) / (2 * R * (R - 1))
    return float(np.mean(term1 - term2))


def spread_skill(ens: np.ndarray, obs: np.ndarray) -> tuple[float, float]:
    """Return ``(rmse_of_ensemble_mean, bias_corrected_spread)``.

    A calibrated ensemble has spread ~= RMSE (Fortin et al. 2014); the SDA
    paper found their ensembles under-dispersive, so always report both.
    """
    R = ens.shape[0]
    mean = ens.mean(axis=0)
    var = ens.var(axis=0, ddof=1)
    obs_b = np.broadcast_to(obs, mean.shape)
    m = np.isfinite(obs_b) & np.isfinite(mean)
    sk = float(np.sqrt(np.mean((mean[m] - obs_b[m]) ** 2)))
    sp = float(np.sqrt(np.mean(var[m]) * (R + 1) / R))
    return sk, sp


def rank_histogram(ens: np.ndarray, obs: np.ndarray, n_bins: int | None = None) -> np.ndarray:
    R = ens.shape[0]
    n_bins = n_bins or (R + 1)
    obs_b = np.broadcast_to(obs, ens.shape[1:])
    m = np.isfinite(obs_b) & np.all(np.isfinite(ens), axis=0)
    ranks = (ens[:, m] < obs_b[m][None]).sum(axis=0)
    return np.bincount(ranks, minlength=R + 1)[: R + 1].astype(float)


# -------------------------------------------------------------- spatial scores
def fss(pred: np.ndarray, obs: np.ndarray, threshold: float, window: int) -> float:
    """Fractions Skill Score (Roberts & Lean 2008) for a single 2-D field pair."""
    from scipy.ndimage import uniform_filter

    pb = (pred >= threshold).astype(float)
    ob = (obs >= threshold).astype(float)
    pf = uniform_filter(pb, size=window, mode="constant")
    of = uniform_filter(ob, size=window, mode="constant")
    num = np.nanmean((pf - of) ** 2)
    den = np.nanmean(pf**2) + np.nanmean(of**2)
    return float(1.0 - num / den) if den > 0 else np.nan


def fss_series(pred: np.ndarray, obs: np.ndarray, thresholds, windows) -> dict:
    """Aggregate FSS over time (accumulate numerator/denominator, not the ratio)."""
    from scipy.ndimage import uniform_filter

    out = {}
    for thr in thresholds:
        for win in windows:
            num = den = 0.0
            for t in range(pred.shape[0]):
                pf = uniform_filter((pred[t] >= thr).astype(float), size=win, mode="constant")
                of = uniform_filter((obs[t] >= thr).astype(float), size=win, mode="constant")
                num += np.nansum((pf - of) ** 2)
                den += np.nansum(pf**2) + np.nansum(of**2)
            out[(thr, win)] = float(1 - num / den) if den > 0 else np.nan
    return out


def categorical(pred: np.ndarray, obs: np.ndarray, threshold: float) -> dict:
    m = np.isfinite(pred) & np.isfinite(obs)
    p, o = pred[m] >= threshold, obs[m] >= threshold
    hits = float((p & o).sum())
    fa = float((p & ~o).sum())
    miss = float((~p & o).sum())
    cn = float((~p & ~o).sum())
    n = hits + fa + miss + cn
    pod = hits / (hits + miss) if hits + miss else np.nan
    far = fa / (hits + fa) if hits + fa else np.nan
    csi = hits / (hits + fa + miss) if hits + fa + miss else np.nan
    hits_rand = (hits + miss) * (hits + fa) / n if n else np.nan
    ets = (hits - hits_rand) / (hits + fa + miss - hits_rand) if (hits + fa + miss - hits_rand) else np.nan
    return dict(POD=pod, FAR=far, CSI=csi, ETS=ets, bias=(hits + fa) / (hits + miss) if hits + miss else np.nan)


def sal(pred: np.ndarray, obs: np.ndarray, thr_factor: float = 1 / 15):
    """Structure-Amplitude-Location (Wernli et al. 2008) via pysteps if available."""
    try:
        from pysteps.verification.salscores import sal as _sal
    except Exception:  # pragma: no cover
        return dict(S=np.nan, A=np.nan, L=np.nan)
    s, a, l = _sal(pred, obs, thr_factor=thr_factor)
    return dict(S=float(s), A=float(a), L=float(l))


def summarize(pred: np.ndarray, obs: np.ndarray, thresholds=(1, 10, 20, 50, 100)) -> dict:
    out = dict(rmse=rmse(pred, obs), mae=mae(pred, obs), bias=bias(pred, obs))
    for thr in thresholds:
        for k, v in categorical(pred, obs, thr).items():
            out[f"{k}@{thr}"] = v
    return out
