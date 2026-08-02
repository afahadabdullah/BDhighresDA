"""Scale-explicit verification for generative precipitation downscaling.

A single field score cannot answer the question this project actually poses.
Three claims live at three different scales, each needing its own null model,
and conflating them is the standard way a downscaling paper overstates itself.

===============================================================================
 Claim                        Null model                 What a win proves
===============================================================================
 A. DOWNSCALING GAIN          ``coarse_input``           The prior synthesises
    The prior adds real       the 0.5-deg conditioning   real structure between
    information below the     field on the fine grid.    the conditioning scale
    conditioning scale.       Zero variance below        and 0.05 deg.
                              its own resolution.

 B. SUB-FOOTPRINT GAIN        ``footprint_perfect``      Located structure that
    With 0.1-deg footprints   the TRUTH's own 0.1-deg    NO observation carries.
    assimilated, the          block mean, upsampled.     This is the strongest
    analysis is still right   Perfect coarse skill,      claim available and the
    BELOW footprint scale.    exactly zero subgrid.      hardest null to beat.

 C. TEXTURE REALISM           ``truth`` spectrum         Member variance is
    Members have the right                               right at each scale --
    variance per scale, not                              necessary but NOT
    just the right mean.                                 sufficient for A or B.
===============================================================================

Claim C is deliberately separated because a field can be simultaneously *too
rough* (member/truth power ratio far above one) and *too narrow* (ensemble
spread below error).  Those are different defects with opposite fixes, and a
scalar "calibration" verdict hides both.

Beating ``footprint_perfect`` is the load-bearing result: that null is handed
the exact 0.1-deg block means of the truth, so any positive skill against it is
information the observations could not have supplied.
"""
from __future__ import annotations

from dataclasses import dataclass, field as _field

import numpy as np

__all__ = [
    "block_mean",
    "upsample_blocks",
    "scale_decompose",
    "eligible_mask",
    "coarse_input_null",
    "footprint_perfect_null",
    "ScaleLadder",
    "scale_ladder",
    "rapsd",
    "spectral_summary",
    "effective_resolution",
    "fss_grid",
    "skillful_scale",
    "variogram",
    "fair_crps",
    "deterministic_scores",
]


# --------------------------------------------------------------------------
# Nested block algebra.  All routines are mask-aware: coastal blocks that are
# only partly land must not be silently averaged against zeros.
# --------------------------------------------------------------------------

def _block_sum(field: np.ndarray, factor: int) -> np.ndarray:
    height, width = field.shape[-2:]
    if factor < 1:
        raise ValueError("factor must be >= 1")
    if height % factor or width % factor:
        raise ValueError(
            f"trailing shape {field.shape[-2:]} is not divisible by factor {factor}"
        )
    shape = (*field.shape[:-2], height // factor, factor, width // factor, factor)
    return field.reshape(shape).sum(axis=(-3, -1))


def upsample_blocks(field: np.ndarray, factor: int) -> np.ndarray:
    """Repeat each coarse value into its nested fine cells."""
    if factor == 1:
        return field
    return np.repeat(np.repeat(field, factor, axis=-2), factor, axis=-1)


def block_mean(field: np.ndarray, factor: int, mask: np.ndarray) -> np.ndarray:
    """Mask-aware nested block mean over the trailing ``(H, W)`` axes."""
    if factor == 1:
        return np.where(np.broadcast_to(mask, field.shape), field, np.nan)
    valid = np.broadcast_to(mask, field.shape) & np.isfinite(field)
    sums = _block_sum(np.where(valid, field, 0.0), factor)
    counts = _block_sum(valid.astype(np.int32), factor)
    return np.divide(
        sums, counts,
        out=np.full(sums.shape, np.nan, dtype=np.float64),
        where=counts > 0,
    )


def scale_decompose(
    field: np.ndarray, factor: int, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Split a field into (block mean upsampled, residual below that scale)."""
    coarse = upsample_blocks(block_mean(field, factor, mask), factor)
    fine_mask = np.broadcast_to(mask, field.shape)
    residual = np.where(fine_mask, field - coarse, np.nan)
    return np.where(fine_mask, coarse, np.nan), residual


def eligible_mask(
    truth: np.ndarray, factor: int, minimum_valid_fraction: float = 1.0
) -> np.ndarray:
    """Fine cells inside sufficiently complete nested blocks.

    With the default of 1.0 a block must be entirely valid, which removes
    coastal footprints whose block mean would otherwise be computed from a
    handful of land cells and compared against a full-block model value.
    """
    if not 0.0 < minimum_valid_fraction <= 1.0:
        raise ValueError("minimum_valid_fraction must lie in (0, 1]")
    valid = np.isfinite(truth)
    if factor == 1:
        return valid
    counts = _block_sum(valid.astype(np.int32), factor)
    required = int(np.ceil(minimum_valid_fraction * factor * factor))
    return valid & upsample_blocks(counts >= required, factor)


# --------------------------------------------------------------------------
# Null models
# --------------------------------------------------------------------------

def coarse_input_null(
    coarse_field: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Claim A null: the conditioning field itself, as a one-member ensemble.

    ``coarse_field`` is the 0.5-deg precipitation analysis already mapped onto
    the fine grid (bilinear or block-constant -- either way it carries no
    genuine information at 0.05 deg).  Skill against this null is the honest
    definition of "what downscaling bought us".
    """
    out = np.where(np.broadcast_to(mask, coarse_field.shape), coarse_field, np.nan)
    return out[None] if out.ndim == coarse_field.ndim else out


def footprint_perfect_null(
    truth: np.ndarray, factor: int, mask: np.ndarray
) -> np.ndarray:
    """Claim B null: the truth's own block mean, upsampled, as one member.

    This null is *given* the correct 0.1-deg footprint means -- better satellite
    information than any real retrieval supplies -- and has exactly zero subgrid
    structure.  Positive skill against it cannot have come from the observations.
    """
    coarse, _ = scale_decompose(truth, factor, mask)
    return coarse[None]


# --------------------------------------------------------------------------
# Scores
# --------------------------------------------------------------------------

def _paired(ensemble: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if ensemble.ndim != truth.ndim + 1:
        raise ValueError(
            f"ensemble {ensemble.shape} needs one member axis before truth {truth.shape}"
        )
    ens = np.asarray(ensemble, dtype=np.float64).reshape(ensemble.shape[0], -1)
    obs = np.asarray(truth, dtype=np.float64).reshape(-1)
    keep = np.isfinite(obs) & np.all(np.isfinite(ens), axis=0)
    return ens[:, keep], obs[keep]


def fair_crps(ensemble: np.ndarray, truth: np.ndarray) -> float:
    """Fair (unbiased for finite ``m``) ensemble CRPS -- Ferro (2014).

    The plain estimator is optimistically biased for small ensembles, which
    matters here because ``m = 16``.
    """
    ens, obs = _paired(ensemble, truth)
    if obs.size == 0:
        return float("nan")
    members = ens.shape[0]
    skill = np.mean(np.abs(ens - obs[None]))
    if members == 1:
        return float(skill)
    ordered = np.sort(ens, axis=0)
    weights = (2 * np.arange(1, members + 1) - members - 1)[:, None]
    spread = np.sum(weights * ordered, axis=0) / (members * (members - 1))
    return float(skill - np.mean(spread))


def deterministic_scores(
    ensemble: np.ndarray, truth: np.ndarray
) -> dict[str, float | int]:
    """Deterministic, probabilistic, and per-member scores on one selection."""
    ens, obs = _paired(ensemble, truth)
    if obs.size == 0:
        return {}
    members = ens.shape[0]
    mean = ens.mean(axis=0)
    error = mean - obs

    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        if a.std() == 0 or b.std() == 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    energy = float(np.mean(obs**2))
    member_corr = np.array([_corr(m, obs) for m in ens])
    finite_corr = member_corr[np.isfinite(member_corr)]
    low, high = np.quantile(ens, [0.05, 0.95], axis=0)
    spread = float(np.mean(np.std(ens, axis=0, ddof=1))) if members > 1 else 0.0
    rmse = float(np.sqrt(np.mean(error**2)))
    return {
        "rmse_mm": rmse,
        "mae_mm": float(np.mean(np.abs(error))),
        "bias_mm": float(np.mean(error)),
        "crps_mm": fair_crps(ensemble, truth),
        "correlation": _corr(mean, obs),
        "spread_mm": spread,
        "spread_skill_ratio": float(spread / rmse) if rmse > 0 else float("nan"),
        "coverage_90": float(np.mean((obs >= low) & (obs <= high))),
        "ensemble_mean_energy_ratio": float(np.mean(mean**2) / energy)
        if energy > 0 else float("nan"),
        "mean_member_rmse_mm": float(np.mean(np.sqrt(np.mean((ens - obs[None]) ** 2, axis=1)))),
        "mean_member_correlation": float(np.mean(finite_corr))
        if finite_corr.size else float("nan"),
        "mean_member_energy_ratio": float(np.mean(np.mean(ens**2, axis=1)) / energy)
        if energy > 0 else float("nan"),
        "truth_rms_mm": float(np.sqrt(energy)),
        "n": int(obs.size),
    }


def skill_against(scores: dict, null: dict) -> dict[str, float]:
    """Fractional MSE and CRPS reduction relative to a null model."""
    out: dict[str, float] = {}
    null_mse = float(null.get("rmse_mm", np.nan)) ** 2
    mse = float(scores.get("rmse_mm", np.nan)) ** 2
    out["mse_skill"] = float(1.0 - mse / null_mse) if null_mse > 0 else float("nan")
    null_crps = float(null.get("crps_mm", np.nan))
    crps = float(scores.get("crps_mm", np.nan))
    out["crps_skill"] = float(1.0 - crps / null_crps) if null_crps > 0 else float("nan")
    return out


# --------------------------------------------------------------------------
# The scale ladder: where does skill actually live?
# --------------------------------------------------------------------------

@dataclass
class ScaleLadder:
    """Scores of the block-mean and residual components at each aggregation."""

    factors: list[int] = _field(default_factory=list)
    degrees: list[float] = _field(default_factory=list)
    aggregated: dict[str, list[dict]] = _field(default_factory=dict)
    residual: dict[str, list[dict]] = _field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "factors": self.factors,
            "degrees": self.degrees,
            "aggregated": self.aggregated,
            "residual": self.residual,
        }


def scale_ladder(
    candidates: dict[str, np.ndarray],
    truth: np.ndarray,
    mask: np.ndarray,
    factors: tuple[int, ...] = (1, 2, 4, 8),
    fine_degrees: float = 0.05,
) -> ScaleLadder:
    """Score every candidate at a ladder of aggregation scales.

    For each factor ``f`` the field is split into the block mean at ``f`` and
    the residual below it.  ``aggregated`` answers "how good is this at scale
    ``f`` and coarser"; ``residual`` answers "how good is this *below* scale
    ``f``".  Reading the two together shows exactly which scales carry skill --
    a downscaler that only improves the aggregated component has not downscaled
    anything, it has corrected a bias.
    """
    ladder = ScaleLadder(
        factors=list(factors),
        degrees=[float(f * fine_degrees) for f in factors],
        aggregated={name: [] for name in candidates},
        residual={name: [] for name in candidates},
    )
    for factor in factors:
        valid = eligible_mask(truth, factor, 1.0) & mask
        truth_coarse, truth_residual = scale_decompose(truth, factor, valid)
        for name, ensemble in candidates.items():
            member_mask = np.broadcast_to(valid, ensemble.shape[1:])
            coarse_members = []
            residual_members = []
            for member in ensemble:
                c, r = scale_decompose(member, factor, member_mask)
                coarse_members.append(c)
                residual_members.append(r)
            ladder.aggregated[name].append(
                deterministic_scores(np.stack(coarse_members), truth_coarse)
            )
            ladder.residual[name].append(
                deterministic_scores(np.stack(residual_members), truth_residual)
            )
    return ladder


# --------------------------------------------------------------------------
# Spectra and effective resolution
# --------------------------------------------------------------------------

def _prepare_for_fft(field: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """Demean over valid cells, zero-fill gaps, apply a separable Hann taper.

    Gap filling with zeros after demeaning injects a small amount of spurious
    high-wavenumber power along the coastline.  We accept it (and say so) rather
    than interpolating, because interpolation would *suppress* exactly the
    high-wavenumber power the diagnostic exists to measure -- a bias in the
    direction that would flatter the model.
    """
    valid = mask & np.isfinite(field)
    if valid.sum() < 16:
        return None
    work = np.zeros(field.shape, dtype=np.float64)
    work[valid] = field[valid] - field[valid].mean()
    rows = np.hanning(field.shape[0])[:, None]
    cols = np.hanning(field.shape[1])[None, :]
    return work * rows * cols


def rapsd(
    field: np.ndarray,
    mask: np.ndarray,
    fine_degrees: float = 0.05,
    n_bins: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Radially averaged power spectral density.

    Returns ``(wavelength_km, power)`` with wavelength descending in index
    order, i.e. large scales first.  Wavelength is reported in kilometres using
    a nominal 111 km per degree, which is adequate at Bangladesh latitudes for
    a diagnostic whose purpose is comparison between fields on one grid.
    """
    prepared = _prepare_for_fft(field, mask)
    if prepared is None:
        return np.array([]), np.array([])
    height, width = prepared.shape
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(prepared))) ** 2
    ky = np.fft.fftshift(np.fft.fftfreq(height))[:, None]
    kx = np.fft.fftshift(np.fft.fftfreq(width))[None, :]
    radius = np.sqrt(ky**2 + kx**2)
    limit = 0.5
    bins = n_bins or min(height, width) // 2
    edges = np.linspace(1.0 / max(height, width), limit, bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    power = np.full(bins, np.nan)
    index = np.digitize(radius.ravel(), edges) - 1
    flat = spectrum.ravel()
    for b in range(bins):
        selected = flat[index == b]
        if selected.size:
            power[b] = selected.mean()
    wavelength_km = fine_degrees * 111.0 / centres
    keep = np.isfinite(power) & (power > 0)
    return wavelength_km[keep][::-1], power[keep][::-1]


def spectral_summary(
    fields: dict[str, np.ndarray],
    truth: np.ndarray,
    mask: np.ndarray,
    fine_degrees: float = 0.05,
) -> dict:
    """Day-averaged RAPSD for each named field plus its ratio to the truth.

    ``fields`` maps a label to an array of shape ``(days, H, W)``; pass a single
    representative member, not the ensemble mean, when the question is texture
    realism.  Averaging members before the FFT removes precisely the
    high-wavenumber variance being tested.
    """
    reference_wavelength: np.ndarray | None = None
    accumulated: dict[str, list[np.ndarray]] = {name: [] for name in fields}
    truth_power: list[np.ndarray] = []
    for day in range(truth.shape[0]):
        wavelength, power = rapsd(truth[day], mask[day] if mask.ndim == 3 else mask,
                                  fine_degrees)
        if wavelength.size == 0:
            continue
        if reference_wavelength is None:
            reference_wavelength = wavelength
        elif wavelength.shape != reference_wavelength.shape:
            continue
        truth_power.append(power)
        for name, stack in fields.items():
            _, p = rapsd(stack[day], mask[day] if mask.ndim == 3 else mask, fine_degrees)
            accumulated[name].append(p if p.shape == power.shape else np.full_like(power, np.nan))
    if reference_wavelength is None or not truth_power:
        return {}
    mean_truth = np.nanmean(np.stack(truth_power), axis=0)
    out = {
        "wavelength_km": reference_wavelength.tolist(),
        "truth_power": mean_truth.tolist(),
        "power": {},
        "power_ratio": {},
        "effective_resolution_km": {},
    }
    for name, stacks in accumulated.items():
        if not stacks:
            continue
        mean_power = np.nanmean(np.stack(stacks), axis=0)
        ratio = np.divide(
            mean_power, mean_truth,
            out=np.full_like(mean_power, np.nan), where=mean_truth > 0,
        )
        out["power"][name] = mean_power.tolist()
        out["power_ratio"][name] = ratio.tolist()
        out["effective_resolution_km"][name] = effective_resolution(
            reference_wavelength, ratio
        )
    return out


def effective_resolution(
    wavelength_km: np.ndarray,
    power_ratio: np.ndarray,
    tolerance: float = 2.0,
) -> float:
    """Shortest wavelength down to which the spectrum stays within tolerance.

    Walking from large scales toward small, this is the last wavelength before
    the model/truth power ratio first leaves ``[1/tolerance, tolerance]``.  It
    is the honest resolution of the product: features smaller than this are
    either damped away or invented.
    """
    wavelength = np.asarray(wavelength_km, dtype=float)
    ratio = np.asarray(power_ratio, dtype=float)
    order = np.argsort(-wavelength)
    wavelength, ratio = wavelength[order], ratio[order]
    low, high = 1.0 / tolerance, tolerance
    previous = float(wavelength[0]) if wavelength.size else float("nan")
    for lam, value in zip(wavelength, ratio):
        if not np.isfinite(value):
            continue
        if value < low or value > high:
            return previous
        previous = float(lam)
    return previous


# --------------------------------------------------------------------------
# Neighbourhood verification
# --------------------------------------------------------------------------

def _fraction(binary: np.ndarray, window: int) -> np.ndarray:
    """Neighbourhood exceedance fraction via a summed-area table."""
    padded = np.pad(binary.astype(np.float64), window, mode="constant")
    integral = padded.cumsum(axis=0).cumsum(axis=1)
    integral = np.pad(integral, ((1, 0), (1, 0)), mode="constant")
    height, width = binary.shape
    half = window // 2
    rows = np.arange(height) + window
    cols = np.arange(width) + window
    r0, r1 = rows - half, rows + half + 1
    c0, c1 = cols - half, cols + half + 1
    total = (
        integral[np.ix_(r1, c1)] - integral[np.ix_(r0, c1)]
        - integral[np.ix_(r1, c0)] + integral[np.ix_(r0, c0)]
    )
    return total / float(window * window)


def fss_grid(
    forecast: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
    thresholds: tuple[float, ...] = (1.0, 5.0, 10.0, 25.0, 50.0),
    windows: tuple[int, ...] = (1, 3, 5, 9, 17, 33),
) -> dict:
    """Fractions skill score over a threshold x neighbourhood grid.

    Also returns the uniform-skill reference ``0.5 + f0/2`` used to define the
    skillful scale, where ``f0`` is the domain exceedance base rate.
    """
    result: dict[str, dict] = {"thresholds": list(thresholds),
                               "windows": list(windows),
                               "fss": {}, "target": {}, "skillful_scale": {}}
    days = truth.shape[0]
    for threshold in thresholds:
        scores = np.zeros(len(windows))
        weights = np.zeros(len(windows))
        base_rates = []
        for day in range(days):
            valid = mask[day] if mask.ndim == 3 else mask
            obs = np.where(valid, truth[day], np.nan)
            pred = np.where(valid, forecast[day], np.nan)
            ok = np.isfinite(obs) & np.isfinite(pred)
            if ok.sum() < 16:
                continue
            obs_bin = np.where(ok, obs >= threshold, 0)
            pred_bin = np.where(ok, pred >= threshold, 0)
            if obs_bin.sum() == 0 and pred_bin.sum() == 0:
                continue
            base_rates.append(float(obs_bin.sum() / ok.sum()))
            for w, window in enumerate(windows):
                po = _fraction(obs_bin, window)[ok]
                pf = _fraction(pred_bin, window)[ok]
                denominator = np.mean(po**2) + np.mean(pf**2)
                if denominator <= 0:
                    continue
                scores[w] += 1.0 - np.mean((pf - po) ** 2) / denominator
                weights[w] += 1.0
        curve = np.divide(scores, weights, out=np.full_like(scores, np.nan),
                          where=weights > 0)
        key = f"{threshold:g}"
        f0 = float(np.mean(base_rates)) if base_rates else float("nan")
        target = 0.5 + f0 / 2.0
        result["fss"][key] = curve.tolist()
        result["target"][key] = target
        result["skillful_scale"][key] = skillful_scale(windows, curve, target)
    return result


def skillful_scale(
    windows: tuple[int, ...] | list[int],
    curve: np.ndarray | list[float],
    target: float,
) -> float:
    """Smallest neighbourhood width (in fine cells) at which FSS clears target."""
    for window, value in zip(windows, np.asarray(curve, dtype=float)):
        if np.isfinite(value) and value >= target:
            return float(window)
    return float("nan")


# --------------------------------------------------------------------------
# Spatial structure
# --------------------------------------------------------------------------

def variogram(
    field: np.ndarray,
    mask: np.ndarray,
    lags: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
) -> dict[str, list[float]]:
    """Isotropic semivariance at axis-aligned lags, averaged over days.

    A downscaler that produces the right total variance but the wrong
    correlation length is caught here and nowhere else.
    """
    semivariance = []
    for lag in lags:
        values = []
        for day in range(field.shape[0]):
            valid = mask[day] if mask.ndim == 3 else mask
            layer = np.where(valid, field[day], np.nan)
            for shifted in (
                layer[lag:, :] - layer[:-lag, :],
                layer[:, lag:] - layer[:, :-lag],
            ):
                finite = shifted[np.isfinite(shifted)]
                if finite.size:
                    values.append(0.5 * np.mean(finite**2))
        semivariance.append(float(np.mean(values)) if values else float("nan"))
    return {"lags": list(lags), "semivariance": semivariance}
