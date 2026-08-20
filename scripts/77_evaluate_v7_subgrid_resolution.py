#!/usr/bin/env python3
"""Test whether a V7 analysis contains *located* skill below 0.1 degrees.

Fine-looking texture is not evidence of resolution.  This evaluator removes
the exact 0.1-degree block mean from V7 and CHIRPS before scoring, then asks
whether the remaining 2x2-cell pattern beats a forecast with no subgrid
information.  A within-block permutation test retains V7's texture amplitude
but destroys its placement, separating useful structure from plausible noise.

CHIRPS is V7 stage B's training target, so its gridded comparison is an
out-of-sample product-agreement test, not independent truth.  When a station
dump is supplied, independently withheld BMD anomalies provide a second check.
Even then, ten selected days are exploratory rather than confirmatory.

Finally, a 0.05-degree grid has a Nyquist wavelength near 0.1 degrees.  The
phrase "0.05-degree resolved" below therefore means skill in the four values
inside a 0.1-degree cell, not literal skill at a 0.05-degree wavelength.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bdhires.eval import scale as S  # noqa: E402


FINE_DEGREES = 0.05
FOOTPRINT_FACTOR = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--map-dump", required=True)
    parser.add_argument("--subgrid-archive", required=True,
                        help="V7 target Zarr containing fine_mm (CHIRPS)")
    parser.add_argument("--station-dump", default=None,
                        help="optional V7 station NPZ for withheld-BMD anomalies")
    parser.add_argument("--arm", default="da_sim_r81")
    parser.add_argument("--comparators", default="background,da_meso",
                        help="comma-separated V7 arms included beside --arm")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--bootstrap-block-days", type=int, default=3,
                        help="circular temporal block length for the skill interval")
    parser.add_argument("--seed", type=int, default=20220503)
    return parser.parse_args()


def _finite(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.floating, float)):
        return _finite(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def correlation(first: np.ndarray, second: np.ndarray) -> float:
    first, second = np.asarray(first, float).ravel(), np.asarray(second, float).ravel()
    keep = np.isfinite(first) & np.isfinite(second)
    if keep.sum() < 3 or first[keep].std() == 0 or second[keep].std() == 0:
        return float("nan")
    return float(np.corrcoef(first[keep], second[keep])[0, 1])


def anomaly_metrics(predicted: np.ndarray, observed: np.ndarray) -> dict:
    predicted, observed = np.asarray(predicted, float), np.asarray(observed, float)
    keep = np.isfinite(predicted) & np.isfinite(observed)
    if not keep.any():
        return {"n": 0}
    predicted, observed = predicted[keep], observed[keep]
    error = predicted - observed
    null_mse = float(np.mean(observed**2))
    model_mse = float(np.mean(error**2))
    active = np.abs(observed) >= 1.0
    return {
        "n": int(len(observed)),
        "correlation": correlation(predicted, observed),
        "rmse_mm": float(np.sqrt(model_mse)),
        "bias_mm": float(np.mean(error)),
        "mse_skill_vs_no_subgrid": (
            float(1.0 - model_mse / null_mse) if null_mse > 0 else float("nan")
        ),
        "variance_ratio": (
            float(np.var(predicted) / np.var(observed))
            if np.var(observed) > 0 else float("nan")
        ),
        "sign_agreement_active": (
            float(np.mean(np.sign(predicted[active]) == np.sign(observed[active])))
            if active.any() else float("nan")
        ),
    }


def strict_mask(truth: np.ndarray, valid: np.ndarray, factor: int) -> np.ndarray:
    return S.eligible_mask(np.where(valid, truth, np.nan), factor, 1.0) & valid


def residuals(field: np.ndarray, valid: np.ndarray, factor: int) -> np.ndarray:
    return S.scale_decompose(field, factor, valid)[1]


def daily_residual_skill(
    candidate: np.ndarray, truth: np.ndarray, valid: np.ndarray, factor: int
) -> np.ndarray:
    output = np.full(len(truth), np.nan)
    for day in range(len(truth)):
        keep = strict_mask(truth[day], valid[day], factor)
        truth_residual = residuals(truth[day], keep, factor)
        candidate_residual = residuals(candidate[day], keep, factor)
        finite = keep & np.isfinite(truth_residual) & np.isfinite(candidate_residual)
        denominator = np.mean(truth_residual[finite] ** 2) if finite.any() else 0.0
        if denominator > 0:
            output[day] = 1.0 - np.mean(
                (candidate_residual[finite] - truth_residual[finite]) ** 2
            ) / denominator
    return output


def day_bootstrap(
    values: np.ndarray, resamples: int, seed: int, block_days: int = 3
) -> dict:
    """Circular day-block bootstrap; grid cells are never treated as replicates."""
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"n_days": 0, "mean": None, "ci_low": None, "ci_high": None}
    rng = np.random.default_rng(seed)
    width = max(1, min(int(block_days), len(values)))
    block_count = int(np.ceil(len(values) / width))
    starts = rng.integers(0, len(values), size=(resamples, block_count))
    index = (
        starts[:, :, None] + np.arange(width)[None, None, :]
    ).reshape(resamples, -1)[:, :len(values)] % len(values)
    estimates = values[index].mean(axis=1)
    low, high = np.percentile(estimates, [2.5, 97.5])
    return {
        "n_days": int(len(values)),
        "mean": float(values.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "resamples": int(resamples),
        "block_days": int(width),
    }


def permute_within_blocks(
    field: np.ndarray, factor: int, rng: np.random.Generator
) -> np.ndarray:
    """Independently permute fine-cell locations inside every coarse block."""
    time, height, width = field.shape
    blocks = field.reshape(time, height // factor, factor, width // factor, factor)
    blocks = blocks.transpose(0, 1, 3, 2, 4).reshape(
        time, height // factor, width // factor, factor * factor
    )
    order = np.argsort(rng.random(blocks.shape), axis=-1)
    shuffled = np.take_along_axis(blocks, order, axis=-1)
    return shuffled.reshape(
        time, height // factor, width // factor, factor, factor
    ).transpose(0, 1, 3, 2, 4).reshape(field.shape)


def placement_permutation_test(
    candidate_residual: np.ndarray,
    truth_residual: np.ndarray,
    valid: np.ndarray,
    factor: int,
    permutations: int,
    seed: int,
) -> dict:
    """Test whether subgrid values are better located than random in each cell."""
    keep = valid & np.isfinite(candidate_residual) & np.isfinite(truth_residual)
    observed_r = correlation(candidate_residual[keep], truth_residual[keep])
    observed_mse = float(np.mean(
        (candidate_residual[keep] - truth_residual[keep]) ** 2
    ))
    rng = np.random.default_rng(seed)
    null_r = np.full(permutations, np.nan)
    null_mse = np.full(permutations, np.nan)
    for index in range(permutations):
        shuffled = permute_within_blocks(candidate_residual, factor, rng)
        null_r[index] = correlation(shuffled[keep], truth_residual[keep])
        null_mse[index] = np.mean((shuffled[keep] - truth_residual[keep]) ** 2)
    finite_r, finite_mse = np.isfinite(null_r), np.isfinite(null_mse)
    return {
        "null": "V7 residual values independently permuted within every 0.1-degree cell",
        "permutations": int(permutations),
        "observed_correlation": observed_r,
        "null_correlation_mean": float(np.nanmean(null_r)),
        "correlation_p_one_sided": float(
            (1 + np.sum(null_r[finite_r] >= observed_r)) / (1 + finite_r.sum())
        ),
        "observed_mse_mm2": observed_mse,
        "null_mse_mean_mm2": float(np.nanmean(null_mse)),
        "mse_p_one_sided": float(
            (1 + np.sum(null_mse[finite_mse] <= observed_mse))
            / (1 + finite_mse.sum())
        ),
    }


def scale_evidence(
    fields: dict[str, np.ndarray],
    truth: np.ndarray,
    valid: np.ndarray,
    factors: tuple[int, ...] = (2, 4, 8),
) -> dict:
    """Return scale-separated deterministic scores for ensemble-mean fields."""
    output: dict[str, dict] = {}
    for name, field in fields.items():
        rows = []
        for factor in factors:
            if truth.shape[-2] % factor or truth.shape[-1] % factor:
                continue
            keep = strict_mask(truth, valid, factor)
            truth_coarse, truth_residual = S.scale_decompose(truth, factor, keep)
            model_coarse, model_residual = S.scale_decompose(field, factor, keep)
            residual_score = anomaly_metrics(model_residual[keep], truth_residual[keep])
            coarse_score = S.deterministic_scores(model_coarse[None], truth_coarse)
            rows.append({
                "factor": int(factor),
                "support_degrees": float(factor * FINE_DEGREES),
                "coarse_component": coarse_score,
                "below_support_component": residual_score,
            })
        output[name] = rows
    return output


def bilinear_sample(
    field: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    station_lat: np.ndarray,
    station_lon: np.ndarray,
) -> np.ndarray:
    y = np.interp(station_lat, lat, np.arange(len(lat)))
    x = np.interp(station_lon, lon, np.arange(len(lon)))
    y0, x0 = np.floor(y).astype(int), np.floor(x).astype(int)
    y0, x0 = np.clip(y0, 0, len(lat) - 1), np.clip(x0, 0, len(lon) - 1)
    y1, x1 = np.clip(y0 + 1, 0, len(lat) - 1), np.clip(x0 + 1, 0, len(lon) - 1)
    wy, wx = y - y0, x - x0
    return (
        field[:, y0, x0] * (1 - wy)[None] * (1 - wx)[None]
        + field[:, y1, x0] * wy[None] * (1 - wx)[None]
        + field[:, y0, x1] * (1 - wy)[None] * wx[None]
        + field[:, y1, x1] * wy[None] * wx[None]
    )


def withheld_bmd_anomalies(
    station_path: Path,
    arm: str,
    model_field: np.ndarray,
    truth: np.ndarray,
    valid: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> dict:
    with np.load(station_path, allow_pickle=False) as station:
        key = f"station_{arm}"
        required = ("observed_mm", "eval_idx", "station_lat", "station_lon", key)
        missing = [name for name in required if name not in station]
        if missing:
            raise ValueError(f"{station_path} lacks {missing}")
        held = np.asarray(station["eval_idx"], int)
        station_lat = np.asarray(station["station_lat"], float)[held]
        station_lon = np.asarray(station["station_lon"], float)[held]
        observed = np.asarray(station["observed_mm"], float)[:, held]
        predicted = np.asarray(station[key], float)[:, :, held].mean(axis=1)
    keep = strict_mask(truth, valid, FOOTPRINT_FACTOR)
    truth_coarse = S.scale_decompose(truth, FOOTPRINT_FACTOR, keep)[0]
    model_coarse = S.scale_decompose(model_field, FOOTPRINT_FACTOR, keep)[0]
    truth_baseline = bilinear_sample(truth_coarse, lat, lon, station_lat, station_lon)
    model_baseline = bilinear_sample(model_coarse, lat, lon, station_lat, station_lon)
    metrics = anomaly_metrics(predicted - model_baseline, observed - truth_baseline)
    metrics["definition"] = (
        "withheld BMD minus CHIRPS 0.1-degree block baseline, compared with "
        "V7 station prediction minus V7's own 0.1-degree block baseline"
    )
    metrics["independence"] = (
        "BMD point values are independent and withheld; the local baseline is CHIRPS"
    )
    return metrics


def load_fields(map_path: Path, arm: str, comparators: list[str]) -> dict:
    with np.load(map_path, allow_pickle=False) as archive:
        required = ("grid_lat", "grid_lon", "valid", f"meanfield_{arm}")
        missing = [name for name in required if name not in archive]
        if missing:
            raise ValueError(f"{map_path} lacks {missing}")
        fields = {arm: np.asarray(archive[f"meanfield_{arm}"], float)}
        for name in comparators:
            key = f"meanfield_{name}"
            if key in archive:
                fields[name] = np.asarray(archive[key], float)
        dates_key = "model_times" if "model_times" in archive else "times"
        return {
            "dates": np.asarray(archive[dates_key]).astype("datetime64[D]"),
            "lat": np.asarray(archive["grid_lat"], float),
            "lon": np.asarray(archive["grid_lon"], float),
            "valid": np.asarray(archive["valid"], bool),
            "fields": fields,
        }


def load_chirps_target(
    archive_path: Path, dates: np.ndarray, target_lat: np.ndarray, target_lon: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    import zarr

    root = zarr.open_group(str(archive_path), mode="r")
    for name in ("time", "lat", "lon", "fine_mm", "fine_valid"):
        if name not in root:
            raise ValueError(f"{archive_path} lacks {name}")
    raw_dates = np.asarray(root["time"][:])
    source_dates = (
        raw_dates.astype("datetime64[D]")
        if np.issubdtype(raw_dates.dtype, np.datetime64)
        else raw_dates.astype("datetime64[ns]").astype("datetime64[D]")
    )
    date_lookup = {day: index for index, day in enumerate(source_dates)}
    missing = [str(day) for day in dates if day not in date_lookup]
    if missing:
        raise ValueError(f"{archive_path} lacks model dates {missing}")
    source_lat = np.asarray(root["lat"][:], float)
    source_lon = np.asarray(root["lon"][:], float)

    def positions(source, target, label):
        index = np.asarray([np.argmin(np.abs(source - value)) for value in target])
        if not np.allclose(source[index], target, atol=1e-5, rtol=0.0):
            raise ValueError(f"map {label} coordinates are not nested in CHIRPS archive")
        if not np.array_equal(index, np.arange(index[0], index[0] + len(index))):
            raise ValueError(f"map {label} coordinates are not contiguous")
        return index

    rows, columns = positions(source_lat, target_lat, "latitude"), positions(
        source_lon, target_lon, "longitude"
    )
    truth = np.stack([
        np.asarray(root["fine_mm"][date_lookup[day], rows[0]:rows[-1] + 1,
                                    columns[0]:columns[-1] + 1], float)
        for day in dates
    ])
    source_valid = np.asarray(
        root["fine_valid"][rows[0]:rows[-1] + 1, columns[0]:columns[-1] + 1], bool
    )
    return truth, np.broadcast_to(source_valid, truth.shape).copy()


def evaluate(
    fields: dict[str, np.ndarray],
    truth: np.ndarray,
    valid: np.ndarray,
    arm: str,
    permutations: int,
    bootstrap_resamples: int,
    bootstrap_block_days: int,
    seed: int,
) -> dict:
    evidence = scale_evidence(fields, truth, valid)
    keep = strict_mask(truth, valid, FOOTPRINT_FACTOR)
    truth_coarse, truth_residual = S.scale_decompose(truth, FOOTPRINT_FACTOR, keep)
    arm_coarse, arm_residual = S.scale_decompose(fields[arm], FOOTPRINT_FACTOR, keep)
    daily = daily_residual_skill(fields[arm], truth, valid, FOOTPRINT_FACTOR)
    bootstrap = day_bootstrap(
        daily, bootstrap_resamples, seed, bootstrap_block_days
    )
    placement = placement_permutation_test(
        arm_residual, truth_residual, keep, FOOTPRINT_FACTOR, permutations, seed + 1
    )
    full_score = S.deterministic_scores(fields[arm][None], np.where(keep, truth, np.nan))
    oracle = S.footprint_perfect_null(truth, FOOTPRINT_FACTOR, keep)
    oracle_score = S.deterministic_scores(oracle, np.where(keep, truth, np.nan))
    strict_skill = S.skill_against(full_score, oracle_score)
    spectra = S.spectral_summary(fields, truth, valid, FINE_DEGREES)
    fss = {
        name: S.fss_grid(field, truth, valid, windows=(1, 3, 5, 9, 17, 33))
        for name, field in {**fields, "footprint_perfect": truth_coarse}.items()
    }
    target_row = evidence[arm][0]["below_support_component"]
    chirps_supported = bool(
        target_row.get("mse_skill_vs_no_subgrid", -np.inf) > 0
        and target_row.get("correlation", -np.inf) > 0
        and placement["correlation_p_one_sided"] <= 0.05
        and (bootstrap["ci_low"] is not None and bootstrap["ci_low"] > 0)
    )
    return {
        "scale_ladder": evidence,
        "target_0p1_residual": target_row,
        "daily_mse_skill_vs_no_subgrid": daily,
        "daily_block_bootstrap": bootstrap,
        "within_footprint_placement_test": placement,
        "strict_full_field_test_vs_truth_coarse_oracle": {
            "model": full_score,
            "oracle": oracle_score,
            "skill_vs_oracle": strict_skill,
        },
        "spectra": spectra,
        "fss": fss,
        "chirps_supported": chirps_supported,
        "_plot": {"truth_residual": truth_residual, "arm_residual": arm_residual},
    }


def plot_summary(result: dict, arm: str, out_path: Path) -> None:
    ladder = result["scale_ladder"]
    names = list(ladder)
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    width = 0.8 / max(len(names), 1)
    degrees = [row["support_degrees"] for row in ladder[arm]]
    x = np.arange(len(degrees))
    for index, name in enumerate(names):
        rows = ladder[name]
        axes[0, 0].bar(
            x + (index - (len(names) - 1) / 2) * width,
            [row["below_support_component"].get("mse_skill_vs_no_subgrid", np.nan)
             for row in rows], width=width, label=name,
        )
        axes[0, 1].bar(
            x + (index - (len(names) - 1) / 2) * width,
            [row["below_support_component"].get("correlation", np.nan) for row in rows],
            width=width, label=name,
        )
    for axis, title in zip(
        axes[0], ("MSE skill below support vs no-subgrid", "Residual pattern correlation")
    ):
        axis.axhline(0, color="black", lw=0.8)
        axis.set_xticks(x, [f"{degree:g}°" for degree in degrees])
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(fontsize=8)

    spectra = result["spectra"]
    wavelength = np.asarray(spectra.get("wavelength_km", []), float)
    for name, ratio in spectra.get("power_ratio", {}).items():
        axes[1, 0].plot(wavelength, ratio, label=name)
    axes[1, 0].axhspan(0.5, 2.0, color="grey", alpha=0.15, label="0.5–2× CHIRPS")
    axes[1, 0].axvline(11.1, color="black", ls="--", lw=0.9,
                       label="0.1° Nyquist wavelength")
    axes[1, 0].set_xscale("log"); axes[1, 0].set_yscale("log")
    axes[1, 0].invert_xaxis(); axes[1, 0].set_xlabel("wavelength (km)")
    axes[1, 0].set_ylabel("power / CHIRPS power")
    axes[1, 0].set_title("Texture amplitude (not placement skill)")
    axes[1, 0].legend(fontsize=7)

    fss = result["fss"]
    for name in (arm, "footprint_perfect"):
        if name not in fss:
            continue
        windows = np.asarray(fss[name]["windows"], float) * FINE_DEGREES * 111.0
        for threshold in ("5", "10", "25"):
            axes[1, 1].plot(
                windows, fss[name]["fss"].get(threshold, []), marker="o",
                label=f"{name}, ≥{threshold} mm",
            )
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_xlabel("neighbourhood width (nominal km)")
    axes[1, 1].set_ylabel("FSS")
    axes[1, 1].set_title("Rain-event location by neighbourhood")
    axes[1, 1].grid(alpha=0.25); axes[1, 1].legend(fontsize=7)
    figure.suptitle(f"V7 {arm}: is the 0.05° grid carrying real subgrid information?")
    figure.savefig(out_path, dpi=160)
    plt.close(figure)


def plot_case(
    dates: np.ndarray,
    truth: np.ndarray,
    field: np.ndarray,
    valid: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    arm: str,
    out_path: Path,
) -> None:
    keep = strict_mask(truth, valid, FOOTPRINT_FACTOR)
    truth_coarse, truth_residual = S.scale_decompose(truth, FOOTPRINT_FACTOR, keep)
    model_coarse, model_residual = S.scale_decompose(field, FOOTPRINT_FACTOR, keep)
    energy = np.nanmean(truth_residual**2, axis=(1, 2))
    day = int(np.nanargmax(energy))
    rain_max = float(np.nanpercentile(np.concatenate([
        truth[day][keep[day]], field[day][keep[day]]
    ]), 99.5))
    residual_max = float(np.nanpercentile(np.abs(np.concatenate([
        truth_residual[day][keep[day]], model_residual[day][keep[day]]
    ])), 99.0))
    panels = (
        (truth[day], "CHIRPS 0.05°", "turbo", 0, rain_max),
        (field[day], arm, "turbo", 0, rain_max),
        (truth_coarse[day], "CHIRPS exact 0.1° mean", "turbo", 0, rain_max),
        (model_coarse[day], "V7 0.1° mean", "turbo", 0, rain_max),
        (truth_residual[day], "CHIRPS below-0.1° residual", "RdBu_r", -residual_max, residual_max),
        (model_residual[day], "V7 below-0.1° residual", "RdBu_r", -residual_max, residual_max),
    )
    extent = (lon.min() - 0.025, lon.max() + 0.025,
              lat.min() - 0.025, lat.max() + 0.025)
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for axis, (values, title, cmap, low, high) in zip(axes.ravel(), panels):
        image = axis.imshow(np.where(keep[day], values, np.nan), origin="lower",
                            extent=extent, cmap=cmap, vmin=low, vmax=high, aspect="auto")
        axis.set_title(title); figure.colorbar(image, ax=axis, shrink=0.82, label="mm/day")
    figure.suptitle(
        f"Scale-separated case: model day {dates[day]}\n"
        "Bottom row removes each field's own exact 0.1° block mean"
    )
    figure.savefig(out_path, dpi=160)
    plt.close(figure)


def report_lines(result: dict, arm: str, dates: np.ndarray) -> list[str]:
    target = result["target_0p1_residual"]
    placement = result["within_footprint_placement_test"]
    bootstrap = result["daily_block_bootstrap"]
    strict = result["strict_full_field_test_vs_truth_coarse_oracle"]["skill_vs_oracle"]
    bmd = result.get("withheld_bmd_subgrid_anomalies")
    supported = result["chirps_supported"]
    if supported and bmd and bmd.get("mse_skill_vs_no_subgrid", -1) > 0 and bmd.get(
        "correlation", -1
    ) > 0:
        verdict = "Supported on this ten-day window by both CHIRPS and withheld BMD anomalies."
    elif supported:
        verdict = "Supported against CHIRPS on this window, but not independently confirmed."
    else:
        verdict = "Not demonstrated on this window."
    lines = [
        f"# V7 `{arm}` subgrid-resolution test",
        "",
        f"**Verdict: {verdict}**",
        "",
        f"Model dates: {dates[0]} through {dates[-1]} ({len(dates)} days).",
        "",
        "## Decisive below-0.1° tests",
        "",
        "| Test | Result | Passing evidence |",
        "|:--|--:|:--|",
        f"| Residual correlation with CHIRPS | {target.get('correlation', np.nan):.3f} | > 0 |",
        f"| Residual MSE skill vs zero subgrid | {target.get('mse_skill_vs_no_subgrid', np.nan):+.3f} | > 0 |",
        f"| Day-bootstrap mean skill (95% CI) | {bootstrap.get('mean', np.nan):+.3f} "
        f"[{bootstrap.get('ci_low', np.nan):+.3f}, {bootstrap.get('ci_high', np.nan):+.3f}] | lower bound > 0 |",
        f"| Within-cell placement permutation p | {placement['correlation_p_one_sided']:.4f} | ≤ 0.05 |",
        f"| Full-field MSE skill vs perfect 0.1° oracle | {strict.get('mse_skill', np.nan):+.3f} | > 0 is exceptionally strong |",
        "",
    ]
    if bmd:
        lines += [
            "## Independent point check",
            "",
            f"At {bmd['n']} withheld BMD station-days, subgrid-anomaly correlation is "
            f"{bmd.get('correlation', np.nan):.3f} and MSE skill versus no subgrid is "
            f"{bmd.get('mse_skill_vs_no_subgrid', np.nan):+.3f}.",
            "",
    ]
    lines += [
        "## Scientific interpretation",
        "",
        "- The residual test removes the exact 0.1° mean from both fields. Broad storm "
        "placement, bias correction, and IMERG adherence therefore cannot create a win.",
        "- The permutation null preserves V7's within-cell variance but scrambles which of "
        "the four 0.05° cells receives it. Beating it tests location, not sharpness.",
        "- The withheld-BMD values are independent, but their local 0.1° reference comes "
        "from CHIRPS; treat this as supporting evidence rather than a fully independent grid.",
        "- CHIRPS is the stage-B training target. May 2022 lies outside V7's configured "
        "1981–2018 training and 2019–2020 validation periods, so this tests temporal "
        "generalization within the same product family—not an independent fine-grid truth.",
        "- A 0.05° grid's shortest representable wavelength is about 0.1° (~11 km). "
        "The defensible claim is skill below 0.1° support, not literal 5.5-km resolution.",
        "- Ten tuning days cannot establish a climatological result. Confirm on withheld "
        "dates with a day-block bootstrap before making a publication-level claim.",
    ]
    return lines


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    comparators = [name.strip() for name in args.comparators.split(",") if name.strip()]
    loaded = load_fields(Path(args.map_dump), args.arm, comparators)
    truth, truth_valid = load_chirps_target(
        Path(args.subgrid_archive), loaded["dates"], loaded["lat"], loaded["lon"]
    )
    valid = loaded["valid"] & truth_valid & np.isfinite(truth)
    for field in loaded["fields"].values():
        valid &= np.isfinite(field)
    result = evaluate(
        loaded["fields"], truth, valid, args.arm, args.permutations,
        args.bootstrap_resamples, args.bootstrap_block_days, args.seed,
    )
    result.pop("_plot", None)
    if args.station_dump:
        result["withheld_bmd_subgrid_anomalies"] = withheld_bmd_anomalies(
            Path(args.station_dump), args.arm, loaded["fields"][args.arm], truth,
            valid, loaded["lat"], loaded["lon"],
        )
    result.update({
        "arm": args.arm,
        "model_dates": loaded["dates"].astype(str).tolist(),
        "fine_grid_degrees": FINE_DEGREES,
        "tested_footprint_degrees": FINE_DEGREES * FOOTPRINT_FACTOR,
        "nyquist_wavelength_degrees": 2 * FINE_DEGREES,
        "claim_scope": (
            "located variation among 0.05-degree cells within 0.1-degree support; "
            "not literal 0.05-degree wavelength resolution"
        ),
    })
    plot_summary(result, args.arm, out_dir / "v7_r81_subgrid_evidence.png")
    plot_case(
        loaded["dates"], truth, loaded["fields"][args.arm], valid,
        loaded["lat"], loaded["lon"], args.arm,
        out_dir / "v7_r81_subgrid_case.png",
    )
    (out_dir / "v7_r81_subgrid_evidence.json").write_text(
        json.dumps(_json_ready(result), indent=2)
    )
    (out_dir / "v7_r81_subgrid_evidence.md").write_text(
        "\n".join(report_lines(result, args.arm, loaded["dates"])) + "\n"
    )
    print("\n".join(report_lines(result, args.arm, loaded["dates"])), flush=True)
    print(f"[subgrid] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
