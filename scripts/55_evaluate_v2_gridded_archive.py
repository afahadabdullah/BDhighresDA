#!/usr/bin/env python
"""Evaluate completed CPC-v2 gridded analyses without calling CHIRPS truth.

The real-data experiment has no gridded truth.  CHIRPS is the training target,
IMERG enters the simultaneous likelihood, CPC conditions the prior, and BMD
gauges are point observations.  Treating any one of those as a perfect fine-grid
verification field would make the downscaling claim circular.

This script therefore keeps three kinds of evidence separate:

1. ``independent_gauges``: ordinary held-out-gauge scores plus *sub-footprint
   gauge anomalies*.  For the latter, an independently withheld gauge anomaly
   is defined relative to the local 0.4-degree mean of IMERG, CHIRPS, or CPC,
   while the model anomaly is its fine prediction minus its own 0.4-degree
   block mean. Agreement robust to all three baselines is the strongest
   real-data evidence that located subgrid structure is useful.
2. ``product_agreement``: daily/monthly spatial correlation and variability
   agreement with CHIRPS, IMERG, and CPC.  These diagnose plausibility and
   observation adherence; they are never labelled skill or truth.
3. ``reference_free_structure``: variance below the 0.4-degree footprint,
   spectral power, variograms, member texture and ensemble coherence.  These
   establish that the model resolves rather than merely upsamples, but texture
   alone cannot prove correct placement.

The script accepts one completed seasonal Zarr immediately, or several stores
for a pooled analysis.  Every figure ships its underlying CSV tables through
``bdhires.paper.save_figure``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bdhires.eval import scale as S  # noqa: E402
from bdhires.paper import save_figure, use_paper_style  # noqa: E402


FOOTPRINT_FACTOR = 8
FINE_DEGREES = 0.05
WET_MM = 1.0
REFERENCE_NAMES = ("chirps", "imerg", "cpc")
SELECTION_START = np.datetime64("2022-05-01", "D")
SELECTION_END = np.datetime64("2022-05-10", "D")


def confirmatory_daily_mask(times: np.ndarray) -> np.ndarray:
    times = np.asarray(times).astype("datetime64[D]")
    return ~((times >= SELECTION_START) & (times <= SELECTION_END))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--zarr", nargs="+", required=True,
                        help="one or more completed seasonal production stores")
    parser.add_argument(
        "--cv-root", default=None,
        help="optional experiment root containing cv/<period>/fold*.npz; "
             "default is inferred from a .../gridded/<period>.zarr path",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--factor", type=int, default=FOOTPRINT_FACTOR)
    parser.add_argument("--texture-members", type=int, default=5,
                        help="evenly spaced members used for spectra/variograms")
    parser.add_argument("--minimum-block-valid", type=float, default=1.0)
    return parser.parse_args()


def finite_float(value) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, float).ravel()
    second = np.asarray(second, float).ravel()
    keep = np.isfinite(first) & np.isfinite(second)
    if keep.sum() < 3 or first[keep].std() == 0 or second[keep].std() == 0:
        return float("nan")
    return float(np.corrcoef(first[keep], second[keep])[0, 1])


def centered_rmse(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, float).ravel()
    second = np.asarray(second, float).ravel()
    keep = np.isfinite(first) & np.isfinite(second)
    if not keep.any():
        return float("nan")
    a = first[keep] - first[keep].mean()
    b = second[keep] - second[keep].mean()
    return float(np.sqrt(np.mean((a - b) ** 2)))


def ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def daily_spatial_correlation(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.asarray([correlation(a, b) for a, b in zip(first, second)], float)


def daily_centered_rmse(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.asarray([centered_rmse(a, b) for a, b in zip(first, second)], float)


def spatial_statistics(field: np.ndarray, valid: np.ndarray) -> dict[str, np.ndarray]:
    field = np.where(valid[None], np.asarray(field, float), np.nan)
    flat = field.reshape(field.shape[0], -1)
    wet = np.where(np.isfinite(flat), flat >= WET_MM, np.nan)
    return {
        "domain_mean_mm": np.nanmean(flat, axis=1),
        "spatial_std_mm": np.nanstd(flat, axis=1),
        "wet_area_fraction": np.nanmean(wet, axis=1),
        "spatial_p95_mm": np.nanpercentile(flat, 95, axis=1),
    }


def block_stack(field: np.ndarray, factor: int, valid: np.ndarray) -> np.ndarray:
    mask = np.broadcast_to(valid, np.asarray(field).shape)
    return S.block_mean(np.asarray(field, float), factor, mask)


def upsample_coarse(field: np.ndarray, factor: int, shape: tuple[int, int]) -> np.ndarray:
    field = np.asarray(field, float)
    expected = (shape[0] // factor, shape[1] // factor)
    if field.shape[-2:] != expected:
        raise ValueError(
            f"coarse product shape {field.shape[-2:]} does not match factor "
            f"{factor} on fine shape {shape}"
        )
    return S.upsample_blocks(field, factor)


def strict_block_mask(valid: np.ndarray, factor: int, minimum: float) -> np.ndarray:
    dummy = np.where(valid, 0.0, np.nan)[None]
    return S.eligible_mask(dummy, factor, minimum)[0] & valid


def residual_stack(field: np.ndarray, factor: int, mask: np.ndarray) -> np.ndarray:
    _, residual = S.scale_decompose(
        np.asarray(field, float), factor, np.broadcast_to(mask, np.asarray(field).shape)
    )
    return residual


def gradient_energy(field: np.ndarray, mask: np.ndarray) -> float:
    values = []
    for layer in np.asarray(field, float):
        work = np.where(mask, layer, np.nan)
        for difference in (np.diff(work, axis=0), np.diff(work, axis=1)):
            finite = difference[np.isfinite(difference)]
            if finite.size:
                values.append(np.mean(finite**2))
    return float(np.mean(values)) if values else float("nan")


def energy(field: np.ndarray, mask: np.ndarray) -> float:
    values = np.asarray(field, float)[..., mask]
    return float(np.nanmean(values**2))


def spatial_anomaly_energy(field: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(field, float)[..., mask]
    selected = selected - np.nanmean(selected, axis=-1, keepdims=True)
    return float(np.nanmean(selected**2))


def fair_crps_per_sample(members: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Fair CRPS for members (sample, member), preserving missing samples."""
    members = np.asarray(members, float)
    truth = np.asarray(truth, float)
    count = members.shape[1]
    output = np.full(truth.shape, np.nan)
    finite = np.isfinite(truth) & np.all(np.isfinite(members), axis=1)
    if not finite.any():
        return output
    selected = members[finite]
    first = np.mean(np.abs(selected - truth[finite, None]), axis=1)
    ordered = np.sort(selected, axis=1)
    weights = 2 * np.arange(1, count + 1) - count - 1
    pair = np.sum(ordered * weights[None], axis=1) / (count * (count - 1))
    output[finite] = first - pair
    return output


def point_metrics(members: np.ndarray, truth: np.ndarray) -> dict:
    members = np.asarray(members, float)
    truth = np.asarray(truth, float)
    keep = np.isfinite(truth) & np.all(np.isfinite(members), axis=1)
    if not keep.any():
        return {"n": 0}
    members, truth = members[keep], truth[keep]
    mean = members.mean(axis=1)
    difference = mean - truth
    wet = truth >= WET_MM
    low, high = np.quantile(members, [0.05, 0.95], axis=1)
    return {
        "n": int(len(truth)),
        "crps_mm": float(np.mean(fair_crps_per_sample(members, truth))),
        "mae_mm": float(np.mean(np.abs(difference))),
        "dry_mae_mm": float(np.mean(np.abs(difference[~wet]))) if (~wet).any() else None,
        "wet_mae_mm": float(np.mean(np.abs(difference[wet]))) if wet.any() else None,
        "bias_mm": float(np.mean(difference)),
        "correlation": finite_float(correlation(mean, truth)),
        "coverage_90": float(np.mean((truth >= low) & (truth <= high))),
    }


def anomaly_metrics(predicted: np.ndarray, observed: np.ndarray) -> dict:
    predicted = np.asarray(predicted, float)
    observed = np.asarray(observed, float)
    keep = np.isfinite(predicted) & np.isfinite(observed)
    if not keep.any():
        return {"n": 0}
    predicted, observed = predicted[keep], observed[keep]
    difference = predicted - observed
    active = np.abs(observed) >= 1.0
    null_mse = float(np.mean(observed**2))
    model_mse = float(np.mean(difference**2))
    return {
        "n": int(len(observed)),
        "correlation": finite_float(correlation(predicted, observed)),
        "rmse_mm": float(np.sqrt(np.mean(difference**2))),
        "mse_skill_vs_no_subgrid": (
            float(1.0 - model_mse / null_mse) if null_mse > 0 else None
        ),
        "bias_mm": float(np.mean(difference)),
        "variance_ratio": ratio(float(np.var(predicted)), float(np.var(observed))),
        "sign_agreement": (
            float(np.mean(np.sign(predicted[active]) == np.sign(observed[active])))
            if active.any() else None
        ),
    }


def bilinear_sample(field: np.ndarray, grid_lat: np.ndarray, grid_lon: np.ndarray,
                    station_lat: np.ndarray, station_lon: np.ndarray) -> np.ndarray:
    """Bilinearly sample (time,lat,lon) without importing the torch DA path."""
    field = np.asarray(field, float)
    lat = np.asarray(grid_lat, float)
    lon = np.asarray(grid_lon, float)
    y = np.interp(station_lat, lat, np.arange(len(lat)))
    x = np.interp(station_lon, lon, np.arange(len(lon)))
    y0 = np.clip(np.floor(y).astype(int), 0, len(lat) - 1)
    x0 = np.clip(np.floor(x).astype(int), 0, len(lon) - 1)
    y1 = np.clip(y0 + 1, 0, len(lat) - 1)
    x1 = np.clip(x0 + 1, 0, len(lon) - 1)
    wy, wx = y - y0, x - x0
    return (
        field[:, y0, x0] * (1 - wy)[None] * (1 - wx)[None]
        + field[:, y1, x0] * wy[None] * (1 - wx)[None]
        + field[:, y0, x1] * (1 - wy)[None] * wx[None]
        + field[:, y1, x1] * wy[None] * wx[None]
    )


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def json_ready(value):
    """Recursively replace NumPy scalars and non-finite floats for strict JSON."""
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def validate_store(path: Path):
    import xarray as xr

    if not path.is_dir():
        raise FileNotFoundError(path)
    dataset = xr.open_zarr(path, consolidated=True)
    if dataset.attrs.get("schema") != "bdhires.physical_ensemble.v1":
        raise ValueError(f"{path}: unsupported schema {dataset.attrs.get('schema')}")
    if not dataset.attrs.get("complete"):
        raise ValueError(f"{path}: store is not marked complete")
    if not dataset.attrs.get("scope", {}).get("assimilate_all_stations"):
        raise ValueError(f"{path}: expected an all-station production analysis")
    required = {
        "precipitation", "ensemble_mean", "ensemble_std", "cpc", "chirps",
        "imerg", "valid", "gauge", "assimilated_station",
    }
    missing = required - set(dataset.variables)
    if missing:
        raise ValueError(f"{path}: missing {sorted(missing)}")
    if not bool(np.asarray(dataset.assimilated_station).all()):
        raise ValueError(f"{path}: not every production station was assimilated")
    times = np.asarray(dataset.time.values).astype("datetime64[D]")
    if len(np.unique(times)) != len(times):
        raise ValueError(f"{path}: duplicate dates")
    return dataset


def load_archive(paths: list[Path], factor: int) -> dict:
    datasets = [validate_store(path) for path in paths]
    methods = datasets[0].method.values.astype(str).tolist()
    lat = np.asarray(datasets[0].lat.values, float)
    lon = np.asarray(datasets[0].lon.values, float)
    valid = np.asarray(datasets[0].valid.values, bool)
    if valid.shape[0] % factor or valid.shape[1] % factor:
        raise ValueError(
            f"fine grid {valid.shape} is not divisible by footprint factor {factor}"
        )
    for path, dataset in zip(paths[1:], datasets[1:]):
        if dataset.method.values.astype(str).tolist() != methods:
            raise ValueError(f"{path}: method order differs")
        if not np.array_equal(dataset.lat.values, lat) or not np.array_equal(dataset.lon.values, lon):
            raise ValueError(f"{path}: fine grid differs")
        if not np.array_equal(dataset.valid.values, valid):
            raise ValueError(f"{path}: valid mask differs")
    times = np.concatenate([
        np.asarray(dataset.time.values).astype("datetime64[D]") for dataset in datasets
    ])
    if len(np.unique(times)) != len(times):
        raise ValueError("seasonal stores contain overlapping dates")
    order = np.argsort(times)

    def concatenate(name: str) -> np.ndarray:
        values = np.concatenate([np.asarray(dataset[name].values) for dataset in datasets], axis=0)
        return values[order]

    # ensemble_mean is method,time,lat,lon, so concatenate explicitly on time.
    mean = np.concatenate([np.asarray(dataset.ensemble_mean.values) for dataset in datasets], axis=1)
    mean = mean[:, order]
    spread = np.concatenate([np.asarray(dataset.ensemble_std.values) for dataset in datasets], axis=1)
    spread = spread[:, order]
    imerg_coarse = concatenate("imerg")
    imerg = upsample_coarse(imerg_coarse, factor, valid.shape)
    return {
        "paths": paths,
        "datasets": datasets,
        "methods": methods,
        "time": times[order],
        "lat": lat,
        "lon": lon,
        "valid": valid,
        "mean": mean,
        "spread": spread,
        "chirps": concatenate("chirps"),
        "cpc": concatenate("cpc"),
        "imerg": imerg,
        "imerg_coarse": imerg_coarse,
    }


def product_fields(archive: dict) -> dict[str, np.ndarray]:
    return {name: archive[name] for name in REFERENCE_NAMES}


def evaluate_daily_and_monthly(archive: dict, factor: int) -> tuple[list[dict], list[dict], list[dict]]:
    methods = archive["methods"]
    times = archive["time"]
    valid = archive["valid"]
    products = product_fields(archive)
    primary = confirmatory_daily_mask(times)
    if not primary.any():
        raise ValueError(
            "no confirmatory days remain after excluding 2022-05-01 through 2022-05-10"
        )
    stats = {name: spatial_statistics(field, valid) for name, field in products.items()}
    for name in REFERENCE_NAMES:
        stats[name]["posterior_spread_mm"] = np.full(len(times), np.nan)
    for index, name in enumerate(methods):
        stats[name] = spatial_statistics(archive["mean"][index], valid)
        stats[name]["posterior_spread_mm"] = np.nanmean(
            archive["spread"][index][:, valid], axis=1
        )

    daily_rows = []
    for position, day in enumerate(times):
        for name in [*methods, *REFERENCE_NAMES]:
            daily_rows.append({
                "date": str(day), "source": name,
                **{metric: float(values[position]) for metric, values in stats[name].items()},
            })

    common = {
        name: block_stack(field, factor, valid) for name, field in products.items()
    }
    matrix_rows = []
    for index, method in enumerate(methods):
        field = archive["mean"][index]
        method_common = block_stack(field, factor, valid)
        row = {"method": method}
        for reference, reference_field in products.items():
            fine_r = daily_spatial_correlation(field, reference_field)[primary]
            common_r = daily_spatial_correlation(method_common, common[reference])[primary]
            common_crmse = daily_centered_rmse(method_common, common[reference])[primary]
            row[f"daily_fine_r_{reference}"] = finite_float(np.nanmean(fine_r))
            row[f"daily_common_r_{reference}"] = finite_float(np.nanmean(common_r))
            row[f"daily_common_crmse_{reference}_mm"] = finite_float(np.nanmean(common_crmse))
            row[f"daily_domain_mean_r_{reference}"] = finite_float(correlation(
                stats[method]["domain_mean_mm"][primary],
                stats[reference]["domain_mean_mm"][primary],
            ))
            row[f"daily_domain_mean_bias_{reference}_mm"] = finite_float(np.nanmean(
                stats[method]["domain_mean_mm"][primary]
                - stats[reference]["domain_mean_mm"][primary]
            ))
        row["confirmatory_days"] = int(primary.sum())
        matrix_rows.append(row)

    monthly_rows, monthly_matrix = [], []
    months = times.astype("datetime64[M]")
    for month in np.unique(months):
        choose = months == month
        monthly_fields = {
            name: np.nanmean(field[choose], axis=0) for name, field in products.items()
        }
        variability_fields = {
            name: np.nanstd(field[choose], axis=0) for name, field in products.items()
        }
        for index, method in enumerate(methods):
            monthly_fields[method] = np.nanmean(archive["mean"][index, choose], axis=0)
            variability_fields[method] = np.nanstd(archive["mean"][index, choose], axis=0)
        for name in [*methods, *REFERENCE_NAMES]:
            values = monthly_fields[name][valid]
            variability = variability_fields[name][valid]
            monthly_rows.append({
                "month": str(month), "source": name,
                "monthly_domain_mean_mm": float(np.nanmean(values)),
                "monthly_spatial_std_mm": float(np.nanstd(values)),
                "within_month_daily_variability_mm": float(np.nanmean(variability)),
                "within_month_variability_spatial_std_mm": float(np.nanstd(variability)),
                "mean_posterior_spread_mm": (
                    finite_float(np.nanmean(stats[name]["posterior_spread_mm"][choose]))
                    if name in methods else None
                ),
                "n_days": int(choose.sum()),
            })
        for method in methods:
            row = {
                "month": str(month), "method": method,
                "confirmatory": str(month) != "2022-05",
            }
            for reference in REFERENCE_NAMES:
                row[f"monthly_mean_r_{reference}"] = finite_float(correlation(
                    monthly_fields[method][valid], monthly_fields[reference][valid]
                ))
                row[f"monthly_mean_bias_{reference}_mm"] = finite_float(np.nanmean(
                    monthly_fields[method][valid] - monthly_fields[reference][valid]
                ))
                row[f"monthly_variability_r_{reference}"] = finite_float(correlation(
                    variability_fields[method][valid], variability_fields[reference][valid]
                ))
                row[f"monthly_variability_ratio_{reference}"] = finite_float(ratio(
                    float(np.nanmean(variability_fields[method][valid] ** 2)),
                    float(np.nanmean(variability_fields[reference][valid] ** 2)),
                ))
            monthly_matrix.append(row)
    return daily_rows, monthly_rows, matrix_rows + monthly_matrix


def selected_member_indices(count: int, requested: int) -> np.ndarray:
    requested = max(1, min(int(requested), count))
    return np.unique(np.linspace(0, count - 1, requested).round().astype(int))


def mean_spectrum(layers: list[np.ndarray], masks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    wavelength = None
    powers = []
    for layer, mask in zip(layers, masks):
        current_wavelength, power = S.rapsd(layer, mask, FINE_DEGREES)
        if not len(power):
            continue
        if wavelength is None:
            wavelength = current_wavelength
        if power.shape == wavelength.shape:
            powers.append(power)
    if wavelength is None or not powers:
        return np.asarray([]), np.asarray([])
    return wavelength, np.nanmean(np.stack(powers), axis=0)


def layer_semivariance(layer: np.ndarray, mask: np.ndarray,
                       lags: tuple[int, ...]) -> np.ndarray:
    values = []
    work = np.where(mask, layer, np.nan)
    for lag in lags:
        parts = []
        for shifted in (
            work[lag:, :] - work[:-lag, :],
            work[:, lag:] - work[:, :-lag],
        ):
            finite = shifted[np.isfinite(shifted)]
            if finite.size:
                parts.append(0.5 * np.mean(finite**2))
        values.append(float(np.mean(parts)) if parts else np.nan)
    return np.asarray(values)


def evaluate_subgrid(archive: dict, factor: int, texture_members: int,
                     minimum_block_valid: float) -> tuple[list[dict], dict, list[dict]]:
    valid = archive["valid"]
    primary = confirmatory_daily_mask(archive["time"])
    block_valid = strict_block_mask(valid, factor, minimum_block_valid)
    if not block_valid.any():
        raise ValueError(
            f"no {factor}x{factor} footprint has the requested valid-land fraction"
        )
    chirps = archive["chirps"][primary]
    chirps_residual = residual_stack(chirps, factor, block_valid)
    chirps_energy = energy(chirps_residual, block_valid)
    chirps_gradient = gradient_energy(chirps_residual, block_valid)
    methods = archive["methods"]
    rows = []

    # Mean-field spectra, residual ladders and variograms.
    spectra: dict = {"wavelength_km": None, "mean_power": {}, "member_power": {}}
    variogram_rows: list[dict] = []
    scale_rows: list[dict] = []
    lags = tuple(lag for lag in (1, 2, 4, 8, 16, 32) if lag < min(valid.shape))
    masks_by_day = [valid] * int(primary.sum())
    wavelength, chirps_power = mean_spectrum(list(chirps), masks_by_day)
    spectra["wavelength_km"] = wavelength.tolist()
    spectra["mean_power"]["chirps"] = chirps_power.tolist()
    chirps_variogram = np.nanmean(np.stack([
        layer_semivariance(layer, valid, lags) for layer in chirps
    ]), axis=0)
    for lag, value in zip(lags, chirps_variogram):
        variogram_rows.append({"source": "chirps", "kind": "reference", "lag_cells": lag,
                               "lag_km": lag * FINE_DEGREES * 111.0,
                               "semivariance": float(value)})

    factors = tuple(
        value for value in (2, 4, 8, 16)
        if valid.shape[0] % value == 0 and valid.shape[1] % value == 0
    )
    chirps_residual_energy_by_factor = {}
    for scale_factor in factors:
        scale_mask = strict_block_mask(valid, scale_factor, minimum_block_valid)
        value = energy(residual_stack(chirps, scale_factor, scale_mask), scale_mask)
        chirps_residual_energy_by_factor[scale_factor] = value
        scale_rows.append({"source": "chirps", "kind": "reference",
                           "factor": scale_factor,
                           "scale_deg": scale_factor * FINE_DEGREES,
                           "residual_variance_mm2": value,
                           "ratio_to_chirps": 1.0})

    for method_index, method in enumerate(methods):
        mean_field = archive["mean"][method_index, primary]
        mean_residual = residual_stack(mean_field, factor, block_valid)
        mean_residual_energy = energy(mean_residual, block_valid)
        mean_total_energy = spatial_anomaly_energy(mean_field, block_valid)
        daily_residual_r = daily_spatial_correlation(mean_residual, chirps_residual)
        member_residual_sum = member_total_sum = 0.0
        member_layers = []
        member_masks = []
        member_variograms = []
        member_count = 0
        for dataset in archive["datasets"]:
            member_indices = selected_member_indices(dataset.sizes["member"], texture_members)
            dataset_times = np.asarray(dataset.time.values).astype("datetime64[D]")
            dataset_primary = confirmatory_daily_mask(dataset_times)
            values = np.asarray(
                dataset.precipitation.sel({"method": method}).isel(
                    time=np.flatnonzero(dataset_primary), member=member_indices
                ).values,
                float,
            )  # time, selected-member, lat, lon
            for day in range(values.shape[0]):
                for member in range(values.shape[1]):
                    layer = values[day, member]
                    residual = residual_stack(layer[None], factor, block_valid)[0]
                    member_residual_sum += energy(residual[None], block_valid)
                    member_total_sum += spatial_anomaly_energy(layer[None], block_valid)
                    member_layers.append(layer)
                    member_masks.append(valid)
                    member_variograms.append(layer_semivariance(layer, valid, lags))
                    member_count += 1
        member_residual_energy = member_residual_sum / member_count
        member_total_energy = member_total_sum / member_count
        _, mean_power = mean_spectrum(list(mean_field), masks_by_day)
        _, member_power = mean_spectrum(member_layers, member_masks)
        spectra["mean_power"][method] = mean_power.tolist()
        spectra["member_power"][method] = member_power.tolist()
        member_variogram = np.nanmean(np.stack(member_variograms), axis=0)
        for lag, value in zip(lags, member_variogram):
            variogram_rows.append({"source": method, "kind": "member_texture",
                                   "lag_cells": lag,
                                   "lag_km": lag * FINE_DEGREES * 111.0,
                                   "semivariance": float(value)})

        high_frequency = wavelength <= factor * FINE_DEGREES * 111.0
        mean_hf_ratio = ratio(float(np.nansum(mean_power[high_frequency])),
                              float(np.nansum(chirps_power[high_frequency])))
        member_hf_ratio = ratio(float(np.nansum(member_power[high_frequency])),
                                float(np.nansum(chirps_power[high_frequency])))
        row = {
            "method": method,
            "footprint_factor": factor,
            "footprint_deg": factor * FINE_DEGREES,
            "subgrid_mean_variance_mm2": mean_residual_energy,
            "subgrid_member_variance_mm2": member_residual_energy,
            "subgrid_mean_variance_ratio_chirps": ratio(mean_residual_energy, chirps_energy),
            "subgrid_member_variance_ratio_chirps": ratio(member_residual_energy, chirps_energy),
            "subgrid_mean_daily_r_chirps": finite_float(np.nanmean(daily_residual_r)),
            "subgrid_gradient_ratio_chirps": ratio(
                gradient_energy(mean_residual, block_valid), chirps_gradient
            ),
            "subgrid_coherent_fraction": ratio(mean_residual_energy, member_residual_energy),
            "subgrid_fraction_member": ratio(member_residual_energy, member_total_energy),
            "subgrid_fraction_mean": ratio(mean_residual_energy, mean_total_energy),
            "high_frequency_mean_power_ratio_chirps": mean_hf_ratio,
            "high_frequency_member_power_ratio_chirps": member_hf_ratio,
            "texture_members_sampled": int(len(selected_member_indices(
                archive["datasets"][0].sizes["member"], texture_members
            ))),
            "confirmatory_days": int(primary.sum()),
        }
        if "background" in methods and method != "background":
            background = archive["mean"][methods.index("background"), primary]
            increment_residual = residual_stack(mean_field - background, factor, block_valid)
            row["subgrid_increment_rms_mm"] = float(np.sqrt(energy(
                increment_residual, block_valid
            )))
        else:
            row["subgrid_increment_rms_mm"] = 0.0 if method == "background" else None
        rows.append(row)

        for scale_factor in factors:
            scale_mask = strict_block_mask(valid, scale_factor, minimum_block_valid)
            value = energy(residual_stack(mean_field, scale_factor, scale_mask), scale_mask)
            scale_rows.append({"source": method, "kind": "ensemble_mean",
                               "factor": scale_factor,
                               "scale_deg": scale_factor * FINE_DEGREES,
                               "residual_variance_mm2": value,
                               "ratio_to_chirps": ratio(
                                   value, chirps_residual_energy_by_factor[scale_factor]
                               )})
    return rows, spectra, variogram_rows + scale_rows


def infer_cv_root(paths: list[Path]) -> Path | None:
    parents = []
    for path in paths:
        if path.parent.name != "gridded":
            return None
        parents.append(path.parent.parent)
    return parents[0] if parents and all(parent == parents[0] for parent in parents) else None


def evaluate_withheld_gauges(paths: list[Path], methods: list[str], factor: int,
                             cv_root: Path | None) -> tuple[list[dict], list[dict]]:
    if cv_root is None:
        return [], []
    point_members = {name: [] for name in methods}
    point_truth = []
    anomalies = {
        name: {reference: {"predicted": [], "observed": []}
               for reference in REFERENCE_NAMES}
        for name in methods
    }
    for path in paths:
        period_dir = cv_root / "cv" / path.stem
        fold_paths = sorted(period_dir.glob("fold[0-4].npz"))
        if len(fold_paths) != 5:
            print(f"[gauges] {period_dir}: need five complete folds; skipping this period")
            continue
        for fold_path in fold_paths:
            dump = np.load(fold_path, allow_pickle=False)
            if dump["variant_names"].astype(str).tolist() != methods:
                raise ValueError(f"{fold_path}: method order differs from Zarr")
            eval_idx = np.asarray(dump["eval_idx"], int)
            primary = confirmatory_daily_mask(dump["times"].astype("datetime64[D]"))
            observed = np.asarray(dump["gauge_mm"][primary][:, eval_idx], float)
            point_truth.append(observed.reshape(-1))
            grid_lat = np.asarray(dump["grid_lat"], float)
            grid_lon = np.asarray(dump["grid_lon"], float)
            station_lat = np.asarray(dump["station_lat"], float)[eval_idx]
            station_lon = np.asarray(dump["station_lon"], float)[eval_idx]
            valid = np.asarray(dump["valid"], bool)
            reference_coarse = {
                "chirps": S.scale_decompose(
                    np.asarray(dump["chirps"][primary], float), factor,
                    np.broadcast_to(valid, dump["chirps"][primary].shape),
                )[0],
                "cpc": S.scale_decompose(
                    np.asarray(dump["condition"][primary], float), factor,
                    np.broadcast_to(valid, dump["condition"][primary].shape),
                )[0],
                "imerg": upsample_coarse(
                    np.asarray(dump["raw_imerg_mm"][primary], float), factor, valid.shape
                ),
            }
            sampled_reference = {
                name: bilinear_sample(field, grid_lat, grid_lon, station_lat, station_lon)
                for name, field in reference_coarse.items()
            }
            for method in methods:
                members = np.asarray(
                    dump[f"station_{method}"][primary][:, :, eval_idx], float
                )
                point_members[method].append(
                    np.moveaxis(members, 1, 2).reshape(-1, members.shape[1])
                )
                fine_mean = np.asarray(dump[f"meanfield_{method}"][primary], float)
                own_coarse = S.scale_decompose(
                    fine_mean, factor, np.broadcast_to(valid, fine_mean.shape)
                )[0]
                own_coarse_at_station = bilinear_sample(
                    own_coarse, grid_lat, grid_lon, station_lat, station_lon
                )
                predicted_anomaly = members.mean(axis=1) - own_coarse_at_station
                for reference in REFERENCE_NAMES:
                    observed_anomaly = observed - sampled_reference[reference]
                    anomalies[method][reference]["predicted"].append(predicted_anomaly.ravel())
                    anomalies[method][reference]["observed"].append(observed_anomaly.ravel())

    if not point_truth:
        return [], []
    truth = np.concatenate(point_truth)
    point_rows, anomaly_rows = [], []
    for method in methods:
        members = np.concatenate(point_members[method])
        point_rows.append({"method": method, **point_metrics(members, truth)})
        for reference in REFERENCE_NAMES:
            predicted = np.concatenate(anomalies[method][reference]["predicted"])
            observed = np.concatenate(anomalies[method][reference]["observed"])
            anomaly_rows.append({
                "method": method,
                "coarse_baseline": reference,
                **anomaly_metrics(predicted, observed),
            })
    return point_rows, anomaly_rows


def aggregate_monthly_matrix(rows: list[dict], methods: list[str]) -> list[dict]:
    output = []
    for method in methods:
        selected = [
            row for row in rows
            if row.get("method") == method and "month" in row
            and row.get("confirmatory", True)
        ]
        if not selected:
            continue
        combined = {"method": method}
        keys = [
            key for key in selected[0]
            if key not in {"month", "method", "confirmatory"}
        ]
        for key in keys:
            values = [row[key] for row in selected if row.get(key) is not None]
            combined[key] = finite_float(np.nanmean(values)) if values else None
        output.append(combined)
    return output


def merge_matrix(methods: list[str], product_rows: list[dict], monthly_rows: list[dict],
                 subgrid_rows: list[dict], point_rows: list[dict],
                 anomaly_rows: list[dict]) -> list[dict]:
    sources = {
        "product": {row["method"]: row for row in product_rows if "month" not in row},
        "monthly": {row["method"]: row for row in aggregate_monthly_matrix(monthly_rows, methods)},
        "subgrid": {row["method"]: row for row in subgrid_rows},
        "point": {row["method"]: row for row in point_rows},
    }
    output = []
    for method in methods:
        row = {"method": method}
        for prefix, mapping in sources.items():
            for key, value in mapping.get(method, {}).items():
                if key != "method":
                    row[f"{prefix}_{key}"] = value
        for anomaly in anomaly_rows:
            if anomaly["method"] != method:
                continue
            reference = anomaly["coarse_baseline"]
            for key, value in anomaly.items():
                if key not in {"method", "coarse_baseline"}:
                    row[f"gauge_anomaly_{key}_{reference}"] = value
        output.append(row)
    return output


def plot_evaluation_matrix(matrix: list[dict], out_dir: Path, sources: list[Path]) -> None:
    plt = use_paper_style()
    methods = [row["method"] for row in matrix]
    labels = [method.replace("v2_simul_s04_", "").replace("v2_", "") for method in methods]
    y = np.arange(len(methods))
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)

    def values(key):
        return np.asarray([row.get(key, np.nan) for row in matrix], float)

    if np.isfinite(values("point_crps_mm")).any():
        axes[0, 0].barh(y, values("point_crps_mm"), color="#C1440E")
        axes[0, 0].set_xlabel("withheld-gauge CRPS (mm/day)")
        axes[0, 0].set_title("A. Independent point value")
    else:
        axes[0, 0].text(0.5, 0.5, "five CV folds not complete", ha="center", va="center")
        axes[0, 0].set_title("A. Independent point value")
    axes[0, 0].set_yticks(y, labels); axes[0, 0].invert_yaxis()

    width = 0.25
    for offset, reference, color in zip((-width, 0, width), REFERENCE_NAMES,
                                        ("#2878B5", "#E09F3E", "#4C956C")):
        axes[0, 1].barh(y + offset, values(f"product_daily_common_r_{reference}"),
                        height=width, label=reference, color=color)
    axes[0, 1].set_xlim(-0.2, 1); axes[0, 1].legend()
    axes[0, 1].set_yticks(y, labels); axes[0, 1].invert_yaxis()
    axes[0, 1].set_xlabel("mean daily spatial r")
    axes[0, 1].set_title("B. Agreement at common 0.4° scale")

    for offset, reference, color in zip((-width, 0, width), REFERENCE_NAMES,
                                        ("#2878B5", "#E09F3E", "#4C956C")):
        axes[0, 2].barh(y + offset, values(f"monthly_monthly_mean_r_{reference}"),
                        height=width, label=reference, color=color)
    axes[0, 2].set_xlim(-0.2, 1); axes[0, 2].set_yticks(y, labels); axes[0, 2].invert_yaxis()
    axes[0, 2].set_xlabel("spatial r"); axes[0, 2].set_title("C. Monthly mean agreement")

    for offset, reference, color in zip((-width, 0, width), REFERENCE_NAMES,
                                        ("#2878B5", "#E09F3E", "#4C956C")):
        axes[1, 0].barh(y + offset, values(f"monthly_monthly_variability_r_{reference}"),
                        height=width, label=reference, color=color)
    axes[1, 0].set_xlim(-0.2, 1); axes[1, 0].set_yticks(y, labels); axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlabel("spatial r"); axes[1, 0].set_title("D. Within-month variability")

    axes[1, 1].barh(y - 0.18, values("subgrid_subgrid_mean_daily_r_chirps"),
                    height=0.35, label="CHIRPS residual r", color="#6A4C93")
    axes[1, 1].barh(y + 0.18, values("subgrid_subgrid_coherent_fraction"),
                    height=0.35, label="coherent fraction", color="#2A9D8F")
    axes[1, 1].set_yticks(y, labels); axes[1, 1].invert_yaxis(); axes[1, 1].legend()
    axes[1, 1].set_title("E. Subgrid placement and coherence")

    if np.isfinite(values("gauge_anomaly_mse_skill_vs_no_subgrid_imerg")).any():
        for offset, reference, color in zip((-width, 0, width), REFERENCE_NAMES,
                                            ("#2878B5", "#E09F3E", "#4C956C")):
            axes[1, 2].barh(
                y + offset,
                values(f"gauge_anomaly_mse_skill_vs_no_subgrid_{reference}"),
                            height=width, label=reference, color=color)
        axes[1, 2].axvline(0, color="black", ls="--", lw=0.8)
        axes[1, 2].set_xlabel("MSE skill vs zero-subgrid null")
    else:
        axes[1, 2].text(0.5, 0.5, "five CV folds not complete", ha="center", va="center")
    axes[1, 2].set_yticks(y, labels); axes[1, 2].invert_yaxis()
    axes[1, 2].set_title("F. Located subgrid evidence")

    figure.suptitle("CPC-v2 real-data evaluation: gauges for value, products for structure")
    save_figure(
        figure, out_dir, "01", "evaluation_matrix", data={"matrix": matrix}, sources=sources,
        caption=(
            "Independent gauge evidence is separated from product agreement. "
            "CHIRPS, IMERG and CPC correlations describe agreement, not truth."
        ),
    )
    plt.close(figure)


def plot_daily_monthly(daily_rows: list[dict], monthly_rows: list[dict],
                       methods: list[str], out_dir: Path, sources: list[Path]) -> None:
    plt = use_paper_style()
    figure, axes = plt.subplots(2, 3, figsize=(15, 7), constrained_layout=True)
    selected_sources = [*methods, *REFERENCE_NAMES]
    colors = {method: plt.cm.tab10(index % 10) for index, method in enumerate(methods)}
    colors.update({"chirps": "black", "imerg": "#E09F3E", "cpc": "#4C956C"})
    styles = {"chirps": "--", "imerg": ":", "cpc": "-."}
    for source in selected_sources:
        selected = [row for row in daily_rows if row["source"] == source]
        axes[0, 0].plot(
            np.asarray([row["date"] for row in selected], dtype="datetime64[D]"),
            [row["domain_mean_mm"] for row in selected],
            label=source, color=colors[source], ls=styles.get(source, "-"), lw=1,
        )
        axes[0, 1].plot(
            np.asarray([row["date"] for row in selected], dtype="datetime64[D]"),
            [row["spatial_std_mm"] for row in selected],
            label=source, color=colors[source], ls=styles.get(source, "-"), lw=1,
        )
        if source in methods:
            axes[0, 2].plot(
                np.asarray([row["date"] for row in selected], dtype="datetime64[D]"),
                [row["posterior_spread_mm"] for row in selected],
                label=source, color=colors[source], lw=1,
            )
        monthly = [row for row in monthly_rows if row["source"] == source]
        month_axis = np.asarray([row["month"] for row in monthly], dtype="datetime64[M]")
        axes[1, 0].plot(month_axis, [row["monthly_domain_mean_mm"] for row in monthly],
                        marker="o", ms=2, label=source, color=colors[source],
                        ls=styles.get(source, "-"))
        axes[1, 1].plot(
            month_axis, [row["within_month_daily_variability_mm"] for row in monthly],
            marker="o", ms=2, label=source, color=colors[source],
            ls=styles.get(source, "-"),
        )
        if source in methods:
            axes[1, 2].plot(
                month_axis, [row["mean_posterior_spread_mm"] for row in monthly],
                marker="o", ms=2, label=source, color=colors[source],
            )
    axes[0, 0].set_title("A. Daily domain mean"); axes[0, 0].set_ylabel("mm/day")
    axes[0, 1].set_title("B. Daily spatial variability"); axes[0, 1].set_ylabel("spatial SD (mm/day)")
    axes[0, 2].set_title("C. Daily posterior spread"); axes[0, 2].set_ylabel("ensemble SD (mm/day)")
    axes[1, 0].set_title("D. Monthly domain mean"); axes[1, 0].set_ylabel("mm/day")
    axes[1, 1].set_title("E. Within-month daily variability"); axes[1, 1].set_ylabel("mean temporal SD (mm/day)")
    axes[1, 2].set_title("F. Monthly mean posterior spread"); axes[1, 2].set_ylabel("ensemble SD (mm/day)")
    axes[0, 0].legend(ncol=3, fontsize=6)
    for axis in axes.ravel():
        axis.tick_params(axis="x", rotation=20)
    save_figure(
        figure, out_dir, "02", "daily_monthly_variability",
        data={"daily": daily_rows, "monthly": monthly_rows}, sources=sources,
        caption="Daily and monthly amount and variability. Product curves are references, not truth.",
    )
    plt.close(figure)


def plot_subgrid_matrix(matrix: list[dict], out_dir: Path, sources: list[Path]) -> None:
    plt = use_paper_style()
    columns = [
        ("subgrid_subgrid_mean_daily_r_chirps", "CHIRPS\nresidual r", "correlation"),
        ("subgrid_subgrid_mean_variance_ratio_chirps", "mean var\n/ CHIRPS", "ratio"),
        ("subgrid_subgrid_member_variance_ratio_chirps", "member var\n/ CHIRPS", "ratio"),
        ("subgrid_high_frequency_member_power_ratio_chirps", "member HF\npower ratio", "ratio"),
        ("subgrid_subgrid_coherent_fraction", "coherent\nfraction", "fraction"),
        ("subgrid_subgrid_fraction_member", "member\nsubgrid fraction", "fraction"),
        ("gauge_anomaly_correlation_imerg", "gauge anomaly r\nIMERG baseline", "correlation"),
        ("gauge_anomaly_correlation_chirps", "gauge anomaly r\nCHIRPS baseline", "correlation"),
        ("gauge_anomaly_correlation_cpc", "gauge anomaly r\nCPC baseline", "correlation"),
        ("gauge_anomaly_mse_skill_vs_no_subgrid_imerg", "zero-subgrid skill\nIMERG baseline", "skill"),
        ("gauge_anomaly_mse_skill_vs_no_subgrid_chirps", "zero-subgrid skill\nCHIRPS baseline", "skill"),
        ("gauge_anomaly_mse_skill_vs_no_subgrid_cpc", "zero-subgrid skill\nCPC baseline", "skill"),
    ]
    raw = np.asarray([[row.get(key, np.nan) for key, _, _ in columns] for row in matrix], float)
    normalized = np.full_like(raw, np.nan)
    for column, (_, _, interpretation) in enumerate(columns):
        values = raw[:, column]
        if interpretation == "correlation":
            normalized[:, column] = np.clip((values + 0.5) / 1.5, 0, 1)
        elif interpretation == "skill":
            normalized[:, column] = np.clip((values + 1.0) / 2.0, 0, 1)
        elif interpretation == "ratio":
            with np.errstate(divide="ignore", invalid="ignore"):
                normalized[:, column] = np.exp(-np.abs(np.log(values)))
        else:
            normalized[:, column] = np.clip(values, 0, 1)
    figure, axis = plt.subplots(figsize=(16, 0.55 * len(matrix) + 2.7))
    image = axis.imshow(normalized, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    for row in range(raw.shape[0]):
        for column in range(raw.shape[1]):
            text = "—" if not np.isfinite(raw[row, column]) else f"{raw[row, column]:.2f}"
            color = "white" if np.isfinite(normalized[row, column]) and normalized[row, column] < 0.45 else "black"
            axis.text(column, row, text, ha="center", va="center", color=color, fontsize=7)
    axis.set_xticks(np.arange(len(columns)), [label for _, label, _ in columns])
    axis.set_yticks(np.arange(len(matrix)), [row["method"] for row in matrix])
    axis.tick_params(axis="x", rotation=25)
    axis.set_title("Subgrid evidence matrix (annotation is raw; ratio colour is closeness to 1)")
    figure.colorbar(image, ax=axis, label="diagnostic agreement index")
    save_figure(
        figure, out_dir, "03", "subgrid_matrix", data={"matrix": matrix}, sources=sources,
        caption=(
            "Subgrid diagnostics. Variance and spectral ratios measure texture, "
            "CHIRPS residual correlation measures product agreement, and withheld-"
            "gauge anomaly correlation is independent located-structure evidence."
        ),
    )
    plt.close(figure)


def plot_subgrid_case(archive: dict, factor: int, out_dir: Path,
                      sources: list[Path]) -> None:
    """Full field and below-footprint residual for the most active CHIRPS day."""
    mask = strict_block_mask(archive["valid"], factor, 1.0)
    primary = confirmatory_daily_mask(archive["time"])
    candidate_indices = np.flatnonzero(primary)
    chirps_residual = residual_stack(archive["chirps"][primary], factor, mask)
    activity = np.nanmean(chirps_residual[:, mask] ** 2, axis=1)
    day_index = int(candidate_indices[np.nanargmax(activity)])
    fields = {
        **{
            method: archive["mean"][index, day_index]
            for index, method in enumerate(archive["methods"])
        },
        "CHIRPS (reference)": archive["chirps"][day_index],
        "IMERG S04 (assimilated)": archive["imerg"][day_index],
        "CPC (conditioning)": archive["cpc"][day_index],
    }
    residuals = {
        name: residual_stack(field[None], factor, mask)[0]
        for name, field in fields.items()
    }
    full_values = np.concatenate([field[archive["valid"]] for field in fields.values()])
    residual_values = np.concatenate([field[mask] for field in residuals.values()])
    full_max = float(np.nanpercentile(full_values, 99))
    residual_max = float(np.nanpercentile(np.abs(residual_values), 99))
    plt = use_paper_style()
    figure, axes = plt.subplots(
        2, len(fields), figsize=(2.25 * len(fields), 5.2), constrained_layout=True,
    )
    table = []
    lon_grid, lat_grid = np.meshgrid(archive["lon"], archive["lat"])
    for column, (name, field) in enumerate(fields.items()):
        first = axes[0, column].imshow(
            np.where(archive["valid"], field, np.nan), origin="lower",
            vmin=0, vmax=full_max, cmap="Blues",
        )
        second = axes[1, column].imshow(
            np.where(mask, residuals[name], np.nan), origin="lower",
            vmin=-residual_max, vmax=residual_max, cmap="RdBu",
        )
        axes[0, column].set_title(name, fontsize=7)
        axes[0, column].set_xticks([]); axes[0, column].set_yticks([])
        axes[1, column].set_xticks([]); axes[1, column].set_yticks([])
        for lat, lon, full, residual in zip(
            lat_grid.ravel(), lon_grid.ravel(), field.ravel(), residuals[name].ravel()
        ):
            table.append({
                "date": str(archive["time"][day_index]), "source": name,
                "lat": float(lat), "lon": float(lon),
                "full_mm": finite_float(full), "subgrid_mm": finite_float(residual),
            })
    axes[0, 0].set_ylabel("full daily field")
    axes[1, 0].set_ylabel(f"residual below {factor * FINE_DEGREES:.1f}°")
    figure.colorbar(first, ax=axes[0, :], shrink=0.75, label="mm/day")
    figure.colorbar(second, ax=axes[1, :], shrink=0.75, label="subgrid residual (mm/day)")
    figure.suptitle(
        f"Most CHIRPS-subgrid-active archived day: {archive['time'][day_index]}"
    )
    save_figure(
        figure, out_dir, "05", "subgrid_case", data={"fields": table}, sources=sources,
        caption=(
            "Full fields and residuals below the 0.4-degree IMERG footprint. "
            "IMERG has zero sub-footprint content by construction; CHIRPS is a "
            "structural reference, not asserted truth."
        ),
    )
    plt.close(figure)


def plot_scale_diagnostics(spectra: dict, curve_rows: list[dict], methods: list[str],
                           out_dir: Path, sources: list[Path]) -> None:
    plt = use_paper_style()
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    wavelength = np.asarray(spectra["wavelength_km"], float)
    mean_power = spectra["mean_power"]
    member_power = spectra["member_power"]
    colors = {method: plt.cm.tab10(index % 10) for index, method in enumerate(methods)}
    axes[0, 0].loglog(wavelength, mean_power["chirps"], color="black", label="CHIRPS")
    for method in methods:
        axes[0, 0].loglog(wavelength, mean_power[method], color=colors[method], label=method)
        ratio_curve = np.divide(
            np.asarray(member_power[method], float), np.asarray(mean_power["chirps"], float),
            out=np.full_like(wavelength, np.nan), where=np.asarray(mean_power["chirps"]) > 0,
        )
        axes[0, 1].semilogx(wavelength, ratio_curve, color=colors[method], label=method)
    axes[0, 0].invert_xaxis(); axes[0, 0].set_title("A. Ensemble-mean spectra")
    axes[0, 0].set_xlabel("wavelength (km)"); axes[0, 0].set_ylabel("power")
    axes[0, 1].invert_xaxis(); axes[0, 1].axhline(1, color="black", ls="--")
    axes[0, 1].axvline(FOOTPRINT_FACTOR * FINE_DEGREES * 111, color="grey", ls=":")
    axes[0, 1].set_title("B. Member texture / CHIRPS power")
    axes[0, 1].set_xlabel("wavelength (km)"); axes[0, 1].set_ylabel("power ratio")

    variograms = [row for row in curve_rows if "lag_cells" in row]
    scales = [row for row in curve_rows if "factor" in row]
    for source in ["chirps", *methods]:
        selected = [row for row in variograms if row["source"] == source]
        axes[1, 0].plot([row["lag_km"] for row in selected],
                        [row["semivariance"] for row in selected], marker="o", ms=2,
                        color="black" if source == "chirps" else colors[source], label=source)
    axes[1, 0].set_title("C. Member texture variogram")
    axes[1, 0].set_xlabel("lag (km)"); axes[1, 0].set_ylabel("semivariance")
    for method in methods:
        selected = [row for row in scales if row["source"] == method]
        axes[1, 1].plot([row["scale_deg"] for row in selected],
                        [row["ratio_to_chirps"] for row in selected], marker="o", ms=3,
                        color=colors[method], label=method)
    axes[1, 1].axhline(1, color="black", ls="--")
    axes[1, 1].axvline(FOOTPRINT_FACTOR * FINE_DEGREES, color="grey", ls=":")
    axes[1, 1].set_title("D. Residual variance below each scale")
    axes[1, 1].set_xlabel("block scale (degrees)"); axes[1, 1].set_ylabel("ratio to CHIRPS")
    axes[0, 0].legend(ncol=2, fontsize=5)
    save_figure(
        figure, out_dir, "04", "scale_diagnostics",
        data={
            "spectra": [
                {"wavelength_km": float(wavelength[index]),
                 **{f"mean_{name}": float(values[index]) for name, values in mean_power.items()},
                 **{f"member_{name}": float(values[index]) for name, values in member_power.items()}}
                for index in range(len(wavelength))
            ],
            "curves": curve_rows,
        },
        sources=sources,
        caption=(
            "Scale-explicit texture diagnostics. Ratios near one mean CHIRPS-like "
            "amplitude, not proof that CHIRPS is true or that features are colocated."
        ),
    )
    plt.close(figure)


def main() -> None:
    args = parse_args()
    paths = [Path(path) for path in args.zarr]
    archive = load_archive(paths, args.factor)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    daily_rows, monthly_rows, product_and_monthly_rows = evaluate_daily_and_monthly(
        archive, args.factor
    )
    product_rows = [row for row in product_and_monthly_rows if "month" not in row]
    monthly_matrix_rows = [row for row in product_and_monthly_rows if "month" in row]
    subgrid_rows, spectra, curve_rows = evaluate_subgrid(
        archive, args.factor, args.texture_members, args.minimum_block_valid
    )
    cv_root = Path(args.cv_root) if args.cv_root else infer_cv_root(paths)
    point_rows, anomaly_rows = evaluate_withheld_gauges(
        paths, archive["methods"], args.factor, cv_root
    )
    matrix = merge_matrix(
        archive["methods"], product_rows, monthly_matrix_rows, subgrid_rows,
        point_rows, anomaly_rows,
    )

    write_rows(out_dir / "daily_domain.csv", daily_rows)
    write_rows(out_dir / "monthly_domain.csv", monthly_rows)
    write_rows(out_dir / "product_matrix.csv", product_rows)
    write_rows(out_dir / "monthly_matrix.csv", monthly_matrix_rows)
    write_rows(out_dir / "subgrid_matrix.csv", subgrid_rows)
    write_rows(out_dir / "withheld_gauge_matrix.csv", point_rows)
    write_rows(out_dir / "withheld_gauge_subgrid_anomalies.csv", anomaly_rows)
    write_rows(out_dir / "evaluation_matrix.csv", matrix)

    payload = {
        "design": {
            "zarr_stores": [str(path) for path in paths],
            "start": str(archive["time"].min()),
            "end": str(archive["time"].max()),
            "days": int(len(archive["time"])),
            "methods": archive["methods"],
            "footprint_factor": args.factor,
            "footprint_degrees": args.factor * FINE_DEGREES,
            "texture_members_sampled": args.texture_members,
            "cv_root": str(cv_root) if cv_root else None,
            "selection_dates_excluded_from_aggregate_daily": [
                str(SELECTION_START), str(SELECTION_END)
            ],
            "selection_month_excluded_from_aggregate_monthly": "2022-05",
        },
        "interpretation": {
            "independent_evidence": (
                "withheld-gauge value scores and withheld-gauge anomalies; absent "
                "until all five CV folds for a period are complete"
            ),
            "product_agreement": (
                "CHIRPS/IMERG/CPC correlations and variability are agreement, not truth"
            ),
            "reference_free_structure": (
                "subgrid variance, spectra, variograms and coherence prove generated "
                "scale content but cannot alone prove correct placement"
            ),
            "chirps_role": (
                "fine-resolution training target and structural reference; explicitly "
                "not assumed to be gridded truth"
            ),
        },
        "evaluation_matrix": matrix,
        "product_matrix": product_rows,
        "monthly_matrix": aggregate_monthly_matrix(monthly_matrix_rows, archive["methods"]),
        "subgrid_matrix": subgrid_rows,
        "withheld_gauge_matrix": point_rows,
        "withheld_gauge_subgrid_anomalies": anomaly_rows,
        "spectra": spectra,
        "scale_curves": curve_rows,
    }
    (out_dir / "evaluation.json").write_text(
        json.dumps(json_ready(payload), indent=2, allow_nan=False) + "\n"
    )
    (out_dir / "README.md").write_text(
        "# Real-data gridded evaluation\n\n"
        f"Evaluated **{len(archive['time'])} days** from {archive['time'].min()} "
        f"through {archive['time'].max()}.\n\n"
        "CHIRPS, IMERG and CPC metrics are product agreement, not truth. "
        "Independent downscaling evidence comes from the five-fold withheld-gauge "
        "scores and sub-footprint anomaly tests when those folds are available. "
        "Spectra, variance and variograms diagnose resolved texture but not its "
        "correct placement.\n"
    )

    plot_evaluation_matrix(matrix, out_dir, paths)
    plot_daily_monthly(daily_rows, monthly_rows, archive["methods"], out_dir, paths)
    plot_subgrid_matrix(matrix, out_dir, paths)
    plot_scale_diagnostics(spectra, curve_rows, archive["methods"], out_dir, paths)
    plot_subgrid_case(archive, args.factor, out_dir, paths)
    for dataset in archive["datasets"]:
        dataset.close()
    print(f"[done] evaluated {len(archive['time'])} days -> {out_dir}")
    print(f"[done] independent gauge folds available: {bool(point_rows)}")


if __name__ == "__main__":
    main()
