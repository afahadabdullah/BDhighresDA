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
2. ``product_agreement``: daily/monthly/May--September spatial correlation and variability
   agreement with CHIRPS, IMERG, and CPC.  These diagnose plausibility and
   observation adherence; they are never labelled skill or truth.
3. ``reference_free_structure``: variance below the 0.4-degree footprint,
   spectral power, variograms, member texture and ensemble coherence.  These
   establish that the model resolves rather than merely upsamples, but texture
   alone cannot prove correct placement.

Production fields are also sampled at gauges that entered the likelihood. That
table is explicitly called ``assimilated_fit`` and is never presented as
independent verification.

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
        help="optional experiment root containing held-out evaluation files; "
            "default is inferred from a .../gridded/<period>.zarr path",
    )
    parser.add_argument(
        "--cv-layout", choices=("five-fold", "single-holdout"), default="five-fold",
        help=(
            "five-fold expects cv/<period>/fold{0..4}.npz; single-holdout expects "
            "evaluation/<period>.npz. The latter is a constrained random 20%% split, "
            "not exhaustive cross-validation."
        ),
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--cpc-source-zarr", default=None,
        help="optional packed checkpoint Zarr used to load same-day original CPC; "
             "default is inferred from each archive's scope.checkpoint_data",
    )
    parser.add_argument("--factor", type=int, default=FOOTPRINT_FACTOR)
    parser.add_argument("--texture-members", type=int, default=5,
                        help="evenly spaced members used for spectra/variograms")
    parser.add_argument("--minimum-block-valid", type=float, default=1.0)
    parser.add_argument(
        "--comparison-zarr", default=None,
        help=(
            "optional completed all-station Zarr from another production run; it is "
            "shown only as a labelled spatial comparison, never as verification truth"
        ),
    )
    parser.add_argument(
        "--comparison-method", default="v2_simul_s04_huber3",
        help="method to extract from --comparison-zarr",
    )
    parser.add_argument(
        "--comparison-label", default="BMD-only Huber3 (prior production)",
        help="label shown for the optional comparison field",
    )
    parser.add_argument(
        "--selection-daily-start", default=None,
        help="first configuration-selection date to exclude from independent daily scores",
    )
    parser.add_argument(
        "--selection-daily-end", default=None,
        help="last configuration-selection date to exclude from independent daily scores",
    )
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
    rmse = float(np.sqrt(np.mean(difference**2)))
    spread = float(np.sqrt(np.mean(np.var(members, axis=1, ddof=1))))
    return {
        "n": int(len(truth)),
        "crps_mm": float(np.mean(fair_crps_per_sample(members, truth))),
        "mae_mm": float(np.mean(np.abs(difference))),
        "dry_mae_mm": float(np.mean(np.abs(difference[~wet]))) if (~wet).any() else None,
        "wet_mae_mm": float(np.mean(np.abs(difference[wet]))) if wet.any() else None,
        "bias_mm": float(np.mean(difference)),
        "rmse_mm": rmse,
        "correlation": finite_float(correlation(mean, truth)),
        "spread_mm": spread,
        "spread_skill_ratio": float(spread / rmse) if rmse > 0 else None,
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


def bilinear_sample_members(
    field: np.ndarray,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    station_lat: np.ndarray,
    station_lon: np.ndarray,
) -> np.ndarray:
    """Bilinearly sample ``(time,member,lat,lon)`` to ``(time,member,station)``.

    Production archives mask ocean cells after sampling.  The diffusion state
    uses zero outside the land mask, so archived NaNs are restored to zero
    before applying the same align-corners bilinear geometry used by the DA
    station operator.
    """
    field = np.asarray(field, float)
    if field.ndim != 4:
        raise ValueError(f"expected time,member,lat,lon; got {field.shape}")
    time, member, nlat, nlon = field.shape
    sampled = bilinear_sample(
        np.nan_to_num(field, nan=0.0).reshape(time * member, nlat, nlon),
        grid_lat, grid_lon, station_lat, station_lon,
    )
    return sampled.reshape(time, member, len(station_lat))


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


def load_spatial_comparison(path: Path, method: str, archive: dict, label: str) -> dict:
    """Load one matched prior-production field for a map-only sensitivity view."""
    dataset = validate_store(path)
    methods = dataset.method.values.astype(str).tolist()
    if method not in methods:
        raise ValueError(f"{path}: comparison method {method!r} not in {methods}")
    if not np.array_equal(dataset.lat.values, archive["lat"]) or not np.array_equal(dataset.lon.values, archive["lon"]):
        raise ValueError(f"{path}: comparison grid differs from the evaluated archive")
    times = np.asarray(dataset.time.values).astype("datetime64[D]")
    if not np.array_equal(times, archive["time"]):
        raise ValueError(
            f"{path}: comparison dates differ; spatial comparison requires an exact matched window"
        )
    values = np.asarray(dataset.ensemble_mean.isel(method=methods.index(method)).values, float)
    scope = dataset.attrs.get("scope", {})
    result = {
        "path": str(path), "method": method, "label": label, "mean": values,
        "scope": scope, "days": int(dataset.sizes["time"]),
    }
    dataset.close()
    return result


def load_same_day_cpc(archive: dict, override: str | None = None) -> tuple[np.ndarray, Path] | None:
    """Load CPC on the gauge/target date from the checkpoint-bound packed store.

    The CPC copied into the production Zarr is the lagged conditioning field.
    This loader goes back to ``scope.checkpoint_data`` and selects the target
    dates directly. If that auditable source is unavailable, CPC is omitted
    from gauge verification rather than silently substituting the lagged input.
    """
    candidates = []
    for dataset in archive["datasets"]:
        scope = dataset.attrs.get("scope", {})
        if isinstance(scope, dict) and scope.get("checkpoint_data"):
            candidates.append(str(scope["checkpoint_data"]))
    if override:
        source_path = Path(override)
    elif candidates and len(set(candidates)) == 1:
        source_path = Path(candidates[0])
    else:
        print("[gauges] same-day CPC source is not uniquely recorded; omitting CPC")
        return None
    if not source_path.is_dir():
        print(f"[gauges] same-day CPC source missing: {source_path}; omitting CPC")
        return None

    import zarr

    root = zarr.open(str(source_path), mode="r")
    channels = [str(name) for name in root.attrs.get("cond_channels", [])]
    if "cpc_precip" not in channels:
        print(f"[gauges] {source_path}: no cpc_precip channel; omitting CPC")
        return None
    cpc_index = channels.index("cpc_precip")
    valid_index = channels.index("cpc_valid") if "cpc_valid" in channels else None
    source_times = np.asarray(
        root["time"][:], dtype="datetime64[ns]"
    ).astype("datetime64[D]")
    time_lookup = {day: index for index, day in enumerate(source_times)}
    missing = [str(day) for day in archive["time"] if day not in time_lookup]
    if missing:
        print(
            f"[gauges] {source_path}: {len(missing)} archive date(s) lack "
            f"same-day CPC ({missing[:5]}); omitting CPC"
        )
        return None
    source_lat = np.asarray(root["lat"][:], float)
    source_lon = np.asarray(root["lon"][:], float)
    lat_index = np.asarray([
        int(np.argmin(np.abs(source_lat - value))) for value in archive["lat"]
    ])
    lon_index = np.asarray([
        int(np.argmin(np.abs(source_lon - value))) for value in archive["lon"]
    ])
    if (
        not np.allclose(source_lat[lat_index], archive["lat"], atol=1.0e-5)
        or not np.allclose(source_lon[lon_index], archive["lon"], atol=1.0e-5)
    ):
        print(
            f"[gauges] {source_path}: CPC grid does not contain the archive grid; "
            "omitting CPC"
        )
        return None
    output = np.empty((len(archive["time"]), *archive["valid"].shape), np.float32)
    for position, day in enumerate(archive["time"]):
        source_position = time_lookup[day]
        layer = np.asarray(root["cond"][source_position, cpc_index], np.float32)
        selected = layer[np.ix_(lat_index, lon_index)]
        if valid_index is not None:
            available = np.asarray(
                root["cond"][source_position, valid_index], bool
            )[np.ix_(lat_index, lon_index)]
            selected = np.where(available, selected, np.nan)
        output[position] = selected
    print(f"[gauges] loaded same-day original CPC from {source_path}")
    return output, source_path


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


def complete_may_sep_years(times: np.ndarray) -> list[int]:
    """Calendar years containing every day from 1 May through 30 September."""
    days = set(np.asarray(times).astype("datetime64[D]").astype(str).tolist())
    years = sorted({int(str(day)[:4]) for day in times})
    complete = []
    for year in years:
        expected = np.arange(
            np.datetime64(f"{year}-05-01", "D"),
            np.datetime64(f"{year}-10-01", "D"),
        ).astype(str)
        if set(expected.tolist()).issubset(days):
            complete.append(year)
    return complete


def _period_field_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict:
    """Average spatial diagnostics over one or more matched gridded periods."""
    candidate = np.asarray(candidate, float)
    reference = np.asarray(reference, float)
    if candidate.ndim == 2:
        candidate, reference = candidate[None], reference[None]
    correlations = daily_spatial_correlation(candidate, reference)
    crmse = daily_centered_rmse(candidate, reference)
    return {
        "r": finite_float(np.nanmean(correlations)),
        "crmse_mm": finite_float(np.nanmean(crmse)),
        "bias_mm": finite_float(np.nanmean(candidate - reference)),
        "variance_ratio": finite_float(ratio(
            float(np.nanmean((candidate - np.nanmean(candidate, axis=(-2, -1), keepdims=True)) ** 2)),
            float(np.nanmean((reference - np.nanmean(reference, axis=(-2, -1), keepdims=True)) ** 2)),
        )),
    }


def evaluate_temporal_scales(archive: dict, factor: int) -> tuple[list[dict], list[dict], dict]:
    """Evaluate gridded mean and temporal variability at daily/monthly/seasonal scales.

    The common-support matrix uses exact 0.4-degree block means for every
    source.  Daily variability is the day-to-day series of domain spatial SD;
    monthly and May--September variability are maps of temporal SD of daily
    precipitation.  The latter two therefore measure weather variability, not
    posterior uncertainty.
    """
    methods = archive["methods"]
    times = archive["time"].astype("datetime64[D]")
    valid = archive["valid"]
    sources = {
        **{method: archive["mean"][index] for index, method in enumerate(methods)},
        **product_fields(archive),
    }
    common = {name: block_stack(field, factor, valid) for name, field in sources.items()}
    primary = confirmatory_daily_mask(times)
    months = times.astype("datetime64[M]")
    years = np.asarray([int(str(day)[:4]) for day in times])
    full_years = complete_may_sep_years(times)
    confirmatory_years = [year for year in full_years if year != 2022]
    seasonal_scope = "confirmatory_full_may_sep"
    if not confirmatory_years:
        # A small single-season archive (and the synthetic unit test) still
        # receives descriptive plots, but it is not mislabeled confirmation.
        confirmatory_years = sorted(set(years[primary].tolist()))
        seasonal_scope = "available_archive_no_complete_confirmatory_may_sep"

    rows = []
    for method in methods:
        for reference in REFERENCE_NAMES:
            daily_mean = _period_field_metrics(
                common[method][primary], common[reference][primary]
            )
            method_sd = np.nanstd(common[method].reshape(len(times), -1), axis=1)[primary]
            reference_sd = np.nanstd(common[reference].reshape(len(times), -1), axis=1)[primary]
            daily_variability = {
                "r": finite_float(correlation(method_sd, reference_sd)),
                "crmse_mm": finite_float(centered_rmse(method_sd, reference_sd)),
                "bias_mm": finite_float(np.nanmean(method_sd - reference_sd)),
                "variance_ratio": finite_float(ratio(
                    float(np.nanmean(method_sd**2)), float(np.nanmean(reference_sd**2))
                )),
            }
            rows.append({
                "method": method, "reference": reference, "scale": "daily",
                "mean_definition": "daily precipitation pattern at 0.4 degree",
                "variability_definition": "daily domain spatial SD time series",
                "n_periods": int(primary.sum()),
                **{f"mean_{key}": value for key, value in daily_mean.items()},
                **{f"variability_{key}": value for key, value in daily_variability.items()},
            })

            month_mean_method, month_mean_reference = [], []
            month_var_method, month_var_reference = [], []
            for month in np.unique(months):
                if str(month) == "2022-05":
                    continue
                choose = months == month
                month_mean_method.append(np.nanmean(common[method][choose], axis=0))
                month_mean_reference.append(np.nanmean(common[reference][choose], axis=0))
                month_var_method.append(np.nanstd(common[method][choose], axis=0))
                month_var_reference.append(np.nanstd(common[reference][choose], axis=0))
            monthly_mean = _period_field_metrics(month_mean_method, month_mean_reference)
            monthly_variability = _period_field_metrics(month_var_method, month_var_reference)
            rows.append({
                "method": method, "reference": reference, "scale": "monthly",
                "mean_definition": "calendar-month mean pattern at 0.4 degree",
                "variability_definition": "within-month temporal SD pattern",
                "n_periods": int(len(month_mean_method)),
                **{f"mean_{key}": value for key, value in monthly_mean.items()},
                **{f"variability_{key}": value for key, value in monthly_variability.items()},
            })

            season_mean_method, season_mean_reference = [], []
            season_var_method, season_var_reference = [], []
            for year in confirmatory_years:
                choose = (years == year) & primary
                if not choose.any():
                    continue
                season_mean_method.append(np.nanmean(common[method][choose], axis=0))
                season_mean_reference.append(np.nanmean(common[reference][choose], axis=0))
                season_var_method.append(np.nanstd(common[method][choose], axis=0))
                season_var_reference.append(np.nanstd(common[reference][choose], axis=0))
            seasonal_mean = _period_field_metrics(season_mean_method, season_mean_reference)
            seasonal_variability = _period_field_metrics(
                season_var_method, season_var_reference
            )
            rows.append({
                "method": method, "reference": reference, "scale": "may_sep",
                "mean_definition": "May-September seasonal mean pattern at 0.4 degree",
                "variability_definition": "within-season daily temporal SD pattern",
                "season_scope": seasonal_scope,
                "years": ",".join(str(year) for year in confirmatory_years),
                "n_periods": int(len(season_mean_method)),
                **{f"mean_{key}": value for key, value in seasonal_mean.items()},
                **{f"variability_{key}": value for key, value in seasonal_variability.items()},
            })

    # Descriptive per-season values retain 2022 but mark its selection leakage.
    daily_stats = {name: spatial_statistics(field, valid) for name, field in sources.items()}
    seasonal_rows = []
    for year in sorted(set(years.tolist())):
        choose = years == year
        label = "may_sep" if year in full_years else (
            "may_jun" if set(int(str(day)[5:7]) for day in times[choose]) <= {5, 6}
            else "available_period"
        )
        for name, field in sources.items():
            mean_field = np.nanmean(field[choose], axis=0)
            variability_field = np.nanstd(field[choose], axis=0)
            seasonal_rows.append({
                "year": year, "period": label, "source": name,
                "start": str(times[choose].min()), "end": str(times[choose].max()),
                "n_days": int(choose.sum()), "complete_may_sep": year in full_years,
                "selection_contaminated": year == 2022,
                "seasonal_domain_mean_mm": float(np.nanmean(mean_field[valid])),
                "seasonal_spatial_std_mm": float(np.nanstd(mean_field[valid])),
                "within_season_daily_variability_mm": float(
                    np.nanmean(variability_field[valid])
                ),
                "daily_domain_mean_variability_mm": float(np.nanstd(
                    daily_stats[name]["domain_mean_mm"][choose]
                )),
                "mean_posterior_spread_mm": (
                    float(np.nanmean(archive["spread"][methods.index(name), choose][:, valid]))
                    if name in methods else None
                ),
            })

    # Calendar-month climatology maps use all confirmatory months available.
    calendar_months = np.arange(5, 10)
    monthly_mean_maps, monthly_variability_maps = {}, {}
    for name, field in sources.items():
        source_means, source_variability = [], []
        for calendar_month in calendar_months:
            means, variability = [], []
            for year in sorted(set(years.tolist())):
                month = np.datetime64(f"{year}-{calendar_month:02d}", "M")
                if str(month) == "2022-05":
                    continue
                choose = months == month
                if choose.any():
                    means.append(np.nanmean(field[choose], axis=0))
                    variability.append(np.nanstd(field[choose], axis=0))
            if means:
                source_means.append(np.nanmean(np.stack(means), axis=0))
                source_variability.append(np.nanmean(np.stack(variability), axis=0))
            else:
                source_means.append(np.full(valid.shape, np.nan))
                source_variability.append(np.full(valid.shape, np.nan))
        monthly_mean_maps[name] = np.stack(source_means)
        monthly_variability_maps[name] = np.stack(source_variability)

    seasonal_mean_maps, seasonal_variability_maps = {}, {}
    for name, field in sources.items():
        means, variability = [], []
        for year in confirmatory_years:
            choose = (years == year) & primary
            if choose.any():
                means.append(np.nanmean(field[choose], axis=0))
                variability.append(np.nanstd(field[choose], axis=0))
        seasonal_mean_maps[name] = np.nanmean(np.stack(means), axis=0)
        seasonal_variability_maps[name] = np.nanmean(np.stack(variability), axis=0)

    grids = {
        "sources": [*methods, *REFERENCE_NAMES],
        "calendar_months": calendar_months,
        "monthly_mean": monthly_mean_maps,
        "monthly_variability": monthly_variability_maps,
        "seasonal_mean": seasonal_mean_maps,
        "seasonal_variability": seasonal_variability_maps,
        "seasonal_scope": seasonal_scope,
        "seasonal_years": confirmatory_years,
    }
    return rows, seasonal_rows, grids


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


def heldout_files(cv_root: Path, period: str, layout: str) -> list[Path]:
    """Return the completed held-out files for a production period.

    The original confirmation archive has five exhaustive folds in a period
    directory.  The combined BMD/BWDB production archive has one explicitly
    constrained random holdout, stored beside its all-station counterpart.
    Keeping both layouts here preserves the full gridded diagnostic suite.
    """
    if layout == "five-fold":
        return sorted((cv_root / "cv" / period).glob("fold[0-4].npz"))
    if layout == "single-holdout":
        path = cv_root / "evaluation" / f"{period}.npz"
        return [path] if path.is_file() else []
    raise ValueError(f"unknown CV layout {layout!r}")


def required_heldout_files(layout: str) -> int:
    return 5 if layout == "five-fold" else 1


def evaluate_withheld_gauges(paths: list[Path], methods: list[str], factor: int,
                             cv_root: Path | None, cv_layout: str) -> tuple[list[dict], list[dict]]:
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
        fold_paths = heldout_files(cv_root, path.stem, cv_layout)
        if len(fold_paths) != required_heldout_files(cv_layout):
            print(f"[gauges] {path.stem}: incomplete {cv_layout} held-out set; skipping this period")
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


def load_withheld_gauge_bundle(
    paths: list[Path], methods: list[str], factor: int, cv_root: Path | None,
    cv_layout: str,
) -> dict | None:
    """Collect every independently withheld station/day exactly once."""
    if cv_root is None:
        return None
    dates, stations, station_lats, station_lons, truth = [], [], [], [], []
    members = {method: [] for method in methods}
    products = {reference: [] for reference in REFERENCE_NAMES}
    for path in paths:
        fold_paths = heldout_files(cv_root, path.stem, cv_layout)
        if len(fold_paths) != required_heldout_files(cv_layout):
            continue
        for fold_path in fold_paths:
            with np.load(fold_path, allow_pickle=False) as dump:
                if dump["variant_names"].astype(str).tolist() != methods:
                    raise ValueError(f"{fold_path}: method order differs from Zarr")
                eval_idx = np.asarray(dump["eval_idx"], int)
                fold_dates = dump["times"].astype("datetime64[D]")
                fold_stations = dump["station_ids"].astype(str)[eval_idx]
                observed = np.asarray(dump["gauge_mm"][:, eval_idx], float)
                dates.append(np.repeat(fold_dates, len(eval_idx)))
                stations.append(np.tile(fold_stations, len(fold_dates)))
                station_lats.append(np.tile(
                    np.asarray(dump["station_lat"], float)[eval_idx], len(fold_dates)
                ))
                station_lons.append(np.tile(
                    np.asarray(dump["station_lon"], float)[eval_idx], len(fold_dates)
                ))
                truth.append(observed.reshape(-1))
                for method in methods:
                    values = np.asarray(dump[f"station_{method}"][:, :, eval_idx], float)
                    members[method].append(
                        np.moveaxis(values, 1, 2).reshape(-1, values.shape[1])
                    )
                grid_lat = np.asarray(dump["grid_lat"], float)
                grid_lon = np.asarray(dump["grid_lon"], float)
                station_lat = np.asarray(dump["station_lat"], float)[eval_idx]
                station_lon = np.asarray(dump["station_lon"], float)[eval_idx]
                valid = np.asarray(dump["valid"], bool)
                product_fields_at_scale = {
                    "chirps": np.asarray(dump["chirps"], float),
                    "cpc": np.asarray(dump["condition"], float),
                    "imerg": upsample_coarse(
                        np.asarray(dump["raw_imerg_mm"], float), factor, valid.shape
                    ),
                }
                for reference, field in product_fields_at_scale.items():
                    products[reference].append(bilinear_sample(
                        np.nan_to_num(field, nan=0.0), grid_lat, grid_lon,
                        station_lat, station_lon,
                    ).reshape(-1))
    if not truth:
        return None
    bundle = {
        "date": np.concatenate(dates),
        "station": np.concatenate(stations),
        "station_lat": np.concatenate(station_lats),
        "station_lon": np.concatenate(station_lons),
        "truth": np.concatenate(truth),
        "members": {method: np.concatenate(parts) for method, parts in members.items()},
        "products": {
            reference: np.concatenate(parts) for reference, parts in products.items()
        },
    }
    keys = np.asarray([
        f"{date}|{station}" for date, station in zip(bundle["date"], bundle["station"])
    ])
    if len(keys) != len(np.unique(keys)):
        raise ValueError("a withheld date/station pair appears in more than one fold")
    return bundle


def load_assimilated_gauge_bundle(archive: dict) -> dict:
    """Sample all-station production fields at gauges that entered the likelihood."""
    methods = archive["methods"]
    dates, stations, truth = [], [], []
    members = {method: [] for method in methods}
    for path, dataset in zip(archive["paths"], archive["datasets"]):
        store_dates = np.asarray(dataset.time.values).astype("datetime64[D]")
        station_ids = dataset.station_id.values.astype(str)
        station_lat = np.asarray(dataset.station_lat.values, float)
        station_lon = np.asarray(dataset.station_lon.values, float)
        observed = np.asarray(dataset.gauge.values, float)
        dates.append(np.repeat(store_dates, len(station_ids)))
        stations.append(np.tile(station_ids, len(store_dates)))
        truth.append(observed.reshape(-1))
        for method_index, method in enumerate(methods):
            print(f"[gauges] sampling assimilated fit: {path.stem} / {method}")
            # ``method`` is a data variable in schema v1 rather than an xarray
            # index coordinate, so positional selection is intentional here.
            field = np.asarray(
                dataset.precipitation.isel(method=method_index).values, float
            )
            sampled = bilinear_sample_members(
                field, archive["lat"], archive["lon"], station_lat, station_lon
            )
            members[method].append(
                np.moveaxis(sampled, 1, 2).reshape(-1, sampled.shape[1])
            )
            del field, sampled
    return {
        "date": np.concatenate(dates),
        "station": np.concatenate(stations),
        "truth": np.concatenate(truth),
        "members": {method: np.concatenate(parts) for method, parts in members.items()},
    }


def attach_same_day_cpc_to_withheld_bundle(
    bundle: dict | None, archive: dict, cpc: np.ndarray
) -> None:
    """Sample a matched same-day CPC grid for every withheld station record."""
    if bundle is None:
        return
    time_lookup = {day: index for index, day in enumerate(archive["time"])}
    sampled = np.full(len(bundle["date"]), np.nan, float)
    for day in np.unique(bundle["date"]):
        if day not in time_lookup:
            continue
        choose = bundle["date"] == day
        sampled[choose] = bilinear_sample(
            np.nan_to_num(cpc[time_lookup[day]][None], nan=0.0),
            archive["lat"], archive["lon"],
            bundle["station_lat"][choose], bundle["station_lon"][choose],
        )[0]
    bundle["products"]["cpc_same_day"] = sampled


def aggregate_gauge_samples(
    bundle: dict, methods: list[str], scale: str
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, dict]]:
    """Aggregate daily station ensembles to monthly or May--September means."""
    dates = bundle["date"].astype("datetime64[D]")
    stations = bundle["station"].astype(str)
    if scale == "monthly":
        labels = dates.astype("datetime64[M]")
        allowed = labels != np.datetime64("2022-05", "M")
        period_labels = np.unique(labels[allowed])
        group_name = lambda label: str(label)
    elif scale == "may_sep":
        years = np.asarray([int(str(day)[:4]) for day in dates])
        full_years = complete_may_sep_years(np.unique(dates))
        selected_years = [year for year in full_years if year != 2022]
        if not selected_years:
            selected_years = sorted(set(years[confirmatory_daily_mask(dates)].tolist()))
        labels = years
        allowed = np.isin(years, selected_years) & confirmatory_daily_mask(dates)
        period_labels = np.asarray(selected_years)
        group_name = lambda label: str(int(label))
    else:
        raise ValueError(f"unsupported gauge aggregation scale: {scale}")

    output_truth = []
    output_members = {method: [] for method in methods}
    variability_truth = []
    variability_predicted = {method: [] for method in methods}
    groups = []
    for label in period_labels:
        in_period = allowed & (labels == label)
        requested_days = len(np.unique(dates[in_period]))
        required = int(np.ceil(0.8 * requested_days))
        for station in np.unique(stations[in_period]):
            choose = in_period & (stations == station)
            finite = np.isfinite(bundle["truth"][choose])
            for method in methods:
                finite &= np.all(np.isfinite(bundle["members"][method][choose]), axis=1)
            if finite.sum() < required or required == 0:
                continue
            observed = bundle["truth"][choose][finite]
            output_truth.append(float(np.mean(observed)))
            variability_truth.append(float(np.std(observed)))
            groups.append(f"{group_name(label)}|{station}")
            for method in methods:
                values = bundle["members"][method][choose][finite]
                output_members[method].append(np.mean(values, axis=0))
                variability_predicted[method].append(float(np.std(values.mean(axis=1))))
    return (
        np.asarray(output_truth, float),
        {method: np.asarray(values, float) for method, values in output_members.items()},
        {
            "group": np.asarray(groups),
            "truth": np.asarray(variability_truth, float),
            "predicted": {
                method: np.asarray(values, float)
                for method, values in variability_predicted.items()
            },
        },
    )


def evaluate_gauge_scales(
    bundle: dict | None, methods: list[str], evaluation_type: str
) -> list[dict]:
    """Daily/monthly/seasonal gauge value and temporal-variability metrics."""
    if bundle is None:
        return []
    rows = []
    primary = confirmatory_daily_mask(bundle["date"])
    full_years = complete_may_sep_years(np.unique(bundle["date"]))
    confirmatory_years = [year for year in full_years if year != 2022]
    gauge_season_scope = "confirmatory_full_may_sep"
    if not confirmatory_years:
        confirmatory_years = sorted({
            int(str(day)[:4]) for day in bundle["date"][primary]
        })
        gauge_season_scope = "available_archive_no_complete_confirmatory_may_sep"
    for method in methods:
        rows.append({
            "evaluation": evaluation_type,
            "independent": evaluation_type == "withheld",
            "scale": "daily", "method": method,
            "temporal_variability_definition": None,
            **point_metrics(bundle["members"][method][primary], bundle["truth"][primary]),
        })
    for scale in ("monthly", "may_sep"):
        truth, members, variability = aggregate_gauge_samples(bundle, methods, scale)
        for method in methods:
            predicted_variability = variability["predicted"][method]
            variability_truth = variability["truth"]
            variability_keep = (
                np.isfinite(predicted_variability) & np.isfinite(variability_truth)
            )
            rows.append({
                "evaluation": evaluation_type,
                "independent": evaluation_type == "withheld",
                "scale": scale, "method": method,
                "season_scope": gauge_season_scope if scale == "may_sep" else None,
                "years": (
                    ",".join(str(year) for year in confirmatory_years)
                    if scale == "may_sep" else None
                ),
                "temporal_variability_definition": (
                    "SD of daily ensemble mean versus SD of daily gauge values "
                    "within each station-period"
                ),
                **point_metrics(members[method], truth),
                "n_variability_groups": int(variability_keep.sum()),
                "temporal_variability_correlation": finite_float(correlation(
                    predicted_variability[variability_keep],
                    variability_truth[variability_keep],
                )),
                "temporal_variability_bias_mm": finite_float(np.nanmean(
                    predicted_variability[variability_keep]
                    - variability_truth[variability_keep]
                )),
                "temporal_variability_ratio": finite_float(ratio(
                    float(np.nanmean(predicted_variability[variability_keep] ** 2)),
                    float(np.nanmean(variability_truth[variability_keep] ** 2)),
                )),
            })
    return rows


def deterministic_metrics(predicted: np.ndarray, observed: np.ndarray) -> dict:
    """Matched deterministic scores for a product or an ensemble mean."""
    predicted = np.asarray(predicted, float)
    observed = np.asarray(observed, float)
    keep = np.isfinite(predicted) & np.isfinite(observed)
    if not keep.any():
        return {"n": 0}
    predicted, observed = predicted[keep], observed[keep]
    difference = predicted - observed
    return {
        "n": int(len(observed)),
        "correlation": finite_float(correlation(predicted, observed)),
        "rmse_mm": float(np.sqrt(np.mean(difference**2))),
        "mae_mm": float(np.mean(np.abs(difference))),
        "bias_mm": float(np.mean(difference)),
    }


def evaluate_long_term_withheld_products(
    bundle: dict | None, methods: list[str]
) -> tuple[list[dict], list[dict]]:
    """Compare all model and reference products against withheld BMD gauges.

    ``pooled_daily`` weights every finite station-day equally. ``station_time``
    first scores the daily time series at each station and then summarizes the
    station scores. ``long_term_mean`` averages each station over all archived
    confirmatory dates and evaluates the 38-station spatial climatology.
    """
    if bundle is None:
        return [], []
    primary = confirmatory_daily_mask(bundle["date"])
    observed = np.asarray(bundle["truth"], float)[primary]
    dates = bundle["date"][primary]
    stations = bundle["station"][primary].astype(str)
    predictions = {
        **{
            method: np.asarray(bundle["members"][method], float)[primary].mean(axis=1)
            for method in methods
        },
        "chirps": np.asarray(bundle["products"]["chirps"], float)[primary],
        "imerg": np.asarray(bundle["products"]["imerg"], float)[primary],
    }
    if "cpc_same_day" in bundle["products"]:
        predictions["cpc"] = np.asarray(
            bundle["products"]["cpc_same_day"], float
        )[primary]
    common = np.isfinite(observed)
    for predicted in predictions.values():
        common &= np.isfinite(predicted)
    if not common.any():
        print("[gauges] no common finite withheld sample across products; skipping comparison")
        return [], []
    observed, dates, stations = observed[common], dates[common], stations[common]
    predictions = {
        source: predicted[common] for source, predicted in predictions.items()
    }
    rows, station_rows = [], []
    for source, predicted in predictions.items():
        pooled = deterministic_metrics(predicted, observed)
        long_term_observed, long_term_predicted = [], []
        for station in np.unique(stations):
            choose = stations == station
            scores = deterministic_metrics(predicted[choose], observed[choose])
            keep = (
                np.isfinite(predicted[choose]) & np.isfinite(observed[choose])
            )
            station_observed_mean = (
                float(np.mean(observed[choose][keep])) if keep.any() else None
            )
            station_predicted_mean = (
                float(np.mean(predicted[choose][keep])) if keep.any() else None
            )
            station_rows.append({
                "source": source,
                "source_type": "analysis" if source in methods else "product",
                "station_id": station,
                "long_term_observed_mm": station_observed_mean,
                "long_term_predicted_mm": station_predicted_mean,
                **scores,
            })
            if keep.any():
                long_term_predicted.append(station_predicted_mean)
                long_term_observed.append(station_observed_mean)
        station_selected = [row for row in station_rows if row["source"] == source]
        station_correlations = np.asarray([
            row.get("correlation", np.nan) for row in station_selected
        ], float)
        finite_station_r = station_correlations[np.isfinite(station_correlations)]
        fisher_mean_r = (
            float(np.tanh(np.mean(np.arctanh(np.clip(
                finite_station_r, -0.999999, 0.999999
            ))))) if finite_station_r.size else None
        )
        long_term = deterministic_metrics(long_term_predicted, long_term_observed)
        rows.append({
            "source": source,
            "source_type": "analysis" if source in methods else "product",
            "archive_start": str(dates.min()), "archive_end": str(dates.max()),
            "archive_days": int(len(np.unique(dates))),
            "seasonal_sampling": "May-Sep 2021-2023 plus May-Jun 2024",
            "selection_dates_excluded": "2022-05-01..2022-05-10",
            "matched_sample_across_all_sources": True,
            "cpc_timing": "same-day original CPC" if source == "cpc" else None,
            "n_stations": int(len(np.unique(stations))),
            **{f"pooled_daily_{key}": value for key, value in pooled.items()},
            "mean_station_temporal_correlation": finite_float(
                np.nanmean(station_correlations)
            ),
            "fisher_mean_station_temporal_correlation": fisher_mean_r,
            "median_station_temporal_correlation": finite_float(
                np.nanmedian(station_correlations)
            ),
            **{f"long_term_station_mean_{key}": value for key, value in long_term.items()},
        })
    return rows, station_rows


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
        axes[0, 0].text(0.5, 0.5, "held-out evaluation unavailable", ha="center", va="center")
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
        axes[1, 2].text(0.5, 0.5, "held-out evaluation unavailable", ha="center", va="center")
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


def save_temporal_grids(grids: dict, archive: dict, out_dir: Path) -> tuple[Path, list[dict]]:
    """Ship compact gridded arrays behind the monthly and seasonal map figures."""
    path = out_dir / "temporal_mean_variability_grids.npz"
    arrays = {
        "lat": archive["lat"], "lon": archive["lon"],
        "calendar_month": grids["calendar_months"],
        "seasonal_year": np.asarray(grids["seasonal_years"], int),
        "seasonal_scope": np.asarray(grids["seasonal_scope"]),
    }
    summary = []
    valid = archive["valid"]
    for source in grids["sources"]:
        for kind in (
            "monthly_mean", "monthly_variability",
            "seasonal_mean", "seasonal_variability",
        ):
            values = np.asarray(grids[kind][source], np.float32)
            arrays[f"{kind}__{source}"] = values
            if values.ndim == 2:
                summary.append({
                    "source": source, "field": kind, "calendar_month": None,
                    "domain_mean_mm": float(np.nanmean(values[valid])),
                    "spatial_std_mm": float(np.nanstd(values[valid])),
                })
            else:
                for index, month in enumerate(grids["calendar_months"]):
                    selected = values[index][valid]
                    finite = selected[np.isfinite(selected)]
                    summary.append({
                        "source": source, "field": kind,
                        "calendar_month": int(month),
                        "domain_mean_mm": float(np.mean(finite)) if finite.size else None,
                        "spatial_std_mm": float(np.std(finite)) if finite.size else None,
                    })
    np.savez_compressed(path, **arrays)
    return path, summary


def _source_label(name: str) -> str:
    return name.replace("v2_simul_s04_", "").replace("v2_", "")


def plot_monthly_grid_maps(
    grids: dict, archive: dict, kind: str, number: str, out_dir: Path,
    source_paths: list[Path], grid_path: Path, summary_rows: list[dict],
) -> None:
    plt = use_paper_style()
    sources = grids["sources"]
    months = grids["calendar_months"]
    values = np.concatenate([
        np.asarray(grids[kind][source], float).ravel() for source in sources
    ])
    values = values[np.isfinite(values)]
    vmax = float(np.percentile(values, 98)) if values.size else 1.0
    figure, axes = plt.subplots(
        len(sources), len(months), figsize=(12, 1.35 * len(sources) + 1.0),
        constrained_layout=True, squeeze=False,
    )
    image = None
    extent = [archive["lon"][0], archive["lon"][-1], archive["lat"][0], archive["lat"][-1]]
    month_names = ("May", "Jun", "Jul", "Aug", "Sep")
    for row, source in enumerate(sources):
        for column, month in enumerate(months):
            axis = axes[row, column]
            image = axis.imshow(
                grids[kind][source][column], origin="lower", extent=extent,
                vmin=0, vmax=vmax, cmap="YlGnBu", aspect="auto",
            )
            axis.grid(False)
            if row == 0:
                axis.set_title(month_names[column])
            if column == 0:
                axis.set_ylabel(_source_label(source))
            axis.set_xticks([]); axis.set_yticks([])
    label = "monthly mean precipitation (mm/day)" if kind == "monthly_mean" else (
        "within-month temporal SD (mm/day)"
    )
    figure.colorbar(image, ax=axes, label=label, shrink=0.75)
    title = "Calendar-month mean fields" if kind == "monthly_mean" else (
        "Calendar-month daily variability fields"
    )
    figure.suptitle(f"{title} (selection month May 2022 excluded)")
    selected_summary = [row for row in summary_rows if row["field"] == kind]
    save_figure(
        figure, out_dir, number, kind, data={"map_summary": selected_summary},
        sources=[*source_paths, grid_path],
        caption=(
            f"{title}. Complete gridded values are in {grid_path.name}; "
            "products are agreement references rather than truth."
        ),
    )
    plt.close(figure)


def plot_seasonal_grid_maps(
    grids: dict, archive: dict, out_dir: Path, source_paths: list[Path],
    grid_path: Path, summary_rows: list[dict],
) -> None:
    plt = use_paper_style()
    sources = grids["sources"]
    all_mean = np.concatenate([
        np.asarray(grids["seasonal_mean"][source], float).ravel() for source in sources
    ])
    all_variability = np.concatenate([
        np.asarray(grids["seasonal_variability"][source], float).ravel()
        for source in sources
    ])
    mean_vmax = float(np.nanpercentile(all_mean, 98))
    variability_vmax = float(np.nanpercentile(all_variability, 98))
    figure, axes = plt.subplots(
        2, len(sources), figsize=(2.0 * len(sources), 5.2),
        constrained_layout=True, squeeze=False,
    )
    extent = [archive["lon"][0], archive["lon"][-1], archive["lat"][0], archive["lat"][-1]]
    images = [None, None]
    for column, source in enumerate(sources):
        for row, (kind, vmax) in enumerate((
            ("seasonal_mean", mean_vmax),
            ("seasonal_variability", variability_vmax),
        )):
            axis = axes[row, column]
            images[row] = axis.imshow(
                grids[kind][source], origin="lower", extent=extent,
                vmin=0, vmax=vmax, cmap="YlGnBu", aspect="auto",
            )
            axis.grid(False); axis.set_xticks([]); axis.set_yticks([])
            if row == 0:
                axis.set_title(_source_label(source), fontsize=7)
        if column == 0:
            axes[0, column].set_ylabel("May–Sep mean")
            axes[1, column].set_ylabel("daily temporal SD")
    figure.colorbar(images[0], ax=axes[0], label="mm/day", shrink=0.75)
    figure.colorbar(images[1], ax=axes[1], label="mm/day", shrink=0.75)
    years = ", ".join(str(year) for year in grids["seasonal_years"])
    scope = grids["seasonal_scope"].replace("_", " ")
    figure.suptitle(
        f"May–September gridded mean and variability ({years}; {scope})"
    )
    selected_summary = [
        row for row in summary_rows if row["field"].startswith("seasonal_")
    ]
    save_figure(
        figure, out_dir, "08", "may_sep_gridded_mean_variability",
        data={"map_summary": selected_summary}, sources=[*source_paths, grid_path],
        caption=(
            "Seasonal mean and within-season daily temporal standard deviation. "
            f"Complete grids are in {grid_path.name}; 2022 is excluded from the "
            "confirmatory seasonal climatology because it contains selection dates."
        ),
    )
    plt.close(figure)


def plot_temporal_scale_matrix(
    rows: list[dict], out_dir: Path, source_paths: list[Path]
) -> None:
    plt = use_paper_style()
    methods = list(dict.fromkeys(row["method"] for row in rows))
    pairs = [(method, reference) for method in methods for reference in REFERENCE_NAMES]
    scales = ("daily", "monthly", "may_sep")
    lookup = {(row["method"], row["reference"], row["scale"]): row for row in rows}
    figure, axes = plt.subplots(1, 2, figsize=(10, 0.42 * len(pairs) + 2.2), constrained_layout=True)
    for axis, key, title in (
        (axes[0], "mean_r", "A. Mean-pattern agreement"),
        (axes[1], "variability_r", "B. Variability agreement"),
    ):
        matrix = np.asarray([
            [lookup[(method, reference, scale)].get(key, np.nan) for scale in scales]
            for method, reference in pairs
        ], float)
        image = axis.imshow(matrix, aspect="auto", vmin=-0.2, vmax=1, cmap="viridis")
        for row_index in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row_index, column]
                axis.text(
                    column, row_index, "—" if not np.isfinite(value) else f"{value:.2f}",
                    ha="center", va="center",
                    color="white" if np.isfinite(value) and value < 0.45 else "black",
                )
        axis.set_xticks(np.arange(3), ("daily", "monthly", "May–Sep"))
        axis.set_yticks(
            np.arange(len(pairs)),
            [f"{_source_label(method)} vs {reference}" for method, reference in pairs],
            fontsize=6,
        )
        axis.set_title(title); axis.grid(False)
    figure.colorbar(image, ax=axes, label="correlation (agreement, not skill)", shrink=0.7)
    figure.suptitle("Gridded mean and variability agreement at common 0.4° support")
    save_figure(
        figure, out_dir, "09", "temporal_scale_grid_matrix",
        data={"matrix": rows}, sources=source_paths,
        caption=(
            "Daily variability is the time series of domain spatial SD; monthly "
            "and seasonal variability are spatial patterns of within-period temporal SD."
        ),
    )
    plt.close(figure)


def plot_gauge_scale_matrix(
    rows: list[dict], methods: list[str], out_dir: Path, source_paths: list[Path]
) -> None:
    plt = use_paper_style()
    evaluations = ("withheld", "assimilated_fit")
    scales = ("daily", "monthly", "may_sep")
    metrics = (
        ("crps_mm", "CRPS"), ("mae_mm", "MAE"), ("bias_mm", "bias"),
        ("correlation", "r"), ("coverage_90", "cov90"),
        ("temporal_variability_correlation", "var r"),
    )
    lookup = {
        (row["evaluation"], row["scale"], row["method"]): row for row in rows
    }
    figure, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for row_index, evaluation in enumerate(evaluations):
        for column, scale in enumerate(scales):
            axis = axes[row_index, column]
            selected = [lookup.get((evaluation, scale, method), {}) for method in methods]
            raw = np.asarray([
                [entry.get(key, np.nan) for key, _ in metrics] for entry in selected
            ], float)
            score = np.full_like(raw, np.nan)
            for metric_index, (key, _) in enumerate(metrics):
                values = raw[:, metric_index]
                finite = np.isfinite(values)
                if not finite.any():
                    continue
                if key in {"crps_mm", "mae_mm"}:
                    low, high = np.nanmin(values), np.nanmax(values)
                    score[finite, metric_index] = (
                        1.0 if high == low else (high - values[finite]) / (high - low)
                    )
                elif key == "bias_mm":
                    magnitude = np.abs(values)
                    high = np.nanmax(magnitude)
                    score[finite, metric_index] = (
                        1.0 if high == 0 else 1.0 - magnitude[finite] / high
                    )
                elif key == "coverage_90":
                    score[finite, metric_index] = np.clip(
                        1.0 - np.abs(values[finite] - 0.9) / 0.9, 0, 1
                    )
                else:
                    score[finite, metric_index] = np.clip((values[finite] + 0.2) / 1.2, 0, 1)
            axis.imshow(score, aspect="auto", vmin=0, vmax=1, cmap="viridis")
            for method_index in range(len(methods)):
                for metric_index in range(len(metrics)):
                    value = raw[method_index, metric_index]
                    axis.text(
                        metric_index, method_index,
                        "—" if not np.isfinite(value) else f"{value:.2f}",
                        ha="center", va="center", fontsize=6,
                        color=("white" if np.isfinite(score[method_index, metric_index])
                               and score[method_index, metric_index] < 0.45 else "black"),
                    )
            axis.set_xticks(np.arange(len(metrics)), [label for _, label in metrics], rotation=20)
            axis.set_yticks(np.arange(len(methods)), [_source_label(method) for method in methods])
            axis.grid(False)
            evidence = "independent withheld" if evaluation == "withheld" else "assimilated fit"
            scale_label = "May–Sep" if scale == "may_sep" else scale
            axis.set_title(f"{evidence}: {scale_label}")
    figure.suptitle(
        "BMD gauge evaluation by temporal scale (annotations raw; colour ranks within panel)"
    )
    save_figure(
        figure, out_dir, "10", "withheld_assimilated_gauge_matrix",
        data={"gauge_matrix": rows}, sources=source_paths,
        caption=(
            "Withheld rows are independent verification. Assimilated-fit rows use "
            "gauges that entered the production likelihood and diagnose fit only."
        ),
    )
    plt.close(figure)


def plot_long_term_withheld_product_comparison(
    rows: list[dict], station_rows: list[dict], out_dir: Path,
    source_paths: list[Path],
) -> None:
    if not rows:
        return
    plt = use_paper_style()
    sources = [row["source"] for row in rows]
    labels = [_source_label(source) for source in sources]
    y = np.arange(len(rows))
    colors = [
        ("#2878B5" if row["source"] == "chirps" else
         "#E09F3E" if row["source"] == "imerg" else
         "#4C956C" if row["source"] == "cpc" else
         plt.cm.tab10(index % 10))
        for index, row in enumerate(rows)
    ]
    figure, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)

    def values(key):
        return np.asarray([row.get(key, np.nan) for row in rows], float)

    panels = (
        (axes[0, 0], "pooled_daily_correlation", "A. Pooled daily correlation", "r"),
        (axes[0, 1], "fisher_mean_station_temporal_correlation", "B. Fisher-mean station temporal correlation", "mean r"),
        (axes[0, 2], "pooled_daily_rmse_mm", "C. Pooled daily RMSE", "mm/day"),
        (axes[1, 0], "pooled_daily_bias_mm", "D. Pooled daily bias", "mm/day"),
        (axes[1, 1], "long_term_station_mean_correlation", "E. Spatial correlation of station means", "r"),
    )
    for axis, key, title, xlabel in panels:
        axis.barh(y, values(key), color=colors)
        if "bias" in key:
            axis.axvline(0, color="black", lw=0.8)
        if "correlation" in key:
            axis.set_xlim(-0.2, 1)
        axis.set_yticks(y, labels); axis.invert_yaxis()
        axis.set_title(title); axis.set_xlabel(xlabel)

    width = 0.38
    axes[1, 2].barh(
        y - width / 2, values("long_term_station_mean_rmse_mm"),
        height=width, label="RMSE", color="#457B9D",
    )
    axes[1, 2].barh(
        y + width / 2, values("long_term_station_mean_bias_mm"),
        height=width, label="bias", color="#D1495B",
    )
    axes[1, 2].axvline(0, color="black", lw=0.8)
    axes[1, 2].set_yticks(y, labels); axes[1, 2].invert_yaxis()
    axes[1, 2].set_title("F. Long-term station-mean error")
    axes[1, 2].set_xlabel("mm/day"); axes[1, 2].legend()
    figure.suptitle(
        "All products against independently withheld BMD gauges, May 2021–June 2024"
    )
    save_figure(
        figure, out_dir, "11", "long_term_withheld_product_comparison",
        data={"summary": rows, "station_scores": station_rows}, sources=source_paths,
        caption=(
            "Available archive days are May--September 2021--2023 and May--June "
            "2024, not a continuous four-year record. The ten May 2022 selection "
            "days are excluded. All targets are independently withheld gauges."
        ),
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


def plot_prior_production_spatial_comparison(
    archive: dict, comparison: dict, factor: int, out_dir: Path, sources: list[Path]
) -> None:
    """Map the current combined-network Huber3 field against prior BMD-only Huber3."""
    mask = strict_block_mask(archive["valid"], factor, 1.0)
    primary = confirmatory_daily_mask(archive["time"])
    candidates = np.flatnonzero(primary)
    chirps_residual = residual_stack(archive["chirps"][primary], factor, mask)
    day_index = int(candidates[np.nanargmax(np.nanmean(chirps_residual[:, mask] ** 2, axis=1))])
    if "v2_simul_s04_huber3" not in archive["methods"]:
        raise ValueError("evaluated archive lacks v2_simul_s04_huber3 for spatial comparison")
    current = archive["mean"][archive["methods"].index("v2_simul_s04_huber3"), day_index]
    previous = comparison["mean"][day_index]
    current_residual = residual_stack(current[None], factor, mask)[0]
    previous_residual = residual_stack(previous[None], factor, mask)[0]
    difference = current - previous
    residual_difference = current_residual - previous_residual
    panels = {
        "Combined BMD+BWDB Huber3": (current, current_residual),
        comparison["label"]: (previous, previous_residual),
        "Combined − BMD-only": (difference, residual_difference),
    }
    full_values = np.concatenate([values[archive["valid"]] for values, _ in panels.values()])
    residual_values = np.concatenate([values[mask] for _, values in panels.values()])
    full_max = float(np.nanpercentile(full_values, 99))
    difference_max = float(np.nanpercentile(np.abs(difference[archive["valid"]]), 99))
    residual_max = float(np.nanpercentile(np.abs(residual_values), 99))
    plt = use_paper_style()
    figure, axes = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True)
    table = []
    mean_image = difference_image = residual_image = None
    for column, (name, (field, residual)) in enumerate(panels.items()):
        if column < 2:
            top = axes[0, column].imshow(np.where(archive["valid"], field, np.nan), origin="lower", vmin=0, vmax=full_max, cmap="Blues")
            mean_image = top
        else:
            top = axes[0, column].imshow(np.where(archive["valid"], field, np.nan), origin="lower", vmin=-difference_max, vmax=difference_max, cmap="RdBu")
            difference_image = top
        bottom = axes[1, column].imshow(np.where(mask, residual, np.nan), origin="lower", vmin=-residual_max, vmax=residual_max, cmap="RdBu")
        residual_image = bottom
        axes[0, column].set_title(name, fontsize=8)
        for axis in axes[:, column]:
            axis.set_xticks([]); axis.set_yticks([]); axis.grid(False)
        table.extend({
            "date": str(archive["time"][day_index]), "panel": name,
            "lat_index": int(row), "lon_index": int(column_index),
            "full_mm": finite_float(field[row, column_index]),
            "subgrid_mm": finite_float(residual[row, column_index]),
        } for row in range(field.shape[0]) for column_index in range(field.shape[1]))
    axes[0, 0].set_ylabel("daily ensemble mean")
    axes[1, 0].set_ylabel("below-0.4° residual")
    figure.colorbar(mean_image, ax=axes[0, :2], shrink=0.75, label="mm/day")
    figure.colorbar(difference_image, ax=axes[0, 2], shrink=0.75, label="difference (mm/day)")
    figure.colorbar(residual_image, ax=axes[1, :], shrink=0.75, label="subgrid residual (mm/day)")
    figure.suptitle(f"Spatial sensitivity to adding BWDB gauges: {archive['time'][day_index]}")
    save_figure(
        figure, out_dir, "12", "prior_bmd_only_spatial_comparison", data={"fields": table},
        sources=sources,
        caption=(
            "Matched Huber3 production fields from the combined BMD+BWDB and earlier BMD-only runs. "
            "This is a spatial sensitivity comparison, not independent verification and not a truth map."
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
    global SELECTION_START, SELECTION_END
    if (args.selection_daily_start is None) != (args.selection_daily_end is None):
        raise ValueError("set both --selection-daily-start and --selection-daily-end, or neither")
    if args.selection_daily_start is not None:
        SELECTION_START = np.datetime64(args.selection_daily_start, "D")
        SELECTION_END = np.datetime64(args.selection_daily_end, "D")
        if SELECTION_END < SELECTION_START:
            raise ValueError("--selection-daily-end precedes --selection-daily-start")
    paths = [Path(path) for path in args.zarr]
    archive = load_archive(paths, args.factor)
    comparison = (
        load_spatial_comparison(
            Path(args.comparison_zarr), args.comparison_method, archive, args.comparison_label
        )
        if args.comparison_zarr else None
    )
    same_day_cpc_result = load_same_day_cpc(archive, args.cpc_source_zarr)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    daily_rows, monthly_rows, product_and_monthly_rows = evaluate_daily_and_monthly(
        archive, args.factor
    )
    temporal_scale_rows, seasonal_rows, temporal_grids = evaluate_temporal_scales(
        archive, args.factor
    )
    product_rows = [row for row in product_and_monthly_rows if "month" not in row]
    monthly_matrix_rows = [row for row in product_and_monthly_rows if "month" in row]
    subgrid_rows, spectra, curve_rows = evaluate_subgrid(
        archive, args.factor, args.texture_members, args.minimum_block_valid
    )
    cv_root = Path(args.cv_root) if args.cv_root else infer_cv_root(paths)
    point_rows, anomaly_rows = evaluate_withheld_gauges(
        paths, archive["methods"], args.factor, cv_root, args.cv_layout
    )
    withheld_bundle = load_withheld_gauge_bundle(
        paths, archive["methods"], args.factor, cv_root, args.cv_layout
    )
    if same_day_cpc_result is not None:
        attach_same_day_cpc_to_withheld_bundle(
            withheld_bundle, archive, same_day_cpc_result[0]
        )
    assimilated_bundle = load_assimilated_gauge_bundle(archive)
    gauge_scale_rows = [
        *evaluate_gauge_scales(withheld_bundle, archive["methods"], "withheld"),
        *evaluate_gauge_scales(
            assimilated_bundle, archive["methods"], "assimilated_fit"
        ),
    ]
    long_term_product_rows, long_term_station_rows = (
        evaluate_long_term_withheld_products(withheld_bundle, archive["methods"])
    )
    matrix = merge_matrix(
        archive["methods"], product_rows, monthly_matrix_rows, subgrid_rows,
        point_rows, anomaly_rows,
    )

    write_rows(out_dir / "daily_domain.csv", daily_rows)
    write_rows(out_dir / "monthly_domain.csv", monthly_rows)
    write_rows(out_dir / "seasonal_domain.csv", seasonal_rows)
    write_rows(out_dir / "temporal_scale_grid_matrix.csv", temporal_scale_rows)
    write_rows(out_dir / "product_matrix.csv", product_rows)
    write_rows(out_dir / "monthly_matrix.csv", monthly_matrix_rows)
    write_rows(out_dir / "subgrid_matrix.csv", subgrid_rows)
    write_rows(out_dir / "withheld_gauge_matrix.csv", point_rows)
    write_rows(out_dir / "withheld_gauge_subgrid_anomalies.csv", anomaly_rows)
    write_rows(out_dir / "gauge_temporal_scale_matrix.csv", gauge_scale_rows)
    write_rows(
        out_dir / "long_term_withheld_product_matrix.csv", long_term_product_rows
    )
    write_rows(
        out_dir / "long_term_withheld_station_scores.csv", long_term_station_rows
    )
    write_rows(out_dir / "evaluation_matrix.csv", matrix)
    temporal_grid_path, temporal_grid_summary = save_temporal_grids(
        temporal_grids, archive, out_dir
    )
    write_rows(out_dir / "temporal_mean_variability_grid_summary.csv", temporal_grid_summary)

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
            "cv_layout": args.cv_layout,
            "selection_dates_excluded_from_aggregate_daily": [
                str(SELECTION_START), str(SELECTION_END)
            ],
            "selection_month_excluded_from_aggregate_monthly": "2022-05",
            "selection_season_excluded_from_aggregate_may_sep": 2022,
            "complete_may_sep_years": complete_may_sep_years(archive["time"]),
            "confirmatory_may_sep_years": temporal_grids["seasonal_years"],
            "same_day_cpc_source": (
                str(same_day_cpc_result[1]) if same_day_cpc_result else None
            ),
            "spatial_comparison": (
                {key: value for key, value in comparison.items() if key != "mean"}
                if comparison else None
            ),
        },
        "interpretation": {
            "independent_evidence": (
                "withheld-gauge value scores and withheld-gauge anomalies; absent "
                f"until all required {args.cv_layout} held-out files are complete"
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
            "variability": (
                "daily variability is domain spatial SD; monthly and seasonal "
                "variability are temporal SD of daily ensemble-mean precipitation. "
                "Posterior ensemble spread is reported separately"
            ),
            "assimilated_gauge_fit": (
                "production gauges entered the likelihood and diagnose fit only; "
                "they are not independent verification"
            ),
            "long_term_product_comparison": (
                "all analysis means, CHIRPS, IMERG and (when its recorded source "
                "is available) same-day original CPC target independently withheld "
                "BMD gauges; lagged archived CPC is never substituted. The archive "
                "has seasonal gaps and is not a continuous May 2021 to June 2024 record"
            ),
        },
        "evaluation_matrix": matrix,
        "product_matrix": product_rows,
        "monthly_matrix": aggregate_monthly_matrix(monthly_matrix_rows, archive["methods"]),
        "temporal_scale_grid_matrix": temporal_scale_rows,
        "seasonal_domain": seasonal_rows,
        "subgrid_matrix": subgrid_rows,
        "withheld_gauge_matrix": point_rows,
        "withheld_gauge_subgrid_anomalies": anomaly_rows,
        "gauge_temporal_scale_matrix": gauge_scale_rows,
        "long_term_withheld_product_matrix": long_term_product_rows,
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
        "Independent downscaling evidence comes from the configured withheld-gauge "
        "scores and sub-footprint anomaly tests when those folds are available. "
        "Production-station scores are explicitly labelled assimilated fit, not "
        "verification. Daily, monthly and May–September temporal variability is "
        "kept separate from posterior ensemble spread. "
        "The long-term product comparison targets withheld BMD gauges and loads "
        "same-day CPC from the recorded source when available; lagged CPC is omitted. "
        "Spectra, variance and variograms diagnose resolved texture but not its "
        "correct placement.\n"
    )

    plot_evaluation_matrix(matrix, out_dir, paths)
    plot_daily_monthly(daily_rows, monthly_rows, archive["methods"], out_dir, paths)
    plot_subgrid_matrix(matrix, out_dir, paths)
    plot_scale_diagnostics(spectra, curve_rows, archive["methods"], out_dir, paths)
    plot_subgrid_case(archive, args.factor, out_dir, paths)
    if comparison:
        plot_prior_production_spatial_comparison(
            archive, comparison, args.factor, out_dir,
            [*paths, Path(comparison["path"])],
        )
    plot_monthly_grid_maps(
        temporal_grids, archive, "monthly_mean", "06", out_dir,
        paths, temporal_grid_path, temporal_grid_summary,
    )
    plot_monthly_grid_maps(
        temporal_grids, archive, "monthly_variability", "07", out_dir,
        paths, temporal_grid_path, temporal_grid_summary,
    )
    plot_seasonal_grid_maps(
        temporal_grids, archive, out_dir, paths,
        temporal_grid_path, temporal_grid_summary,
    )
    plot_temporal_scale_matrix(temporal_scale_rows, out_dir, paths)
    plot_gauge_scale_matrix(gauge_scale_rows, archive["methods"], out_dir, paths)
    plot_long_term_withheld_product_comparison(
        long_term_product_rows, long_term_station_rows, out_dir,
        [*paths, *([same_day_cpc_result[1]] if same_day_cpc_result else [])],
    )
    for dataset in archive["datasets"]:
        dataset.close()
    print(f"[done] evaluated {len(archive['time'])} days -> {out_dir}")
    print(f"[done] independent gauge folds available: {bool(point_rows)}")


if __name__ == "__main__":
    main()
