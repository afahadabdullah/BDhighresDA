#!/usr/bin/env python
"""Evaluate the saved BRISHTI-05 analysis over Bangladesh for the June contract.

The analysis is read from completed legacy-lineage all-station production Zarr
stores. It does not rerun DA. The 2023 contract is evaluated day by day and
as a monthly field; every other complete June in the supplied stores forms a
leave-2023-out climatology.  All spatial calculations and maps use the supplied
Bangladesh ADM0 polygon, intersected with the model-valid mask.

Evidence is deliberately labelled by role: production BMD gauges diagnose
assimilated fit, IMERG diagnoses assimilated-product adherence, CHIRPS is a
learned fine-grid structural reference, and CPC is loaded on the same target
product day from its original native archive. No gridded product is designated
as truth.
"""

from __future__ import annotations

import argparse
import calendar
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


METHOD_DEFAULT = "v2_simul_s04_ig010"
PRODUCT_NAME = "BRISHTI-05"
PRODUCT_FULL_NAME = (
    "Bangladesh Rainfall Integration of Satellite, Hydrometeorological, and "
    "Terrestrial Information at 0.05°"
)
TARGET_YEAR = 2023
TARGET_MONTH = 6
WET_MM = 1.0
REFERENCE_ORDER = ("imerg_native_0p1", "chirps", "cpc_native_0p5")
SOURCE_LABELS = {
    "analysis": PRODUCT_NAME,
    "imerg_native_0p1": "IMERG native 0.1°",
    "chirps": "CHIRPS 0.05°",
    "cpc_native_0p5": "CPC original 0.5°",
    "bmd": "BMD gauges",
}
EVIDENCE_ROLES = {
    "analysis": f"{PRODUCT_NAME} saved all-station posterior analysis",
    "imerg_native_0p1": (
        "native 0.1-degree version of the product assimilated after S04 coarsening; "
        "0.1-degree detail was not assimilated, but product family is not independent"
    ),
    "chirps": "training-target analysis; structural comparison only",
    "cpc_native_0p5": (
        "original NOAA CPC Global Unified 0.5-degree product-day gauge analysis; "
        "comparison analysis only"
    ),
    "bmd": "all stations entered likelihood; assimilated fit, not independent verification",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--zarr", nargs="+", required=True)
    parser.add_argument("--boundary-geojson", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--method", default=METHOD_DEFAULT)
    parser.add_argument("--year", type=int, default=TARGET_YEAR)
    parser.add_argument("--month", type=int, default=TARGET_MONTH)
    parser.add_argument(
        "--months", nargs="+", type=int, default=None,
        help="one or more contiguous calendar months; overrides --month (e.g. 5 6 7 8)",
    )
    parser.add_argument("--factor", type=int, default=8,
                        help="assimilated IMERG S04 factor, retained as experiment metadata")
    parser.add_argument(
        "--native-imerg", nargs="+", required=True,
        help="prepared native 0.1-degree IMERG netCDFs covering every requested June",
    )
    parser.add_argument(
        "--cpc-dir", default="data/raw/cpc",
        help="directory containing original NOAA precip.YYYY.nc CPC 0.5-degree files",
    )
    return parser.parse_args()


def evaluation_months(args: argparse.Namespace) -> tuple[int, ...]:
    months = tuple(args.months) if args.months is not None else (args.month,)
    if not months or any(month < 1 or month > 12 for month in months):
        raise ValueError(f"months must be in 1..12, got {months}")
    if tuple(sorted(set(months))) != months:
        raise ValueError(f"months must be sorted and unique, got {months}")
    if any(right != left + 1 for left, right in zip(months, months[1:])):
        raise ValueError(f"months must be contiguous, got {months}")
    return months


def period_label(months: tuple[int, ...]) -> str:
    names = [calendar.month_name[month] for month in months]
    return names[0] if len(names) == 1 else f"{names[0]}–{names[-1]}"


def period_tag(months: tuple[int, ...]) -> str:
    return "_".join(calendar.month_abbr[month].lower() for month in months)


def expected_period_days(year: int, months: tuple[int, ...]) -> int:
    return sum(calendar.monthrange(year, month)[1] for month in months)


def load_shared_evaluator():
    path = Path(__file__).with_name("55_evaluate_v2_gridded_archive.py")
    spec = importlib.util.spec_from_file_location("bdhires_v2_archive_eval", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def background_day_offset(archive: dict) -> int:
    offsets = []
    for dataset in archive["datasets"]:
        scope = dataset.attrs.get("scope", {})
        if not isinstance(scope, dict) or "background_day_offset" not in scope:
            raise ValueError("every production Zarr must record scope.background_day_offset")
        offsets.append(int(scope["background_day_offset"]))
    if len(set(offsets)) != 1:
        raise ValueError(f"production Zarr stores have different background offsets: {offsets}")
    return offsets[0]


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def json_ready(value):
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


def geometry_from_geojson(payload: dict) -> dict:
    kind = payload.get("type")
    if kind == "FeatureCollection":
        features = payload.get("features", [])
        if len(features) != 1:
            raise ValueError("Bangladesh ADM0 GeoJSON must contain exactly one feature")
        return geometry_from_geojson(features[0])
    if kind == "Feature":
        return payload["geometry"]
    if kind not in {"Polygon", "MultiPolygon"}:
        raise ValueError(f"expected Polygon or MultiPolygon, got {kind}")
    return payload


def polygon_parts(geometry: dict) -> list[list[np.ndarray]]:
    coordinates = geometry["coordinates"]
    polygons = [coordinates] if geometry["type"] == "Polygon" else coordinates
    return [[np.asarray(ring, float) for ring in polygon] for polygon in polygons]


def country_mask(lat: np.ndarray, lon: np.ndarray, geometry: dict) -> np.ndarray:
    from matplotlib.path import Path as MplPath

    xx, yy = np.meshgrid(np.asarray(lon, float), np.asarray(lat, float))
    points = np.column_stack([xx.ravel(), yy.ravel()])
    inside = np.zeros(len(points), bool)
    for polygon in polygon_parts(geometry):
        if not polygon:
            continue
        part = MplPath(polygon[0]).contains_points(points, radius=1.0e-9)
        for hole in polygon[1:]:
            part &= ~MplPath(hole).contains_points(points, radius=1.0e-9)
        inside |= part
    return inside.reshape(xx.shape)


def points_in_country(lat: np.ndarray, lon: np.ndarray, geometry: dict) -> np.ndarray:
    """Test paired station coordinates, with a small tolerance for border gauges."""
    from matplotlib.path import Path as MplPath

    points = np.column_stack([np.asarray(lon, float), np.asarray(lat, float)])
    inside = np.zeros(len(points), bool)
    for polygon in polygon_parts(geometry):
        part = MplPath(polygon[0]).contains_points(points, radius=0.02)
        for hole in polygon[1:]:
            part &= ~MplPath(hole).contains_points(points, radius=-0.02)
        inside |= part
    return inside


def boundary_bounds(geometry: dict) -> tuple[float, float, float, float]:
    points = np.concatenate([ring for polygon in polygon_parts(geometry) for ring in polygon])
    return float(points[:, 0].min()), float(points[:, 0].max()), float(points[:, 1].min()), float(points[:, 1].max())


def _ascending_grid(values: np.ndarray, coordinate: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    coordinate = np.asarray(coordinate, float)
    if coordinate[0] <= coordinate[-1]:
        return values, coordinate
    return np.flip(values, axis=axis), coordinate[::-1]


def _validate_resolution(name: str, coordinate: np.ndarray, expected: float) -> None:
    spacing = np.diff(np.asarray(coordinate, float))
    if spacing.size == 0 or not np.allclose(spacing, expected, atol=2.0e-4):
        raise ValueError(
            f"{name} is not on the expected {expected:g}-degree regular grid; "
            f"median spacing={np.median(spacing) if spacing.size else np.nan}"
        )


def load_native_imerg(paths: list[Path], wanted: np.ndarray) -> dict:
    """Load exact BMD-window IMERG by its 03 UTC end-date label."""
    import xarray as xr

    by_day: dict[np.datetime64, np.ndarray] = {}
    grid_lat = grid_lon = None
    provenance = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        with xr.open_dataset(path) as dataset:
            required = {"precipitation", "time", "lat", "lon"}
            if not required.issubset(set(dataset.variables) | set(dataset.coords)):
                raise ValueError(f"{path}: native IMERG requires {sorted(required)}")
            end_hour = dataset.attrs.get("bmd_accumulation_end_hour_utc")
            if end_hour is None or int(end_hour) != 3:
                raise ValueError(f"{path}: expected BMD-window end hour 03 UTC, got {end_hour}")
            units = str(dataset.precipitation.attrs.get("units", "")).lower().replace(" ", "")
            if units not in {"mm/day", "mmday-1", "mmd-1", "mmday^-1", "mmd^-1"}:
                raise ValueError(f"{path}: IMERG precipitation units are {units!r}, not mm/day")
            values = np.asarray(dataset.precipitation.values, float)
            lat = np.asarray(dataset.lat.values, float)
            lon = np.asarray(dataset.lon.values, float)
            values, lat = _ascending_grid(values, lat, -2)
            values, lon = _ascending_grid(values, lon, -1)
            _validate_resolution(str(path), lat, 0.1)
            _validate_resolution(str(path), lon, 0.1)
            if grid_lat is None:
                grid_lat, grid_lon = lat, lon
            elif not np.allclose(lat, grid_lat) or not np.allclose(lon, grid_lon):
                raise ValueError(f"{path}: native IMERG grid differs from earlier files")
            days = np.asarray(dataset.time.values).astype("datetime64[D]")
            for index, day in enumerate(days):
                if day in by_day:
                    raise ValueError(f"native IMERG date {day} appears more than once")
                by_day[day] = values[index]
        provenance.append(str(path))
    missing = [str(day) for day in wanted if day not in by_day]
    if missing:
        raise ValueError(f"native IMERG lacks {len(missing)} requested dates: {missing[:8]}")
    return {
        "values": np.stack([by_day[day] for day in wanted]),
        "lat": grid_lat, "lon": grid_lon, "resolution_degrees": 0.1,
        "paths": provenance,
        "timing": "BMD 03:00-03:00 UTC accumulation window",
    }


def load_native_cpc(cpc_dir: Path, wanted: np.ndarray, bounds) -> dict:
    """Load original NOAA CPC 0.5-degree cells without fine-grid interpolation."""
    import xarray as xr

    years = sorted({int(str(day)[:4]) for day in wanted})
    by_day: dict[np.datetime64, np.ndarray] = {}
    grid_lat = grid_lon = None
    paths = []
    lon_min, lon_max, lat_min, lat_max = bounds
    for year in years:
        path = cpc_dir / f"precip.{year}.nc"
        if not path.is_file():
            raise FileNotFoundError(
                f"missing original CPC file {path}; run scripts/02b_download_cpc.py "
                f"--start {min(years)} --end {max(years)} --out {cpc_dir}"
            )
        with xr.open_dataset(path) as dataset:
            if "precip" not in dataset:
                raise ValueError(f"{path}: missing original CPC variable 'precip'")
            field = dataset["precip"]
            lat_name = "lat" if "lat" in field.coords else "latitude"
            lon_name = "lon" if "lon" in field.coords else "longitude"
            lat = np.asarray(field[lat_name].values, float)
            lon = np.asarray(field[lon_name].values, float)
            lat_keep = (lat >= lat_min - 0.75) & (lat <= lat_max + 0.75)
            lon_keep = (lon >= lon_min - 0.75) & (lon <= lon_max + 0.75)
            values = np.asarray(field.isel({lat_name: np.flatnonzero(lat_keep),
                                            lon_name: np.flatnonzero(lon_keep)}).values, float)
            lat = lat[lat_keep]
            lon = lon[lon_keep]
            values, lat = _ascending_grid(values, lat, -2)
            values, lon = _ascending_grid(values, lon, -1)
            _validate_resolution(str(path), lat, 0.5)
            _validate_resolution(str(path), lon, 0.5)
            values = np.where(np.isfinite(values) & (values >= 0) & (values <= 1000),
                              values, np.nan)
            if grid_lat is None:
                grid_lat, grid_lon = lat, lon
            elif not np.allclose(lat, grid_lat) or not np.allclose(lon, grid_lon):
                raise ValueError(f"{path}: original CPC regional grid differs")
            days = np.asarray(field.time.values).astype("datetime64[D]")
            for index, day in enumerate(days):
                if day in by_day:
                    raise ValueError(f"original CPC date {day} appears more than once")
                by_day[day] = values[index]
        paths.append(str(path))
    missing = [str(day) for day in wanted if day not in by_day]
    if missing:
        raise ValueError(f"original CPC lacks {len(missing)} requested dates: {missing[:8]}")
    return {
        "values": np.stack([by_day[day] for day in wanted]),
        "lat": grid_lat, "lon": grid_lon, "resolution_degrees": 0.5,
        "paths": paths,
        "timing": (
            "original CPC calendar-day accumulation; approximately 3 hours offset "
            "from the BMD/IMERG 03 UTC window"
        ),
    }


def target_cell_edges(centres: np.ndarray) -> np.ndarray:
    centres = np.asarray(centres, float)
    if len(centres) < 2:
        raise ValueError("at least two target centres are required")
    middle = 0.5 * (centres[:-1] + centres[1:])
    return np.r_[centres[0] - (middle[0] - centres[0]), middle,
                 centres[-1] + (centres[-1] - middle[-1])]


def regrid_cell_average(values: np.ndarray, source_lat: np.ndarray, source_lon: np.ndarray,
                        target_lat: np.ndarray, target_lon: np.ndarray,
                        source_mask: np.ndarray | None = None) -> np.ndarray:
    """Area-weighted centre-bin average onto actual target product cells."""
    values = np.asarray(values, float)
    lat_edges, lon_edges = target_cell_edges(target_lat), target_cell_edges(target_lon)
    ilat = np.digitize(source_lat, lat_edges) - 1
    ilon = np.digitize(source_lon, lon_edges) - 1
    lat_ok = (ilat >= 0) & (ilat < len(target_lat))
    lon_ok = (ilon >= 0) & (ilon < len(target_lon))
    yy, xx = np.meshgrid(np.arange(len(source_lat)), np.arange(len(source_lon)), indexing="ij")
    keep = lat_ok[:, None] & lon_ok[None, :]
    if source_mask is not None:
        keep &= np.asarray(source_mask, bool)
    source_y, source_x = yy[keep], xx[keep]
    target_y, target_x = ilat[source_y], ilon[source_x]
    weights = np.cos(np.deg2rad(np.asarray(source_lat)[source_y]))
    flat = values[..., source_y, source_x]
    output = np.full((*values.shape[:-2], len(target_lat), len(target_lon)), np.nan)
    for target_row in range(len(target_lat)):
        for target_col in range(len(target_lon)):
            choose = (target_y == target_row) & (target_x == target_col)
            if not choose.any():
                continue
            selected = flat[..., choose]
            finite = np.isfinite(selected)
            weighted = np.where(finite, selected * weights[choose], 0.0).sum(axis=-1)
            denominator = np.where(finite, weights[choose], 0.0).sum(axis=-1)
            output[..., target_row, target_col] = np.divide(
                weighted, denominator, out=np.full(weighted.shape, np.nan),
                where=denominator > 0,
            )
    return output


def correlation(first: np.ndarray, second: np.ndarray) -> float:
    first, second = np.asarray(first, float).ravel(), np.asarray(second, float).ravel()
    keep = np.isfinite(first) & np.isfinite(second)
    if keep.sum() < 3 or np.std(first[keep]) == 0 or np.std(second[keep]) == 0:
        return float("nan")
    return float(np.corrcoef(first[keep], second[keep])[0, 1])


def deterministic_metrics(predicted: np.ndarray, observed: np.ndarray) -> dict:
    predicted, observed = np.asarray(predicted, float), np.asarray(observed, float)
    keep = np.isfinite(predicted) & np.isfinite(observed)
    if not keep.any():
        return {"n": 0}
    predicted, observed = predicted[keep], observed[keep]
    difference = predicted - observed
    return {
        "n": int(keep.sum()),
        "correlation": correlation(predicted, observed),
        "mae_mm": float(np.mean(np.abs(difference))),
        "bias_mm": float(np.mean(difference)),
        "rmse_mm": float(np.sqrt(np.mean(difference**2))),
    }


def field_metrics(candidate: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> dict:
    candidate = np.asarray(candidate, float)[mask]
    reference = np.asarray(reference, float)[mask]
    keep = np.isfinite(candidate) & np.isfinite(reference)
    if not keep.any():
        return {"n_cells": 0}
    candidate, reference = candidate[keep], reference[keep]
    difference = candidate - reference
    centered = (candidate - candidate.mean()) - (reference - reference.mean())
    return {
        "n_cells": int(keep.sum()),
        "correlation": correlation(candidate, reference),
        "bias_mm": float(np.mean(difference)),
        "rmse_mm": float(np.sqrt(np.mean(difference**2))),
        "centered_rmse_mm": float(np.sqrt(np.mean(centered**2))),
        "spatial_sd_ratio": float(np.std(candidate) / np.std(reference)) if np.std(reference) else float("nan"),
    }


def coarse_block_mean(field: np.ndarray, factor: int, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    field = np.asarray(field, float)
    ny, nx = field.shape[-2:]
    if ny % factor or nx % factor:
        raise ValueError(f"grid {(ny, nx)} is not divisible by factor {factor}")
    lead = field.shape[:-2]
    work = np.where(mask, field, np.nan).reshape(*lead, ny // factor, factor, nx // factor, factor)
    finite = np.isfinite(work)
    count = finite.sum(axis=(-3, -1))
    total = np.where(finite, work, 0.0).sum(axis=(-3, -1))
    coarse = np.divide(total, count, out=np.full(total.shape, np.nan), where=count > 0)
    fraction = mask.reshape(ny // factor, factor, nx // factor, factor).mean(axis=(1, 3))
    valid = fraction >= 0.5
    return np.where(valid, coarse, np.nan), valid


def masked_temporal_stat(field: np.ndarray, mask: np.ndarray, statistic: str) -> np.ndarray:
    """Calculate a time statistic only on requested country cells."""
    field = np.asarray(field, float)
    output = np.full(mask.shape, np.nan, float)
    values = field[:, mask]
    if statistic == "mean":
        output[mask] = np.nanmean(values, axis=0)
    elif statistic == "std":
        output[mask] = np.nanstd(values, axis=0)
    else:
        raise ValueError(f"unsupported statistic {statistic}")
    return output


def masked_field_average(fields: list[np.ndarray], mask: np.ndarray) -> np.ndarray:
    output = np.full(mask.shape, np.nan, float)
    output[mask] = np.nanmean(np.stack(fields)[:, mask], axis=0)
    return output


def spatial_daily_rows(times: np.ndarray, fields: dict[str, np.ndarray], masks: dict[str, np.ndarray],
                       spread: np.ndarray) -> list[dict]:
    rows = []
    for index, day in enumerate(times):
        for source, field in fields.items():
            values = np.asarray(field[index], float)[masks[source]]
            finite = values[np.isfinite(values)]
            rows.append({
                "date": str(day), "source": source,
                "domain_mean_mm": float(np.mean(finite)),
                "spatial_sd_mm": float(np.std(finite)),
                "wet_area_fraction": float(np.mean(finite >= WET_MM)),
                "spatial_p95_mm": float(np.percentile(finite, 95)),
                "posterior_spread_mm": (
                    float(np.nanmean(np.asarray(spread[index], float)[masks["analysis"]]))
                    if source == "analysis" else None
                ),
            })
    return rows


def load_target_members(dataset, method_index: int, dates: np.ndarray) -> np.ndarray:
    store_dates = np.asarray(dataset.time.values).astype("datetime64[D]")
    positions = [int(np.where(store_dates == day)[0][0]) for day in dates]
    return np.asarray(
        dataset.precipitation.isel(method=method_index, time=positions).values, float
    )


def load_station_bundle(dataset, method_index: int, dates: np.ndarray, members: np.ndarray,
                        archive: dict, products: dict[str, dict], shared) -> dict:
    store_dates = np.asarray(dataset.time.values).astype("datetime64[D]")
    positions = [int(np.where(store_dates == day)[0][0]) for day in dates]
    station_id = dataset.station_id.values.astype(str)
    station_lat = np.asarray(dataset.station_lat.values, float)
    station_lon = np.asarray(dataset.station_lon.values, float)
    observed = np.asarray(dataset.gauge.isel(time=positions).values, float)
    sampled_members = shared.bilinear_sample_members(
        members, archive["lat"], archive["lon"], station_lat, station_lon
    )
    sampled_products = {
        source: shared.bilinear_sample(
            np.nan_to_num(product["values"], nan=0.0),
            product["lat"], product["lon"],
            station_lat, station_lon,
        ) for source, product in products.items()
    }
    return {
        "station_id": station_id, "station_lat": station_lat, "station_lon": station_lon,
        "observed": observed, "members": sampled_members,
        "products": sampled_products,
    }


def gauge_evaluation(bundle: dict, shared) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    observed = bundle["observed"]
    members = bundle["members"]
    ensemble_mean = members.mean(axis=1)
    sample_members = np.moveaxis(members, 1, 2).reshape(-1, members.shape[1])
    flat_observed = observed.reshape(-1)
    rows = [{
        "source": "analysis", "source_type": "ensemble_analysis",
        "evaluation": "assimilated_fit", "independent": False,
        **shared.point_metrics(sample_members, flat_observed),
    }]
    predictions = {"analysis": ensemble_mean, **bundle["products"]}
    for source, predicted in bundle["products"].items():
        rows.append({
            "source": source, "source_type": "comparison_analysis",
            "evaluation": "all_station_comparison", "independent": False,
            **deterministic_metrics(predicted, observed),
        })

    station_rows = []
    for station_index, station in enumerate(bundle["station_id"]):
        row = {
            "station_id": station,
            "lat": float(bundle["station_lat"][station_index]),
            "lon": float(bundle["station_lon"][station_index]),
            "observed_period_mean_mm": float(np.nanmean(observed[:, station_index])),
            "observed_within_period_daily_sd_mm": float(np.nanstd(observed[:, station_index])),
        }
        for source, predicted in predictions.items():
            row[f"{source}_period_mean_mm"] = float(np.nanmean(predicted[:, station_index]))
            row[f"{source}_within_period_daily_sd_mm"] = float(np.nanstd(predicted[:, station_index]))
        station_rows.append(row)

    variability_rows = []
    same_station_rows = []
    observed_mean = np.nanmean(observed, axis=0)
    observed_sd = np.nanstd(observed, axis=0)
    for source, predicted in predictions.items():
        predicted_mean = np.nanmean(predicted, axis=0)
        predicted_sd = np.nanstd(predicted, axis=0)
        mean_metrics = deterministic_metrics(predicted_mean, observed_mean)
        metrics = deterministic_metrics(predicted_sd, observed_sd)
        variance_ratio = float(np.nanmean(predicted_sd**2) / np.nanmean(observed_sd**2))
        variability_rows.append({
            "source": source,
            "metric": "station_within_period_daily_sd",
            "evaluation": "all_station_variability_fit",
            "independent": False,
            "variability_ratio": variance_ratio,
            **metrics,
        })
        same_station_rows.append({
            "source": source,
            "n_stations": int(len(observed_sd)),
            "evaluation": "same_locations_assimilated_fit",
            "independent": False,
            **{f"period_mean_{key}": value for key, value in mean_metrics.items()},
            **{f"daily_sd_{key}": value for key, value in metrics.items()},
            "daily_sd_variance_ratio": variance_ratio,
            "daily_sd_amplitude_ratio": float(np.sqrt(variance_ratio)),
        })
    return rows, station_rows, variability_rows, same_station_rows


def gauge_daily_network_rows(bundle: dict, product_times: np.ndarray,
                             bmd_times: np.ndarray) -> list[dict]:
    fields = {
        "bmd": bundle["observed"],
        "analysis": bundle["members"].mean(axis=1),
        **bundle["products"],
    }
    rows = []
    for index, (product_day, bmd_day) in enumerate(zip(product_times, bmd_times)):
        for source, field in fields.items():
            values = np.asarray(field[index], float)
            finite = values[np.isfinite(values)]
            rows.append({
                "product_date": str(product_day),
                "bmd_observation_date": str(bmd_day),
                "source_archive_date": str(
                    bmd_day if source in {"bmd", "imerg_native_0p1"} else product_day
                ),
                "source": source,
                "network_mean_mm": float(np.mean(finite)),
                "network_station_sd_mm": float(np.std(finite)),
                "wet_station_fraction": float(np.mean(finite >= WET_MM)),
                "n_stations": int(len(finite)),
            })
    return rows


def make_map_axes(axes, geometry: dict, bounds: tuple[float, float, float, float]) -> None:
    lon_min, lon_max, lat_min, lat_max = bounds
    for axis in np.asarray(axes).ravel():
        axis.set_facecolor("white")
        for polygon in polygon_parts(geometry):
            for ring in polygon:
                axis.plot(ring[:, 0], ring[:, 1], color="black", lw=0.55, zorder=5)
        axis.set_xlim(lon_min - 0.15, lon_max + 0.15)
        axis.set_ylim(lat_min - 0.15, lat_max + 0.15)
        axis.set_aspect("equal")
        axis.set_xlabel("longitude")
        axis.set_ylabel("latitude")


def white_cmap(name: str):
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(name).copy()
    cmap.set_bad("white")
    return cmap


def save_figure(figure, out_dir: Path, stem: str) -> None:
    figure.savefig(out_dir / f"{stem}.png", dpi=220, bbox_inches="tight", facecolor="white")
    figure.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")


def plot_field_grid(fields: dict[str, np.ndarray], grids: dict[str, dict],
                    masks: dict[str, np.ndarray], geometry: dict, bounds,
                    out_dir: Path, stem: str,
                    title: str, cmap: str, symmetric: bool = False,
                    station_overlay: dict | None = None) -> None:
    import matplotlib.pyplot as plt

    sources = list(fields)
    columns = 2
    rows = int(np.ceil(len(sources) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(10.8, 4.6 * rows), squeeze=False)
    values = np.concatenate([
        np.asarray(fields[source], float)[masks[source]] for source in sources
    ])
    finite = values[np.isfinite(values)]
    if symmetric:
        limit = float(np.percentile(np.abs(finite), 98))
        vmin, vmax = -limit, limit
    else:
        vmin, vmax = 0.0, float(np.percentile(finite, 99))
    for axis, source in zip(axes.ravel(), sources):
        layer = np.where(masks[source], fields[source], np.nan)
        image = axis.pcolormesh(grids[source]["lon"], grids[source]["lat"], layer,
                                shading="auto", cmap=white_cmap(cmap),
                                vmin=vmin, vmax=vmax, rasterized=True)
        if station_overlay is not None and source == "analysis":
            axis.scatter(station_overlay["lon"], station_overlay["lat"],
                         c=station_overlay["value"], cmap=white_cmap(cmap),
                         vmin=vmin, vmax=vmax, edgecolors="black", linewidths=0.35,
                         s=17, zorder=7)
        label = SOURCE_LABELS.get(source, source)
        if station_overlay is not None and source == "analysis":
            label += " (BMD dots)"
        axis.set_title(label)
        figure.colorbar(image, ax=axis, shrink=0.78, label="mm/day")
    for axis in axes.ravel()[len(sources):]:
        axis.set_visible(False)
    make_map_axes(axes.ravel()[:len(sources)], geometry, bounds)
    figure.suptitle(title, fontsize=13)
    figure.tight_layout()
    save_figure(figure, out_dir, stem)
    plt.close(figure)


def plot_station_residual_maps(bundle: dict, mask: np.ndarray, geometry: dict, bounds,
                               out_dir: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    observed = bundle["observed"]
    predictions = {"analysis": bundle["members"].mean(axis=1), **bundle["products"]}
    residuals = {
        source: np.nanmean(predicted - observed, axis=0)
        for source, predicted in predictions.items()
    }
    finite = np.concatenate([values[np.isfinite(values)] for values in residuals.values()])
    limit = max(1.0, float(np.percentile(np.abs(finite), 95)))
    figure, axes = plt.subplots(2, 2, figsize=(10.8, 9.2), squeeze=False)
    choose = bundle.get("inside_country", np.ones(len(bundle["station_id"]), bool))
    for axis, (source, values) in zip(axes.ravel(), residuals.items()):
        image = axis.scatter(
            bundle["station_lon"][choose], bundle["station_lat"][choose], c=values[choose],
            cmap=white_cmap("RdBu_r"), vmin=-limit, vmax=limit,
            edgecolors="black", linewidths=0.45, s=34, zorder=7,
        )
        axis.set_title(SOURCE_LABELS[source])
        figure.colorbar(image, ax=axis, shrink=0.78, label="mean residual (mm/day)")
    make_map_axes(axes, geometry, bounds)
    figure.suptitle(title + "\nstation period-mean residual: source minus BMD")
    figure.tight_layout()
    save_figure(figure, out_dir, "09_station_mean_residual_maps")
    plt.close(figure)


def plot_daily_series(daily_rows: list[dict], out_dir: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    dates = sorted({row["product_date"] for row in daily_rows})
    figure, axes = plt.subplots(3, 1, figsize=(11.5, 8.5), sharex=True)
    metrics = (("domain_mean_mm", "Bangladesh mean (mm/day)"),
               ("spatial_sd_mm", "spatial SD (mm/day)"),
               ("wet_area_fraction", "wet-area fraction (≥1 mm/day)"))
    for source in ("analysis", *REFERENCE_ORDER):
        selected = {
            row["product_date"]: row for row in daily_rows if row["source"] == source
        }
        for axis, (metric, ylabel) in zip(axes, metrics):
            axis.plot(dates, [selected[day][metric] for day in dates], marker="o", ms=2.5,
                      lw=1.25, label=SOURCE_LABELS[source])
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.2)
    axes[0].legend(ncol=4, fontsize=8)
    axes[-1].tick_params(axis="x", rotation=45)
    axes[-1].set_xlabel("produced-analysis / gridded-product date")
    figure.suptitle(title)
    figure.tight_layout()
    save_figure(figure, out_dir, "03_daily_domain_variability")
    plt.close(figure)


def plot_matrix(rows: list[dict], out_dir: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    metrics = ["mean_r", "mean_bias_mm", "mean_rmse_mm", "mean_sd_ratio",
               "variability_r", "variability_bias_mm", "variability_rmse_mm",
               "variability_sd_ratio", "mean_daily_spatial_r", "mean_daily_crmse_mm"]
    labels = ["mean r", "mean bias", "mean RMSE", "mean SD ratio", "daily-SD r",
              "daily-SD bias", "daily-SD RMSE", "daily-SD ratio", "daily pattern r",
              "daily cRMSE"]
    table = np.asarray([[row.get(metric, np.nan) for metric in metrics] for row in rows], float)
    scaled = np.zeros_like(table)
    for column in range(table.shape[1]):
        values = table[:, column]
        metric = metrics[column]
        if metric in {"mean_bias_mm", "variability_bias_mm"}:
            utility = -np.abs(values)
        elif metric in {"mean_rmse_mm", "variability_rmse_mm", "mean_daily_crmse_mm"}:
            utility = -values
        elif metric in {"mean_sd_ratio", "variability_sd_ratio"}:
            utility = -np.abs(np.log(np.maximum(values, 1.0e-12)))
        else:
            utility = values
        span = np.nanmax(utility) - np.nanmin(utility)
        scaled[:, column] = ((utility - np.nanmin(utility)) / span if span else 0.5)
    figure, axis = plt.subplots(figsize=(15, 3.6))
    image = axis.imshow(scaled, cmap="YlGn", vmin=0, vmax=1, aspect="auto")
    for row in range(table.shape[0]):
        for column in range(table.shape[1]):
            axis.text(column, row, f"{table[row, column]:.2f}", ha="center", va="center", fontsize=8)
    axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    axis.set_yticks(
        range(len(rows)),
        [
            f"{SOURCE_LABELS[row['reference']]} ({row['native_support_degrees']:g}°)"
            for row in rows
        ],
    )
    axis.set_title(f"{title} agreement matrix (raw values; colour ranks within metric)")
    figure.colorbar(image, ax=axis, shrink=0.7, label="relative agreement within column")
    figure.tight_layout()
    save_figure(figure, out_dir, "04_product_agreement_matrix")
    plt.close(figure)


def plot_gauges(bundle: dict, gauge_rows: list[dict], out_dir: Path, title: str,
                label: str) -> None:
    import matplotlib.pyplot as plt

    observed = bundle["observed"]
    predictions = {"analysis": bundle["members"].mean(axis=1), **bundle["products"]}
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    sources = list(predictions)
    x = np.arange(len(sources))
    lookup = {row["source"]: row for row in gauge_rows}
    axes[0, 0].bar(x - 0.17, [lookup[source]["mae_mm"] for source in sources], 0.34, label="MAE")
    axes[0, 0].bar(x + 0.17, [lookup[source]["rmse_mm"] for source in sources], 0.34, label="RMSE")
    axes[0, 0].set_xticks(x, [SOURCE_LABELS[source] for source in sources], rotation=25, ha="right")
    axes[0, 0].set_ylabel("mm/day"); axes[0, 0].legend(); axes[0, 0].set_title("All station-days")
    axes[0, 1].bar(x, [lookup[source]["correlation"] for source in sources])
    axes[0, 1].set_xticks(x, [SOURCE_LABELS[source] for source in sources], rotation=25, ha="right")
    axes[0, 1].set_ylim(-0.1, 1.0); axes[0, 1].set_ylabel("correlation")
    axes[0, 1].set_title("Daily values versus BMD")
    days = np.arange(observed.shape[0])
    axes[1, 0].plot(days, np.nanmean(observed, axis=1), color="black", lw=2, label="BMD")
    for source, predicted in predictions.items():
        axes[1, 0].plot(days, np.nanmean(predicted, axis=1), lw=1.2, label=SOURCE_LABELS[source])
    axes[1, 0].set_xlabel("paired BMD observation index (+1 day)"); axes[1, 0].set_ylabel("network mean (mm/day)")
    axes[1, 0].legend(fontsize=7, ncol=2); axes[1, 0].set_title("Daily network mean")
    observed_mean = np.nanmean(observed, axis=0)
    limit = float(np.nanpercentile(np.concatenate([observed_mean, *[np.nanmean(p, axis=0) for p in predictions.values()]]), 99))
    for source, predicted in predictions.items():
        axes[1, 1].scatter(observed_mean, np.nanmean(predicted, axis=0), s=18, alpha=0.75,
                           label=SOURCE_LABELS[source])
    axes[1, 1].plot([0, limit], [0, limit], color="black", ls="--", lw=1)
    axes[1, 1].set_xlim(0, limit); axes[1, 1].set_ylim(0, limit)
    axes[1, 1].set_xlabel(f"BMD station {label} mean"); axes[1, 1].set_ylabel("source period mean")
    axes[1, 1].legend(fontsize=7); axes[1, 1].set_title("Paired station-period mean")
    figure.suptitle(
        title + "\nBMD labels are +1 day; scores are assimilated fit, not verification"
    )
    figure.tight_layout()
    save_figure(figure, out_dir, "05_all_station_gauge_fit")
    plt.close(figure)


def plot_station_variability(bundle: dict, same_station_rows: list[dict], out_dir: Path,
                              title: str, label: str) -> None:
    """Show the exact same-station daily-SD comparison for every product."""
    import matplotlib.pyplot as plt

    observed_sd = np.nanstd(bundle["observed"], axis=0)
    predictions = {"analysis": bundle["members"].mean(axis=1), **bundle["products"]}
    lookup = {row["source"]: row for row in same_station_rows}
    all_sd = np.concatenate([
        observed_sd, *[np.nanstd(field, axis=0) for field in predictions.values()]
    ])
    maximum = float(np.nanpercentile(all_sd, 99))
    maximum = max(1.0, maximum)
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 9.2), squeeze=False)
    for axis, (source, field) in zip(axes.ravel(), predictions.items()):
        predicted_sd = np.nanstd(field, axis=0)
        axis.scatter(observed_sd, predicted_sd, s=31, alpha=0.8)
        axis.plot([0, maximum], [0, maximum], "k--", lw=1)
        axis.set_xlim(0, maximum); axis.set_ylim(0, maximum)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(SOURCE_LABELS[source])
        metrics = lookup[source]
        axis.text(
            0.04, 0.96,
            f"r = {metrics['daily_sd_correlation']:.2f}\n"
            f"bias = {metrics['daily_sd_bias_mm']:+.2f} mm/day\n"
            f"amplitude ratio = {metrics['daily_sd_amplitude_ratio']:.2f}",
            transform=axis.transAxes, va="top", fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.82},
        )
        axis.set_xlabel(f"BMD station {label} daily SD (mm/day)")
        axis.set_ylabel("source station daily SD (mm/day)")
        axis.grid(alpha=0.2)
    figure.suptitle(
        title + "\nSame 39 stations and paired dates; BMD is assimilated fit, not verification"
    )
    figure.tight_layout()
    save_figure(figure, out_dir, "10_same_station_daily_variability")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    requested_months = evaluation_months(args)
    requested_label = period_label(requested_months)
    requested_tag = period_tag(requested_months)
    shared = load_shared_evaluator()
    paths = [Path(path) for path in args.zarr]
    archive = shared.load_archive(paths, args.factor)
    if args.method not in archive["methods"]:
        raise ValueError(f"{args.method!r} not in archived methods {archive['methods']}")
    method_index = archive["methods"].index(args.method)

    boundary_path = Path(args.boundary_geojson)
    geometry = geometry_from_geojson(json.loads(boundary_path.read_text()))
    boundary_metadata_path = boundary_path.with_suffix(".metadata.json")
    boundary_metadata = (
        json.loads(boundary_metadata_path.read_text())
        if boundary_metadata_path.is_file() else {}
    )
    fine_mask = country_mask(archive["lat"], archive["lon"], geometry) & archive["valid"]
    if fine_mask.sum() < 100:
        raise ValueError(f"Bangladesh mask contains only {fine_mask.sum()} model cells")
    bounds = boundary_bounds(geometry)

    times = archive["time"].astype("datetime64[D]")
    years = np.asarray([int(str(day)[:4]) for day in times])
    months = np.asarray([int(str(day)[5:7]) for day in times])
    requested_month_mask = np.isin(months, requested_months)
    available_years = []
    for year in sorted(set(years[requested_month_mask].tolist())):
        choose = (years == year) & requested_month_mask
        expected = expected_period_days(year, requested_months)
        if choose.sum() == expected:
            available_years.append(year)
    if args.year not in available_years:
        raise ValueError(
            f"target {requested_label} {args.year} is incomplete; "
            f"complete years={available_years}"
        )
    baseline_years = [year for year in available_years if year != args.year]
    if not baseline_years:
        raise ValueError(
            f"leave-target-year-out {requested_label} climatology needs at least one other year"
        )

    day_offset = background_day_offset(archive)
    if day_offset != -1:
        raise ValueError(
            f"this evaluation expects the documented BMD +1-day contract, but "
            f"scope.background_day_offset={day_offset:+d}"
        )
    all_period = requested_month_mask & np.isin(years, available_years)
    june_bmd_times = times[all_period]
    june_product_times = june_bmd_times + np.timedelta64(day_offset, "D")
    # Prepared IMERG is stamped with the BMD 03 UTC window end date. The saved
    # analysis, CPC conditioning day, and CHIRPS conditioning field are D-1.
    # Load IMERG at D (the exact retrieval assimilated), then associate all
    # products with the D-1 produced/model date in daily comparison tables.
    native_imerg = load_native_imerg(
        [Path(path) for path in args.native_imerg], june_bmd_times
    )
    native_cpc = load_native_cpc(Path(args.cpc_dir), june_product_times, bounds)

    grids = {
        "analysis": {"lat": archive["lat"], "lon": archive["lon"],
                     "resolution_degrees": 0.05},
        "chirps": {"lat": archive["lat"], "lon": archive["lon"],
                   "resolution_degrees": 0.05},
        "imerg_native_0p1": native_imerg,
        "cpc_native_0p5": native_cpc,
    }
    masks = {
        "analysis": fine_mask,
        "chirps": fine_mask,
        "imerg_native_0p1": country_mask(native_imerg["lat"], native_imerg["lon"], geometry),
        "cpc_native_0p5": country_mask(native_cpc["lat"], native_cpc["lon"], geometry),
    }
    product_lookup = {day: index for index, day in enumerate(june_product_times)}
    observation_lookup = {day: index for index, day in enumerate(june_bmd_times)}

    target = (years == args.year) & requested_month_mask
    target_bmd_times = times[target]
    target_product_times = target_bmd_times + np.timedelta64(day_offset, "D")
    product_positions = np.asarray([product_lookup[day] for day in target_product_times])
    observation_positions = np.asarray([observation_lookup[day] for day in target_bmd_times])
    target_fields = {
        "analysis": np.asarray(archive["mean"][method_index, target], float),
        # CHIRPS stored on Zarr row D is already the D-1 conditioning field.
        "chirps": np.asarray(archive["chirps"][target], float),
        "imerg_native_0p1": native_imerg["values"][observation_positions],
        "cpc_native_0p5": native_cpc["values"][product_positions],
    }
    target_spread = np.asarray(archive["spread"][method_index, target], float)
    masked_target = {source: np.where(masks[source][None], field, np.nan)
                     for source, field in target_fields.items()}

    target_dataset = next(dataset for dataset in archive["datasets"]
                          if np.isin(target_bmd_times, np.asarray(dataset.time.values).astype("datetime64[D]")).all())
    target_members = load_target_members(target_dataset, method_index, target_bmd_times)
    target_products = {
        source: {**grids[source], "values": target_fields[source]}
        for source in REFERENCE_ORDER
    }
    stations = load_station_bundle(target_dataset, method_index, target_bmd_times, target_members,
                                   archive, target_products, shared)
    stations["inside_country"] = points_in_country(
        stations["station_lat"], stations["station_lon"], geometry
    )
    if not stations["inside_country"].all():
        print(
            f"[mask] warning: {(~stations['inside_country']).sum()} station(s) lie "
            "outside the ADM0 polygon and will be omitted from maps, but retained "
            "in the requested all-station fit tables"
        )
    gauge_rows, station_rows, gauge_variability_rows, same_station_rows = gauge_evaluation(
        stations, shared
    )
    gauge_daily_rows = gauge_daily_network_rows(
        stations, target_product_times, target_bmd_times
    )

    daily_rows = spatial_daily_rows(
        target_product_times, masked_target, masks, target_spread
    )
    for row in daily_rows:
        position = int(np.where(target_product_times == np.datetime64(row["date"], "D"))[0][0])
        row["product_date"] = row.pop("date")
        row["bmd_observation_date"] = str(target_bmd_times[position])
        row["source_archive_date"] = (
            row["bmd_observation_date"]
            if row["source"] == "imerg_native_0p1" else row["product_date"]
        )
    daily_matrix_rows = []
    common_0p5_fields = {
        source: (
            target_fields[source] if source == "cpc_native_0p5" else
            regrid_cell_average(
                target_fields[source], grids[source]["lat"], grids[source]["lon"],
                native_cpc["lat"], native_cpc["lon"], masks[source],
            )
        ) for source in target_fields
    }
    for reference in REFERENCE_ORDER:
        reference_grid = grids[reference]
        analysis_native = (
            target_fields["analysis"] if reference == "chirps" else
            regrid_cell_average(
                target_fields["analysis"], archive["lat"], archive["lon"],
                reference_grid["lat"], reference_grid["lon"], fine_mask,
            )
        )
        native_daily = [
            field_metrics(analysis_native[i], target_fields[reference][i], masks[reference])
            for i in range(len(target_product_times))
        ]
        common_daily = [
            field_metrics(common_0p5_fields["analysis"][i],
                          common_0p5_fields[reference][i], masks["cpc_native_0p5"])
            for i in range(len(target_product_times))
        ]
        for product_day, bmd_day, native, common in zip(
            target_product_times, target_bmd_times, native_daily, common_daily
        ):
            daily_matrix_rows.append({
                "product_date": str(product_day),
                "bmd_observation_date": str(bmd_day),
                "reference_source_date": str(
                    bmd_day if reference == "imerg_native_0p1" else product_day
                ),
                "reference": reference,
                "native_support_degrees": reference_grid["resolution_degrees"],
                **{f"native_{key}": value for key, value in native.items()},
                **{f"common_0p5_{key}": value for key, value in common.items()},
            })

    monthly_fields = {
        source: masked_temporal_stat(field, masks[source], "mean")
        for source, field in masked_target.items()
    }
    variability_fields = {
        source: masked_temporal_stat(field, masks[source], "std")
        for source, field in masked_target.items()
    }
    matrix_rows = []
    for reference in REFERENCE_ORDER:
        reference_grid = grids[reference]
        analysis_native_daily = (
            target_fields["analysis"] if reference == "chirps" else
            regrid_cell_average(
                target_fields["analysis"], archive["lat"], archive["lon"],
                reference_grid["lat"], reference_grid["lon"], fine_mask,
            )
        )
        analysis_native_mean = masked_temporal_stat(
            analysis_native_daily, masks[reference], "mean"
        )
        analysis_native_variability = masked_temporal_stat(
            analysis_native_daily, masks[reference], "std"
        )
        mean_metrics = field_metrics(
            analysis_native_mean, monthly_fields[reference], masks[reference]
        )
        variability_metrics = field_metrics(
            analysis_native_variability, variability_fields[reference], masks[reference]
        )
        common_analysis_mean = masked_temporal_stat(
            common_0p5_fields["analysis"], masks["cpc_native_0p5"], "mean"
        )
        common_reference_mean = masked_temporal_stat(
            common_0p5_fields[reference], masks["cpc_native_0p5"], "mean"
        )
        common_analysis_variability = masked_temporal_stat(
            common_0p5_fields["analysis"], masks["cpc_native_0p5"], "std"
        )
        common_reference_variability = masked_temporal_stat(
            common_0p5_fields[reference], masks["cpc_native_0p5"], "std"
        )
        common_mean_metrics = field_metrics(
            common_analysis_mean, common_reference_mean, masks["cpc_native_0p5"]
        )
        common_variability_metrics = field_metrics(
            common_analysis_variability, common_reference_variability,
            masks["cpc_native_0p5"],
        )
        selected_daily = [row for row in daily_matrix_rows if row["reference"] == reference]
        matrix_rows.append({
            "reference": reference,
            "native_support_degrees": reference_grid["resolution_degrees"],
            **{f"mean_{key.replace('correlation', 'r').replace('spatial_sd_ratio', 'sd_ratio')}": value
               for key, value in mean_metrics.items() if key != "n_cells"},
            **{f"variability_{key.replace('correlation', 'r').replace('spatial_sd_ratio', 'sd_ratio')}": value
               for key, value in variability_metrics.items() if key != "n_cells"},
            **{f"common_0p5_mean_{key.replace('correlation', 'r').replace('spatial_sd_ratio', 'sd_ratio')}": value
               for key, value in common_mean_metrics.items() if key != "n_cells"},
            **{f"common_0p5_variability_{key.replace('correlation', 'r').replace('spatial_sd_ratio', 'sd_ratio')}": value
               for key, value in common_variability_metrics.items() if key != "n_cells"},
            "mean_daily_spatial_r": float(np.nanmean([row["native_correlation"] for row in selected_daily])),
            "mean_daily_crmse_mm": float(np.nanmean([row["native_centered_rmse_mm"] for row in selected_daily])),
            "mean_daily_common_0p5_r": float(np.nanmean([row["common_0p5_correlation"] for row in selected_daily])),
            "mean_daily_common_0p5_crmse_mm": float(np.nanmean([row["common_0p5_centered_rmse_mm"] for row in selected_daily])),
        })

    yearly_fields: dict[int, dict[str, np.ndarray]] = {}
    climatology_rows = []
    for year in available_years:
        choose = (years == year) & requested_month_mask
        year_bmd_times = times[choose]
        year_product_times = year_bmd_times + np.timedelta64(day_offset, "D")
        june_positions = np.asarray([product_lookup[day] for day in year_product_times])
        year_observation_positions = np.asarray([
            observation_lookup[day] for day in year_bmd_times
        ])
        year_sources = {
            "analysis": np.asarray(archive["mean"][method_index, choose], float),
            "chirps": np.asarray(archive["chirps"][choose], float),
            "imerg_native_0p1": native_imerg["values"][year_observation_positions],
            "cpc_native_0p5": native_cpc["values"][june_positions],
        }
        yearly_fields[year] = {}
        for source, field in year_sources.items():
            mean_field = masked_temporal_stat(field, masks[source], "mean")
            variability_field = masked_temporal_stat(field, masks[source], "std")
            yearly_fields[year][f"{source}_mean"] = mean_field
            yearly_fields[year][f"{source}_variability"] = variability_field
            climatology_rows.append({
                "year": year, "source": source, "days": int(choose.sum()),
                "product_start": str(year_product_times[0]),
                "product_end": str(year_product_times[-1]),
                "bmd_observation_start": str(year_bmd_times[0]),
                "bmd_observation_end": str(year_bmd_times[-1]),
                "native_resolution_degrees": grids[source]["resolution_degrees"],
                "domain_mean_mm": float(np.nanmean(mean_field[masks[source]])),
                "spatial_sd_of_period_mean_mm": float(np.nanstd(mean_field[masks[source]])),
                "mean_within_period_daily_sd_mm": float(np.nanmean(variability_field[masks[source]])),
            })
    climatology_mean = {
        source: masked_field_average(
            [yearly_fields[year][f"{source}_mean"] for year in baseline_years], masks[source]
        ) for source in target_fields
    }
    climatology_variability = {
        source: masked_field_average(
            [yearly_fields[year][f"{source}_variability"] for year in baseline_years],
            masks[source],
        ) for source in target_fields
    }
    anomalies = {source: monthly_fields[source] - climatology_mean[source]
                 for source in target_fields}
    climatology_matrix_rows = []
    for reference in REFERENCE_ORDER:
        reference_grid = grids[reference]
        analysis_year_means, analysis_year_variability = [], []
        for baseline_year in baseline_years:
            choose = (years == baseline_year) & requested_month_mask
            analysis_days = np.asarray(archive["mean"][method_index, choose], float)
            analysis_native_days = (
                analysis_days if reference == "chirps" else
                regrid_cell_average(
                    analysis_days, archive["lat"], archive["lon"],
                    reference_grid["lat"], reference_grid["lon"], fine_mask,
                )
            )
            analysis_year_means.append(masked_temporal_stat(
                analysis_native_days, masks[reference], "mean"
            ))
            analysis_year_variability.append(masked_temporal_stat(
                analysis_native_days, masks[reference], "std"
            ))
        analysis_climate_native = masked_field_average(
            analysis_year_means, masks[reference]
        )
        analysis_variability_native = masked_field_average(
            analysis_year_variability, masks[reference]
        )
        mean_metrics = field_metrics(
            analysis_climate_native, climatology_mean[reference], masks[reference]
        )
        variability_metrics = field_metrics(
            analysis_variability_native, climatology_variability[reference], masks[reference]
        )
        climatology_matrix_rows.append({
            "reference": reference,
            "years": ",".join(map(str, baseline_years)),
            "native_support_degrees": reference_grid["resolution_degrees"],
            **{f"climatology_mean_{key}": value for key, value in mean_metrics.items()},
            **{f"climatology_daily_sd_{key}": value for key, value in variability_metrics.items()},
        })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_rows(out_dir / "daily_domain_variability.csv", daily_rows)
    write_rows(out_dir / "daily_spatial_agreement.csv", daily_matrix_rows)
    write_rows(out_dir / f"{requested_tag}{args.year}_product_matrix.csv", matrix_rows)
    write_rows(out_dir / "all_station_gauge_fit.csv", gauge_rows)
    write_rows(out_dir / "all_station_daily_variability_fit.csv", gauge_variability_rows)
    write_rows(out_dir / "same_station_period_matrix.csv", same_station_rows)
    write_rows(out_dir / "all_station_daily_network.csv", gauge_daily_rows)
    write_rows(out_dir / f"station_{requested_tag}{args.year}_summary.csv", station_rows)
    write_rows(out_dir / f"available_{requested_tag}_climatology.csv", climatology_rows)
    write_rows(out_dir / f"{requested_tag}_climatology_product_matrix.csv", climatology_matrix_rows)
    field_payload = {
        "product_times": target_product_times.astype(str),
        "bmd_observation_times": target_bmd_times.astype(str),
        "available_complete_period_years": np.asarray(available_years),
        "baseline_years": np.asarray(baseline_years),
    }
    for source in target_fields:
        field_payload[f"{source}_lat"] = grids[source]["lat"]
        field_payload[f"{source}_lon"] = grids[source]["lon"]
        field_payload[f"{source}_bangladesh_mask"] = masks[source]
        field_payload[f"{requested_tag}{args.year}_{source}_mean"] = monthly_fields[source]
        field_payload[f"{requested_tag}{args.year}_{source}_daily_sd"] = variability_fields[source]
        field_payload[f"baseline_{source}_mean"] = climatology_mean[source]
        field_payload[f"baseline_{source}_daily_sd"] = climatology_variability[source]
        field_payload[f"{requested_tag}{args.year}_{source}_anomaly"] = anomalies[source]
    np.savez_compressed(
        out_dir / f"{requested_tag}_bangladesh_fields.npz",
        **field_payload,
    )

    title = f"{PRODUCT_NAME} — Bangladesh {requested_label} {args.year} production contract"
    station_map = stations["inside_country"]
    overlay = {"lon": stations["station_lon"][station_map],
               "lat": stations["station_lat"][station_map],
               "value": np.nanmean(stations["observed"], axis=0)[station_map]}
    plot_field_grid(monthly_fields, grids, masks, geometry, bounds,
                    out_dir, f"01_{requested_tag}{args.year}_mean_maps", title + " period mean", "turbo",
                    station_overlay=overlay)
    plot_field_grid(variability_fields, grids, masks, geometry, bounds,
                    out_dir, f"02_{requested_tag}{args.year}_daily_variability_maps",
                    title + " within-period daily variability", "magma",
                    station_overlay={
                        "lon": stations["station_lon"][station_map],
                        "lat": stations["station_lat"][station_map],
                        "value": np.nanstd(stations["observed"], axis=0)[station_map],
                    })
    plot_daily_series(daily_rows, out_dir, title)
    plot_matrix(matrix_rows, out_dir, f"{requested_label} {args.year} {PRODUCT_NAME}")
    plot_gauges(stations, gauge_rows, out_dir, title, requested_label)
    plot_station_variability(stations, same_station_rows, out_dir, title, requested_label)
    plot_field_grid(climatology_mean, grids, masks, geometry, bounds,
                    out_dir, f"06_leave{args.year}out_{requested_tag}_climatology_maps",
                    f"{requested_label} climatology ({', '.join(map(str, baseline_years))}; excludes {args.year})",
                    "turbo")
    plot_field_grid(
        climatology_variability, grids, masks, geometry, bounds, out_dir,
        f"07_leave{args.year}out_{requested_tag}_variability_climatology_maps",
        f"Within-{requested_label} daily-SD climatology ({', '.join(map(str, baseline_years))}; excludes {args.year})",
        "magma",
    )
    plot_field_grid(anomalies, grids, masks, geometry, bounds,
                    out_dir, f"08_{requested_tag}{args.year}_anomaly_maps",
                    f"{requested_label} {args.year} minus leave-{args.year}-out climatology",
                    "RdBu_r", symmetric=True)
    plot_station_residual_maps(stations, fine_mask, geometry, bounds, out_dir, title)

    report = {
        "design": {
            "method": args.method, "target_year": args.year,
            "months": list(requested_months), "period_label": requested_label,
            "product_name": PRODUCT_NAME,
            "product_full_name": PRODUCT_FULL_NAME,
            "lineage_method_key": args.method,
            "target_days": len(target_product_times),
            "ensemble_members": int(target_members.shape[1]),
            "zarr_stores": [str(path) for path in paths],
            "boundary_geojson": str(boundary_path),
            "boundary_metadata": {
                key: boundary_metadata.get(key) for key in (
                    "boundaryName", "boundaryISO", "boundaryYear", "boundaryType",
                    "boundarySource", "boundaryLicense", "licenseDetail", "licenseSource",
                    "gjDownloadURL", "simplifiedGeometryGeoJSON",
                ) if boundary_metadata.get(key) is not None
            },
            "bangladesh_grid_cells_by_product": {
                source: int(masks[source].sum()) for source in masks
            },
            "gauge_stations_total": int(len(stations["station_id"])),
            "gauge_stations_inside_boundary_for_maps": int(stations["inside_country"].sum()),
            "available_complete_period_years": available_years,
            "leave_target_year_out_climatology_years": baseline_years,
            "date_contract": {
                "bmd_observation_dates": [str(target_bmd_times[0]), str(target_bmd_times[-1])],
                "produced_analysis_and_product_dates": [
                    str(target_product_times[0]), str(target_product_times[-1])
                ],
                "bmd_minus_product_days": int(-day_offset),
                "native_imerg_source_dates": [
                    str(target_bmd_times[0]), str(target_bmd_times[-1])
                ],
                "native_imerg_source_date_note": (
                    "the prepared 03 UTC window is stamped by its BMD end-date; "
                    "it is associated with the preceding produced/model day"
                ),
            },
            "native_imerg_sources": native_imerg["paths"],
            "native_imerg_timing": native_imerg["timing"],
            "original_cpc_sources": native_cpc["paths"],
            "original_cpc_timing": native_cpc["timing"],
            "assimilated_imerg_support_degrees": args.factor * 0.05,
            "comparison_supports_degrees": {
                source: grids[source]["resolution_degrees"] for source in target_fields
            },
        },
        "evidence_roles": EVIDENCE_ROLES,
        "period_product_matrix": matrix_rows,
        "period_climatology_product_matrix": climatology_matrix_rows,
        "all_station_gauge_fit": gauge_rows,
        "all_station_daily_variability_fit": gauge_variability_rows,
        "same_station_period_matrix": same_station_rows,
        "all_station_daily_network": gauge_daily_rows,
        "interpretation": {
            "mask": "all map pixels and spatial scores are inside Bangladesh ADM0 and model-valid land only",
            "outside_country": "NaN and rendered white",
            "climatology": (
                "mean across complete available requested periods excluding target year; "
                "not a 30-year climate normal"
            ),
            "gauge_warning": "all production gauges were assimilated, so scores diagnose fit rather than independent skill",
            "support_aware_scoring": (
                "analysis is area-averaged to each product's actual native support; "
                "a second matrix puts every source on original CPC 0.5-degree cells"
            ),
            "imerg_warning": (
                "the compared IMERG is native 0.1 degree, while this arm assimilated "
                "the same product coarsened to S04 0.4 degree; 0.1-degree placement "
                "was not assimilated, but this is not a fully independent product"
            ),
            "cpc_warning": (
                "original CPC is evaluated on native 0.5-degree cells without fine-grid "
                "interpolation; its calendar-day window is about 3 hours offset from BMD"
            ),
            "chirps_warning": (
                "CHIRPS was the training-target analysis and is used only as a "
                "structural comparison"
            ),
            "verification_status": (
                "no gridded source is designated as truth; BMD is an assimilated "
                "observation fit in this all-station production run"
            ),
        },
    }
    (out_dir / "evaluation.json").write_text(json.dumps(json_ready(report), indent=2, allow_nan=False) + "\n")
    (out_dir / "README.md").write_text(
        f"# {PRODUCT_NAME}: Bangladesh production-contract evaluation, {args.year}\n\n"
        f"**{PRODUCT_FULL_NAME}.** Reproducibility lineage method: `{args.method}`; "
        f"{target_members.shape[1]} ensemble members; "
        f"{len(target_product_times)} target days for {requested_label}. Produced-analysis/product dates "
        f"are {target_product_times[0]} through {target_product_times[-1]}; paired BMD "
        f"labels are {target_bmd_times[0]} through {target_bmd_times[-1]} (+1 day). "
        f"{requested_label} climatology uses complete observation-label years "
        f"{', '.join(map(str, baseline_years))} and excludes {args.year}.\n\n"
        "All spatial maps and matrices use the Bangladesh ADM0 polygon intersected "
        "with the model-valid mask; everything outside is missing and plotted white. "
        "The BMD matrix is assimilated fit. IMERG is read at native 0.1-degree "
        "support, CHIRPS at 0.05 degrees, and original NOAA CPC at 0.5 degrees. "
        "Analysis fields are area-averaged to each comparison support. "
        "No gridded source is designated as truth. See `evaluation.json` for the "
        "complete design.\n"
        "\nBoundary attribution and license are recorded verbatim in the downloaded "
        "geoBoundaries metadata snapshot stored beside the GeoJSON.\n"
    )
    print(f"evaluation complete: {out_dir}")


if __name__ == "__main__":
    main()
