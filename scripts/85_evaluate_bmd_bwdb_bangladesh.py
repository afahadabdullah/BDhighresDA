#!/usr/bin/env python
"""Evaluate the BMD+BWDB Huber3 analysis over Bangladesh ADM0 boundary.

Reads completed all-station production Zarr stores from the BMD+BWDB archive
(data/processed/v2_bmd_bwdb_huber3_2021_2024). Evaluates day-by-day and
period-aggregated fields over Bangladesh. All spatial calculations and maps use
the Bangladesh ADM0 polygon, clipped so that all areas outside Bangladesh are
masked to NaN and rendered in pure white.

Evidence roles:
- analysis: BRISHTI-05 (BMD+BWDB Huber3) posterior analysis
- gauges: Combined BMD and BWDB station network (assimilated fit)
- imerg_native_0p1: Native 0.1-degree IMERG retrieval
- chirps: CHIRPS 0.05-degree conditioning reference
- cpc_native_0p5: NOAA CPC Global Unified 0.5-degree native product-day gauge analysis
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


METHOD_DEFAULT = "v2_simul_s04_huber3"
PRODUCT_NAME = "BRISHTI-05 (BMD+BWDB)"
PRODUCT_FULL_NAME = (
    "Bangladesh Rainfall Integration of Satellite, Hydrometeorological, and "
    "Terrestrial Information at 0.05° (BMD+BWDB Huber3)"
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
    "gauges": "BMD+BWDB gauges",
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
    "gauges": "all BMD+BWDB stations entered likelihood; assimilated fit, not independent verification",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--zarr", nargs="+", required=True,
                        help="path(s) to seasonal Zarr stores (2021_may_sep, 2022_may_sep, etc.)")
    parser.add_argument("--boundary-geojson", required=True,
                        help="GeoJSON file containing Bangladesh ADM0 boundary polygon")
    parser.add_argument("--out-dir", required=True,
                        help="output directory for evaluation reports, tables, and figures")
    parser.add_argument("--method", default=METHOD_DEFAULT,
                        help=f"method to extract from Zarr (default: {METHOD_DEFAULT})")
    parser.add_argument("--year", type=int, default=TARGET_YEAR,
                        help=f"target evaluation year (default: {TARGET_YEAR})")
    parser.add_argument("--month", type=int, default=TARGET_MONTH,
                        help=f"target evaluation month (default: {TARGET_MONTH})")
    parser.add_argument(
        "--months", nargs="+", type=int, default=None,
        help="one or more contiguous calendar months; overrides --month (e.g. 5 6 7 8)",
    )
    parser.add_argument("--factor", type=int, default=8,
                        help="assimilated IMERG S04 factor, retained as experiment metadata")
    parser.add_argument(
        "--native-imerg", nargs="+", required=True,
        help="prepared native 0.1-degree IMERG netCDFs covering every requested season",
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


def deterministic_metrics(predicted: np.ndarray, observed: np.ndarray) -> dict:
    predicted = np.asarray(predicted, float)
    observed = np.asarray(observed, float)
    mask = np.isfinite(predicted) & np.isfinite(observed)
    if not mask.any():
        return {
            "n": 0, "mae_mm": None, "rmse_mm": None, "bias_mm": None,
            "correlation": None, "centered_rmse_mm": None, "spatial_sd_ratio": None,
        }
    p = predicted[mask]
    o = observed[mask]
    diff = p - o
    var_p = float(np.var(p))
    var_o = float(np.var(o))
    corr = (
        float(np.cov(p, o)[0, 1] / np.sqrt(var_p * var_o))
        if var_p > 0 and var_o > 0 else (1.0 if np.allclose(p, o) else 0.0)
    )
    crmse = float(np.sqrt(np.mean((diff - np.mean(diff)) ** 2)))
    sd_ratio = float(np.sqrt(var_p / var_o)) if var_o > 0 else None
    return {
        "n": int(mask.sum()),
        "mae_mm": float(np.mean(np.abs(diff))),
        "rmse_mm": float(np.sqrt(np.mean(diff ** 2))),
        "bias_mm": float(np.mean(diff)),
        "correlation": corr,
        "centered_rmse_mm": crmse,
        "spatial_sd_ratio": sd_ratio,
    }


def field_metrics(predicted: np.ndarray, observed: np.ndarray, mask: np.ndarray) -> dict:
    return deterministic_metrics(predicted[mask], observed[mask])


def regrid_cell_average(field: np.ndarray, src_lat: np.ndarray, src_lon: np.ndarray,
                        tgt_lat: np.ndarray, tgt_lon: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    field = np.asarray(field, float)
    is_3d = field.ndim == 3
    if not is_3d:
        field = field[None, :, :]
    n_time, n_src_y, n_src_x = field.shape
    n_tgt_y = len(tgt_lat)
    n_tgt_x = len(tgt_lon)

    dlat = float(abs(tgt_lat[1] - tgt_lat[0])) if len(tgt_lat) > 1 else 0.5
    dlon = float(abs(tgt_lon[1] - tgt_lon[0])) if len(tgt_lon) > 1 else 0.5
    half_lat = dlat / 2.0
    half_lon = dlon / 2.0

    out = np.full((n_time, n_tgt_y, n_tgt_x), np.nan, dtype=float)

    y_indices = []
    x_indices = []
    for y_idx, lat_c in enumerate(tgt_lat):
        y_mask = (src_lat >= lat_c - half_lat - 1e-6) & (src_lat <= lat_c + half_lat + 1e-6)
        y_indices.append(np.flatnonzero(y_mask))
    for x_idx, lon_c in enumerate(tgt_lon):
        x_mask = (src_lon >= lon_c - half_lon - 1e-6) & (src_lon <= lon_c + half_lon + 1e-6)
        x_indices.append(np.flatnonzero(x_mask))

    for y_idx in range(n_tgt_y):
        ys = y_indices[y_idx]
        if len(ys) == 0:
            continue
        for x_idx in range(n_tgt_x):
            xs = x_indices[x_idx]
            if len(xs) == 0:
                continue
            box = field[:, ys[:, None], xs[None, :]]
            if mask is not None:
                box_mask = mask[ys[:, None], xs[None, :]]
                box = np.where(box_mask[None, :, :], box, np.nan)
            valid_count = np.sum(np.isfinite(box), axis=(1, 2))
            mean_val = np.nanmean(box, axis=(1, 2))
            out[:, y_idx, x_idx] = np.where(valid_count > 0, mean_val, np.nan)

    return out if is_3d else out[0]


def masked_temporal_stat(field: np.ndarray, mask: np.ndarray, stat: str = "mean") -> np.ndarray:
    field = np.asarray(field, float)
    masked = np.where(mask[None, :, :], field, np.nan)
    if stat == "mean":
        return np.nanmean(masked, axis=0)
    elif stat == "std":
        return np.nanstd(masked, axis=0)
    raise ValueError(f"unknown stat: {stat}")


def masked_field_average(fields: list[np.ndarray], mask: np.ndarray) -> np.ndarray:
    stacked = np.stack(fields, axis=0)
    return np.nanmean(np.where(mask[None, :, :], stacked, np.nan), axis=0)


def spatial_daily_rows(dates: np.ndarray, fields: dict[str, np.ndarray],
                       masks: dict[str, np.ndarray], spread: np.ndarray) -> list[dict]:
    rows = []
    for day_idx, date in enumerate(dates):
        for source, field in fields.items():
            vals = field[source][day_idx] if isinstance(field, dict) else field[day_idx]
            m = masks[source]
            valid_vals = vals[m]
            finite = valid_vals[np.isfinite(valid_vals)]
            mean_val = float(np.mean(finite)) if len(finite) else np.nan
            sd_val = float(np.std(finite)) if len(finite) else np.nan
            wet_frac = float(np.mean(finite >= WET_MM)) if len(finite) else np.nan
            row = {
                "date": str(date),
                "source": source,
                "domain_mean_mm": mean_val,
                "spatial_sd_mm": sd_val,
                "wet_area_fraction": wet_frac,
            }
            if source == "analysis" and spread is not None:
                day_spread = spread[day_idx][m]
                row["posterior_spread_mm"] = float(np.nanmean(day_spread))
            rows.append(row)
    return rows


def load_native_imerg(paths: list[Path], dates: np.ndarray) -> dict:
    import xarray as xr

    datasets = [xr.open_dataset(path) for path in paths]
    ds = xr.concat(datasets, dim="time") if len(datasets) > 1 else datasets[0]
    time_vals = np.asarray(ds.time.values).astype("datetime64[D]")
    indices = [int(np.where(time_vals == d)[0][0]) for d in dates if d in time_vals]
    if len(indices) != len(dates):
        missing = [d for d in dates if d not in time_vals]
        raise ValueError(f"native IMERG missing dates: {missing[:5]} (total {len(missing)})")
    sub = ds.isel(time=indices)
    var_name = "precipitation" if "precipitation" in sub else "precip"
    lat = np.asarray(sub.lat.values, float)
    lon = np.asarray(sub.lon.values, float)
    vals = np.asarray(sub[var_name].values, float)
    return {"lat": lat, "lon": lon, "values": vals, "resolution_degrees": 0.1}


def load_native_cpc(cpc_dir: Path, dates: np.ndarray, bounds: tuple[float, float, float, float]) -> dict:
    import xarray as xr

    lon_min, lon_max, lat_min, lat_max = bounds
    years = sorted(set(int(str(d)[:4]) for d in dates))
    datasets = []
    for y in years:
        nc_path = cpc_dir / f"precip.{y}.nc"
        if not nc_path.is_file():
            raise FileNotFoundError(f"CPC file missing: {nc_path}")
        datasets.append(xr.open_dataset(nc_path))
    ds = xr.concat(datasets, dim="time") if len(datasets) > 1 else datasets[0]
    time_vals = np.asarray(ds.time.values).astype("datetime64[D]")
    indices = [int(np.where(time_vals == d)[0][0]) for d in dates]
    sub = ds.isel(time=indices)
    lat = np.asarray(sub.lat.values, float)
    lon = np.asarray(sub.lon.values, float)
    if (lon > 180).any():
        lon = np.where(lon > 180, lon - 360, lon)
        order = np.argsort(lon)
        lon = lon[order]
        sub = sub.isel(lon=order)
    lat_mask = (lat >= lat_min - 0.75) & (lat <= lat_max + 0.75)
    lon_mask = (lon >= lon_min - 0.75) & (lon <= lon_max + 0.75)
    sub = sub.isel(lat=lat_mask, lon=lon_mask)
    lat = np.asarray(sub.lat.values, float)
    lon = np.asarray(sub.lon.values, float)
    if lat[1] < lat[0]:
        lat = lat[::-1]
        sub = sub.isel(lat=slice(None, None, -1))
    vals = np.asarray(sub.precip.values, float)
    return {"lat": lat, "lon": lon, "values": vals, "resolution_degrees": 0.5}


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
                             obs_times: np.ndarray) -> list[dict]:
    fields = {
        "gauges": bundle["observed"],
        "analysis": bundle["members"].mean(axis=1),
        **bundle["products"],
    }
    rows = []
    for index, (product_day, obs_day) in enumerate(zip(product_times, obs_times)):
        for source, field in fields.items():
            values = np.asarray(field[index], float)
            finite = values[np.isfinite(values)]
            rows.append({
                "product_date": str(product_day),
                "gauge_observation_date": str(obs_day),
                "source_archive_date": str(
                    obs_day if source in {"gauges", "imerg_native_0p1"} else product_day
                ),
                "source": source,
                "network_mean_mm": float(np.mean(finite)) if len(finite) else np.nan,
                "network_station_sd_mm": float(np.std(finite)) if len(finite) else np.nan,
                "wet_station_fraction": float(np.mean(finite >= WET_MM)) if len(finite) else np.nan,
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
            label += " (BMD+BWDB dots)"
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
            edgecolors="black", linewidths=0.45, s=26, zorder=7,
        )
        axis.set_title(SOURCE_LABELS[source])
        figure.colorbar(image, ax=axis, shrink=0.78, label="mean residual (mm/day)")
    make_map_axes(axes, geometry, bounds)
    figure.suptitle(title + "\nstation period-mean residual: source minus BMD+BWDB gauges")
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
    axes[0, 0].set_ylabel("mm/day"); axes[0, 0].legend(); axes[0, 0].set_title("All BMD+BWDB station-days")
    axes[0, 1].bar(x, [lookup[source]["correlation"] for source in sources])
    axes[0, 1].set_xticks(x, [SOURCE_LABELS[source] for source in sources], rotation=25, ha="right")
    axes[0, 1].set_ylim(-0.1, 1.0); axes[0, 1].set_ylabel("correlation")
    axes[0, 1].set_title("Daily values versus BMD+BWDB")
    days = np.arange(observed.shape[0])
    axes[1, 0].plot(days, np.nanmean(observed, axis=1), color="black", lw=2, label="BMD+BWDB")
    for source, predicted in predictions.items():
        axes[1, 0].plot(days, np.nanmean(predicted, axis=1), lw=1.2, label=SOURCE_LABELS[source])
    axes[1, 0].set_xlabel("paired observation index (+1 day)"); axes[1, 0].set_ylabel("network mean (mm/day)")
    axes[1, 0].legend(fontsize=7, ncol=2); axes[1, 0].set_title("Daily network mean")
    observed_mean = np.nanmean(observed, axis=0)
    limit = float(np.nanpercentile(np.concatenate([observed_mean, *[np.nanmean(p, axis=0) for p in predictions.values()]]), 99))
    for source, predicted in predictions.items():
        axes[1, 1].scatter(observed_mean, np.nanmean(predicted, axis=0), s=18, alpha=0.75,
                           label=SOURCE_LABELS[source])
    axes[1, 1].plot([0, limit], [0, limit], color="black", ls="--", lw=1)
    axes[1, 1].set_xlim(0, limit); axes[1, 1].set_ylim(0, limit)
    axes[1, 1].set_xlabel(f"Station {label} mean (BMD+BWDB)"); axes[1, 1].set_ylabel("source period mean")
    axes[1, 1].legend(fontsize=7); axes[1, 1].set_title("Paired station-period mean")
    figure.suptitle(
        title + "\nStation labels are +1 day; scores are assimilated fit, not verification"
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
        axis.scatter(observed_sd, predicted_sd, s=26, alpha=0.75)
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
        axis.set_xlabel(f"Station {label} daily SD (BMD+BWDB, mm/day)")
        axis.set_ylabel("source station daily SD (mm/day)")
        axis.grid(alpha=0.2)
    figure.suptitle(
        title + f"\nSame {len(observed_sd)} BMD+BWDB stations and paired dates; assimilated fit, not verification"
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
    all_obs_times = times[all_period]
    all_product_times = all_obs_times + np.timedelta64(day_offset, "D")

    native_imerg = load_native_imerg(
        [Path(path) for path in args.native_imerg], all_obs_times
    )
    native_cpc = load_native_cpc(Path(args.cpc_dir), all_product_times, bounds)

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
    product_lookup = {day: index for index, day in enumerate(all_product_times)}
    observation_lookup = {day: index for index, day in enumerate(all_obs_times)}

    target = (years == args.year) & requested_month_mask
    target_obs_times = times[target]
    target_product_times = target_obs_times + np.timedelta64(day_offset, "D")
    product_positions = np.asarray([product_lookup[day] for day in target_product_times])
    observation_positions = np.asarray([observation_lookup[day] for day in target_obs_times])
    target_fields = {
        "analysis": np.asarray(archive["mean"][method_index, target], float),
        "chirps": np.asarray(archive["chirps"][target], float),
        "imerg_native_0p1": native_imerg["values"][observation_positions],
        "cpc_native_0p5": native_cpc["values"][product_positions],
    }
    target_spread = np.asarray(archive["spread"][method_index, target], float)
    masked_target = {source: np.where(masks[source][None], field, np.nan)
                     for source, field in target_fields.items()}

    target_dataset = next(dataset for dataset in archive["datasets"]
                          if np.isin(target_obs_times, np.asarray(dataset.time.values).astype("datetime64[D]")).all())
    target_members = load_target_members(target_dataset, method_index, target_obs_times)
    target_products = {
        source: {**grids[source], "values": target_fields[source]}
        for source in REFERENCE_ORDER
    }
    stations = load_station_bundle(target_dataset, method_index, target_obs_times, target_members,
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
        stations, target_product_times, target_obs_times
    )

    daily_rows = spatial_daily_rows(
        target_product_times, masked_target, masks, target_spread
    )
    for row in daily_rows:
        position = int(np.where(target_product_times == np.datetime64(row["date"], "D"))[0][0])
        row["product_date"] = row.pop("date")
        row["gauge_observation_date"] = str(target_obs_times[position])
        row["source_archive_date"] = (
            row["gauge_observation_date"]
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
        for product_day, obs_day, native, common in zip(
            target_product_times, target_obs_times, native_daily, common_daily
        ):
            daily_matrix_rows.append({
                "product_date": str(product_day),
                "gauge_observation_date": str(obs_day),
                "reference_source_date": str(
                    obs_day if reference == "imerg_native_0p1" else product_day
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
        year_obs_times = times[choose]
        year_product_times = year_obs_times + np.timedelta64(day_offset, "D")
        period_positions = np.asarray([product_lookup[day] for day in year_product_times])
        year_observation_positions = np.asarray([
            observation_lookup[day] for day in year_obs_times
        ])
        year_sources = {
            "analysis": np.asarray(archive["mean"][method_index, choose], float),
            "chirps": np.asarray(archive["chirps"][choose], float),
            "imerg_native_0p1": native_imerg["values"][year_observation_positions],
            "cpc_native_0p5": native_cpc["values"][period_positions],
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
                "gauge_observation_start": str(year_obs_times[0]),
                "gauge_observation_end": str(year_obs_times[-1]),
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
    anomalies = {
        source: monthly_fields[source] - climatology_mean[source]
        for source in target_fields
    }
    climatology_matrix_rows = []
    for reference in REFERENCE_ORDER:
        reference_grid = grids[reference]
        analysis_climate_native = (
            climatology_mean["analysis"] if reference == "chirps" else
            regrid_cell_average(
                climatology_mean["analysis"], archive["lat"], archive["lon"],
                reference_grid["lat"], reference_grid["lon"], fine_mask,
            )
        )
        analysis_variability_native = (
            climatology_variability["analysis"] if reference == "chirps" else
            regrid_cell_average(
                climatology_variability["analysis"], archive["lat"], archive["lon"],
                reference_grid["lat"], reference_grid["lon"], fine_mask,
            )
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
        "gauge_observation_times": target_obs_times.astype(str),
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

    title = f"{PRODUCT_NAME} — Bangladesh {requested_label} {args.year}"
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
            "product_name": PRODUCT_NAME,
            "product_full_name": PRODUCT_FULL_NAME,
            "method": args.method,
            "target_year": int(args.year),
            "target_months": list(requested_months),
            "target_period_label": requested_label,
            "target_period_tag": requested_tag,
            "target_days": int(target.sum()),
            "available_complete_period_years": [int(y) for y in available_years],
            "baseline_years": [int(y) for y in baseline_years],
            "background_day_offset": day_offset,
            "boundary_source": boundary_metadata.get("boundarySource", "geoBoundaries-BGD-ADM0"),
            "boundary_release_year": boundary_metadata.get("boundaryYear", "unknown"),
            "n_bangladesh_model_cells": int(fine_mask.sum()),
            "n_assimilated_stations": int(len(stations["station_id"])),
            "n_stations_inside_bangladesh": int(station_map.sum()),
        },
        "evidence_roles": EVIDENCE_ROLES,
        "product_agreement_matrix": matrix_rows,
        "all_station_fit": gauge_rows,
        "same_station_variability_fit": same_station_rows,
        "climatology_agreement_matrix": climatology_matrix_rows,
    }
    with (out_dir / "evaluation.json").open("w") as handle:
        json.dump(json_ready(report), handle, indent=2)
    print(f"Completed BMD+BWDB Bangladesh evaluation for {requested_label} {args.year} -> {out_dir}")


if __name__ == "__main__":
    main()
