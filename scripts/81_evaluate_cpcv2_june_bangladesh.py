#!/usr/bin/env python
"""Evaluate the saved CPCv2 simultaneous product over Bangladesh for June.

The analysis is read from completed all-station production Zarr stores.  It
does not rerun DA or regenerate CPCv2.  June 2023 is evaluated day by day and
as a monthly field; every other complete June in the supplied stores forms a
leave-2023-out climatology.  All spatial calculations and maps use the supplied
Bangladesh ADM0 polygon, intersected with the model-valid mask.

Evidence is deliberately labelled by role: production BMD gauges diagnose
assimilated fit, IMERG diagnoses assimilated-product adherence, CHIRPS is a
learned fine-grid structural reference, and CPC is loaded on the same target
day from the checkpoint-bound source.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


METHOD_DEFAULT = "v2_simul_s04_ig010"
TARGET_YEAR = 2023
TARGET_MONTH = 6
WET_MM = 1.0
REFERENCE_ORDER = ("imerg", "chirps", "cpc_same_day")
SOURCE_LABELS = {
    "analysis": "CPCv2 simultaneous",
    "imerg": "IMERG S04",
    "chirps": "CHIRPS 0.05°",
    "cpc_same_day": "CPC same day",
    "bmd": "BMD gauges",
}
EVIDENCE_ROLES = {
    "analysis": "saved all-station CPCv2 posterior",
    "imerg": "assimilated product; agreement is adherence, not verification",
    "chirps": "training target; structural reference, not truth",
    "cpc_same_day": "same-day external gridded product; not perfect truth",
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
    parser.add_argument("--factor", type=int, default=8)
    parser.add_argument("--cpc-source-zarr", default=None)
    return parser.parse_args()


def load_shared_evaluator():
    path = Path(__file__).with_name("55_evaluate_v2_gridded_archive.py")
    spec = importlib.util.spec_from_file_location("bdhires_v2_archive_eval", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def spatial_daily_rows(times: np.ndarray, fields: dict[str, np.ndarray], mask: np.ndarray,
                       spread: np.ndarray) -> list[dict]:
    rows = []
    for index, day in enumerate(times):
        for source, field in fields.items():
            values = np.asarray(field[index], float)[mask]
            finite = values[np.isfinite(values)]
            rows.append({
                "date": str(day), "source": source,
                "domain_mean_mm": float(np.mean(finite)),
                "spatial_sd_mm": float(np.std(finite)),
                "wet_area_fraction": float(np.mean(finite >= WET_MM)),
                "spatial_p95_mm": float(np.percentile(finite, 95)),
                "posterior_spread_mm": (
                    float(np.nanmean(np.asarray(spread[index], float)[mask]))
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
                        archive: dict, fields: dict[str, np.ndarray], shared) -> dict:
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
            np.nan_to_num(field, nan=0.0), archive["lat"], archive["lon"],
            station_lat, station_lon,
        ) for source, field in fields.items() if source != "analysis"
    }
    return {
        "station_id": station_id, "station_lat": station_lat, "station_lon": station_lon,
        "observed": observed, "members": sampled_members,
        "products": sampled_products,
    }


def gauge_evaluation(bundle: dict, shared) -> tuple[list[dict], list[dict], list[dict]]:
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
            "source": source, "source_type": "gridded_product",
            "evaluation": "all_station_comparison", "independent": False,
            **deterministic_metrics(predicted, observed),
        })

    station_rows = []
    for station_index, station in enumerate(bundle["station_id"]):
        row = {
            "station_id": station,
            "lat": float(bundle["station_lat"][station_index]),
            "lon": float(bundle["station_lon"][station_index]),
            "observed_june_mean_mm": float(np.nanmean(observed[:, station_index])),
            "observed_daily_sd_mm": float(np.nanstd(observed[:, station_index])),
        }
        for source, predicted in predictions.items():
            row[f"{source}_june_mean_mm"] = float(np.nanmean(predicted[:, station_index]))
            row[f"{source}_daily_sd_mm"] = float(np.nanstd(predicted[:, station_index]))
        station_rows.append(row)

    variability_rows = []
    observed_sd = np.nanstd(observed, axis=0)
    for source, predicted in predictions.items():
        predicted_sd = np.nanstd(predicted, axis=0)
        metrics = deterministic_metrics(predicted_sd, observed_sd)
        variability_rows.append({
            "source": source,
            "metric": "station_within_june_daily_sd",
            "evaluation": "all_station_variability_fit",
            "independent": False,
            "variability_ratio": float(np.nanmean(predicted_sd**2) / np.nanmean(observed_sd**2)),
            **metrics,
        })
    return rows, station_rows, variability_rows


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


def plot_field_grid(fields: dict[str, np.ndarray], lat: np.ndarray, lon: np.ndarray,
                    mask: np.ndarray, geometry: dict, bounds, out_dir: Path, stem: str,
                    title: str, cmap: str, symmetric: bool = False,
                    station_overlay: dict | None = None) -> None:
    import matplotlib.pyplot as plt

    sources = list(fields)
    columns = 2
    rows = int(np.ceil(len(sources) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(10.8, 4.6 * rows), squeeze=False)
    values = np.concatenate([np.asarray(fields[source], float)[mask] for source in sources])
    finite = values[np.isfinite(values)]
    if symmetric:
        limit = float(np.percentile(np.abs(finite), 98))
        vmin, vmax = -limit, limit
    else:
        vmin, vmax = 0.0, float(np.percentile(finite, 99))
    for axis, source in zip(axes.ravel(), sources):
        layer = np.where(mask, fields[source], np.nan)
        image = axis.pcolormesh(lon, lat, layer, shading="auto", cmap=white_cmap(cmap),
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
    figure.suptitle(title + "\nstation June-mean residual: source minus BMD")
    figure.tight_layout()
    save_figure(figure, out_dir, "09_station_mean_residual_maps")
    plt.close(figure)


def plot_daily_series(daily_rows: list[dict], out_dir: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    dates = sorted({row["date"] for row in daily_rows})
    figure, axes = plt.subplots(3, 1, figsize=(11.5, 8.5), sharex=True)
    metrics = (("domain_mean_mm", "Bangladesh mean (mm/day)"),
               ("spatial_sd_mm", "spatial SD (mm/day)"),
               ("wet_area_fraction", "wet-area fraction (≥1 mm/day)"))
    for source in ("analysis", *REFERENCE_ORDER):
        selected = {row["date"]: row for row in daily_rows if row["source"] == source}
        for axis, (metric, ylabel) in zip(axes, metrics):
            axis.plot(dates, [selected[day][metric] for day in dates], marker="o", ms=2.5,
                      lw=1.25, label=SOURCE_LABELS[source])
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.2)
    axes[0].legend(ncol=4, fontsize=8)
    axes[-1].tick_params(axis="x", rotation=45)
    figure.suptitle(title)
    figure.tight_layout()
    save_figure(figure, out_dir, "03_daily_domain_variability")
    plt.close(figure)


def plot_matrix(rows: list[dict], out_dir: Path) -> None:
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
    axis.set_yticks(range(len(rows)), [SOURCE_LABELS[row["reference"]] for row in rows])
    axis.set_title("June 2023 CPCv2 agreement matrix (raw values; colour ranks within metric)")
    figure.colorbar(image, ax=axis, shrink=0.7, label="relative agreement within column")
    figure.tight_layout()
    save_figure(figure, out_dir, "04_product_agreement_matrix")
    plt.close(figure)


def plot_gauges(bundle: dict, gauge_rows: list[dict], out_dir: Path, title: str) -> None:
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
    axes[1, 0].set_xlabel("June day index"); axes[1, 0].set_ylabel("network mean (mm/day)")
    axes[1, 0].legend(fontsize=7, ncol=2); axes[1, 0].set_title("Daily network mean")
    observed_mean = np.nanmean(observed, axis=0)
    limit = float(np.nanpercentile(np.concatenate([observed_mean, *[np.nanmean(p, axis=0) for p in predictions.values()]]), 99))
    for source, predicted in predictions.items():
        axes[1, 1].scatter(observed_mean, np.nanmean(predicted, axis=0), s=18, alpha=0.75,
                           label=SOURCE_LABELS[source])
    axes[1, 1].plot([0, limit], [0, limit], color="black", ls="--", lw=1)
    axes[1, 1].set_xlim(0, limit); axes[1, 1].set_ylim(0, limit)
    axes[1, 1].set_xlabel("BMD station June mean"); axes[1, 1].set_ylabel("source June mean")
    axes[1, 1].legend(fontsize=7); axes[1, 1].set_title("Station climatology for June 2023")
    figure.suptitle(title + "\nBMD scores are assimilated fit, not independent verification")
    figure.tight_layout()
    save_figure(figure, out_dir, "05_all_station_gauge_fit")
    plt.close(figure)


def main() -> None:
    args = parse_args()
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
    mask = country_mask(archive["lat"], archive["lon"], geometry) & archive["valid"]
    if mask.sum() < 100:
        raise ValueError(f"Bangladesh mask contains only {mask.sum()} model cells")
    bounds = boundary_bounds(geometry)

    times = archive["time"].astype("datetime64[D]")
    years = np.asarray([int(str(day)[:4]) for day in times])
    months = np.asarray([int(str(day)[5:7]) for day in times])
    available_years = []
    for year in sorted(set(years[months == args.month].tolist())):
        choose = (years == year) & (months == args.month)
        expected = int((np.datetime64(f"{year}-{args.month % 12 + 1:02d}-01") -
                        np.datetime64(f"{year}-{args.month:02d}-01")) / np.timedelta64(1, "D")) if args.month < 12 else 31
        if choose.sum() == expected:
            available_years.append(year)
    if args.year not in available_years:
        raise ValueError(f"target June {args.year} is incomplete; complete years={available_years}")
    baseline_years = [year for year in available_years if year != args.year]
    if not baseline_years:
        raise ValueError("leave-target-year-out June climatology needs at least one other year")

    all_june = (months == args.month) & np.isin(years, available_years)
    june_archive = dict(archive)
    june_archive["time"] = archive["time"][all_june]
    june_archive["datasets"] = archive["datasets"]
    cpc_result = shared.load_same_day_cpc(june_archive, args.cpc_source_zarr)
    if cpc_result is None:
        raise RuntimeError("same-day CPC could not be loaded; pass --cpc-source-zarr explicitly")
    cpc_all_june = cpc_result[0]
    cpc_lookup = {day: cpc_all_june[index] for index, day in enumerate(june_archive["time"])}

    target = (years == args.year) & (months == args.month)
    target_times = times[target]
    target_fields = {
        "analysis": np.asarray(archive["mean"][method_index, target], float),
        "imerg": np.asarray(archive["imerg"][target], float),
        "chirps": np.asarray(archive["chirps"][target], float),
        "cpc_same_day": np.stack([cpc_lookup[day] for day in target_times]),
    }
    target_spread = np.asarray(archive["spread"][method_index, target], float)
    masked_target = {source: np.where(mask[None], field, np.nan)
                     for source, field in target_fields.items()}

    target_dataset = next(dataset for dataset in archive["datasets"]
                          if np.isin(target_times, np.asarray(dataset.time.values).astype("datetime64[D]")).all())
    target_members = load_target_members(target_dataset, method_index, target_times)
    stations = load_station_bundle(target_dataset, method_index, target_times, target_members,
                                   archive, target_fields, shared)
    stations["inside_country"] = points_in_country(
        stations["station_lat"], stations["station_lon"], geometry
    )
    if not stations["inside_country"].all():
        print(
            f"[mask] warning: {(~stations['inside_country']).sum()} station(s) lie "
            "outside the ADM0 polygon and will be omitted from maps, but retained "
            "in the requested all-station fit tables"
        )
    gauge_rows, station_rows, gauge_variability_rows = gauge_evaluation(stations, shared)

    daily_rows = spatial_daily_rows(target_times, masked_target, mask, target_spread)
    daily_matrix_rows = []
    common_fields = {}
    for source, field in masked_target.items():
        common_fields[source], common_mask = coarse_block_mean(field, args.factor, mask)
    for reference in REFERENCE_ORDER:
        fine_daily = [field_metrics(target_fields["analysis"][i], target_fields[reference][i], mask)
                      for i in range(len(target_times))]
        coarse_daily = [field_metrics(common_fields["analysis"][i], common_fields[reference][i], common_mask)
                        for i in range(len(target_times))]
        for day, fine, coarse in zip(target_times, fine_daily, coarse_daily):
            daily_matrix_rows.append({
                "date": str(day), "reference": reference,
                **{f"fine_{key}": value for key, value in fine.items()},
                **{f"common_0p4_{key}": value for key, value in coarse.items()},
            })

    monthly_fields = {
        source: masked_temporal_stat(field, mask, "mean")
        for source, field in masked_target.items()
    }
    variability_fields = {
        source: masked_temporal_stat(field, mask, "std")
        for source, field in masked_target.items()
    }
    monthly_common = {
        source: coarse_block_mean(field, args.factor, mask)[0]
        for source, field in monthly_fields.items()
    }
    variability_common = {
        source: coarse_block_mean(field, args.factor, mask)[0]
        for source, field in variability_fields.items()
    }
    matrix_rows = []
    for reference in REFERENCE_ORDER:
        mean_metrics = field_metrics(monthly_fields["analysis"], monthly_fields[reference], mask)
        variability_metrics = field_metrics(variability_fields["analysis"], variability_fields[reference], mask)
        common_mean_metrics = field_metrics(
            monthly_common["analysis"], monthly_common[reference], common_mask
        )
        common_variability_metrics = field_metrics(
            variability_common["analysis"], variability_common[reference], common_mask
        )
        selected_daily = [row for row in daily_matrix_rows if row["reference"] == reference]
        matrix_rows.append({
            "reference": reference,
            **{f"mean_{key.replace('correlation', 'r').replace('spatial_sd_ratio', 'sd_ratio')}": value
               for key, value in mean_metrics.items() if key != "n_cells"},
            **{f"variability_{key.replace('correlation', 'r').replace('spatial_sd_ratio', 'sd_ratio')}": value
               for key, value in variability_metrics.items() if key != "n_cells"},
            **{f"common_0p4_mean_{key.replace('correlation', 'r').replace('spatial_sd_ratio', 'sd_ratio')}": value
               for key, value in common_mean_metrics.items() if key != "n_cells"},
            **{f"common_0p4_variability_{key.replace('correlation', 'r').replace('spatial_sd_ratio', 'sd_ratio')}": value
               for key, value in common_variability_metrics.items() if key != "n_cells"},
            "mean_daily_spatial_r": float(np.nanmean([row["fine_correlation"] for row in selected_daily])),
            "mean_daily_crmse_mm": float(np.nanmean([row["fine_centered_rmse_mm"] for row in selected_daily])),
            "mean_daily_common_0p4_r": float(np.nanmean([row["common_0p4_correlation"] for row in selected_daily])),
            "mean_daily_common_0p4_crmse_mm": float(np.nanmean([row["common_0p4_centered_rmse_mm"] for row in selected_daily])),
        })

    yearly_fields: dict[int, dict[str, np.ndarray]] = {}
    climatology_rows = []
    for year in available_years:
        choose = (years == year) & (months == args.month)
        year_times = times[choose]
        year_sources = {
            "analysis": np.asarray(archive["mean"][method_index, choose], float),
            "imerg": np.asarray(archive["imerg"][choose], float),
            "chirps": np.asarray(archive["chirps"][choose], float),
            "cpc_same_day": np.stack([cpc_lookup[day] for day in year_times]),
        }
        yearly_fields[year] = {}
        for source, field in year_sources.items():
            mean_field = masked_temporal_stat(field, mask, "mean")
            variability_field = masked_temporal_stat(field, mask, "std")
            yearly_fields[year][f"{source}_mean"] = mean_field
            yearly_fields[year][f"{source}_variability"] = variability_field
            climatology_rows.append({
                "year": year, "source": source, "days": int(choose.sum()),
                "domain_mean_mm": float(np.nanmean(mean_field[mask])),
                "spatial_sd_of_june_mean_mm": float(np.nanstd(mean_field[mask])),
                "mean_within_june_daily_sd_mm": float(np.nanmean(variability_field[mask])),
            })
    climatology_mean = {
        source: masked_field_average(
            [yearly_fields[year][f"{source}_mean"] for year in baseline_years], mask
        ) for source in target_fields
    }
    climatology_variability = {
        source: masked_field_average(
            [yearly_fields[year][f"{source}_variability"] for year in baseline_years], mask
        ) for source in target_fields
    }
    anomalies = {source: monthly_fields[source] - climatology_mean[source]
                 for source in target_fields}
    climatology_matrix_rows = []
    for reference in REFERENCE_ORDER:
        mean_metrics = field_metrics(climatology_mean["analysis"], climatology_mean[reference], mask)
        variability_metrics = field_metrics(
            climatology_variability["analysis"], climatology_variability[reference], mask
        )
        climatology_matrix_rows.append({
            "reference": reference,
            "years": ",".join(map(str, baseline_years)),
            **{f"climatology_mean_{key}": value for key, value in mean_metrics.items()},
            **{f"climatology_daily_sd_{key}": value for key, value in variability_metrics.items()},
        })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_rows(out_dir / "daily_domain_variability.csv", daily_rows)
    write_rows(out_dir / "daily_spatial_agreement.csv", daily_matrix_rows)
    write_rows(out_dir / "june2023_product_matrix.csv", matrix_rows)
    write_rows(out_dir / "all_station_gauge_fit.csv", gauge_rows)
    write_rows(out_dir / "all_station_daily_variability_fit.csv", gauge_variability_rows)
    write_rows(out_dir / "station_june2023_summary.csv", station_rows)
    write_rows(out_dir / "available_june_climatology.csv", climatology_rows)
    write_rows(out_dir / "june_climatology_product_matrix.csv", climatology_matrix_rows)
    np.savez_compressed(
        out_dir / "june_bangladesh_fields.npz",
        lat=archive["lat"], lon=archive["lon"], bangladesh_mask=mask,
        target_times=target_times.astype(str), available_june_years=np.asarray(available_years),
        baseline_years=np.asarray(baseline_years),
        **{f"june2023_{source}_mean": field for source, field in monthly_fields.items()},
        **{f"june2023_{source}_daily_sd": field for source, field in variability_fields.items()},
        **{f"baseline_{source}_mean": field for source, field in climatology_mean.items()},
        **{f"baseline_{source}_daily_sd": field for source, field in climatology_variability.items()},
        **{f"june2023_{source}_anomaly": field for source, field in anomalies.items()},
    )

    title = f"Bangladesh June {args.year}: {args.method}"
    station_map = stations["inside_country"]
    overlay = {"lon": stations["station_lon"][station_map],
               "lat": stations["station_lat"][station_map],
               "value": np.nanmean(stations["observed"], axis=0)[station_map]}
    plot_field_grid(monthly_fields, archive["lat"], archive["lon"], mask, geometry, bounds,
                    out_dir, "01_june2023_mean_maps", title + " monthly mean", "turbo",
                    station_overlay=overlay)
    plot_field_grid(variability_fields, archive["lat"], archive["lon"], mask, geometry, bounds,
                    out_dir, "02_june2023_daily_variability_maps",
                    title + " within-June daily variability", "magma",
                    station_overlay={
                        "lon": stations["station_lon"][station_map],
                        "lat": stations["station_lat"][station_map],
                        "value": np.nanstd(stations["observed"], axis=0)[station_map],
                    })
    plot_daily_series(daily_rows, out_dir, title)
    plot_matrix(matrix_rows, out_dir)
    plot_gauges(stations, gauge_rows, out_dir, title)
    plot_field_grid(climatology_mean, archive["lat"], archive["lon"], mask, geometry, bounds,
                    out_dir, "06_leave2023out_june_climatology_maps",
                    f"June climatology ({', '.join(map(str, baseline_years))}; excludes {args.year})",
                    "turbo")
    plot_field_grid(
        climatology_variability, archive["lat"], archive["lon"], mask,
        geometry, bounds, out_dir,
        "07_leave2023out_june_variability_climatology_maps",
        f"Within-June daily-SD climatology ({', '.join(map(str, baseline_years))}; excludes {args.year})",
        "magma",
    )
    plot_field_grid(anomalies, archive["lat"], archive["lon"], mask, geometry, bounds,
                    out_dir, "08_june2023_anomaly_maps",
                    f"June {args.year} minus leave-{args.year}-out June climatology",
                    "RdBu_r", symmetric=True)
    plot_station_residual_maps(stations, mask, geometry, bounds, out_dir, title)

    report = {
        "design": {
            "method": args.method, "target_year": args.year, "month": args.month,
            "target_days": len(target_times), "ensemble_members": int(target_members.shape[1]),
            "zarr_stores": [str(path) for path in paths],
            "boundary_geojson": str(boundary_path),
            "boundary_metadata": {
                key: boundary_metadata.get(key) for key in (
                    "boundaryName", "boundaryISO", "boundaryYear", "boundaryType",
                    "boundarySource", "boundaryLicense", "licenseDetail", "licenseSource",
                    "gjDownloadURL", "simplifiedGeometryGeoJSON",
                ) if boundary_metadata.get(key) is not None
            },
            "bangladesh_grid_cells": int(mask.sum()),
            "gauge_stations_total": int(len(stations["station_id"])),
            "gauge_stations_inside_boundary_for_maps": int(stations["inside_country"].sum()),
            "available_complete_june_years": available_years,
            "leave_target_year_out_climatology_years": baseline_years,
            "same_day_cpc_source": str(cpc_result[1]),
            "footprint_factor": args.factor,
        },
        "evidence_roles": EVIDENCE_ROLES,
        "june2023_product_matrix": matrix_rows,
        "june_climatology_product_matrix": climatology_matrix_rows,
        "all_station_gauge_fit": gauge_rows,
        "all_station_daily_variability_fit": gauge_variability_rows,
        "interpretation": {
            "mask": "all map pixels and spatial scores are inside Bangladesh ADM0 and model-valid land only",
            "outside_country": "NaN and rendered white",
            "climatology": "mean across complete available Junes excluding target year; not a 30-year climate normal",
            "gauge_warning": "all production gauges were assimilated, so scores diagnose fit rather than independent skill",
            "imerg_warning": "IMERG S04 entered this simultaneous arm, so agreement diagnoses adherence",
            "chirps_warning": "CHIRPS was the training target and is not assumed to be truth",
        },
    }
    (out_dir / "evaluation.json").write_text(json.dumps(json_ready(report), indent=2, allow_nan=False) + "\n")
    (out_dir / "README.md").write_text(
        f"# Bangladesh June {args.year} CPCv2 evaluation\n\n"
        f"Saved method: `{args.method}`; {target_members.shape[1]} ensemble members; "
        f"{len(target_times)} target days. June climatology uses complete years "
        f"{', '.join(map(str, baseline_years))} and excludes {args.year}.\n\n"
        "All spatial maps and matrices use the Bangladesh ADM0 polygon intersected "
        "with the model-valid mask; everything outside is missing and plotted white. "
        "The BMD matrix is assimilated fit, IMERG is assimilated-product adherence, "
        "CHIRPS is a learned structural reference, and CPC is loaded for the same day. "
        "None is labelled perfect truth. See `evaluation.json` for the complete design.\n"
        "\nBoundary attribution: geoBoundaries `gbOpen` Bangladesh ADM0, licensed "
        "CC BY 4.0; the downloaded metadata snapshot is stored beside the GeoJSON.\n"
    )
    print(f"evaluation complete: {out_dir}")


if __name__ == "__main__":
    main()
