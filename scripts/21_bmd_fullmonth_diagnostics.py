#!/usr/bin/env python3
"""Full-month real-observation DA diagnostics pooled over spatial BMD folds.

The primary reference is always a BMD gauge withheld from the corresponding
assimilation fold.  Gridded panels describe increments, spread and texture;
they deliberately do not claim gridded error reduction because no independent
0.05-degree rainfall truth exists for this real-observation experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cartopy  # noqa: E402
import cartopy.crs as ccrs  # noqa: E402
import cartopy.feature as cfeature  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter  # noqa: E402


METHODS = {
    "Background": ("background_at_stations", "background", "#7D8597"),
    "Gauges only": ("gauge_analysis_at_stations", "analysis_gauge", "#0077B6"),
    "IMERG only": ("imerg_analysis_at_stations", "analysis_imerg", "#F4A261"),
    "Simultaneous": ("combined_analysis_at_stations", "analysis_combined", "#D1495B"),
}
THRESHOLDS = (1.0, 10.0, 25.0, 50.0)
INTENSITY_EDGES = (0.0, 1.0, 5.0, 10.0, 25.0, 50.0, np.inf)
INTENSITY_LABELS = ("0-1", "1-5", "5-10", "10-25", "25-50", ">50")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dumps", nargs="+", required=True)
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument(
        "--out-verification",
        default="data/processed/bmd_imerg_fullmonth_verification.png",
    )
    parser.add_argument(
        "--out-spatial",
        default="data/processed/bmd_imerg_fullmonth_spatial_impact.png",
    )
    parser.add_argument(
        "--out-report",
        default="data/processed/bmd_imerg_fullmonth_diagnostics.json",
    )
    parser.add_argument(
        "--cartopy-data-dir",
        default="data/static/cartopy",
        help="Writable persistent cache for Natural Earth boundary files.",
    )
    parser.add_argument("--reliability-threshold", type=float, default=10.0)
    return parser.parse_args()


def crps_values(ensemble: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Fair finite-ensemble CRPS at each point; ensemble is ``(member, ...)``."""
    ensemble = np.asarray(ensemble, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    valid = np.isfinite(observed) & np.all(np.isfinite(ensemble), axis=0)
    output = np.full(observed.shape, np.nan, dtype=np.float64)
    if not valid.any():
        return output
    members = ensemble[:, valid]
    truth = observed[valid]
    member_count = members.shape[0]
    first = np.mean(np.abs(members - truth[None]), axis=0)
    second = np.sum(
        np.abs(members[:, None, :] - members[None, :, :]), axis=(0, 1)
    ) / (2.0 * member_count * (member_count - 1))
    output[valid] = first - second
    return output


def rank_histogram(ensemble: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Tie-randomised rank counts, important for zero-inflated precipitation."""
    valid = np.isfinite(observed) & np.all(np.isfinite(ensemble), axis=0)
    members = ensemble[:, valid]
    truth = observed[valid]
    rng = np.random.default_rng(202405)
    below = (members < truth[None]).sum(axis=0)
    equal = (members == truth[None]).sum(axis=0)
    ranks = below + (rng.random(len(below)) * (equal + 1)).astype(int)
    return np.bincount(ranks, minlength=ensemble.shape[0] + 1)


def reliability(probability: np.ndarray, event: np.ndarray, bins: int = 10):
    edges = np.linspace(0.0, 1.0, bins + 1)
    centres, frequencies, counts = [], [], []
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        inside = (probability >= lower) & (
            probability <= upper if index == bins - 1 else probability < upper
        )
        counts.append(int(inside.sum()))
        centres.append(float(np.mean(probability[inside])) if inside.any() else np.nan)
        frequencies.append(float(np.mean(event[inside])) if inside.any() else np.nan)
    return np.asarray(centres), np.asarray(frequencies), np.asarray(counts)


def radial_spectrum(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.nan_to_num(field, nan=0.0).astype(np.float64)
    values -= values.mean()
    power = np.abs(np.fft.fftshift(np.fft.fft2(values))) ** 2
    height, width = values.shape
    y, x = np.indices((height, width))
    radius = np.hypot(y - height // 2, x - width // 2).astype(int)
    bins = min(height, width) // 2
    totals = np.bincount(radius.ravel(), weights=power.ravel(), minlength=bins)[:bins]
    counts = np.bincount(radius.ravel(), minlength=bins)[:bins]
    spectrum = np.divide(totals, counts, out=np.full(bins, np.nan), where=counts > 0)
    return np.arange(1, bins), spectrum[1:]


def haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    value = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(np.clip(value, 0, 1)))


def pooled_metrics(ensemble: np.ndarray, observed: np.ndarray) -> dict:
    valid = np.isfinite(observed) & np.all(np.isfinite(ensemble), axis=0)
    members = ensemble[:, valid]
    truth = observed[valid]
    mean = np.mean(members, axis=0)
    error = mean - truth
    spread = np.sqrt(np.mean(np.var(members, axis=0, ddof=1)))
    rmse = np.sqrt(np.mean(error**2))
    low, high = np.quantile(members, [0.05, 0.95], axis=0)
    return {
        "n": int(valid.sum()),
        "crps_mm": float(np.nanmean(crps_values(ensemble, observed))),
        "rmse_mm": float(rmse),
        "mae_mm": float(np.mean(np.abs(error))),
        "bias_mm": float(np.mean(error)),
        "spread_skill": float(spread / rmse) if rmse else np.nan,
        "coverage_90": float(np.mean((truth >= low) & (truth <= high))),
        "correlation": (
            float(np.corrcoef(mean, truth)[0, 1])
            if np.std(mean) > 0 and np.std(truth) > 0
            else np.nan
        ),
    }


def extent_from_centres(lat: np.ndarray, lon: np.ndarray) -> list[float]:
    dy = float(np.median(np.diff(lat)))
    dx = float(np.median(np.diff(lon)))
    return [lon[0] - dx / 2, lon[-1] + dx / 2, lat[0] - dy / 2, lat[-1] + dy / 2]


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def transformed_innovations(data, report: dict) -> tuple[np.ndarray, np.ndarray]:
    """Gauge and IMERG prior innovations using exactly the assimilated R models."""
    stats_path = Path(report["scope"]["checkpoint_stats"])
    stats = json.loads(stats_path.read_text())
    from bdhires.transforms import PrecipTransform

    transform = PrecipTransform.from_dict(stats["precip_transform"])
    background_stations = np.asarray(data["background_at_stations"], dtype=np.float64)
    gauge = np.asarray(data["gauge_mm"], dtype=np.float64)
    background_transformed = transform.forward(np.clip(background_stations, 0, None))
    gauge_transformed = transform.forward(np.clip(gauge, 0, None))
    gauge_cfg = report["observation_error"]["gauges"]
    gauge_r = gauge_cfg["sigma_transformed"] ** 2 + gauge_cfg[
        "representativeness_transformed"
    ] ** 2
    gauge_expected = np.sqrt(
        np.var(background_transformed, axis=1, ddof=1) + gauge_r
    )
    gauge_normalised = (
        gauge_transformed - np.mean(background_transformed, axis=1)
    ) / np.maximum(gauge_expected, 1e-6)
    gauge_normalised = gauge_normalised[np.isfinite(gauge_normalised)]

    background = np.asarray(data["background"], dtype=np.float64)
    imerg = np.asarray(data["imerg"], dtype=np.float64)
    imerg_error = np.asarray(data["imerg_random_error"], dtype=np.float64)
    valid = np.asarray(data["valid"]).astype(bool)
    factor = background.shape[-1] // imerg.shape[-1]
    days, members, height, width = background.shape
    coarse = np.nan_to_num(background, nan=0.0).reshape(
        days, members, height // factor, factor, width // factor, factor
    ).mean(axis=(3, 5))
    predicted_transformed = transform.forward(np.clip(coarse, 0, None))
    observation_transformed = transform.forward(np.clip(imerg, 0, None))
    imerg_cfg = report["observation_error"]["imerg"]
    lower = np.clip(imerg - imerg_error, 0, None)
    upper = np.clip(imerg + imerg_error, 0, None)
    native_sigma = 0.5 * np.abs(transform.forward(upper) - transform.forward(lower))
    sigma = np.maximum(native_sigma, imerg_cfg["sigma_floor_transformed"])
    satellite_r = (
        sigma**2 + imerg_cfg["representativeness_transformed"] ** 2
    ) * imerg_cfg["correlation_variance_inflation"]
    block_valid = valid.reshape(
        height // factor, factor, width // factor, factor
    ).mean(axis=(1, 3)) >= 0.999
    stride = int(imerg_cfg["footprint_stride"])
    thinning = np.zeros(block_valid.shape, dtype=bool)
    offset = stride // 2
    thinning[offset::stride, offset::stride] = True
    keep = block_valid & thinning
    expected = np.sqrt(np.var(predicted_transformed, axis=1, ddof=1) + satellite_r)
    normalised = (
        observation_transformed - np.mean(predicted_transformed, axis=1)
    ) / np.maximum(expected, 1e-6)
    keep_all = np.broadcast_to(keep, normalised.shape) & np.isfinite(normalised)
    return gauge_normalised, normalised[keep_all]


def add_map_context(
    axis,
    projection,
    extent,
    longitude_ticks,
    latitude_ticks,
    *,
    label_left,
    label_bottom,
):
    """Add Bangladesh-region political boundaries and geographic labels."""
    axis.set_extent(extent, crs=projection)
    axis.add_feature(
        cfeature.COASTLINE.with_scale("10m"),
        edgecolor="black",
        facecolor="none",
        linewidth=0.65,
        zorder=5,
    )
    axis.add_feature(
        cfeature.BORDERS.with_scale("10m"),
        edgecolor="black",
        facecolor="none",
        linewidth=0.65,
        zorder=5,
    )
    axis.add_feature(
        cfeature.STATES.with_scale("10m"),
        edgecolor="black",
        facecolor="none",
        linewidth=0.30,
        linestyle=":",
        alpha=0.65,
        zorder=5,
    )
    gridlines = axis.gridlines(
        crs=projection,
        draw_labels=True,
        x_inline=False,
        y_inline=False,
        xlocs=mticker.FixedLocator(longitude_ticks),
        ylocs=mticker.FixedLocator(latitude_ticks),
        linewidth=0.30,
        color="black",
        alpha=0.25,
        linestyle=":",
    )
    gridlines.top_labels = False
    gridlines.right_labels = False
    gridlines.left_labels = label_left
    gridlines.bottom_labels = label_bottom
    gridlines.xformatter = LongitudeFormatter()
    gridlines.yformatter = LatitudeFormatter()
    gridlines.xlabel_style = {"size": 7}
    gridlines.ylabel_style = {"size": 7}


def show_map(
    axis,
    field,
    extent,
    cmap,
    lower,
    upper,
    title,
    unit,
    figure,
    projection,
    longitude_ticks,
    latitude_ticks,
    *,
    label_left,
    label_bottom,
):
    colours = plt.get_cmap(cmap).copy()
    colours.set_bad("white")
    image = axis.imshow(
        field,
        origin="lower",
        extent=extent,
        cmap=colours,
        vmin=lower,
        vmax=upper,
        interpolation="nearest",
        transform=projection,
    )
    add_map_context(
        axis,
        projection,
        extent,
        longitude_ticks,
        latitude_ticks,
        label_left=label_left,
        label_bottom=label_bottom,
    )
    axis.set_title(title, fontsize=10)
    bar = figure.colorbar(image, ax=axis, shrink=0.82)
    bar.set_label(unit, fontsize=8)
    bar.ax.tick_params(labelsize=7)


def main() -> None:
    args = parse_args()
    cartopy_data_dir = Path(args.cartopy_data_dir).resolve()
    cartopy_data_dir.mkdir(parents=True, exist_ok=True)
    cartopy.config["data_dir"] = cartopy_data_dir
    if len(args.dumps) != len(args.reports):
        raise ValueError("--dumps and --reports must have equal length")
    if len(args.dumps) < 2:
        raise ValueError("full-month diagnostics require multiple spatial folds")
    summary = json.loads(Path(args.summary).read_text())
    reports = [json.loads(Path(path).read_text()) for path in args.reports]

    station_chunks = {name: [] for name in METHODS}
    observed_chunks = []
    distance_chunks = []
    station_added_value = None
    station_gauge_crps = None
    station_sim_crps = None
    daily_crps = {name: [] for name in METHODS}
    fold_differences = []
    all_eval_indices = []

    spatial_sums = {}
    spatial_count = 0
    spectrum_sums = {}
    spectrum_count = {}
    distance_edges = np.asarray([0, 1, 2, 3, 5, 8, 12, 20, 30, 50, np.inf])
    increment_sums = {"Gauges only": np.zeros(len(distance_edges) - 1), "Simultaneous": np.zeros(len(distance_edges) - 1)}
    increment_counts = {key: np.zeros(len(distance_edges) - 1, dtype=np.int64) for key in increment_sums}
    innovation_gauge = innovation_imerg = None

    reference_ids = reference_names = reference_lat = reference_lon = None
    grid_lat = grid_lon = valid_grid = None
    dates = None
    member_count = None

    for fold_position, (dump_path, report) in enumerate(zip(args.dumps, reports)):
        with np.load(dump_path, allow_pickle=False) as data:
            station_ids = np.asarray(data["station_id"]).astype(str)
            station_names = np.asarray(data["station_name"]).astype(str)
            station_lat = np.asarray(data["station_lat"], dtype=float)
            station_lon = np.asarray(data["station_lon"], dtype=float)
            eval_idx = np.asarray(data["eval_idx"], dtype=int)
            assim_idx = np.asarray(data["assim_idx"], dtype=int)
            time = np.asarray(data["time"]).astype("datetime64[ns]").astype("datetime64[D]")
            observed = np.asarray(data["gauge_mm"], dtype=float)[:, eval_idx]
            if reference_ids is None:
                reference_ids, reference_names = station_ids, station_names
                reference_lat, reference_lon = station_lat, station_lon
                dates = time
                member_count = int(np.asarray(data["background"]).shape[1])
                grid_lat = np.asarray(data["grid_lat"], dtype=float)
                grid_lon = np.asarray(data["grid_lon"], dtype=float)
                valid_grid = np.asarray(data["valid"]).astype(bool)
                station_added_value = np.full(len(station_ids), np.nan)
                station_gauge_crps = np.full(len(station_ids), np.nan)
                station_sim_crps = np.full(len(station_ids), np.nan)
                innovation_gauge, innovation_imerg = transformed_innovations(data, report)
            else:
                if not np.array_equal(station_ids, reference_ids):
                    raise ValueError("fold dumps disagree on station ordering")
                if not np.array_equal(time, dates):
                    raise ValueError("fold dumps disagree on dates")
            all_eval_indices.extend(eval_idx.tolist())
            observed_chunks.append(observed.ravel())

            nearest = np.full(len(eval_idx), np.inf)
            for position, index in enumerate(eval_idx):
                nearest[position] = np.min(
                    haversine_km(
                        station_lat[index], station_lon[index],
                        station_lat[assim_idx], station_lon[assim_idx],
                    )
                )
            distance_chunks.append(np.broadcast_to(nearest[None], observed.shape).ravel())

            fold_ensembles = {}
            for name, (station_key, _, _) in METHODS.items():
                values = np.asarray(data[station_key], dtype=float)[:, :, eval_idx]
                ensemble = np.moveaxis(values, 1, 0)
                fold_ensembles[name] = ensemble
                station_chunks[name].append(ensemble.reshape(ensemble.shape[0], -1))
                point_crps = crps_values(ensemble, observed)
                daily_crps[name].append(np.nanmean(point_crps, axis=1))
            gauge_station = np.nanmean(
                crps_values(fold_ensembles["Gauges only"], observed), axis=0
            )
            simultaneous_station = np.nanmean(
                crps_values(fold_ensembles["Simultaneous"], observed), axis=0
            )
            station_gauge_crps[eval_idx] = gauge_station
            station_sim_crps[eval_idx] = simultaneous_station
            station_added_value[eval_idx] = gauge_station - simultaneous_station
            fold_differences.append(
                float(np.nanmean(simultaneous_station - gauge_station))
            )

            background = np.asarray(data["background"], dtype=float)
            gauge_grid = np.asarray(data["analysis_gauge"], dtype=float)
            imerg_grid = np.asarray(data["analysis_imerg"], dtype=float)
            simultaneous_grid = np.asarray(data["analysis_combined"], dtype=float)
            grid_methods = {
                "Background": background,
                "Gauges only": gauge_grid,
                "IMERG only": imerg_grid,
                "Simultaneous": simultaneous_grid,
            }
            means = {name: np.nanmean(value, axis=1) for name, value in grid_methods.items()}
            spreads = {name: np.nanstd(value, axis=1, ddof=1) for name, value in grid_methods.items()}
            fields = {
                "gauge_abs_increment": np.abs(means["Gauges only"] - means["Background"]),
                "simultaneous_abs_increment": np.abs(means["Simultaneous"] - means["Background"]),
                "satellite_abs_contribution": np.abs(means["Simultaneous"] - means["Gauges only"]),
                "gauge_spread_change": spreads["Gauges only"] - spreads["Background"],
                "simultaneous_spread_change": spreads["Simultaneous"] - spreads["Background"],
            }
            for key, value in fields.items():
                spatial_sums.setdefault(key, np.zeros(valid_grid.shape, dtype=np.float64))
                spatial_sums[key][valid_grid] += np.nansum(value[:, valid_grid], axis=0)
            spatial_count += len(time)

            latitude_grid, longitude_grid = np.meshgrid(grid_lat, grid_lon, indexing="ij")
            distance_cells = np.full(valid_grid.shape, np.inf)
            resolution_km = float(np.median(np.diff(grid_lat))) * 111.0
            for index in assim_idx:
                distance_cells = np.minimum(
                    distance_cells,
                    haversine_km(
                        latitude_grid, longitude_grid,
                        station_lat[index], station_lon[index],
                    ) / resolution_km,
                )
            for name in increment_sums:
                increment = np.abs(means[name] - means["Background"])
                for bin_index, (lower, upper) in enumerate(zip(distance_edges[:-1], distance_edges[1:])):
                    cells = valid_grid & (distance_cells >= lower) & (distance_cells < upper)
                    values = increment[:, cells]
                    finite = np.isfinite(values)
                    increment_sums[name][bin_index] += np.nansum(values)
                    increment_counts[name][bin_index] += int(finite.sum())

            for name, (_, grid_key, _) in METHODS.items():
                members = np.asarray(data[grid_key], dtype=float)[:, 0]
                for day in range(len(time)):
                    k, power = radial_spectrum(np.where(valid_grid, members[day], 0.0))
                    spectrum_sums[name] = spectrum_sums.get(name, np.zeros_like(power)) + power
                    spectrum_count[name] = spectrum_count.get(name, 0) + 1
            if fold_position == 0:
                context_fields = {
                    "CHIRPS context": np.asarray(data["chirps"], dtype=float),
                    "Raw IMERG context": np.repeat(
                        np.repeat(np.asarray(data["imerg"], dtype=float), 2, axis=1),
                        2,
                        axis=2,
                    )[:, : valid_grid.shape[0], : valid_grid.shape[1]],
                }
                for name, values in context_fields.items():
                    for day in range(len(time)):
                        k, power = radial_spectrum(np.where(valid_grid, values[day], 0.0))
                        spectrum_sums[name] = spectrum_sums.get(name, np.zeros_like(power)) + power
                        spectrum_count[name] = spectrum_count.get(name, 0) + 1

    expected_indices = list(range(len(reference_ids)))
    if sorted(all_eval_indices) != expected_indices:
        raise ValueError("folds must withhold every station exactly once")
    observed_all = np.concatenate(observed_chunks)
    distance_all = np.concatenate(distance_chunks)
    ensembles = {
        name: np.concatenate(chunks, axis=1) for name, chunks in station_chunks.items()
    }
    scores = {name: pooled_metrics(value, observed_all) for name, value in ensembles.items()}

    crps_point = {name: crps_values(value, observed_all) for name, value in ensembles.items()}
    rank_fractions = {}
    rank_deviation = {}
    for name, ensemble in ensembles.items():
        counts = rank_histogram(ensemble, observed_all)
        fraction = counts / max(1, counts.sum())
        rank_fractions[name] = fraction
        rank_deviation[name] = float(np.sum(np.abs(fraction - 1 / len(fraction))))

    # ---------------- verification suite ---------------------------------
    figure, axes = plt.subplots(3, 3, figsize=(19, 15), constrained_layout=True)
    ranks = np.arange(member_count + 1)
    width = 0.8 / len(METHODS)
    axis = axes[0, 0]
    for position, (name, (_, _, colour)) in enumerate(METHODS.items()):
        axis.bar(
            ranks - 0.4 + width / 2 + position * width,
            rank_fractions[name], width, color=colour, alpha=0.9, label=name,
        )
    axis.axhline(1 / (member_count + 1), color="black", ls="--", lw=1, label="flat")
    axis.set(xlabel="Rank of withheld BMD among members", ylabel="Relative frequency")
    axis.set_title("A. Rank histogram\nU = under-dispersed; dome = over-dispersed", fontsize=10)
    axis.legend(fontsize=7, frameon=False)
    axis.grid(alpha=0.2)

    axis = axes[0, 1]
    maximum = 0.0
    spread_skill_curves = {}
    for name, ensemble in ensembles.items():
        mean = np.mean(ensemble, axis=0)
        spread = np.std(ensemble, axis=0, ddof=1)
        error = mean - observed_all
        valid = np.isfinite(spread) & np.isfinite(error)
        edges = np.unique(np.quantile(spread[valid], np.linspace(0, 1, 11)))
        centres, rmses, counts = [], [], []
        minimum_bin_count = max(5, int(valid.sum() // 50))
        for lower, upper in zip(edges[:-1], edges[1:]):
            inside = valid & (spread >= lower) & (spread <= upper)
            if inside.sum() >= minimum_bin_count:
                centres.append(float(np.mean(spread[inside])))
                rmses.append(float(np.sqrt(np.mean(error[inside] ** 2))))
                counts.append(int(inside.sum()))
        maximum = max(maximum, *(centres or [0]), *(rmses or [0]))
        spread_skill_curves[name] = {"spread_mm": centres, "rmse_mm": rmses, "n": counts}
        axis.plot(centres, rmses, marker="o", ms=4, color=METHODS[name][2], label=name)
    axis.plot([0, maximum], [0, maximum], "k--", lw=1, label="1:1")
    axis.set(xlabel="Ensemble spread (mm day$^{-1}$)", ylabel="RMSE of ensemble mean (mm day$^{-1}$)")
    axis.set_title("B. Spread-skill at withheld BMD\nbelow line = over-confident", fontsize=10)
    axis.legend(fontsize=7, frameon=False)
    axis.grid(alpha=0.2)

    axis = axes[0, 2]
    positions = np.arange(len(INTENSITY_LABELS))
    crps_by_intensity = {}
    for method_position, (name, (_, _, colour)) in enumerate(METHODS.items()):
        values, counts = [], []
        for lower, upper in zip(INTENSITY_EDGES[:-1], INTENSITY_EDGES[1:]):
            inside = (observed_all >= lower) & (observed_all < upper)
            counts.append(int(np.sum(inside)))
            values.append(float(np.nanmean(crps_point[name][inside])) if inside.any() else np.nan)
        crps_by_intensity[name] = {"crps_mm": values, "n": counts}
        axis.bar(
            positions - 0.4 + width / 2 + method_position * width,
            values, width, color=colour, label=name,
        )
    axis.set_xticks(positions, INTENSITY_LABELS)
    axis.set(xlabel="Observed BMD intensity (mm day$^{-1}$)", ylabel="CRPS (mm day$^{-1}$)")
    axis.set_title("C. CRPS by intensity\nwithheld gauges are the target", fontsize=10)
    axis.legend(fontsize=7, frameon=False)
    axis.grid(axis="y", alpha=0.2)

    axis = axes[1, 0]
    threshold = args.reliability_threshold
    reliability_report = {}
    event = observed_all >= threshold
    for name, ensemble in ensembles.items():
        probability = np.mean(ensemble >= threshold, axis=0)
        centres, frequencies, counts = reliability(probability, event)
        reliability_report[name] = {
            "forecast_probability": centres, "observed_frequency": frequencies, "n": counts,
        }
        axis.plot(centres, frequencies, marker="o", ms=4, color=METHODS[name][2], label=name)
    axis.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel=f"Forecast probability of ≥{threshold:g} mm", ylabel="Observed frequency")
    axis.set_title(f"D. Reliability at ≥{threshold:g} mm\nwithheld BMD events", fontsize=10)
    axis.legend(fontsize=7, frameon=False)
    axis.grid(alpha=0.2)

    axis = axes[1, 1]
    brier_scores = {}
    threshold_positions = np.arange(len(THRESHOLDS))
    for method_position, (name, ensemble) in enumerate(ensembles.items()):
        values = []
        for level in THRESHOLDS:
            probability = np.mean(ensemble >= level, axis=0)
            values.append(float(np.mean((probability - (observed_all >= level)) ** 2)))
        brier_scores[name] = values
        axis.bar(
            threshold_positions - 0.4 + width / 2 + method_position * width,
            values, width, color=METHODS[name][2], label=name,
        )
    axis.set_xticks(threshold_positions, [f"≥{value:g}" for value in THRESHOLDS])
    axis.set(xlabel="BMD rainfall threshold (mm day$^{-1}$)", ylabel="Brier score")
    axis.set_title("E. Event probability skill\nlower is better", fontsize=10)
    axis.legend(fontsize=7, frameon=False)
    axis.grid(axis="y", alpha=0.2)

    axis = axes[1, 2]
    spectrum_styles = {
        "Background": "--", "Gauges only": "-", "IMERG only": "-",
        "Simultaneous": "-", "CHIRPS context": ":", "Raw IMERG context": ":",
    }
    spectrum_colours = {name: value[2] for name, value in METHODS.items()}
    spectrum_colours.update({"CHIRPS context": "#5E548E", "Raw IMERG context": "#2A9D8F"})
    spectra = {}
    for name in spectrum_sums:
        spectra[name] = spectrum_sums[name] / spectrum_count[name]
        axis.loglog(k, spectra[name], ls=spectrum_styles[name], color=spectrum_colours[name], label=name)
    axis.set(xlabel="Wavenumber (higher = finer scale)", ylabel="Power")
    axis.set_title("F. Power spectrum\nstructural comparison only; no gridded truth", fontsize=10)
    axis.legend(fontsize=6.5, frameon=False)
    axis.grid(alpha=0.2, which="both")

    axis = axes[2, 0]
    increment_profile = {}
    finite_edges = distance_edges.copy()
    finite_edges[-1] = max(distance_edges[-2] * 1.5, 60)
    centres = np.sqrt(np.maximum(finite_edges[:-1], 0.2) * finite_edges[1:])
    for name in increment_sums:
        values = np.divide(
            increment_sums[name], increment_counts[name],
            out=np.full_like(increment_sums[name], np.nan), where=increment_counts[name] > 0,
        )
        increment_profile[name] = {"distance_cells": centres, "increment_mm": values, "n": increment_counts[name]}
        axis.plot(centres, values, marker="o", ms=5, color=METHODS[name][2], label=name)
    axis.set_xscale("log")
    axis.set(xlabel="Distance to nearest assimilated BMD gauge (0.05° cells)", ylabel="Mean |analysis − background| (mm day$^{-1}$)")
    axis.set_title("G. Increment reach\nsimultaneous also contains dense satellite guidance", fontsize=10)
    axis.legend(fontsize=7, frameon=False)
    axis.grid(alpha=0.2, which="both")

    axis = axes[2, 1]
    innovation_report = {}
    for name, values, colour in [
        ("BMD gauges", innovation_gauge, "#0077B6"),
        ("IMERG footprints", innovation_imerg, "#F4A261"),
    ]:
        finite = values[np.isfinite(values)]
        axis.hist(finite, bins=60, range=(-5, 5), density=True, histtype="step", lw=1.8, color=colour, label=f"{name} (sd {np.std(finite):.2f})")
        innovation_report[name] = {
            "n": int(len(finite)), "mean": float(np.mean(finite)), "sd": float(np.std(finite)),
            "fraction_abs_gt_3": float(np.mean(np.abs(finite) > 3)),
        }
    normal_x = np.linspace(-5, 5, 400)
    axis.plot(normal_x, np.exp(-normal_x**2 / 2) / np.sqrt(2 * np.pi), "k-", lw=1.2, label="N(0,1)")
    axis.set(xlim=(-5, 5), xlabel=r"$(y-H(x_b))/\sqrt{\sigma_b^2+R}$", ylabel="Density")
    axis.set_title("H. Normalised prior innovations\nsd ≈ 1 when B and R are consistent", fontsize=10)
    axis.legend(fontsize=7, frameon=False)
    axis.grid(alpha=0.2)

    axis = axes[2, 2]
    axis.axis("off")
    gate = summary["fusion_gate"]
    bootstrap = gate.get("day_block_bootstrap", {})
    ci = bootstrap.get("ci_95_mm", [None, None])
    ci_text = "unavailable" if ci[0] is None else f"[{ci[0]:+.2f}, {ci[1]:+.2f}]"
    summary_text = (
        f"days                    {len(dates)}\n"
        f"members                 {member_count}\n"
        f"withheld station-days   {len(observed_all)}\n"
        f"spatial folds           {len(args.dumps)}\n\n"
        f"gauges-only CRPS        {scores['Gauges only']['crps_mm']:.2f}\n"
        f"simultaneous CRPS       {scores['Simultaneous']['crps_mm']:.2f}\n"
        f"sim − gauges CRPS       {gate['pooled_difference_mm']:+.2f}\n"
        f"day-block 95% CI        {ci_text}\n"
        f"sim fold wins           {gate['simultaneous_fold_wins']}/{gate['required_fold_wins']} required\n\n"
        f"gauge cover90           {scores['Gauges only']['coverage_90']:.2f}\n"
        f"sim cover90             {scores['Simultaneous']['coverage_90']:.2f}\n"
        f"gauge innovation sd     {innovation_report['BMD gauges']['sd']:.2f}\n"
        f"IMERG innovation sd     {innovation_report['IMERG footprints']['sd']:.2f}\n\n"
        f"FORMAL GATE             {'PASS' if gate['passes'] else 'FAIL'}\n"
        f"provisional method      {summary['provisional_recommendation']}\n\n"
        "2018 is inside prior training;\n"
        "this is development evidence,\n"
        "not independent product skill."
    )
    axis.text(0, 1, summary_text, va="top", ha="left", family="monospace", fontsize=9.5)
    axis.set_title("I. Consistency and selection summary", fontsize=10)

    figure.suptitle(
        "Full-May real-observation data-assimilation verification suite\n"
        f"{len(dates)} exact BMD 03:00-to-03:00 UTC days | {member_count} members | "
        f"{len(reference_ids)} stations withheld once across {len(args.dumps)} folds\n"
        "All skill and calibration panels use withheld BMD; spectral curves are context only",
        fontsize=14,
    )
    Path(args.out_verification).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out_verification, dpi=150)
    plt.close(figure)

    # ---------------- spatial impact, explicitly not spatial verification --
    spatial = {key: np.where(valid_grid, value / spatial_count, np.nan) for key, value in spatial_sums.items()}
    map_extent = extent_from_centres(grid_lat, grid_lon)
    map_projection = ccrs.PlateCarree()
    longitude_ticks = np.arange(
        np.ceil(map_extent[0]), np.floor(map_extent[1]) + 1, 1
    )
    latitude_ticks = np.arange(
        np.ceil(map_extent[2]), np.floor(map_extent[3]) + 1, 1
    )
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(19, 11),
        constrained_layout=True,
        subplot_kw={"projection": map_projection},
    )
    increment_limit = max(
        1.0,
        float(np.nanpercentile(
            np.stack([spatial["gauge_abs_increment"], spatial["simultaneous_abs_increment"]]), 99
        )),
    )
    contribution_limit = max(1.0, float(np.nanpercentile(spatial["satellite_abs_contribution"], 99)))
    spread_limit = max(
        1.0,
        float(np.nanpercentile(
            np.abs(np.stack([spatial["gauge_spread_change"], spatial["simultaneous_spread_change"]])), 98
        )),
    )
    map_panels = [
        (0, 0, spatial["gauge_abs_increment"], "magma", 0, increment_limit, "A. Mean |gauges-only − background|"),
        (0, 1, spatial["simultaneous_abs_increment"], "magma", 0, increment_limit, "B. Mean |simultaneous − background|"),
        (0, 2, spatial["satellite_abs_contribution"], "magma", 0, contribution_limit, "C. Mean |simultaneous − gauges-only|\nsatellite's additional movement"),
        (1, 0, spatial["gauge_spread_change"], "RdBu_r", -spread_limit, spread_limit, "D. Gauges-only spread change\nblue = reduced spread"),
        (1, 1, spatial["simultaneous_spread_change"], "RdBu_r", -spread_limit, spread_limit, "E. Simultaneous spread change\nblue = reduced spread"),
    ]
    for row, column, field, cmap, lower, upper, title in map_panels:
        show_map(
            axes[row, column],
            field,
            map_extent,
            cmap,
            lower,
            upper,
            title,
            "mm day$^{-1}$",
            figure,
            map_projection,
            longitude_ticks,
            latitude_ticks,
            label_left=column == 0,
            label_bottom=row == 1,
        )

    axis = axes[1, 2]
    base = np.where(valid_grid, 0.5, np.nan)
    axis.imshow(
        base,
        origin="lower",
        extent=map_extent,
        cmap="Greys",
        vmin=0,
        vmax=1,
        alpha=0.18,
        transform=map_projection,
    )
    add_map_context(
        axis,
        map_projection,
        map_extent,
        longitude_ticks,
        latitude_ticks,
        label_left=False,
        label_bottom=True,
    )
    station_limit = max(0.1, float(np.nanpercentile(np.abs(station_added_value), 95)))
    image = axis.scatter(
        reference_lon, reference_lat, c=station_added_value, s=70,
        cmap="BrBG", vmin=-station_limit, vmax=station_limit,
        edgecolors="black", linewidths=0.5,
        transform=map_projection,
        zorder=6,
    )
    strongest = np.argsort(np.nan_to_num(np.abs(station_added_value), nan=-1))[-6:]
    for index in strongest:
        axis.annotate(
            reference_names[index],
            (reference_lon[index], reference_lat[index]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=6,
            transform=map_projection,
            zorder=7,
        )
    axis.set_title("F. Cross-validated satellite added value at BMD\ngreen = simultaneous CRPS better", fontsize=10)
    bar = figure.colorbar(image, ax=axis, shrink=0.82)
    bar.set_label("gauges-only CRPS − simultaneous CRPS (mm day$^{-1}$)", fontsize=8)
    bar.ax.tick_params(labelsize=7)
    for axis in axes.flat:
        if axis is axes[1, 2]:
            continue
        axis.scatter(
            reference_lon,
            reference_lat,
            s=7,
            c="black",
            alpha=0.45,
            transform=map_projection,
            zorder=6,
        )
    figure.suptitle(
        "Full-May DA spatial impact across rotated BMD networks — not gridded validation\n"
        "Maps A–E show how the posterior changed; only panel F uses independent withheld observations",
        fontsize=14,
    )
    Path(args.out_spatial).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out_spatial, dpi=150)
    plt.close(figure)

    diagnostic_report = {
        "scope": {
            "dates": [str(value) for value in dates],
            "days": int(len(dates)),
            "members": member_count,
            "folds": len(args.dumps),
            "stations": len(reference_ids),
            "withheld_station_days": int(len(observed_all)),
            "primary_reference": "BMD gauges withheld exactly once across spatial folds",
            "spatial_note": "Grid maps show posterior impact, not error reduction against a gridded truth.",
            "map_projection": "Cartopy PlateCarree with 10m Natural Earth boundaries",
            "cartopy_data_dir": str(cartopy_data_dir),
        },
        "pooled_metrics": scores,
        "fold_simultaneous_minus_gauges_crps_mm": fold_differences,
        "rank_histogram_fraction": rank_fractions,
        "rank_histogram_deviation": rank_deviation,
        "spread_skill_curves": spread_skill_curves,
        "crps_by_bmd_intensity": {
            "bins_mm": list(INTENSITY_LABELS), **crps_by_intensity,
        },
        "reliability_at_mm": threshold,
        "reliability": reliability_report,
        "brier_score": {"thresholds_mm": list(THRESHOLDS), **brier_scores},
        "normalised_innovations": innovation_report,
        "increment_vs_gauge_distance": increment_profile,
        "station_added_value": {
            reference_names[index]: {
                "station_id": reference_ids[index],
                "lat": reference_lat[index],
                "lon": reference_lon[index],
                "gauges_only_crps_mm": station_gauge_crps[index],
                "simultaneous_crps_mm": station_sim_crps[index],
                "gauges_minus_simultaneous_crps_mm": station_added_value[index],
            }
            for index in range(len(reference_ids))
        },
        "selection": summary["fusion_gate"],
        "provisional_recommendation": summary["provisional_recommendation"],
        "caveat": "May 2018 is inside checkpoint training and is not independent model validation.",
    }
    Path(args.out_report).write_text(
        json.dumps(json_safe(diagnostic_report), indent=2, allow_nan=False) + "\n"
    )
    print(f"wrote {args.out_verification}")
    print(f"wrote {args.out_spatial}")
    print(f"wrote {args.out_report}")
    print(
        f"full-month gate: simultaneous - gauges CRPS "
        f"{summary['fusion_gate']['pooled_difference_mm']:+.3f}; "
        f"recommendation {summary['provisional_recommendation']}"
    )


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    main()
