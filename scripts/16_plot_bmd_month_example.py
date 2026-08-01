#!/usr/bin/env python
"""Plot station, metric-matrix, calibration, and spatial BMD DA diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", default="data/processed/bmd_may2018_example.npz")
    parser.add_argument("--report", default="data/processed/bmd_may2018_example.json")
    parser.add_argument(
        "--out-diagnostics", default="data/processed/bmd_may2018_diagnostics.png"
    )
    parser.add_argument("--out-spatial", default="data/processed/bmd_may2018_spatial.png")
    return parser.parse_args()


def rank_histogram(ensemble: np.ndarray, observed: np.ndarray) -> np.ndarray:
    valid = np.isfinite(observed) & np.all(np.isfinite(ensemble), axis=0)
    ranks = (ensemble[:, valid] < observed[valid][None]).sum(axis=0)
    return np.bincount(ranks, minlength=ensemble.shape[0] + 1)


def add_station_markers(axis, lon, lat, assim_idx, eval_idx, labels=False):
    axis.scatter(
        lon[assim_idx], lat[assim_idx], s=24, c="black", edgecolors="white",
        linewidths=0.35, label="assimilated", zorder=5,
    )
    axis.scatter(
        lon[eval_idx], lat[eval_idx], s=52, facecolors="none", edgecolors="#00D5FF",
        linewidths=1.5, label="withheld", zorder=6,
    )
    if labels:
        for index in eval_idx:
            axis.annotate(str(index + 1), (lon[index], lat[index]), xytext=(3, 3),
                          textcoords="offset points", fontsize=6, color="#007C91")


def main() -> None:
    args = parse_args()
    try:
        data = np.load(args.dump, allow_pickle=False)
        # Access the legacy troublemaker immediately: NpzFile defers array
        # decoding until __getitem__, so np.load alone does not raise.
        _ = data["station_name"]
    except ValueError as exc:
        if "Object arrays cannot be loaded" not in str(exc):
            raise
        # Backward compatibility for dumps created before station labels were
        # stored as fixed-width Unicode. These files are generated locally by
        # script 15; never enable pickle for an untrusted external NPZ.
        data.close()
        data = np.load(args.dump, allow_pickle=True)
        print("WARNING: loading legacy locally-generated object-string labels")
    report = json.loads(Path(args.report).read_text())
    time = data["time"].astype("datetime64[ns]").astype("datetime64[D]")
    names = data["station_name"].astype(str)
    lat, lon = data["station_lat"], data["station_lon"]
    assim_idx, eval_idx = data["assim_idx"], data["eval_idx"]
    gauge = data["gauge_mm"]
    background_station = data["background_at_stations"]
    gauge_station = (
        data["gauge_analysis_at_stations"]
        if "gauge_analysis_at_stations" in data
        else data["analysis_at_stations"]
    )
    imerg_station = (
        data["imerg_analysis_at_stations"]
        if "imerg_analysis_at_stations" in data
        else data["analysis_at_stations"]
    )
    combined_station = (
        data["combined_analysis_at_stations"]
        if "combined_analysis_at_stations" in data
        else data["analysis_at_stations"]
    )
    sequential_station = (
        data["sequential_analysis_at_stations"]
        if "sequential_analysis_at_stations" in data
        else data["analysis_at_stations"]
    )
    has_imerg = bool(report.get("scope", {}).get("imerg_assimilated", False))
    imerg_error = report.get("observation_error", {}).get("imerg") or {}
    sequential_config = report.get("sequential_update", {})
    control_text = (
        f"stride {imerg_error.get('footprint_stride', '?')}; "
        f"R inflation {imerg_error.get('correlation_variance_inflation', np.nan):.1f}×; "
        f"gauge localization {sequential_config.get('support_km', np.nan):.0f} km"
        if has_imerg and imerg_error
        else "gauge-only configuration"
    )

    # -------------------- station-space diagnostic suite --------------------
    figure, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    axis = axes[0, 0]
    add_station_markers(axis, lon, lat, assim_idx, eval_idx, labels=True)
    axis.set_xlabel("Longitude (°E)")
    axis.set_ylabel("Latitude (°N)")
    axis.set_title("A. BMD network and spatial holdout")
    axis.legend(loc="lower left", fontsize=8)
    axis.grid(alpha=0.2)

    axis = axes[0, 1]
    available = np.isfinite(gauge).T.astype(float)
    axis.imshow(available, aspect="auto", interpolation="nearest", cmap="Greys",
                vmin=0, vmax=1)
    axis.set_yticks(np.arange(len(names)))
    axis.set_yticklabels(names, fontsize=6)
    tick = np.arange(0, len(time), 5)
    axis.set_xticks(tick)
    axis.set_xticklabels([str(value)[5:] for value in time[tick]], rotation=45, ha="right")
    axis.set_title("B. Daily gauge availability")
    axis.set_xlabel("Date in 2018")

    axis = axes[0, 2]
    daily_mean = np.nanmean(gauge, axis=1)
    daily_max = np.nanmax(gauge, axis=1)
    wet_fraction = np.nanmean(gauge > 0, axis=1)
    axis.plot(time, daily_mean, color="#006D77", marker="o", ms=3, label="station mean")
    axis.plot(time, daily_max, color="#D1495B", alpha=0.75, label="station maximum")
    axis.set_ylabel("BMD rainfall (mm day$^{-1}$)")
    axis.tick_params(axis="x", rotation=45)
    twin = axis.twinx()
    twin.plot(time, wet_fraction, color="#E9C46A", lw=2, label="wet fraction")
    twin.set_ylim(0, 1)
    twin.set_ylabel("Fraction of stations wet")
    lines = axis.lines + twin.lines
    axis.legend(lines, [line.get_label() for line in lines], fontsize=8, loc="upper right")
    axis.set_title("C. Observed May rainfall sequence")

    gauge_report = report["withheld_gauges"]
    metric_names = ["RMSE", "MAE", "Bias", "CRPS", "Spread/skill", "Cover90"]
    keys = ["rmse_mm", "mae_mm", "bias_mm", "crps_mm", "spread_skill", "coverage_90"]
    matrix = np.array(
        [
            [gauge_report["background"].get(key, np.nan) for key in keys],
            [
                gauge_report.get("gauges_only", gauge_report["analysis"]).get(key, np.nan)
                for key in keys
            ],
            [
                gauge_report.get("imerg_only", gauge_report["analysis"]).get(
                    key, np.nan
                )
                for key in keys
            ],
            [
                gauge_report.get(
                    "simultaneous",
                    gauge_report.get("gauges_plus_imerg", gauge_report["analysis"]),
                ).get(key, np.nan)
                for key in keys
            ],
            [
                gauge_report.get("imerg_then_gauges", gauge_report["analysis"]).get(
                    key, np.nan
                )
                for key in keys
            ],
        ],
        dtype=float,
    )
    # Normalize each column only for colouring; annotations retain physical values.
    colour = np.empty_like(matrix)
    for column in range(matrix.shape[1]):
        values = matrix[:, column]
        lo, hi = np.nanmin(values), np.nanmax(values)
        colour[:, column] = 0.5 if hi == lo else (values - lo) / (hi - lo)
    axis = axes[1, 0]
    # Neutral shading avoids implying that the same direction is desirable for
    # error, bias, spread/skill and coverage. Interpret the annotated values.
    axis.imshow(colour, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(np.arange(len(metric_names)))
    axis.set_xticklabels(metric_names, rotation=35, ha="right")
    axis.set_yticks(np.arange(5))
    axis.set_yticklabels(
        ["Background", "Gauges", "IMERG", "Simultaneous", "IMERG → gauges"]
    )
    for row in range(5):
        for column in range(len(metric_names)):
            value = matrix[row, column]
            axis.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=8)
    axis.set_title("D. Withheld-gauge metric matrix\n(column-normalized shading)")

    observed = gauge[:, eval_idx]
    background_eval = np.moveaxis(background_station[:, :, eval_idx], 1, 0)
    gauge_eval = np.moveaxis(gauge_station[:, :, eval_idx], 1, 0)
    imerg_eval = np.moveaxis(imerg_station[:, :, eval_idx], 1, 0)
    combined_eval = np.moveaxis(combined_station[:, :, eval_idx], 1, 0)
    sequential_eval = np.moveaxis(sequential_station[:, :, eval_idx], 1, 0)
    background_mean = np.nanmean(background_eval, axis=0)
    gauge_mean = np.nanmean(gauge_eval, axis=0)
    imerg_mean = np.nanmean(imerg_eval, axis=0)
    combined_mean = np.nanmean(combined_eval, axis=0)
    sequential_mean = np.nanmean(sequential_eval, axis=0)
    valid_obs = np.isfinite(observed)
    limit = float(
        np.nanpercentile(
            np.concatenate(
                [observed[valid_obs], combined_mean[valid_obs], sequential_mean[valid_obs]]
            ),
            99,
        )
    )
    limit = max(10.0, limit)
    axis = axes[1, 1]
    axis.scatter(observed[valid_obs], background_mean[valid_obs], s=13, alpha=0.45,
                 color="#6C757D", label="background")
    axis.scatter(observed[valid_obs], gauge_mean[valid_obs], s=13, alpha=0.45,
                 color="#0077B6", label="gauges")
    if has_imerg:
        axis.scatter(observed[valid_obs], imerg_mean[valid_obs], s=13, alpha=0.4,
                     color="#F4A261", label="IMERG")
        axis.scatter(observed[valid_obs], combined_mean[valid_obs], s=13, alpha=0.45,
                     color="#D1495B", label="simultaneous")
        axis.scatter(observed[valid_obs], sequential_mean[valid_obs], s=18, alpha=0.55,
                     color="#2A9D8F", label="IMERG → gauges")
    axis.plot([0, limit], [0, limit], "k--", lw=1)
    axis.set_xlim(0, limit)
    axis.set_ylim(0, limit)
    axis.set_xlabel("Withheld BMD observation (mm day$^{-1}$)")
    axis.set_ylabel("Ensemble mean (mm day$^{-1}$)")
    axis.legend(fontsize=8)
    axis.set_title("E. Withheld station-space comparison")

    axis = axes[1, 2]
    ranks_background = rank_histogram(background_eval, observed)
    ranks_gauge = rank_histogram(gauge_eval, observed)
    ranks_imerg = rank_histogram(imerg_eval, observed)
    ranks_combined = rank_histogram(combined_eval, observed)
    ranks_sequential = rank_histogram(sequential_eval, observed)
    ranks = np.arange(len(ranks_background))
    width = 0.16
    axis.bar(ranks - 2 * width, ranks_background / ranks_background.sum(), width,
             color="#ADB5BD", label="background")
    axis.bar(ranks - width, ranks_gauge / ranks_gauge.sum(), width,
             color="#0077B6", label="gauges")
    if has_imerg:
        axis.bar(ranks, ranks_imerg / ranks_imerg.sum(), width,
                 color="#F4A261", label="IMERG")
        axis.bar(ranks + width, ranks_combined / ranks_combined.sum(), width,
                 color="#D1495B", label="simultaneous")
        axis.bar(ranks + 2 * width, ranks_sequential / ranks_sequential.sum(), width,
                 color="#2A9D8F", label="IMERG → gauges")
    axis.axhline(1 / len(ranks), color="black", ls="--", lw=1, label="flat")
    axis.set_xlabel("Ensemble rank")
    axis.set_ylabel("Relative frequency")
    axis.legend(fontsize=8)
    axis.set_title("F. Withheld-gauge rank histogram")

    figure.suptitle(
        "BMD May 2018 real-observation DA process evaluation\n"
        f"{len(assim_idx)} stations assimilated; "
        f"{len(eval_idx)} spatially spread stations withheld; "
        f"IMERG {'assimilated' if has_imerg else 'not assimilated'}; "
        f"2018 is inside checkpoint training years\n{control_text}",
        fontsize=15,
    )
    Path(args.out_diagnostics).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out_diagnostics, dpi=180)
    plt.close(figure)

    # -------------------------- spatial case suite --------------------------
    chirps, condition = data["chirps"], data["condition"]
    background = data["background"]
    analysis_gauge = data["analysis_gauge"] if "analysis_gauge" in data else data["analysis"]
    analysis_imerg = data["analysis_imerg"] if "analysis_imerg" in data else data["analysis"]
    analysis_combined = (
        data["analysis_combined"] if "analysis_combined" in data else data["analysis"]
    )
    analysis = (
        data["analysis_sequential"] if "analysis_sequential" in data else data["analysis"]
    )
    imerg = data["imerg"] if "imerg" in data else None
    valid_grid = data["valid"].astype(bool)
    background_mean_field = np.where(
        valid_grid[None], np.nan_to_num(background, nan=0.0).mean(axis=1), np.nan
    )
    gauge_mean_field = np.where(
        valid_grid[None], np.nan_to_num(analysis_gauge, nan=0.0).mean(axis=1), np.nan
    )
    imerg_mean_field = np.where(
        valid_grid[None], np.nan_to_num(analysis_imerg, nan=0.0).mean(axis=1), np.nan
    )
    combined_mean_field = np.where(
        valid_grid[None], np.nan_to_num(analysis_combined, nan=0.0).mean(axis=1), np.nan
    )
    analysis_mean_field = np.where(
        valid_grid[None], np.nan_to_num(analysis, nan=0.0).mean(axis=1), np.nan
    )
    analysis_spread = np.where(
        valid_grid[None], np.nan_to_num(analysis, nan=0.0).std(axis=1, ddof=1), np.nan
    )
    network_total = np.nansum(gauge, axis=1)
    selected_days = [
        int(np.argmax(network_total)),
        int(np.argmax(np.nanmax(gauge, axis=1))),
        int(np.argmin(np.abs(network_total - np.nanmedian(network_total)))),
    ]
    # Preserve order while preventing duplicate cases.
    selected_days = list(dict.fromkeys(selected_days))
    for candidate in np.argsort(network_total)[::-1]:
        if len(selected_days) >= 3:
            break
        if int(candidate) not in selected_days:
            selected_days.append(int(candidate))

    grid_lat, grid_lon = data["grid_lat"], data["grid_lon"]
    extent = [grid_lon[0], grid_lon[-1], grid_lat[0], grid_lat[-1]]
    n_columns = 10 if has_imerg else 7
    figure, axes = plt.subplots(
        len(selected_days), n_columns, figsize=(3.3 * n_columns, 10), constrained_layout=True
    )
    if len(selected_days) == 1:
        axes = axes[None, :]
    if has_imerg:
        column_titles = [
            "CPC condition", "IMERG 0.1° observation", "CHIRPS reference",
            "Background mean", "Gauges-only mean", "IMERG-only mean",
            "Simultaneous mean", "IMERG → gauges mean",
            "Sequential − background", "Sequential spread",
        ]
    else:
        column_titles = [
            "CPC condition", "CHIRPS reference", "Background mean",
            "Gauges-only mean", "Analysis mean", "Analysis − background",
            "Analysis spread",
        ]
    for row, day in enumerate(selected_days):
        precip_fields = [condition[day]]
        if has_imerg:
            precip_fields.append(imerg[day])
        if has_imerg:
            precip_fields.extend(
                [
                    chirps[day], background_mean_field[day], gauge_mean_field[day],
                    imerg_mean_field[day], combined_mean_field[day],
                    analysis_mean_field[day],
                ]
            )
        else:
            precip_fields.extend(
                [
                    chirps[day], background_mean_field[day], gauge_mean_field[day],
                    analysis_mean_field[day],
                ]
            )
        rain_vmax = max(
            10.0,
            max(float(np.nanpercentile(field, 99)) for field in precip_fields),
        )
        increment = analysis_mean_field[day] - background_mean_field[day]
        increment_limit = max(1.0, float(np.nanpercentile(np.abs(increment), 99)))
        spread_limit = max(1.0, float(np.nanpercentile(analysis_spread[day], 99)))
        images = []
        for column, field in enumerate(precip_fields):
            image = axes[row, column].imshow(
                field, origin="lower", extent=extent, cmap="viridis", vmin=0, vmax=rain_vmax
            )
            images.append(image)
        increment_column = len(precip_fields)
        spread_column = increment_column + 1
        images.append(
            axes[row, increment_column].imshow(
                increment, origin="lower", extent=extent, cmap="RdBu_r",
                                vmin=-increment_limit, vmax=increment_limit)
        )
        images.append(
            axes[row, spread_column].imshow(analysis_spread[day], origin="lower", extent=extent,
                                cmap="magma", vmin=0, vmax=spread_limit)
        )
        analysis_column = 7 if has_imerg else 4
        add_station_markers(axes[row, analysis_column], lon, lat, assim_idx, eval_idx)
        axes[row, 0].set_ylabel(
            f"{time[day]}\nBMD mean {np.nanmean(gauge[day]):.1f} mm",
            fontsize=9,
        )
        for column, axis in enumerate(axes[row]):
            if row == 0:
                axis.set_title(column_titles[column], fontsize=10)
            axis.set_xlim(extent[0], extent[1])
            axis.set_ylim(extent[2], extent[3])
            axis.tick_params(labelsize=7)
            figure.colorbar(images[column], ax=axis, orientation="horizontal", fraction=0.045,
                            pad=0.04)
    figure.suptitle(
        "BMD May 2018 real-observation DA spatial cases\n"
        "CHIRPS is a consistency reference; withheld BMD gauges provide the score; "
        "IMERG is thinned/correlation-weighted native V07B without fitted bias correction\n"
        + control_text,
        fontsize=14,
    )
    Path(args.out_spatial).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out_spatial, dpi=180)
    plt.close(figure)
    print(f"wrote {args.out_diagnostics}")
    print(f"wrote {args.out_spatial}")


if __name__ == "__main__":
    main()
