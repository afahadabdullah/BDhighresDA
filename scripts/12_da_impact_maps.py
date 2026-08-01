#!/usr/bin/env python
"""Spatial maps of what the assimilation actually did, before and after.

Reads the ``.npz`` from ``scripts/10_osse.py --dump``.  The verification suite
(``11_da_diagnostics.py``) says whether the DA is statistically sound; this says
*where* it helped, which is the question a domain-wide score cannot answer.

Two figures.

CASES -- one row per selected day, six columns:
    A  CHIRPS truth, with assimilated stations as filled circles and withheld
       stations as open squares.  Every other panel shares those markers, so the
       eye can always ask "is this change near an observation?"
    B  background ensemble mean          (before)
    C  analysis ensemble mean            (after)
    D  background error, mean - truth
    E  analysis error, mean - truth
    F  ERROR REDUCTION, |background error| - |analysis error|.
       Green means the assimilation improved that cell, brown means it made it
       worse.  This is the panel that matters: a DA that lowers the domain score
       while degrading large regions is not doing what it looks like it is
       doing.

AGGREGATE -- averaged over every day in the dump:
    A  mean error reduction, with the station network on top
    B  mean |analysis - background|, the raw size of the increment
    C  change in ensemble spread, negative where assimilation reduced uncertainty
    D  error reduction against distance to the nearest assimilated station --
       the spatial version of the localisation question
    E  withheld-station scatter, background and analysis against truth
    F  fraction of land area improved, by intensity of the observed rainfall

    python scripts/12_da_impact_maps.py --dump data/processed/osse_dump.npz
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", required=True)
    parser.add_argument(
        "--cases",
        type=int,
        default=3,
        help="days to map, spread across the wet-to-dry range of the dump",
    )
    parser.add_argument(
        "--out-cases", default="data/processed/da_impact_cases.png"
    )
    parser.add_argument(
        "--out-aggregate", default="data/processed/da_impact_aggregate.png"
    )
    parser.add_argument("--out-report", default="data/processed/da_impact.json")
    return parser.parse_args()


def add_stations(axis, lat, lon, assim_idx, eval_idx, size: float = 22) -> None:
    """Assimilated = filled, withheld = open.  Consistent on every panel."""
    axis.scatter(
        lon[assim_idx], lat[assim_idx], s=size, c="white", edgecolors="black",
        linewidths=0.7, marker="o", zorder=6, label="assimilated",
    )
    axis.scatter(
        lon[eval_idx], lat[eval_idx], s=size * 1.3, facecolors="none",
        edgecolors="black", linewidths=1.1, marker="s", zorder=6,
        label="withheld",
    )


def show(axis, values, extent, cmap, vmin, vmax, title, unit, figure):
    colours = plt.get_cmap(cmap).copy()
    colours.set_bad("white")
    image = axis.imshow(
        values, origin="lower", extent=extent, cmap=colours,
        vmin=vmin, vmax=vmax, interpolation="nearest",
    )
    axis.set_title(title, fontsize=10)
    axis.tick_params(labelsize=7)
    bar = figure.colorbar(image, ax=axis, shrink=0.84)
    bar.set_label(unit, fontsize=8)
    bar.ax.tick_params(labelsize=7)
    return image


def main() -> None:
    args = parse_args()
    data = np.load(args.dump, allow_pickle=False)
    background = data["background"]      # (D, M, H, W)
    analysis = data["analysis"]
    truth = data["truth"]                # (D, H, W)
    valid = data["valid"].astype(bool)
    assim_idx, eval_idx = data["assim_idx"], data["eval_idx"]
    lat, lon = data["station_lat"], data["station_lon"]
    truth_at_stations = data["truth_at_stations"]
    pseudo_satellite = (
        bool(data["pseudo_satellite_enabled"])
        if "pseudo_satellite_enabled" in data.files
        else False
    )
    day_labels = [str(d) for d in data["days"]]
    n_days = truth.shape[0]

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from bdhires.grids import get_grid

    grid = get_grid(str(data["grid_name"]))
    extent = [grid.lon_min, grid.lon_max, grid.lat_min, grid.lat_max]

    background_mean = background.mean(axis=1)
    analysis_mean = analysis.mean(axis=1)
    background_error = background_mean - truth
    analysis_error = analysis_mean - truth
    # Positive = the assimilation moved this cell closer to the truth.
    reduction = np.abs(background_error) - np.abs(analysis_error)
    increment = analysis_mean - background_mean
    spread_change = analysis.std(axis=1, ddof=1) - background.std(axis=1, ddof=1)

    # -- CASES ---------------------------------------------------------------
    domain_mean = np.array(
        [np.nanmean(np.where(valid, truth[d], np.nan)) for d in range(n_days)]
    )
    order = np.argsort(domain_mean)
    chosen = order[
        np.linspace(0, n_days - 1, min(args.cases, n_days)).astype(int)
    ][::-1]

    rows = len(chosen)
    figure, axes = plt.subplots(
        rows, 6, figsize=(24, 4.3 * rows), constrained_layout=True, squeeze=False
    )
    for row, day in enumerate(chosen):
        finite = valid & np.isfinite(truth[day])
        rain_max = max(5.0, float(np.nanpercentile(truth[day][finite], 99)))
        error_limit = max(
            2.0,
            float(np.nanpercentile(np.abs(background_error[day][finite]), 99)),
        )
        panels = [
            (np.where(valid, truth[day], np.nan), "viridis", 0, rain_max,
             "A.  CHIRPS truth", "mm day$^{-1}$"),
            (np.where(valid, background_mean[day], np.nan), "viridis", 0, rain_max,
             "B.  Background mean (before)", "mm day$^{-1}$"),
            (np.where(valid, analysis_mean[day], np.nan), "viridis", 0, rain_max,
             "C.  Analysis mean (after)", "mm day$^{-1}$"),
            (np.where(valid, background_error[day], np.nan), "RdBu_r",
             -error_limit, error_limit, "D.  Background error", "mm day$^{-1}$"),
            (np.where(valid, analysis_error[day], np.nan), "RdBu_r",
             -error_limit, error_limit, "E.  Analysis error", "mm day$^{-1}$"),
            # BrBG: negative -> brown, positive -> green. Verified against the
            # colormap rather than assumed -- PuOr_r puts POSITIVE at orange,
            # which contradicted the caption in the first version of this figure.
            (np.where(valid, reduction[day], np.nan), "BrBG",
             -error_limit, error_limit,
             "F.  Error reduction\ngreen = DA helped, brown = DA hurt",
             "mm day$^{-1}$"),
        ]
        for column, (values, cmap, vmin, vmax, title, unit) in enumerate(panels):
            axis = axes[row, column]
            show(axis, values, extent, cmap, vmin, vmax,
                 title if row == 0 else "", unit, figure)
            add_stations(axis, lat, lon, assim_idx, eval_idx)
            if column == 0:
                axis.set_ylabel("Latitude (deg N)", fontsize=8)
            if row == rows - 1:
                axis.set_xlabel("Longitude (deg E)", fontsize=8)
        improved = float(
            np.mean(reduction[day][valid & np.isfinite(truth[day])] > 0)
        )
        axes[row, 0].annotate(
            f"{day_labels[day]}\n{domain_mean[day]:.1f} mm day$^{{-1}}$\n"
            f"{improved:.0%} of land improved",
            xy=(0, 0.5), xycoords="axes fraction",
            xytext=(-92, 0), textcoords="offset points",
            ha="center", va="center", fontsize=10, fontweight="bold",
        )
    axes[0, 0].legend(fontsize=7.5, loc="lower left", framealpha=0.85)
    figure.suptitle(
        "BDhighresDA - where did the assimilation change the field?\n"
        f"{Path(args.dump).name}   |   network '{data['network']}'   |   "
        f"{len(assim_idx)} assimilated (filled circles), "
        f"{len(eval_idx)} withheld (open squares)\n"
        "Column F is the one to read: positive means the analysis is closer to "
        "CHIRPS than the background was",
        fontsize=14,
    )
    Path(args.out_cases).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out_cases, dpi=110)
    plt.close(figure)

    # -- AGGREGATE -----------------------------------------------------------
    figure, axes = plt.subplots(2, 3, figsize=(19, 11), constrained_layout=True)
    finite_all = valid[None] & np.isfinite(truth)

    def day_mean(stack: np.ndarray) -> np.ndarray:
        """Average over days on land only.

        Ocean cells are NaN on every day, so a plain nanmean over the full grid
        warns about empty slices and returns NaN anyway.  Restrict to land.
        """
        out = np.full(stack.shape[1:], np.nan, dtype=np.float64)
        out[valid] = np.nanmean(stack[:, valid], axis=0)
        return out

    mean_reduction = day_mean(reduction)
    mean_increment = day_mean(np.abs(increment))
    mean_spread_change = day_mean(spread_change)

    limit = float(np.nanpercentile(np.abs(mean_reduction), 98))
    show(axes[0, 0], mean_reduction, extent, "BrBG", -limit, limit,
         "A.  Mean error reduction\ngreen = DA helped, brown = DA hurt",
         "mm day$^{-1}$", figure)
    add_stations(axes[0, 0], lat, lon, assim_idx, eval_idx)
    axes[0, 0].legend(fontsize=7.5, loc="lower left", framealpha=0.85)

    show(axes[0, 1], mean_increment, extent, "magma", 0,
         float(np.nanpercentile(mean_increment, 99)),
         "B.  Mean |analysis - background|\nhow much the DA moved things",
         "mm day$^{-1}$", figure)
    add_stations(axes[0, 1], lat, lon, assim_idx, eval_idx)

    spread_limit = float(np.nanpercentile(np.abs(mean_spread_change), 98))
    show(axes[0, 2], mean_spread_change, extent, "RdBu_r",
         -spread_limit, spread_limit,
         "C.  Change in ensemble spread\nblue = uncertainty reduced",
         "mm day$^{-1}$", figure)
    add_stations(axes[0, 2], lat, lon, assim_idx, eval_idx)

    # D. error reduction vs distance to the nearest assimilated station
    lat_grid, lon_grid = np.meshgrid(grid.lat, grid.lon, indexing="ij")
    distance = np.full(grid.shape, np.inf)
    for index in assim_idx:
        distance = np.minimum(
            distance,
            np.hypot(
                (lat_grid - lat[index]) / grid.res,
                (lon_grid - lon[index]) / grid.res * np.cos(np.radians(lat_grid)),
            ),
        )
    axis = axes[1, 0]
    edges = np.array([0, 1, 2, 3, 5, 8, 12, 20, 30, 1e9])
    centres, values, fractions = [], [], []
    for low, high in zip(edges[:-1], edges[1:]):
        inside = valid & (distance >= low) & (distance < high)
        cells = np.broadcast_to(inside, reduction.shape) & np.isfinite(reduction)
        if inside.sum() > 20:
            centres.append(float(distance[inside].mean()))
            values.append(float(np.nanmean(reduction[cells])))
            fractions.append(float(np.mean(reduction[cells] > 0)))
    axis.plot(centres, values, marker="o", ms=5, color="#2166ac")
    axis.axhline(0.0, color="black", lw=1)
    axis.set_xscale("log")
    axis.set_xlabel("Distance to nearest assimilated station (cells)")
    axis.set_ylabel("Mean error reduction (mm day$^{-1}$)")
    axis.set_title(
        "D.  Error reduction versus gauge distance\n"
        + (
            "dense satellite is present: distance is descriptive only"
            if pseudo_satellite
            else "above zero = still helping"
        ),
        fontsize=10,
    )
    axis.grid(alpha=0.25, which="both")

    # E. withheld-station scatter
    axis = axes[1, 1]
    rows_idx = np.clip(
        np.round((lat - grid.lat_min) / grid.res - 0.5).astype(int), 0, grid.nlat - 1
    )
    cols_idx = np.clip(
        np.round((lon - grid.lon_min) / grid.res - 0.5).astype(int), 0, grid.nlon - 1
    )
    truth_eval = truth_at_stations[:, eval_idx].ravel()
    background_eval = background_mean[:, rows_idx, cols_idx][:, eval_idx].ravel()
    analysis_eval = analysis_mean[:, rows_idx, cols_idx][:, eval_idx].ravel()
    axis.scatter(truth_eval, background_eval, s=12, alpha=0.45,
                 color="#3b78b4", label="background")
    axis.scatter(truth_eval, analysis_eval, s=12, alpha=0.45,
                 color="#c1442e", label="analysis")
    top = float(np.nanmax([truth_eval.max(), background_eval.max(),
                           analysis_eval.max()]))
    axis.plot([0, top], [0, top], color="black", ls="--", lw=1)
    axis.set_xlabel("CHIRPS truth at withheld stations (mm day$^{-1}$)")
    axis.set_ylabel("Ensemble mean (mm day$^{-1}$)")
    axis.set_title("E.  Withheld stations, before and after\n"
                   "closer to the 1:1 line is better", fontsize=10)
    axis.legend(fontsize=8, frameon=False)
    axis.grid(alpha=0.25)

    # F. improved fraction by observed intensity
    axis = axes[1, 2]
    bins = [0, 1, 5, 10, 25, 50, 1e9]
    labels = ["0-1", "1-5", "5-10", "10-25", "25-50", ">50"]
    flat_truth = truth[finite_all]
    flat_reduction = reduction[finite_all]
    improved, counts = [], []
    for low, high in zip(bins[:-1], bins[1:]):
        inside = (flat_truth >= low) & (flat_truth < high)
        counts.append(int(inside.sum()))
        improved.append(
            float(np.mean(flat_reduction[inside] > 0)) if inside.sum() > 20 else np.nan
        )
    axis.bar(np.arange(len(labels)), improved, color="#2166ac")
    axis.axhline(0.5, color="black", ls="--", lw=1, label="no better than chance")
    axis.set_xticks(np.arange(len(labels)), labels)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Observed intensity (mm day$^{-1}$)")
    axis.set_ylabel("Fraction of cells improved")
    axis.set_title("F.  Where does the DA help?\nby rainfall intensity", fontsize=10)
    axis.legend(fontsize=8, frameon=False)
    axis.grid(axis="y", alpha=0.25)

    overall = float(np.mean(flat_reduction > 0))
    figure.suptitle(
        "BDhighresDA - assimilation impact averaged over "
        f"{n_days} days   |   network '{data['network']}'   |   "
        f"{overall:.0%} of land cells improved overall\n"
        f"{len(assim_idx)} assimilated (filled circles), {len(eval_idx)} "
        "withheld (open squares)",
        fontsize=14,
    )
    figure.savefig(args.out_aggregate, dpi=115)
    plt.close(figure)

    report = {
        "dump": str(args.dump),
        "network": str(data["network"]),
        "days": n_days,
        "fraction_of_land_improved": overall,
        "mean_error_reduction_mm": float(np.nanmean(flat_reduction)),
        "improved_by_intensity": dict(zip(labels, improved)),
        "cells_by_intensity": dict(zip(labels, counts)),
        "reduction_vs_distance": {
            "distance_cells": centres,
            "mean_reduction_mm": values,
            "fraction_improved": fractions,
        },
    }
    Path(args.out_report).write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(f"land cells improved overall: {overall:.1%}")
    print(f"mean error reduction: {report['mean_error_reduction_mm']:+.3f} mm/day")
    print("improved fraction by intensity:")
    for label, value in zip(labels, improved):
        print(f"  {label:>6s} mm  {value:.1%}" if np.isfinite(value)
              else f"  {label:>6s} mm  (too few cells)")
    print(f"\nwrote {args.out_cases}")
    print(f"wrote {args.out_aggregate}")
    print(f"wrote {args.out_report}")


if __name__ == "__main__":
    main()
