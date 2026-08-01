#!/usr/bin/env python
"""Spatial reconstruction suite and metric matrices for a dumped CHIRPS OSSE.

This complements ``11_da_diagnostics.py`` and ``12_da_impact_maps.py``.  It is
deliberately scale-aware: a dense 0.1-degree pseudo-satellite makes coarse
footprint skill comparatively easy, while the scientifically interesting claim
is whether the prior reconstructs realistic 0.05-degree departures *within*
those footprints.
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

from bdhires.grids import get_grid  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", required=True)
    parser.add_argument("--cases", type=int, default=4)
    parser.add_argument(
        "--out-reconstruction",
        default="data/processed/osse_reconstruction_maps.png",
    )
    parser.add_argument(
        "--out-matrix", default="data/processed/osse_metric_matrix.png"
    )
    parser.add_argument(
        "--out-report", default="data/processed/osse_scale_summary.json"
    )
    return parser.parse_args()


def ensemble_score(ensemble: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Scores after flattening all non-member dimensions."""
    members = ensemble.shape[0]
    ens = ensemble.reshape(members, -1)
    obs = truth.reshape(-1)
    keep = np.isfinite(obs) & np.all(np.isfinite(ens), axis=0)
    ens, obs = ens[:, keep], obs[keep]
    if not obs.size:
        return {}
    mean = ens.mean(axis=0)
    difference = mean - obs
    spread = ens.std(axis=0, ddof=1) if members > 1 else np.zeros_like(mean)
    pairwise = 0.0
    for first in range(members):
        for second in range(members):
            pairwise += np.mean(np.abs(ens[first] - ens[second]))
    crps = np.mean(np.abs(ens - obs[None])) - 0.5 * pairwise / members**2
    low, high = np.quantile(ens, [0.05, 0.95], axis=0)
    correlation = (
        float(np.corrcoef(mean, obs)[0, 1])
        if mean.std() > 0 and obs.std() > 0
        else float("nan")
    )
    return {
        "rmse_mm": float(np.sqrt(np.mean(difference**2))),
        "mae_mm": float(np.mean(np.abs(difference))),
        "bias_mm": float(np.mean(difference)),
        "crps_mm": float(crps),
        "correlation": correlation,
        "spread_mm": float(np.mean(spread)),
        "coverage_90": float(np.mean((obs >= low) & (obs <= high))),
        "variance_ratio": float(np.var(mean) / np.var(obs))
        if np.var(obs) > 0
        else float("nan"),
        "n": int(obs.size),
    }


def coarsen(field: np.ndarray, factor: int) -> np.ndarray:
    """Non-overlapping physical block means over the final two dimensions."""
    height, width = field.shape[-2:]
    if height % factor or width % factor:
        raise ValueError(f"shape {field.shape} is not divisible by factor {factor}")
    shape = (*field.shape[:-2], height // factor, factor, width // factor, factor)
    return field.reshape(shape).mean(axis=(-3, -1))


def upsample_blocks(field: np.ndarray, factor: int) -> np.ndarray:
    return np.repeat(np.repeat(field, factor, axis=-2), factor, axis=-1)


def subgrid_component(field: np.ndarray, factor: int) -> np.ndarray:
    """0.05-degree departure from each field's own 0.1-degree footprint mean."""
    return field - upsample_blocks(coarsen(field, factor), factor)


def bilinear_sample(field: np.ndarray, lat: np.ndarray, lon: np.ndarray, grid) -> np.ndarray:
    """Bilinear physical-space sample for arrays ending in (H, W)."""
    values = np.nan_to_num(field, nan=0.0)
    row = (np.asarray(lat) - grid.lat[0]) / grid.res
    col = (np.asarray(lon) - grid.lon[0]) / grid.res
    r0 = np.floor(row).astype(int)
    c0 = np.floor(col).astype(int)
    r0 = np.clip(r0, 0, grid.nlat - 1)
    c0 = np.clip(c0, 0, grid.nlon - 1)
    r1 = np.clip(r0 + 1, 0, grid.nlat - 1)
    c1 = np.clip(c0 + 1, 0, grid.nlon - 1)
    wr = row - r0
    wc = col - c0
    return (
        values[..., r0, c0] * (1 - wr) * (1 - wc)
        + values[..., r1, c0] * wr * (1 - wc)
        + values[..., r0, c1] * (1 - wr) * wc
        + values[..., r1, c1] * wr * wc
    )


def combined_score_by_day(ensemble: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Treat day/cell pairs as samples but preserve ensemble as axis zero."""
    return ensemble_score(np.moveaxis(ensemble, 1, 0), truth)


def directional_change(background: dict, analysis: dict, metric: str) -> float:
    """Positive always means analysis is better."""
    before, after = background.get(metric, np.nan), analysis.get(metric, np.nan)
    if not np.isfinite(before) or not np.isfinite(after):
        return float("nan")
    if metric == "correlation":
        return float(100.0 * (after - before))  # percentage-point style delta
    if metric == "coverage_90":
        b_error, a_error = abs(before - 0.90), abs(after - 0.90)
        return float(100.0 * (b_error - a_error) / b_error) if b_error else np.nan
    before = abs(before) if metric == "bias_mm" else before
    after = abs(after) if metric == "bias_mm" else after
    return float(100.0 * (before - after) / before) if before else np.nan


def heatmap(
    axis,
    values: np.ndarray,
    row_labels: list[str],
    column_labels: list[str],
    title: str,
    *,
    cmap: str = "viridis",
    symmetric: bool = False,
    formats: list[str] | None = None,
) -> None:
    finite = values[np.isfinite(values)]
    if symmetric:
        limit = max(float(np.max(np.abs(finite))) if finite.size else 1.0, 1e-6)
        image = axis.imshow(values, cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
    else:
        image = axis.imshow(values, cmap=cmap, aspect="auto")
    axis.set_xticks(np.arange(len(column_labels)), column_labels, rotation=20, ha="right")
    axis.set_yticks(np.arange(len(row_labels)), row_labels)
    axis.set_title(title, fontsize=11)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            fmt = formats[row] if formats else ".2f"
            label = "--" if not np.isfinite(value) else format(value, fmt)
            axis.text(column, row, label, ha="center", va="center", fontsize=8,
                      color="black", bbox={"facecolor": "white", "alpha": 0.62,
                                             "edgecolor": "none", "pad": 1.0})
    plt.colorbar(image, ax=axis, fraction=0.045, pad=0.03)


def plot_reconstruction(data, grid, factor: int, n_cases: int, output: Path) -> None:
    background = data["background"]
    analysis = data["analysis"]
    truth = data["truth"]
    valid = data["valid"].astype(bool)
    days = data["days"].astype(str)
    station_lat, station_lon = data["station_lat"], data["station_lon"]
    assim_idx, eval_idx = data["assim_idx"], data["eval_idx"]
    satellite = (
        data["pseudo_satellite_mm"]
        if "pseudo_satellite_mm" in data.files
        else coarsen(truth, factor)
    )

    domain_mean = np.nanmean(truth, axis=(1, 2))
    order = np.argsort(domain_mean)
    quantiles = np.linspace(0.10, 0.95, min(max(n_cases, 1), len(order)))
    selected = np.unique(order[np.round(quantiles * (len(order) - 1)).astype(int)])
    n_rows = len(selected)
    selected_bg_mean = np.nanmean(background[selected], axis=1)
    selected_an_mean = np.nanmean(analysis[selected], axis=1)
    rain_values = np.concatenate(
        [
            truth[selected][:, valid].ravel(),
            selected_bg_mean[:, valid].ravel(),
            selected_an_mean[:, valid].ravel(),
            analysis[selected, 0][:, valid].ravel(),
        ]
    )
    rain_max = max(1.0, float(np.nanpercentile(rain_values, 99.5)))
    selected_errors = np.concatenate(
        [
            (selected_bg_mean - truth[selected])[:, valid].ravel(),
            (selected_an_mean - truth[selected])[:, valid].ravel(),
        ]
    )
    error_max = max(1.0, float(np.nanpercentile(np.abs(selected_errors), 99)))
    selected_spread = np.nanstd(analysis[selected], axis=1, ddof=1)
    spread_max = max(
        1.0, float(np.nanpercentile(selected_spread[:, valid], 99))
    )
    figure, axes = plt.subplots(
        n_rows, 8, figsize=(24, 3.65 * n_rows), squeeze=False, constrained_layout=True
    )
    extent = [
        grid.lon[0] - grid.res / 2,
        grid.lon[-1] + grid.res / 2,
        grid.lat[0] - grid.res / 2,
        grid.lat[-1] + grid.res / 2,
    ]
    precip_images, error_images, spread_images = [], [], []
    titles = [
        "CHIRPS truth\n0.05° nature run",
        "Pseudo-satellite\n0.1° footprints",
        "Background mean\n0.05°",
        "Analysis mean\n0.05°",
        "Analysis member 1\ntexture check",
        "Background error\nmean − truth",
        "Analysis error\nmean − truth",
        "Analysis spread\nstandard deviation",
    ]
    for column, title in enumerate(titles):
        axes[0, column].set_title(title, fontsize=10)

    for row, day_index in enumerate(selected):
        bg_mean = np.nanmean(background[day_index], axis=0)
        an_mean = np.nanmean(analysis[day_index], axis=0)
        errors = [bg_mean - truth[day_index], an_mean - truth[day_index]]
        spread = np.nanstd(analysis[day_index], axis=0, ddof=1)
        panels = [
            truth[day_index],
            satellite[day_index],
            bg_mean,
            an_mean,
            analysis[day_index, 0],
            errors[0],
            errors[1],
            spread,
        ]
        for column, field in enumerate(panels):
            if column == 1:
                image = axes[row, column].imshow(
                    field, origin="lower", extent=extent, cmap="viridis",
                    vmin=0, vmax=rain_max, interpolation="nearest", aspect="auto"
                )
                precip_images.append(image)
            elif column <= 4:
                image = axes[row, column].imshow(
                    field, origin="lower", extent=extent, cmap="viridis",
                    vmin=0, vmax=rain_max, aspect="auto"
                )
                precip_images.append(image)
            elif column <= 6:
                image = axes[row, column].imshow(
                    field, origin="lower", extent=extent, cmap="RdBu_r",
                    vmin=-error_max, vmax=error_max, aspect="auto"
                )
                error_images.append(image)
            else:
                image = axes[row, column].imshow(
                    field, origin="lower", extent=extent, cmap="magma",
                    vmin=0, vmax=spread_max, aspect="auto"
                )
                spread_images.append(image)
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])

        for column in (1, 3):
            axes[row, column].scatter(
                station_lon[assim_idx], station_lat[assim_idx], s=10,
                c="black", marker="o", linewidths=0.2, edgecolors="white",
                label="assimilated" if row == 0 else None,
            )
            axes[row, column].scatter(
                station_lon[eval_idx], station_lat[eval_idx], s=20,
                facecolors="none", edgecolors="#00e5ff", marker="o", linewidths=1.0,
                label="withheld" if row == 0 else None,
            )
        bg_rmse = np.sqrt(np.nanmean(errors[0][valid] ** 2))
        an_rmse = np.sqrt(np.nanmean(errors[1][valid] ** 2))
        truth_sub = subgrid_component(truth[day_index], factor)
        an_sub = subgrid_component(an_mean, factor)
        sub_keep = valid & np.isfinite(truth_sub) & np.isfinite(an_sub)
        sub_corr = (
            np.corrcoef(truth_sub[sub_keep], an_sub[sub_keep])[0, 1]
            if sub_keep.sum() > 2
            else np.nan
        )
        axes[row, 0].set_ylabel(
            f"{days[day_index]}\nmean {domain_mean[day_index]:.1f} mm d⁻¹\n"
            f"RMSE {bg_rmse:.1f}→{an_rmse:.1f}\nsubgrid r={sub_corr:.2f}",
            fontsize=8.5,
        )

    figure.colorbar(precip_images[-1], ax=axes[:, :5], orientation="horizontal",
                    fraction=0.025, pad=0.025, label="Daily precipitation (mm day⁻¹)")
    figure.colorbar(error_images[-1], ax=axes[:, 5:7], orientation="horizontal",
                    fraction=0.025, pad=0.025, label="Error (mm day⁻¹)")
    figure.colorbar(spread_images[-1], ax=axes[:, 7], orientation="horizontal",
                    fraction=0.025, pad=0.025, label="Spread (mm day⁻¹)")
    axes[0, 3].legend(loc="lower left", fontsize=6.5, frameon=True)
    figure.suptitle(
        "CHIRPS OSSE reconstruction: coarse observations versus fine-scale recovery\n"
        "Filled black = assimilated gauges; cyan rings = withheld gauges. "
        "Colour ranges are shared across all cases.",
        fontsize=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=130)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    data = np.load(args.dump, allow_pickle=False)
    background = data["background"]       # (D, M, H, W)
    analysis = data["analysis"]
    truth = data["truth"]                 # (D, H, W)
    factor = int(data["satellite_factor"]) if "satellite_factor" in data.files else 2
    grid = get_grid(str(data["grid_name"]))

    plot_reconstruction(
        data, grid, factor, args.cases, Path(args.out_reconstruction)
    )

    # Combine day and space while keeping ensemble-member axis first.
    bg_all = np.moveaxis(background, 1, 0)
    an_all = np.moveaxis(analysis, 1, 0)
    field_scores = {
        "background": ensemble_score(bg_all, truth),
        "analysis": ensemble_score(an_all, truth),
    }
    background_coarse, analysis_coarse, truth_coarse = (
        coarsen(background, factor),
        coarsen(analysis, factor),
        coarsen(truth, factor),
    )
    coarse_scores = {
        "background": ensemble_score(np.moveaxis(background_coarse, 1, 0), truth_coarse),
        "analysis": ensemble_score(np.moveaxis(analysis_coarse, 1, 0), truth_coarse),
    }
    background_subgrid = subgrid_component(background, factor)
    analysis_subgrid = subgrid_component(analysis, factor)
    truth_subgrid = subgrid_component(truth, factor)
    subgrid_scores = {
        "background": ensemble_score(np.moveaxis(background_subgrid, 1, 0), truth_subgrid),
        "analysis": ensemble_score(np.moveaxis(analysis_subgrid, 1, 0), truth_subgrid),
    }

    station_lat, station_lon = data["station_lat"], data["station_lon"]
    eval_idx = data["eval_idx"]
    background_stations = bilinear_sample(background, station_lat, station_lon, grid)
    analysis_stations = bilinear_sample(analysis, station_lat, station_lon, grid)
    truth_stations = data["truth_at_stations"]
    withheld_scores = {
        "background": ensemble_score(
            np.moveaxis(background_stations[:, :, eval_idx], 1, 0),
            truth_stations[:, eval_idx],
        ),
        "analysis": ensemble_score(
            np.moveaxis(analysis_stations[:, :, eval_idx], 1, 0),
            truth_stations[:, eval_idx],
        ),
    }

    scopes = {
        "Fine field": field_scores,
        "Coarse 0.1°": coarse_scores,
        "Subgrid 0.05°": subgrid_scores,
        "Withheld gauges": withheld_scores,
    }
    raw_metrics = [
        ("RMSE", "rmse_mm", ".2f"),
        ("MAE", "mae_mm", ".2f"),
        ("CRPS", "crps_mm", ".2f"),
        ("|Bias|", "bias_mm", ".2f"),
        ("Correlation", "correlation", ".3f"),
        ("Spread", "spread_mm", ".2f"),
        ("Coverage 90%", "coverage_90", ".3f"),
    ]
    field_matrix = np.array(
        [
            [
                abs(field_scores[name][key]) if key == "bias_mm" else field_scores[name][key]
                for name in ("background", "analysis")
            ]
            for _, key, _ in raw_metrics
        ]
    )
    change_metrics = [
        ("RMSE reduction", "rmse_mm"),
        ("MAE reduction", "mae_mm"),
        ("CRPS reduction", "crps_mm"),
        ("|Bias| reduction", "bias_mm"),
        ("Correlation gain", "correlation"),
    ]
    improvement_matrix = np.array(
        [
            [directional_change(scope["background"], scope["analysis"], key)
             for scope in scopes.values()]
            for _, key in change_metrics
        ]
    )
    scale_metrics = [
        ("Coarse RMSE", coarse_scores, "rmse_mm", ".2f"),
        ("Coarse correlation", coarse_scores, "correlation", ".3f"),
        ("Subgrid RMSE", subgrid_scores, "rmse_mm", ".2f"),
        ("Subgrid correlation", subgrid_scores, "correlation", ".3f"),
        ("Subgrid variance ratio", subgrid_scores, "variance_ratio", ".2f"),
    ]
    scale_matrix = np.array(
        [[scores[name][key] for name in ("background", "analysis")]
         for _, scores, key, _ in scale_metrics]
    )

    daily_matrix = []
    for day in range(len(truth)):
        bg = ensemble_score(background[day], truth[day])["rmse_mm"]
        an = ensemble_score(analysis[day], truth[day])["rmse_mm"]
        daily_matrix.append([bg, an, 100 * (bg - an) / bg if bg else np.nan])
    daily_matrix = np.asarray(daily_matrix)

    figure, axes = plt.subplots(2, 2, figsize=(17, 13), constrained_layout=True)
    heatmap(
        axes[0, 0], field_matrix, [x[0] for x in raw_metrics],
        ["Background", "Analysis"], "A. Full 0.05° field — absolute metrics",
        formats=[x[2] for x in raw_metrics],
    )
    heatmap(
        axes[0, 1], improvement_matrix, [x[0] for x in change_metrics],
        list(scopes), "B. Directional improvement (positive = better, %)",
        cmap="RdBu", symmetric=True, formats=[".1f"] * len(change_metrics),
    )
    heatmap(
        axes[1, 0], scale_matrix, [x[0] for x in scale_metrics],
        ["Background", "Analysis"],
        "C. Scale separation — does 0.05° structure improve?",
        formats=[x[3] for x in scale_metrics],
    )
    heatmap(
        axes[1, 1], daily_matrix,
        [str(day) for day in data["days"].astype(str)],
        ["Background RMSE", "Analysis RMSE", "Reduction %"],
        "D. Daily 0.05° field RMSE matrix",
        formats=[".2f"] * len(daily_matrix),
    )
    figure.suptitle(
        "CHIRPS pseudo-satellite + station OSSE metric matrix\n"
        "Coarse scores measure information directly observed at 0.1°; subgrid "
        "scores isolate structure that only the generative prior can supply.",
        fontsize=14,
    )
    Path(args.out_matrix).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out_matrix, dpi=135)
    plt.close(figure)

    report = {
        "dump": str(args.dump),
        "checkpoint": str(data["checkpoint"]) if "checkpoint" in data.files else None,
        "checkpoint_epoch": int(data["checkpoint_epoch"])
        if "checkpoint_epoch" in data.files
        else None,
        "network": str(data["network"]),
        "n_assimilated": int(len(data["assim_idx"])),
        "n_withheld": int(len(eval_idx)),
        "pseudo_satellite": bool(data["pseudo_satellite_enabled"])
        if "pseudo_satellite_enabled" in data.files
        else False,
        "field_0p05": field_scores,
        "footprint_0p1": coarse_scores,
        "subgrid_0p05": subgrid_scores,
        "withheld_gauges": withheld_scores,
        "directional_improvement_percent": {
            scope_name: {
                key: directional_change(scores["background"], scores["analysis"], key)
                for _, key in change_metrics
            }
            for scope_name, scores in scopes.items()
        },
        "gate": {
            "field_crps_improved": field_scores["analysis"]["crps_mm"]
            < field_scores["background"]["crps_mm"],
            "subgrid_rmse_improved": subgrid_scores["analysis"]["rmse_mm"]
            < subgrid_scores["background"]["rmse_mm"],
            "subgrid_correlation_improved": subgrid_scores["analysis"]["correlation"]
            > subgrid_scores["background"]["correlation"],
            "coverage_near_nominal": abs(field_scores["analysis"]["coverage_90"] - 0.90)
            <= 0.10,
        },
        "interpretation": (
            "This CHIRPS-on-CHIRPS experiment is an optimistic upper bound. "
            "Do not approve real IMERG/BMD DA from footprint skill alone: require "
            "improvement in subgrid RMSE/correlation and acceptable ensemble coverage."
        ),
    }
    Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_report).write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.out_reconstruction}")
    print(f"wrote {args.out_matrix}")
    print(f"wrote {args.out_report}")


if __name__ == "__main__":
    main()
