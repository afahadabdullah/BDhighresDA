#!/usr/bin/env python
"""Plot held-out ERA5 input, CHIRPS target, and best-checkpoint predictions.

By default, the script selects three held-out days nearest the 50th, 90th, and
99th percentiles of Bangladesh-domain mean CHIRPS precipitation.  This gives a
compact dry/typical-to-extreme visual stress test without presenting the three
cases as an unbiased full-test score.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cartopy  # noqa: E402
import cartopy.crs as ccrs  # noqa: E402
import cartopy.feature as cfeature  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bdhires.da import SamplerConfig  # noqa: E402
from bdhires.da.sampler import sample  # noqa: E402
from bdhires.data import DatasetConfig, PrecipDataset  # noqa: E402
from bdhires.grids import WIDE, crop_offsets, get_grid  # noqa: E402
from bdhires.models import RectifiedFlow, UNet, select_weights  # noqa: E402
from bdhires.transforms import (  # noqa: E402
    load_climatology,
    CondTransform,
    PrecipTransform,
    ResidualSpec,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/da.yaml")
    parser.add_argument("--ckpt", default="runs/prior_h100/best.pt")
    parser.add_argument("--members", type=int, default=16)
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        default=[0.50, 0.90, 0.99],
        help="Quantiles of domain-mean CHIRPS used to select held-out cases.",
    )
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--out-figure",
        default="data/processed/test_prediction_maps.png",
        help="Map-comparison suite.",
    )
    parser.add_argument(
        "--out-metrics-figure",
        default="data/processed/test_prediction_metrics.png",
        help="Case-metric comparison suite.",
    )
    parser.add_argument(
        "--out-case-dir",
        default="data/processed/test_prediction_cases",
        help="Directory for one high-resolution spatial figure per case.",
    )
    parser.add_argument(
        "--out-report",
        default="data/processed/test_prediction_report.json",
    )
    parser.add_argument(
        "--stats",
        default=None,
        help="override data.stats from the config. Each checkpoint is bound to "
             "the statistics it was trained with and they are NOT "
             "interchangeable, so this avoids editing da.yaml between runs.",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=None,
        help="override background_sampler.cfg_scale, for sweeping without "
             "writing a config file per value.",
    )
    parser.add_argument(
        "--cartopy-data-dir",
        default="data/static/cartopy",
        help="Writable persistent cache for Natural Earth boundary files.",
    )
    args = parser.parse_args()
    if args.members < 2:
        parser.error("--members must be at least 2")
    if not args.quantiles or any(q < 0.0 or q > 1.0 for q in args.quantiles):
        parser.error("--quantiles must contain values between 0 and 1")
    return args


def metric_pair(prediction: np.ndarray, target: np.ndarray, valid: np.ndarray) -> dict:
    keep = valid & np.isfinite(prediction) & np.isfinite(target)
    predicted = prediction[keep].astype(np.float64)
    observed = target[keep].astype(np.float64)
    difference = predicted - observed
    if len(predicted) < 2:
        raise ValueError("not enough valid pixels for test metrics")
    if np.std(predicted) > 0 and np.std(observed) > 0:
        correlation = float(np.corrcoef(predicted, observed)[0, 1])
    else:
        correlation = None
    return {
        "n_pixels": int(len(predicted)),
        "bias_mm": float(np.mean(difference)),
        "mae_mm": float(np.mean(np.abs(difference))),
        "rmse_mm": float(np.sqrt(np.mean(difference**2))),
        "spatial_correlation": correlation,
        "prediction_max_mm": float(np.max(predicted)),
        "target_max_mm": float(np.max(observed)),
    }


def crps_ensemble(
    members: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
) -> float:
    """Mean empirical ensemble CRPS over valid grid cells."""
    ensemble = members[:, valid].astype(np.float64)
    observed = target[valid].astype(np.float64)
    term_observation = np.mean(np.abs(ensemble - observed[None]), axis=0)
    term_ensemble = 0.5 * np.mean(
        np.abs(ensemble[:, None] - ensemble[None, :]),
        axis=(0, 1),
    )
    return float(np.mean(term_observation - term_ensemble))


def load_best_model(
    checkpoint: dict,
    cond_channels: int,
    image_size: int,
    device: torch.device,
) -> tuple[UNet, dict, dict]:
    training_config = checkpoint["cfg"]
    model = UNet(
        in_channels=1,
        cond_channels=cond_channels,
        out_channels=1,
        image_size=image_size,
        **training_config["model"],
    )
    # EMA when the run used it, the online weights when it did not.
    model.load_state_dict(select_weights(checkpoint), strict=True)
    metadata = {
        key: checkpoint.get(key)
        for key in (
            "epoch", "step", "val_loss", "best_val_loss",
            "crps", "best_crps", "weights", "selected_by",
        )
        if checkpoint.get(key) is not None
    }
    return model.to(device).eval(), training_config, metadata


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def add_map_context(
    axis,
    projection,
    extent: list[float],
    longitude_ticks: np.ndarray,
    latitude_ticks: np.ndarray,
    *,
    label_left: bool,
    label_bottom: bool,
) -> None:
    """Add consistent boundaries and geographic labels to one map panel."""
    axis.set_extent(extent, crs=projection)
    axis.add_feature(
        cfeature.COASTLINE.with_scale("10m"),
        edgecolor="black",
        facecolor="none",
        linewidth=0.65,
        zorder=4,
    )
    axis.add_feature(
        cfeature.BORDERS.with_scale("10m"),
        edgecolor="black",
        facecolor="none",
        linewidth=0.60,
        zorder=4,
    )
    axis.add_feature(
        cfeature.STATES.with_scale("10m"),
        edgecolor="black",
        facecolor="none",
        linewidth=0.30,
        linestyle=":",
        alpha=0.70,
        zorder=4,
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
        alpha=0.28,
        linestyle=":",
    )
    gridlines.top_labels = False
    gridlines.right_labels = False
    gridlines.left_labels = label_left
    gridlines.bottom_labels = label_bottom
    gridlines.xformatter = LongitudeFormatter()
    gridlines.yformatter = LatitudeFormatter()
    gridlines.xlabel_style = {"size": 8}
    gridlines.ylabel_style = {"size": 8}


def add_metric_annotation(axis, text: str) -> None:
    axis.text(
        0.02,
        0.02,
        text,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        zorder=6,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.82,
            "pad": 2.5,
        },
    )


def map_column_titles(members: int) -> list[str]:
    """Column headers for the map suite.

    Lettered so panels can be referred to unambiguously in notes and captions.
    Columns A-C share a rainfall colour scale within each row and D-E share a
    symmetric error scale, which the subtitles state explicitly -- a reader
    should not have to infer which panels are comparable.
    """
    return [
        "A.  ERA5 input\nTotal precipitation (mm day$^{-1}$)",
        "B.  CHIRPS target\nObserved truth (mm day$^{-1}$)",
        f"C.  Model prediction\n{members}-member ensemble mean (mm day$^{{-1}}$)",
        "D.  ERA5 error\nInput − CHIRPS (mm day$^{-1}$)",
        "E.  Model error\nEnsemble mean − CHIRPS (mm day$^{-1}$)",
        "F.  Predictive uncertainty\nEnsemble standard deviation (mm day$^{-1}$)",
    ]


def add_row_label(axis, case: dict) -> None:
    """Identify the row in the left margin, clear of the latitude labels."""
    axis.annotate(
        f"{case['date']}\n"
        f"q{int(round(case['quantile'] * 100)):02d} rainfall case\n"
        f"{case['domain_mean_target_mm']:.1f} mm day$^{{-1}}$",
        xy=(0, 0.5),
        xycoords="axes fraction",
        xytext=(-72, 0),
        textcoords="offset points",
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
    )


def case_map_panels(
    case: dict,
    rain_cmap,
    error_cmap,
    spread_cmap,
) -> list[tuple[np.ndarray, object, float, float, str]]:
    """Return six consistently scaled spatial panels for one held-out case."""
    pooled = np.concatenate(
        [
            case["era5_input"][case["valid"]],
            case["target"][case["valid"]],
            case["prediction"][case["valid"]],
        ]
    )
    rain_max = max(5.0, float(np.percentile(pooled, 99.0)))
    error_limit = max(
        2.0,
        float(
            np.percentile(
                np.abs(
                    np.concatenate(
                        [
                            case["era5_error"][case["valid"]],
                            case["error"][case["valid"]],
                        ]
                    )
                ),
                99.0,
            )
        ),
    )
    spread_max = max(
        1.0,
        float(np.percentile(case["spread"][case["valid"]], 99.0)),
    )
    return [
        (
            case["era5_input"],
            rain_cmap,
            0.0,
            rain_max,
            f"RMSE {case['input_metrics']['rmse_mm']:.2f} mm day$^{{-1}}$\n"
            f"Spatial r {case['input_metrics']['spatial_correlation']:.2f}",
        ),
        (
            case["target"],
            rain_cmap,
            0.0,
            rain_max,
            f"Domain mean {case['domain_mean_target_mm']:.2f} mm day$^{{-1}}$\n"
            f"Maximum {case['prediction_metrics']['target_max_mm']:.1f} mm day$^{{-1}}$",
        ),
        (
            case["prediction"],
            rain_cmap,
            0.0,
            rain_max,
            f"RMSE {case['prediction_metrics']['rmse_mm']:.2f} mm day$^{{-1}}$\n"
            f"Spatial r {case['prediction_metrics']['spatial_correlation']:.2f}",
        ),
        (
            case["era5_error"],
            error_cmap,
            -error_limit,
            error_limit,
            f"Bias {case['input_metrics']['bias_mm']:+.2f} mm day$^{{-1}}$\n"
            f"MAE {case['input_metrics']['mae_mm']:.2f} mm day$^{{-1}}$",
        ),
        (
            case["error"],
            error_cmap,
            -error_limit,
            error_limit,
            f"Bias {case['prediction_metrics']['bias_mm']:+.2f} mm day$^{{-1}}$\n"
            f"MAE {case['prediction_metrics']['mae_mm']:.2f} mm day$^{{-1}}$",
        ),
        (
            case["spread"],
            spread_cmap,
            0.0,
            spread_max,
            f"Mean spread {case['prediction_metrics']['mean_spread_mm']:.2f} "
            "mm day$^{-1}$\n"
            f"90% coverage "
            f"{case['prediction_metrics']['interval_90_coverage'] * 100:.1f}%",
        ),
    ]


def save_individual_case_figures(
    cases: list[dict],
    output_dir: Path,
    members: int,
    grid,
    map_projection,
    rain_cmap,
    error_cmap,
    spread_cmap,
) -> list[Path]:
    """Write one large 2x3 spatial diagnostic for every selected case."""
    output_dir.mkdir(parents=True, exist_ok=True)
    extent = [grid.lon_min, grid.lon_max, grid.lat_min, grid.lat_max]
    longitude_ticks = np.arange(
        np.ceil(grid.lon_min),
        np.floor(grid.lon_max) + 1,
        2,
    )
    latitude_ticks = np.arange(
        np.ceil(grid.lat_min),
        np.floor(grid.lat_max) + 1,
        2,
    )
    titles = map_column_titles(members)
    outputs = []
    for case in cases:
        panels = case_map_panels(case, rain_cmap, error_cmap, spread_cmap)
        figure, axes = plt.subplots(
            2,
            3,
            figsize=(19, 12),
            constrained_layout=True,
            squeeze=False,
            subplot_kw={"projection": map_projection},
        )
        images = []
        for index, (values, cmap, vmin, vmax, annotation) in enumerate(panels):
            row, column = divmod(index, 3)
            axis = axes[row, column]
            image = axis.imshow(
                values,
                origin="lower",
                extent=extent,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
                transform=map_projection,
            )
            add_map_context(
                axis,
                map_projection,
                extent,
                longitude_ticks,
                latitude_ticks,
                label_left=True,
                label_bottom=True,
            )
            axis.set_title(titles[index], fontsize=12, pad=9)
            add_metric_annotation(axis, annotation)
            images.append(image)

        rainfall_colorbar = figure.colorbar(
            images[0],
            ax=axes[0, :].tolist(),
            orientation="horizontal",
            shrink=0.72,
            aspect=38,
            pad=0.035,
        )
        rainfall_colorbar.set_label(
            "Daily precipitation (mm day$^{-1}$)",
            fontsize=10,
        )
        error_colorbar = figure.colorbar(
            images[3],
            ax=axes[1, 0:2].tolist(),
            orientation="horizontal",
            shrink=0.72,
            aspect=28,
            pad=0.035,
        )
        error_colorbar.set_label(
            "Signed error: forecast − CHIRPS (mm day$^{-1}$)",
            fontsize=10,
        )
        spread_colorbar = figure.colorbar(
            images[5],
            ax=axes[1, 2],
            orientation="horizontal",
            shrink=0.86,
            aspect=18,
            pad=0.035,
        )
        spread_colorbar.set_label(
            "Ensemble standard deviation (mm day$^{-1}$)",
            fontsize=10,
        )
        quantile_label = int(round(case["quantile"] * 100))
        figure.suptitle(
            f"BDhighresDA held-out q{quantile_label:02d} rainfall case · "
            f"{case['date']}\n"
            f"ERA5-conditioned best-EMA background; {members}-member ensemble; "
            "errors are forecast − CHIRPS\n"
            "Rainfall fields share one scale; both errors share one symmetric scale",
            fontsize=15,
        )
        output = output_dir / f"{case['date']}_q{quantile_label:02d}.png"
        partial = output.with_suffix(output.suffix + ".part")
        figure.savefig(partial, format="png", dpi=200)
        plt.close(figure)
        partial.replace(output)
        outputs.append(output)
    return outputs


def save_metrics_figure(cases: list[dict], output: Path, members: int) -> None:
    """Create a second figure with directly comparable case metrics."""
    labels = [
        f"q{int(round(case['quantile'] * 100)):02d}\n{case['date']}"
        for case in cases
    ]
    positions = np.arange(len(cases), dtype=np.float64)
    width = 0.36
    era5_color = "#3b78b4"
    model_color = "#e07a3f"
    ensemble_color = "#2a9d78"
    target_color = "#555555"

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(17, 10),
        constrained_layout=True,
    )

    def grouped_bars(
        axis,
        first: list[float],
        second: list[float],
        first_label: str,
        second_label: str,
        ylabel: str,
        title: str,
    ) -> None:
        first_bars = axis.bar(
            positions - width / 2,
            first,
            width,
            color=era5_color,
            label=first_label,
        )
        second_bars = axis.bar(
            positions + width / 2,
            second,
            width,
            color=model_color,
            label=second_label,
        )
        axis.bar_label(first_bars, fmt="%.2f", padding=2, fontsize=8)
        axis.bar_label(second_bars, fmt="%.2f", padding=2, fontsize=8)
        axis.set_xticks(positions, labels)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False)
        axis.margins(y=0.16)

    grouped_bars(
        axes[0, 0],
        [case["input_metrics"]["rmse_mm"] for case in cases],
        [case["prediction_metrics"]["rmse_mm"] for case in cases],
        "ERA5 input",
        "Model ensemble mean",
        "RMSE (mm day$^{-1}$)",
        "A. Deterministic grid-cell RMSE",
    )
    grouped_bars(
        axes[0, 1],
        [case["input_metrics"]["mae_mm"] for case in cases],
        [case["prediction_metrics"]["crps_mm"] for case in cases],
        "ERA5 deterministic CRPS (= MAE)",
        f"Model ensemble CRPS ({members} members)",
        "CRPS (mm day$^{-1}$)",
        "B. Probabilistic error (lower is better)",
    )
    grouped_bars(
        axes[0, 2],
        [case["input_metrics"]["bias_mm"] for case in cases],
        [case["prediction_metrics"]["bias_mm"] for case in cases],
        "ERA5 input",
        "Model ensemble mean",
        "Mean error (mm day$^{-1}$)",
        "C. Domain-mean bias",
    )
    axes[0, 2].axhline(0.0, color="black", linewidth=0.8)

    grouped_bars(
        axes[1, 0],
        [
            case["input_metrics"]["spatial_correlation"]
            for case in cases
        ],
        [
            case["prediction_metrics"]["spatial_correlation"]
            for case in cases
        ],
        "ERA5 input",
        "Model ensemble mean",
        "Pearson correlation",
        "D. Spatial pattern correlation",
    )
    axes[1, 0].set_ylim(-0.1, 1.0)

    coverage = [
        case["prediction_metrics"]["interval_90_coverage"] for case in cases
    ]
    coverage_bars = axes[1, 1].bar(
        positions,
        coverage,
        width=0.55,
        color=ensemble_color,
        label="Observed coverage",
    )
    axes[1, 1].bar_label(
        coverage_bars,
        labels=[f"{value * 100:.1f}%" for value in coverage],
        padding=2,
        fontsize=8,
    )
    axes[1, 1].axhline(
        0.90,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="Nominal 90%",
    )
    axes[1, 1].set_xticks(positions, labels)
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].set_ylabel("Fraction of valid grid cells")
    axes[1, 1].set_title("E. Ensemble 90% interval coverage")
    axes[1, 1].grid(axis="y", alpha=0.25)
    axes[1, 1].legend(frameon=False)

    max_width = 0.24
    target_bars = axes[1, 2].bar(
        positions - max_width,
        [case["prediction_metrics"]["target_max_mm"] for case in cases],
        max_width,
        color=target_color,
        label="CHIRPS target",
    )
    input_bars = axes[1, 2].bar(
        positions,
        [case["input_metrics"]["prediction_max_mm"] for case in cases],
        max_width,
        color=era5_color,
        label="ERA5 input",
    )
    model_bars = axes[1, 2].bar(
        positions + max_width,
        [
            case["prediction_metrics"]["prediction_max_mm"]
            for case in cases
        ],
        max_width,
        color=model_color,
        label="Model ensemble mean",
    )
    for bars in (target_bars, input_bars, model_bars):
        axes[1, 2].bar_label(bars, fmt="%.1f", padding=2, fontsize=7)
    axes[1, 2].set_xticks(positions, labels)
    axes[1, 2].set_ylabel("Maximum (mm day$^{-1}$)")
    axes[1, 2].set_title("F. Maximum daily precipitation")
    axes[1, 2].grid(axis="y", alpha=0.25)
    axes[1, 2].legend(frameon=False)
    axes[1, 2].margins(y=0.16)

    figure.suptitle(
        "BDhighresDA held-out case metrics - model versus its own ERA5 input\n"
        f"{members}-member ensemble scored against CHIRPS on the target grid\n"
        "The model must beat the ERA5 bars: a conditional generator that scores "
        "worse than the field it is conditioned on has not learned to use it\n"
        "Target-selected cases are diagnostic, not an aggregate test score",
        fontsize=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    figure.savefig(partial, format="png", dpi=190)
    plt.close(figure)
    partial.replace(output)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    checkpoint_path = Path(args.ckpt)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"best checkpoint not found: {checkpoint_path}")
    cartopy_data_dir = Path(args.cartopy_data_dir).resolve()
    cartopy_data_dir.mkdir(parents=True, exist_ok=True)
    cartopy.config["data_dir"] = cartopy_data_dir

    config = yaml.safe_load(config_path.read_text())
    if args.stats:
        config["data"]["stats"] = args.stats
    if args.cfg_scale is not None:
        config.setdefault("background_sampler", config["sampler"])
        config["background_sampler"]["cfg_scale"] = args.cfg_scale
    print(f"statistics: {config['data']['stats']}")
    stats = json.loads(Path(config["data"]["stats"]).read_text())
    transform = PrecipTransform.from_dict(stats["precip_transform"])
    grid = get_grid(config["data"]["grid"])
    crop_origin = crop_offsets(WIDE, grid)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    training_config = checkpoint["cfg"]
    dataset = PrecipDataset(
        DatasetConfig(
            root=config["data"]["zarr"],
            crop=grid.nlon,
            random_crop=False,
            crop_origin=crop_origin,
            seasonal_encoding=training_config["data"].get(
                "seasonal_encoding", True
            ),
        ),
        transform,
        cond_mean=np.asarray(stats["cond_mean"], dtype=np.float32),
        cond_std=np.asarray(stats["cond_std"], dtype=np.float32),
        cond_transform=CondTransform.from_stats(stats),
        residual=ResidualSpec.from_stats(stats),
        climatology=load_climatology(config["data"]["stats"], stats),
    )
    valid = dataset.fixed_valid > 0
    if valid.shape != grid.shape:
        raise ValueError(
            f"cropped validity mask {valid.shape} does not match {grid.shape}"
        )

    test_years = training_config["data"]["years"]["test"]
    start = np.datetime64(args.start or f"{test_years[0]}-01-01")
    end = np.datetime64(args.end or f"{test_years[1]}-12-31")
    eligible = np.where((dataset.time >= start) & (dataset.time <= end))[0]
    if not len(eligible):
        raise ValueError(f"no test dates between {start} and {end}")

    spatial_slices = dataset.fixed_spatial_slices()
    domain_means = np.empty(len(eligible), dtype=np.float64)
    for position, index in enumerate(eligible):
        target = np.asarray(
            dataset.z["target"][int(index)][spatial_slices],
            dtype=np.float32,
        )
        domain_means[position] = np.nanmean(np.where(valid, target, np.nan))

    selected: list[tuple[float, int, float]] = []
    used: set[int] = set()
    for quantile in args.quantiles:
        threshold = float(np.quantile(domain_means, quantile))
        order = np.argsort(np.abs(domain_means - threshold))
        position = next(int(p) for p in order if int(eligible[p]) not in used)
        index = int(eligible[position])
        used.add(index)
        selected.append((float(quantile), index, float(domain_means[position])))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, training_config, checkpoint_metadata = load_best_model(
        checkpoint,
        dataset.total_cond_channels,
        grid.nlon,
        device,
    )
    del checkpoint
    # The background is UNGUIDED: no observations exist to pull inflated members
    # back, so it uses its own sampler block with prior_temperature = 1 and no
    # Langevin correctors.  Fall back to `sampler` only for old configs.
    base_sampler = SamplerConfig(
        **config.get("background_sampler", config["sampler"])
    )
    base_sampler = replace(base_sampler, mask_fill=dataset.mask_fill)
    flow = RectifiedFlow()
    mask = torch.from_numpy(valid.astype(np.float32)[None, None]).to(device)
    era5_tp_index = int(config["data"].get("era5_tp_cond_index", 0))
    condition_transform = CondTransform.from_stats(stats)
    residual_spec = ResidualSpec.from_stats(stats)
    condition_mean = np.asarray(stats["cond_mean"], dtype=np.float32)
    condition_std = np.asarray(stats["cond_std"], dtype=np.float32)

    cases = []
    print(
        f"testing {checkpoint_path} on {len(selected)} held-out cases with "
        f"{args.members} members using {device}",
        flush=True,
    )
    for case_number, (quantile, index, domain_mean) in enumerate(selected):
        item = dataset[index]
        target = np.asarray(
            dataset.z["target"][index][spatial_slices],
            dtype=np.float32,
        )
        target = np.where(valid, target, np.nan)
        case_valid = valid & np.isfinite(target)
        standardized_era5 = item["cond"][era5_tp_index].numpy()
        era5_input = (
            standardized_era5 * condition_std[era5_tp_index]
            + condition_mean[era5_tp_index]
        )
        # Undo the variance-stabilising transform as well, or panel A plots
        # log-space numbers on an axis labelled mm/day.
        era5_input = condition_transform.inverse_channel(era5_input, era5_tp_index)
        era5_input = np.where(valid, era5_input, np.nan)

        sampler = replace(base_sampler, seed=args.seed + case_number)
        base = item["base"][None].to(device)
        with torch.inference_mode():
            generated = sample(
                model,
                item["cond"][None].to(device),
                (args.members, 1, grid.nlat, grid.nlon),
                device,
                cfg=sampler,
                flow=flow,
                mask=mask,
                to_precip=lambda x, b=base: residual_spec.decode(x, b),
            )
        # Decode the network variable into transformed-precipitation space
        # (identity unless the checkpoint was trained on residuals).
        members = transform.inverse(
            residual_spec.decode(generated, base)[:, 0].cpu().numpy()
        )
        members = np.where(valid[None], members, np.nan)
        if not np.isfinite(members[:, valid]).all() or np.any(members[:, valid] < 0):
            raise ValueError("generated precipitation is non-finite or negative")
        prediction = np.mean(members, axis=0)
        spread = np.std(members, axis=0, ddof=1)
        prediction_metrics = metric_pair(prediction, target, case_valid)
        input_metrics = metric_pair(era5_input, target, case_valid)
        prediction_metrics.update(
            mean_spread_mm=float(np.mean(spread[case_valid])),
            crps_mm=crps_ensemble(members, target, case_valid),
            interval_90_coverage=float(
                np.mean(
                    (
                        target[case_valid]
                        >= np.quantile(members[:, case_valid], 0.05, axis=0)
                    )
                    & (
                        target[case_valid]
                        <= np.quantile(members[:, case_valid], 0.95, axis=0)
                    )
                )
            ),
        )
        date = str(dataset.time[index].astype("datetime64[D]"))
        print(
            f"{date} q={quantile:.2f}: "
            f"ERA5 RMSE={input_metrics['rmse_mm']:.3f}, "
            f"model RMSE={prediction_metrics['rmse_mm']:.3f}, "
            f"r={prediction_metrics['spatial_correlation']:.3f}",
            flush=True,
        )
        cases.append(
            {
                "quantile": quantile,
                "index": index,
                "date": date,
                "domain_mean_target_mm": domain_mean,
                "era5_input": era5_input,
                "target": target,
                "prediction": prediction,
                "era5_error": era5_input - target,
                "error": prediction - target,
                "spread": spread,
                "valid": case_valid,
                "input_metrics": input_metrics,
                "prediction_metrics": prediction_metrics,
            }
        )

    map_projection = ccrs.PlateCarree()
    figure, axes = plt.subplots(
        len(cases),
        6,
        figsize=(27, 5.2 * len(cases)),
        constrained_layout=True,
        squeeze=False,
        subplot_kw={"projection": map_projection},
    )
    extent = [grid.lon_min, grid.lon_max, grid.lat_min, grid.lat_max]
    rain_cmap = plt.get_cmap("viridis").copy()
    rain_cmap.set_bad("white")
    error_cmap = plt.get_cmap("RdBu_r").copy()
    error_cmap.set_bad("white")
    spread_cmap = plt.get_cmap("magma").copy()
    spread_cmap.set_bad("white")
    column_titles = map_column_titles(args.members)
    longitude_ticks = np.arange(
        np.ceil(grid.lon_min),
        np.floor(grid.lon_max) + 1,
        2,
    )
    latitude_ticks = np.arange(
        np.ceil(grid.lat_min),
        np.floor(grid.lat_max) + 1,
        2,
    )

    for row, case in enumerate(cases):
        panels = case_map_panels(case, rain_cmap, error_cmap, spread_cmap)
        row_images = []
        for column, (values, cmap, vmin, vmax, annotation) in enumerate(panels):
            axis = axes[row, column]
            image = axis.imshow(
                values,
                origin="lower",
                extent=extent,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
                transform=map_projection,
            )
            add_map_context(
                axis,
                map_projection,
                extent,
                longitude_ticks,
                latitude_ticks,
                label_left=column == 0,
                label_bottom=row == len(cases) - 1,
            )
            if row == 0:
                axis.set_title(column_titles[column], fontsize=11, pad=10)
            if column == 0:
                add_row_label(axis, case)
            add_metric_annotation(axis, annotation)
            row_images.append(image)

        rain_colorbar = figure.colorbar(
            row_images[0],
            ax=axes[row, 0:3].tolist(),
            orientation="horizontal",
            shrink=0.76,
            aspect=35,
            pad=0.035,
        )
        rain_colorbar.set_label("Daily precipitation (mm day$^{-1}$)", fontsize=9)
        error_colorbar = figure.colorbar(
            row_images[3],
            ax=axes[row, 3:5].tolist(),
            orientation="horizontal",
            shrink=0.72,
            aspect=28,
            pad=0.035,
        )
        error_colorbar.set_label(
            "Signed error: forecast − CHIRPS (mm day$^{-1}$)",
            fontsize=9,
        )
        spread_colorbar = figure.colorbar(
            row_images[5],
            ax=axes[row, 5],
            orientation="horizontal",
            shrink=0.88,
            aspect=18,
            pad=0.035,
        )
        spread_colorbar.set_label(
            "Ensemble standard deviation (mm day$^{-1}$)",
            fontsize=9,
        )

    epoch_label = checkpoint_metadata.get("epoch")
    weights_label = (
        "online weights (no EMA)"
        if checkpoint_metadata.get("weights") == "model"
        else "EMA weights"
    )
    figure.suptitle(
        "BDhighresDA held-out ERA5-conditioned background comparison\n"
        f"{checkpoint_path}"
        + (f"  (epoch {epoch_label + 1})" if epoch_label is not None else "")
        + f", {weights_label}   |   statistics {config['data']['stats']}   |   "
        f"test period {start} to {end}\n"
        f"{args.members}-member ensemble   |   sampler: {base_sampler.n_steps} steps, "
        f"prior temperature {base_sampler.prior_temperature:g}, "
        f"CFG w={base_sampler.cfg_scale:g}, "
        f"schedule power {base_sampler.schedule_power:g}\n"
        "Columns A-C share a rainfall colour scale within each row; D-E share a "
        "symmetric error scale; rows are independent days\n"
        "Cases are selected by domain-mean CHIRPS quantile and are diagnostic, "
        "not an aggregate test score   |   Natural Earth 10 m boundaries",
        fontsize=14,
    )
    output_figure = Path(args.out_figure)
    output_figure.parent.mkdir(parents=True, exist_ok=True)
    partial_figure = output_figure.with_suffix(output_figure.suffix + ".part")
    figure.savefig(partial_figure, format="png", dpi=180)
    plt.close(figure)
    partial_figure.replace(output_figure)
    output_metrics_figure = Path(args.out_metrics_figure)
    save_metrics_figure(cases, output_metrics_figure, args.members)
    case_figure_paths = save_individual_case_figures(
        cases,
        Path(args.out_case_dir),
        args.members,
        grid,
        map_projection,
        rain_cmap,
        error_cmap,
        spread_cmap,
    )

    report_cases = []
    for case in cases:
        report_cases.append(
            {
                key: value
                for key, value in case.items()
                if key
                not in {
                    "era5_input",
                    "target",
                    "prediction",
                    "era5_error",
                    "error",
                    "spread",
                    "valid",
                }
            }
        )
    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_metadata": checkpoint_metadata,
        "git_commit": git_commit(),
        "config": str(config_path),
        "period": [str(start), str(end)],
        "selection": "nearest domain-mean CHIRPS precipitation quantiles",
        "members": args.members,
        "sampler": asdict(base_sampler),
        "crop_origin_row_col": list(crop_origin),
        "grid": grid.name,
        "figure": str(output_figure),
        "figures": {
            "map_comparison": str(output_figure),
            "metric_summary": str(output_metrics_figure),
            "individual_cases": [str(path) for path in case_figure_paths],
        },
        "map_projection": "Cartopy PlateCarree",
        "boundaries": "Natural Earth 10m coastline, national borders, and admin-1",
        "cartopy_data_dir": str(cartopy_data_dir),
        "cases": report_cases,
        "note": (
            "Metrics describe target-selected example days and are diagnostic, "
            "not aggregate test-set skill estimates."
        ),
    }
    output_report = Path(args.out_report)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    partial_report = output_report.with_suffix(output_report.suffix + ".part")
    partial_report.write_text(json.dumps(report, indent=2) + "\n")
    partial_report.replace(output_report)

    for figure_path in (output_figure, output_metrics_figure, *case_figure_paths):
        if figure_path.stat().st_size < 100_000:
            raise ValueError(
                f"diagnostic figure is unexpectedly small: {figure_path}"
            )
    print(f"TEST PREDICTION DIAGNOSTICS PASSED; wrote {output_figure}")
    print(f"wrote {output_metrics_figure}")
    for case_figure_path in case_figure_paths:
        print(f"wrote {case_figure_path}")
    print(f"wrote {output_report}")


if __name__ == "__main__":
    main()
