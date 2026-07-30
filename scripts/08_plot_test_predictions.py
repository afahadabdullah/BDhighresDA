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
from bdhires.models import RectifiedFlow, UNet  # noqa: E402
from bdhires.transforms import PrecipTransform  # noqa: E402


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
        default="data/processed/test_prediction_panels.png",
    )
    parser.add_argument(
        "--out-report",
        default="data/processed/test_prediction_panels.json",
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
    if "ema" not in checkpoint:
        raise ValueError("the checkpoint does not contain EMA weights")
    training_config = checkpoint["cfg"]
    model = UNet(
        in_channels=1,
        cond_channels=cond_channels,
        out_channels=1,
        image_size=image_size,
        **training_config["model"],
    )
    model.load_state_dict(checkpoint["ema"], strict=True)
    metadata = {
        key: checkpoint.get(key)
        for key in ("epoch", "step", "val_loss", "best_val_loss")
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
    base_sampler = SamplerConfig(**config["sampler"])
    flow = RectifiedFlow()
    mask = torch.from_numpy(valid.astype(np.float32)[None, None]).to(device)
    era5_tp_index = int(config["data"].get("era5_tp_cond_index", 0))
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
        era5_input = np.where(valid, era5_input, np.nan)

        sampler = replace(base_sampler, seed=args.seed + case_number)
        with torch.inference_mode():
            generated = sample(
                model,
                item["cond"][None].to(device),
                (args.members, 1, grid.nlat, grid.nlon),
                device,
                cfg=sampler,
                flow=flow,
                mask=mask,
            )
        members = transform.inverse(generated[:, 0].cpu().numpy())
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
        5,
        figsize=(23, 4.6 * len(cases)),
        constrained_layout=True,
        squeeze=False,
        subplot_kw={"projection": map_projection},
    )
    extent = [grid.lon_min, grid.lon_max, grid.lat_min, grid.lat_max]
    rain_cmap = plt.get_cmap("viridis").copy()
    rain_cmap.set_bad("white")
    error_cmap = plt.get_cmap("RdBu_r").copy()
    error_cmap.set_bad("white")

    for row, case in enumerate(cases):
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
            float(np.percentile(np.abs(case["error"][case["valid"]]), 99.0)),
        )
        spread_max = max(
            1.0,
            float(np.percentile(case["spread"][case["valid"]], 99.0)),
        )
        panels = [
            (
                case["era5_input"],
                "ERA5 precipitation input\n"
                f"RMSE {case['input_metrics']['rmse_mm']:.2f} mm",
                rain_cmap,
                0.0,
                rain_max,
                "mm day$^{-1}$",
            ),
            (
                case["target"],
                f"CHIRPS target · {case['date']} · "
                f"q{int(round(case['quantile'] * 100)):02d}\n"
                f"domain mean {case['domain_mean_target_mm']:.2f} mm",
                rain_cmap,
                0.0,
                rain_max,
                "mm day$^{-1}$",
            ),
            (
                case["prediction"],
                f"Prediction mean · {args.members} members\n"
                f"RMSE {case['prediction_metrics']['rmse_mm']:.2f} mm; "
                f"r {case['prediction_metrics']['spatial_correlation']:.2f}",
                rain_cmap,
                0.0,
                rain_max,
                "mm day$^{-1}$",
            ),
            (
                case["error"],
                "Prediction − target\n"
                f"bias {case['prediction_metrics']['bias_mm']:+.2f} mm",
                error_cmap,
                -error_limit,
                error_limit,
                "mm day$^{-1}$",
            ),
            (
                case["spread"],
                "Ensemble standard deviation\n"
                f"mean {case['prediction_metrics']['mean_spread_mm']:.2f} mm",
                rain_cmap,
                0.0,
                spread_max,
                "mm day$^{-1}$",
            ),
        ]
        for column, (values, title, cmap, vmin, vmax, colorbar_label) in enumerate(
            panels
        ):
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
            axis.set_extent(extent, crs=map_projection)
            axis.add_feature(
                cfeature.COASTLINE.with_scale("10m"),
                edgecolor="black",
                facecolor="none",
                linewidth=0.7,
                zorder=4,
            )
            axis.add_feature(
                cfeature.BORDERS.with_scale("10m"),
                edgecolor="black",
                facecolor="none",
                linewidth=0.65,
                zorder=4,
            )
            axis.add_feature(
                cfeature.STATES.with_scale("10m"),
                edgecolor="black",
                facecolor="none",
                linewidth=0.35,
                linestyle=":",
                alpha=0.75,
                zorder=4,
            )
            gridlines = axis.gridlines(
                crs=map_projection,
                draw_labels=True,
                x_inline=False,
                y_inline=False,
                xlocs=mticker.FixedLocator(
                    np.arange(
                        np.ceil(grid.lon_min),
                        np.floor(grid.lon_max) + 1,
                        2,
                    )
                ),
                ylocs=mticker.FixedLocator(
                    np.arange(
                        np.ceil(grid.lat_min),
                        np.floor(grid.lat_max) + 1,
                        2,
                    )
                ),
                linewidth=0.35,
                color="black",
                alpha=0.35,
                linestyle=":",
            )
            gridlines.top_labels = False
            gridlines.right_labels = False
            gridlines.xformatter = LongitudeFormatter()
            gridlines.yformatter = LatitudeFormatter()
            gridlines.xlabel_style = {"size": 7}
            gridlines.ylabel_style = {"size": 7}
            axis.set_title(title, fontsize=10)
            figure.colorbar(
                image,
                ax=axis,
                fraction=0.046,
                pad=0.03,
                label=colorbar_label,
            )

    figure.suptitle(
        "BDhighresDA held-out best-checkpoint background predictions\n"
        f"{start} to {end}; EMA checkpoint; {args.members}-member ensemble; "
        f"sampler steps={base_sampler.n_steps}, temperature="
        f"{base_sampler.prior_temperature:g}",
        fontsize=15,
    )
    figure.text(
        0.5,
        0.001,
        "Coastlines, national borders, and first-order boundaries: "
        "Natural Earth via Cartopy (10 m)",
        ha="center",
        fontsize=8,
    )
    output_figure = Path(args.out_figure)
    output_figure.parent.mkdir(parents=True, exist_ok=True)
    partial_figure = output_figure.with_suffix(output_figure.suffix + ".part")
    figure.savefig(partial_figure, format="png", dpi=180)
    plt.close(figure)
    partial_figure.replace(output_figure)

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

    if output_figure.stat().st_size < 100_000:
        raise ValueError(f"diagnostic figure is unexpectedly small: {output_figure}")
    print(f"TEST PREDICTION DIAGNOSTICS PASSED; wrote {output_figure}")
    print(f"wrote {output_report}")


if __name__ == "__main__":
    main()
