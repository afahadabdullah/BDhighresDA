#!/usr/bin/env python3
"""Create the mandatory pre-training normalization diagnostic figure.

The single large PNG contains raw and normalized maps plus sampled
distributions for CHIRPS and all six ERA5 predictors, followed by the seven
static fields. A JSON sidecar records numerical checks used by the GH200
training preflight.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import zarr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bdhires.transforms import CondTransform, PrecipTransform  # noqa: E402


FIELD_LABELS = {
    "target": "CHIRPS precipitation",
    "era5_tp": "ERA5 precipitation",
    "era5_tcwv": "ERA5 column water vapour",
    "era5_cape": "ERA5 CAPE",
    "era5_u10": "ERA5 10 m zonal wind",
    "era5_v10": "ERA5 10 m meridional wind",
    "era5_msl": "ERA5 mean sea-level pressure",
}
FIELD_UNITS = {
    "target": "mm day⁻¹",
    "era5_tp": "mm day⁻¹",
    "era5_tcwv": "kg m⁻²",
    "era5_cape": "J kg⁻¹",
    "era5_u10": "m s⁻¹",
    "era5_v10": "m s⁻¹",
    "era5_msl": "hPa",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(repository: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def plot_units(name: str, values: np.ndarray) -> np.ndarray:
    if name == "era5_msl":
        return values / 100.0
    return values


def robust_limits(values: np.ndarray, *, symmetric: bool = False) -> tuple[float, float]:
    finite = np.asarray(values)[np.isfinite(values)]
    if not len(finite):
        raise ValueError("cannot determine plot limits from an empty field")
    if symmetric:
        limit = float(np.percentile(np.abs(finite), 99))
        return -max(limit, 1e-6), max(limit, 1e-6)
    low, high = np.percentile(finite, [1, 99])
    if low == high:
        high = low + 1e-6
    return float(low), float(high)


def open_group(path: Path):
    try:
        return zarr.open_group(str(path), mode="r", zarr_format=2)
    except TypeError:
        return zarr.open_group(str(path), mode="r", zarr_version=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr", default="data/processed/bd_wide.zarr")
    parser.add_argument("--stats", default="data/processed/stats.json")
    parser.add_argument(
        "--out",
        default="data/processed/normalization_diagnostics.png",
    )
    parser.add_argument(
        "--report",
        default="data/processed/normalization_diagnostics.json",
    )
    parser.add_argument("--sample-days", type=int, default=256)
    parser.add_argument("--pixels-per-day", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.sample_days < 16:
        parser.error("--sample-days must be at least 16")
    if args.pixels_per_day < 128:
        parser.error("--pixels-per-day must be at least 128")

    repository = Path(__file__).resolve().parents[1]
    zarr_path = (repository / args.zarr).resolve()
    stats_path = (repository / args.stats).resolve()
    output = (repository / args.out).resolve()
    report_path = (repository / args.report).resolve()
    stats = json.loads(stats_path.read_text())
    root = open_group(zarr_path)
    if not root.attrs.get("complete", False):
        raise ValueError(f"{zarr_path} is not marked complete")

    time = np.asarray(root["time"][:]).astype("datetime64[ns]")
    years = time.astype("datetime64[Y]").astype(int) + 1970
    train_start, train_end = map(int, stats["train_years"])
    training_indices = np.flatnonzero(
        (years >= train_start) & (years <= train_end)
    )
    if len(training_indices) < args.sample_days:
        raise ValueError(
            f"only {len(training_indices)} training days are available, fewer "
            f"than requested sample {args.sample_days}"
        )

    condition_channels = list(root.attrs.get("cond_channels", []))
    static_channels = list(root.attrs.get("static_channels", []))
    if condition_channels != list(stats["cond_channels"]):
        raise ValueError("Zarr and statistics condition-channel metadata differ")
    if static_channels != list(stats["static_channels"]):
        raise ValueError("Zarr and statistics static-channel metadata differ")
    if condition_channels != [
        "era5_tp",
        "era5_tcwv",
        "era5_cape",
        "era5_u10",
        "era5_v10",
        "era5_msl",
    ]:
        raise ValueError(f"unexpected condition channels: {condition_channels}")

    valid = np.asarray(root["valid"][:]) > 0.5
    static = np.asarray(root["static"][:], dtype=np.float32)
    latitude = np.asarray(root["lat"][:], dtype=np.float64)
    longitude = np.asarray(root["lon"][:], dtype=np.float64)
    if not np.isfinite(static).all() or not valid.any():
        raise ValueError("static fields or land-validity mask are invalid")

    rng = np.random.default_rng(args.seed)
    sampled_days = np.sort(
        rng.choice(training_indices, size=args.sample_days, replace=False)
    )
    land_flat = np.flatnonzero(valid.ravel())
    all_flat = np.arange(valid.size)
    target_pixels = rng.choice(
        land_flat,
        size=min(args.pixels_per_day, len(land_flat)),
        replace=False,
    )
    condition_pixels = rng.choice(
        all_flat,
        size=min(args.pixels_per_day, len(all_flat)),
        replace=False,
    )

    raw_samples: dict[str, list[np.ndarray]] = {
        "target": [],
        **{name: [] for name in condition_channels},
    }
    land_means = np.empty(len(sampled_days), dtype=np.float64)
    latitude_weights = np.cos(np.deg2rad(latitude))[:, None] * valid
    weight_sum = latitude_weights.sum()
    for sample_index, time_index in enumerate(sampled_days):
        target = np.asarray(root["target"][int(time_index)], dtype=np.float32)
        condition = np.asarray(root["cond"][int(time_index)], dtype=np.float32)
        if not np.isfinite(target[valid]).all() or not np.isfinite(condition).all():
            raise ValueError(f"non-finite training field at index {time_index}")
        raw_samples["target"].append(target.ravel()[target_pixels])
        for channel_index, name in enumerate(condition_channels):
            raw_samples[name].append(
                condition[channel_index].ravel()[condition_pixels]
            )
        land_means[sample_index] = np.nansum(
            target * latitude_weights
        ) / weight_sum
    samples = {
        name: np.concatenate(chunks).astype(np.float64)
        for name, chunks in raw_samples.items()
    }

    wet_rank = int(round(0.9 * (len(sampled_days) - 1)))
    selected_position = np.argsort(land_means)[wet_rank]
    selected_index = int(sampled_days[selected_position])
    selected_date = time[selected_index].astype("datetime64[D]")
    target_map = np.asarray(root["target"][selected_index], dtype=np.float32)
    condition_maps = np.asarray(root["cond"][selected_index], dtype=np.float32)

    transform = PrecipTransform.from_dict(stats["precip_transform"])
    normalized_samples = {
        "target": transform.forward(samples["target"]),
    }
    normalized_maps = {
        "target": np.where(valid, transform.forward(target_map), np.nan),
    }
    condition_mean = np.asarray(stats["cond_mean"], dtype=np.float64)
    condition_std = np.asarray(stats["cond_std"], dtype=np.float64)
    if (
        len(condition_mean) != len(condition_channels)
        or len(condition_std) != len(condition_channels)
        or not np.isfinite(condition_mean).all()
        or not np.isfinite(condition_std).all()
        or np.any(condition_std <= 0)
    ):
        raise ValueError("condition normalization statistics are invalid")
    # cond_mean/cond_std are computed AFTER the variance-stabilising transform
    # (06_compute_stats.py), so the transform has to be applied here before
    # standardising.  Standardising raw values with transformed-space constants
    # gives nonsense -- sqrt-space CAPE statistics against raw J/kg produced a
    # sampled mean of 34.8 sigma, which is what this diagnostic caught.
    condition_transform = CondTransform.from_stats(stats)
    for channel_index, name in enumerate(condition_channels):
        normalized_samples[name] = (
            condition_transform.forward_channel(samples[name], channel_index)
            - condition_mean[channel_index]
        ) / condition_std[channel_index]
        normalized_maps[name] = (
            condition_transform.forward_channel(
                condition_maps[channel_index], channel_index
            )
            - condition_mean[channel_index]
        ) / condition_std[channel_index]

    metrics = {}
    passed = True
    for name in ["target", *condition_channels]:
        values = normalized_samples[name]
        mean = float(np.mean(values))
        standard_deviation = float(np.std(values))
        finite = bool(np.isfinite(values).all())
        approximately_standard = (
            finite
            and abs(mean) <= 0.5
            and 0.5 <= standard_deviation <= 1.5
        )
        metrics[name] = {
            "sample_mean": mean,
            "sample_std": standard_deviation,
            "finite": finite,
            "approximately_standard": approximately_standard,
        }
        passed = passed and approximately_standard

    fields = ["target", *condition_channels]
    figure = plt.figure(figsize=(22, 34), constrained_layout=True)
    grid = figure.add_gridspec(
        9,
        4,
        height_ratios=[1, 1, 1, 1, 1, 1, 1, 0.9, 0.9],
    )
    extent = [
        float(longitude.min()),
        float(longitude.max()),
        float(latitude.min()),
        float(latitude.max()),
    ]
    figure.suptitle(
        "BDhighresDA pre-training normalization diagnostics\n"
        f"maps: {selected_date} (90th-percentile wet sampled day); "
        f"distributions: {args.sample_days} training days, "
        f"{args.pixels_per_day} pixels/day",
        fontsize=16,
    )

    for row, name in enumerate(fields):
        raw_map_axis = figure.add_subplot(grid[row, 0])
        normalized_map_axis = figure.add_subplot(grid[row, 1])
        raw_hist_axis = figure.add_subplot(grid[row, 2])
        normalized_hist_axis = figure.add_subplot(grid[row, 3])

        raw_map = target_map if name == "target" else condition_maps[row - 1]
        if name == "target":
            raw_map = np.where(valid, raw_map, np.nan)
        raw_map_for_plot = plot_units(name, raw_map)
        raw_sample_for_plot = plot_units(name, samples[name])
        symmetric = name in {"era5_u10", "era5_v10"}
        raw_low, raw_high = robust_limits(
            raw_map_for_plot,
            symmetric=symmetric,
        )
        raw_cmap = "RdBu_r" if symmetric else "viridis"
        raw_image = raw_map_axis.imshow(
            raw_map_for_plot,
            origin="lower",
            extent=extent,
            cmap=raw_cmap,
            vmin=raw_low,
            vmax=raw_high,
            aspect="auto",
        )
        figure.colorbar(raw_image, ax=raw_map_axis, fraction=0.046, pad=0.03)
        raw_map_axis.set_title(
            f"{FIELD_LABELS[name]} — raw [{FIELD_UNITS[name]}]",
            fontsize=10,
        )
        raw_map_axis.set_ylabel("latitude")
        raw_map_axis.set_xlabel("longitude")

        normalized_image = normalized_map_axis.imshow(
            normalized_maps[name],
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-3,
            vmax=3,
            aspect="auto",
        )
        figure.colorbar(
            normalized_image,
            ax=normalized_map_axis,
            fraction=0.046,
            pad=0.03,
        )
        normalized_map_axis.set_title(
            f"{FIELD_LABELS[name]} — normalized [σ]",
            fontsize=10,
        )
        normalized_map_axis.set_xlabel("longitude")

        raw_hist_axis.hist(
            raw_sample_for_plot,
            bins=80,
            density=True,
            color="#3b82f6",
            alpha=0.8,
        )
        raw_hist_axis.set_xlim(*robust_limits(raw_sample_for_plot))
        raw_hist_axis.set_title("Raw sampled distribution", fontsize=10)
        raw_hist_axis.set_xlabel(FIELD_UNITS[name])
        raw_hist_axis.set_ylabel("density")
        raw_hist_axis.grid(alpha=0.2)

        normalized_hist_axis.hist(
            normalized_samples[name],
            bins=80,
            density=True,
            color="#f97316",
            alpha=0.8,
        )
        norm_low, norm_high = robust_limits(normalized_samples[name])
        clipped_low = max(norm_low, -6)
        clipped_high = min(norm_high, 6)
        if clipped_low >= clipped_high:
            clipped_low, clipped_high = -6, 6
        normalized_hist_axis.set_xlim(clipped_low, clipped_high)
        normalized_hist_axis.axvline(0, color="black", linewidth=1)
        normalized_hist_axis.set_title(
            "Normalized sampled distribution\n"
            f"mean={metrics[name]['sample_mean']:.3f}, "
            f"std={metrics[name]['sample_std']:.3f}",
            fontsize=10,
        )
        normalized_hist_axis.set_xlabel("standard deviations")
        normalized_hist_axis.set_ylabel("density")
        normalized_hist_axis.grid(alpha=0.2)

    static_axes = [
        figure.add_subplot(grid[7, column]) for column in range(4)
    ] + [
        figure.add_subplot(grid[8, column]) for column in range(3)
    ]
    summary_axis = figure.add_subplot(grid[8, 3])
    for index, (axis, name) in enumerate(zip(static_axes, static_channels)):
        values = static[index]
        symmetric = name in {
            "slope",
            "sin_lon",
            "cos_lon",
            "sin_lat",
            "cos_lat",
        }
        low, high = robust_limits(values, symmetric=symmetric)
        image = axis.imshow(
            values,
            origin="lower",
            extent=extent,
            cmap="RdBu_r" if symmetric else "terrain",
            vmin=low,
            vmax=high,
            aspect="auto",
        )
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
        axis.set_title(f"Static: {name} (not standardized)", fontsize=10)
        axis.set_xlabel("longitude")
        axis.set_ylabel("latitude")
    summary_axis.axis("off")
    summary_axis.text(
        0.02,
        0.98,
        "Numerical QA\n\n"
        f"Training years: {train_start}–{train_end}\n"
        f"Selected date: {selected_date}\n"
        f"Land fraction: {valid.mean():.2%}\n"
        f"Condition channels: {len(condition_channels)}\n"
        f"Static channels: {len(static_channels)}\n\n"
        "Pass criteria for sampled normalized fields:\n"
        "• all finite\n"
        "• |mean| ≤ 0.5\n"
        "• 0.5 ≤ standard deviation ≤ 1.5\n\n"
        f"Result: {'PASSED' if passed else 'FAILED'}",
        va="top",
        ha="left",
        fontsize=11,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure_partial = output.with_suffix(output.suffix + ".part")
    figure.savefig(
        figure_partial,
        format="png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)
    if not figure_partial.is_file() or figure_partial.stat().st_size < 100_000:
        raise ValueError("normalization diagnostic figure was not written correctly")
    figure_partial.replace(output)

    report = {
        "passed": passed,
        "git_commit": git_commit(repository),
        "zarr": str(zarr_path.relative_to(repository)),
        "stats": str(stats_path.relative_to(repository)),
        "stats_sha256": sha256(stats_path),
        "figure": str(output.relative_to(repository)),
        "figure_sha256": sha256(output),
        "selected_date": str(selected_date),
        "training_years": [train_start, train_end],
        "sample_days": args.sample_days,
        "pixels_per_day": args.pixels_per_day,
        "seed": args.seed,
        "normalization_metrics": metrics,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_partial = report_path.with_suffix(report_path.suffix + ".part")
    report_partial.write_text(json.dumps(report, indent=2) + "\n")
    report_partial.replace(report_path)

    if not passed:
        raise RuntimeError(
            f"normalization diagnostics FAILED; inspect {output} and {report_path}"
        )
    print(f"NORMALIZATION DIAGNOSTICS PASSED; wrote {output}", flush=True)
    print(f"wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
