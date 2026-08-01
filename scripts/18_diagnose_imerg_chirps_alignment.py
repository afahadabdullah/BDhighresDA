#!/usr/bin/env python3
"""Separate temporal lag from spatial displacement in IMERG versus CHIRPS.

The BMD experiment stores CHIRPS on the 0.05-degree model grid and IMERG on
the native nested 0.1-degree grid.  This diagnostic first applies the same
physical 2x2 footprint mean to CHIRPS, then searches small time lags and
spatial shifts using pattern correlation.  It is a diagnostic only: a shift
that maximizes five days of data is not a correction to apply to IMERG.
"""

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
    parser.add_argument(
        "--dump", default="data/processed/bmd_imerg_controlled_20180501_05.npz"
    )
    parser.add_argument(
        "--out-json",
        default="data/processed/imerg_chirps_alignment_20180501_05.json",
    )
    parser.add_argument(
        "--out-plot",
        default="data/processed/imerg_chirps_alignment_20180501_05.png",
    )
    parser.add_argument("--max-lag", type=int, default=2)
    parser.add_argument("--max-shift", type=int, default=4)
    return parser.parse_args()


def aggregate_chirps(
    chirps: np.ndarray, valid: np.ndarray, factor: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Average fine CHIRPS to exact coarse footprints, retaining full land cells."""

    if chirps.ndim != 3 or valid.shape != chirps.shape[1:]:
        raise ValueError("CHIRPS must be (time, y, x) and valid must match its grid")
    _, height, width = chirps.shape
    if height % factor or width % factor:
        raise ValueError(f"factor {factor} does not divide CHIRPS shape {(height, width)}")
    fine_valid = np.asarray(valid) > 0.5
    coarse_valid = fine_valid.reshape(
        height // factor, factor, width // factor, factor
    ).all(axis=(1, 3))
    masked = np.where(fine_valid[None], chirps, 0.0)
    coarse = masked.reshape(
        len(chirps), height // factor, factor, width // factor, factor
    ).mean(axis=(2, 4))
    coarse[:, ~coarse_valid] = np.nan
    return coarse, coarse_valid


def time_pair(
    target: np.ndarray, source: np.ndarray, lag: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return CHIRPS[t] and IMERG[t + lag]."""

    if lag < 0:
        return target[-lag:], source[:lag]
    if lag > 0:
        return target[:-lag], source[lag:]
    return target, source


def spatial_pair(
    target: np.ndarray,
    source: np.ndarray,
    valid: np.ndarray,
    dy: int,
    dx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shift source onto target; positive dx/east and dy/north."""

    height, width = target.shape[-2:]
    if abs(dy) >= height or abs(dx) >= width:
        raise ValueError("spatial shift leaves no overlap")
    if dy >= 0:
        target_y, source_y = slice(dy, height), slice(0, height - dy)
    else:
        target_y, source_y = slice(0, height + dy), slice(-dy, height)
    if dx >= 0:
        target_x, source_x = slice(dx, width), slice(0, width - dx)
    else:
        target_x, source_x = slice(0, width + dx), slice(-dx, width)
    return (
        target[..., target_y, target_x],
        source[..., source_y, source_x],
        valid[target_y, target_x],
    )


def scores(
    target: np.ndarray, source: np.ndarray, valid: np.ndarray
) -> dict[str, float | int]:
    """Return pooled raw and spatial-anomaly scores over paired daily fields."""

    mask = np.broadcast_to(valid, target.shape).copy()
    mask &= np.isfinite(target) & np.isfinite(source)
    sample_count = int(mask.sum())
    if sample_count < 10:
        return {
            "pattern_correlation": float("nan"),
            "raw_correlation": float("nan"),
            "rmse_mm": float("nan"),
            "sample_count": sample_count,
        }
    target_values = target[mask].astype(np.float64)
    source_values = source[mask].astype(np.float64)
    raw_correlation = float(np.corrcoef(target_values, source_values)[0, 1])
    rmse = float(np.sqrt(np.mean((source_values - target_values) ** 2)))

    target_anomaly = np.full(target.shape, np.nan, dtype=np.float64)
    source_anomaly = np.full(source.shape, np.nan, dtype=np.float64)
    for day in range(len(target)):
        day_mask = mask[day]
        if day_mask.sum() < 3:
            continue
        target_anomaly[day, day_mask] = (
            target[day, day_mask] - np.mean(target[day, day_mask])
        )
        source_anomaly[day, day_mask] = (
            source[day, day_mask] - np.mean(source[day, day_mask])
        )
    anomaly_mask = np.isfinite(target_anomaly) & np.isfinite(source_anomaly)
    pattern_correlation = float(
        np.corrcoef(target_anomaly[anomaly_mask], source_anomaly[anomaly_mask])[0, 1]
    )
    return {
        "pattern_correlation": pattern_correlation,
        "raw_correlation": raw_correlation,
        "rmse_mm": rmse,
        "sample_count": sample_count,
    }


def evaluate(
    chirps: np.ndarray,
    imerg: np.ndarray,
    valid: np.ndarray,
    lag: int,
    dy: int,
    dx: int,
) -> dict[str, float | int]:
    target, source = time_pair(chirps, imerg, lag)
    target, source, shifted_valid = spatial_pair(target, source, valid, dy, dx)
    result = scores(target, source, shifted_valid)
    result.update({"lag_days": lag, "dy_cells": dy, "dx_cells": dx})
    return result


def best_result(results: list[dict[str, float | int]]) -> dict[str, float | int]:
    finite = [item for item in results if np.isfinite(item["pattern_correlation"])]
    if not finite:
        raise ValueError("no finite alignment correlations")
    return max(finite, key=lambda item: float(item["pattern_correlation"]))


def json_safe(value):
    """Replace non-finite diagnostics with JSON null while keeping strict JSON."""

    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    return value


def diagnosis(
    baseline: dict[str, float | int],
    temporal: dict[str, float | int],
    spatial: dict[str, float | int],
) -> tuple[str, str]:
    timing_gain = float(temporal["pattern_correlation"]) - float(
        baseline["pattern_correlation"]
    )
    spatial_gain = float(spatial["pattern_correlation"]) - float(
        baseline["pattern_correlation"]
    )
    timing_signal = abs(int(temporal["lag_days"])) > 0 and timing_gain >= 0.05
    spatial_signal = (
        abs(int(spatial["dx_cells"])) + abs(int(spatial["dy_cells"])) > 0
        and spatial_gain >= 0.05
    )
    if timing_signal and spatial_signal:
        label = "mixed_timing_and_spatial_signal"
    elif timing_signal:
        label = "possible_timing_signal"
    elif spatial_signal:
        label = "possible_spatial_displacement"
    else:
        label = "no_clear_fixed_lag_or_shift"
    explanation = (
        "Exploratory five-day diagnosis only. A timing signal requires a nonzero "
        "lag to improve pattern correlation by at least 0.05; a spatial signal "
        "uses the same threshold. Confirm any result on at least one full month "
        "before changing timestamps or coordinates."
    )
    return label, explanation


def main() -> None:
    args = parse_args()
    if args.max_lag < 0 or args.max_shift < 0:
        raise ValueError("max lag and max shift must be nonnegative")
    with np.load(args.dump, allow_pickle=False) as data:
        chirps_fine = np.asarray(data["chirps"], dtype=np.float64)
        imerg = np.asarray(data["imerg"], dtype=np.float64)
        valid_fine = np.asarray(data["valid"], dtype=np.float64)
        time = np.asarray(data["time"]).astype("datetime64[ns]").astype("datetime64[D]")
        fine_lat = np.asarray(data["grid_lat"], dtype=np.float64)
        fine_lon = np.asarray(data["grid_lon"], dtype=np.float64)

    chirps, valid = aggregate_chirps(chirps_fine, valid_fine, factor=2)
    if imerg.shape != chirps.shape:
        raise ValueError(f"IMERG {imerg.shape} and 0.1-degree CHIRPS {chirps.shape} differ")
    if len(time) < 2 * args.max_lag + 1:
        raise ValueError(
            f"{len(time)} days cannot support +/-{args.max_lag}-day search; "
            "reduce --max-lag"
        )

    lags = list(range(-args.max_lag, args.max_lag + 1))
    shifts = list(range(-args.max_shift, args.max_shift + 1))
    zero_shift_by_lag = [evaluate(chirps, imerg, valid, lag, 0, 0) for lag in lags]
    same_day_by_shift = [
        evaluate(chirps, imerg, valid, 0, dy, dx) for dy in shifts for dx in shifts
    ]
    joint = [
        evaluate(chirps, imerg, valid, lag, dy, dx)
        for lag in lags
        for dy in shifts
        for dx in shifts
    ]
    baseline = next(item for item in zero_shift_by_lag if item["lag_days"] == 0)
    best_temporal = best_result(zero_shift_by_lag)
    best_spatial = best_result(same_day_by_shift)
    best_joint = best_result(joint)

    per_day = []
    for day in range(len(time)):
        candidates = [
            evaluate(chirps[day : day + 1], imerg[day : day + 1], valid, 0, dy, dx)
            for dy in shifts
            for dx in shifts
        ]
        selected = best_result(candidates)
        selected["date"] = str(time[day])
        per_day.append(selected)

    label, explanation = diagnosis(baseline, best_temporal, best_spatial)
    report = {
        "scope": {
            "dump": str(args.dump),
            "dates": [str(value) for value in time],
            "days": len(time),
            "comparison_grid_degrees": 0.1,
            "chirps_aggregation": "exact 2x2 mean; fully land-covered footprints only",
            "max_lag_days": args.max_lag,
            "max_shift_cells": args.max_shift,
            "shift_cell_degrees": 0.1,
        },
        "conventions": {
            "lag_days": "IMERG date minus CHIRPS date; +1 compares CHIRPS[t] with IMERG[t+1]",
            "dx_cells": "shift applied to IMERG; positive is east/right",
            "dy_cells": "shift applied to IMERG; positive is north/up",
        },
        "same_date_unshifted": baseline,
        "best_time_lag_without_spatial_shift": best_temporal,
        "best_spatial_shift_on_same_date": best_spatial,
        "best_joint_lag_and_shift": best_joint,
        "gains_over_same_date_unshifted": {
            "timing_pattern_correlation": float(best_temporal["pattern_correlation"])
            - float(baseline["pattern_correlation"]),
            "spatial_pattern_correlation": float(best_spatial["pattern_correlation"])
            - float(baseline["pattern_correlation"]),
            "joint_pattern_correlation": float(best_joint["pattern_correlation"])
            - float(baseline["pattern_correlation"]),
        },
        "per_day_best_same_date_spatial_shift": per_day,
        "diagnosis": {"label": label, "interpretation": explanation},
        "zero_shift_lag_scores": zero_shift_by_lag,
    }

    # Plot a compact decision suite, emphasizing that support is matched first.
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    axis = axes[0, 0]
    axis.plot(
        lags,
        [item["pattern_correlation"] for item in zero_shift_by_lag],
        marker="o",
        color="#007C91",
    )
    axis.axvline(0, color="black", ls="--", lw=1)
    axis.set_xticks(lags)
    axis.set_xlabel("Lag (days): IMERG date − CHIRPS date")
    axis.set_ylabel("Pattern correlation")
    axis.set_title("A. Timing test (no spatial shift)")
    axis.grid(alpha=0.25)

    axis = axes[0, 1]
    shift_matrix = np.full((len(shifts), len(shifts)), np.nan)
    for item in same_day_by_shift:
        shift_matrix[shifts.index(int(item["dy_cells"])), shifts.index(int(item["dx_cells"]))] = item[
            "pattern_correlation"
        ]
    image = axis.imshow(
        shift_matrix,
        origin="lower",
        extent=[shifts[0] - 0.5, shifts[-1] + 0.5, shifts[0] - 0.5, shifts[-1] + 0.5],
        cmap="viridis",
        aspect="equal",
    )
    axis.scatter(best_spatial["dx_cells"], best_spatial["dy_cells"], marker="x", s=90, c="white")
    axis.set_xlabel("IMERG shift east (+) / west (−), 0.1° cells")
    axis.set_ylabel("IMERG shift north (+) / south (−), 0.1° cells")
    axis.set_title("B. Same-day spatial-shift test")
    figure.colorbar(image, ax=axis, label="Pattern correlation", shrink=0.8)

    axis = axes[0, 2]
    axis.axhline(0, color="black", lw=0.8)
    axis.plot(
        np.arange(len(per_day)),
        [item["dx_cells"] for item in per_day],
        marker="o",
        label="east/west dx",
    )
    axis.plot(
        np.arange(len(per_day)),
        [item["dy_cells"] for item in per_day],
        marker="s",
        label="north/south dy",
    )
    axis.set_xticks(np.arange(len(time)))
    axis.set_xticklabels([str(value)[5:] for value in time], rotation=35)
    axis.set_ylabel("Best same-day shift (0.1° cells)")
    axis.set_title("C. Is the spatial shift consistent by day?")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25)

    land_count = valid.sum()
    daily_mean = np.nansum(chirps * valid[None], axis=(1, 2)) / land_count
    representative = int(np.nanargmax(daily_mean))
    precipitation_max = max(1.0, float(
        np.nanpercentile(
            np.concatenate([chirps[representative][valid], imerg[representative][valid]]),
            99,
        )
    ))
    extent = [fine_lon[0] - 0.025, fine_lon[-1] + 0.025,
              fine_lat[0] - 0.025, fine_lat[-1] + 0.025]
    for axis, field, title in (
        (axes[1, 0], chirps[representative], "D. CHIRPS aggregated to 0.1°"),
        (axes[1, 1], imerg[representative], "E. IMERG native 0.1°"),
    ):
        shown = axis.imshow(field, origin="lower", extent=extent, cmap="viridis", vmin=0, vmax=precipitation_max)
        axis.set_title(title + f"\n{time[representative]}")
        axis.set_xlabel("Longitude (°E)")
        axis.set_ylabel("Latitude (°N)")
        figure.colorbar(shown, ax=axis, label="mm day$^{-1}$", shrink=0.8)

    axis = axes[1, 2]
    difference = imerg[representative] - chirps[representative]
    error_limit = max(1.0, float(np.nanpercentile(np.abs(difference[valid]), 98)))
    shown = axis.imshow(
        difference,
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        vmin=-error_limit,
        vmax=error_limit,
    )
    axis.set_title("F. Same-date difference\nIMERG − footprint-matched CHIRPS")
    axis.set_xlabel("Longitude (°E)")
    axis.set_ylabel("Latitude (°N)")
    figure.colorbar(shown, ax=axis, label="mm day$^{-1}$", shrink=0.8)
    figure.suptitle(
        "IMERG–CHIRPS timing versus spatial-alignment diagnostic\n"
        f"{label}; five days are exploratory, not sufficient for a coordinate correction",
        fontsize=15,
    )

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_plot).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(
        json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n"
    )
    figure.savefig(args.out_plot, dpi=180)
    plt.close(figure)
    print(f"same-date unshifted correlation {baseline['pattern_correlation']:.3f}")
    print(
        "best time-only: "
        f"lag {best_temporal['lag_days']:+d} day, r={best_temporal['pattern_correlation']:.3f}"
    )
    print(
        "best same-day spatial: "
        f"dx {best_spatial['dx_cells']:+d}, dy {best_spatial['dy_cells']:+d} cells, "
        f"r={best_spatial['pattern_correlation']:.3f}"
    )
    print(f"diagnosis: {label}")
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_plot}")


if __name__ == "__main__":
    main()
