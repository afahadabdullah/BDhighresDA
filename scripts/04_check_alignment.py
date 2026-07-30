#!/usr/bin/env python3
"""Verify that packed ERA5 and CHIRPS daily precipitation peak at lag zero."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def lag_correlations(
    target: np.ndarray,
    background: np.ndarray,
    lags: list[int],
) -> dict[int, float]:
    """Correlate target[t] with background[t + lag] for each requested lag."""
    target = np.asarray(target, dtype=np.float64)
    background = np.asarray(background, dtype=np.float64)
    if target.shape != background.shape or target.ndim != 1:
        raise ValueError("target and background must be one-dimensional peers")
    correlations = {}
    for lag in lags:
        if lag < 0:
            x = target[-lag:]
            y = background[:lag]
        elif lag > 0:
            x = target[:-lag]
            y = background[lag:]
        else:
            x = target
            y = background
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() < 3:
            raise ValueError(f"lag {lag} has fewer than three valid pairs")
        correlations[lag] = float(np.corrcoef(x[finite], y[finite])[0, 1])
    return correlations


def area_mean_series(root, block_days: int) -> tuple[np.ndarray, np.ndarray]:
    valid = np.asarray(root["valid"][:]) > 0.5
    latitude = np.asarray(root["lat"][:], dtype=np.float64)
    weights = np.cos(np.deg2rad(latitude))[:, None] * valid
    denominator = weights.sum()
    if denominator <= 0:
        raise ValueError("packed store has an empty land-validity mask")

    channels = list(root.attrs.get("cond_channels", []))
    if "era5_tp" not in channels:
        raise ValueError("packed store does not identify an era5_tp channel")
    tp_index = channels.index("era5_tp")
    ntime = root["time"].shape[0]
    target_mean = np.empty(ntime, dtype=np.float64)
    background_mean = np.empty(ntime, dtype=np.float64)
    for start in range(0, ntime, block_days):
        stop = min(start + block_days, ntime)
        target = np.asarray(root["target"][start:stop], dtype=np.float64)
        background = np.asarray(
            root["cond"][start:stop, tp_index, :, :],
            dtype=np.float64,
        )
        target_finite = np.isfinite(target)
        target_numerator = np.nansum(target * weights[None], axis=(1, 2))
        target_denominator = np.sum(
            target_finite * weights[None],
            axis=(1, 2),
        )
        target_mean[start:stop] = target_numerator / target_denominator
        background_mean[start:stop] = np.sum(
            background * weights[None],
            axis=(1, 2),
        ) / denominator
        print(f"read alignment block {start}:{stop}", flush=True)
    return target_mean, background_mean


def open_zarr_group(path: Path):
    import zarr

    try:
        return zarr.open_group(str(path), mode="r", zarr_format=2)
    except TypeError:
        return zarr.open_group(str(path), mode="r", zarr_version=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr", default="data/processed/bd_wide.zarr")
    parser.add_argument(
        "--out",
        default="data/processed/alignment_qc.json",
    )
    parser.add_argument("--block-days", type=int, default=31)
    parser.add_argument("--lags", nargs="+", type=int, default=[-2, -1, 0, 1, 2])
    args = parser.parse_args()
    if args.block_days < 1:
        parser.error("--block-days must be positive")
    if 0 not in args.lags:
        parser.error("--lags must include zero")

    root = open_zarr_group(Path(args.zarr))
    if not root.attrs.get("complete", False):
        raise ValueError(f"{args.zarr} is not marked complete")
    time = np.asarray(root["time"][:]).astype("datetime64[ns]")
    expected = np.arange(
        time[0].astype("datetime64[D]"),
        time[-1].astype("datetime64[D]") + np.timedelta64(1, "D"),
        dtype="datetime64[D]",
    )
    if not np.array_equal(time.astype("datetime64[D]"), expected):
        raise ValueError("packed time axis is not unique, daily and contiguous")

    target_mean, background_mean = area_mean_series(root, args.block_days)
    correlations = lag_correlations(target_mean, background_mean, args.lags)
    peak_lag = max(correlations, key=correlations.get)
    passed = peak_lag == 0

    print("Lag convention: correlate CHIRPS[t] with ERA5[t + lag]")
    print(" lag    Pearson r")
    for lag in sorted(correlations):
        print(f"{lag:+4d}    {correlations[lag]:.6f}")
    print(f"peak lag: {peak_lag:+d}")

    report = {
        "zarr": str(args.zarr),
        "time_start": str(time[0]),
        "time_end": str(time[-1]),
        "n_days": int(len(time)),
        "series": "cos(latitude)-weighted WIDE-domain land mean precipitation",
        "lag_convention": "correlate CHIRPS[t] with ERA5[t + lag]",
        "correlations": {str(lag): value for lag, value in correlations.items()},
        "peak_lag_days": int(peak_lag),
        "passed": passed,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    partial.write_text(json.dumps(report, indent=2) + "\n")
    partial.replace(output)

    if not passed:
        raise RuntimeError(
            f"alignment FAILED: maximum correlation occurs at lag {peak_lag:+d}, "
            "not lag 0"
        )
    print(f"alignment PASSED; wrote {output}", flush=True)


if __name__ == "__main__":
    main()
