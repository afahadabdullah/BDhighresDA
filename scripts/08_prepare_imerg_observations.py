#!/usr/bin/env python
"""Prepare IMERG V07B observations for real-data DA.

The scientifically supported BMD path accumulates half-hourly rates over the
24-hour reporting window ending at 03:00 UTC on the BMD archive date.  The
legacy calendar-day input remains available only for reproducibility.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bdhires.imerg import load_imerg_bmd_windows, load_imerg_daily  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/imerg_halfhourly")
    parser.add_argument(
        "--source-frequency",
        choices=("half-hourly", "daily"),
        default="half-hourly",
        help="half-hourly is required for exact BMD 03:00-03:00 UTC windows",
    )
    parser.add_argument("--start", default="2018-05-01")
    parser.add_argument("--end", default="2018-05-31")
    parser.add_argument(
        "--min-count",
        type=int,
        default=None,
        help="valid intervals required per footprint (default 48 half-hourly, 40 daily)",
    )
    parser.add_argument(
        "--accumulation-end-hour-utc",
        type=int,
        default=3,
        help="BMD archive day ends at this UTC hour (default: 3)",
    )
    parser.add_argument("--out", default="data/processed/imerg_bd_may2018.nc")
    parser.add_argument("--report", default="data/processed/imerg_bd_may2018_qc.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    min_count = args.min_count
    if min_count is None:
        min_count = 48 if args.source_frequency == "half-hourly" else 40
    if args.source_frequency == "half-hourly":
        imerg = load_imerg_bmd_windows(
            args.input,
            args.start,
            args.end,
            min_count=min_count,
            accumulation_end_hour_utc=args.accumulation_end_hour_utc,
        )
        product = "GPM_3IMERGHH"
        source = "NASA GES DISC half-hourly Final Run granules"
        accumulation_window = (
            f"previous-day {args.accumulation_end_hour_utc:02d}:00 UTC to selected-day "
            f"{args.accumulation_end_hour_utc:02d}:00 UTC"
        )
    else:
        imerg = load_imerg_daily(
            args.input, args.start, args.end, min_count=min_count
        )
        product = "GPM_3IMERGDF"
        source = "NASA GES DISC daily Final Run granules"
        accumulation_window = "calendar day 00:00-24:00 UTC; not aligned to BMD"
    valid = np.isfinite(imerg.precipitation)
    if not valid.any():
        raise ValueError("no valid regional IMERG footprints after quality control")
    dataset = xr.Dataset(
        data_vars={
            "precipitation": (
                ("time", "lat", "lon"),
                imerg.precipitation,
                {
                    "units": "mm/day",
                    "long_name": "IMERG Final precipitation over the prepared 24-hour window",
                },
            ),
            "randomError": (
                ("time", "lat", "lon"),
                imerg.random_error,
                {
                    "units": "mm/day",
                    "long_name": "IMERG random-error standard deviation over the prepared window",
                    "aggregation": imerg.random_error_aggregation,
                },
            ),
            "precipitation_cnt": (
                ("time", "lat", "lon"),
                imerg.count,
                {"units": "count", "long_name": "valid half-hourly retrieval count"},
            ),
        },
        coords={"time": imerg.time, "lat": imerg.lat, "lon": imerg.lon},
        attrs={
            "product": product,
            "version": "V07B",
            "source": source,
            "source_frequency": imerg.source_frequency,
            "bmd_accumulation_end_hour_utc": int(imerg.accumulation_end_hour_utc),
            "accumulation_window": accumulation_window,
            "window_duration_hours": 24,
            "time_coordinate_semantics": "BMD archive date; end of accumulation window",
            "random_error_aggregation": imerg.random_error_aggregation,
            "spatial_support": "exact 0.1-degree footprints nested over BD 0.05-degree grid",
            "quality_control": f"precipitation_cnt >= {imerg.min_count}; finite nonnegative values",
            "bias_correction": "none",
        },
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    dataset.to_netcdf(temporary)
    temporary.replace(output)

    daily_valid = valid.sum(axis=(1, 2))
    report = {
        "product": product,
        "version": "V07B",
        "source_frequency": imerg.source_frequency,
        "period": {"start": args.start, "end": args.end, "days": int(len(imerg.time))},
        "accumulation": {
            "window": accumulation_window,
            "end_hour_utc": int(imerg.accumulation_end_hour_utc),
            "duration_hours": 24,
            "time_coordinate": "BMD archive date and window end",
            "half_hourly_rate_to_depth": (
                "sum(precipitation_mm_per_hour * 0.5_hour)"
                if args.source_frequency == "half-hourly"
                else None
            ),
            "random_error": imerg.random_error_aggregation,
        },
        "source_directory": str(Path(args.input)),
        "source_file_count": len(imerg.source_files),
        "source_files": list(imerg.source_files),
        "grid": {
            "shape": list(imerg.precipitation.shape[1:]),
            "resolution_degrees": 0.1,
            "lat_range_centres": [float(imerg.lat[0]), float(imerg.lat[-1])],
            "lon_range_centres": [float(imerg.lon[0]), float(imerg.lon[-1])],
        },
        "quality_control": {
            "minimum_half_hourly_count": imerg.min_count,
            "possible_footprints": int(valid.size),
            "valid_footprints": int(valid.sum()),
            "valid_fraction": float(valid.mean()),
            "daily_valid_min": int(daily_valid.min()),
            "daily_valid_median": float(np.median(daily_valid)),
            "daily_valid_max": int(daily_valid.max()),
        },
        "precipitation_mm_day": {
            "mean": float(np.nanmean(imerg.precipitation)),
            "p99": float(np.nanpercentile(imerg.precipitation, 99)),
            "max": float(np.nanmax(imerg.precipitation)),
        },
        "random_error_mm_day": {
            "median": float(np.nanmedian(imerg.random_error)),
            "p90": float(np.nanpercentile(imerg.random_error, 90)),
            "max": float(np.nanmax(imerg.random_error)),
        },
        "warning": (
            "This bounded process experiment uses native V07B values without a fitted "
            "IMERG-to-reference bias correction; do not treat it as final product skill."
            if args.source_frequency == "half-hourly"
            else "Calendar-day IMERG is not aligned to BMD 03:00 UTC reporting days and "
            "must not be used for BMD method selection."
        ),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"[imerg] {len(imerg.time)} BMD windows from {len(imerg.source_files)} "
        f"{imerg.source_frequency} files, {valid.mean():.1%} valid footprints, "
        f"randomError median {np.nanmedian(imerg.random_error):.2f} mm/day"
    )
    print(f"wrote {output}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
