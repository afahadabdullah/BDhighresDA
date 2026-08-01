#!/usr/bin/env python
"""Validate and crop existing daily IMERG V07B files for real-data DA."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bdhires.imerg import load_imerg_daily  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/imerg")
    parser.add_argument("--start", default="2018-05-01")
    parser.add_argument("--end", default="2018-05-31")
    parser.add_argument("--min-count", type=int, default=40)
    parser.add_argument("--out", default="data/processed/imerg_bd_may2018.nc")
    parser.add_argument("--report", default="data/processed/imerg_bd_may2018_qc.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    imerg = load_imerg_daily(
        args.input, args.start, args.end, min_count=args.min_count
    )
    valid = np.isfinite(imerg.precipitation)
    if not valid.any():
        raise ValueError("no valid regional IMERG footprints after quality control")
    dataset = xr.Dataset(
        data_vars={
            "precipitation": (
                ("time", "lat", "lon"),
                imerg.precipitation,
                {"units": "mm/day", "long_name": "IMERG Final daily precipitation"},
            ),
            "randomError": (
                ("time", "lat", "lon"),
                imerg.random_error,
                {"units": "mm/day", "long_name": "IMERG random RMS error estimate"},
            ),
            "precipitation_cnt": (
                ("time", "lat", "lon"),
                imerg.count,
                {"units": "count", "long_name": "valid half-hourly retrieval count"},
            ),
        },
        coords={"time": imerg.time, "lat": imerg.lat, "lon": imerg.lon},
        attrs={
            "product": "GPM_3IMERGDF",
            "version": "V07B",
            "source": "NASA GES DISC daily Final Run granules",
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
        "product": "GPM_3IMERGDF",
        "version": "V07B",
        "period": {"start": args.start, "end": args.end, "days": int(len(imerg.time))},
        "source_directory": str(Path(args.input)),
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
        ),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"[imerg] {len(imerg.time)} days, {valid.mean():.1%} valid footprints, "
        f"randomError median {np.nanmedian(imerg.random_error):.2f} mm/day"
    )
    print(f"wrote {output}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
