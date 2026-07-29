#!/usr/bin/env python
"""Download ERA5 single-level + pressure-level daily predictors over the wide domain.

Requires a CDS API key in ~/.cdsapirc (https://cds.climate.copernicus.eu/how-to-api).

    python scripts/00_download_era5.py --start 1981 --end 2025 --out data/raw/era5

Notes
-----
* We download HOURLY fields and aggregate to daily here rather than using the
  "daily statistics" application, because precipitation needs a 00-00 UTC sum
  aligned with CHIRPS (which is a 00-00 UTC daily total) while the state
  variables want a daily mean.  Getting this alignment wrong is the single
  most common source of a spurious 1-day lag in downscaling studies.
* ERA5 total precipitation at hour H is the accumulation over the PRECEDING
  hour, so the daily total for day D is the sum of hours 01:00 (D) .. 00:00 (D+1).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bdhires.grids import WIDE  # noqa: E402

SINGLE = [
    "total_precipitation",
    "total_column_water_vapour",
    "2m_temperature",
    "2m_dewpoint_temperature",
    "mean_sea_level_pressure",
    "convective_available_potential_energy",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure",
    "vertical_integral_of_eastward_water_vapour_flux",
    "vertical_integral_of_northward_water_vapour_flux",
]

PRESSURE = {
    "variable": [
        "u_component_of_wind",
        "v_component_of_wind",
        "specific_humidity",
        "temperature",
        "vertical_velocity",
        "geopotential",
    ],
    "pressure_level": ["850", "700", "500", "200"],
}

PAD = 1.0  # degrees of halo so the regridder has neighbours at the edges


def area():
    lo, la, hi, ha = WIDE.bbox
    return [ha + PAD, lo - PAD, la - PAD, hi + PAD]  # N, W, S, E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1981)
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument("--out", default="data/raw/era5")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print("area (N,W,S,E):", area())
        print("single-level vars:", len(SINGLE))
        print("pressure-level fields:", len(PRESSURE["variable"]) * len(PRESSURE["pressure_level"]))
        return

    import cdsapi

    c = cdsapi.Client()
    for year in range(args.start, args.end + 1):
        for month in range(1, 13):
            tag = f"{year}{month:02d}"
            days = [f"{d:02d}" for d in range(1, 32)]
            hours = [f"{h:02d}:00" for h in range(24)]

            f1 = out / f"era5_sfc_{tag}.nc"
            if not f1.exists():
                c.retrieve(
                    "reanalysis-era5-single-levels",
                    dict(product_type="reanalysis", variable=SINGLE, year=str(year),
                         month=f"{month:02d}", day=days, time=hours, area=area(),
                         data_format="netcdf"),
                    str(f1),
                )
                print("wrote", f1, flush=True)

            f2 = out / f"era5_pl_{tag}.nc"
            if not f2.exists():
                c.retrieve(
                    "reanalysis-era5-pressure-levels",
                    dict(product_type="reanalysis", **PRESSURE, year=str(year),
                         month=f"{month:02d}", day=days,
                         time=[f"{h:02d}:00" for h in (0, 6, 12, 18)],
                         area=area(), data_format="netcdf"),
                    str(f2),
                )
                print("wrote", f2, flush=True)


if __name__ == "__main__":
    main()
