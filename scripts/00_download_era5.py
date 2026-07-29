#!/usr/bin/env python
"""Download ERA5 single-level + pressure-level daily predictors over the wide domain.

Requires a CDS API key in ~/.cdsapirc (https://cds.climate.copernicus.eu/how-to-api).

    python scripts/00_download_era5.py --start 1981 --end 2025 --out data/raw/era5

Notes
-----
* Only five conditioning channels are downloaded by default -- see the CORE
  list below for why.  Use ``--extended`` for the ablation set.
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

# ---------------------------------------------------------------------------
# CONDITIONING SET -- deliberately minimal: 5 channels.
#
# With ~14k daily training samples, every extra channel is capacity spent on
# something the network must learn to ignore.  These five answer the only
# questions that matter for a daily rainfall total over the Bengal delta:
#
#   tp     how much rain did the background model itself produce?
#          (a MODEL field, not an observation -- that is why it belongs here)
#   tcwv   how much moisture is available in the column?
#   ivte   how much moisture is being transported, and from where?
#   ivtn     -> the monsoon flux hitting the Meghalaya barrier is the single
#              mechanism behind the domain's rainfall maximum
#   cape   is the atmosphere unstable enough to convect it out?
#
# All five are SINGLE-LEVEL, so no pressure-level request is needed at all --
# the CDS download shrinks by roughly an order of magnitude.
#
# Everything else (winds and humidity on levels, CIN, BLH, stability indices,
# shear, dewpoint depression) is available behind --extended and should be
# treated as an ABLATION, not a default.  Add channels only if the validation
# CRPS actually improves.
# ---------------------------------------------------------------------------

CORE = [
    "total_precipitation",
    "total_column_water_vapour",
    "vertical_integral_of_eastward_water_vapour_flux",
    "vertical_integral_of_northward_water_vapour_flux",
    "convective_available_potential_energy",
]

EXTENDED_SINGLE = [
    "mean_sea_level_pressure",
    "2m_temperature",
    "2m_dewpoint_temperature",
    "convective_inhibition",
    "convective_precipitation",
    "vertical_integral_of_divergence_of_moisture_flux",
]

EXTENDED_PRESSURE = {
    "variable": ["u_component_of_wind", "v_component_of_wind",
                 "specific_humidity", "vertical_velocity"],
    "pressure_level": ["850", "500"],
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
    ap.add_argument("--extended", action="store_true",
                    help="also fetch the optional ablation channels (adds a "
                         "pressure-level request and roughly 10x the volume)")
    ap.add_argument("--ensemble", action="store_true",
                    help="also fetch the 10-member ERA5 EDA (0.5 deg, 3-hourly). "
                         "Gives each analysis member its own background, which is "
                         "the physically correct source of background-error spread "
                         "-- see docs/METHODOLOGY.md Section 6.")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    single = CORE + (EXTENDED_SINGLE if args.extended else [])

    if args.dry_run:
        print("area (N,W,S,E):", area())
        print(f"core single-level channels: {len(CORE)}")
        for v in CORE:
            print(f"    {v}")
        if args.extended:
            npl = len(EXTENDED_PRESSURE["variable"]) * len(EXTENDED_PRESSURE["pressure_level"])
            print(f"extended: +{len(EXTENDED_SINGLE)} single-level, +{npl} pressure-level")
        print(f"total ERA5 channels: {len(single) + (len(EXTENDED_PRESSURE['variable']) * len(EXTENDED_PRESSURE['pressure_level']) if args.extended else 0)}")
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
                    dict(product_type="reanalysis", variable=single, year=str(year),
                         month=f"{month:02d}", day=days, time=hours, area=area(),
                         data_format="netcdf"),
                    str(f1),
                )
                print("wrote", f1, flush=True)

            if args.ensemble:
                f3 = out / f"era5_eda_{tag}.nc"
                if not f3.exists():
                    c.retrieve(
                        "reanalysis-era5-single-levels",
                        dict(product_type="ensemble_members",
                             variable=["total_column_water_vapour", "mean_sea_level_pressure",
                                       "2m_temperature", "total_precipitation"],
                             year=str(year), month=f"{month:02d}", day=days,
                             time=[f"{h:02d}:00" for h in range(0, 24, 3)],
                             area=area(), data_format="netcdf"),
                        str(f3),
                    )
                    print("wrote", f3, flush=True)

            if not args.extended:
                continue

            f2 = out / f"era5_pl_{tag}.nc"
            if not f2.exists():
                c.retrieve(
                    "reanalysis-era5-pressure-levels",
                    dict(product_type="reanalysis", **EXTENDED_PRESSURE, year=str(year),
                         month=f"{month:02d}", day=days,
                         time=[f"{h:02d}:00" for h in (0, 6, 12, 18)],
                         area=area(), data_format="netcdf"),
                    str(f2),
                )
                print("wrote", f2, flush=True)


if __name__ == "__main__":
    main()
