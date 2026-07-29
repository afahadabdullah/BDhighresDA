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
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import xarray as xr

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


def validate_era5(path: Path) -> None:
    """Raise if *path* is not a readable regional ERA5 NetCDF file."""
    with xr.open_dataset(path) as ds:
        names = set(ds.coords) | set(ds.dims)
        if not ({"time", "valid_time"} & names):
            raise ValueError(f"{path} has no time coordinate")
        if not ({"latitude", "lat"} & names):
            raise ValueError(f"{path} has no latitude coordinate")
        if not ({"longitude", "lon"} & names):
            raise ValueError(f"{path} has no longitude coordinate")
        if not ds.data_vars:
            raise ValueError(f"{path} contains no ERA5 variables")


def publish_download(download: Path, target: Path) -> None:
    """Validate a CDS response and atomically publish it as one NetCDF file.

    Since the November 2024 CDS converter update, a request containing fields
    with different GRIB ``stepType`` values can be returned as a ZIP archive
    containing multiple NetCDF files.  The core request does exactly that:
    total precipitation is accumulated while the other predictors are
    instantaneous.  Merge those members here so downstream code still sees
    one monthly file.
    """
    if not zipfile.is_zipfile(download):
        validate_era5(download)
        download.replace(target)
        return

    staged = target.with_suffix(target.suffix + ".ready")
    staged.unlink(missing_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=f".{target.stem}-", dir=target.parent
    ) as tmp_name:
        tmp = Path(tmp_name)
        with zipfile.ZipFile(download) as archive:
            members = [
                info for info in archive.infolist()
                if not info.is_dir()
                and Path(info.filename).suffix.lower() in {".nc", ".nc4"}
            ]
            if not members:
                names = ", ".join(info.filename for info in archive.infolist())
                raise ValueError(
                    f"{download} is a ZIP archive with no NetCDF members: {names}"
                )

            extracted = []
            for index, member in enumerate(members):
                # Do not use extract(): archive paths are untrusted and may
                # contain absolute paths or ".." components.
                member_path = tmp / f"{index:03d}_{Path(member.filename).name}"
                with archive.open(member) as source, member_path.open("wb") as dest:
                    shutil.copyfileobj(source, dest)
                validate_era5(member_path)
                extracted.append(member_path)

        print(
            f"merging {len(extracted)} NetCDF members from {download.name}",
            flush=True,
        )
        datasets = [xr.open_dataset(path) for path in extracted]
        merged = None
        try:
            merged = xr.merge(
                datasets,
                compat="no_conflicts",
                join="outer",
                combine_attrs="override",
            )
            merged.to_netcdf(staged, engine="netcdf4")
        finally:
            if merged is not None:
                merged.close()
            for dataset in datasets:
                dataset.close()

    validate_era5(staged)
    staged.replace(target)
    download.unlink()


def retrieve_atomic(client, dataset: str, request: dict, target: Path) -> None:
    """Retrieve and validate one CDS request before publishing *target*."""
    if target.exists():
        try:
            validate_era5(target)
            print("already complete", target, flush=True)
            return
        except (OSError, ValueError) as exc:
            print(f"removing invalid ERA5 file {target}: {exc}", flush=True)
            target.unlink()

    partial = target.with_suffix(target.suffix + ".part")
    if partial.exists():
        print("recovering completed CDS download", partial, flush=True)
        try:
            publish_download(partial, target)
            print("wrote", target, flush=True)
            return
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            print(f"removing invalid partial download {partial}: {exc}", flush=True)
            partial.unlink()

    print("requesting", target, flush=True)
    client.retrieve(dataset, request, str(partial))
    publish_download(partial, target)
    print("wrote", target, flush=True)


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
            retrieve_atomic(
                c,
                "reanalysis-era5-single-levels",
                dict(
                    product_type="reanalysis",
                    variable=single,
                    year=str(year),
                    month=f"{month:02d}",
                    day=days,
                    time=hours,
                    area=area(),
                    data_format="netcdf",
                    download_format="unarchived",
                ),
                f1,
            )

            if args.ensemble:
                f3 = out / f"era5_eda_{tag}.nc"
                retrieve_atomic(
                    c,
                    "reanalysis-era5-single-levels",
                    dict(
                        product_type="ensemble_members",
                        variable=[
                            "total_column_water_vapour",
                            "mean_sea_level_pressure",
                            "2m_temperature",
                            "total_precipitation",
                        ],
                        year=str(year),
                        month=f"{month:02d}",
                        day=days,
                        time=[f"{h:02d}:00" for h in range(0, 24, 3)],
                        area=area(),
                        data_format="netcdf",
                        download_format="unarchived",
                    ),
                    f3,
                )

            if not args.extended:
                continue

            f2 = out / f"era5_pl_{tag}.nc"
            retrieve_atomic(
                c,
                "reanalysis-era5-pressure-levels",
                dict(
                    product_type="reanalysis",
                    **EXTENDED_PRESSURE,
                    year=str(year),
                    month=f"{month:02d}",
                    day=days,
                    time=[f"{h:02d}:00" for h in (0, 6, 12, 18)],
                    area=area(),
                    data_format="netcdf",
                    download_format="unarchived",
                ),
                f2,
            )


if __name__ == "__main__":
    main()
