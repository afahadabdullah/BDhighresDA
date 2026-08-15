#!/usr/bin/env python
"""Download CHIRPS daily 0.05 deg rainfall and subset to the wide domain.

CHIRPS is the training TARGET.  It is land-only (ocean cells are the fill value
-9999), gauge-blended, and available 1981-present at exactly 0.05 deg -- which
is why it, rather than ERA5-Land (9 km) or IMERG (0.1 deg), is the truth here.

    python scripts/01_download_chirps.py --start 1981 --end 2025 --out data/raw/chirps

The global daily files are ~1 GB/yr. Downloads are resumable and written via
``.part`` files; each global file is subset immediately and deleted unless
``--keep-global`` is passed.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bdhires.grids import get_grid  # noqa: E402

BASE = (
    "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p05/"
    "chirps-v2.0.{year}.days_p05.nc"
)
# CHIRPS v3 (0.05 deg, 1981-present) lives under products/CHIRPS/v3.0/ ;
# switch BASE when you are ready to move -- v3 improves the satellite
# calibration over South Asia but changes the climatology, so do not mix.


def validate_chirps(path: Path) -> None:
    """Raise if *path* is not a readable CHIRPS NetCDF file."""
    with xr.open_dataset(path) as ds:
        missing_dims = {"time", "latitude", "longitude"} - set(ds.dims)
        if missing_dims:
            raise ValueError(f"{path} is missing dimensions: {sorted(missing_dims)}")
        if "precip" not in ds:
            raise ValueError(f"{path} does not contain the CHIRPS 'precip' variable")


def download_year(url: str, destination: Path) -> None:
    """Resume *url* into a temporary file and atomically publish it."""
    if destination.exists():
        try:
            validate_chirps(destination)
            return
        except (OSError, ValueError) as exc:
            print(f"existing download is incomplete ({exc}); resuming it", flush=True)
            destination.replace(destination.with_suffix(destination.suffix + ".part"))

    partial = destination.with_suffix(destination.suffix + ".part")
    command = [
        "wget",
        "--continue",
        "--tries=10",
        "--timeout=60",
        "--waitretry=10",
        "--retry-connrefused",
        "--output-document",
        str(partial),
        url,
    ]

    # A stale partial file can occasionally have the correct byte count but
    # invalid contents (for example, after a proxy error). Retry once from
    # scratch if NetCDF validation fails after wget reports success.
    for attempt in range(2):
        print("downloading", url, flush=True)
        subprocess.run(command, check=True)
        try:
            validate_chirps(partial)
        except (OSError, ValueError):
            if attempt:
                raise
            print("download validation failed; retrying once from scratch", flush=True)
            partial.unlink(missing_ok=True)
            continue
        partial.replace(destination)
        return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1981)
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument("--out", default="data/raw/chirps")
    ap.add_argument(
        "--grid",
        default="wide",
        choices=("wide", "wide_cpc"),
        help="wide_cpc is the CPC-edge-aligned 240x240 V3-SG domain",
    )
    ap.add_argument("--keep-global", action="store_true")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print the requested URLs and output files without downloading",
    )
    args = ap.parse_args()

    if args.start > args.end:
        ap.error("--start must be less than or equal to --end")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    grid = get_grid(args.grid)
    lo, la, hi, ha = grid.bbox

    for year in range(args.start, args.end + 1):
        sub = out / f"chirps_{grid.name}_{year}.nc"
        url = BASE.format(year=year)
        glob_f = out / Path(url).name

        if args.dry_run:
            print(f"{year}: {url} -> {sub}")
            continue

        if sub.exists():
            try:
                validate_chirps(sub)
                print("already complete", sub, flush=True)
                continue
            except (OSError, ValueError) as exc:
                print(f"removing invalid regional file {sub}: {exc}", flush=True)
                sub.unlink()

        download_year(url, glob_f)
        partial_sub = sub.with_suffix(sub.suffix + ".part")
        with xr.open_dataset(glob_f) as ds:
            regional = ds.sel(longitude=slice(lo, hi), latitude=slice(la, ha))
            regional["precip"] = regional["precip"].where(
                regional["precip"] > -100
            )  # -9999 -> NaN
            regional = regional.sortby("latitude")  # project convention
            sizes = dict(regional.sizes)
            regional.to_netcdf(
                partial_sub,
                encoding={"precip": {"zlib": True, "complevel": 4}},
            )
        validate_chirps(partial_sub)
        partial_sub.replace(sub)
        print("wrote", sub, sizes, flush=True)
        if not args.keep_global:
            glob_f.unlink()


if __name__ == "__main__":
    main()
