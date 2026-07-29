#!/usr/bin/env python
"""Download CHIRPS daily 0.05 deg rainfall and subset to the wide domain.

CHIRPS is the training TARGET.  It is land-only (ocean cells are the fill value
-9999), gauge-blended, and available 1981-present at exactly 0.05 deg -- which
is why it, rather than ERA5-Land (9 km) or IMERG (0.1 deg), is the truth here.

    python scripts/01_download_chirps.py --start 1981 --end 2025 --out data/raw/chirps

The global daily files are ~1 GB/yr; we subset immediately after download and
delete the global file unless ``--keep-global`` is passed.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bdhires.grids import WIDE  # noqa: E402

BASE = (
    "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p05/"
    "chirps-v2.0.{year}.days_p05.nc"
)
# CHIRPS v3 (0.05 deg, 1981-present) lives under products/CHIRPS/v3.0/ ;
# switch BASE when you are ready to move -- v3 improves the satellite
# calibration over South Asia but changes the climatology, so do not mix.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1981)
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument("--out", default="data/raw/chirps")
    ap.add_argument("--keep-global", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lo, la, hi, ha = WIDE.bbox

    for year in range(args.start, args.end + 1):
        sub = out / f"chirps_wide_{year}.nc"
        if sub.exists():
            continue
        url = BASE.format(year=year)
        glob_f = out / Path(url).name
        if not glob_f.exists():
            print("downloading", url, flush=True)
            subprocess.run(["wget", "-q", "--show-progress", "-O", str(glob_f), url], check=True)
        with xr.open_dataset(glob_f) as ds:
            ds = ds.sel(longitude=slice(lo, hi), latitude=slice(la, ha))
            ds["precip"] = ds["precip"].where(ds["precip"] > -100)  # -9999 -> NaN
            ds = ds.sortby("latitude")  # enforce ascending latitude convention
            ds.to_netcdf(sub, encoding={"precip": {"zlib": True, "complevel": 4}})
        print("wrote", sub, ds.sizes, flush=True)
        if not args.keep_global:
            glob_f.unlink()


if __name__ == "__main__":
    main()
