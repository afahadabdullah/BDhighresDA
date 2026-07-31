#!/usr/bin/env python
"""Download the CPC Global Unified Gauge-Based Analysis of Daily Precipitation.

0.5 degree, daily, land only, 1979-present, from NOAA PSL.  Produced by optimal
interpolation of GTS gauge reports, it is a contemporaneous analysis rather than
a forecast.  It can therefore condition a CPC-to-CHIRPS statistical-downscaling
experiment, but that experiment must not be interpreted as forecast downscaling.

One netCDF per year, ~50 MB each, so the whole 1981-2025 record is a few GB.

    python scripts/02b_download_cpc.py --start 1981 --end 2025 --out data/raw/cpc
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

BASE = "https://downloads.psl.noaa.gov/Datasets/cpc_global_precip"
FILENAME = "precip.{year}.nc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1981)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--out", default="data/raw/cpc")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="re-download files that are already present",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="exit non-zero if any requested year fails or is absent",
    )
    args = parser.parse_args()
    if args.start > args.end:
        parser.error("--start must not exceed --end")
    if args.start < 1979:
        parser.error("CPC daily starts in 1979")
    return args


def download(url: str, destination: Path) -> None:
    """Fetch to a .part file and rename, so an interrupted run leaves no stub."""
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            total = int(response.headers.get("Content-Length", 0))
            written = 0
            with partial.open("wb") as handle:
                while chunk := response.read(1 << 20):
                    handle.write(chunk)
                    written += len(chunk)
            if total and written != total:
                raise IOError(
                    f"{url}: expected {total} bytes, received {written}"
                )
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    partial.replace(destination)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    failures = []
    for year in range(args.start, args.end + 1):
        name = FILENAME.format(year=year)
        destination = out_dir / name
        if destination.is_file() and not args.overwrite:
            print(f"  {name}: already present ({destination.stat().st_size/2**20:.0f} MiB)")
            continue
        url = f"{BASE}/{name}"
        try:
            download(url, destination)
            print(f"  {name}: {destination.stat().st_size/2**20:.0f} MiB", flush=True)
        except Exception as exc:                      # noqa: BLE001
            # The current year is often incomplete or absent; do not abort the
            # whole record for it.
            print(f"  {name}: FAILED ({exc})", flush=True)
            failures.append(year)

    present = sorted(int(p.stem.split(".")[1]) for p in out_dir.glob("precip.*.nc"))
    print(f"\n{len(present)} year(s) on disk: {present[0]}-{present[-1]}"
          if present else "\nno files downloaded")
    if failures:
        print(f"failed: {failures}")
    requested = set(range(args.start, args.end + 1))
    absent = sorted(requested - set(present))
    if args.require_complete and absent:
        print(f"required years absent: {absent}")
        sys.exit(1)
    if len(failures) == args.end - args.start + 1:
        sys.exit(1)


if __name__ == "__main__":
    main()
