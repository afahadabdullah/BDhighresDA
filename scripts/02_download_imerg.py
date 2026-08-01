#!/usr/bin/env python
"""Download GPM IMERG Final Run daily precipitation (0.1 deg) from GES DISC.

IMERG is an observation of where and how much it rained at 0.1 degrees.  It is
kept outside the training Zarr and enters the inference-time likelihood through
the physical block-average observation operator; it is not a conditioning
channel.

Availability: 2000-06-01 onwards (V07 Final). The prior can still train before
the satellite era because IMERG is used only at inference.

Auth: needs an Earthdata login and a ~/.netrc entry:
    machine urs.earthdata.nasa.gov login <user> password <pass>

    python scripts/02_download_imerg.py --start 2000 --end 2025 --out data/raw/imerg
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

BASE = "https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGDF.07"
FNAME = "3B-DAY.MS.MRG.3IMERG.{ymd}-S000000-E235959.V07B.nc4"


def daterange(a: date, b: date):
    d = a
    while d <= b:
        yield d
        d += timedelta(days=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2000)
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument("--out", default="data/raw/imerg")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    a = max(date(args.start, 1, 1), date(2000, 6, 1))
    b = date(args.end, 12, 31)

    urls = []
    for d in daterange(a, b):
        fn = FNAME.format(ymd=d.strftime("%Y%m%d"))
        if (out / fn).exists():
            continue
        urls.append(f"{BASE}/{d.year}/{d.month:02d}/{fn}")

    if not urls:
        print("nothing to download")
        return
    listfile = out / "_urls.txt"
    listfile.write_text("\n".join(urls))
    print(f"{len(urls)} files to fetch")

    # wget handles the Earthdata OAuth redirect chain given ~/.netrc + a cookie jar
    subprocess.run(
        ["xargs", "-a", str(listfile), "-n", "1", "-P", str(args.jobs),
         "wget", "--load-cookies", str(Path.home() / ".urs_cookies"),
         "--save-cookies", str(Path.home() / ".urs_cookies"),
         "--keep-session-cookies", "--auth-no-challenge=on", "--content-disposition",
         "-q", "-nc", "-P", str(out)],
        check=True,
    )
    listfile.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
