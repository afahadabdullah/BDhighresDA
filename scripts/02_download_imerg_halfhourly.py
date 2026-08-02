#!/usr/bin/env python3
"""Download regional IMERG V07B half-hourly subsets for BMD reporting days.

The BMD archive date labels the end of a 24-hour window at 03:00 UTC. Thus
May 2018 BMD data require IMERG intervals beginning 2018-04-30 03:00 and ending
with the interval beginning 2018-05-31 02:30. Explicit ``wget -O`` paths avoid
the long query-string filenames produced by the GES DISC subset service.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from urllib.parse import urlencode


SERVICE = "https://gpm1.gesdisc.eosdis.nasa.gov/daac-bin/OTF/HTTP_services.cgi"
SHORTNAME = "GPM_3IMERGHH"
DATASET = "GPM_3IMERGHH.07"
VERSION = "V07B"
PRINT_LOCK = threading.Lock()


@dataclass(frozen=True)
class Request:
    start: datetime
    source_name: str
    output_name: str
    url: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bmd-start", default="2018-05-01")
    parser.add_argument("--bmd-end", default="2018-05-31")
    parser.add_argument("--end-hour-utc", type=int, default=3)
    parser.add_argument("--out", default="data/imerg_halfhourly")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument(
        "--bbox",
        default="20.3,87.6,26.7,94.0",
        help="GES DISC south,west,north,east subset; default covers the BD grid",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def bmd_interval_starts(
    first_day: date, last_day: date, end_hour_utc: int
) -> list[datetime]:
    """Return unique half-hour starts spanning inclusive BMD archive dates."""
    if last_day < first_day:
        raise ValueError("--bmd-end precedes --bmd-start")
    if not 0 <= end_hour_utc <= 23:
        raise ValueError("--end-hour-utc must be between 0 and 23")
    first_end = datetime.combine(first_day, time(hour=end_hour_utc))
    final_end = datetime.combine(last_day, time(hour=end_hour_utc))
    start = first_end - timedelta(days=1)
    intervals = int((final_end - start) / timedelta(minutes=30))
    expected = ((last_day - first_day).days + 1) * 48
    if intervals != expected:
        raise AssertionError(f"generated {intervals} intervals; expected {expected}")
    return [start + timedelta(minutes=30 * index) for index in range(intervals)]


def request_for(interval_start: datetime, bbox: str) -> Request:
    interval_end = interval_start + timedelta(minutes=29, seconds=59)
    minute_of_day = interval_start.hour * 60 + interval_start.minute
    source_name = (
        f"3B-HHR.MS.MRG.3IMERG.{interval_start:%Y%m%d}-"
        f"S{interval_start:%H%M%S}-E{interval_end:%H%M%S}."
        f"{minute_of_day:04d}.{VERSION}.HDF5"
    )
    output_name = source_name + ".SUB.nc4"
    day_of_year = interval_start.timetuple().tm_yday
    source_path = (
        f"/data/GPM_L3/{DATASET}/{interval_start:%Y}/{day_of_year:03d}/{source_name}"
    )
    parameters = {
        "FILENAME": source_path,
        "VARIABLES": "precipitation,randomError",
        "SERVICE": "L34RS_GPM",
        "BBOX": bbox,
        "FORMAT": "bmM0Lw",
        "VERSION": "1.02",
        "DATASET_VERSION": "07",
        "SHORTNAME": SHORTNAME,
        "LABEL": output_name,
    }
    return Request(
        start=interval_start,
        source_name=source_name,
        output_name=output_name,
        url=SERVICE + "?" + urlencode(parameters),
    )


def valid_netcdf(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 512:
        return False
    with path.open("rb") as handle:
        magic = handle.read(8)
    return magic.startswith(b"CDF") or magic == b"\x89HDF\r\n\x1a\n"


def download_one(request: Request, output: Path, cookie_file: Path) -> str:
    destination = output / request.output_name
    if valid_netcdf(destination):
        return "skipped"
    partial = destination.with_suffix(destination.suffix + ".part")
    partial.unlink(missing_ok=True)
    command = [
        "wget",
        "--auth-no-challenge=on",
        "--tries=5",
        "--timeout=120",
        "--waitretry=5",
        "--retry-connrefused",
        "-q",
        "-O",
        str(partial),
        request.url,
    ]
    if cookie_file.is_file():
        command[1:1] = ["--load-cookies", str(cookie_file)]
    try:
        subprocess.run(command, check=True)
        if not valid_netcdf(partial):
            preview = partial.read_bytes()[:160].decode("utf-8", errors="replace")
            raise RuntimeError(
                f"GES DISC returned a non-NetCDF response for {request.output_name}: "
                f"{preview!r}"
            )
        partial.replace(destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return "downloaded"


def main() -> None:
    args = parse_args()
    first_day = date.fromisoformat(args.bmd_start)
    last_day = date.fromisoformat(args.bmd_end)
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    bbox_values = [float(value) for value in args.bbox.split(",")]
    if len(bbox_values) != 4:
        raise ValueError("--bbox must contain south,west,north,east")
    south, west, north, east = bbox_values
    if not south < north or not west < east:
        raise ValueError("--bbox bounds are not ordered south,west,north,east")
    bbox = ",".join(f"{value:g}" for value in bbox_values)

    intervals = bmd_interval_starts(first_day, last_day, args.end_hour_utc)
    requests = [request_for(value, bbox) for value in intervals]
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / (
        f"_urls_bmd_{first_day:%Y%m%d}_{last_day:%Y%m%d}_end{args.end_hour_utc:02d}utc.txt"
    )
    manifest.write_text("\n".join(request.url for request in requests) + "\n")

    print(f"BMD dates: {first_day} through {last_day}")
    print(
        f"IMERG interval starts: {intervals[0]:%Y-%m-%d %H:%M} through "
        f"{intervals[-1]:%Y-%m-%d %H:%M} UTC"
    )
    print(f"Granules required: {len(requests)}")
    print(f"Subset bbox south,west,north,east: {bbox}")
    print(f"First output: {requests[0].output_name}")
    print(f"Last output:  {requests[-1].output_name}")
    print(f"URL manifest: {manifest}")
    if args.dry_run:
        print("dry run: no downloads requested")
        return

    cookie_file = Path.home() / ".urs_cookies"
    completed = 0
    counts = {"downloaded": 0, "skipped": 0}

    def worker(request: Request) -> str:
        result = download_one(request, output, cookie_file)
        nonlocal completed
        with PRINT_LOCK:
            completed += 1
            counts[result] += 1
            if completed == 1 or completed % 25 == 0 or completed == len(requests):
                print(
                    f"[{completed:04d}/{len(requests)}] {result}: {request.output_name}",
                    flush=True,
                )
        return result

    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(worker, request): request for request in requests}
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                request = futures[future]
                errors.append(f"{request.output_name}: {exc}")
                with PRINT_LOCK:
                    print(f"ERROR {errors[-1]}", file=sys.stderr, flush=True)
    if errors:
        shown = "\n".join(errors[:20])
        raise RuntimeError(
            f"{len(errors)} IMERG downloads failed; rerun safely to resume:\n{shown}"
        )
    print(
        f"complete: {counts['downloaded']} downloaded, {counts['skipped']} already valid, "
        f"{len(requests)} total"
    )


if __name__ == "__main__":
    main()
