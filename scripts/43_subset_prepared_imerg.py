#!/usr/bin/env python
"""Slice an already-prepared IMERG file to an exact date window.

``scripts/15_bmd_month_example.py`` compares the IMERG time axis to the
checkpoint dates with ``np.array_equal``, so a prepared file must span the
requested window and nothing else. The monthly archive in ``data/processed``
(``imerg_bd_aligned_YYYYMM01_YYYYMMDD.nc``) therefore cannot serve a ten-day
experiment even though it contains exactly the right days.

Re-running ``scripts/08_prepare_imerg_observations.py`` would rebuild those days
from the half-hourly granules -- an hour of work, a dependency on granules that
may since have been cleaned up, and a chance to get a flag wrong. The monthly
files were written by ``slurm/download_imerg_halfhourly_2021_2024.sbatch`` with
``--source-frequency half-hourly --min-count 48 --accumulation-end-hour-utc 3``,
which is precisely what a rebuild would use, so the accumulation has already
been done correctly and only needs cutting.

What this script guarantees, because ``scripts/15`` checks all of it:

* every requested day is present -- a missing day is a hard error, never a
  silently shorter file, since a short file fails much later and less legibly;
* ``bmd_accumulation_end_hour_utc``, ``source_frequency`` and the ``mm/day``
  units survive the round trip, because they are the attributes the
  assimilation validates before it will read the data;
* the window is genuinely accumulated to 03:00 UTC -- a calendar-day file is
  rejected here rather than assimilated as though it were a BMD day.

Windows spanning a month boundary are handled by passing several monthly files;
they are concatenated on time and de-duplicated before the cut.

Example
-------
    python scripts/43_subset_prepared_imerg.py \\
        --input data/processed/imerg_bd_aligned_20220501_20220531.nc \\
        --start 2022-05-01 --end 2022-05-10 \\
        --out data/processed/imerg_prepared_ing2022/imerg_aligned_20220501_20220510.nc \\
        --report data/processed/imerg_prepared_ing2022/imerg_aligned_20220501_20220510_qc.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import xarray as xr

REQUIRED_VARS = ("precipitation", "randomError", "precipitation_cnt")
MM_PER_DAY = {"mm/day", "mmday-1", "mmd-1", "mmday^-1", "mmd^-1"}
BMD_END_HOUR = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input", nargs="+", required=True,
        help="Prepared IMERG files or globs. Files that do not overlap the "
             "window are skipped without being read, so the whole archive "
             "glob is a valid argument; overlapping ones are concatenated, "
             "which is what a window crossing a month needs.",
    )
    parser.add_argument("--start", required=True, help="First BMD day, YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="Last BMD day, inclusive")
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", default=None,
                        help="QC sidecar. Defaults to <out>_qc.json.")
    parser.add_argument(
        "--allow-day-shift", action="store_true",
        help="Accept a daily product already shifted onto the BMD convention "
             "(bmd_window_alignment='day-shift'), such as the CPC pseudo-"
             "satellite. Off by default: a whole-day shift approximates the "
             "3-hour offset, it does not reproduce it.",
    )
    return parser.parse_args()


def resolve_inputs(patterns: list[str]) -> list[Path]:
    """Expand globs, keep order stable, and fail on a pattern matching nothing.

    A glob that silently matches nothing is how a 'subset' quietly becomes an
    empty file, so an unmatched pattern is an error rather than a skip.
    """
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern)) or ([pattern] if Path(pattern).exists() else [])
        if not matches:
            raise FileNotFoundError(f"no prepared IMERG file matches {pattern!r}")
        for match in matches:
            path = Path(match)
            if path not in paths:
                paths.append(path)
    return paths


def validate_source(dataset: xr.Dataset, path: Path, allow_day_shift: bool) -> None:
    """Reject anything ``scripts/15`` would reject, here where the error is cheap."""
    missing = [v for v in REQUIRED_VARS if v not in dataset]
    if missing:
        raise ValueError(f"{path} lacks required IMERG variables {missing}")

    for variable in ("precipitation", "randomError"):
        units = str(dataset[variable].attrs.get("units", ""))
        if units.lower().replace(" ", "") not in MM_PER_DAY:
            raise ValueError(f"{path} {variable} units are {units!r}; expected mm/day")

    end_hour = dataset.attrs.get("bmd_accumulation_end_hour_utc")
    try:
        end_hour = int(end_hour)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{path} does not declare bmd_accumulation_end_hour_utc, so it was not "
            "written by scripts/08 and cannot be assumed to be BMD-aligned"
        ) from exc
    if end_hour != BMD_END_HOUR:
        raise ValueError(
            f"{path} ends its accumulation at {end_hour:02d}:00 UTC; BMD daily "
            "rainfall ends at 03:00 UTC, and subsetting cannot fix that"
        )

    source_frequency = str(dataset.attrs.get("source_frequency", ""))
    alignment = str(dataset.attrs.get("bmd_window_alignment", ""))
    if source_frequency != "half-hourly":
        if alignment != "day-shift":
            raise ValueError(
                f"{path} was not prepared from half-hourly IMERG and does not "
                "declare bmd_window_alignment='day-shift'; it cannot represent "
                "the BMD 03:00-03:00 UTC window"
            )
        if not allow_day_shift:
            raise ValueError(
                f"{path} is a day-shifted daily product. Pass --allow-day-shift "
                "if that is intended; its observation errors are optimistic."
            )


def load_concatenated(paths: list[Path], start: str, end: str,
                      allow_day_shift: bool) -> tuple[xr.Dataset, list[Path]]:
    """Open the sources that overlap the window, validate them, concatenate.

    Files are screened on their time coordinate BEFORE their data is read, so
    that ``--input 'data/processed/imerg_bd_aligned_*.nc'`` can be pointed at the
    whole multi-year archive without pulling every month into memory. Screening
    on filename would be faster still and is exactly the kind of shortcut that
    breaks the first time a file is renamed.
    """
    first = np.datetime64(start, "D")
    last = np.datetime64(end, "D")
    pieces, used = [], []
    for path in paths:
        dataset = xr.open_dataset(path)
        days = np.asarray(dataset.time.values).astype("datetime64[D]")
        if days.size == 0 or days[-1] < first or days[0] > last:
            dataset.close()
            continue
        validate_source(dataset, path, allow_day_shift)
        pieces.append(dataset.load())
        dataset.close()
        used.append(path)

    if not pieces:
        raise ValueError(
            f"none of the {len(paths)} prepared file(s) contain any day in "
            f"{start}..{end}"
        )

    if len(pieces) == 1:
        combined = pieces[0]
    else:
        combined = xr.concat(pieces, dim="time", combine_attrs="override")
        combined = combined.sortby("time")
        # Duplicated days appear when monthly files overlap; keeping the first
        # is safe only because they were built identically, which validate_source
        # has just confirmed.
        days = np.asarray(combined.time.values).astype("datetime64[D]")
        _, keep = np.unique(days, return_index=True)
        combined = combined.isel(time=np.sort(keep))
    return combined, used


def cut_window(dataset: xr.Dataset, start: str, end: str) -> xr.Dataset:
    """Select exactly the requested days, or say precisely which are missing."""
    wanted = np.arange(np.datetime64(start, "D"),
                       np.datetime64(end, "D") + np.timedelta64(1, "D"))
    available = np.asarray(dataset.time.values).astype("datetime64[D]")

    missing = np.setdiff1d(wanted, available)
    if missing.size:
        raise ValueError(
            f"{missing.size} of {wanted.size} requested day(s) are absent from the "
            f"source: {', '.join(str(d) for d in missing[:10])}"
            + (" ..." if missing.size > 10 else "")
            + f". The source spans {available[0]}..{available[-1]}. Prepare the "
            "missing days with scripts/08_prepare_imerg_observations.py."
        )

    index = np.searchsorted(available, wanted)
    subset = dataset.isel(time=index)
    # Write the day, not the sub-daily timestamp: scripts/15 compares dates, and
    # a stray hour here would survive the comparison but confuse every plot.
    subset = subset.assign_coords(time=wanted)
    return subset


def build_report(subset: xr.Dataset, sources: list[Path], start: str, end: str) -> dict:
    precipitation = subset["precipitation"].values
    valid = np.isfinite(precipitation)
    daily_valid = valid.sum(axis=(1, 2))
    error = subset["randomError"].values
    return {
        "derivation": "subset of prepared IMERG; no re-accumulation was performed",
        "product": subset.attrs.get("product"),
        "version": subset.attrs.get("version"),
        "source_frequency": subset.attrs.get("source_frequency"),
        "period": {"start": start, "end": end, "days": int(subset.sizes["time"])},
        "accumulation": {
            "end_hour_utc": int(subset.attrs.get("bmd_accumulation_end_hour_utc")),
            "window": subset.attrs.get("accumulation_window"),
            "duration_hours": int(subset.attrs.get("window_duration_hours", 24)),
            "time_coordinate": "BMD archive date and window end",
        },
        "source_files": [str(p) for p in sources],
        "grid": {
            "shape": list(precipitation.shape[1:]),
            "lat_range_centres": [float(subset.lat.values[0]), float(subset.lat.values[-1])],
            "lon_range_centres": [float(subset.lon.values[0]), float(subset.lon.values[-1])],
        },
        "quality_control": {
            "possible_footprints": int(valid.size),
            "valid_footprints": int(valid.sum()),
            "valid_fraction": float(valid.mean()),
            "daily_valid_min": int(daily_valid.min()),
            "daily_valid_median": float(np.median(daily_valid)),
            "daily_valid_max": int(daily_valid.max()),
        },
        "precipitation_mm_day": {
            "mean": float(np.nanmean(precipitation)),
            "p99": float(np.nanpercentile(precipitation, 99)),
            "max": float(np.nanmax(precipitation)),
        },
        "random_error_mm_day": {
            "median": float(np.nanmedian(error)),
            "p90": float(np.nanpercentile(error, 90)),
            "max": float(np.nanmax(error)),
        },
        "warning": (
            "Native V07B values with no fitted bias correction. IMERG carries a "
            "measured +5.56 mm/day bias at BMD gauges; a Gaussian likelihood "
            "assumes unbiasedness, so the satellite arm is expected to be biased."
        ),
    }


def main() -> None:
    args = parse_args()
    candidates = resolve_inputs(args.input)

    combined, sources = load_concatenated(candidates, args.start, args.end,
                                          args.allow_day_shift)
    subset = cut_window(combined, args.start, args.end)

    subset.attrs = dict(combined.attrs)
    subset.attrs.update({
        "subset_of": "; ".join(str(p) for p in sources),
        "subset_window": f"{args.start}..{args.end}",
        "subset_tool": "scripts/43_subset_prepared_imerg.py",
        "derivation": "temporal subset only; accumulation inherited unchanged",
    })

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    subset.to_netcdf(temporary)
    temporary.replace(output)

    report_path = Path(args.report) if args.report else \
        output.with_name(output.stem + "_qc.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(build_report(subset, sources, args.start,
                                                   args.end), indent=2) + "\n")

    days = np.asarray(subset.time.values).astype("datetime64[D]")
    print(f"[subset] {days.size} BMD windows {days[0]}..{days[-1]} "
          f"from {len(sources)} prepared file(s)")
    print(f"wrote {output}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
