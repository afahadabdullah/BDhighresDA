#!/usr/bin/env python3
"""Extract regional daily ERA5 predictors from Earthmover's public ARCO store.

The source is the free, quarterly updated Icechunk v2 / Zarr v3 repository in
the AWS Open Data Registry.  No CDS or AWS credentials are required.

    python scripts/00_download_era5.py \
        --start 1981 --end 2025 --out data/raw/era5

Each output is a compact annual NetCDF file at native ERA5 0.25-degree
resolution, cropped to the WIDE domain plus a one-degree interpolation halo.
Hourly source data are aggregated before they are saved:

* ``tp`` is summed over 01:00(D) through 00:00(D+1), then converted m -> mm.
* ``tcwv``, ``cape``, ``u10``, ``v10`` and ``msl`` are averaged over
  00:00 through 23:00 UTC on day D.

Icechunk v2 requires Python 3.12 or newer.  On Prism, use the dedicated
``bdda-earthmover`` environment described in the README.
"""
from __future__ import annotations

import argparse
import calendar
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bdhires.grids import WIDE  # noqa: E402

BUCKET = "earthmover-icechunk-era5"
PREFIX = "icechunkV2"
REGION = "us-east-1"
GROUP = "single/temporal"
BRANCH = "main"
PAD = 1.0

# Earthmover surface-only replacement for the former CDS five-channel set.
# The wide spatial fields of u10/v10 provide flow direction; msl supplies the
# synoptic circulation pattern.  ERA5 tp already embeds the model's full
# dynamical and moisture-convergence calculation.
VARIABLES = ("tp", "tcwv", "cape", "u10", "v10", "msl")
STATE_VARIABLES = tuple(variable for variable in VARIABLES if variable != "tp")


def bounds() -> tuple[float, float, float, float]:
    """Return regional bounds as north, west, south, east."""
    west, south, east, north = WIDE.bbox
    return north + PAD, west - PAD, south - PAD, east + PAD


def expected_days(year: int) -> int:
    return 366 if calendar.isleap(year) else 365


def validate_year(path: Path, year: int) -> None:
    """Raise if *path* is not a complete daily regional ERA5 year."""
    with xr.open_dataset(path) as dataset:
        missing = set(VARIABLES) - set(dataset.data_vars)
        extra = set(dataset.data_vars) - set(VARIABLES)
        if missing or extra:
            raise ValueError(
                f"{path} variable mismatch; missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )

        required = {"time", "latitude", "longitude"}
        absent = required - (set(dataset.coords) | set(dataset.dims))
        if absent:
            raise ValueError(f"{path} is missing coordinates {sorted(absent)}")

        days = pd.DatetimeIndex(dataset.time.values)
        expected = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
        if not days.equals(expected):
            raise ValueError(
                f"{path} has an incomplete time axis: "
                f"{days[0] if len(days) else 'empty'} to "
                f"{days[-1] if len(days) else 'empty'}, {len(days)} values"
            )

        north, west, south, east = bounds()
        lat_min = float(dataset.latitude.min())
        lat_max = float(dataset.latitude.max())
        lon_min = float(dataset.longitude.min())
        lon_max = float(dataset.longitude.max())
        resolution = 0.25
        if (
            lat_min > south + 1e-6
            or lat_max < north - resolution - 1e-6
            or lon_min > west + 1e-6
            or lon_max < east - resolution - 1e-6
        ):
            raise ValueError(
                f"{path} does not cover the requested halo: "
                f"lat={lat_min}..{lat_max}, lon={lon_min}..{lon_max}"
            )

        # Reading the first and last day catches truncated/corrupt NetCDF
        # payloads without loading the full annual file a second time.
        edge = dataset[list(VARIABLES)].isel(time=[0, -1]).load()
        for variable in VARIABLES:
            if not np.isfinite(edge[variable].values).any():
                raise ValueError(f"{path} has no finite edge values for {variable}")


def aggregate_year(source: xr.Dataset, year: int) -> xr.Dataset:
    """Return one correctly aligned daily year from an hourly source dataset."""
    start = pd.Timestamp(year=year, month=1, day=1)
    stop = pd.Timestamp(year=year + 1, month=1, day=1)
    hourly_expected = pd.date_range(start, stop, freq="h")

    north, west, south, east = bounds()
    window = source[list(VARIABLES)].sel(
        valid_time=slice(start, stop),
        latitude=slice(north, south),  # Earthmover latitude is descending
        longitude=slice(west, east),
    )

    hourly_actual = pd.DatetimeIndex(window.valid_time.values)
    if not hourly_actual.equals(hourly_expected):
        raise ValueError(
            f"Earthmover has an incomplete hourly window for {year}: "
            f"expected {len(hourly_expected)}, found {len(hourly_actual)}"
        )
    if window.sizes["latitude"] == 0 or window.sizes["longitude"] == 0:
        raise ValueError(f"Earthmover regional selection is empty for {year}")

    # State variables use 00:00..23:00 on D.  The inclusive selection includes
    # 00:00 on Jan 1 of the following year, so drop its final sample.
    states = window[list(STATE_VARIABLES)].isel(valid_time=slice(0, -1))
    states_daily = states.resample(valid_time="1D").mean(keep_attrs=True)

    # ERA5 tp at time H is the accumulation over the preceding hour.  Drop
    # 00:00 on Jan 1, retain 00:00 on Jan 1 of the next year, and shift the
    # labels back one hour before summing.
    precipitation = window["tp"].isel(valid_time=slice(1, None))
    precipitation = precipitation.assign_coords(
        valid_time=precipitation.valid_time - np.timedelta64(1, "h")
    )
    precipitation_daily = precipitation.resample(valid_time="1D").sum(
        keep_attrs=True
    )
    precipitation_daily = (precipitation_daily * 1000.0).astype("float32")
    precipitation_daily.attrs.update(
        units="mm day-1",
        long_name="ERA5 total precipitation, 00:00-24:00 UTC daily total",
        aggregation="sum of preceding-hour accumulations 01:00(D)-00:00(D+1)",
    )

    daily = xr.merge(
        [precipitation_daily.to_dataset(name="tp"), states_daily],
        compat="no_conflicts",
        combine_attrs="drop_conflicts",
    ).astype("float32")
    daily = daily.rename(valid_time="time")
    daily = daily.transpose("time", "latitude", "longitude")
    daily.attrs.update(
        title="Daily regional ERA5 predictors for BDhighresDA",
        source=(
            "Earthmover public ERA5 ARCO Icechunk store "
            "s3://earthmover-icechunk-era5/icechunkV2"
        ),
        source_group=GROUP,
        source_branch=BRANCH,
        source_resolution="0.25 degree hourly",
        temporal_resolution="daily",
        state_aggregation="mean of 00:00-23:00 UTC",
        spatial_subset=(
            f"north={north}, west={west}, south={south}, east={east}"
        ),
        license="CC-BY-4.0",
    )

    if daily.sizes["time"] != expected_days(year):
        raise ValueError(
            f"daily aggregation produced {daily.sizes['time']} days for {year}"
        )
    return daily


def open_earthmover() -> xr.Dataset:
    """Open the public temporal-layout ERA5 group with native Dask chunks."""
    if sys.version_info < (3, 12):
        raise RuntimeError(
            "Earthmover's Icechunk v2 store requires Python 3.12 or newer. "
            "Create the dedicated environment from environment-earthmover.yml."
        )

    try:
        import dask  # noqa: F401
        import icechunk
        import pcodec  # noqa: F401  # registers the Zarr PCodec decoder
    except ImportError as exc:
        raise RuntimeError(
            "Missing Earthmover dependency. Create the dedicated environment "
            "from environment-earthmover.yml."
        ) from exc

    storage = icechunk.s3_storage(
        bucket=BUCKET,
        prefix=PREFIX,
        region=REGION,
        anonymous=True,
    )
    repository = icechunk.Repository.open(storage)
    session = repository.readonly_session(branch=BRANCH)
    dataset = xr.open_zarr(
        session.store,
        group=GROUP,
        consolidated=False,
        chunks={},
    )

    missing = set(VARIABLES) - set(dataset.data_vars)
    if missing:
        raise ValueError(
            f"Earthmover group {GROUP} is missing variables {sorted(missing)}"
        )
    return dataset


def write_year(source: xr.Dataset, year: int, out: Path, workers: int) -> None:
    """Aggregate, write and validate one annual file atomically."""
    target = out / f"era5_daily_{year}.nc"
    partial = target.with_suffix(target.suffix + ".part")

    if target.exists():
        try:
            validate_year(target, year)
            print("already complete", target, flush=True)
            return
        except (OSError, ValueError) as exc:
            print(f"removing invalid output {target}: {exc}", flush=True)
            target.unlink()

    if partial.exists():
        try:
            validate_year(partial, year)
            partial.replace(target)
            print("recovered", target, flush=True)
            return
        except (OSError, ValueError):
            partial.unlink()

    print(f"aggregating Earthmover ERA5 for {year}", flush=True)
    daily = aggregate_year(source, year)
    nlat = daily.sizes["latitude"]
    nlon = daily.sizes["longitude"]
    encoding = {
        variable: {
            "dtype": "float32",
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
            "chunksizes": (31, nlat, nlon),
        }
        for variable in VARIABLES
    }

    import dask
    from dask.diagnostics import ProgressBar

    with dask.config.set(scheduler="threads", num_workers=workers):
        with ProgressBar():
            daily.to_netcdf(
                partial,
                engine="netcdf4",
                encoding=encoding,
                compute=True,
            )
    daily.close()

    validate_year(partial, year)
    partial.replace(target)
    print("wrote", target, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1981)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--out", default="data/raw/era5")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.start > args.end:
        parser.error("--start must be less than or equal to --end")
    if args.workers < 1:
        parser.error("--workers must be positive")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print("source:", f"s3://{BUCKET}/{PREFIX}", GROUP)
        print("years:", args.start, "-", args.end)
        print("variables:", ", ".join(VARIABLES))
        print("bounds (N,W,S,E):", bounds())
        print("output:", out / "era5_daily_YYYY.nc")
        return

    source = open_earthmover()
    try:
        print(
            f"opened Earthmover {GROUP}: "
            f"{source.valid_time.values[0]} to {source.valid_time.values[-1]}",
            flush=True,
        )
        for year in range(args.start, args.end + 1):
            write_year(source, year, out, args.workers)
    finally:
        source.close()


if __name__ == "__main__":
    main()
