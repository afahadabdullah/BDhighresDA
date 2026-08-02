"""Strict ingestion of GPM IMERG Final V07 observations.

IMERG is an observation stream, not a model-conditioning channel.  This
module reads either the original calendar-day granules or half-hourly
granules independently of the checkpoint-bound training Zarr.  Half-hourly
data can be accumulated over the BMD reporting window, which ends at 03:00
UTC rather than at calendar-day midnight.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

from .grids import BD, Grid


_DAILY_NAME = re.compile(
    r"^3B-DAY\.MS\.MRG\.3IMERG\.(?P<date>\d{8})-S000000-E235959\.V07B\.nc4$"
)
_HALF_HOURLY_NAME = re.compile(
    r"^3B-HHR\.MS\.MRG\.3IMERG\.(?P<date>\d{8})-"
    r"S(?P<start>\d{6})-E(?P<end>\d{6})\.\d{4}\.V07B\.HDF5"
    r"(?:\.SUB\.nc4)?$"
)


@dataclass(frozen=True)
class ImergDaily:
    """Regional daily IMERG observations and their native uncertainty."""

    time: np.ndarray
    precipitation: np.ndarray
    random_error: np.ndarray
    count: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    source_files: tuple[str, ...]
    min_count: int
    source_frequency: str = "daily"
    accumulation_end_hour_utc: int = 0
    random_error_aggregation: str = "native daily product"


def _dates(start: str | np.datetime64, end: str | np.datetime64) -> list[date]:
    first = date.fromisoformat(str(np.datetime64(start, "D")))
    last = date.fromisoformat(str(np.datetime64(end, "D")))
    if last < first:
        raise ValueError(f"end date {last} precedes start date {first}")
    return [first + timedelta(days=offset) for offset in range((last - first).days + 1)]


def discover_imerg_files(
    directory: str | Path,
    start: str | np.datetime64,
    end: str | np.datetime64,
) -> list[Path]:
    """Return one exact V07B daily granule per requested date.

    A missing or duplicate day is fatal.  Silently assimilating an incomplete
    month would make the gauge/IMERG comparison scientifically ambiguous.
    """

    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"IMERG directory does not exist: {directory}")
    by_date: dict[date, list[Path]] = {}
    for path in sorted(directory.glob("3B-DAY.MS.MRG.3IMERG.*.V07B.nc4")):
        match = _DAILY_NAME.match(path.name)
        if match:
            key = date.fromisoformat(
                f"{match['date'][:4]}-{match['date'][4:6]}-{match['date'][6:]}"
            )
            by_date.setdefault(key, []).append(path)

    output: list[Path] = []
    missing: list[str] = []
    duplicate: list[str] = []
    for day in _dates(start, end):
        paths = by_date.get(day, [])
        if not paths:
            missing.append(day.isoformat())
        elif len(paths) > 1:
            duplicate.append(f"{day.isoformat()}: {[p.name for p in paths]}")
        else:
            output.append(paths[0])
    if missing or duplicate:
        parts = []
        if missing:
            parts.append("missing dates: " + ", ".join(missing))
        if duplicate:
            parts.append("duplicate dates: " + "; ".join(duplicate))
        raise ValueError("invalid IMERG daily inventory; " + " | ".join(parts))
    return output


def _half_hour_start(path: Path) -> datetime | None:
    """Return the UTC interval start encoded in an IMERG half-hourly name."""

    match = _HALF_HOURLY_NAME.match(path.name)
    if match is None:
        return None
    start = datetime.strptime(match["date"] + match["start"], "%Y%m%d%H%M%S")
    encoded_end = datetime.strptime(match["date"] + match["end"], "%Y%m%d%H%M%S")
    expected_end = start + timedelta(minutes=29, seconds=59)
    if encoded_end != expected_end:
        raise ValueError(
            f"{path.name} does not encode one 30-minute interval: "
            f"{start} to {encoded_end}"
        )
    if start.minute not in (0, 30) or start.second:
        raise ValueError(f"{path.name} is not aligned to a UTC half hour")
    return start


def _bmd_window_bounds(day: date, end_hour_utc: int) -> tuple[datetime, datetime]:
    if not 0 <= end_hour_utc <= 23:
        raise ValueError("accumulation end hour must be between 0 and 23 UTC")
    end = datetime.combine(day, time(hour=end_hour_utc))
    return end - timedelta(days=1), end


def discover_imerg_half_hourly_files(
    directory: str | Path,
    start: str | np.datetime64,
    end: str | np.datetime64,
    *,
    accumulation_end_hour_utc: int = 3,
) -> list[Path]:
    """Return the exact 48 half-hour granules for every requested BMD date.

    The selected BMD archive date labels the *end* of its 24-hour reporting
    window.  For example, 2018-05-01 requires IMERG intervals beginning at
    2018-04-30 03:00 through 2018-05-01 02:30 UTC.  Missing or duplicate
    intervals are fatal so a partial Earthdata download cannot silently
    produce a low daily accumulation.
    """

    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"IMERG directory does not exist: {directory}")

    by_start: dict[datetime, list[Path]] = {}
    for path in sorted(directory.rglob("3B-HHR.MS.MRG.3IMERG.*.V07B.HDF5*")):
        interval_start = _half_hour_start(path)
        if interval_start is not None:
            by_start.setdefault(interval_start, []).append(path)

    requested: list[Path] = []
    missing: list[str] = []
    duplicate: list[str] = []
    for day in _dates(start, end):
        window_start, _ = _bmd_window_bounds(day, accumulation_end_hour_utc)
        for offset in range(48):
            interval_start = window_start + timedelta(minutes=30 * offset)
            paths = by_start.get(interval_start, [])
            if not paths:
                missing.append(interval_start.isoformat(timespec="minutes") + "Z")
            elif len(paths) > 1:
                duplicate.append(
                    f"{interval_start.isoformat(timespec='minutes')}Z: "
                    f"{[path.name for path in paths]}"
                )
            else:
                requested.append(paths[0])

    if missing or duplicate:
        parts = []
        if missing:
            shown = ", ".join(missing[:12])
            suffix = f" ... ({len(missing)} total)" if len(missing) > 12 else ""
            parts.append("missing intervals: " + shown + suffix)
        if duplicate:
            shown = "; ".join(duplicate[:6])
            suffix = f" ... ({len(duplicate)} total)" if len(duplicate) > 6 else ""
            parts.append("duplicate intervals: " + shown + suffix)
        raise ValueError("invalid IMERG half-hourly inventory; " + " | ".join(parts))
    return requested


def _open_granule(
    path: Path,
    required: frozenset[str] = frozenset(
        {"precipitation", "randomError", "precipitation_cnt"}
    ),
) -> xr.Dataset:
    """Open either a conventional NetCDF root or the usual GPM ``Grid`` group."""

    errors: list[str] = []
    for group in (None, "Grid"):
        try:
            dataset = xr.open_dataset(path, group=group)
        except Exception as exc:  # pragma: no cover - engine-specific detail
            errors.append(f"group={group!r}: {exc}")
            continue
        if required.issubset(dataset):
            return dataset
        dataset.close()
    raise ValueError(
        f"{path} does not expose required variables {sorted(required)}; "
        + " | ".join(errors)
    )


def _coarse_centres(grid: Grid, factor: int) -> tuple[np.ndarray, np.ndarray]:
    if factor < 1 or grid.nlat % factor or grid.nlon % factor:
        raise ValueError(f"factor {factor} does not divide grid {grid.name} shape {grid.shape}")
    lat = grid.lat.reshape(grid.nlat // factor, factor).mean(axis=1)
    lon = grid.lon.reshape(grid.nlon // factor, factor).mean(axis=1)
    return lat, lon


def _coordinate_names(dataset: xr.Dataset) -> tuple[str, str]:
    lat = next((name for name in ("lat", "latitude") if name in dataset.coords), None)
    lon = next((name for name in ("lon", "longitude") if name in dataset.coords), None)
    if lat is None or lon is None:
        raise ValueError(f"IMERG coordinates not found; available={list(dataset.coords)}")
    return lat, lon


def _regional_array(
    dataset: xr.Dataset,
    variable: str,
    expected_lat: np.ndarray,
    expected_lon: np.ndarray,
) -> np.ndarray:
    lat_name, lon_name = _coordinate_names(dataset)
    array = dataset[variable].squeeze(drop=True)
    if lat_name not in array.dims or lon_name not in array.dims:
        raise ValueError(f"{variable} dimensions {array.dims} lack {lat_name}/{lon_name}")
    array = array.transpose(lat_name, lon_name)
    source_lat = np.asarray(dataset[lat_name].values, dtype=np.float64)
    source_lon = np.asarray(dataset[lon_name].values, dtype=np.float64)
    source_lon = np.where(source_lon > 180.0, source_lon - 360.0, source_lon)
    lat_index = np.array([int(np.argmin(np.abs(source_lat - value))) for value in expected_lat])
    lon_index = np.array([int(np.argmin(np.abs(source_lon - value))) for value in expected_lon])
    if not np.allclose(source_lat[lat_index], expected_lat, atol=2e-3, rtol=0):
        raise ValueError("IMERG latitude centres do not nest on the requested grid")
    if not np.allclose(source_lon[lon_index], expected_lon, atol=2e-3, rtol=0):
        raise ValueError("IMERG longitude centres do not nest on the requested grid")
    return np.asarray(array.isel({lat_name: lat_index, lon_name: lon_index}).values)


def _require_mm_per_day(dataset: xr.Dataset, path: Path) -> None:
    accepted = {"mm/day", "mmday-1", "mmd-1", "mmday^-1", "mmd^-1"}
    for variable in ("precipitation", "randomError"):
        units = str(dataset[variable].attrs.get("units", ""))
        normalized = units.lower().replace(" ", "")
        if normalized not in accepted:
            raise ValueError(
                f"{path} {variable} units are {units!r}, expected daily millimetres; "
                "do not assimilate or apply a factor-of-24 conversion implicitly"
            )


def _require_mm_per_hour(dataset: xr.Dataset, path: Path) -> None:
    accepted = {
        "mm/hr",
        "mmh-1",
        "mmhr-1",
        "mmhour-1",
        "mmh^-1",
        "mmhr^-1",
        "mmhour^-1",
    }
    for variable in ("precipitation", "randomError"):
        units = str(dataset[variable].attrs.get("units", ""))
        normalized = units.lower().replace(" ", "")
        if normalized not in accepted:
            raise ValueError(
                f"{path} {variable} units are {units!r}, expected millimetres per hour"
            )


def load_imerg_daily(
    directory: str | Path,
    start: str | np.datetime64,
    end: str | np.datetime64,
    *,
    grid: Grid = BD,
    factor: int = 2,
    min_count: int = 40,
) -> ImergDaily:
    """Validate and crop daily IMERG V07B granules to ``grid``.

    Precipitation and random error are already in mm/day in the daily V07
    product.  A footprint is withheld when fewer than ``min_count`` of the 48
    half-hourly estimates contributed, or when either value is invalid.
    """

    if not 0 <= min_count <= 48:
        raise ValueError("min_count must be between 0 and 48")
    files = discover_imerg_files(directory, start, end)
    expected_lat, expected_lon = _coarse_centres(grid, factor)
    precip, error, count = [], [], []
    for path in files:
        dataset = _open_granule(path)
        try:
            _require_mm_per_day(dataset, path)
            p = _regional_array(dataset, "precipitation", expected_lat, expected_lon)
            e = _regional_array(dataset, "randomError", expected_lat, expected_lon)
            c = _regional_array(dataset, "precipitation_cnt", expected_lat, expected_lon)
        finally:
            dataset.close()
        p = p.astype(np.float32)
        e = e.astype(np.float32)
        c = c.astype(np.int16)
        valid = np.isfinite(p) & np.isfinite(e) & (p >= 0) & (e >= 0) & (c >= min_count)
        precip.append(np.where(valid, p, np.nan).astype(np.float32))
        error.append(np.where(valid, e, np.nan).astype(np.float32))
        count.append(c)
    return ImergDaily(
        time=np.array([np.datetime64(day.isoformat(), "ns") for day in _dates(start, end)]),
        precipitation=np.stack(precip),
        random_error=np.stack(error),
        count=np.stack(count),
        lat=expected_lat,
        lon=expected_lon,
        source_files=tuple(path.name for path in files),
        min_count=min_count,
    )


def load_imerg_bmd_windows(
    directory: str | Path,
    start: str | np.datetime64,
    end: str | np.datetime64,
    *,
    grid: Grid = BD,
    factor: int = 2,
    min_count: int = 48,
    accumulation_end_hour_utc: int = 3,
) -> ImergDaily:
    """Accumulate half-hourly V07B rates over exact BMD reporting days.

    Each ``precipitation`` value is a half-hour mean rate in mm/hr, so daily
    depth is ``sum(rate * 0.5 hr)``.  Half-hourly ``randomError`` is converted
    to depth in the same way and accumulated in quadrature.  The latter is an
    independence baseline; unknown residual temporal dependence must be
    assessed with the DA observation-error sensitivity rather than assumed
    away.

    By default all 48 intervals must be valid at a footprint.  A lower
    ``min_count`` is allowed only when explicitly requested and does not scale
    partial accumulations to compensate for missing intervals.
    """

    if not 1 <= min_count <= 48:
        raise ValueError("min_count must be between 1 and 48")
    files = discover_imerg_half_hourly_files(
        directory,
        start,
        end,
        accumulation_end_hour_utc=accumulation_end_hour_utc,
    )
    expected_lat, expected_lon = _coarse_centres(grid, factor)
    output_precip: list[np.ndarray] = []
    output_error: list[np.ndarray] = []
    output_count: list[np.ndarray] = []
    half_hour_hours = np.float32(0.5)
    required = frozenset({"precipitation", "randomError"})

    for day_position, day in enumerate(_dates(start, end)):
        window_files = files[48 * day_position : 48 * (day_position + 1)]
        precipitation_sum = np.zeros((len(expected_lat), len(expected_lon)), np.float64)
        error_variance = np.zeros_like(precipitation_sum)
        count = np.zeros_like(precipitation_sum, dtype=np.int16)

        for path in window_files:
            dataset = _open_granule(path, required=required)
            try:
                _require_mm_per_hour(dataset, path)
                precipitation_rate = _regional_array(
                    dataset, "precipitation", expected_lat, expected_lon
                ).astype(np.float64)
                error_rate = _regional_array(
                    dataset, "randomError", expected_lat, expected_lon
                ).astype(np.float64)
            finally:
                dataset.close()
            valid = (
                np.isfinite(precipitation_rate)
                & np.isfinite(error_rate)
                & (precipitation_rate >= 0)
                & (error_rate >= 0)
            )
            precipitation_depth = precipitation_rate * half_hour_hours
            error_depth = error_rate * half_hour_hours
            precipitation_sum[valid] += precipitation_depth[valid]
            error_variance[valid] += error_depth[valid] ** 2
            count[valid] += 1

        valid_day = count >= min_count
        output_precip.append(
            np.where(valid_day, precipitation_sum, np.nan).astype(np.float32)
        )
        output_error.append(
            np.where(valid_day, np.sqrt(error_variance), np.nan).astype(np.float32)
        )
        output_count.append(count)

    return ImergDaily(
        time=np.array(
            [
                np.datetime64(day.isoformat(), "ns")
                + np.timedelta64(accumulation_end_hour_utc, "h")
                for day in _dates(start, end)
            ]
        ),
        precipitation=np.stack(output_precip),
        random_error=np.stack(output_error),
        count=np.stack(output_count),
        lat=expected_lat,
        lon=expected_lon,
        source_files=tuple(path.name for path in files),
        min_count=min_count,
        source_frequency="half-hourly",
        accumulation_end_hour_utc=accumulation_end_hour_utc,
        random_error_aggregation=(
            "sqrt(sum((half-hourly randomError in mm/hr * 0.5 hr)^2)); "
            "temporal-independence baseline"
        ),
    )
