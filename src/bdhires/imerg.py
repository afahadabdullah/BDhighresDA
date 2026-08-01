"""Strict ingestion of daily GPM IMERG Final V07 observations.

IMERG is an observation stream, not a model-conditioning channel.  This
module therefore reads the original daily granules independently of the
checkpoint-bound training Zarr and extracts the exact 0.1-degree footprints
that nest over the 0.05-degree Bangladesh grid.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

from .grids import BD, Grid


_DAILY_NAME = re.compile(
    r"^3B-DAY\.MS\.MRG\.3IMERG\.(?P<date>\d{8})-S000000-E235959\.V07B\.nc4$"
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


def _open_granule(path: Path) -> xr.Dataset:
    """Open either a conventional NetCDF root or the usual GPM ``Grid`` group."""

    errors: list[str] = []
    for group in (None, "Grid"):
        try:
            dataset = xr.open_dataset(path, group=group)
        except Exception as exc:  # pragma: no cover - engine-specific detail
            errors.append(f"group={group!r}: {exc}")
            continue
        if {"precipitation", "randomError", "precipitation_cnt"}.issubset(dataset):
            return dataset
        dataset.close()
    raise ValueError(
        f"{path} does not expose precipitation, randomError, and precipitation_cnt; "
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
