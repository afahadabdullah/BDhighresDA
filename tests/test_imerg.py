from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pytest
import xarray as xr

from bdhires.grids import BD, Grid
from bdhires.imerg import (
    discover_imerg_files,
    discover_imerg_half_hourly_files,
    load_imerg_bmd_windows,
    load_imerg_daily,
)


def _write(path: Path, day: int, bad_count: bool = False) -> None:
    lat = BD.lat.reshape(64, 2).mean(axis=1)
    lon = BD.lon.reshape(64, 2).mean(axis=1)
    # Preserve IMERG's unusual time/lon/lat storage order to test orientation.
    values = day + np.arange(64, dtype=np.float32)[:, None] + 0.01 * np.arange(64)[None]
    count = np.full((64, 64), 48, dtype=np.int16)
    if bad_count:
        count[2, 3] = 12
    xr.Dataset(
        {
            "precipitation": (
                ("time", "lon", "lat"), values[None], {"units": "mm/day"}
            ),
            "randomError": (
                ("time", "lon", "lat"), np.full((1, 64, 64), 2.0),
                {"units": "mm/day"},
            ),
            "precipitation_cnt": (("time", "lon", "lat"), count[None]),
        },
        coords={"time": [np.datetime64(f"2018-05-{day:02d}")], "lon": lon, "lat": lat},
    ).to_netcdf(path)


def test_load_imerg_daily_orients_and_masks(tmp_path: Path) -> None:
    for day in (1, 2):
        path = tmp_path / f"3B-DAY.MS.MRG.3IMERG.201805{day:02d}-S000000-E235959.V07B.nc4"
        _write(path, day, bad_count=day == 2)
    result = load_imerg_daily(tmp_path, "2018-05-01", "2018-05-02", min_count=40)
    assert result.precipitation.shape == (2, 64, 64)
    assert result.precipitation[0, 5, 7] == pytest.approx(1 + 7 + 0.01 * 5)
    assert np.isnan(result.precipitation[1, 3, 2])
    assert result.count[1, 3, 2] == 12


def test_discover_imerg_files_rejects_missing_date(tmp_path: Path) -> None:
    path = tmp_path / "3B-DAY.MS.MRG.3IMERG.20180501-S000000-E235959.V07B.nc4"
    _write(path, 1)
    with pytest.raises(ValueError, match="missing dates: 2018-05-02"):
        discover_imerg_files(tmp_path, "2018-05-01", "2018-05-02")


def test_load_imerg_daily_rejects_wrong_units(tmp_path: Path) -> None:
    path = tmp_path / "3B-DAY.MS.MRG.3IMERG.20180501-S000000-E235959.V07B.nc4"
    _write(path, 1)
    with xr.open_dataset(path) as source:
        dataset = source.load()
    dataset["precipitation"].attrs["units"] = "mm/hr"
    dataset.to_netcdf(path, mode="w")
    with pytest.raises(ValueError, match="expected daily millimetres"):
        load_imerg_daily(tmp_path, "2018-05-01", "2018-05-01")


TINY = Grid("tiny", lon_min=90.0, lat_min=22.0, nlon=2, nlat=2, res=0.1)


def _write_half_hour(
    directory: Path,
    interval_start: datetime,
    *,
    rate: float = 2.0,
    error_rate: float = 1.0,
    invalidate: tuple[int, int] | None = None,
) -> Path:
    interval_end = interval_start + timedelta(minutes=29, seconds=59)
    minute_of_day = interval_start.hour * 60 + interval_start.minute
    path = directory / (
        "3B-HHR.MS.MRG.3IMERG."
        f"{interval_start:%Y%m%d}-S{interval_start:%H%M%S}-"
        f"E{interval_end:%H%M%S}.{minute_of_day:04d}.V07B.HDF5.SUB.nc4"
    )
    # Store lon/lat order to exercise the same orientation path as real IMERG.
    precipitation = np.full((1, 2, 2), rate, np.float32)
    random_error = np.full((1, 2, 2), error_rate, np.float32)
    if invalidate is not None:
        lon_index, lat_index = invalidate
        precipitation[0, lon_index, lat_index] = np.nan
    xr.Dataset(
        {
            "precipitation": (
                ("time", "lon", "lat"), precipitation, {"units": "mm/hr"}
            ),
            "randomError": (
                ("time", "lon", "lat"), random_error, {"units": "mm/hr"}
            ),
        },
        coords={
            "time": [np.datetime64(interval_start)],
            "lon": TINY.lon,
            "lat": TINY.lat,
        },
    ).to_netcdf(path)
    return path


def _write_bmd_window(directory: Path, *, invalidate_first: bool = False) -> None:
    start = datetime(2018, 4, 30, 3)
    for offset in range(48):
        _write_half_hour(
            directory,
            start + timedelta(minutes=30 * offset),
            invalidate=(1, 0) if invalidate_first and offset == 0 else None,
        )


def test_load_half_hourly_imerg_uses_exact_bmd_window(tmp_path: Path) -> None:
    _write_bmd_window(tmp_path, invalidate_first=True)
    files = discover_imerg_half_hourly_files(
        tmp_path, "2018-05-01", "2018-05-01", accumulation_end_hour_utc=3
    )
    assert len(files) == 48
    assert "20180430-S030000" in files[0].name
    assert "20180501-S023000" in files[-1].name

    result = load_imerg_bmd_windows(
        tmp_path, "2018-05-01", "2018-05-01", grid=TINY, factor=1, min_count=48
    )
    assert result.source_frequency == "half-hourly"
    assert result.accumulation_end_hour_utc == 3
    assert result.time[0] == np.datetime64("2018-05-01T03:00")
    assert result.precipitation[0, 1, 1] == pytest.approx(48.0)
    assert result.random_error[0, 1, 1] == pytest.approx(np.sqrt(12.0))
    # Invalid lon index 1 / lat index 0 becomes row 0 / column 1 after transpose.
    assert result.count[0, 0, 1] == 47
    assert np.isnan(result.precipitation[0, 0, 1])


def test_half_hourly_inventory_rejects_partial_download(tmp_path: Path) -> None:
    start = datetime(2018, 4, 30, 3)
    for offset in range(47):
        _write_half_hour(tmp_path, start + timedelta(minutes=30 * offset))
    with pytest.raises(ValueError, match="missing intervals: 2018-05-01T02:30Z"):
        discover_imerg_half_hourly_files(tmp_path, "2018-05-01", "2018-05-01")
