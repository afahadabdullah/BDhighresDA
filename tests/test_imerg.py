from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from bdhires.grids import BD
from bdhires.imerg import discover_imerg_files, load_imerg_daily


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
