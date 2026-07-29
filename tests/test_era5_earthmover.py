from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr


def load_downloader():
    path = Path(__file__).resolve().parents[1] / "scripts" / "00_download_era5.py"
    spec = importlib.util.spec_from_file_location("download_era5", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(("year", "days"), [(1981, 365), (1984, 366)])
def test_hourly_to_daily_alignment(year: int, days: int) -> None:
    module = load_downloader()
    time = pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq="h")
    latitude = np.array([29.75, 15.0])
    longitude = np.array([83.0, 97.75])
    shape = (len(time), len(latitude), len(longitude))

    values = {
        "tp": (
            ("valid_time", "latitude", "longitude"),
            np.full(shape, 0.001, dtype="float32"),
        )
    }
    hour = (np.arange(len(time), dtype="float32") % 24)[:, None, None]
    for index, variable in enumerate(module.STATE_VARIABLES):
        values[variable] = (
            ("valid_time", "latitude", "longitude"),
            np.broadcast_to(hour + index, shape).copy(),
        )

    source = xr.Dataset(
        values,
        coords={
            "valid_time": time,
            "latitude": latitude,
            "longitude": longitude,
        },
    )
    daily = module.aggregate_year(source, year).load()

    assert daily.sizes["time"] == days
    assert str(daily.time.values[0]).startswith(f"{year}-01-01")
    assert str(daily.time.values[-1]).startswith(f"{year}-12-31")
    np.testing.assert_allclose(daily["tp"], 24.0, rtol=1e-6)
    for index, variable in enumerate(module.STATE_VARIABLES):
        np.testing.assert_allclose(daily[variable], 11.5 + index, rtol=1e-6)
