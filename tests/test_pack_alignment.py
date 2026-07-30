from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import xarray as xr


def load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_conservative_regrid_preserves_uniform_field() -> None:
    module = load_script("04_regrid_and_pack.py")
    source_lat = np.array([0.5, 1.5])
    source_lon = np.array([10.5, 11.5])
    target_lat = np.array([0.25, 0.75, 1.25, 1.75])
    target_lon = np.array([10.25, 10.75, 11.25, 11.75])
    source = xr.DataArray(
        np.full((3, 2, 2), 7.5, dtype=np.float32),
        dims=("time", "lat", "lon"),
        coords={
            "time": np.arange(3),
            "lat": source_lat,
            "lon": source_lon,
        },
    )
    result = module.conservative_precipitation(
        source,
        target_lat,
        target_lon,
    )
    assert result.shape == (3, 4, 4)
    np.testing.assert_allclose(result, 7.5, rtol=1e-6)


def test_alignment_correlation_peaks_at_zero() -> None:
    module = load_script("04_check_alignment.py")
    rng = np.random.default_rng(4)
    target = rng.gamma(shape=1.5, scale=5.0, size=500)
    background = target + rng.normal(scale=0.2, size=target.shape)
    correlations = module.lag_correlations(
        target,
        background,
        [-2, -1, 0, 1, 2],
    )
    assert max(correlations, key=correlations.get) == 0
