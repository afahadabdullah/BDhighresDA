import numpy as np

from bdhires.ensrf import gaspari_cohn, localized_serial_ensrf
from bdhires.grids import Grid
from bdhires.transforms import PrecipTransform


def test_gaspari_cohn_is_compact() -> None:
    weights = gaspari_cohn(np.array([0.0, 50.0, 100.0, 150.0]), 100.0)
    assert weights[0] == 1.0
    assert 0.0 < weights[1] < 1.0
    assert weights[2] == 0.0
    assert weights[3] == 0.0


def test_localized_ensrf_reduces_gauge_innovation_without_remote_change() -> None:
    grid = Grid("toy", lon_min=90.0, lat_min=23.0, nlon=12, nlat=12, res=0.1)
    transform = PrecipTransform(kind="log1p", eps=0.1, mu=0.0, sd=1.0)
    members = 12
    rng = np.random.default_rng(3)
    ensemble = np.full((members, grid.nlat, grid.nlon), 5.0, dtype=np.float32)
    # Give the ensemble covariance at the observed location while preserving a
    # positive rainfall state for every member.
    ensemble += rng.normal(0.0, 1.0, members)[:, None, None]
    before_remote = ensemble[:, -1, -1].copy()
    station_lat = np.array([grid.lat[2]])
    station_lon = np.array([grid.lon[2]])
    observed = np.array([12.0], dtype=np.float32)
    updated, diagnostic = localized_serial_ensrf(
        ensemble,
        observed,
        station_lat,
        station_lon,
        grid,
        transform,
        np.ones(grid.shape, dtype=bool),
        observation_variance=0.1**2,
        localization_km=80.0,
        seed=4,
    )
    before_error = abs(float(ensemble[:, 2, 2].mean()) - observed[0])
    after_error = abs(float(updated[:, 2, 2].mean()) - observed[0])
    assert after_error < before_error
    np.testing.assert_allclose(updated[:, -1, -1], before_remote, atol=1e-5)
    assert (
        diagnostic["innovation_rmse_after_transformed"]
        < diagnostic["innovation_rmse_before_transformed"]
    )
