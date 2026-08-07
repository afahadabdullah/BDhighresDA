"""Point-vs-area error estimation from the gauge network.

These tests use synthetic fields with a KNOWN nugget, sill and range, so a
failure means the estimator is wrong rather than that the weather was unusual.
"""

from __future__ import annotations

import numpy as np
import pytest

from bdhires.eval.representativeness import (
    VariogramFit,
    aggregation_decomposition,
    block_dispersion_variance,
    empirical_variogram,
    fit_variogram,
    haversine_km,
    representativeness_sigma,
)


def test_haversine_matches_known_separations():
    # One degree of latitude is ~111.2 km anywhere.
    assert haversine_km(23.0, 90.0, 24.0, 90.0) == pytest.approx(111.2, abs=0.5)
    # Dhaka to Chittagong is ~215 km.
    assert haversine_km(23.767, 90.383, 22.335, 91.834) == pytest.approx(215, abs=10)
    assert haversine_km(23.0, 90.0, 23.0, 90.0) == pytest.approx(0.0, abs=1e-9)


def _gaussian_field_at_stations(lat, lon, fit, n_days, seed=0):
    """Draw days from a Gaussian process with the given variogram.

    Covariance is ``total_sill - gamma(h)``, which is the standard conversion
    for a second-order stationary field, so a variogram estimated back off these
    draws must recover the parameters that generated them.
    """
    rng = np.random.default_rng(seed)
    from bdhires.eval.representativeness import pair_distances_km

    distance = pair_distances_km(lat, lon)
    covariance = fit.total_sill - fit(distance)
    covariance[np.diag_indices_from(covariance)] = fit.total_sill
    # Nugget is a pure discontinuity at zero: it lives on the diagonal only.
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    root = eigenvectors @ np.diag(np.sqrt(np.clip(eigenvalues, 0, None)))
    return rng.normal(size=(n_days, len(lat))) @ root.T


def test_variogram_recovers_the_parameters_that_generated_the_field():
    truth = VariogramFit(nugget=2.0, sill=8.0, range_km=60.0)
    rng = np.random.default_rng(0)
    lat = 21.0 + 5.0 * rng.random(40)
    lon = 88.0 + 4.0 * rng.random(40)
    values = _gaussian_field_at_stations(lat, lon, truth, n_days=800, seed=1)

    edges = np.linspace(0, 400, 21)
    empirical = empirical_variogram(values, lat, lon, edges)
    fitted = fit_variogram(
        empirical["distance_km"], empirical["gamma"], empirical["n_pairs"]
    )

    assert fitted.total_sill == pytest.approx(truth.total_sill, rel=0.15)
    assert fitted.range_km == pytest.approx(truth.range_km, rel=0.6)
    assert fitted.nugget == pytest.approx(truth.nugget, abs=1.5)


def test_pure_nugget_field_fits_a_flat_variogram():
    """White noise between stations must produce sill ~ 0, not a spurious range."""
    rng = np.random.default_rng(3)
    lat = 21.0 + 5.0 * rng.random(30)
    lon = 88.0 + 4.0 * rng.random(30)
    values = rng.normal(scale=np.sqrt(5.0), size=(600, 30))

    empirical = empirical_variogram(values, lat, lon, np.linspace(0, 400, 21))
    fitted = fit_variogram(
        empirical["distance_km"], empirical["gamma"], empirical["n_pairs"]
    )

    assert fitted.total_sill == pytest.approx(5.0, rel=0.2)
    assert fitted.nugget_fraction > 0.8


def test_pure_nugget_is_not_aliased_into_a_tiny_range():
    """The failure mode that motivated ``min_range_km``.

    A pure-nugget field and an exponential with a 3 km range look identical at
    every separation a 30-station network can resolve.  Left unbounded, the fit
    prefers the latter and reports nugget 0, which understates the point-to-cell
    error for a 5 km cell by ~28%.
    """
    rng = np.random.default_rng(17)
    lat = 21.0 + 5.0 * rng.random(30)
    lon = 88.0 + 4.0 * rng.random(30)
    values = rng.normal(scale=np.sqrt(4.0), size=(700, 30))

    empirical = empirical_variogram(values, lat, lon, np.linspace(0, 400, 21))
    bounded = fit_variogram(
        empirical["distance_km"], empirical["gamma"], empirical["n_pairs"],
        min_separation_km=empirical["min_separation_km"],
    )
    unbounded = fit_variogram(
        empirical["distance_km"], empirical["gamma"], empirical["n_pairs"],
        min_range_km=0.5,
    )

    smallest_usable_bin = np.nanmin(
        np.where(np.isfinite(empirical["gamma"]), empirical["distance_km"], np.nan)
    )
    assert bounded.nugget_fraction > 0.9
    assert bounded.range_km >= smallest_usable_bin * 0.999
    # The bounded fit must not understate the 5 km representativeness error.
    assert representativeness_sigma(bounded, 5.0) >= representativeness_sigma(
        unbounded, 5.0
    )
    assert representativeness_sigma(bounded, 5.0) == pytest.approx(2.0, rel=0.15)


def test_variogram_never_returns_a_negative_nugget():
    """The constrained fit is the reason this holds even on noisy short bins."""
    rng = np.random.default_rng(7)
    lat = 21.0 + 5.0 * rng.random(25)
    lon = 88.0 + 4.0 * rng.random(25)
    truth = VariogramFit(nugget=0.0, sill=6.0, range_km=40.0)
    values = _gaussian_field_at_stations(lat, lon, truth, n_days=120, seed=5)

    empirical = empirical_variogram(values, lat, lon, np.linspace(0, 400, 15))
    fitted = fit_variogram(
        empirical["distance_km"], empirical["gamma"], empirical["n_pairs"]
    )

    assert fitted.nugget >= 0.0
    assert fitted.sill >= 0.0


def test_thin_bins_are_dropped_rather_than_fitted_through():
    rng = np.random.default_rng(11)
    lat = np.array([23.0, 23.1, 23.2, 26.5])       # one far outlier station
    lon = np.array([90.0, 90.1, 90.2, 88.5])
    values = rng.normal(size=(50, 4))

    empirical = empirical_variogram(
        values, lat, lon, np.linspace(0, 500, 26), min_pairs=100
    )

    assert np.isnan(empirical["gamma"]).any()
    assert empirical["min_separation_km"] > 0


def test_block_dispersion_is_the_nugget_for_a_pure_nugget_variogram():
    """Every distinct pair in the cell sees the nugget, so gammabar == nugget."""
    pure = VariogramFit(nugget=3.0, sill=0.0, range_km=50.0)
    assert block_dispersion_variance(pure, cell_km=5.0) == pytest.approx(3.0, rel=1e-6)


def test_block_dispersion_grows_with_cell_size_and_vanishes_at_zero():
    fit = VariogramFit(nugget=1.0, sill=9.0, range_km=50.0)
    tiny = block_dispersion_variance(fit, cell_km=0.001)
    cell = block_dispersion_variance(fit, cell_km=5.0)
    footprint = block_dispersion_variance(fit, cell_km=50.0)

    assert tiny == pytest.approx(fit.nugget, rel=0.05)
    assert cell > tiny
    assert footprint > cell
    assert footprint < fit.total_sill        # bounded by the sill
    assert block_dispersion_variance(fit, cell_km=0.0) == 0.0


def test_representativeness_sigma_is_the_root_of_the_dispersion():
    fit = VariogramFit(nugget=1.0, sill=9.0, range_km=50.0)
    assert representativeness_sigma(fit, 5.0) ** 2 == pytest.approx(
        block_dispersion_variance(fit, 5.0), rel=1e-9
    )


def test_block_dispersion_is_insensitive_to_the_discretisation():
    fit = VariogramFit(nugget=0.5, sill=4.0, range_km=30.0)
    coarse = block_dispersion_variance(fit, 5.0, n_sub=8)
    fine = block_dispersion_variance(fit, 5.0, n_sub=24)
    assert coarse == pytest.approx(fine, rel=0.02)


def test_aggregation_decomposition_recovers_a_known_split():
    """MSE(N) = 4 + 16/N must come back as systematic 4, random 16."""
    windows = np.array([1, 5, 10, 30], float)
    mse = 4.0 + 16.0 / windows

    out = aggregation_decomposition(windows, mse)

    assert out["systematic_mse"] == pytest.approx(4.0, rel=1e-6)
    assert out["random_mse"] == pytest.approx(16.0, rel=1e-6)
    assert out["systematic_rmse"] == pytest.approx(2.0, rel=1e-6)
    assert out["r_squared"] == pytest.approx(1.0, abs=1e-9)
    assert not out["intercept_was_negative"]


def test_pure_random_error_decomposes_to_a_zero_floor():
    """A product with no persistent offset must show systematic ~ 0."""
    windows = np.array([1, 5, 10, 30], float)
    out = aggregation_decomposition(windows, 9.0 / windows)

    assert out["systematic_mse"] == pytest.approx(0.0, abs=1e-6)
    assert out["random_mse"] == pytest.approx(9.0, rel=1e-6)


def test_decomposition_flags_error_that_grows_with_averaging():
    """MSE = a + b/N cannot rise with N at all, so this must not be quoted.

    It shows up as a negative SLOPE, not a negative intercept -- the intercept
    stays comfortably positive and would look like a perfectly respectable
    representativeness floor if only that flag were checked.
    """
    windows = np.array([1, 5, 10, 30], float)
    out = aggregation_decomposition(windows, np.array([1.0, 4.0, 7.0, 20.0]))

    assert out["error_grows_with_averaging"]
    assert not out["model_is_valid"]
    assert out["systematic_mse"] > 0        # positive, and meaningless


def test_a_healthy_decomposition_is_marked_valid():
    windows = np.array([1, 5, 10, 30], float)
    out = aggregation_decomposition(windows, 4.0 + 16.0 / windows)

    assert out["model_is_valid"]
    assert not out["error_grows_with_averaging"]
    assert not out["intercept_was_negative"]


def test_empirical_variogram_rejects_a_bad_shape():
    with pytest.raises(ValueError, match="must be"):
        empirical_variogram(np.zeros(10), [23.0], [90.0], np.linspace(0, 100, 5))


def test_fit_needs_enough_usable_bins():
    with pytest.raises(ValueError, match="at least 3"):
        fit_variogram(np.array([10.0, 20.0]), np.array([1.0, 2.0]))
