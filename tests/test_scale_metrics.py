"""Invariants for the scale-explicit verification module.

These tests exist to catch the failure that would be hardest to notice in a
result table: a scale decomposition that silently leaks coarse information into
the subgrid component would make claim B look strong for the wrong reason.
"""
from __future__ import annotations

import numpy as np
import pytest

from bdhires.eval import scale as S


@pytest.fixture
def field():
    rng = np.random.default_rng(3)
    data = rng.normal(5.0, 2.0, (4, 16, 16))
    data[:, :2, :2] = np.nan          # ocean corner
    return data


# --------------------------------------------------------------------------
# Block algebra
# --------------------------------------------------------------------------

def test_residual_has_zero_block_mean(field):
    """The defining property: subgrid carries no footprint-scale information."""
    mask = S.eligible_mask(field, 2, 1.0)
    _, residual = S.scale_decompose(field, 2, mask)
    assert np.nanmax(np.abs(S.block_mean(residual, 2, mask))) < 1e-10


def test_decomposition_is_exact(field):
    mask = S.eligible_mask(field, 4, 1.0)
    coarse, residual = S.scale_decompose(field, 4, mask)
    total = coarse + residual
    assert np.allclose(total[mask], field[mask])


def test_factor_one_leaves_field_untouched(field):
    mask = np.isfinite(field)
    coarse, residual = S.scale_decompose(field, 1, mask)
    assert np.allclose(coarse[mask], field[mask])
    assert np.nanmax(np.abs(residual)) < 1e-12


def test_strict_mask_drops_partial_blocks():
    truth = np.ones((1, 4, 4))
    truth[:, 0, 0] = np.nan
    strict = S.eligible_mask(truth, 2, 1.0)
    relaxed = S.eligible_mask(truth, 2, 0.5)
    assert strict[0, :2, :2].sum() == 0       # whole block rejected
    assert relaxed[0, :2, :2].sum() == 3      # partial block retained
    assert strict[0, 2:, 2:].all()


def test_block_mean_ignores_missing_cells():
    field = np.array([[[1.0, 3.0], [np.nan, 5.0]]])
    mask = np.isfinite(field)
    assert S.block_mean(field, 2, mask)[0, 0, 0] == pytest.approx(3.0)


def test_non_divisible_shape_is_rejected(field):
    with pytest.raises(ValueError):
        S.block_mean(field, 5, np.isfinite(field))


# --------------------------------------------------------------------------
# Null models
# --------------------------------------------------------------------------

def test_footprint_perfect_null_has_no_subgrid(field):
    mask = S.eligible_mask(field, 2, 1.0)
    null = S.footprint_perfect_null(field, 2, mask)
    _, residual = S.scale_decompose(null[0], 2, mask)
    assert np.nanmax(np.abs(residual)) < 1e-10


def test_footprint_perfect_null_is_exact_at_footprint_scale(field):
    """The null must be *perfect* coarsely -- that is what makes it hard."""
    mask = S.eligible_mask(field, 2, 1.0)
    null = S.footprint_perfect_null(field, 2, mask)
    truth_coarse, _ = S.scale_decompose(field, 2, mask)
    scores = S.deterministic_scores(null, truth_coarse)
    assert scores["rmse_mm"] == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------
# Scores
# --------------------------------------------------------------------------

def test_perfect_forecast_scores_unit_skill():
    truth = np.array([[[-1.0, 1.0], [2.0, -2.0]]])
    perfect = np.stack([truth, truth])
    null = S.deterministic_scores(np.zeros((1, *truth.shape)), truth)
    skill = S.skill_against(S.deterministic_scores(perfect, truth), null)
    assert skill["mse_skill"] == pytest.approx(1.0)
    assert skill["crps_skill"] == pytest.approx(1.0)


def test_right_amplitude_wrong_location_earns_no_skill():
    """The whole point of scoring against a null: sharpness is not skill."""
    truth = np.array([[[-1.0, 1.0], [2.0, -2.0]]])
    inverted = np.stack([-truth, -truth])
    null = S.deterministic_scores(np.zeros((1, *truth.shape)), truth)
    scores = S.deterministic_scores(inverted, truth)
    assert scores["mean_member_energy_ratio"] == pytest.approx(1.0)
    assert S.skill_against(scores, null)["mse_skill"] < 0.0


def test_fair_crps_below_plain_crps_for_small_ensembles():
    rng = np.random.default_rng(0)
    ensemble = rng.normal(size=(6, 4000))
    truth = rng.normal(size=4000)
    fair = S.fair_crps(ensemble, truth)
    plain = float(
        np.mean(np.abs(ensemble - truth))
        - 0.5 * np.mean(np.abs(ensemble[:, None] - ensemble[None, :]))
    )
    assert fair < plain


def test_fair_crps_matches_mae_for_one_member():
    ensemble = np.array([[1.0, 2.0, 3.0]])
    truth = np.array([1.5, 2.5, 2.0])
    assert S.fair_crps(ensemble, truth) == pytest.approx(
        np.mean(np.abs(ensemble[0] - truth)))


def test_empty_selection_returns_empty_dict():
    truth = np.full((1, 2, 2), np.nan)
    assert S.deterministic_scores(np.zeros((1, 1, 2, 2)), truth) == {}


# --------------------------------------------------------------------------
# Spectra
# --------------------------------------------------------------------------

def test_smoothing_degrades_effective_resolution():
    """A block-averaged field must report a coarser effective resolution."""
    rng = np.random.default_rng(7)
    truth = rng.normal(size=(3, 32, 32))
    mask = np.ones_like(truth, dtype=bool)
    smoothed = np.stack([
        S.scale_decompose(day, 4, np.ones((32, 32), bool))[0] for day in truth
    ])
    summary = S.spectral_summary({"sharp": truth, "smooth": smoothed},
                                 truth, mask)
    assert (summary["effective_resolution_km"]["smooth"]
            > summary["effective_resolution_km"]["sharp"])


def test_identical_field_has_unit_power_ratio():
    rng = np.random.default_rng(1)
    truth = rng.normal(size=(2, 32, 32))
    mask = np.ones_like(truth, dtype=bool)
    summary = S.spectral_summary({"same": truth}, truth, mask)
    ratio = np.asarray(summary["power_ratio"]["same"])
    assert np.allclose(ratio[np.isfinite(ratio)], 1.0, atol=1e-6)


def test_effective_resolution_stops_at_first_excursion():
    wavelength = np.array([100.0, 50.0, 25.0, 12.0])
    ratio = np.array([1.0, 1.1, 5.0, 5.0])       # leaves the band at 25 km
    assert S.effective_resolution(wavelength, ratio, tolerance=2.0) == 50.0


# --------------------------------------------------------------------------
# Neighbourhood verification
# --------------------------------------------------------------------------

def test_fss_is_one_against_itself():
    rng = np.random.default_rng(2)
    truth = rng.gamma(2.0, 5.0, (3, 16, 16))
    mask = np.ones_like(truth, dtype=bool)
    grid = S.fss_grid(truth, truth, mask, thresholds=(5.0,), windows=(1, 3, 5))
    assert np.allclose(grid["fss"]["5"], 1.0)
    assert grid["skillful_scale"]["5"] == 1.0


def test_fss_improves_with_neighbourhood_for_displaced_field():
    """A one-cell shift should be forgiven as the neighbourhood widens."""
    truth = np.zeros((1, 32, 32))
    truth[0, 12:20, 12:20] = 20.0
    shifted = np.roll(truth, 3, axis=2)
    mask = np.ones_like(truth, dtype=bool)
    grid = S.fss_grid(shifted, truth, mask, thresholds=(10.0,), windows=(1, 9, 17))
    curve = grid["fss"]["10"]
    assert curve[0] < curve[-1]


def test_neighbourhood_fraction_is_a_local_mean():
    binary = np.zeros((5, 5))
    binary[2, 2] = 1.0
    fraction = S._fraction(binary, 3)
    assert fraction[2, 2] == pytest.approx(1 / 9)
    assert fraction[0, 0] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Ladder and variogram
# --------------------------------------------------------------------------

def test_ladder_reports_one_row_per_factor(field):
    mask = np.isfinite(field)
    ladder = S.scale_ladder({"copy": field[None]}, field, mask, factors=(1, 2, 4))
    assert ladder.degrees == [0.05, 0.1, 0.2]
    assert len(ladder.aggregated["copy"]) == 3
    assert all(row["rmse_mm"] == pytest.approx(0.0, abs=1e-9)
               for row in ladder.aggregated["copy"])


def test_variogram_grows_with_lag_for_smooth_fields():
    yy, xx = np.mgrid[0:32, 0:32]
    smooth = np.stack([np.sin(2 * np.pi * yy / 32) * np.cos(2 * np.pi * xx / 32)])
    mask = np.ones_like(smooth, dtype=bool)
    result = S.variogram(smooth, mask, lags=(1, 2, 4, 8))
    values = result["semivariance"]
    assert all(values[i] < values[i + 1] for i in range(len(values) - 1))
