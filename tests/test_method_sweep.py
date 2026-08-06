"""Tests for the simultaneous-DA method sweep (scripts 27-29).

The sampler paths in script 28 need a GPU and a checkpoint, so what is tested
here is everything that decides whether the sweep's *conclusions* are right:
the bias-correction maths, the per-sample CRPS decomposition, the bootstrap and
the bullseye diagnostic.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load(stem: str):
    path = next(ROOT.glob(f"scripts/{stem}*.py"))
    spec = importlib.util.spec_from_file_location(f"_{stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


qm = _load("27_fit_imerg_bias_correction")
summary = _load("29_summarize_method_sweep")


# ---------------------------------------------------------------- quantile map


def test_identity_map_leaves_a_matched_distribution_alone():
    rng = np.random.default_rng(0)
    values = rng.gamma(0.6, 12.0, size=6000).astype(np.float32)
    quantiles = np.linspace(0, 1, 41)
    source_knots, target_knots, cut = qm.fit_quantile_map(values, values, quantiles, 0.1)
    corrected = qm.apply_quantile_map(values, source_knots, target_knots, cut)
    wet = values >= 0.1
    assert np.mean(corrected[wet]) == pytest.approx(np.mean(values[wet]), rel=0.05)


def test_quantile_map_removes_a_multiplicative_wet_bias():
    """A satellite reading 1.8x the reference must come back to the reference."""
    rng = np.random.default_rng(1)
    truth = rng.gamma(0.6, 10.0, size=8000).astype(np.float32)
    biased = (truth * 1.8).astype(np.float32)
    quantiles = np.linspace(0, 1, 41)
    source_knots, target_knots, cut = qm.fit_quantile_map(biased, truth, quantiles, 0.1)
    corrected = qm.apply_quantile_map(biased, source_knots, target_knots, cut)
    raw_bias = float(np.mean(biased - truth))
    corrected_bias = float(np.mean(corrected - truth))
    assert raw_bias > 3.0
    assert abs(corrected_bias) < 0.1 * abs(raw_bias)


def test_frequency_adaptation_matches_the_wet_day_fraction():
    """IMERG's drizzle must be zeroed BEFORE the quantile map, not after."""
    rng = np.random.default_rng(2)
    truth = np.where(rng.random(6000) < 0.4, rng.gamma(0.7, 9.0, 6000), 0.0)
    drizzly = truth + rng.uniform(0.0, 0.6, 6000)      # over-detects light rain
    quantiles = np.linspace(0, 1, 41)
    source_knots, target_knots, cut = qm.fit_quantile_map(drizzly, truth, quantiles, 0.1)
    corrected = qm.apply_quantile_map(drizzly, source_knots, target_knots, cut)
    assert qm.wet_frequency(truth, 0.1) < 0.45      # CHIRPS-like: mostly dry
    assert qm.wet_frequency(drizzly, 0.1) > 0.85    # IMERG-like: rains everywhere
    assert qm.wet_frequency(corrected, 0.1) == pytest.approx(
        qm.wet_frequency(truth, 0.1), abs=0.02
    )


def test_map_extrapolates_the_tail_as_a_ratio_not_a_clip():
    """Clipping the top knot would manufacture a dry bias in extremes."""
    source_knots = np.array([0.0, 10.0, 50.0], np.float32)
    target_knots = np.array([0.0, 8.0, 40.0], np.float32)
    corrected = qm.apply_quantile_map(
        np.array([200.0], np.float32), source_knots, target_knots, 0.0
    )
    assert corrected[0] == pytest.approx(200.0 * 40.0 / 50.0, rel=1e-5)


def test_values_below_the_frequency_cut_become_exactly_zero():
    source_knots = np.array([0.0, 10.0], np.float32)
    target_knots = np.array([0.0, 10.0], np.float32)
    corrected = qm.apply_quantile_map(
        np.array([0.05, 0.4, 5.0], np.float32), source_knots, target_knots, 0.5
    )
    assert corrected[0] == 0.0 and corrected[1] == 0.0
    assert corrected[2] > 0.0


def test_map_preserves_nan_so_missing_footprints_stay_masked():
    source_knots = np.array([0.0, 10.0], np.float32)
    target_knots = np.array([0.0, 10.0], np.float32)
    corrected = qm.apply_quantile_map(
        np.array([np.nan, 5.0], np.float32), source_knots, target_knots, 0.0
    )
    assert np.isnan(corrected[0]) and np.isfinite(corrected[1])


def test_pooled_slice_stays_inside_the_array():
    assert qm.pooled_slice(0, 5, 10) == slice(0, 3)
    assert qm.pooled_slice(9, 5, 10) == slice(7, 10)
    assert qm.pooled_slice(5, 5, 10) == slice(3, 8)


# ------------------------------------------------------------------ CRPS split


def test_per_sample_crps_averages_to_the_library_scalar():
    """The bootstrap is only valid if the per-sample split is the same metric."""
    from bdhires.eval import crps_ensemble

    rng = np.random.default_rng(3)
    members = rng.gamma(0.8, 8.0, size=(7, 16, 5))       # (T, M, S)
    observed = rng.gamma(0.8, 8.0, size=(7, 5))
    per_sample = summary.crps_per_sample(members, observed)
    scalar = crps_ensemble(np.moveaxis(members, 1, 0), observed)
    assert np.nanmean(per_sample) == pytest.approx(scalar, rel=1e-10)


def test_per_sample_crps_marks_missing_observations():
    rng = np.random.default_rng(4)
    members = rng.gamma(0.8, 8.0, size=(4, 8, 3))
    observed = rng.gamma(0.8, 8.0, size=(4, 3))
    observed[1, 2] = np.nan
    per_sample = summary.crps_per_sample(members, observed)
    assert np.isnan(per_sample[1, 2])
    assert np.isfinite(per_sample[0, 0])


def test_crps_is_zero_for_a_perfect_deterministic_ensemble():
    observed = np.array([[3.0, 12.0]])
    members = np.repeat(observed[:, None, :], 9, axis=1)
    assert np.nanmax(np.abs(summary.crps_per_sample(members, observed))) < 1e-9


# ------------------------------------------------------------------- bootstrap


def test_bootstrap_interval_brackets_a_real_difference():
    rng = np.random.default_rng(5)
    difference = rng.normal(0.8, 0.3, size=(40, 6))
    mean, low, high = summary.circular_block_bootstrap(difference, 3, 2000, seed=0)
    assert mean == pytest.approx(float(np.nanmean(difference)))
    assert low < 0.8 < high
    assert low > 0.0, "a 0.8 +/- 0.3 signal over 40 days should exclude zero"


def test_bootstrap_interval_straddles_zero_for_noise():
    rng = np.random.default_rng(6)
    difference = rng.normal(0.0, 1.0, size=(40, 6))
    _, low, high = summary.circular_block_bootstrap(difference, 3, 2000, seed=0)
    assert low < 0.0 < high


def test_short_windows_give_much_wider_intervals_than_long_ones():
    """The five-day sweep must not be able to claim a small CRPS win.

    Asserted as a width ratio rather than as "the interval contains zero": a
    five-day sample can easily have a large sample mean by chance, and the
    interval correctly centres on whatever was observed. What must hold is that
    five days buys a far less precise estimate than a season does.
    """
    rng = np.random.default_rng(7)
    short = rng.normal(0.05, 1.0, size=(5, 8))
    long = rng.normal(0.05, 1.0, size=(80, 8))
    _, short_low, short_high = summary.circular_block_bootstrap(short, 3, 2000, seed=0)
    _, long_low, long_high = summary.circular_block_bootstrap(long, 3, 2000, seed=0)
    assert (short_high - short_low) > 2.5 * (long_high - long_low)
    # A 2.6% CRPS difference is about 0.16 mm/day at these magnitudes; five days
    # cannot resolve it, and the summariser must not pretend otherwise.
    assert (short_high - short_low) > 0.16


def test_bootstrap_tolerates_missing_pairs():
    rng = np.random.default_rng(8)
    difference = rng.normal(0.5, 0.2, size=(20, 4))
    difference[np.array([2, 5]), :] = np.nan
    mean, low, high = summary.circular_block_bootstrap(difference, 3, 500, seed=0)
    assert np.isfinite(mean) and np.isfinite(low) and np.isfinite(high)


# ------------------------------------------------------- torch-dependent parts
#
# Script 28 imports torch at module scope, so on a machine without it the tests
# below are skipped individually -- a module-level importorskip would silently
# skip the numpy tests above too, which are the ones that matter most.

try:
    sweep = _load("28_simultaneous_method_sweep")
except ImportError as exc:  # pragma: no cover - environment dependent
    sweep = None
    _sweep_error = str(exc)
else:
    _sweep_error = ""

needs_sweep = pytest.mark.skipif(
    sweep is None, reason=f"script 28 needs torch: {_sweep_error}"
)


@needs_sweep
def test_every_group_carries_the_background_reference():
    """The increment diagnostic differences against 'background' by name."""
    for group in sweep.GROUPS:
        names = [variant.name for variant in sweep.resolve_variants(group, None)]
        assert "background" in names, group


@needs_sweep
def test_variant_names_are_unique_within_every_group():
    for group, variants in sweep.GROUPS.items():
        names = [variant.name for variant in variants]
        assert len(names) == len(set(names)), group


@needs_sweep
def test_unknown_variant_names_are_rejected():
    with pytest.raises(ValueError, match="unknown variant"):
        sweep.resolve_variants("core", ["not_a_real_variant"])


@needs_sweep
def test_two_step_variants_request_both_streams():
    for variant in sweep.GROUPS["all"]:
        if variant.algorithm == "twostep_ensrf":
            assert variant.uses_imerg and variant.uses_gauges, variant.name


@needs_sweep
def test_increment_locality_detects_a_bullseye():
    """A disc of increment around a gauge must give a large locality ratio."""
    distance = np.tile(np.linspace(0.0, 300.0, 60), (60, 1)).astype(np.float32)
    valid = np.ones_like(distance, dtype=bool)
    background = np.zeros((3, 60, 60), np.float32)
    bullseye = np.exp(-((distance / 40.0) ** 2))[None].repeat(3, axis=0)
    edges = np.array([0, 25, 50, 100, 150, 250, 1e4], dtype=float)
    values, counts = sweep.increment_locality(bullseye, background, distance, valid, edges)
    assert values[0] > 5.0 * values[-2]
    assert counts[0] > 0


@needs_sweep
def test_increment_locality_is_flat_for_a_uniform_increment():
    distance = np.tile(np.linspace(0.0, 300.0, 60), (60, 1)).astype(np.float32)
    valid = np.ones_like(distance, dtype=bool)
    background = np.zeros((3, 60, 60), np.float32)
    uniform = np.full((3, 60, 60), 2.0, np.float32)
    edges = np.array([0, 25, 50, 100, 150, 250, 1e4], dtype=float)
    values, _ = sweep.increment_locality(uniform, background, distance, valid, edges)
    finite = values[np.isfinite(values)]
    assert np.allclose(finite, 2.0, atol=1e-5)


@needs_sweep
def test_jensen_estimators_separate_mean_from_median():
    """Inverting a convex transform over a spread ensemble lifts the mean only."""
    from bdhires.transforms import PrecipTransform

    transform = PrecipTransform(kind="log1p", eps=0.1)
    rng = np.random.default_rng(9)
    transformed = rng.normal(1.0, 0.9, size=(64, 200))
    members = transform.inverse(transformed)
    estimators = sweep.jensen_estimators(members, transform)
    assert np.mean(estimators["mean"]) > np.mean(estimators["median"])
    assert np.mean(estimators["transform_mean"]) < np.mean(estimators["mean"])
