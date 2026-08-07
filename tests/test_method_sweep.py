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
    """Import a numbered script as a module.

    The module MUST be registered in ``sys.modules`` before ``exec_module``.
    On Python 3.10 ``dataclasses._is_type`` resolves a class's module with
    ``sys.modules.get(cls.__module__).__dict__`` while checking for
    ``KW_ONLY``; for a module that was never registered that lookup returns
    ``None`` and every ``@dataclass`` in the file dies with
    ``AttributeError: 'NoneType' object has no attribute '__dict__'``.
    Python 3.12 made the lookup tolerant, so this only bites on the cluster.
    """
    path = next(ROOT.glob(f"scripts/{stem}*.py"))
    name = f"_{stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
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


# ------------------------------------------------------------- stratification


def test_may_and_june_fall_in_different_season_bins():
    """The reason a May-June window is mis-corrected in season mode.

    June lands in JJAS, whose map is fitted mostly on July-August peak-monsoon
    intensities. Applying that to monsoon-onset June mis-scales it, which is
    what degraded the short 2024 holdout.
    """
    may = np.array(["2024-05-15"], dtype="datetime64[D]")
    june = np.array(["2024-06-15"], dtype="datetime64[D]")
    assert qm.stratify(may, "season")[0] == "MAM"
    assert qm.stratify(june, "season")[0] == "JJAS"
    assert qm.stratify(may, "month")[0] == "M05"
    assert qm.stratify(june, "month")[0] == "M06"


def test_strata_for_covers_the_full_cycle():
    assert len(qm.strata_for("season")) == 4
    assert len(qm.strata_for("month")) == 12
    with pytest.raises(ValueError, match="unknown season mode"):
        qm.strata_for("fortnight")


def test_season_of_still_means_season_mode():
    dates = np.array(["2024-05-15", "2024-06-15"], dtype="datetime64[D]")
    assert list(qm.season_of(dates)) == list(qm.stratify(dates, "season"))


def test_apply_rejects_a_stratification_mismatch():
    """Applying a 4-season map with 12 month labels would silently misalign."""
    knots = np.zeros((4, 5, 2, 2), np.float32)
    cuts = np.zeros((4, 2, 2), np.float32)
    with pytest.raises(ValueError, match="strata"):
        qm.apply_map_to_series(
            np.zeros((1, 2, 2), np.float32),
            np.array(["M05"], dtype=object),
            knots, knots, cuts, strata=qm.MONTH_ORDER,
        )


def test_month_stratification_isolates_a_month_dependent_bias():
    """Pooling months only matters when the bias RELATIONSHIP differs by month.

    A monotone bias is recovered by per-cell quantile mapping however the months
    are pooled -- pooling changes knot density, not the shape of the transfer
    function. But when onset and peak-monsoon months are biased differently, a
    map fitted on the pooled distribution splits the difference and mis-corrects
    both. This asserts the fit recovers each month's own relationship when the
    months are kept separate.
    """
    rng = np.random.default_rng(11)
    quantiles = np.linspace(0, 1, 31)
    onset_truth = rng.gamma(0.6, 6.0, 4000)
    peak_truth = rng.gamma(0.6, 22.0, 4000)
    onset_imerg = onset_truth * 1.15 + 2.2          # additive-ish, light rain
    peak_imerg = peak_truth * 1.85                  # multiplicative, deep convection

    separate = []
    for source, target in ((onset_imerg, onset_truth), (peak_imerg, peak_truth)):
        sk, tk, cut = qm.fit_quantile_map(source, target, quantiles, 0.1)
        separate.append(float(np.mean(qm.apply_quantile_map(source, sk, tk, cut) - target)))

    pooled_source = np.concatenate([onset_imerg, peak_imerg])
    pooled_target = np.concatenate([onset_truth, peak_truth])
    sk, tk, cut = qm.fit_quantile_map(pooled_source, pooled_target, quantiles, 0.1)
    pooled_onset_bias = float(
        np.mean(qm.apply_quantile_map(onset_imerg, sk, tk, cut) - onset_truth)
    )

    assert abs(separate[0]) < 0.5, separate
    assert abs(separate[1]) < 2.0, separate
    assert abs(pooled_onset_bias) > 2 * abs(separate[0]) or abs(pooled_onset_bias) > 1.0


def test_load_zarr_time_decodes_the_int64_view_the_packer_writes():
    """Regression test for a real failure on the cluster.

    scripts/04_regrid_and_pack.py writes ``time`` as
    ``datetime64[ns]).view("i8")`` -- a plain int64 array on disk, not a
    datetime64 array. Casting that straight to ``datetime64[D]`` (skipping the
    ``datetime64[ns]`` step) reinterprets nanosecond counts as day counts and
    produces dates thousands of years off, which silently failed every date
    lookup against real 2021-2024 requests. ``load_zarr_time`` must match
    ``bdhires.data.zarr_dataset.PrecipDataset``'s decoding exactly.
    """

    class _FakeStore(dict):
        pass

    dates = np.array(["2021-05-01", "2021-05-02", "2024-06-30"], dtype="datetime64[ns]")
    raw_i8_store = _FakeStore(time=dates.view("i8"))
    decoded = qm.load_zarr_time(raw_i8_store)
    assert decoded.dtype == np.dtype("datetime64[D]")
    assert list(decoded) == list(dates.astype("datetime64[D]"))

    # A store that already holds a real datetime64 array must also work.
    datetime_store = _FakeStore(time=dates)
    assert list(qm.load_zarr_time(datetime_store)) == list(dates.astype("datetime64[D]"))


def test_naive_astype_D_would_have_produced_the_original_bug():
    """Documents the exact failure mode the fix replaces."""
    dates = np.array(["2021-05-01"], dtype="datetime64[ns]")
    raw = dates.view("i8")
    wrong = raw.astype("datetime64[D]")
    assert wrong[0] != dates.astype("datetime64[D]")[0]


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
except Exception as exc:  # pragma: no cover - environment dependent
    # Deliberately broad. A missing torch raises ImportError, but anything else
    # that goes wrong while importing script 28 -- an interpreter-version quirk
    # in dataclasses, a syntax error mid-edit -- must not take the numpy tests
    # above down with it as a COLLECTION error, which is what a narrow
    # ImportError clause allowed to happen.
    sweep = None
    _sweep_error = f"{type(exc).__name__}: {exc}"
else:
    _sweep_error = ""

needs_sweep = pytest.mark.skipif(
    sweep is None, reason=f"script 28 did not import -- {_sweep_error}"
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
def test_fitted_sigma_floor_looks_up_the_measured_intensity_bin():
    """Regression test for the imerg_qm_loyo.json fit (2021 holdout).

    Real fitted sd is ~1.1-1.2 across intensity bins, over 3x the config's
    guessed sigma_obs=0.35. bias_correct variants must use the measured value,
    not silently fall back to the guess, for any bin with enough samples.
    """
    bins = [0.0, 1.0, 5.0, 10.0, 25.0, 50.0, 1e9]
    sigma = [1.170, 1.162, 1.163, 1.188, float("nan"), float("nan")]
    values = np.array([0.05, 0.5, 3.0, 7.0, 30.0, 80.0], np.float32)
    out = sweep.fitted_sigma_floor(values, bins, sigma, fallback=0.35)
    assert out[0] == out[1] == np.float32(1.170)      # both in [0, 1)
    assert out[2] == np.float32(1.162)                # [1, 5)
    assert out[3] == np.float32(1.163)                # [5, 10)
    assert out[4] == 0.35 and out[5] == 0.35          # undersampled bins: fall back
    assert np.all(out[:4] > 3.0 * 0.35), "measured sd must dominate the old guess"


@needs_sweep
def test_fitted_sigma_floor_passes_through_without_a_fit():
    values = np.array([1.0, 2.0], np.float32)
    assert sweep.fitted_sigma_floor(values, None, None, fallback=0.35) == 0.35
    assert sweep.fitted_sigma_floor(values, [], [], fallback=0.35) == 0.35


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
