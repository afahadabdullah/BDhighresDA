"""End-to-end tests for the OSSE downscaling evaluator and paper suite.

The scale primitives themselves live in ``bdhires.eval.scale`` and are covered
by ``test_scale_metrics.py``.  What is tested here is the part a unit test of
the primitives cannot reach: that the scripts assemble the right *claims* from a
dump, that claim A degrades loudly on a dump predating ``coarse_base_mm``, and
that every artifact the manuscript depends on is actually written.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EVAL = _load("_osse_downscaling", "22_evaluate_osse_downscaling.py")
SUITE = _load("_osse_paper_suite", "24_osse_paper_suite.py")
MERGE = _load("_osse_chunk_merge", "25_merge_osse_chunks.py")


# --------------------------------------------------------------------------
# Synthetic dump: a known-good downscaler over a known coarse input
# --------------------------------------------------------------------------

def _block_upsample(field: np.ndarray, factor: int) -> np.ndarray:
    valid = np.isfinite(field)
    shape = (field.shape[0], field.shape[1] // factor, factor,
             field.shape[2] // factor, factor)
    sums = np.where(valid, field, 0.0).reshape(shape).sum((2, 4))
    counts = valid.reshape(shape).sum((2, 4))
    mean = np.divide(sums, counts, out=np.full(sums.shape, np.nan), where=counts > 0)
    return np.repeat(np.repeat(mean, factor, 1), factor, 2)


def make_dump(path: Path, *, with_coarse: bool = True, skill: float = 0.7,
              days: int = 6, members: int = 4, size: int = 32) -> Path:
    rng = np.random.default_rng(19)
    yy, xx = np.mgrid[0:size, 0:size]
    truth = np.stack([
        np.clip(6 * np.exp(-((yy - size / 2) ** 2 + (xx - size / 2) ** 2) / 90)
                + 2 * np.sin(2 * np.pi * (yy + d) / 6)
                + rng.normal(0, 1, (size, size)), 0, None)
        for d in range(days)
    ])
    valid = np.ones((days, size, size), dtype=bool)
    valid[:, :4, :4] = False
    truth[~valid] = np.nan

    coarse = _block_upsample(truth, 8)
    background = np.stack([
        coarse + skill * (truth - coarse) + rng.normal(0, 0.5, truth.shape)
        for _ in range(members)
    ])
    footprint = _block_upsample(truth, 2)
    analysis = np.stack([
        m - _block_upsample(m, 2) + footprint + rng.normal(0, 0.2, truth.shape)
        for m in background
    ])
    for stack in (background, analysis):
        stack[:, ~valid] = np.nan

    # Match the real dump emitted by scripts/10_osse.py: day first, then member.
    payload = dict(
        background=np.moveaxis(background, 0, 1).astype(np.float32),
        analysis=np.moveaxis(analysis, 0, 1).astype(np.float32),
        truth=truth.astype(np.float32),
        array_layout=np.str_("day,member,latitude,longitude"),
        valid=valid,
        satellite_factor=np.int32(2),
        pseudo_satellite_enabled=np.bool_(True),
        days=np.array([f"2021-07-{d + 1:02d}" for d in range(days)]),
        network=np.str_("40"),
        obs_error=np.str_("realistic"),
        observation_mode=np.str_("combined"),
        checkpoint=np.str_("synthetic"),
        grid_lat=np.linspace(20.3, 26.7, size).astype(np.float32),
        grid_lon=np.linspace(87.6, 94.0, size).astype(np.float32),
        station_lat=np.linspace(21, 26, 6).astype(np.float32),
        station_lon=np.linspace(88, 93, 6).astype(np.float32),
        assim_idx=np.arange(4), eval_idx=np.arange(4, 6),
    )
    if with_coarse:
        payload["coarse_base_mm"] = coarse.astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return path


@pytest.fixture
def dump(tmp_path) -> dict:
    return EVAL.load_dump(make_dump(tmp_path / "ensemble.npz"))


def test_production_dump_is_normalized_to_member_first(tmp_path):
    loaded = EVAL.load_dump(make_dump(
        tmp_path / "ensemble.npz", days=6, members=4))
    assert loaded["background"].shape[:2] == (4, 6)
    assert loaded["analysis"].shape[:2] == (4, 6)
    assert loaded["truth"].shape[0] == 6


# --------------------------------------------------------------------------
# Claim structure
# --------------------------------------------------------------------------

def test_claims_are_scored_against_their_own_nulls(dump):
    report, *_ = EVAL.evaluate_claims(dump, factor=2, minimum_valid_fraction=1.0)
    claim_a = report["claim_a_downscaling_gain"]
    claim_b = report["claim_b_sub_footprint_gain"]
    assert claim_a["scored_on"] == "full field"
    assert "footprint mean" in claim_b["null_model"]
    # A downscaler that recovers 70% of the fine detail must beat both nulls.
    assert claim_a["background"]["mse_skill"] > 0.2
    assert claim_b["analysis"]["mse_skill"] > 0.2


def test_claim_a_reports_background_not_analysis(dump):
    """Claim A must not be able to borrow skill from assimilated observations."""
    report, *_ = EVAL.evaluate_claims(dump, factor=2, minimum_valid_fraction=1.0)
    claim_a = report["claim_a_downscaling_gain"]
    assert "background" in claim_a
    assert "analysis_for_reference" in claim_a
    assert "analysis" not in claim_a


def test_missing_coarse_base_degrades_loudly(tmp_path):
    """An older dump must disable claim A, never silently mislabel it."""
    path = make_dump(tmp_path / "old.npz", with_coarse=False)
    report, *_ = EVAL.evaluate_claims(EVAL.load_dump(path), 2, 1.0)
    claim_a = report["claim_a_downscaling_gain"]
    assert "unavailable" in claim_a
    assert "rerun" in claim_a["unavailable"]
    assert report["claim_b_sub_footprint_gain"]["analysis"]["mse_skill"] > 0.0


def test_analysis_beats_background_at_footprint_scale(dump):
    """Sanity check: the analysis was nudged onto the true footprint means."""
    report, *_ = EVAL.evaluate_claims(dump, factor=2, minimum_valid_fraction=1.0)
    component = report["footprint_component"]
    assert component["analysis"]["rmse_mm"] < component["background"]["rmse_mm"]


def test_stratifications_are_populated(dump):
    report, *_ = EVAL.evaluate_claims(dump, factor=2, minimum_valid_fraction=1.0)
    assert report["by_intensity"], "intensity stratification is empty"
    assert set(report["by_year"]) == {"2021"}


def test_missing_required_array_is_rejected(tmp_path):
    path = tmp_path / "broken.npz"
    np.savez(path, truth=np.zeros((1, 4, 4)))
    with pytest.raises(SystemExit):
        EVAL.load_dump(path)


def test_chunk_exactness_diagnostics_merge_by_worst_case():
    """Per-month round-off maxima vary and must not be static metadata."""
    values = [np.float32(0.0), np.float32(2.4e-6), np.float32(8.0e-7)]
    assert MERGE.finite_max(values) == pytest.approx(2.4e-6)
    assert "exact_satellite_max_abs_error_mm" in MERGE.MAX_DIAGNOSTICS
    assert "exact_gauge_max_abs_error_transformed" in MERGE.MAX_DIAGNOSTICS


# --------------------------------------------------------------------------
# Structure diagnostics
# --------------------------------------------------------------------------

def test_structure_reports_every_diagnostic(dump):
    _, mask, *_ = EVAL.evaluate_claims(dump, 2, 1.0)
    structure = EVAL.evaluate_structure(dump, mask, fine_degrees=0.05)
    assert structure["spectra"]["effective_resolution_km"]
    assert len(structure["scale_ladder"]["degrees"]) == len(EVAL.LADDER_FACTORS)
    assert "analysis_mean" in structure["fss"]
    assert "truth" in structure["variogram"]


def test_coarse_input_has_the_worst_effective_resolution(dump):
    """The null must look blurrier than the model, or the metric is inverted."""
    _, mask, *_ = EVAL.evaluate_claims(dump, 2, 1.0)
    effective = EVAL.evaluate_structure(
        dump, mask, 0.05)["spectra"]["effective_resolution_km"]
    assert effective["coarse_input"] > effective["analysis_member"]


# --------------------------------------------------------------------------
# Artifacts
# --------------------------------------------------------------------------

def test_cli_writes_every_manuscript_artifact(tmp_path):
    dump_path = make_dump(tmp_path / "ensemble.npz")
    out = tmp_path / "out"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/22_evaluate_osse_downscaling.py"),
         "--dump", str(dump_path), "--skip-figure",
         "--out-report", str(out / "downscaling.json"),
         "--out-curve-data", str(out / "curves.npz"),
         "--out-spatial-data", str(out / "spatial.nc")],
        capture_output=True, text=True, env=environment,
    )
    assert result.returncode == 0, result.stderr
    assert (out / "downscaling.json").exists()
    assert (out / "curves.npz").exists()

    report = json.loads((out / "downscaling.json").read_text())
    assert report["claim_b_sub_footprint_gain"]["analysis"]["mse_skill"] > 0.0

    with np.load(out / "curves.npz") as curves:
        assert "spectra_wavelength_km" in curves
        assert any(k.startswith("fss__") for k in curves.files)
        assert any(k.startswith("ladder_") for k in curves.files)


def test_netcdf_export_is_georeferenced(tmp_path):
    xr = pytest.importorskip("xarray")
    dump = EVAL.load_dump(make_dump(tmp_path / "ensemble.npz"))
    _, mask, truth_sub, bg_sub, an_sub = EVAL.evaluate_claims(dump, 2, 1.0)
    target = tmp_path / "spatial.nc"
    EVAL.save_spatial_data(target, dump, mask, 2, truth_sub, bg_sub, an_sub)
    with xr.open_dataset(target) as ds:
        assert {"truth_subgrid_mm", "analysis_subgrid_mean_mm",
                "coarse_input_mm"} <= set(ds.data_vars)
        assert ds.latitude.min() > 20 and ds.latitude.max() < 27
        assert ds.station_role.values.tolist() == [1, 1, 1, 1, 2, 2]


# --------------------------------------------------------------------------
# Paper suite
# --------------------------------------------------------------------------

def test_calibration_columns_rank_by_distance_to_target():
    """Spread/skill of 1.4 must not outrank 1.0 just because it is larger."""
    values = np.array([0.5, 1.0, 1.4])
    assert SUITE._best_row(values, 1.0) == 1
    assert SUITE._best_row(values, "max") == 2
    assert SUITE._best_row(values, "min") == 0


def test_observation_value_uses_withheld_gauges_and_best_single_source(tmp_path):
    def scores(analysis_crps: float) -> dict:
        return {
            "background": {"crps_mm": 10.0, "rmse_mm": 14.0, "mae_mm": 8.0,
                           "bias_mm": 2.0, "correlation": 0.30,
                           "spread_skill_ratio": 0.70, "coverage_90": 0.80},
            "analysis": {"crps_mm": analysis_crps,
                         "rmse_mm": 10.0 + analysis_crps / 10,
                         "mae_mm": 6.0, "bias_mm": 1.0,
                         "correlation": 0.60,
                         "spread_skill_ratio": 0.90, "coverage_90": 0.88},
        }

    labels_and_crps = [
        ("gauges_exact_bmd", 6.0),
        ("satellite_exact_bmd", 5.0),
        ("simultaneous_exact_bmd", 4.0),
    ]
    arms = []
    for label, crps in labels_and_crps:
        arms.append({
            "label": label, "pretty": SUITE.PRETTY[label],
            "scale": {scope: scores(crps) for scope, _ in SUITE.SCALE_SCOPES},
        })

    selected = SUITE.select_observation_arms(arms)
    matrix = SUITE.build_observation_value_matrix(selected)
    assert matrix.shape == (3, len(SUITE.OBSERVATION_VALUE_COLUMNS))
    assert matrix[2, 0] == pytest.approx(60.0)  # simultaneous withheld CRPSS
    assert SUITE.combined_synergy(selected, "withheld_gauges") == pytest.approx(20.0)

    SUITE.write_observation_value_data(
        tmp_path / "observation_value.csv",
        tmp_path / "observation_value.json",
        selected,
    )
    payload = json.loads((tmp_path / "observation_value.json").read_text())
    assert payload["primary_target"].startswith("withheld pseudo-gauges")
    assert payload["simultaneous_synergy_percent"]["withheld_gauges"] == pytest.approx(20.0)


def test_50_gauge_density_triplet_is_selected_and_labelled():
    arms = [
        {"label": label, "scale": {"withheld_gauges": {}}}
        for label in (
            "gauges_exact_50", "satellite_exact_50", "simultaneous_exact_50"
        )
    ]
    selected = SUITE.select_observation_arms(arms)
    assert [arm["label"] for arm in selected] == [
        "gauges_exact_50", "satellite_exact_50", "simultaneous_exact_50"
    ]
    assert SUITE.observation_target_description(selected) == (
        "withheld pseudo-gauges at 10 of 50 spread locations"
    )


def test_spread_skill_is_derived_for_older_scale_summaries():
    scores = {"analysis": {"spread_mm": 4.0, "rmse_mm": 5.0}}
    assert SUITE._spread_skill(scores) == pytest.approx(0.8)


def test_suite_builds_all_paper_artifacts(tmp_path):
    root = tmp_path / "osse_paper"
    for label, skill in (("gauges_realistic_40", 0.4),
                         ("simultaneous_realistic_40", 0.75)):
        directory = root / label
        make_dump(directory / "ensemble.npz", skill=skill)
        dump = EVAL.load_dump(directory / "ensemble.npz")
        report, mask, *_ = EVAL.evaluate_claims(dump, 2, 1.0)
        report["structure"] = EVAL.evaluate_structure(dump, mask, 0.05)
        report["network"] = "40"
        (directory / "downscaling.json").write_text(
            json.dumps(report, indent=2, default=float))
        EVAL.save_curve_data(directory / "downscaling_curves.npz",
                             report["structure"])
        (directory / "osse_report.json").write_text(json.dumps({"results": [{
            "network": "40",
            "withheld_background": {"crps_mm": 6.9, "spread_skill": 0.6,
                                    "coverage_90": 0.8},
            "withheld_analysis": {"crps_mm": 6.9 * (1 - skill),
                                  "spread_skill": 0.9, "coverage_90": 0.85},
            "withheld_improvement_crps_mm": skill * 100,
            "field_improvement_crps_mm": skill * 60}]}))

    arms = [SUITE.load_arm(label, path)
            for label, path in SUITE.discover_arms(root, None)]
    assert [a["label"] for a in arms] == ["gauges_realistic_40",
                                          "simultaneous_realistic_40"]

    out = root / "paper"
    out.mkdir(parents=True, exist_ok=True)
    matrix = SUITE.build_matrix(arms, SUITE.DOWNSCALING_COLUMNS, "downscaling")
    SUITE.write_latex(out / "table.tex", arms, SUITE.DOWNSCALING_COLUMNS,
                      matrix, "caption", "tab:x")
    SUITE.write_tidy_csv(out / "metrics.csv", arms)
    SUITE.write_combined_curves(out / "curves.npz", arms)
    SUITE.write_results_markdown(out / "RESULTS.md", arms,
                                 "simultaneous_realistic_40", matrix, matrix)

    latex = (out / "table.tex").read_text()
    assert r"\begin{table*}" in latex and r"\textbf" in latex
    # The better arm must win the headline claim-A column.
    assert matrix[1, 0] > matrix[0, 0]

    csv_text = (out / "metrics.csv").read_text()
    assert "claim_a_downscaling_gain.background.mse_skill" in csv_text
    assert "by_intensity" in csv_text

    summary = (out / "RESULTS.md").read_text()
    assert "Downscaling gain" in summary and "Sub-footprint gain" in summary
    assert "CHIRPS supplies both the nature truth" in summary

    with np.load(out / "curves.npz") as curves:
        assert any(k.startswith("simultaneous_realistic_40::") for k in curves.files)


def test_empty_root_is_rejected(tmp_path):
    with pytest.raises(SystemExit):
        SUITE.discover_arms(tmp_path, None)
