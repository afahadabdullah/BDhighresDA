"""Regression tests for the corrected v4 short-window DA diagnostic."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")


def load_script(number: int, name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{number}_{name}.py"
    spec = importlib.util.spec_from_file_location(f"v4_test_script_{number}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_imerg_s04_crop_has_the_frozen_legacy_bd_phase():
    sampler = load_script(60, "v4_subgrid_da_test")
    crop = sampler.legacy_bd_crop((slice(70, 230), slice(60, 220)))
    assert crop == (16, 144, 12, 140)
    assert crop[1] - crop[0] == crop[3] - crop[2] == 128


def test_v4_date_selection_requires_observation_and_condition_days():
    sampler = load_script(60, "v4_subgrid_da_test")
    days = np.asarray(
        ["2022-04-30", "2022-05-01", "2022-05-02"], dtype="datetime64[ns]"
    )
    root = {"time": days.astype(np.int64)}
    observed, conditioned, observed_index, condition_index = sampler.date_indices(
        root, "2022-05-01", "2022-05-02", -1
    )
    assert np.array_equal(
        observed, np.asarray(["2022-05-01", "2022-05-02"], dtype="datetime64[D]")
    )
    assert np.array_equal(
        conditioned, np.asarray(["2022-04-30", "2022-05-01"], dtype="datetime64[D]")
    )
    assert np.array_equal(observed_index, [1, 2])
    assert np.array_equal(condition_index, [0, 1])

    with pytest.raises(ValueError, match="lacks requested"):
        sampler.date_indices(root, "2022-05-02", "2022-05-03", -1)


def test_station_crps_uses_members_and_withheld_observations():
    evaluator = load_script(61, "evaluate_v4_subgrid_da_test")
    truth = np.asarray([[2.0, 5.0], [3.0, 7.0]], np.float32)
    prediction = np.stack([truth - 1.0, truth + 1.0], axis=1)
    metrics = evaluator.ensemble_station_metrics(prediction, truth)
    assert metrics["n"] == 4
    assert metrics["crps_mm_day"] == pytest.approx(0.5)
    assert metrics["rmse_mm_day"] == pytest.approx(0.0)
    assert metrics["bias_mm_day"] == pytest.approx(0.0)
    assert metrics["correlation"] == pytest.approx(1.0)
    assert metrics["coverage90"] == pytest.approx(1.0)


def test_optional_metrics_render_without_invalid_json_values():
    evaluator = load_script(61, "evaluate_v4_subgrid_da_test")
    payload = evaluator.finite_or_none(
        {"missing": np.nan, "valid": np.float32(0.25), "count": np.int64(3)}
    )
    assert payload == {"missing": None, "valid": 0.25, "count": 3}
    assert evaluator.metric_text(None) == "—"
    assert evaluator.metric_text(0.1254) == "0.125"


def test_matrix_plot_accepts_missing_optional_metrics(tmp_path):
    evaluator = load_script(61, "evaluate_v4_subgrid_da_test")
    point = {
        name: {
            "crps_mm_day": 1.0,
            "rmse_mm_day": 2.0,
            "bias_mm_day": 0.0,
            "dry_mae_mm_day": None,
            "wet_mae_mm_day": 3.0,
        }
        for name in evaluator.POINT_METHODS
    }
    structure = {
        name: {
            "chirps_mean_pattern_r": 0.5,
            "chirps_subgrid_pattern_r": None,
            "cpc_pattern_r": 0.6,
            "imerg_pattern_r": 0.7,
        }
        for name in evaluator.MAP_METHODS
    }
    gridded = {
        name: {"crps_mm_day": 1.5, "subgrid_anomaly_crps_mm_day": 0.8}
        for name in evaluator.MAP_METHODS
    }
    output = tmp_path / "matrix.png"
    evaluator.plot_matrix(
        {
            "withheld_gauges": point,
            "spatial_product_agreement": structure,
            "gridded_chirps_agreement": gridded,
        },
        output,
    )
    assert output.stat().st_size > 1_000
