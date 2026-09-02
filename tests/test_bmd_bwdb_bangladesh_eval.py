"""Focused tests for the BMD+BWDB Bangladesh evaluation script and launcher."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def module():
    path = ROOT / "scripts" / "85_evaluate_bmd_bwdb_bangladesh.py"
    spec = importlib.util.spec_from_file_location("bmd_bwdb_bangladesh_eval", path)
    loaded = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(loaded)
    return loaded


def test_default_method_and_network_label():
    evaluator = module()
    assert evaluator.METHOD_DEFAULT == "v2_simul_s04_huber3"
    assert "BMD+BWDB" in evaluator.SOURCE_LABELS["gauges"]
    assert "BMD+BWDB" in evaluator.EVIDENCE_ROLES["gauges"]


def test_country_mask_obeys_polygon_and_hole():
    evaluator = module()
    geometry = {
        "type": "Polygon",
        "coordinates": [
            [[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]],
            [[1, 1], [3, 1], [3, 3], [1, 3], [1, 1]],
        ],
    }
    mask = evaluator.country_mask(np.array([0.5, 2.0, 3.5]),
                                  np.array([0.5, 2.0, 3.5]), geometry)
    assert mask[0, 0]
    assert not mask[1, 1]
    assert mask[2, 2]
    points = evaluator.points_in_country(np.array([0.5, 2.0, 5.0]),
                                         np.array([0.5, 2.0, 5.0]), geometry)
    assert points.tolist() == [True, False, False]


def test_field_metrics_are_zero_for_identical_fields():
    evaluator = module()
    field = np.arange(16, dtype=float).reshape(4, 4)
    metrics = evaluator.field_metrics(field, field, np.ones((4, 4), bool))
    assert metrics["correlation"] == 1.0
    assert metrics["bias_mm"] == 0.0
    assert metrics["rmse_mm"] == 0.0
    assert metrics["spatial_sd_ratio"] == 1.0


def test_area_regrid_closes_on_native_cells():
    evaluator = module()
    source_lat = np.array([0.25, 0.75, 1.25, 1.75])
    source_lon = source_lat.copy()
    target_lat = np.array([0.5, 1.5])
    target_lon = target_lat.copy()
    values = np.arange(16, dtype=float).reshape(1, 4, 4)
    result = evaluator.regrid_cell_average(
        values, source_lat, source_lon, target_lat, target_lon,
        np.ones((4, 4), bool),
    )
    assert result.shape == (1, 2, 2)
    assert np.allclose(result[0, 0, 0], np.mean(values[0, :2, :2]), atol=0.02)
    assert np.allclose(result[0, 1, 1], np.mean(values[0, 2:, 2:]), atol=0.02)


def test_contiguous_period_selection_and_labels():
    evaluator = module()

    class Args:
        month = 6
        months = [5, 6, 7, 8]

    months = evaluator.evaluation_months(Args())
    assert months == (5, 6, 7, 8)
    assert evaluator.period_label(months) == "May–August"
    assert evaluator.period_tag(months) == "may_jun_jul_aug"
    assert evaluator.expected_period_days(2023, months) == 123


def test_daily_network_rows_preserve_dates_and_gauge_label():
    evaluator = module()
    bundle = {
        "observed": np.array([[2.0, 4.0]]),
        "members": np.array([[[1.0, 3.0], [3.0, 5.0]]]),
        "products": {"chirps": np.array([[1.5, 3.5]])},
    }
    rows = evaluator.gauge_daily_network_rows(
        bundle,
        np.array([np.datetime64("2023-05-31")]),
        np.array([np.datetime64("2023-06-01")]),
    )
    assert {row["source"] for row in rows} == {"gauges", "analysis", "chirps"}
    assert all(row["product_date"] == "2023-05-31" for row in rows)
    assert all(row["gauge_observation_date"] == "2023-06-01" for row in rows)
    assert next(row for row in rows if row["source"] == "gauges")["source_archive_date"] == "2023-06-01"
    assert next(row for row in rows if row["source"] == "chirps")["source_archive_date"] == "2023-05-31"
    analysis = next(row for row in rows if row["source"] == "analysis")
    assert analysis["network_mean_mm"] == 3.0


def test_launcher_references_bwdb_and_huber3():
    sbatch_src = (ROOT / "slurm" / "bmd_bwdb_bangladesh_eval.sbatch").read_text()
    assert "v2_bmd_bwdb_huber3_2021_2024" in sbatch_src
    assert "v2_simul_s04_huber3" in sbatch_src
    assert "scripts/85_evaluate_bmd_bwdb_bangladesh.py" in sbatch_src
    assert "--boundary-geojson" in sbatch_src
    assert "#SBATCH --partition=grace" in sbatch_src

    submit_src = (ROOT / "slurm" / "submit_bmd_bwdb_bangladesh_eval.sh").read_text()
    assert "v2_bmd_bwdb_huber3_2021_2024" in submit_src
    assert "slurm/bmd_bwdb_bangladesh_eval.sbatch" in submit_src


def test_data_checker_fair_crps():
    checker_path = ROOT / "scripts" / "86_check_bmd_bwdb_2021_2024_archive.py"
    spec = importlib.util.spec_from_file_location("bmd_bwdb_checker", checker_path)
    loaded = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(loaded)

    # Perfect ensemble matching truth should have fair CRPS = 0
    members = np.ones((5, 30)) * 4.0
    truth = np.ones(5) * 4.0
    crps = loaded.fair_crps_per_sample(members, truth)
    assert np.allclose(crps, 0.0)


def test_checker_sbatch_references():
    sbatch_src = (ROOT / "slurm" / "bmd_bwdb_data_checker.sbatch").read_text()
    assert "86_check_bmd_bwdb_2021_2024_archive.py" in sbatch_src
    assert "v2_bmd_bwdb_huber3_2021_2024" in sbatch_src

