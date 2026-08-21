"""Focused tests for the BRISHTI-05 Bangladesh June evaluation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def module():
    path = ROOT / "scripts" / "81_evaluate_cpcv2_june_bangladesh.py"
    spec = importlib.util.spec_from_file_location("cpcv2_june_bangladesh", path)
    loaded = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(loaded)
    return loaded


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


def test_bmd_date_contract_comes_from_frozen_archive():
    evaluator = module()

    class Dataset:
        attrs = {"scope": {"background_day_offset": -1}}

    assert evaluator.background_day_offset({"datasets": [Dataset(), Dataset()]}) == -1


def test_daily_network_rows_preserve_product_and_bmd_dates():
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
    assert {row["source"] for row in rows} == {"bmd", "analysis", "chirps"}
    assert all(row["product_date"] == "2023-05-31" for row in rows)
    assert all(row["bmd_observation_date"] == "2023-06-01" for row in rows)
    assert next(row for row in rows if row["source"] == "bmd")["source_archive_date"] == "2023-06-01"
    assert next(row for row in rows if row["source"] == "chirps")["source_archive_date"] == "2023-05-31"
    analysis = next(row for row in rows if row["source"] == "analysis")
    assert analysis["network_mean_mm"] == 3.0


def test_launcher_uses_saved_stores_and_gpu_node():
    source = (ROOT / "slurm" / "cpcv2_june2023_bangladesh_eval.sbatch").read_text()
    assert "#SBATCH --partition=grace" in source
    assert "#SBATCH --gres=gpu:1" in source
    assert "2023_may_sep.zarr" in source
    assert "v2_simul_s04_ig010" in source
    assert "imerg_native/2023_may_sep.nc" in source
    assert "precip.2023.nc" in source
    assert "BRISHTI-05" in source
    assert "scripts/28_simultaneous_method_sweep.py" not in source
    assert "--boundary-geojson" in source
    assert "--native-imerg" in source
    assert "--cpc-dir" in source
    assert "2023-05-31 through 2023-06-29" in source
    assert "2023-06-01 through 2023-06-30 (+1 day)" in source
    assert "--cpc-source-zarr" not in source
