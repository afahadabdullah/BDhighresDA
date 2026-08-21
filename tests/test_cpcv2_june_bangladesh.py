"""Focused tests for the Bangladesh-only CPCv2 June evaluation."""

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


def test_launcher_uses_saved_stores_and_gpu_node():
    source = (ROOT / "slurm" / "cpcv2_june2023_bangladesh_eval.sbatch").read_text()
    assert "#SBATCH --partition=grace" in source
    assert "#SBATCH --gres=gpu:1" in source
    assert "2023_may_sep.zarr" in source
    assert "v2_simul_s04_ig010" in source
    assert "scripts/28_simultaneous_method_sweep.py" not in source
    assert "--boundary-geojson" in source
