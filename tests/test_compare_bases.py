"""Regression tests for the CPC/CHIRPS candidate-base comparison."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "13_compare_bases.py"
    spec = importlib.util.spec_from_file_location("compare_bases", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_score_ignores_missing_cpc_pixels_pairwise() -> None:
    module = load_module()
    cpc = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)
    chirps = np.array([[1.0, 2.0], [2.0, 4.0]], dtype=np.float32)

    result = module.score(cpc, chirps, np.ones_like(cpc, dtype=bool), module.PrecipTransform())

    assert result["n"] == 3
    assert np.isfinite(result["correlation"])
    assert np.isfinite(result["transformed_correlation"])
    assert np.isfinite(result["rmse_mm"])


def test_score_reports_no_overlap_without_warnings_or_crash() -> None:
    module = load_module()
    result = module.score(
        np.full((2, 2), np.nan, dtype=np.float32),
        np.ones((2, 2), dtype=np.float32),
        np.ones((2, 2), dtype=bool),
        module.PrecipTransform(),
    )

    assert result["n"] == 0
    assert np.isnan(result["correlation"])
