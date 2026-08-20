"""Scientific guardrails for the V7 below-0.1-degree evaluator."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import numpy as np


def _module():
    root = Path(__file__).parents[1]
    # ``bdhires.eval.__init__`` also exposes torch-based training monitors.  The
    # evaluator itself needs only the NumPy scale module, so keep this unit test
    # runnable in lightweight environments without torch.
    if "bdhires.eval.scale" not in sys.modules:
        package = types.ModuleType("bdhires.eval")
        package.__path__ = []
        sys.modules["bdhires.eval"] = package
        scale_path = root / "src/bdhires/eval/scale.py"
        scale_spec = importlib.util.spec_from_file_location(
            "bdhires.eval.scale", scale_path
        )
        scale = importlib.util.module_from_spec(scale_spec)
        sys.modules["bdhires.eval.scale"] = scale
        assert scale_spec.loader is not None
        scale_spec.loader.exec_module(scale)
        package.scale = scale
    path = root / "scripts/77_evaluate_v7_subgrid_resolution.py"
    spec = importlib.util.spec_from_file_location("v7_subgrid_resolution", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def synthetic_fields():
    rng = np.random.default_rng(4)
    coarse = rng.gamma(2.0, 4.0, (5, 4, 4))
    coarse = np.repeat(np.repeat(coarse, 2, axis=1), 2, axis=2)
    texture = np.tile(np.array([[-2.0, 2.0], [1.0, -1.0]]), (5, 4, 4))
    truth = coarse + texture
    valid = np.ones_like(truth, dtype=bool)
    return truth, coarse, valid


def test_scale_test_rewards_located_subgrid_not_smooth_upsampling():
    module = _module()
    truth, smooth, valid = synthetic_fields()
    evidence = module.scale_evidence(
        {"located": truth.copy(), "smooth": smooth}, truth, valid, factors=(2,)
    )
    located = evidence["located"][0]["below_support_component"]
    smooth_score = evidence["smooth"][0]["below_support_component"]
    assert located["correlation"] > 0.999
    assert located["mse_skill_vs_no_subgrid"] > 0.999
    assert smooth_score["mse_skill_vs_no_subgrid"] == 0.0


def test_within_cell_permutation_rejects_correctly_located_pattern():
    module = _module()
    truth, _, valid = synthetic_fields()
    keep = module.strict_mask(truth, valid, 2)
    residual = module.residuals(truth, keep, 2)
    result = module.placement_permutation_test(
        residual, residual, keep, factor=2, permutations=199, seed=8
    )
    assert result["observed_correlation"] > 0.999
    assert result["correlation_p_one_sided"] <= 0.01
    assert result["mse_p_one_sided"] <= 0.01


def test_daily_bootstrap_does_not_treat_grid_cells_as_independent_days():
    module = _module()
    result = module.day_bootstrap(np.array([0.1, 0.2, 0.3]), 200, 3, block_days=2)
    assert result["n_days"] == 3
    assert result["block_days"] == 2
    assert result["ci_low"] <= result["mean"] <= result["ci_high"]


def test_v7_launcher_runs_r81_subgrid_evaluator():
    root = Path(__file__).parents[1]
    launcher = (root / "slurm/v7_imerg_ingestion_sweep.sbatch").read_text()
    assert "scripts/77_evaluate_v7_subgrid_resolution.py" in launcher
    assert "--arm da_sim_r81" in launcher
    assert "--station-dump \"$V7_DUMP\"" in launcher
