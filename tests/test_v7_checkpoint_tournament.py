"""Tests for the matched V7 checkpoint-pair tournament."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "scripts/78_rank_v7_checkpoint_pairs.py"
    spec = importlib.util.spec_from_file_location("v7_checkpoint_pairs", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_fair_station_crps_rewards_perfect_ensemble():
    module = _module()
    truth = np.array([1.0, 4.0, 9.0])
    perfect = np.broadcast_to(truth, (4, 3)).copy()
    displaced = perfect + 2.0
    assert np.allclose(module.fair_crps_per_station(perfect, truth), 0.0)
    assert np.all(module.fair_crps_per_station(displaced, truth) > 0.0)


def test_paired_bootstrap_uses_candidate_minus_reference_sign():
    module = _module()
    result = module.paired_bootstrap(
        np.array([1.0, 1.2, 1.4]), np.array([2.0, 2.2, 2.4]), 500, 8
    )
    assert result["candidate_minus_reference_crps_mm"] == -1.0
    assert result["ci_high"] < 0.0


def test_launcher_is_four_pair_factorial_with_fixed_r81():
    root = Path(__file__).parents[1]
    source = (root / "slurm/v7_checkpoint_tournament_may03.sbatch").read_text()
    for label in ("frozen_frozen", "latest_frozen", "frozen_latest", "latest_latest"):
        assert f"run_pair {label}" in source
    assert "--imerg-r-only 81" in source
    assert "--imerg-r-sweep 81" not in source
    assert "--seed \"$SEED\"" in source
    assert "--members \"$MEMBERS\" --n-steps \"$STEPS\"" in source


def test_ranker_refuses_missing_seed_or_split_drift():
    module = _module()
    baseline = {
        "report": {
            "model_dates": ["2022-05-03"], "gauge_dates": ["2022-05-04"],
            "members": 16, "n_steps": 50, "seed": 20220503,
            "observations": "real", "gauge_day_offset": 1, "imerg_day_offset": 1,
            "meso_gauge_sigma_transformed": .1,
            "meso_gauge_representativeness": .25, "fine_gauge_sigma_mm": 3,
            "arm_imerg_r": {"da_sim_r81": 81},
        },
        "arrays": {
            "times": np.array(["2022-05-04"]),
            "model_times": np.array(["2022-05-03"]),
            "station_ids": np.array(["a", "b"]),
            "eval_idx": np.array([1]), "assim_idx": np.array([0]),
            "observed_mm": np.array([[1.0, 2.0]]),
        },
    }
    runs = {label: {**baseline, "report": dict(baseline["report"]),
                    "arrays": dict(baseline["arrays"])} for label in module.LABELS}
    runs["latest_latest"]["report"]["seed"] = 99
    try:
        module.validate_matched(runs)
    except ValueError as error:
        assert "seed" in str(error)
    else:
        raise AssertionError("seed drift was accepted")
