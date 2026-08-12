import ast
import importlib.util
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "_v2_simultaneous_refinement_summary",
    ROOT / "scripts" / "53_summarize_v2_simultaneous_refinement.py",
)
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


def refinement_catalogue() -> ast.List:
    tree = ast.parse((ROOT / "scripts" / "28_simultaneous_method_sweep.py").read_text())
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "V2_SIMULTANEOUS_REFINE"
                for target in node.targets
            )
        ):
            return node.value
    raise AssertionError("V2_SIMULTANEOUS_REFINE catalogue not found")


def fake_fold(group: str, names: list[str], fold: int = 0) -> dict:
    scope = {
        "start": "2022-05-01",
        "end": "2022-05-10",
        "members": 30,
        "checkpoint": "runs/prior_h100_cpc_v2/best.pt",
        "checkpoint_data": "data/processed/bd_wide_cpc.zarr",
        "checkpoint_stats": "data/processed/stats_cpc_v2.json",
        "background_day_offset": -1,
        "seed": 201805,
        "holdout_folds": 5,
        "holdout_fold": fold,
        "group": group,
        "analysis_sampler_n_steps": 50,
        "analysis_sampler_n_corrections": 2,
        "analysis_sampler_heun": True,
        "precip_transform": {"kind": "sqrt"},
        "config_overrides": [
            {"path": "observations.imerg.factor", "value": 8},
            {"path": "observations.imerg.error_corr_cells", "value": 0.75},
        ],
    }
    specs = {name: {} for name in names}
    for steps, name in SUMMARY.ODE_ARMS.items():
        if name in specs:
            specs[name] = {"n_steps": steps, "n_corrections": 0}
    if SUMMARY.OPERATIONAL_N100 in specs:
        specs[SUMMARY.OPERATIONAL_N100] = {"n_steps": 100, "n_corrections": None}
    background = np.arange(12, dtype=np.float32).reshape(1, 3, 2, 2)
    dump = {
        "variant_names": np.asarray(names),
        "times": np.asarray(["2022-05-01"]),
        "station_ids": np.asarray(["A", "B"]),
        "eval_idx": np.asarray([fold % 2]),
        "assim_idx": np.asarray([1 - fold % 2]),
        "station_background": np.zeros((1, 3, 2), dtype=np.float32),
        "meanfield_background": background.mean(axis=1),
        "station_lat": np.asarray([23.0, 24.0]),
        "station_lon": np.asarray([90.0, 91.0]),
        "grid_lat": np.asarray([22.0, 23.0]),
        "grid_lon": np.asarray([89.0, 90.0]),
        "gauge_mm": np.ones((1, 2), dtype=np.float32),
        "condition": np.ones((1, 2, 2), dtype=np.float32),
        "chirps": np.ones((1, 2, 2), dtype=np.float32),
        "raw_imerg_mm": np.ones((1, 1, 1), dtype=np.float32),
        "valid": np.ones((2, 2), dtype=bool),
    }
    report = {
        "scope": scope,
        "variants": {name: {"spec": specs[name]} for name in names},
    }
    return {"fold": fold, "dump": dump, "report": report}


def test_cross_run_pairing_accepts_exact_completed_controls():
    reference = [fake_fold("v2_ingestion_s04", SUMMARY.REFERENCE_NAMES)]
    candidates = [
        fake_fold("v2_simultaneous_refine", SUMMARY.CANDIDATE_NAMES)
    ]
    SUMMARY.validate_cross_run_pairing(reference, candidates)


def test_summary_expected_arms_match_sampler_catalogue():
    catalogue = refinement_catalogue()
    names = [call.args[0].value for call in catalogue.elts]
    assert names == SUMMARY.NEW_ARMS


def test_cross_run_pairing_rejects_changed_background():
    reference = [fake_fold("v2_ingestion_s04", SUMMARY.REFERENCE_NAMES)]
    candidates = [
        fake_fold("v2_simultaneous_refine", SUMMARY.CANDIDATE_NAMES)
    ]
    candidates[0]["dump"]["station_background"][0, 0, 0] = 1.0
    with unittest.TestCase().assertRaisesRegex(ValueError, "paired inputs differ"):
        SUMMARY.validate_cross_run_pairing(reference, candidates)


def test_ode_cost_is_isolated_from_corrector_cost():
    reference = [fake_fold("v2_ingestion_s04", SUMMARY.REFERENCE_NAMES)]
    candidates = [
        fake_fold("v2_simultaneous_refine", SUMMARY.CANDIDATE_NAMES)
    ]
    current = SUMMARY.effective_sampler(reference, SUMMARY.CURRENT)
    n25 = SUMMARY.effective_sampler(candidates, SUMMARY.ODE_ARMS[25])
    n50 = SUMMARY.effective_sampler(candidates, SUMMARY.ODE_ARMS[50])
    n100 = SUMMARY.effective_sampler(candidates, SUMMARY.ODE_ARMS[100])
    operational_n100 = SUMMARY.effective_sampler(
        candidates, SUMMARY.OPERATIONAL_N100
    )

    assert current == {
        "n_steps": 50,
        "n_corrections_per_level": 2,
        "heun": True,
        "integration_guidance_evaluations": 99,
        "corrector_guidance_evaluations": 98,
        "total_guidance_evaluations": 197,
    }
    assert [n25["total_guidance_evaluations"], n50["total_guidance_evaluations"],
            n100["total_guidance_evaluations"]] == [49, 99, 199]
    assert all(
        item["corrector_guidance_evaluations"] == 0
        for item in (n25, n50, n100)
    )
    assert operational_n100["total_guidance_evaluations"] == 397
