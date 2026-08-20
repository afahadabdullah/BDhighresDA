"""Tests for the V7-only native/S04 ingestion ranking."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "v7_ingestion_ranking", ROOT / "scripts" / "76_rank_v7_imerg_ingestion.py"
)
RANKING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RANKING)


def arm(crps, rmse, spread_skill):
    return {
        "mean": {
            "crps_mm": crps, "mae_mm": crps + 1.0, "bias_mm": 0.1,
            "rmse_mm": rmse, "spread_mm": rmse * spread_skill,
            "spread_skill": spread_skill,
        },
        "days": [{"withheld": {"crps_mm": crps}}],
        "pattern_r": {"imerg_0p1": 0.5},
    }


def test_ranking_selects_s04_and_retains_near_tie():
    results = {
        "members": 16, "n_steps": 50, "imerg_r_multiplier": 9.0,
        "arm_imerg_r": {"da_sim_r27": 27.0, "da_sim_s04_corr_g010_l2": 3.53},
        "arm_imerg_stream": {"da_sim_r27": "native", "da_sim_s04_corr_g010_l2": "s04"},
        "arm_guidance_gamma": {"da_sim_r27": 0.001, "da_sim_s04_corr_g010_l2": 0.01},
        "arm_huber_delta": {"da_sim_r27": 3.0, "da_sim_s04_corr_g010_l2": None},
        "arms": {
            "background": arm(5.0, 8.0, 1.0),
            "da_meso": arm(4.0, 7.0, 0.9),
            "da_sim_r27": arm(3.83, 6.5, 0.7),
            "da_sim_s04_corr_g010_l2": arm(3.80, 6.4, 0.75),
        },
    }
    summary = RANKING.build_summary(results)
    assert summary["winner"] == "da_sim_s04_corr_g010_l2"
    assert summary["overall_da_winner"] == "da_sim_s04_corr_g010_l2"
    assert summary["winner_by_support_deg"] == {
        "0.1": "da_sim_r27", "0.4": "da_sim_s04_corr_g010_l2"
    }
    assert summary["co_winners_within_1pct"] == [
        "da_sim_s04_corr_g010_l2", "da_sim_r27"
    ]
    assert summary["ranking"][0]["support_deg"] == 0.4
    assert summary["ranking"][0]["loss"] == "L2"
    assert abs(summary["winner_minus_gauges_crps_mm"] + 0.2) < 1.0e-12


def test_v7_only_launcher_does_not_run_cpcv2():
    launcher = (ROOT / "slurm" / "v7_imerg_ingestion_sweep.sbatch").read_text()
    assert "scripts/72_v7_two_stage_osse.py" in launcher
    assert "scripts/76_rank_v7_imerg_ingestion.py" in launcher
    assert "scripts/28_simultaneous_method_sweep.py" not in launcher
    assert "cpcv2_station_ensembles" not in launcher
