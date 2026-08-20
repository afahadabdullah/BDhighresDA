"""Unit tests for the raw-output guardrail in the V7/CPCv2 day comparison."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "v7_cpcv2_comparison", ROOT / "scripts" / "73_compare_v7_cpcv2_day.py"
)
COMPARISON = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARISON)


def write_matched_dumps(
    tmp_path: Path, *, changed_observation: bool = False,
    extra_v7_simultaneous: bool = False,
):
    """Create matching dumps with deliberately different station ordering."""
    v7_path = tmp_path / "v7.npz"
    cpc_path = tmp_path / "cpcv2.npz"
    observations = np.asarray([[1.0, 2.0, 3.0]], np.float32)  # A, B, C
    v7_base = np.asarray([[[1.0, 2.2, 3.0], [1.1, 2.0, 2.9], [0.9, 1.8, 3.1]]])
    cpc_base = np.asarray([[[3.0, 2.4, 1.1], [3.1, 2.2, 0.9], [2.9, 2.1, 1.0]]])
    np.savez_compressed(
        v7_path,
        times=np.asarray(["2022-05-04"]),
        model_times=np.asarray(["2022-05-03"]),
        station_ids=np.asarray(["A", "B", "C"]),
        station_lat=np.asarray([23.0, 24.0, 25.0]),
        station_lon=np.asarray([90.0, 91.0, 92.0]),
        eval_idx=np.asarray([1]),
        assim_idx=np.asarray([0, 2]),
        observed_mm=observations,
        station_da_meso=v7_base,
        station_da_sim=v7_base + 0.05,
        **(
            {
                "station_da_sim_r27": v7_base + 0.10,
                "station_da_sim_r81": v7_base + 0.15,
            }
            if extra_v7_simultaneous else {}
        ),
    )
    cpc_observed = observations[:, [2, 1, 0]].copy()
    if changed_observation:
        cpc_observed[0, 1] += 1.0
    np.savez_compressed(
        cpc_path,
        times=np.asarray(["2022-05-04"]),
        model_times=np.asarray(["2022-05-03"]),
        station_ids=np.asarray(["C", "B", "A"]),
        station_lat=np.asarray([25.0, 24.0, 23.0]),
        station_lon=np.asarray([92.0, 91.0, 90.0]),
        eval_idx=np.asarray([1]),
        assim_idx=np.asarray([0, 2]),
        gauge_mm=cpc_observed,
        station_guided_s6_g010_t100=cpc_base,
        station_v2_simul_s04_ig010=cpc_base + 0.05,
    )
    return v7_path, cpc_path


def test_comparison_aligns_station_order_before_scoring(tmp_path):
    v7_path, cpc_path = write_matched_dumps(tmp_path)
    report = COMPARISON.compare_dumps(v7_path, cpc_path)

    assert report["scope"]["observation_dates"] == ["2022-05-04"]
    assert report["scope"]["withheld_station_ids"] == ["B"]
    assert report["scope"]["withheld_station_days"] == 1
    assert set(report["comparisons"]) == {"gauges_only", "simultaneous"}
    for result in report["comparisons"].values():
        assert result["v7"]["station_days"] == 1
        assert np.isfinite(result["v7_minus_cpcv2"]["crps_mm"])


def test_comparison_refuses_different_bmd_values(tmp_path):
    v7_path, cpc_path = write_matched_dumps(tmp_path, changed_observation=True)
    with pytest.raises(ValueError, match="BMD values differ"):
        COMPARISON.compare_dumps(v7_path, cpc_path)


def test_comparison_scores_optional_v7_r_sweep_against_same_cpc_arm(tmp_path):
    v7_path, cpc_path = write_matched_dumps(tmp_path, extra_v7_simultaneous=True)
    report = COMPARISON.compare_dumps(v7_path, cpc_path)

    assert list(report["comparisons"]) == [
        "gauges_only", "simultaneous", "simultaneous_r27", "simultaneous_r81"
    ]
    assert report["comparisons"]["simultaneous_r27"]["v7_arm"] == "da_sim_r27"
    assert report["comparisons"]["simultaneous_r27"]["cpcv2_arm"] == (
        "v2_simul_s04_ig010"
    )


def test_cpcv2_comparison_group_contains_only_the_two_selected_winners():
    source = (ROOT / "scripts" / "28_simultaneous_method_sweep.py").read_text()
    assert '"v2_comparison": V2_COMPARISON' in source
    group = COMPARISON.COMPARISONS
    assert group["gauges_only"] == ("da_meso", "guided_s6_g010_t100")
    assert group["simultaneous"] == ("da_sim", "v2_simul_s04_ig010")
