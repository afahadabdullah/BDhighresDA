"""Smoke coverage for the matched V7/CPCv2 spatial diagnostics."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "scripts/74_plot_v7_cpcv2_diagnostics.py"
    spec = importlib.util.spec_from_file_location("v7_cpcv2_diagnostics", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_diagnostics_aligns_reordered_stations_and_crops_cpc_grid(tmp_path):
    diagnostics = _module()
    times = np.array(["2022-05-02", "2022-05-03"])
    ids = np.array(["a", "b", "c", "d"])
    members = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4) + 1.0
    observed = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]])
    cpc_to_v7 = np.array([2, 0, 3, 1])
    cpc_ids = ids[cpc_to_v7]
    v7_path = tmp_path / "v7.npz"
    cpc_path = tmp_path / "cpc.npz"
    map_path = tmp_path / "v7_maps.npz"
    np.savez_compressed(
        v7_path, times=times, station_ids=ids, eval_idx=np.array([2, 3]),
        assim_idx=np.array([0, 1]), observed_mm=observed,
        station_lat=np.array([0.2, 0.3, 0.4, 0.5]),
        station_lon=np.array([0.2, 0.3, 0.4, 0.5]),
        arm_names=np.array(["da_meso", "da_sim", "da_sim_r27_g010_l2"]),
        station_da_meso=members, station_da_sim=members + 1.0,
        station_da_sim_r27_g010_l2=members + 1.5,
    )
    grid = np.arange(2 * 6 * 6, dtype=np.float32).reshape(2, 6, 6) + 1.0
    np.savez_compressed(
        cpc_path, times=times, station_ids=cpc_ids, eval_idx=np.array([0, 2]),
        gauge_mm=observed[:, cpc_to_v7], grid_lat=np.arange(6) * 0.05 + 0.025,
        grid_lon=np.arange(6) * 0.05 + 0.025, valid=np.ones((6, 6), bool),
        station_guided_s6_g010_t100=members[:, :, cpc_to_v7],
        station_v2_simul_s04_ig010=members[:, :, cpc_to_v7] + 1.0,
        meanfield_background=grid,
        meanfield_guided_s6_g010_t100=grid + 1.0,
        meanfield_v2_simul_s04_ig010=grid + 2.0,
    )
    target = np.arange(4) * 0.05 + 0.075
    np.savez_compressed(
        map_path, times=times, grid_lat=target, grid_lon=target,
        valid=np.ones((2, 4, 4), bool), meanfield_background=grid[:, 1:5, 1:5],
        meanfield_da_meso=grid[:, 1:5, 1:5] + 1.0,
        meanfield_da_sim=grid[:, 1:5, 1:5] + 2.0,
    )

    got_times, scores, stations = diagnostics.aligned_station_data(v7_path, cpc_path)
    v7, cpc = diagnostics.map_data(map_path, cpc_path, got_times)
    assert got_times.astype(str).tolist() == ["2022-05-02", "2022-05-03"]
    assert scores["gauges_only"]["v7"]["n"] == [2, 2]
    assert scores["simultaneous_r27_g010_l2"]["v7"]["n"] == [2, 2]
    assert cpc["fields"]["simultaneous"].shape == (2, 4, 4)
    out_dir = tmp_path / "diagnostics"
    out_dir.mkdir()
    diagnostics.plot_skill(got_times, scores, out_dir / "skill.png")
    diagnostics.plot_subgrid_timeseries(got_times, v7, cpc, out_dir / "texture.png")
    diagnostics.plot_day_maps(got_times, v7, cpc, stations, out_dir)
    assert (out_dir / "skill.png").is_file()
    assert (out_dir / "subgrid_maps_2022-05-02.png").is_file()
