from __future__ import annotations

import json
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _method_score(crps: float) -> dict:
    return {
        "n": 12,
        "crps_mm": crps,
        "rmse_mm": crps * 2,
        "mae_mm": crps * 1.4,
        "bias_mm": crps * 0.2,
        "correlation": 0.6,
        "spread_skill": 0.9,
        "coverage_90": 0.8,
        "thresholds": {
            threshold: {"brier_score": 0.1 + 0.01 * position}
            for position, threshold in enumerate(("1", "10", "25", "50"))
        },
    }


def test_fullmonth_diagnostics_render_from_rotated_folds(tmp_path: Path):
    if importlib.util.find_spec("cartopy") is None:
        return
    rng = np.random.default_rng(4)
    days, members, height, width, stations = 6, 4, 8, 8, 10
    grid_lat = np.linspace(20.5, 24.0, height)
    grid_lon = np.linspace(88.0, 91.5, width)
    station_lat = np.linspace(20.7, 23.8, stations)
    station_lon = np.linspace(88.2, 91.3, stations)
    rows = np.clip(np.searchsorted(grid_lat, station_lat), 0, height - 1)
    cols = np.clip(np.searchsorted(grid_lon, station_lon), 0, width - 1)
    valid = np.ones((height, width), dtype=bool)
    dates = np.arange(
        np.datetime64("2018-05-01"), np.datetime64("2018-05-07")
    )
    time_ns = dates.astype("datetime64[ns]").astype("i8")
    background = rng.gamma(1.8, 4.0, (days, members, height, width)).astype("f4")
    analysis_gauge = np.clip(background * 0.88 + 0.7, 0, None).astype("f4")
    analysis_imerg = np.clip(background * 0.95 + 1.0, 0, None).astype("f4")
    analysis_combined = np.clip(background * 0.90 + 0.9, 0, None).astype("f4")
    truth = np.clip(background.mean(axis=1) * 0.85 + rng.normal(0, 1, (days, height, width)), 0, None)
    gauge_mm = truth[:, rows, cols].astype("f4")

    def at_stations(values):
        return values[:, :, rows, cols]

    imerg = truth.reshape(days, height // 2, 2, width // 2, 2).mean(axis=(2, 4)).astype("f4")
    imerg_error = np.full_like(imerg, 3.0)
    stats = tmp_path / "stats.json"
    stats.write_text(
        json.dumps(
            {"precip_transform": {"kind": "log1p", "eps": 0.1, "mu": 1.0, "sd": 0.8}}
        )
    )

    dumps, reports, evaluations = [], [], []
    for fold in range(5):
        eval_idx = np.arange(fold, stations, 5)
        assim_idx = np.setdiff1d(np.arange(stations), eval_idx)
        prefix = tmp_path / f"fold{fold}"
        dump = prefix.with_suffix(".npz")
        report = prefix.with_suffix(".json")
        evaluation = tmp_path / f"fold{fold}_evaluation.json"
        np.savez_compressed(
            dump,
            background=background,
            analysis=analysis_combined,
            analysis_gauge=analysis_gauge,
            analysis_imerg=analysis_imerg,
            analysis_combined=analysis_combined,
            chirps=truth,
            condition=truth * 0.8,
            imerg=imerg,
            imerg_random_error=imerg_error,
            gauge_mm=gauge_mm,
            background_at_stations=at_stations(background),
            analysis_at_stations=at_stations(analysis_combined),
            gauge_analysis_at_stations=at_stations(analysis_gauge),
            imerg_analysis_at_stations=at_stations(analysis_imerg),
            combined_analysis_at_stations=at_stations(analysis_combined),
            station_id=np.asarray([f"S{index:02d}" for index in range(stations)]),
            station_name=np.asarray([f"Station {index:02d}" for index in range(stations)]),
            station_lat=station_lat,
            station_lon=station_lon,
            assim_idx=assim_idx,
            eval_idx=eval_idx,
            time=time_ns,
            background_time=(dates - np.timedelta64(1, "D")).astype("datetime64[ns]").astype("i8"),
            grid_lat=grid_lat,
            grid_lon=grid_lon,
            valid=valid,
        )
        report.write_text(
            json.dumps(
                {
                    "scope": {"checkpoint_stats": str(stats)},
                    "observation_error": {
                        "gauges": {
                            "sigma_transformed": 0.1,
                            "representativeness_transformed": 0.25,
                        },
                        "imerg": {
                            "sigma_floor_transformed": 0.35,
                            "representativeness_transformed": 0.1,
                            "correlation_variance_inflation": float(2 * np.pi),
                            "footprint_stride": 3,
                        },
                    },
                }
            )
        )
        crps = {"Background": 6.0, "Gauges only": 4.0, "IMERG only": 6.2, "Simultaneous": 4.2}
        evaluation.write_text(
            json.dumps(
                {
                    "scope": {
                        "holdout_fold": fold,
                        "holdout_folds": 5,
                        "station_days": int(days * len(eval_idx)),
                        "assimilated_stations": int(len(assim_idx)),
                        "withheld_stations": int(len(eval_idx)),
                        "dates": [str(value) for value in dates],
                        "background_day_offset": -1,
                        "withheld_station_ids": [f"S{index:02d}" for index in eval_idx],
                        "withheld_station_names": [f"Station {index:02d}" for index in eval_idx],
                    },
                    "probabilistic_methods": {
                        name: _method_score(value) for name, value in crps.items()
                    },
                    "daily_crps_mm": {
                        name: [value + 0.02 * day for day in range(days)]
                        for name, value in crps.items()
                    },
                }
            )
        )
        dumps.append(str(dump))
        reports.append(str(report))
        evaluations.append(str(evaluation))

    summary_json = tmp_path / "summary.json"
    summary_plot = tmp_path / "summary.png"
    environment = os.environ.copy()
    environment["MPLCONFIGDIR"] = str(tmp_path / "mpl")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/20_summarize_bmd_rotated_folds.py"),
            "--evaluations",
            *evaluations,
            "--out-json",
            str(summary_json),
            "--out-plot",
            str(summary_plot),
        ],
        cwd=ROOT,
        env=environment,
        check=True,
    )
    verification = tmp_path / "verification.png"
    spatial = tmp_path / "spatial.png"
    diagnostics = tmp_path / "diagnostics.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/21_bmd_fullmonth_diagnostics.py"),
            "--dumps",
            *dumps,
            "--reports",
            *reports,
            "--summary",
            str(summary_json),
            "--out-verification",
            str(verification),
            "--out-spatial",
            str(spatial),
            "--out-report",
            str(diagnostics),
            "--cartopy-data-dir",
            str(tmp_path / "cartopy"),
        ],
        cwd=ROOT,
        env=environment,
        check=True,
    )
    output = json.loads(diagnostics.read_text())
    assert output["scope"]["withheld_station_days"] == days * stations
    assert output["scope"]["primary_reference"].startswith("BMD gauges")
    assert output["normalised_innovations"]["IMERG footprints"]["n"] > 0
    assert verification.stat().st_size > 10_000
    assert spatial.stat().st_size > 10_000
    assert summary_plot.stat().st_size > 10_000
