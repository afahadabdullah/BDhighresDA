from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SPEC = importlib.util.spec_from_file_location(
    "_v2_confirmatory_summary",
    ROOT / "scripts" / "54_summarize_v2_confirmatory.py",
)
SUMMARY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SUMMARY
SPEC.loader.exec_module(SUMMARY)

from bdhires.zarr_output import write_physical_ensemble_zarr  # noqa: E402


def confirmatory_catalogue() -> ast.List:
    tree = ast.parse((ROOT / "scripts" / "28_simultaneous_method_sweep.py").read_text())
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "V2_CONFIRMATORY"
                for target in node.targets
            )
        ):
            return node.value
    raise AssertionError("V2_CONFIRMATORY catalogue not found")


def test_frozen_archive_catalogue_matches_summary_methods():
    catalogue = confirmatory_catalogue()
    analysis_names = [call.args[0].value for call in catalogue.elts]
    assert ["background", *analysis_names] == SUMMARY.METHODS
    assert analysis_names == [
        "guided_s6_g010_t100",
        "v2_simultaneous_s04_t100",
        "v2_simul_s04_ig010",
        "v2_simul_s04_huber3",
    ]


def test_selection_dates_and_may_2022_are_excluded_only_from_primary_scores():
    dates = np.asarray(
        ["2022-05-01", "2022-05-10", "2022-05-11", "2022-06-01"],
        dtype="datetime64[D]",
    )
    data = {
        "date": dates,
        "station": np.asarray(["A"] * len(dates)),
        "truth": np.asarray([0.0, 2.0, 3.0, 4.0]),
        "members": {
            name: np.tile(np.asarray([[0.0, 1.0, 2.0]]), (len(dates), 1))
            for name in SUMMARY.METHODS
        },
    }
    assert SUMMARY.daily_mask(data, confirmatory=True).tolist() == [False, False, True, True]
    assert SUMMARY.daily_mask(data, confirmatory=False).tolist() == [True] * 4
    assert SUMMARY.monthly_samples(data, confirmatory=True)["month"].astype(str).tolist() == [
        "2022-06"
    ]
    assert SUMMARY.monthly_samples(data, confirmatory=False)["month"].astype(str).tolist() == [
        "2022-05", "2022-06"
    ]


def test_segmented_bootstrap_never_bridges_seasonal_gaps():
    dates = np.asarray(
        ["2021-09-29", "2021-09-30", "2022-05-01", "2022-05-02"],
        dtype="datetime64[D]",
    )
    result = SUMMARY.segmented_block_bootstrap(
        dates, np.asarray([1.0, 2.0, 3.0, 4.0]), 3, 100, 7
    )
    assert result["n_segments"] == 2
    assert result["n_days"] == 4


def test_zarr_archive_is_xarray_compatible_and_preserves_real_zeros(tmp_path):
    import xarray as xr

    grid = SimpleNamespace(
        lat=np.asarray([0.0, 0.05], np.float32),
        lon=np.asarray([90.0, 90.05, 90.10], np.float32),
        lat_min=-0.025,
        lon_min=89.975,
        res=0.05,
    )
    valid = np.asarray([[True, True, False], [True, True, True]])
    base = np.zeros((2, 3, 2, 3), np.float32)
    base[:, :, valid] = np.arange(2 * 3 * int(valid.sum()), dtype=np.float32).reshape(
        2, 3, -1
    )
    base[:, :, ~valid] = np.nan
    fields = {"background": base, "analysis": base + np.where(valid, 1.0, np.nan)}
    output = tmp_path / "archive.zarr"
    write_physical_ensemble_zarr(
        output,
        fields=fields,
        method_specs={"background": {"streams": "none"}, "analysis": {"streams": "both"}},
        selected_times=np.asarray(["2021-05-01", "2021-05-02"], dtype="datetime64[D]"),
        grid=grid,
        valid=valid,
        condition=np.zeros((2, 2, 3), np.float32),
        chirps=np.ones((2, 2, 3), np.float32),
        raw_imerg_mm=np.zeros((2, 1, 1), np.float32),
        imerg_factor=8,
        station_ids=np.asarray(["A", "B"]),
        station_lat=np.asarray([0.0, 0.05]),
        station_lon=np.asarray([90.0, 90.10]),
        gauge_mm=np.asarray([[0.0, 1.0], [2.0, np.nan]], np.float32),
        assim_idx=np.asarray([0, 1]),
        scope={
            "start": "2021-05-01",
            "end": "2021-05-02",
            "assimilate_all_stations": True,
        },
    )

    dataset = xr.open_zarr(output, consolidated=True)
    assert dataset.attrs["complete"] is True
    assert dataset.precipitation.dims == ("method", "time", "member", "lat", "lon")
    assert dataset.method.values.astype(str).tolist() == ["background", "analysis"]
    assert dataset.member.values.tolist() == [0, 1, 2]
    assert dataset.valid.values.tolist() == valid.tolist()
    assert dataset.cpc.values[0, 0, 0] == 0.0
    assert dataset.imerg.values[0, 0, 0] == 0.0
    assert dataset.gauge.values[0, 0] == 0.0
    np.testing.assert_allclose(
        dataset.ensemble_mean.sel({"method": "analysis"}).values[:, valid],
        np.mean(fields["analysis"][:, :, valid], axis=1),
    )
    assert np.isnan(dataset.precipitation.values[..., 0, 2]).all()
    dataset.close()


def test_zarr_writer_refuses_to_overwrite_a_completed_archive(tmp_path):
    existing = tmp_path / "archive.zarr"
    existing.mkdir()
    with unittest.TestCase().assertRaisesRegex(FileExistsError, "refusing to overwrite"):
        write_physical_ensemble_zarr(
            existing,
            fields={}, method_specs={}, selected_times=np.asarray([]), grid=None,
            valid=np.asarray([]), condition=np.asarray([]), chirps=np.asarray([]),
            raw_imerg_mm=None, imerg_factor=8, station_ids=np.asarray([]),
            station_lat=np.asarray([]), station_lon=np.asarray([]), gauge_mm=np.asarray([]),
            assim_idx=np.asarray([]), scope={},
        )


def test_full_four_period_validation_and_selection_guard(tmp_path):
    grid = SimpleNamespace(
        lat=np.asarray([22.0, 22.05], np.float32),
        lon=np.asarray([88.0, 88.05], np.float32),
        lat_min=21.975,
        lon_min=87.975,
        res=0.05,
    )
    valid = np.ones((2, 2), bool)
    station_ids = np.asarray(["A", "B", "C", "D", "E"])
    station_lat = np.linspace(22.0, 22.04, 5)
    station_lon = np.linspace(88.0, 88.04, 5)
    method_offsets = dict(zip(SUMMARY.METHODS, [2.0, 1.0, 0.5, 0.2, 0.3]))
    dump_paths, report_paths, stores = [], [], []

    for period, (start, end) in SUMMARY.PERIODS.items():
        times = np.arange(
            np.datetime64(start, "D"),
            np.datetime64(end, "D") + np.timedelta64(1, "D"),
        )
        day = np.arange(len(times))[:, None]
        station = np.arange(len(station_ids))[None, :]
        gauge = ((day + station) % 8).astype(np.float32)
        ensembles = {}
        for name, offset in method_offsets.items():
            ensembles[name] = (
                gauge[:, None, :]
                + offset
                + np.asarray([-0.2, 0.0, 0.2], np.float32)[None, :, None]
            ).astype(np.float32)

        period_dir = tmp_path / "cv" / period
        period_dir.mkdir(parents=True)
        for fold in range(5):
            dump_path = period_dir / f"fold{fold}.npz"
            report_path = period_dir / f"fold{fold}.json"
            np.savez_compressed(
                dump_path,
                times=times.astype(str),
                station_ids=station_ids,
                station_lat=station_lat,
                station_lon=station_lon,
                variant_names=np.asarray(SUMMARY.METHODS),
                eval_idx=np.asarray([fold]),
                assim_idx=np.asarray([index for index in range(5) if index != fold]),
                gauge_mm=gauge,
                **{f"station_{name}": values for name, values in ensembles.items()},
            )
            report_path.write_text(json.dumps({
                "scope": {
                    "start": start,
                    "end": end,
                    "n_days": len(times),
                    "members": 3,
                    "checkpoint": "v2.pt",
                    "checkpoint_data": "cpc.zarr",
                    "checkpoint_stats": "stats.json",
                    "background_day_offset": -1,
                    "seed": 201805,
                    "group": "v2_confirmatory",
                    "precip_transform": {"kind": "sqrt"},
                    "config_overrides": [{"path": "factor", "value": 8}],
                    "holdout_folds": 5,
                    "holdout_fold": fold,
                    "assimilate_all_stations": False,
                }
            }))
            dump_paths.append(dump_path)
            report_paths.append(report_path)

        fields = {}
        for name, offset in method_offsets.items():
            daily = np.broadcast_to(
                (gauge.mean(axis=1) + offset)[:, None, None, None],
                (len(times), 3, 2, 2),
            ).copy()
            daily += np.asarray([-0.2, 0.0, 0.2])[None, :, None, None]
            fields[name] = daily.astype(np.float32)
        store = tmp_path / "gridded" / f"{period}.zarr"
        write_physical_ensemble_zarr(
            store,
            fields=fields,
            method_specs={name: {} for name in SUMMARY.METHODS},
            selected_times=times,
            grid=grid,
            valid=valid,
            condition=np.zeros((len(times), 2, 2), np.float32),
            chirps=np.ones((len(times), 2, 2), np.float32),
            raw_imerg_mm=np.ones((len(times), 1, 1), np.float32),
            imerg_factor=8,
            station_ids=station_ids,
            station_lat=station_lat,
            station_lon=station_lon,
            gauge_mm=gauge,
            assim_idx=np.arange(5),
            scope={"start": start, "end": end, "assimilate_all_stations": True},
        )
        stores.append(store)

    files = SUMMARY.load_and_validate(dump_paths, report_paths)
    data = SUMMARY.collect(files)
    primary = SUMMARY.score_scope(data, confirmatory=True)
    descriptive = SUMMARY.score_scope(data, confirmatory=False)
    catalog = SUMMARY.validate_zarr_stores(stores)
    fold_plots = SUMMARY.plot_fold_diagnostics(files, tmp_path / "fold_plots")
    assert primary["n_dates"] == 510
    assert descriptive["n_dates"] == 520
    assert primary["daily"][SUMMARY.BACKGROUND]["n"] == 2550
    assert primary["monthly"][SUMMARY.BACKGROUND]["n"] == 80
    assert sum(entry["days"] for entry in catalog) == 520
    assert len(fold_plots) == 20 and all(Path(path).is_file() for path in fold_plots)
    json.dumps({"primary": primary, "catalog": catalog}, allow_nan=False)
