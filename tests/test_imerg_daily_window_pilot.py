"""Timing/units and pairing checks without a checkpoint, GPU, or network."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bdhires.grids import BD
from bdhires.imerg import validate_prepared_time_convention

spec = importlib.util.spec_from_file_location("_daily_window_pilot", ROOT / "scripts/101_compare_imerg_daily_windows.py")
PILOT = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PILOT)


class DailyWindowPilotTests(unittest.TestCase):
    def test_station_source_falls_back_when_wide_table_is_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root / "data/stations/data_2020_2025"
            directory.mkdir(parents=True)
            catalog = directory / "Stations.csv"
            catalog.write_text("catalogue")
            bwdb = root / "bwdb.xlsx"
            bwdb.write_text("source")
            args = SimpleNamespace(bmd_catalog=str(catalog), bwdb=str(bwdb),
                                   bmd_wide=str(root / "missing_wide.csv"), bmd_data_dir=None)
            previous = Path.cwd()
            try:
                os.chdir(root)
                with self.assertRaisesRegex(FileNotFoundError, "no per-station BMD CSVs"):
                    PILOT.station_preparation_arguments(args, root / "out.csv", root)
                (directory / "Dhaka.csv").write_text("date,precip_mm\n2022-05-01,10\n")
                command = PILOT.station_preparation_arguments(args, root / "out.csv", root)
                self.assertIn("--bmd-data-dir", command)
                self.assertNotIn("--bmd-wide", command)
                Path(args.bmd_wide).write_text("wide history")
                command = PILOT.station_preparation_arguments(args, root / "out.csv", root)
                self.assertIn("--bmd-wide", command)
                args.bmd_data_dir = str(directory)
                command = PILOT.station_preparation_arguments(args, root / "out.csv", root)
                self.assertIn("--bmd-data-dir", command)
            finally:
                os.chdir(previous)

    def test_station_directory_preparation_preserves_qc_windows_and_provenance(self):
        spec = importlib.util.spec_from_file_location("_pilot_station_prep", ROOT / "scripts/99_prepare_production_stations.py")
        prep = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prep)
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            catalog = folder / "Stations.csv"
            catalog.write_text("StationNumber,Station,StationId,Latitude,Longitude\n1,Dhaka,0,23.76,90.38\n")
            daily = folder / "Dhaka.csv"
            daily.write_text("Datetime,Rainfall\n2022-05-01,10\n2022-05-02,***\n2022-05-03,12\n2022-05-04,13\n2022-05-05,14\n")
            workbook = folder / "bwdb.xlsx"
            workbook.write_text("source bytes")
            dates = pd.date_range("2022-05-01", "2022-05-05")
            bwdb = pd.DataFrame({"station_id": "BWDB_1", "name": "test", "lat": 23.8, "lon": 90.4,
                                 "date": dates, "precip_mm": 20., "source": "BWDB", "accumulation_end_hour_utc": 3})
            argv = ["prep", "--bmd-data-dir", str(folder), "--bmd-stations", str(catalog),
                    "--bwdb-xlsx", str(workbook), "--start", "2022-05-01", "--end", "2022-05-05",
                    "--min-bmd", "1", "--min-bwdb", "1", "--out", str(folder / "combined.csv"),
                    "--summary", str(folder / "summary.csv"), "--report", str(folder / "manifest.json")]
            with patch.object(sys, "argv", argv), patch.object(prep.prep82, "read_bwdb", return_value=(bwdb, {})):
                prep.main()
            combined = pd.read_csv(folder / "combined.csv")
            bmd = combined.loc[combined.source == "BMD"]
            self.assertEqual(len(bmd), 5)
            self.assertTrue(np.isnan(bmd.iloc[1].precip_mm))
            self.assertEqual(bmd.iloc[0].date, "2022-05-01")
            self.assertTrue((bmd.accumulation_end_hour_utc == 0).all())
            self.assertTrue((combined.loc[combined.source == "BWDB", "accumulation_end_hour_utc"] == 3).all())
            manifest = json.loads((folder / "manifest.json").read_text())
            self.assertEqual(manifest["input_sha256"][str(daily)], PILOT.sha(daily))
            self.assertNotIn(str(folder / "combined.csv"), manifest["input_sha256"])
            self.assertEqual(manifest["eligible_station_counts"], {"BMD": 1, "BWDB": 1})
            with self.assertRaisesRegex(ValueError, "cannot replace the wide BMD history"):
                prep.read_bmd(SimpleNamespace(bmd_data_dir=str(folder)), pd.Timestamp("2001-01-01"), pd.Timestamp("2001-01-05"))

    def test_calendar_ingestion_requires_truthful_metadata_and_explicit_opt_in(self):
        ds = xr.Dataset(attrs={"product": "GPM_3IMERGDF", "source_frequency": "daily",
                               "bmd_accumulation_end_hour_utc": 0, "window_duration_hours": 24,
                               "time_coordinate_semantics": "UTC calendar day; window start"})
        with self.assertRaisesRegex(ValueError, "explicit experimental opt-in"):
            validate_prepared_time_convention(ds)
        validate_prepared_time_convention(ds, allow_calendar_day=True)
        ds.attrs["bmd_accumulation_end_hour_utc"] = 3
        with self.assertRaisesRegex(ValueError, "native daily IMERG"):
            validate_prepared_time_convention(ds, allow_calendar_day=True)
        ds.attrs["source_frequency"] = "half-hourly"
        validate_prepared_time_convention(ds)

    def test_calendar_preparation_uses_previous_dates_native_mm_day_and_full_count(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            dates = np.arange(np.datetime64("2022-04-30"), np.datetime64("2022-05-05"))
            lat = BD.lat.reshape(64, 2).mean(axis=1)
            lon = BD.lon.reshape(64, 2).mean(axis=1)
            for i, day in enumerate(dates):
                count = np.full((1, 64, 64), 48, np.int16)
                count[0, 0, 0] = 47
                name = f"3B-DAY.MS.MRG.3IMERG.{str(day).replace('-', '')}-S000000-E235959.V07B.nc4"
                xr.Dataset(
                    {"precipitation": (("time", "lat", "lon"), np.full((1, 64, 64), 10. + i), {"units": "mm / day"}),
                     "randomError": (("time", "lat", "lon"), np.full((1, 64, 64), 2.), {"units": "mm/day"}),
                     "precipitation_cnt": (("time", "lat", "lon"), count)},
                    coords={"time": [day.astype("datetime64[ns]")], "lat": lat, "lon": lon}).to_netcdf(folder / name)
            output = folder / "prepared.nc"
            sources = PILOT.prepare_calendar_imerg(folder, output)
            self.assertEqual(len(sources), 5)
            with xr.open_dataset(output) as ds:
                np.testing.assert_array_equal(ds.time.values.astype("datetime64[D]"), dates)
                np.testing.assert_array_equal(ds.precipitation[:, 1, 1], np.arange(10., 15.))
                np.testing.assert_array_equal(ds.randomError[:, 1, 1], np.full(5, 2.))
                self.assertTrue(np.isnan(ds.precipitation[:, 0, 0]).all())
                self.assertEqual(ds.attrs["source_frequency"], "daily")
                self.assertEqual(ds.attrs["bmd_accumulation_end_hour_utc"], 0)
            coarse = folder / "coarse.nc"
            result = subprocess.run([sys.executable, str(ROOT / "scripts/44_coarsen_imerg_observations.py"),
                                     "--input", str(output), "--factor", "8", "--out", str(coarse)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            PILOT.enforce_complete_footprints(coarse, calendar=True)
            with xr.open_dataset(coarse) as ds:
                validate_prepared_time_convention(ds, allow_calendar_day=True)
                self.assertEqual(ds.sizes["lat"], 16)
                self.assertTrue(np.isnan(ds.precipitation[:, 0, 0]).all())
                self.assertEqual(ds.precipitation[0, 1, 1], 10.)

    def test_dry_run_is_bounded_and_does_not_write_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "output"
            result = subprocess.run([sys.executable, str(ROOT / "scripts/101_compare_imerg_daily_windows.py"),
                                     "--dry-run", "--root", str(root)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(root.exists())
            self.assertEqual(result.stdout.count("--members 1"), 2)
            self.assertIn("--start 2022-04-30 --end 2022-05-04", result.stdout)
            self.assertIn("--background-day-offset -1 --seed 202205", result.stdout)
            self.assertIn("--background-day-offset 0 --seed 202206", result.stdout)
            self.assertEqual(result.stdout.count("--allow-calendar-day-imerg"), 1)

    def test_slurm_job_forwards_options_and_stops_if_cuda_check_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            # Mock the node and interpreter boundary; no Slurm or CUDA required.
            (folder / "uname").write_text("#!/bin/sh\necho aarch64\n")
            (folder / "uname").chmod(0o755)
            interpreter = folder / "python"
            interpreter.write_text("#!/bin/sh\nif [ \"$1\" = -c ]; then exit \"${MOCK_CUDA_EXIT:-0}\"; fi\n"
                                   f"exec {sys.executable} \"$@\"\n")
            interpreter.chmod(0o755)
            output = folder / "results"
            env = {**os.environ, "SLURM_SUBMIT_DIR": str(ROOT), "PYTHON_BIN": str(interpreter),
                   "PATH": str(folder) + os.pathsep + os.environ["PATH"]}
            command = ["bash", str(ROOT / "slurm/imerg_daily_window_pilot.sbatch"),
                       "--dry-run", "--root", str(output), "--seed", "123"]
            result = subprocess.run(command, capture_output=True, text=True, env=env, cwd=folder)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--seed 123", result.stdout)
            self.assertIn("--seed 124", result.stdout)
            self.assertIn(str(output / "calendar.npz"), result.stdout)
            self.assertFalse(output.exists())
            result = subprocess.run(command, capture_output=True, text=True,
                                    env={**env, "MOCK_CUDA_EXIT": "17"}, cwd=folder)
            self.assertEqual(result.returncode, 17)
            self.assertNotIn("--members 1", result.stdout)

    def make_results(self, folder):
        dates = np.arange(np.datetime64("2022-05-01"), np.datetime64("2022-05-06"))
        calendar = dates - np.timedelta64(1, "D")
        land = np.ones((4, 4), bool)
        land[0, 0] = False
        fields = np.full((5, 4, 4), 10., np.float32)
        fields[:, 0, 0] = np.nan
        common = {"model_times": calendar.astype(str), "station_ids": np.array(["SOB_1", "BMD_2"]),
                  "station_lat": np.array([22., 23.]), "station_lon": np.array([90., 91.]),
                  "assim_idx": np.array([0, 1]), "eval_idx": np.array([], dtype=int),
                  "gauge_mm": np.full((5, 2), 10.), "condition": fields.copy(), "valid": land,
                  "grid_lat": np.arange(4) + 22., "grid_lon": np.arange(4) + 90.,
                  "meanfield_background": fields.copy(), "chirps": fields.copy(),
                  "raw_imerg_mm": np.full((5, 2, 2), 10.)}
        scopes = {"members": 1, "checkpoint_stats": "stats.json", "assimilate_all_stations": True,
                  "checkpoint": "best.pt", "checkpoint_data": "data.zarr",
                  "group": "v2_bmd_bwdb_superob_winner", "config_overrides": {},
                  "analysis_sampler_n_steps": 50, "analysis_sampler_n_corrections": 2,
                  "analysis_sampler_heun": True}
        for label, labels, increment, imerg_path in (("reporting", dates, 0., "imerg_reporting_s04.nc"),
                                                    ("calendar", calendar, 2., "imerg_calendar_s04.nc")):
            np.savez_compressed(folder / f"{label}.npz", **common, times=labels.astype(str),
                                **{f"meanfield_{PILOT.METHOD}": fields + increment,
                                   f"station_{PILOT.METHOD}": np.full((5, 1, 2), 10. + increment)})
            (folder / f"{label}.json").write_text(json.dumps({"scope": scopes}))
            xr.Dataset({"randomError": (("time", "lat", "lon"), np.full((5, 2, 2), 2.))}).to_netcdf(folder / imerg_path)
        (folder / "experiment.json").write_text(json.dumps({"statistics": "stats.json", "caveats": [], "input_sha256": {}}))
        return SimpleNamespace(root=str(folder))

    def test_summary_compares_same_event_and_uses_common_chirps(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            args = self.make_results(folder)
            with patch.object(PILOT, "plot_comparison"):
                PILOT.summarize(args)
            summary = json.loads((folder / "comparison.json").read_text())
            self.assertEqual(summary["analysis_mm_day"]["mae"], 2.)
            self.assertEqual(summary["analysis_mm_day"]["n"], 75)
            self.assertEqual(summary["five_day_total_mm"]["mae"], 10.)
            self.assertEqual(summary["matched_background_max_abs_difference"], 0.)
            self.assertEqual(summary["analysis_vs_common_chirps_a"]["mae"], 0.)
            self.assertEqual(summary["analysis_vs_common_chirps_b"]["mae"], 2.)
            self.assertEqual(len(summary["daily"]), 5)
            # Render a synthetic result once so layout can be checked without claiming real rainfall output.
            with np.load(folder / "reporting.npz") as dump:
                a = dump[f"meanfield_{PILOT.METHOD}"]
                PILOT.plot_comparison(folder, dump, a, a + 2., dump["valid"], dump["times"])
            self.assertTrue((folder / "comparison_maps.png").stat().st_size > 1000)

    def test_summary_rejects_unmatched_background_draws_or_dates(self):
        for bad_key in ("meanfield_background", "model_times"):
            with self.subTest(key=bad_key), tempfile.TemporaryDirectory() as temp:
                folder = Path(temp)
                args = self.make_results(folder)
                with np.load(folder / "calendar.npz") as source:
                    data = {key: source[key] for key in source.files}
                if bad_key == "model_times":
                    data[bad_key] = data[bad_key].astype("datetime64[D]") + np.timedelta64(1, "D")
                else:
                    data[bad_key] += 1.
                np.savez_compressed(folder / "calendar.npz", **data)
                with self.assertRaisesRegex(ValueError, "background"):
                    PILOT.summarize(args)


if __name__ == "__main__":
    unittest.main()
