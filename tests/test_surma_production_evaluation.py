"""Diagnostics preserve paired samples, completed archives and dependencies."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/surma-evaluation-test-mpl")
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_surma_eval", ROOT/"scripts/105_evaluate_surma_production.py")
E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
from bdhires.zarr_output import write_physical_ensemble_zarr


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def fixture(root):
    status = E.load_module("_status_fixture", ROOT/"scripts/104_surma_production_status.py")
    periods = status.quarters(2001, 2001)
    grid = SimpleNamespace(lat=np.linspace(20, 21, 16), lon=np.linspace(89, 90, 16),
        lat_min=20-1/30, lon_min=89-1/30, res=1/15)
    ids = np.asarray(["BMD_1", "BMD_2", "BWDB_1", "BWDB_2"])
    slat = np.asarray([20.3, 20.4, 20.5, 20.6]); slon = np.asarray([89.3, 89.4, 89.5, 89.6])
    records = []
    for p in periods:
        label = p["label"]
        times = np.arange(np.datetime64(p["start"]), np.datetime64(p["end"])+np.timedelta64(1, "D"))
        base = (np.arange(len(times)) % 15).astype(float)
        fields = np.broadcast_to(base[:, None, None, None], (len(times), 30, 16, 16)).copy()
        fields += np.arange(30)[None, :, None, None]/30
        scope = {"start": p["start"], "end": p["end"], "members": 30,
                 "background_day_offset": -1, "assimilate_all_stations": True}
        station = np.repeat(fields[:, :, 0, 0, None], 4, axis=2)
        truth = np.repeat(base[:, None], 4, axis=1)
        physical = {"background": fields+2, E.ANALYSIS: fields}
        path = root/"gridded"/f"{label}.zarr"
        write_physical_ensemble_zarr(path, fields=physical, method_specs={s: {} for s in physical},
            selected_times=times, grid=grid, valid=np.ones((16, 16), bool),
            condition=np.repeat((base+1)[:, None, None], 16, axis=1).repeat(16, axis=2),
            chirps=np.repeat((base+.5)[:, None, None], 16, axis=1).repeat(16, axis=2),
            raw_imerg_mm=np.repeat((base+.2)[:, None, None], 2, axis=1).repeat(2, axis=2),
            imerg_factor=8, station_ids=ids, station_lat=slat, station_lon=slon,
            gauge_mm=truth, assim_idx=np.arange(4), scope=scope)
        folder = root/"stations"/label; folder.mkdir(parents=True)
        pd.DataFrame({"date": np.repeat(times.astype(str), 4), "station_id": np.tile(ids, len(times)),
            "source": np.tile(["BMD", "BMD", "BWDB", "BWDB"], len(times)),
            "lat": np.tile(slat, len(times)), "lon": np.tile(slon, len(times)),
            "precip_mm": truth.ravel()}).to_csv(folder/"combined_daily.csv", index=False)
        pd.DataFrame({"station_id": ids, "eligible_for_analysis": True}).to_csv(folder/"station_summary.csv", index=False)
        report = root/"production_metadata"/f"{label}.json"
        write(report, {"scope": scope}); arrays = report.with_suffix(".npz")
        np.savez_compressed(arrays, times=times.astype(str), gauge_mm=truth,
            eval_idx=np.asarray([], int), assim_idx=np.arange(4), station_lat=slat, station_lon=slon,
            station_background=station+2, **{f"station_{E.ANALYSIS}": station})
        record = {"status": "complete", "period": {k: p[k] for k in ("label", "start", "end")},
                  "days": len(times), "field_store": str(path),
                  "report_sha256": E.sha(report), "station_array_sha256": E.sha(arrays)}
        prepared = root/"prepared"/f"{label}.json"
        write(prepared, {"files": [{"path": str(folder/"combined_daily.csv"), "sha256": E.sha(folder/"combined_daily.csv")}]})
        record["prepared_manifest_sha256"] = E.sha(prepared)
        write(root/"validated"/f"{label}.json", record); records.append(record)
    write(root/"production_manifest.json", {"status": "complete", "members": 30,
        "start": "2001-01-01", "end": "2001-12-31", "days": 365, "shards": records})
    boundary = root/"boundary.geojson"
    write(boundary, {"type": "Polygon", "coordinates": [[[88.9, 19.9], [90.1, 19.9],
        [90.1, 21.1], [88.9, 21.1], [88.9, 19.9]]]})
    return boundary


class EvaluationTests(unittest.TestCase):
    def test_sampling_never_extrapolates_or_fills_missing_corner(self):
        values = np.asarray([[0., 2.], [2., 4.]])
        coordinates = np.asarray([0., 1.])
        sampled = E.bilinear(values, coordinates, coordinates,
                             np.asarray([.5, 0, 2]), np.asarray([.5, 0, .5]))
        np.testing.assert_allclose(sampled[:2], [2, 0]); self.assertTrue(np.isnan(sampled[2]))
        values[1, 1] = np.nan
        sampled = E.bilinear(values, coordinates, coordinates, np.asarray([.5, 0]), np.asarray([.5, 0]))
        self.assertTrue(np.isnan(sampled[0])); self.assertEqual(sampled[1], 0)

    def test_common_scores_and_monthly_totals_have_identical_samples(self):
        frame = pd.DataFrame({"observed": [0., 10., 100.], **{s: [1., 11., 99.] for s in E.SOURCES},
            **{f"{s}_{key}": [1., 1., 1.] for s in ("analysis", "background") for key in ("crps", "spread", "covered")}})
        frame.loc[2, "chirps"] = np.nan
        scored = E.common_scores(frame)
        self.assertEqual(scored.n.tolist(), [2]*5); np.testing.assert_allclose(scored.bias_mm, 1.)
        daily = pd.DataFrame({"date": pd.date_range("2001-01-01", periods=31), "station_id": "s1",
            "network": "BMD", "observed": 1., **{s: 2. for s in E.SOURCES}})
        daily.loc[:2, "imerg"] = np.nan
        total = E.monthly_station(daily)
        self.assertEqual(total.paired_days.tolist(), [28]); self.assertEqual(total.observed.tolist(), [28])
        daily.loc[3, "imerg"] = np.nan; self.assertTrue(E.monthly_station(daily).empty)

    def test_temporal_moments_extremes_and_missing_day(self):
        a = E.Accumulator((1, 1, 1))
        for x in (0., 0., 2., 20.): a.add(np.full((1, 1, 1), x), np.ones((1, 1), bool))
        result = a.result(4)
        self.assertEqual(result["total"].item(), 22)
        self.assertEqual(result["cdd"].item(), 2); self.assertEqual(result["cwd"].item(), 2)
        self.assertEqual(result["r20_days"].item(), 1); self.assertEqual(result["rx1day"].item(), 20)
        self.assertAlmostEqual(result["daily_sd"].item(), np.std([0, 0, 2, 20]))
        a.add(np.zeros((1, 1, 1)), np.zeros((1, 1), bool))
        self.assertTrue(np.isnan(a.result(5)["total"]).all())

    def test_fair_crps_matches_pairwise_definition(self):
        members = np.asarray([[0., 1.], [1., 4.], [3., 5.]]); obs = np.asarray([1., 2.])
        pair = np.abs(members[:, None]-members[None, :]).sum(axis=(0, 1))/(2*3*2)
        np.testing.assert_allclose(E.fair_crps(members, obs), np.abs(members-obs).mean(axis=0)-pair)

    def test_final_gate_and_full_rendered_synthetic_year(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/"production"; out = Path(temp)/"evaluation"
            with self.assertRaises(FileNotFoundError): E.require_final(root)
            boundary = fixture(root); manifest, periods = E.require_final(root)
            self.assertEqual(len(periods), 4)
            original_manifest = manifest.copy(); manifest["shards"] = manifest["shards"][:3]
            write(root/"production_manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "unique"): E.require_final(root)
            write(root/"production_manifest.json", original_manifest)
            with contextlib.redirect_stdout(io.StringIO()):
                E.main(["--root", str(root), "--out-dir", str(out), "--boundary-geojson", str(boundary)])
            report = json.loads((out/"evaluation.json").read_text())
            self.assertEqual(report["status"], "complete"); self.assertEqual(report["days"], 365)
            self.assertEqual(len(report["figures"]), 15)
            for stem in report["figures"]:
                self.assertGreater((out/f"{stem}.png").stat().st_size, 1000)
                self.assertGreater((out/f"{stem}.pdf").stat().st_size, 1000)
                self.assertTrue((out/"data"/f"{stem}_manifest.json").is_file())
            scores = pd.read_csv(out/"original_gauge_scores.csv"); self.assertEqual(set(scores.n), {730})
            self.assertEqual(len(pd.read_csv(out/"monthly_gauge_totals.csv")), 48)
            self.assertEqual(len(pd.read_csv(out/"annual_domain.csv")), 5)
            with patch.object(E, "process", side_effect=ValueError("bad diagnostic input")):
                with self.assertRaises(ValueError):
                    E.main(["--root", str(root), "--out-dir", str(out), "--boundary-geojson", str(boundary)])
            self.assertFalse((out/"evaluation.json").exists())

    def test_submit_wrapper_depends_on_final_validator(self):
        with tempfile.TemporaryDirectory() as temp:
            command = Path(temp)/"sbatch"; capture = Path(temp)/"arguments"
            command.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$SURMA_TEST_CAPTURE"\nprintf "123456;cluster\\n"\n')
            command.chmod(0o755)
            env = {**os.environ, "PATH": temp+":"+os.environ["PATH"], "SURMA_TEST_CAPTURE": str(capture)}
            result = subprocess.run(["bash", str(ROOT/"slurm/submit_surma_production_evaluation.sh"), "37940179"],
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--dependency=afterok:37940179", capture.read_text().splitlines())
            self.assertIn("Evaluation job: 123456", result.stdout)
            result = subprocess.run(["bash", str(ROOT/"slurm/submit_surma_production_evaluation.sh"), "bad"],
                                    env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__": unittest.main()
