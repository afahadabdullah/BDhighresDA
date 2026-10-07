"""Engineering checks for QC, withholding, provenance and paired pilot scoring."""
import ast
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_improve", ROOT/"scripts/106_surma_improvement.py")
I = importlib.util.module_from_spec(spec); spec.loader.exec_module(I)


class ImprovementTests(unittest.TestCase):
    def test_qc_uses_paired_days_and_flags_without_mutating(self):
        rows = []
        for j, sid in enumerate(["BMD_1", "BWDB_1", "BWDB_2", "BWDB_3"]):
            for day in pd.date_range("2020-11-01", "2020-11-30"):
                value = 60.0 if j == 0 else 1.0
                rows.append(dict(station_id=sid, date=day, lat=23+j*.015, lon=90, precip_mm=value))
        frame = pd.DataFrame(rows); before = frame.copy(deep=True)
        neighbors, flags = I.qc_tables(frame)
        row = neighbors[neighbors.station_id == "BMD_1"].iloc[0]
        self.assertEqual(row.minimum_paired_days, 30)
        self.assertEqual(row.paired_station_total_mm, 1800)
        self.assertEqual(row.nearby_bwdb_total_mm, 30)
        self.assertIn("BMD_1", set(flags.station_id))
        pd.testing.assert_frame_equal(before, frame)
        # Inadequate pairing must not fabricate a dry neighbor monthly total.
        frame.loc[(frame.station_id == "BWDB_3") & (frame.date.dt.day > 5), "precip_mm"] = np.nan
        neighbors, flags = I.qc_tables(frame)
        self.assertNotIn("BMD_1", set(neighbors.get("station_id", [])))

    def test_reviewed_exclusions_are_inclusive_and_leave_source_intact(self):
        frame = pd.DataFrame(dict(station_id=["BMD_1"]*3,
            date=pd.date_range("2020-01-01", periods=3), precip_mm=[1., 2., 3.]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"review.csv"
            path.write_text("station_id,start,end,reason\nBMD_1,2020-01-01,2020-01-02,confirmed source transcription\n")
            result, count = I.apply_exclusions(frame, path)
            self.assertEqual(count, 2)
            self.assertEqual(result.precip_mm.isna().sum(), 2)
            self.assertEqual(frame.precip_mm.tolist(), [1, 2, 3])
            path.write_text("station_id,start,end,reason\nBMD_1,2020-01-01,2020-01-02,\n")
            with self.assertRaises(ValueError): I.apply_exclusions(frame, path)

    def test_split_removes_whole_cells_and_spatial_buffer(self):
        sites = pd.DataFrame({"lat": [21+.32*(j//8)+.01*(j%2) for j in range(80)],
                              "lon": [88+.32*(j%8) for j in range(80)]},
                              index=[f"BWDB_{j}" for j in range(80)])
        held, retained = I.split_stations(sites, .1, 202205, 20)
        again = I.split_stations(sites, .1, 202205, 20)
        self.assertEqual((held, retained), again)
        self.assertFalse(held & retained)
        self.assertFalse(set(I.cells(sites.loc[sorted(held)])) & set(I.cells(sites.loc[sorted(retained)])))
        idx_h = [sites.index.get_loc(s) for s in held]; idx_a = [sites.index.get_loc(s) for s in retained]
        self.assertGreaterEqual(I.distance(sites)[np.ix_(idx_h, idx_a)].min(), 20)

    def test_windows_reject_overlap_and_original_selection_dates(self):
        default = I.windows(ROOT/"configs/surma_improvement_windows.json")
        self.assertEqual(sum(w["days"] for w in default), 120)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"windows.json"
            path.write_text(json.dumps({"windows": [{"label": "selection", "role": "tune",
                "start": "2022-05-01", "end": "2022-05-05"}]}))
            with self.assertRaises(ValueError): I.windows(path)

    def test_scores_include_false_alarms_and_probabilistic_extremes(self):
        members = np.array([[40, 60, 0], [60, 80, 20]], dtype=float)
        observed = np.array([60, 0, 0], dtype=float)
        scores = I.score_members(members, observed)
        self.assertEqual(scores["rain50_events"], 1)
        self.assertEqual(scores["rain50_hits"], 1)
        self.assertEqual(scores["rain50_false_alarms"], 1)
        self.assertAlmostEqual(scores["rain50_brier"], (0.25+1)/3)
        self.assertEqual(scores["rain100_events"], 0)
        self.assertIsNone(scores["rain100_pod"])
        self.assertAlmostEqual(I.score_members(np.array([[0.], [2.]]), np.array([1.]))["fair_crps_mm"], 0)

    def test_prepare_builds_real_superobs_with_withheld_values_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); source = base/"source"; out = base/"pilot"
            folder = source/"stations"/"2020_q1"; folder.mkdir(parents=True)
            rows = []
            for j in range(80):
                for day in pd.date_range("2020-01-01", "2020-03-31"):
                    rows.append(dict(station_id=f"BWDB_{j}", lat=21+.32*((j//2)//8)+.01*(j%2),
                        lon=88+.32*((j//2)%8)+.01*(j%2), date=day, precip_mm=float((day.day+j)%20)))
            pd.DataFrame(rows).to_csv(folder/"combined_daily.csv", index=False)
            imerg = source/"imerg_s04"/"2020_q1.nc"; imerg.parent.mkdir(); imerg.write_bytes(b"stub-imerg")
            stats = base/"stats.json"; stats.write_text(json.dumps({"precip_transform": {"kind": "sqrt", "eps": .1, "mu": 1., "sd": 2.}}))
            ckpt = base/"best.pt"; ckpt.write_bytes(b"stub-checkpoint")
            config = base/"config.yaml"; config.write_text("sampler: {}\n")
            predictor = base/"predictors.zarr"; predictor.mkdir()
            I.write_json(source/"predictor_validation.json", {"predictors": {"path": str(predictor)}})
            I.write_json(source/"prepared"/"2020_q1.json", {"statistics_sha256": I.sha(stats),
                "files": [I.record(folder/"combined_daily.csv"), I.record(imerg)]})
            windows = base/"windows.json"; windows.write_text(json.dumps({"windows": [
                {"label": "tune", "role": "tune", "start": "2020-01-01", "end": "2020-01-03"},
                {"label": "test", "role": "test", "start": "2020-02-01", "end": "2020-02-03"}]}))
            args = SimpleNamespace(root=str(source), out_dir=str(out), windows=str(windows), exclusions=None,
                ckpt=str(ckpt), stats=str(stats), config=str(config), min_coverage=.8,
                withhold=.1, seed=202205, buffer_km=20, members=3)
            with contextlib.redirect_stdout(io.StringIO()): I.prepare(args)
            plan = I.load_plan(args)
            held = set((out/"holdout_ids.txt").read_text().splitlines())
            for w in plan["windows"]:
                table = pd.read_csv(w["stations"])
                self.assertEqual(set(table.station_id) & held, held)
                original = pd.read_csv(out/"prepared"/w["label"]/"original_qc.csv")
                pd.testing.assert_frame_equal(table[table.station_id.isin(held)][["station_id", "date", "precip_mm"]].sort_values(["station_id", "date"]).reset_index(drop=True),
                    original[original.station_id.isin(held)][["station_id", "date", "precip_mm"]].sort_values(["station_id", "date"]).reset_index(drop=True))
            with self.assertRaises(ValueError): I.prepare(args)

    def test_summary_requires_complete_receipts_and_common_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            case = dict(label="tune", role="tune", start="2020-01-01", end="2020-01-02")
            plan = {"shared_files": [], "source_files": [], "windows": [case], "qc_review": "none", "excluded_station_days": 0}
            I.write_json(out/"pilot_plan.json", plan)
            folder = out/"runs"/"tune"; folder.mkdir(parents=True)
            obs = np.array([[10., 60.], [0., 20.]])
            ensembles = {"station_"+name: np.broadcast_to(obs[:, None, :], (2, 3, 2)).copy() for name in I.VARIANTS}
            ensembles["station_improve_t125"][0, :, 0] = np.nan
            arrays = folder/"sweep.npz"
            np.savez(arrays, times=["2020-01-01", "2020-01-02"], eval_idx=[0, 1], station_ids=["BMD_1", "BWDB_1"], gauge_mm=obs,
                variant_names=I.VARIANTS, **ensembles)
            report = folder/"sweep.json"; report.write_text("{}")
            I.write_json(folder/"complete.json", {"plan_sha256": I.sha(out/"pilot_plan.json"), "files": [I.record(arrays), I.record(report)]})
            args = SimpleNamespace(out_dir=str(out))
            with contextlib.redirect_stdout(io.StringIO()): I.summarize(args)
            scores = pd.read_csv(out/"summary"/"withheld_scores.csv")
            self.assertEqual(set(scores[scores.network == "ALL"].n), {3})
            self.assertEqual(set(scores[scores.network == "BMD"].n), {1})
            self.assertEqual(set(scores[scores.network == "BWDB"].n), {2})
            report.write_text('{"changed": true}')
            with self.assertRaises(ValueError): I.summarize(args)

    def test_tempered_heun_evaluates_both_endpoints_including_gate_crossing(self):
        # Execute the real integration function with a tiny NumPy torch shim.
        # This checks drift integration independently of a torch installation;
        # it is not a GPU/learned-prior skill test.
        tree = ast.parse((ROOT/"src/bdhires/da/sampler.py").read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "assimilate")
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function], type_ignores=[])
        torch = SimpleNamespace(manual_seed=lambda s: None, randn=lambda shape, device: np.zeros(shape),
            full=lambda shape, value, device: np.full(shape, value), no_grad=contextlib.nullcontext)
        ns = {"torch": torch, "GuidanceConfig": lambda: None, "apply_mask": lambda x, mask, fill: x,
              "make_schedule": lambda cfg, device: [0., .5, 1.]}
        exec(compile(ast.fix_missing_locations(module), "sampler-test", "exec"), ns)
        class Model:
            def eval(self): pass
            def __call__(self, x, t, cond): return np.zeros_like(x)
        flow = SimpleNamespace(x0_hat=lambda x, t, u: np.ones_like(x))
        cfg = SimpleNamespace(seed=1, cfg_scale=1, noise_scale=0, prior_temperature=1.25,
            n_corrections=0, n_steps=2, temperature_t_start=.15, heun=True, mask_fill=0)
        # First step: Heun's second evaluation supplies .1. Last Euler step: .2.
        result = ns["assimilate"](Model(), None, (1, 1, 1, 1), "cpu", cfg=cfg, flow=flow)
        self.assertAlmostEqual(float(result.item()), .3)
        cfg.prior_temperature = 1.
        result = ns["assimilate"](Model(), None, (1, 1, 1, 1), "cpu", cfg=cfg, flow=flow)
        self.assertEqual(float(result.item()), 0.)

    def test_submission_uses_dynamic_array_and_afterok_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory); bin_dir = tmp/"bin"; bin_dir.mkdir()
            I.write_json(tmp/"pilot_plan.json", {"status": "prepared_research_pilot", "windows": [{}, {}, {}]})
            sbatch = bin_dir/"sbatch"
            sbatch.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$TEST_SBATCH_LOG"\ncase "$*" in *--dependency=afterok:123*) echo 456;; *) echo 123;; esac\n')
            sbatch.chmod(0o755)
            env = {**os.environ, "PATH": str(bin_dir)+":"+os.environ["PATH"], "PYTHON_BIN": sys.executable,
                   "SURMA_IMPROVE_OUT": str(tmp), "SURMA_IMPROVE_CONCURRENCY": "4", "TEST_SBATCH_LOG": str(tmp/"calls")}
            result = subprocess.run(["bash", str(ROOT/"slurm/submit_surma_improvement.sh")], env=env, capture_output=True, text=True, check=True)
            calls = (tmp/"calls").read_text().splitlines()
            self.assertEqual(len(calls), 2)
            self.assertIn("--array=0-2%4", calls[0])
            self.assertIn("--dependency=afterok:123", calls[1])
            self.assertIn("Dependent summary: 456", result.stdout)


if __name__ == "__main__": unittest.main()
