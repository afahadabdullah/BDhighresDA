"""Checks for conditional scores, real aggregation mapping and immutable pilots."""
import ast
import contextlib
from dataclasses import dataclass, replace
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_tail_tests", ROOT/"scripts/107_surma_tail_diagnostics.py")
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
I = T.I


def completed_fixture(root):
    folder = root/"old"; folder.mkdir()
    held = folder/"holdout_ids.txt"; held.write_text("BMD_1\nBWDB_1\n")
    code = root/"old_runtime.py"; code.write_text("old code")
    cases = []
    for label, month, role in [("tune", 1, "tune"), ("assessment", 2, "test")]:
        dates = pd.date_range(f"2020-{month:02d}-01", periods=2)
        prepared = folder/"prepared"/label; prepared.mkdir(parents=True)
        rows = []
        for date in dates:
            for sid, lat, lon, value in [("BMD_1", 25., 90., 60.), ("BWDB_1", 26., 90., 0.),
                                       ("BMD_2", 23.01, 90.01, 80.), ("BWDB_2", 23.02, 90.02, 20.)]:
                rows.append(dict(station_id=sid, date=date, lat=lat, lon=lon, precip_mm=value))
        original = prepared/"original_qc.csv"; pd.DataFrame(rows).to_csv(original, index=False)
        cell = int(I.cells(pd.DataFrame({"lat": [23.01], "lon": [90.01]})).iloc[0])
        superrows = [r for r in rows if r["station_id"] in {"BMD_1", "BWDB_1"}]
        superrows += [dict(station_id=f"SOB_{cell}", date=date, lat=23.015, lon=90.015, precip_mm=50.) for date in dates]
        station_table = prepared/"superob.csv"; pd.DataFrame(superrows).to_csv(station_table, index=False)
        w = dict(label=label, role=role, start=str(dates[0].date()), end=str(dates[-1].date()), days=2,
                 quarter="2020_q1", stations=str(station_table), files=[I.record(original), I.record(station_table)])
        cases.append(w)
    qc = folder/"qc"; qc.mkdir()
    pd.DataFrame([dict(station_id="BMD_2", month="2020-01", reason="review only")]).to_csv(qc/"review_flags.csv", index=False)
    plan = dict(status="prepared_research_pilot", windows=cases, variants=I.VARIANTS,
        holdout=str(held), shared_files=[I.record(held), I.record(code)], source_files=[],
        members=3, qc_review="no corrections", excluded_station_days=0)
    I.write_json(folder/"pilot_plan.json", plan)
    for w in cases:
        run = folder/"runs"/w["label"]; run.mkdir(parents=True)
        obs = np.array([[60., 0., 50.], [60., 0., 50.]])
        members = np.array([[[20., 0., 50.], [40., 0., 50.], [80., 0., 50.]]]*2)
        arrays = run/"sweep.npz"
        np.savez(arrays, times=pd.date_range(w["start"], w["end"]).strftime("%Y-%m-%d").to_numpy(dtype=str),
            eval_idx=[0, 1], station_ids=["BMD_1", "BWDB_1", "SOB_9200360"],
            gauge_mm=obs, variant_names=I.VARIANTS, **{"station_"+n: members for n in I.VARIANTS})
        report = run/"sweep.json"; report.write_text("{}")
        I.write_json(run/"complete.json", {"plan_sha256": I.sha(folder/"pilot_plan.json"), "files": [I.record(arrays), I.record(report)]})
    return folder, code


class TailTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is checked on the GPU node before sampling")
    def test_real_per_stream_autograd_preflight(self):
        T.tail_gradient_check()

    def test_conditional_coverage_distinguishes_interval_side_and_member_possibility(self):
        scores = T.conditional_scores(np.array([[20., 0.], [40., 0.], [80., 0.]]), np.array([100., 0.]))
        self.assertEqual(scores["above_interval_fraction"], .5)
        self.assertEqual(scores["below_interval_fraction"], 0)
        self.assertEqual(scores["rain50_any_member_fraction"], .5)
        self.assertEqual(scores["rain50_pod"], 0)
        self.assertAlmostEqual(scores["rain50_probability_on_events"], 1/3)

    def test_aggregation_maps_actual_cells_and_excludes_held_gauges(self):
        original = pd.DataFrame({"station_id": ["BMD_1", "BMD_2", "BWDB_2"],
            "lat": [25., 23.01, 23.02], "lon": [90., 90.01, 90.02],
            "date": pd.to_datetime(["2020-01-01"]*3), "precip_mm": [100., 80., 20.]})
        cell = int(I.cells(original.iloc[1:2]).iloc[0])
        table = pd.DataFrame({"station_id": ["BMD_1", f"SOB_{cell}"],
            "date": pd.to_datetime(["2020-01-01"]*2), "precip_mm": [100., 50.]})
        result = T.aggregation_rows(original, table, {"BMD_1"})
        self.assertEqual(result.station_id.tolist(), ["BMD_2"])
        self.assertEqual(result.original_minus_assimilated_mm.iloc[0], 30)
        self.assertEqual(result.reporting_cell_members.iloc[0], 2)
        self.assertEqual(result.assimilated_fraction_of_original.iloc[0], .625)
        table.loc[1, "station_id"] = "SOB_999"
        with self.assertRaises(ValueError): T.aggregation_rows(original, table, {"BMD_1"})

    def test_completed_results_allow_code_changes_only_for_reading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); folder, code = completed_fixture(root)
            code.write_text("new code")
            plan, changes = T.load_archive(folder)
            self.assertEqual(changes, [str(code.resolve())])
            with self.assertRaises(ValueError): T.load_archive(folder, strict_code=True)
            T.window_arrays(folder, plan, plan["windows"][0])
            Path(plan["windows"][0]["stations"]).write_text("changed data")
            with self.assertRaises(ValueError): T.load_archive(folder)

    def test_saved_diagnostics_are_stratified_and_leave_archive_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); folder, code = completed_fixture(root)
            before = I.sha(folder/"pilot_plan.json")
            args = SimpleNamespace(pilot=str(folder), out_dir=str(root/"next"))
            with contextlib.redirect_stdout(io.StringIO()): T.diagnose(args)
            out = root/"next"/"diagnostics_before"
            scores = pd.read_csv(out/"intensity_scores.csv")
            heavy = scores[(scores.window == "POOLED") & (scores.role == "test") &
                           (scores.network == "BMD") & (scores.rainfall_bin == "rain50_100")]
            self.assertEqual(set(heavy.n), {2})
            self.assertEqual(set(heavy.rain50_pod), {0})
            events = pd.read_csv(out/"heavy_events.csv")
            self.assertEqual(set(events.station_id), {"BMD_1"})
            self.assertEqual(set(events.member_max_mm), {80})
            shortlist = pd.read_csv(out/"review_flags_for_pilot.csv")
            self.assertEqual(shortlist.window.tolist(), ["tune"])
            self.assertEqual(I.sha(folder/"pilot_plan.json"), before)

    def test_next_preparation_preserves_split_and_relabels_seen_test_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); parent = root/"old"; parent.mkdir()
            held = parent/"holdout_ids.txt"; held.write_text("BWDB_0\nBWDB_1\n")
            files = []
            for label, start, role in [("tune", "2020-01-01", "tune"), ("test", "2020-02-01", "test")]:
                folder = parent/"prepared"/label; folder.mkdir(parents=True)
                rows = []
                # Held-out sites are well away from the retained clusters.
                for j in range(80):
                    lat = 25.8+j*.02 if j < 2 else 21+.32*((j//2)//8)+.01*(j%2)
                    lon = 92.8 if j < 2 else 88+.32*((j//2)%8)+.01*(j%2)
                    for day in pd.date_range(start, periods=3):
                        rows.append(dict(station_id=f"BWDB_{j}", lat=lat, lon=lon, date=day, precip_mm=float((j+day.day)%20)))
                original = folder/"original_qc.csv"; pd.DataFrame(rows).to_csv(original, index=False)
                imerg = folder/"imerg.nc"; imerg.write_bytes(b"placeholder")
                files.append(dict(label=label, role=role, start=start,
                    end=str((pd.Timestamp(start)+pd.Timedelta(days=2)).date()), days=3, quarter="2020_q1",
                    imerg=str(imerg), files=[I.record(original)]))
            stats = root/"stats.json"; stats.write_text(json.dumps({"precip_transform": {"kind": "sqrt", "eps": .1, "mu": 1., "sd": 2.}}))
            ckpt = root/"model.pt"; ckpt.write_bytes(b"placeholder")
            config = root/"da.yaml"; config.write_text("placeholder")
            plan = dict(windows=files, shared_files=[I.record(held)], source_files=[], holdout=str(held),
                checkpoint=str(ckpt), config=str(config), stats=str(stats), predictors=str(root/"predictors.zarr"),
                fill_known_cpc_gaps=False, members=3, seed=202205, min_coverage=.8,
                held_stations=2, retained_stations=78, buffer_km=20, excluded_station_days=0)
            I.write_json(parent/"pilot_plan.json", plan)
            args = SimpleNamespace(pilot=str(parent), out_dir=str(root/"next"), exclusions=None)
            with contextlib.redirect_stdout(io.StringIO()): T.prepare(args)
            result, changes = T.load_archive(root/"next", strict_code=True)
            self.assertEqual(result["variants"], T.VARIANTS)
            self.assertEqual(result["windows"][1]["role"], "retest")
            self.assertEqual(T.held_ids(result), {"BWDB_0", "BWDB_1"})
            with self.assertRaises(ValueError): T.prepare(args)

    def test_per_observation_huber_keeps_satellite_threshold_unchanged(self):
        # Execute the actual likelihood function with a small ndarray backend.
        # GPU/autograd integration is a separate runtime check on PRISM.
        class Tensor(np.ndarray):
            def __new__(cls, values): return np.ndarray.view(np.asarray(values, dtype=float), cls)
            def view(self, *shape): return self.reshape(shape)
            def clamp_min(self, value): return np.maximum(self, value)
            def to(self, dtype): return self.astype(dtype)
            def abs(self): return np.abs(self)
            def sqrt(self): return np.sqrt(self)
            def flatten(self, start_dim=0): return self.reshape((*self.shape[:start_dim], -1))
            def sum(self, dim=None, **kwargs): return np.ndarray.sum(self, axis=dim, **kwargs)
        backend = SimpleNamespace(isfinite=np.isfinite, zeros_like=np.zeros_like,
            where=lambda c, a, b: Tensor(np.where(c, a, b)))
        tree = ast.parse((ROOT/"src/bdhires/da/guidance.py").read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "obs_log_likelihood")
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), fn], type_ignores=[])
        ns = {"torch": backend}; exec(compile(ast.fix_missing_locations(module), "likelihood-test", "exec"), ns)
        cfg = SimpleNamespace(gamma=0., huber_delta=Tensor([5., 3.]))
        result = ns["obs_log_likelihood"](Tensor([[[10., 10.]]]), Tensor([[[0., 0.]]]), Tensor([1., 1.]), Tensor([.5]), cfg)
        self.assertAlmostEqual(float(result.item()), -63.)
        cfg.huber_delta = 3.
        result = ns["obs_log_likelihood"](Tensor([[[10., 10.]]]), Tensor([[[0., 0.]]]), Tensor([1., 1.]), Tensor([.5]), cfg)
        self.assertAlmostEqual(float(result.item()), -51.)

    def test_tail_variants_change_one_factor_and_preserve_baseline(self):
        tree = ast.parse((ROOT/"scripts/28_simultaneous_method_sweep.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Variant")
        ns = {"dataclass": dataclass, "replace": replace, "__name__": __name__}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), "variant-test", "exec"), ns)
        base = ns["Variant"](name=I.BASELINE, huber_delta=3., prior_temperature=1.,
            gauge_component_spread_cells=6., secondary_r_multiplier=4.)
        ns.update(_IMPROVEMENT_BASE=base, CORE=[ns["Variant"](name="background", streams="none")])
        assignment = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "V2_TAIL_IMPROVEMENT" for t in n.targets))
        exec(compile(ast.Module(body=[assignment], type_ignores=[]), "variant-test", "exec"), ns)
        variants = ns["V2_TAIL_IMPROVEMENT"]
        self.assertEqual([v.name for v in variants], T.VARIANTS)
        self.assertEqual(variants[2].gauge_huber_delta, 5.)
        self.assertEqual(variants[2].huber_delta, 3.)
        self.assertEqual(variants[3].gauge_weight, 1.25)
        self.assertEqual(variants[4].gauge_weight, .75)
        self.assertTrue(all(v.prior_temperature == 1. for v in variants[1:]))


if __name__ == "__main__": unittest.main()
