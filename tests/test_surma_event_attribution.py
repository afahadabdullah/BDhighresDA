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
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_events_test", ROOT/"scripts/108_surma_event_attribution.py")
E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)


class EventTests(unittest.TestCase):
    def test_aggregation_counts_threshold_crossing_without_moving_unmerged_values(self):
        d = pd.DataFrame(dict(role=["retest"]*4, aggregated=[True, True, True, False],
            precip_mm=[120., 100., 60., 110.], assimilated_mm=[80., 105., 40., 110.]))
        d["original_minus_assimilated_mm"] = d.precip_mm-d.assimilated_mm
        d["assimilated_fraction_of_original"] = d.assimilated_mm/d.precip_mm
        result = E.aggregation_summary(d).set_index("aggregated")
        self.assertEqual(result.at[True, "n_original_ge100"], 2)
        self.assertEqual(result.at[True, "n_ge100_reduced_below100"], 1)
        self.assertEqual(result.at[True, "n_reduced_below50"], 1)
        self.assertEqual(result.at[False, "mean_reduction_mm"], 0)

    def test_month_flags_apply_to_each_inclusive_station_day(self):
        flags = pd.DataFrame(dict(station_id=["BWDB_X", "BWDB_X"], reason=["month", "day"],
            review_start=pd.to_datetime(["2020-01-01", "2020-01-31"]),
            review_end=pd.to_datetime(["2020-01-31", "2020-01-31"])))
        self.assertEqual(E.flag_reasons(flags, "BWDB_X", pd.Timestamp("2020-01-31")), "day | month")
        self.assertEqual(E.flag_reasons(flags, "BWDB_Y", pd.Timestamp("2020-01-31")), "")
        self.assertEqual(E.flag_reasons(flags, "BWDB_X", pd.Timestamp("2020-02-01")), "")

    def test_neighbors_exclude_self_missing_wrong_date_and_distant_stations(self):
        frame = pd.DataFrame(dict(station_id=["BWDB_X", "BWDB_Y", "BWDB_Z", "BMD_1", "BWDB_F", "BWDB_T"],
            date=pd.to_datetime(["2020-01-01"]*5+["2020-01-02"]),
            lat=[23., 23.01, 23.02, 23.03, 26., 23.], lon=[90.]*6,
            precip_mm=[100., 80., np.nan, 70., 90., 110.]))
        self.assertEqual(E.nearby(frame, "BWDB_X", pd.Timestamp("2020-01-01"), 23., 90.).station_id.tolist(), ["BWDB_Y"])
        self.assertEqual(E.nearby(frame, "BWDB_X", pd.Timestamp("2020-01-01"), 23., 90., bwdb_only=False).station_id.tolist(), ["BWDB_Y", "BMD_1"])

    def test_downloaded_input_relocation_checks_content_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder); target = archive/"prepared"/"test"/"original_qc.csv"
            target.parent.mkdir(parents=True); target.write_text("original data")
            record = dict(path="/panfs/project/data/processed/surma_tail_pilot/prepared/test/original_qc.csv", sha256=E.I.sha(target))
            self.assertEqual(E.local_record(record, archive), target)
            target.write_text("changed data")
            with self.assertRaises(ValueError): E.local_record(record, archive)

    def test_stream_controls_retain_matching_gamma_spreading_and_error_budget(self):
        tree = ast.parse((ROOT/"scripts/28_simultaneous_method_sweep.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Variant")
        ns = dict(dataclass=dataclass, replace=replace, __name__=__name__)
        exec(compile(ast.Module(body=[cls], type_ignores=[]), "variant-test", "exec"), ns)
        base = ns["Variant"](name=E.I.BASELINE, huber_delta=3., prior_temperature=1., imerg_stride=1,
            gauge_component_spread_cells=6., gauge_guidance_gamma=.01, imerg_guidance_gamma=.001, secondary_r_multiplier=4.)
        ns.update(_IMPROVEMENT_BASE=base, CORE=[ns["Variant"](name="background", streams="none")])
        assignment = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "V2_EVENT_ATTRIBUTION" for t in n.targets))
        exec(compile(ast.Module(body=[assignment], type_ignores=[]), "variant-test", "exec"), ns)
        _, gauge, satellite, both = ns["V2_EVENT_ATTRIBUTION"]
        self.assertEqual(both, base)
        self.assertEqual(gauge.streams, "gauges"); self.assertEqual(gauge.guidance_spread_cells, 6.)
        self.assertIsNone(gauge.gauge_component_spread_cells)
        self.assertEqual(gauge.gauge_guidance_gamma, both.gauge_guidance_gamma)
        self.assertEqual(satellite.streams, "imerg"); self.assertEqual(satellite.guidance_spread_cells, 0.)
        self.assertEqual(satellite.imerg_guidance_gamma, both.imerg_guidance_gamma)
        for v in (gauge, satellite):
            self.assertEqual(v.huber_delta, 3.); self.assertEqual(v.secondary_r_multiplier, 4.)
            self.assertEqual(v.prior_temperature, 1.); self.assertEqual(v.imerg_stride, 1)

    def test_prepare_keeps_parent_observations_and_fails_on_changed_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); parent = root/"parent"; parent.mkdir(); review = root/"review"; review.mkdir()
            original = parent/"original_qc.csv"
            pd.DataFrame(dict(station_id=["BWDB_X"]*3, date=pd.date_range("2020-01-01", periods=3), precip_mm=[120., 80., 0.])).to_csv(original, index=False)
            stations = parent/"superob.csv"; stations.write_text("unchanged observations")
            imerg = parent/"imerg.nc"; imerg.write_bytes(b"unchanged satellite")
            held = parent/"holdout.txt"; held.write_text("BWDB_X\n")
            shared = []
            for name in ("model.pt", "config.yaml", "stats.json"):
                path = parent/name; path.write_text("placeholder"); shared.append(E.I.record(path))
            plan = dict(checkpoint=str(parent/"model.pt"), config=str(parent/"config.yaml"), stats=str(parent/"stats.json"),
                predictors="predictors.zarr", fill_known_cpc_gaps=False, members=30, seed=1, min_coverage=.8,
                held_stations=1, retained_stations=1, buffer_km=20, holdout=str(held), shared_files=shared, source_files=[],
                windows=[dict(label="parent_case", role="test", start="2020-01-01", end="2020-01-03", days=3,
                    quarter="2020_q1", stations=str(stations), representation=.53, imerg=str(imerg),
                    files=[E.I.record(p) for p in (original, stations, imerg)])])
            E.I.write_json(parent/"pilot_plan.json", plan)
            candidates = review/"supported_event_candidates.csv"
            pd.DataFrame([dict(window="parent_case", date="2020-01-01", station_id="BWDB_X", observed_mm=120.,
                source_matches=True, neighbor_supported=True, qc_reasons="")]).to_csv(candidates, index=False)
            flags = review/"review_flags_for_pilot.csv"
            pd.DataFrame(columns=["station_id", "window", "review_start", "review_end", "reason"]).to_csv(flags, index=False)
            E.I.write_json(review/"review_provenance.json", dict(pilot_plan=E.I.record(parent/"pilot_plan.json"), inputs=[E.I.record(flags)], candidate_file=E.I.record(candidates)))
            windows = root/"windows.json"; windows.write_text(json.dumps(dict(windows=[dict(label="event", parent_window="parent_case", start="2020-01-01", end="2020-01-02")])))
            args = SimpleNamespace(pilot=str(parent), out_dir=str(root/"next"), windows=str(windows), review_dir=str(review))
            with contextlib.redirect_stdout(io.StringIO()): E.prepare(args)
            prepared = json.loads((root/"next/pilot_plan.json").read_text())
            w = prepared["windows"][0]
            self.assertEqual(w["stations"], str(stations)); self.assertEqual(w["representation"], .53)
            self.assertEqual(w["files"], plan["windows"][0]["files"])
            self.assertEqual(w["days"], 2); self.assertEqual(w["role"], "retest")
            self.assertEqual(prepared["variants"], E.VARIANTS)
            pd.DataFrame([dict(station_id="BWDB_OTHER", window="parent_case", review_start="2020-01-01", review_end="2020-01-01", reason="needs review")]).to_csv(flags, index=False)
            E.I.write_json(review/"review_provenance.json", dict(pilot_plan=E.I.record(parent/"pilot_plan.json"), inputs=[E.I.record(flags)], candidate_file=E.I.record(candidates)))
            args.out_dir = str(root/"flagged")
            with self.assertRaisesRegex(ValueError, "flagged input"): E.prepare(args)
            candidates.write_text("changed source review")
            args.out_dir = str(root/"another")
            with self.assertRaises(ValueError): E.prepare(args)


if __name__ == "__main__": unittest.main()
