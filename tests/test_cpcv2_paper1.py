"""Publication gates and scoring checks using synthetic fixtures, not model results."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "_cpcv2_paper1", ROOT / "scripts/90_evaluate_cpcv2_paper1.py")
PAPER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PAPER
SPEC.loader.exec_module(PAPER)


class PaperEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.contract = json.loads(PAPER.DEFAULT_CONTRACT.read_text())
        self.contract["periods"] = {"2021_may_sep": ["2021-06-01", "2021-06-30"]}
        self.profile = self.contract["profiles"]["superob-final"]
        self.method = self.profile["method"]
        self.methods = ["background", self.method]
        self.dates = np.arange(np.datetime64("2021-06-01"), np.datetime64("2021-07-01"))
        ids = np.asarray(["BMD_A", "BWDB_A", "BWDB_NEAR", "SOB_1"])
        observed = np.arange(len(self.dates), dtype=float)[:, None] % 10 + np.arange(4)[None]
        noise = np.linspace(-0.5, 0.5, self.contract["members"])[None, :, None]
        self.dump = {
            "times": self.dates.astype(str),
            "model_times": (self.dates - np.timedelta64(1, "D")).astype(str),
            "station_ids": ids, "station_lat": np.arange(4, dtype=float) + 20,
            "station_lon": np.arange(4, dtype=float) + 89,
            "eval_idx": np.asarray([0, 1]), "assim_idx": np.asarray([2, 3]),
            "variant_names": np.asarray(self.methods), "gauge_mm": observed,
            "station_background": observed[:, None] + 2 + noise,
            f"station_{self.method}": observed[:, None] + 0.7 + noise,
        }
        self.report = {
            "scope": {
                "start": "2021-06-01", "end": "2021-06-30", "n_days": 30,
                "checkpoint": "runs/prior_h100_cpc_v2/best.pt",
                "checkpoint_data": "data/processed/bd_wide_cpc.zarr",
                "checkpoint_stats": "data/processed/stats_cpc_v2.json",
                "precip_transform": {"kind": "sqrt"}, "seed": 202205,
                "members": 30, "background_day_offset": -1,
                "holdout_folds": 1, "holdout_fold": 0,
                "assimilate_all_stations": False,
                "withheld_station_ids": ids[:2].tolist(),
                "group": "v2_bmd_bwdb_superob_winner",
                "analysis_sampler_n_steps": 50, "analysis_sampler_n_corrections": 2,
                "analysis_sampler_heun": True,
                "config_overrides": ["observations.imerg.factor: 8 -> 8",
                                     "observations.imerg.error_corr_cells: 0.75 -> 0.75",
                                     "observations.gauges.representativeness: 0.25 -> 0.45"],
            },
            "variants": {self.method: {"spec": self.profile["expected_spec"]}},
        }

    def tearDown(self):
        self.temp.cleanup()

    def write_fixture(self):
        folder = self.root / "stations/2021_may_sep"
        (folder / "holdout").mkdir(parents=True, exist_ok=True)
        (folder / "holdout/fold0.txt").write_text("BMD_A\nBWDB_A\n")
        (folder / "preparation_manifest.json").write_text(json.dumps({
            "analysis_selection": {"minimum_retained_neighbour_km": 3}}))
        (folder / "superob_eval_0.25.json").write_text(json.dumps({
            "stations_held_out": 2, "cell_deg": 0.25}))
        base = self.root / "evaluation/2021_may_sep"
        base.parent.mkdir(parents=True, exist_ok=True)
        np.savez(base.with_suffix(".npz"), **self.dump)
        base.with_suffix(".json").write_text(json.dumps(self.report))

    def validate(self, dump=None, report=None):
        return PAPER.validate_fold(dump or self.dump, report or self.report,
                                   self.contract, self.profile, "2021_may_sep", 0, self.methods)

    def test_rejects_assimilated_fit_overlap_and_time_leakage(self):
        report = copy.deepcopy(self.report)
        report["scope"]["assimilate_all_stations"] = True
        with self.assertRaisesRegex(ValueError, "held-out"):
            self.validate(report=report)
        dump = dict(self.dump, assim_idx=np.asarray([1, 2, 3]))
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.validate(dump=dump)
        dump = dict(self.dump, model_times=self.dates.astype(str))
        with self.assertRaisesRegex(ValueError, "D-1"):
            self.validate(dump=dump)

    def test_rejects_wrong_checkpoint_and_method_contract(self):
        report = copy.deepcopy(self.report)
        report["scope"]["checkpoint"] = "runs/prior_h100_cpc_graphflow_g0/best.pt"
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            self.validate(report=report)
        report = copy.deepcopy(self.report)
        report["variants"][self.method]["spec"]["secondary_r_multiplier"] = 1.0
        with self.assertRaisesRegex(ValueError, "secondary_r_multiplier"):
            self.validate(report=report)

    def test_archived_overrides_and_cli_entries_have_identical_effective_values(self):
        expected = {
            "observations.imerg.factor": 8,
            "observations.imerg.error_corr_cells": 0.75,
            "observations.gauges.representativeness": 0.45,
        }
        self.assertEqual(PAPER.normalize_overrides(self.report["scope"]["config_overrides"]), expected)
        self.assertEqual(PAPER.normalize_overrides([f"{key}={value}" for key, value in expected.items()]), expected)
        self.assertEqual(PAPER.normalize_overrides(expected), expected)
        self.assertEqual(PAPER.normalize_overrides([
            {"path": key, "value": value} for key, value in expected.items()]), expected)
        self.validate()

    def test_override_parser_handles_duplicates_and_rejects_malformed_records(self):
        self.assertEqual(PAPER.normalize_overrides([
            "observations.imerg.factor: 8 -> 4", "observations.imerg.factor: 4 -> 8"]),
            {"observations.imerg.factor": 8})
        for malformed in (["observations.imerg.factor"], [{"path": "x"}], "x=8"):
            with self.assertRaisesRegex(ValueError, "override"):
                PAPER.normalize_overrides(malformed)

    def test_changed_recorded_footprint_is_still_rejected(self):
        report = copy.deepcopy(self.report)
        report["scope"]["config_overrides"][0] = "observations.imerg.factor: 8 -> 4"
        with self.assertRaisesRegex(ValueError, "S04 factor"):
            self.validate(report=report)

    def test_rejects_incomplete_dates_and_synthetic_withheld_nodes(self):
        dump = dict(self.dump, times=self.dump["times"][:-1])
        with self.assertRaisesRegex(ValueError, "dates"):
            self.validate(dump=dump)
        dump = dict(self.dump, eval_idx=np.asarray([0, 3]), assim_idx=np.asarray([1, 2]))
        with self.assertRaisesRegex(ValueError, "synthetic"):
            self.validate(dump=dump)

    def test_common_samples_and_selection_exclusion(self):
        scorer = PAPER.scoring_module()
        data = {
            "date": np.asarray(["2022-05-01", "2022-06-01", "2022-06-02", "2022-06-03"],
                               dtype="datetime64[D]"),
            "station": np.asarray(["BMD_A"] * 4),
            "period": np.asarray(["2022_may_sep"] * 4),
            "source": np.asarray(["BMD"] * 4), "truth": np.full(4, 5.0),
            "members": {"background": np.full((4, 30), 7.0),
                        self.method: np.full((4, 30), 6.0)},
        }
        data["members"][self.method][2, 0] = np.nan
        rows, paired, keep, counts = PAPER.score_samples(
            data, self.methods, self.profile, scorer, [3, 7], 20, 0)
        self.assertEqual(keep.tolist(), [False, True, False, True])
        self.assertEqual(counts["scored_station_days"], 2)
        self.assertTrue(all(r["n"] == 2 for r in rows if r["group"] == "pooled"))
        self.assertAlmostEqual(paired[0]["difference"], 1.0)
        self.assertEqual(paired[0]["n_segments"], 2)

    def test_temporal_scores_require_coverage_and_omit_selected_month(self):
        data = {
            "date": np.concatenate([np.arange(np.datetime64("2022-05-01"), np.datetime64("2022-06-01")),
                                    np.arange(np.datetime64("2022-06-01"), np.datetime64("2022-07-01")),
                                    np.asarray(["2024-05-01", "2024-05-02"], dtype="datetime64[D]")]),
        }
        length = len(data["date"])
        data.update(station=np.asarray(["BMD_A"] * length), truth=np.full(length, 5.0),
                    members={name: np.full((length, 30), 5.0) for name in self.methods})
        rows = PAPER.temporal_scores(data, self.methods, self.profile)
        self.assertEqual({r["period"] for r in rows}, {"2022-06"})
        self.assertTrue(all(r["scale"] == "monthly" for r in rows))
        self.assertTrue(all("crps" not in r for r in rows))

    def test_seasonal_scores_require_full_may_sep(self):
        dates = np.arange(np.datetime64("2021-05-01"), np.datetime64("2021-10-01"))
        data = {"date": dates, "station": np.asarray(["BMD_A"] * len(dates)),
                "truth": np.ones(len(dates)),
                "members": {name: np.ones((len(dates), 30)) for name in self.methods}}
        rows = PAPER.temporal_scores(data, self.methods, self.profile)
        self.assertEqual(sum(r["scale"] == "may_sep" for r in rows), 2)
        data["date"] = dates[:61]
        data["station"] = data["station"][:61]
        data["truth"] = data["truth"][:61]
        data["members"] = {k: v[:61] for k, v in data["members"].items()}
        self.assertFalse(any(r["scale"] == "may_sep" for r in PAPER.temporal_scores(data, self.methods, self.profile)))

    def test_loader_preserves_original_gauges_and_source_labels(self):
        self.write_fixture()
        data, _, scopes, warnings = PAPER.load_samples(
            self.root, ["2021_may_sep"], self.contract, self.profile, self.methods)
        self.assertEqual(len(data["truth"]), 60)
        self.assertEqual(set(data["source"]), {"BMD", "BWDB"})
        self.assertEqual(set(data["station"]), {"BMD_A", "BWDB_A"})
        self.assertEqual(len(scopes), 1)
        self.assertTrue(any("SOB_" in warning for warning in warnings))

    def test_cluster_centroids_can_change_between_periods(self):
        self.write_fixture()
        self.contract["periods"]["2023_may_sep"] = ["2023-06-01", "2023-06-30"]
        shutil.copytree(self.root / "stations/2021_may_sep", self.root / "stations/2023_may_sep")
        second_dump = {key: value.copy() for key, value in self.dump.items()}
        for key in ("times", "model_times"):
            second_dump[key] = np.char.replace(second_dump[key], "2021", "2023")
        second_dump["station_lat"][3] += 0.01
        second_report = copy.deepcopy(self.report)
        second_report["scope"].update(start="2023-06-01", end="2023-06-30")
        second_report["scope"]["config_overrides"][0] = "observations.imerg.factor: 4 -> 8"
        second_report["scope"]["config_overrides"][2] = "observations.gauges.representativeness: 0.25 -> 0.60"
        np.savez(self.root / "evaluation/2023_may_sep.npz", **second_dump)
        (self.root / "evaluation/2023_may_sep.json").write_text(json.dumps(second_report))
        data, _, _, _ = PAPER.load_samples(self.root, list(self.contract["periods"]),
                                           self.contract, self.profile, self.methods)
        self.assertEqual(len(data["truth"]), 120)

    def test_original_five_folds_are_exhaustive_and_scored_separately(self):
        profile = self.contract["profiles"]["bmd-reference"]
        methods = ["background", profile["method"]]
        ids = np.asarray(["BMD_" + str(i) for i in range(5)])
        base_dir = self.root / "cv/2021_may_sep"
        base_dir.mkdir(parents=True)
        observed = np.ones((30, 5))
        for fold in range(5):
            dump = {
                "times": self.dump["times"], "model_times": self.dump["model_times"],
                "station_ids": ids, "station_lat": np.arange(5, dtype=float) + 20,
                "station_lon": np.arange(5, dtype=float) + 89,
                "eval_idx": np.asarray([fold]), "assim_idx": np.asarray([i for i in range(5) if i != fold]),
                "variant_names": np.asarray(methods), "gauge_mm": observed,
                "station_background": np.ones((30, 30, 5)) * 3,
                f"station_{profile['method']}": np.ones((30, 30, 5)) * 2,
            }
            report = copy.deepcopy(self.report)
            report["scope"].update(holdout_fold=fold, holdout_folds=5,
                                    withheld_station_ids=[str(ids[fold])], group="v2_confirmatory")
            report["variants"] = {profile["method"]: {"spec": profile["expected_spec"]}}
            np.savez(base_dir / f"fold{fold}.npz", **dump)
            (base_dir / f"fold{fold}.json").write_text(json.dumps(report))
        data, _, scopes, _ = PAPER.load_samples(self.root, ["2021_may_sep"],
                                               self.contract, profile, methods)
        self.assertEqual(len(data["truth"]), 150)
        self.assertEqual(len(set(data["station"])), 5)
        self.assertEqual(len(scopes), 5)

    def test_missing_input_never_produces_scores(self):
        self.write_fixture()
        contract_path = self.root / "contract.json"
        contract_path.write_text(json.dumps(self.contract))
        (self.root / "evaluation/2021_may_sep.json").unlink()
        args = ["--contract", str(contract_path), "--root", str(self.root),
                "--out-dir", str(self.root / "output")]
        self.assertEqual(PAPER.main(args), 2)
        output = self.root / "output/superob-final"
        self.assertFalse((output / "paper1_evaluation.json").exists())
        self.assertEqual(json.loads((output / "input_audit.json").read_text())["status"], "missing_inputs")

    def test_end_to_end_withheld_evaluation(self):
        self.write_fixture()
        contract_path = self.root / "contract.json"
        contract_path.write_text(json.dumps(self.contract))
        self.assertEqual(PAPER.main([
            "--contract", str(contract_path), "--root", str(self.root),
            "--out-dir", str(self.root / "output"), "--bootstrap", "20"]), 0)
        output = self.root / "output/superob-final"
        result = json.loads((output / "paper1_evaluation.json").read_text())
        self.assertEqual(result["counts"]["scored_station_days"], 60)
        self.assertEqual(result["status"], "withheld_evaluation_complete")
        self.assertTrue((output / "daily_withheld_scores.csv").is_file())
        self.assertTrue((output / "temporal_withheld_scores.csv").is_file())
        self.assertAlmostEqual(result["paired_crps"][0]["difference"], 1.3)


if __name__ == "__main__":
    unittest.main()
