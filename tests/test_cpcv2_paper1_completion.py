"""Missing-evidence and holdout-leakage checks using synthetic inputs only."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("completion", ROOT / "scripts/92_complete_cpcv2_paper1.py")
COMPLETE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPLETE)
EVALSPEC = importlib.util.spec_from_file_location("evaluation_tests", ROOT / "tests/test_cpcv2_paper1.py")
EVAL = importlib.util.module_from_spec(EVALSPEC)
EVALSPEC.loader.exec_module(EVAL)


class CompletionTests(unittest.TestCase):
    def test_tied_zero_ranks_and_probability_endpoints(self):
        members = np.zeros((10000, 30))
        ranks, coverage, scores, reliability = COMPLETE.calibration(members, np.zeros(10000), np.random.default_rng(7))
        self.assertEqual(ranks.sum(), 10000)
        self.assertTrue((ranks > 0).all())
        self.assertTrue(all(r["empirical"] == 1 for r in coverage))
        self.assertTrue(all(r["brier"] == 0 for r in scores))
        self.assertTrue(all(r["n"] == 10000 for r in reliability if r["bin"] == 0))
        _, _, scores, reliability = COMPLETE.calibration(np.ones((2, 30)) * 100, np.array([0., 100.]), np.random.default_rng(7))
        self.assertEqual(scores[-1]["brier"], .5)
        self.assertEqual(scores[-1]["mean_csi"], .5)
        self.assertEqual(reliability[-1]["n"], 2)
        self.assertEqual(reliability[-1]["mean_probability"], 1.)

    def test_idw_missing_and_coincident_reports(self):
        distances = np.array([[0., 10.], [1., 2.]])
        np.testing.assert_allclose(COMPLETE.idw(distances, np.array([5., 100.])), [5., 24.])
        np.testing.assert_allclose(COMPLETE.idw(distances, np.array([np.nan, 100.])), [100., 100.])
        self.assertTrue(np.isnan(COMPLETE.idw(distances, np.array([np.nan, np.nan]))).all())

    def test_selection_exclusion_is_entire_month(self):
        data = {"date": np.array(["2022-04-30", "2022-05-01", "2022-05-31", "2022-06-01"], dtype="datetime64[D]"),
                "truth": np.ones(4), "members": {"background": np.ones((4, 30))}}
        keep = COMPLETE.independent_mask(data, {"selection_start": "2022-05-01", "selection_end": "2022-05-31"})
        self.assertEqual(keep.tolist(), [True, False, False, True])

    def test_missing_inputs_remain_pending_without_score_files(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "output"
            result = COMPLETE.main(["--root", str(Path(temp) / "absent"), "--output", str(output), "--require-complete"])
            self.assertEqual(result, 2)
            manifest = json.loads((output / "completion_manifest.json").read_text())
            self.assertTrue(all(r["status"] == "pending" for r in manifest["slots"].values()))
            self.assertEqual(manifest["outputs"], [])
            self.assertFalse(list(output.glob("*.tex")))

    def test_archive_to_artifacts_and_no_withheld_idw_leakage(self):
        fixture = EVAL.PaperEvaluationTests(methodName="test_rejects_wrong_checkpoint_and_method_contract")
        fixture.setUp()
        try:
            fixture.write_fixture()
            contract_path = fixture.root / "contract.json"
            contract_path.write_text(json.dumps(fixture.contract))
            raw = []
            for day_index, day in enumerate(fixture.dates):
                for i, station in enumerate(fixture.dump["station_ids"][:3]):
                    raw.append({"date": str(day), "station_id": station,
                                "lat": fixture.dump["station_lat"][i], "lon": fixture.dump["station_lon"][i],
                                "precip_mm": fixture.dump["gauge_mm"][day_index, i]})
            COMPLETE.write_csv(fixture.root / "stations/2021_may_sep/combined_daily.csv", raw)
            output = fixture.root / "artifacts"
            self.assertEqual(COMPLETE.main(["--root", str(fixture.root), "--contract", str(contract_path), "--output", str(output), "--boundary-geojson", str(fixture.boundary)]), 0)
            manifest = json.loads((output / "completion_manifest.json").read_text())
            self.assertEqual(manifest["scored_station_days"], 60)
            for key in ("network", "calibration", "interpolation"):
                self.assertEqual(manifest["slots"][key]["status"], "generated")
            predictions = COMPLETE.read_csv(output / "interpolation_station_days.csv")
            # Only BWDB_NEAR is retained in the raw fixture: predictions are
            # its reports, never the withheld station's own rainfall.
            for row in predictions:
                day = np.where(fixture.dates == np.datetime64(row["date"]))[0][0]
                self.assertAlmostEqual(float(row["idw_mm"]), fixture.dump["gauge_mm"][day, 2])
            for file in ("fig_network.pdf", "fig_calibration.pdf"):
                self.assertTrue((output / file).read_bytes().startswith(b"%PDF"))
            # A subsequent run missing raw reports removes their stale fills.
            (fixture.root / "stations/2021_may_sep/combined_daily.csv").unlink()
            COMPLETE.main(["--root", str(fixture.root), "--contract", str(contract_path), "--output", str(output), "--boundary-geojson", str(fixture.boundary)])
            self.assertFalse((output / "tab_interpolation.tex").exists())
            self.assertFalse((output / "fig_network.pdf").exists())
        finally:
            fixture.tearDown()

    def test_history_rejects_test_dates_and_changed_cases(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "history.jsonl"
            rows = [{"epoch": i, "step": 10 * i, "members": 8, "mean_crps_mm": 4.,
                     "cases": [{"date": "2021-07-01", "quantile": .5}]} for i in (1, 2)]
            path.write_text("\n".join(json.dumps(r) for r in rows))
            with self.assertRaisesRegex(ValueError, "2019"):
                COMPLETE.build_training(path, Path(temp))
            rows[0]["cases"][0]["date"] = "2019-07-01"
            rows[1]["cases"][0]["date"] = "2020-07-01"
            path.write_text("\n".join(json.dumps(r) for r in rows))
            with self.assertRaisesRegex(ValueError, "cases"):
                COMPLETE.build_training(path, Path(temp))

    def test_compute_rejects_different_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "timing.json"
            path.write_text(json.dumps({"checkpoint_sha256": "wrong"}))
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                COMPLETE.build_compute(path, {}, Path(temp))

    def test_selection_rejects_verification_dates(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            profiles = []
            for name in ("raw", "final"):
                prefix = path / name
                np.savez(Path(str(prefix) + ".npz"), times=np.array(["2023-05-01"]))
                Path(str(prefix) + ".json").write_text(json.dumps({"scope": {
                    "assimilate_all_stations": False, "checkpoint": "runs/prior_h100_cpc_v2/best.pt"}}))
                profiles.append({"label": name, "prefix": str(prefix), "method": COMPLETE.FINAL})
            selection = path / "selection.json"
            selection.write_text(json.dumps({"profiles": profiles}))
            contract = json.loads((ROOT / "configs/paper1_cpcv2_final.json").read_text())
            with self.assertRaisesRegex(ValueError, "May 2022"):
                COMPLETE.build_selection(selection, contract, path, None, lambda p: None)

    def test_optional_artifact_generation_from_recorded_schemas(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            contract = json.loads((ROOT / "configs/paper1_cpcv2_final.json").read_text())
            profiles = []
            for name, shift in (("raw", 2.), ("final", 1.)):
                prefix = path / name
                np.savez(Path(str(prefix) + ".npz"), times=np.array(["2022-05-01", "2022-05-02"]),
                         station_ids=np.array(["BMD_A", "BWDB_B"]), eval_idx=np.array([0]), assim_idx=np.array([1]),
                         gauge_mm=np.array([[0., 8.], [10., 7.]]),
                         **{"station_" + COMPLETE.FINAL: np.broadcast_to(np.array([[0., 8.], [10., 7.]])[:, None] + shift, (2, 30, 2))})
                Path(str(prefix) + ".json").write_text(json.dumps({"scope": {
                    "assimilate_all_stations": False, "checkpoint": "runs/prior_h100_cpc_v2/best.pt",
                    "checkpoint_stats": "stats_cpc_v2.json", "seed": 7, "withheld_station_ids": ["BMD_A"]}}))
                profiles.append({"label": name, "prefix": str(prefix), "method": COMPLETE.FINAL})
            selection = path / "selection.json"
            selection.write_text(json.dumps({"profiles": profiles}))
            COMPLETE.build_selection(selection, contract, path, COMPLETE.module90().scoring_module(), lambda p: None)
            scores = COMPLETE.read_csv(path / "selection_scores.csv")
            self.assertEqual([float(r["crps"]) for r in scores], [2., 1.])
            self.assertEqual([int(r["n"]) for r in scores], [2, 2])
            history = path / "history.jsonl"
            history.write_text("\n".join(json.dumps({"epoch": i, "step": i * 10, "members": 8,
                "mean_crps_mm": 5. - i, "cases": [{"date": "2019-07-01", "quantile": .5}]}) for i in (1, 2)))
            COMPLETE.build_training(history, path)
            self.assertEqual(COMPLETE.read_csv(path / "training_curve.csv")[0]["epoch"], "2")
            compute = path / "compute.json"
            rows = [{"stage": s, "hardware": "Fixture", "gpus": 1, "wall_seconds": 3600,
                     "record_source": "Synthetic test only", "days": 10, "members": 30, "steps": 50,
                     "correctors": 2 if s == "analysis" else 0, "grid": [128, 128]}
                    for s in ("training", "background", "analysis")]
            # Use the actual evaluated identity; all timings here are temporary
            # synthetic fixtures and never written to manuscript outputs.
            compute.write_text(json.dumps({"checkpoint_sha256": "a04a3d9ae9109f905e06c32bfd55252daf1229d17c98b404e265064b89f210ea", "measurements": rows}))
            COMPLETE.build_compute(compute, contract, path)
            self.assertEqual(float(COMPLETE.read_csv(path / "compute_summary.csv")[1]["seconds_per_ensemble_day"]), 360.)
            for filename in ("tab_selection.tex", "fig_training.pdf", "tab_compute.tex"):
                self.assertTrue((path / filename).is_file())


if __name__ == "__main__":
    unittest.main()
