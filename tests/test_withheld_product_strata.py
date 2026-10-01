"""Matched product comparisons use gauge truth, real calendar groups and exclusions."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("product_strata_eval", Path(__file__).resolve().parents[1] / "scripts/55_evaluate_v2_gridded_archive.py")
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


class ProductStrataTests(unittest.TestCase):
    def setUp(self):
        self.selection = mod.SELECTION_START, mod.SELECTION_END
        mod.SELECTION_START, mod.SELECTION_END = np.datetime64("2022-05-01"), np.datetime64("2022-05-31")
        days = np.concatenate([np.arange("2021-05-01", "2021-06-01", dtype="datetime64[D]"),
                               np.arange("2022-05-01", "2022-06-01", dtype="datetime64[D]"),
                               np.arange("2023-05-01", "2023-10-01", dtype="datetime64[D]")])
        truth = 2 + np.arange(len(days) * 2) % 3
        self.bundle = {"date": np.repeat(days, 2), "station": np.tile(["BMD_A", "BWDB_B"], len(days)),
                       "truth": truth.astype(float),
                       "members": {"background": np.repeat((truth + 1.)[:, None], 3, axis=1),
                                   "analysis": np.repeat((truth + .5)[:, None], 3, axis=1)},
                       "products": {"chirps": truth + 2., "imerg": truth + 3.,
                                    "cpc_same_day": truth + 4., "cpc": truth + 1000.}}
        self.bundle["products"]["imerg"][1] = np.nan

    def tearDown(self):
        mod.SELECTION_START, mod.SELECTION_END = self.selection

    def test_intersection_exclusion_and_same_day_cpc(self):
        rows = mod.evaluate_withheld_product_strata(self.bundle, ["background", "analysis"])
        pooled = {r["source"]: r for r in rows if r["group"] == "pooled"}
        self.assertEqual({r["n"] for r in pooled.values()}, {367})
        self.assertEqual(pooled["analysis"]["daily_sample_attrition"], 1)
        self.assertAlmostEqual(pooled["analysis"]["rmse_mm"], .5)
        self.assertAlmostEqual(pooled["cpc"]["rmse_mm"], 4.)
        self.assertFalse(any(r["group"] == "year" and r["label"] == "2022" for r in rows))
        networks = {r["label"]: r["n"] for r in rows if r["group"] == "network" and r["source"] == "analysis"}
        self.assertEqual(networks, {"BMD": 184, "BWDB": 183})

    def test_calendar_coverage_and_shared_period_means(self):
        rows = mod.evaluate_withheld_product_strata(self.bundle, ["background", "analysis"])
        groups = [r for r in rows if r["group"] == "temporal"]
        self.assertEqual({r["n"] for r in groups if r["label"] == "monthly"}, {12})
        self.assertEqual({r["n"] for r in groups if r["label"] == "may_sep"}, {2})
        self.assertTrue(all(abs(r["rmse_mm"] - .5) < 1e-12 for r in groups if r["source"] == "analysis"))
        # Remove a shared date in a full season: only that station-season is lost.
        idx = np.flatnonzero((self.bundle["date"] == np.datetime64("2023-07-01")) & (self.bundle["station"] == "BMD_A"))[0]
        self.bundle["products"]["chirps"][idx] = np.nan
        rows = mod.evaluate_withheld_product_strata(self.bundle, ["background", "analysis"])
        self.assertEqual({r["n"] for r in rows if r["group"] == "temporal" and r["label"] == "may_sep"}, {1})

    def test_lagged_cpc_and_duplicate_pairs_are_rejected(self):
        same_day = self.bundle["products"].pop("cpc_same_day")
        with self.assertRaisesRegex(ValueError, "same-day CPC"):
            mod.evaluate_withheld_product_strata(self.bundle, ["background", "analysis"])
        self.bundle["products"]["cpc_same_day"] = same_day
        self.bundle["station"][1] = "BMD_A"
        with self.assertRaisesRegex(ValueError, "duplicate"):
            mod.evaluate_withheld_product_strata(self.bundle, ["background", "analysis"])


if __name__ == "__main__":
    unittest.main()
