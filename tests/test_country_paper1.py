"""Country-only scoring and map clipping checks using synthetic geometry."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bdhires.country import geometry, points_inside, grid_mask, map_layer


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / name)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class CountryTests(unittest.TestCase):
    def setUp(self):
        self.country = geometry({"type": "MultiPolygon", "coordinates": [
            [[[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]],
             [[1, 1], [1, 2], [2, 2], [2, 1], [1, 1]]],
            [[[5, 5], [6, 5], [6, 6], [5, 6], [5, 5]]]]})

    def test_polygon_holes_islands_and_no_border_buffer(self):
        self.assertEqual(points_inside([.5, 1.5, 5.5, .5, np.nan], [.5, 1.5, 5.5, 4.001, 0], self.country).tolist(),
                         [True, False, True, False, False])
        mask = grid_mask([.5, 1.5, 5.5], [.5, 1.5, 5.5], self.country)
        self.assertFalse(mask[1, 1])
        self.assertTrue(mask[2, 2])

    def test_all_scoring_arrays_use_identical_country_selection(self):
        mod = load("90_evaluate_cpcv2_paper1.py")
        with tempfile.TemporaryDirectory() as temp:
            boundary = Path(temp) / "country.geojson"
            boundary.write_text(json.dumps(self.country))
            data = {"station_lat": np.array([.5, 1.5, .5]), "station_lon": np.array([.5, 1.5, 4.001]),
                    "station": np.array(["IN", "HOLE", "OUT"]), "truth": np.array([1., 99., 1000.]),
                    "members": {"background": np.array([[2., 2.], [3., 3.], [4., 4.]])}}
            filtered, region = mod.country_samples(data, boundary)
            self.assertEqual(filtered["station"].tolist(), ["IN"])
            self.assertEqual(filtered["members"]["background"].shape, (1, 2))
            self.assertEqual(region["excluded_station_days"], 2)

    def test_raster_mask_and_vector_clip_do_not_paint_outside(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        artist = map_layer(ax, np.ones((7, 7)), np.arange(7) + .5, np.arange(7) + .5, self.country)
        self.assertTrue(np.ma.getmaskarray(artist.get_array())[0, 4])
        self.assertTrue(np.ma.getmaskarray(artist.get_array())[1, 1])
        self.assertIsNotNone(artist.get_clip_path())
        self.assertEqual(tuple(artist.cmap.get_bad()), (1., 1., 1., 1.))
        self.assertTrue(len(ax.lines) >= 3)
        plt.close(fig)

    def test_station_moment_pool_matches_direct_scores(self):
        mod = load("94_recompute_paper1_bangladesh_exports.py")
        truth = np.array([1., 3., 5., 8.])
        pred = np.array([2., 1., 7., 9.])
        error = pred - truth
        station = [{"n": 4, "n_wet": 4, "crps": 1., "mae": np.mean(abs(error)), "bias": np.mean(error),
                    "rmse": np.sqrt(np.mean(error ** 2)), "spread": 2., "coverage_90": .75, "wet_mae": np.mean(abs(error)), "dry_mae": ""}]
        months = [{"n_days": 2, "observed_mean_mm_day": y.mean(), "predicted_mean_mm_day": p.mean(),
                   "observed_daily_temporal_sd_mm": y.std(), "predicted_daily_temporal_sd_mm": p.std()}
                  for y, p in ((truth[:2], pred[:2]), (truth[2:], pred[2:]))]
        self.assertAlmostEqual(mod.pool(station, months)["correlation"], np.corrcoef(truth, pred)[0, 1])
        months[0]["n_days"] = 1
        with self.assertRaisesRegex(ValueError, "exact daily sample"):
            mod.pool(station, months)

    def test_outside_grid_values_do_not_enter_domain_statistics(self):
        mod = load("55_evaluate_v2_gridded_archive.py")
        lat = lon = np.arange(7) + .5
        mask = grid_mask(lat, lon, self.country)
        field = np.where(mask, 2., 100000.)[None]
        archive = {"lat": lat, "lon": lon, "valid": np.ones((7, 7), bool),
                   "mean": field[None], "spread": field[None],
                   "chirps": field, "cpc": field, "imerg": field}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "country.geojson"
            path.write_text(json.dumps(self.country))
            region = mod.restrict_archive_country(archive, path)
            self.assertEqual(region["scored_grid_cells"], int(mask.sum()))
            for key in ("mean", "spread", "chirps", "cpc", "imerg"):
                self.assertTrue(np.isnan(archive[key][..., ~mask]).all())
            self.assertEqual(mod.spatial_statistics(archive["chirps"], archive["valid"])["domain_mean_mm"][0], 2.)

    def test_gridded_withheld_scores_and_bundle_exclude_outside_sites(self):
        mod = load("55_evaluate_v2_gridded_archive.py")
        methods = ["background", "dense_s6_bwdb_r4"]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "evaluation").mkdir()
            arrays = {"times": np.array(["2021-06-01", "2021-06-02"]),
                      "station_ids": np.array(["IN", "OUT", "HOLE"]),
                      "station_lat": np.array([.5, .5, 1.5]), "station_lon": np.array([.5, 4.001, 1.5]),
                      "eval_idx": np.array([0, 1, 2]), "variant_names": np.array(methods),
                      "gauge_mm": np.array([[1., 1000., 1000.], [1., 1000., 1000.]]),
                      "grid_lat": np.arange(8) + .5, "grid_lon": np.arange(8) + .5,
                      "valid": np.ones((8, 8), bool), "chirps": np.ones((2, 8, 8)),
                      "condition": np.ones((2, 8, 8)), "raw_imerg_mm": np.ones((2, 4, 4))}
            for method in methods:
                arrays["station_" + method] = np.ones((2, 2, 3)) * 2
                arrays["meanfield_" + method] = np.ones((2, 8, 8)) * 2
            np.savez(root / "evaluation/2021_may_sep.npz", **arrays)
            paths = [root / "2021_may_sep.zarr"]
            rows, _ = mod.evaluate_withheld_gauges(paths, methods, 2, root, "single-holdout", self.country)
            self.assertTrue(all(r["n"] == 2 and r["rmse_mm"] == 1. for r in rows))
            bundle = mod.load_withheld_gauge_bundle(paths, methods, 2, root, "single-holdout", self.country)
            self.assertEqual(bundle["station"].tolist(), ["IN", "IN"])
            self.assertEqual(bundle["truth"].tolist(), [1., 1.])


if __name__ == "__main__":
    unittest.main()
