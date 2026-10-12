"""Native-product exports preserve calendar support and cannot relabel stale scores."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_script(name, pattern):
    spec = importlib.util.spec_from_file_location(name, next((ROOT / "scripts").glob(pattern)))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evidence = load_script("native_export", "96_*.py")
renderer = load_script("native_renderer", "110_*.py")


class NativePaperTests(unittest.TestCase):
    def native_case_fixture(self, folder):
        import xarray as xr
        path = Path(folder) / "imerg_july.nc"
        lat, lon = np.array([23., 23.1]), np.array([90., 90.1])
        fields = np.array([[[1., 2.], [3., 4.]], [[11., 12.], [13., 14.]]])
        dataset = xr.Dataset({"precipitation": (("time", "lat", "lon"), fields)},
                             coords={"time": np.array(["2021-07-01", "2021-07-02"], dtype="datetime64[ns]"),
                                     "lat": lat, "lon": lon},
                             attrs={"version": "V07B", "source_frequency": "half-hourly",
                                    "bmd_accumulation_end_hour_utc": 3, "window_duration_hours": 24})
        dataset.precipitation.attrs["units"] = "mm/day"
        dataset.to_netcdf(path, engine="scipy")
        native = {"lat": lat, "lon": lon, "case_date": np.asarray("2024-05-29"),
                  "case_mm": np.full((2, 2), 999.), "seasonal_mean": np.full((2, 2), 15.)}
        manifest = {"native_imerg_sources": [{"path": str(path), "sha256": renderer.digest(path)}]}
        return path, native, manifest, fields

    def test_actual_csv_date_loads_verified_native_field_without_rescoring(self):
        with tempfile.TemporaryDirectory() as folder:
            path, native, manifest, fields = self.native_case_fixture(folder)
            manifest["native_imerg_sources"][0]["path"] = "/unavailable/original/path.nc"
            actual, provenance = renderer.match_native_case(native, ["2021-07-02"], manifest, [path])
            self.assertEqual(str(actual["case_date"]), "2021-07-02")
            np.testing.assert_array_equal(actual["case_mm"], fields[1])
            np.testing.assert_array_equal(actual["seasonal_mean"], native["seasonal_mean"])
            np.testing.assert_array_equal(native["case_mm"], np.full((2, 2), 999.))
            self.assertEqual(provenance["sha256"], renderer.digest(path))

    def test_case_reload_rejects_unverified_source_or_unavailable_date(self):
        with tempfile.TemporaryDirectory() as folder:
            path, native, manifest, _ = self.native_case_fixture(folder)
            manifest["native_imerg_sources"][0]["sha256"] = "wrong hash"
            with self.assertRaisesRegex(ValueError, "source hashes"):
                renderer.match_native_case(native, ["2021-07-01"], manifest, [path])
            with self.assertRaisesRegex(ValueError, "found 0"):
                renderer.match_native_case(native, ["2021-07-03"], manifest, [path])
            with self.assertRaisesRegex(ValueError, "exactly one date"):
                renderer.match_native_case(native, ["2021-07-01", "2021-07-02"], manifest, [path])

    def test_matching_cached_case_does_not_need_original_sources(self):
        native = {"case_date": np.asarray("2021-07-01"), "case_mm": np.ones((2, 2))}
        actual, provenance = renderer.match_native_case(native, ["2021-07-01"], {})
        self.assertIs(actual, native)
        self.assertEqual(provenance["source"], "native_imerg_grids.npz")

    def test_seasonal_export_excludes_selection_and_incomplete_years(self):
        dates = np.concatenate([np.arange(f"{year}-05-01", f"{year}-10-01", dtype="datetime64[D]")
                                for year in (2021, 2022, 2023)] +
                               [np.arange("2024-05-01", "2024-07-01", dtype="datetime64[D]")])
        years = dates.astype("datetime64[Y]").astype(str)
        field = np.asarray([{"2021": 10., "2022": 1000., "2023": 20., "2024": 300.}[y] for y in years])
        native = {"time": dates, "lat": np.array([23., 23.1]), "lon": np.array([90., 90.1]),
                  "values": np.broadcast_to(field[:, None, None], (len(dates), 2, 2))}
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            evidence.export_native_grids(out, native, "2024-05-29")
            with np.load(out / "native_imerg_grids.npz", allow_pickle=False) as grids:
                np.testing.assert_array_equal(grids["seasonal_year"], [2021, 2023])
                np.testing.assert_array_equal(grids["seasonal_mean"], np.full((2, 2), 15.))
                np.testing.assert_array_equal(grids["lat"], native["lat"])
                np.testing.assert_array_equal(grids["case_mm"], np.full((2, 2), 300.))
                self.assertEqual(str(grids["case_date"]), "2024-05-29")

    def test_missing_native_inputs_and_stale_outputs_cannot_render(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            manifest = {"status": "matched_product_scores_complete", "outputs": []}
            (out / "evidence_manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "--native-imerg"):
                renderer.load_scores(out)
            manifest["native_imerg_sources"] = [{"path": "actual_native.nc"}]
            (out / "native_imerg_comparison.csv").write_text("stale old scores")
            (out / "evidence_manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "output hash"):
                renderer.load_scores(out)


if __name__ == "__main__":
    unittest.main()
