"""Direct same-UTC-day comparison, with independent rainfall/error validity."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bdhires.grids import BD

spec = importlib.util.spec_from_file_location("_direct_imerg_compare", ROOT / "scripts/102_compare_imerg_daily_halfhourly.py")
COMPARE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(COMPARE)


class DirectImergComparisonTests(unittest.TestCase):
    def make_inputs(self, root):
        daily, half = root / "daily", root / "half"
        daily.mkdir()
        half.mkdir()
        lat = BD.lat.reshape(64, 2).mean(axis=1)
        lon = BD.lon.reshape(64, 2).mean(axis=1)
        coords = {"lat": lat, "lon": lon}
        rates, errors = [], []
        for i in range(48):
            p = np.full((64, 64), 1. + i / 48)
            e = np.full_like(p, .2 + i / 480)
            p[2, 2] = e[2, 2] = 0  # dry values must remain valid
            rates.append(p.copy())
            errors.append(e.copy())
            if i == 0:
                p[0, 0] = np.nan
                e[1, 1] = -9999.9  # bad error must not invalidate valid rainfall
            start = np.datetime64("2022-05-01T00:00:00") + np.timedelta64(30 * i, "m")
            end = start + np.timedelta64(1799, "s")
            stamp = str(start).split("T")[1].replace(":", "")
            stop = str(end).split("T")[1].replace(":", "")
            name = f"3B-HHR.MS.MRG.3IMERG.20220501-S{stamp}-E{stop}.{i*30:04d}.V07B.HDF5.SUB.nc4"
            xr.Dataset({"precipitation": (("lat", "lon"), p, {"units": "mm/hr"}),
                        "randomError": (("lat", "lon"), e, {"units": "mm/hr"})},
                       coords=coords).to_netcdf(half / name)
        p = .5 * np.sum(rates, axis=0)
        e = np.sqrt(24 * np.mean(np.square(errors), axis=0))
        e[1, 1] = 12000.  # an upstream positive anomaly remains visible, never clipped
        pc = np.full((64, 64), 48)
        ec = pc.copy()
        ec[1, 1] = 47
        xr.Dataset({"precipitation": (("lat", "lon"), p, {"units": "mm/day"}),
                    "randomError": (("lat", "lon"), e, {"units": "mm/day"}),
                    "precipitation_cnt": (("lat", "lon"), pc),
                    "randomError_cnt": (("lat", "lon"), ec)}, coords=coords).to_netcdf(
                        daily / "3B-DAY.MS.MRG.3IMERG.20220501-S000000-E235959.V07B.nc4")
        return daily, half

    def test_same_utc_day_rainfall_and_documented_errors(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            daily, half = self.make_inputs(root)
            result = COMPARE.compare("2022-05-01", daily, half, root / "out")
            self.assertEqual(result["half_hourly_files"], 48)
            self.assertEqual(result["rainfall_daily_vs_half_hourly_total_mm_day"]["n"], 4095)
            self.assertLess(result["rainfall_daily_vs_half_hourly_total_mm_day"]["mae"], 1e-10)
            self.assertEqual(result["precip_count_mismatches"], 1)
            self.assertEqual(result["error_count_mismatches"], 0)
            self.assertLess(result["error_daily_vs_documented_daily_formula_mm_day"]["mae"], 1e-10)
            self.assertGreater(result["error_daily_vs_half_hourly_quadrature_mm_day"]["mae"], .1)
            self.assertFalse(result["error_encoding_check"]["all_cells_match"])
            self.assertEqual(result["native_distributions"]["randomError"]["above_1000"], 1)
            with xr.open_dataset(root / "out/fields.nc") as ds:
                self.assertTrue(np.isfinite(ds.rainfall_difference[1, 1]))
                self.assertTrue(np.isnan(ds.rainfall_difference[0, 0]))
                self.assertEqual(float(ds.daily_random_error[1, 1]), 12000.)
                self.assertEqual(float(ds.half_hourly_total[2, 2]), 0.)
            # Default NetCDF NaN fill metadata must be represented as JSON safely.
            data = json.loads((root / "out/comparison.json").read_text())
            self.assertEqual(data["native_metadata"]["randomError"]["encoding"]["_FillValue"], "nan")

    def test_sum_squared_encoding_is_checked_cellwise_including_zero(self):
        squared = np.array([[0., 2.76480079, 122.36616516], [4227.13916016, 10455.63755, 25409.65625]])
        raw = squared.astype(np.float32).astype(float)
        result = COMPARE.error_encoding_diagnostics(raw, squared, np.ones_like(raw, dtype=bool))
        self.assertEqual(result["matching_cells"], 6)
        self.assertTrue(result["all_cells_match"])
        self.assertLess(result["candidate_quadrature_from_daily_vs_halfhourly_mm_day"]["max_abs_difference"], .00001)
        raw[0, 1] += 1.
        result = COMPARE.error_encoding_diagnostics(raw, squared, np.ones_like(raw, dtype=bool))
        self.assertEqual(result["matching_cells"], 5)
        self.assertFalse(result["all_cells_match"])
        mask = np.ones_like(raw, dtype=bool)
        mask[0, 1] = False
        self.assertTrue(COMPARE.error_encoding_diagnostics(raw, squared, mask)["all_cells_match"])

    def test_missing_interval_fails_before_writing_results(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            daily, half = self.make_inputs(root)
            next(half.glob("*.nc4")).unlink()
            with self.assertRaisesRegex(ValueError, "missing intervals"):
                COMPARE.compare("2022-05-01", daily, half, root / "out")
            self.assertFalse((root / "out").exists())

    def test_audit_preserves_inputs_and_records_incomplete_error_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            lat = BD.lat.reshape(64, 2).mean(axis=1)
            lon = BD.lon.reshape(64, 2).mean(axis=1)
            files = []
            for i in range(5):
                path = root / f"3B-DAY.MS.MRG.3IMERG.20220{430+i if i == 0 else 500+i}-S000000-E235959.V07B.nc4"
                values = np.full((64, 64), 2.)
                values[0, 0] = 12000.
                counts = np.full((64, 64), 48)
                ec = counts.copy()
                ec[0, 0] = 47
                xr.Dataset({"precipitation": (("lat", "lon"), np.full((64, 64), 10.)),
                            "randomError": (("lat", "lon"), values),
                            "precipitation_cnt": (("lat", "lon"), counts),
                            "randomError_cnt": (("lat", "lon"), ec)},
                           coords={"lat": lat, "lon": lon}).to_netcdf(path)
                files.append(path)
            for label, value in (("reporting", 2.), ("calendar", 2000.)):
                path = root / f"imerg_{label}_s04.nc"
                xr.Dataset({"precipitation": (("time", "lat", "lon"), np.full((5, 16, 16), 10.)),
                            "randomError": (("time", "lat", "lon"), np.full((5, 16, 16), value)),
                            "precipitation_cnt": (("time", "lat", "lon"), np.full((5, 16, 16), 48))}).to_netcdf(path)
                files.append(path)
            import hashlib
            hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
            (root / "experiment.json").write_text(json.dumps({"input_sha256": hashes}))
            result = COMPARE.audit(root)
            self.assertEqual(result["calendar_to_reporting_error_ratio"]["median"], 1000.)
            self.assertEqual(result["raw_daily"][0]["accepted_precipitation_with_incomplete_error_count"], 1)
            self.assertEqual(result["raw_daily"][0]["largest_accepted_errors"][0]["randomError"], 12000.)
            self.assertEqual({str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}, hashes)
            files[-1].write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "input changed"):
                COMPARE.audit(root)


if __name__ == "__main__":
    unittest.main()
