"""Ensure CPU station readers do not initialize the optional tensor backend."""
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PackageImportTests(unittest.TestCase):
    def run_fresh(self, code):
        env = dict(os.environ, PYTHONPATH=str(ROOT/"src"), PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                                env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

    def test_cpu_station_reader_does_not_even_attempt_torch_import(self):
        self.run_fresh("""
import importlib.abc
import importlib.util
import sys
attempts = []
class DetectTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            attempts.append(fullname)
sys.meta_path.insert(0, DetectTorch())
import bdhires.bmd
from bdhires import Grid, get_grid
spec = importlib.util.spec_from_file_location('production', 'scripts/99_prepare_production_stations.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert not attempts, attempts
assert 'bdhires.transforms' not in sys.modules
assert get_grid('bd').nlat == 128
""")

    def test_package_transform_export_is_still_the_real_numpy_transform(self):
        self.run_fresh("""
import bdhires
import numpy as np
from bdhires import PrecipTransform
from bdhires.transforms import PrecipTransform as DirectTransform
assert PrecipTransform is DirectTransform
assert bdhires.PrecipTransform is PrecipTransform
values = np.array([0., 1., 100.])
transform = PrecipTransform(kind='sqrt', mu=1., sd=2.)
assert np.allclose(transform.inverse(transform.forward(values)), values)
exports = {}
exec('from bdhires import *', exports)
assert exports['PrecipTransform'] is DirectTransform
try:
    bdhires.missing_attribute
except AttributeError:
    pass
else:
    raise AssertionError('unknown package attribute must fail')
""")


if __name__ == '__main__': unittest.main()
