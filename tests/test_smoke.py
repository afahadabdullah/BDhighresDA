"""pytest wrapper around scripts/smoke_test.py so CI can run it."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_smoke():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "smoke_test.py")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
