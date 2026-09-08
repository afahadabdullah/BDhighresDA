#!/usr/bin/env python3
"""Run gfs_joint_iw125 GraphFlow DA and generate 2D spatial diagnostic maps.

Convenience entry point for the ``gfs_joint_iw125`` assimilation arm
(simultaneous joint flow guidance with 1.25x IMERG likelihood weight).

Usage:
1. Generate plots directly from screening dump:
       python scripts/run_gfs_joint_iw125_da.py \
           --plot-only \
           --dump runs/graphflow_g0_multimesh/da_screen_may2022/graphflow_g0_multimesh/fold0.npz \
           --out-dir runs/graphflow_g0_multimesh/da_screen_may2022/gfs_joint_iw125_diagnostics

2. Or run targeted DA on GPU and plot:
       python scripts/run_gfs_joint_iw125_da.py \
           --graphflow-ckpt runs/prior_h100_cpc_graphflow_g0/best.pt \
           --unet-ckpt runs/prior_h100_cpc_v2/best.pt \
           --stations data/processed/v2_simultaneous_refinement/ing2022_s04/fold0_bmd.csv \
           --imerg data/processed/imerg_prepared_ing2022/imerg_0p4deg_20220501_20220510.nc \
           --out-dir runs/graphflow_g0_multimesh/gfs_joint_iw125_da
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location("runner", ROOT / "scripts/run_gfs_twostep_l100_da.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

if __name__ == "__main__":
    if "--arm" not in sys.argv:
        sys.argv.extend(["--arm", "gfs_joint_iw125"])
    runner.main()
