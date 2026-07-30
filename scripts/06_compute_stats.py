#!/usr/bin/env python
"""Compute normalisation statistics on the TRAINING period only.

    python scripts/06_compute_stats.py --zarr data/processed/bd_wide.zarr \
        --train-years 1981 2018 --transform log1p --out data/processed/stats.json

Writes the precipitation transform constants and per-channel conditioning
mean/std.  Leaking validation/test statistics into these numbers is a classic
and hard-to-spot bug, hence the explicit year range.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bdhires.transforms import PrecipTransform  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", required=True)
    ap.add_argument("--train-years", nargs=2, type=int, required=True)
    ap.add_argument("--transform", default="log1p", choices=["log1p", "sqrt", "cbrt", "none"])
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--sample-days", type=int, default=1500)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.train_years[0] > args.train_years[1]:
        ap.error("the first --train-years value must not exceed the second")
    if args.sample_days < 1:
        ap.error("--sample-days must be positive")

    z = zarr.open(args.zarr, mode="r")
    if not z.attrs.get("complete", False):
        raise ValueError(f"{args.zarr} is not marked complete")
    time = np.asarray(z["time"][:]).astype("datetime64[ns]")
    yrs = time.astype("datetime64[Y]").astype(int) + 1970
    idx = np.where((yrs >= args.train_years[0]) & (yrs <= args.train_years[1]))[0]
    if not len(idx):
        raise ValueError(
            f"{args.zarr} contains no days in training years {args.train_years}"
        )
    rng = np.random.default_rng(0)
    sub = np.sort(rng.choice(idx, size=min(args.sample_days, len(idx)), replace=False))

    tgt = np.stack([np.asarray(z["target"][int(i)]) for i in sub])
    valid = np.asarray(z["valid"][:]) > 0
    p = tgt[:, valid]
    p = p[np.isfinite(p)]
    if not len(p):
        raise ValueError("the training sample contains no finite land precipitation")

    tf = PrecipTransform(kind=args.transform, eps=args.eps).fit(p)

    cond = np.stack([np.asarray(z["cond"][int(i)]) for i in sub])   # (N, C, H, W)
    if not np.isfinite(cond).all():
        raise ValueError("the sampled ERA5 conditions contain non-finite values")
    cm = np.mean(cond, axis=(0, 2, 3), dtype=np.float64)
    cs = np.std(cond, axis=(0, 2, 3), dtype=np.float64) + 1e-6
    channels = list(z.attrs.get("cond_channels", []))
    if len(channels) != cond.shape[1]:
        raise ValueError(
            f"condition metadata lists {len(channels)} channels but the array "
            f"contains {cond.shape[1]}"
        )
    if not np.isfinite(cm).all() or not np.isfinite(cs).all() or np.any(cs <= 0):
        raise ValueError("computed condition statistics are invalid")

    stats = dict(
        precip_transform=tf.to_dict(),
        cond_mean=cm.tolist(),
        cond_std=cs.tolist(),
        cond_channels=channels,
        static_channels=list(z.attrs.get("static_channels", [])),
        train_years=args.train_years,
        n_days_sampled=int(len(sub)),
        precip_stats=dict(
            mean_mm=float(p.mean()), wet_frac=float((p >= 1).mean()),
            p99_mm=float(np.percentile(p, 99)), max_mm=float(p.max()),
        ),
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    partial.write_text(json.dumps(stats, indent=2) + "\n")
    partial.replace(output)
    print(json.dumps({k: v for k, v in stats.items() if k != "cond_mean" and k != "cond_std"},
                     indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
