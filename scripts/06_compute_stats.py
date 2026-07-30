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
from bdhires.transforms import (  # noqa: E402
    CondTransform,
    PrecipTransform,
    ResidualSpec,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", required=True)
    ap.add_argument("--train-years", nargs=2, type=int, required=True)
    ap.add_argument("--transform", default="log1p", choices=["log1p", "sqrt", "cbrt", "none"])
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--sample-days", type=int, default=1500)
    ap.add_argument(
        "--no-cond-transform",
        action="store_true",
        help="reproduce pre-2026 statistics: standardise raw ERA5 channels with "
             "no variance stabilisation (not recommended; see "
             "docs/DIAGNOSIS_epoch119.md item 1)",
    )
    ap.add_argument(
        "--residual",
        action="store_true",
        help="parameterise the target as a correction to ERA5 tp in transformed "
             "space instead of the absolute field",
    )
    ap.add_argument(
        "--era5-tp-index",
        type=int,
        default=0,
        help="index of era5_tp within the conditioning stack (the residual base)",
    )
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
    channels = list(z.attrs.get("cond_channels", []))
    if len(channels) != cond.shape[1]:
        raise ValueError(
            f"condition metadata lists {len(channels)} channels but the array "
            f"contains {cond.shape[1]}"
        )

    # Residual statistics come from the RAW tp channel, before the conditioning
    # transform is applied below -- the residual base lives in precipitation
    # space (PrecipTransform), not in conditioning space.
    residual = ResidualSpec(enabled=False)
    residual_summary = None
    if args.residual:
        if not 0 <= args.era5_tp_index < cond.shape[1]:
            raise ValueError(
                f"--era5-tp-index {args.era5_tp_index} is outside the "
                f"{cond.shape[1]} conditioning channels"
            )
        base_name = channels[args.era5_tp_index]
        if "tp" not in base_name:
            raise ValueError(
                f"--era5-tp-index {args.era5_tp_index} selects {base_name!r}, "
                f"which does not look like total precipitation"
            )
        finite = np.isfinite(tgt) & valid[None]
        target_t = tf.forward(np.where(finite, tgt, 0.0).astype(np.float64))
        base_t = tf.forward(
            np.clip(cond[:, args.era5_tp_index], 0.0, None).astype(np.float64)
        )
        difference = (target_t - base_t)[finite]
        if not np.isfinite(difference).all():
            raise ValueError("the residual target contains non-finite values")
        residual = ResidualSpec(
            enabled=True,
            mean=float(difference.mean()),
            std=float(difference.std() + 1e-12),
            base_channel=int(args.era5_tp_index),
        )
        encoded = (difference - residual.mean) / residual.std
        residual_summary = dict(
            base_channel_name=base_name,
            raw_mean=residual.mean,
            raw_std=residual.std,
            # the network's actual target: should be ~N(0, 1)
            encoded_mean=float(encoded.mean()),
            encoded_std=float(encoded.std()),
            encoded_abs_max=float(np.abs(encoded).max()),
            # how much of CHIRPS the base already explains
            base_correlation=float(
                np.corrcoef(target_t[finite], base_t[finite])[0, 1]
            ),
        )
        print("residual target:", json.dumps(residual_summary, indent=2))

    # Variance-stabilise the skewed predictors BEFORE standardising, so that
    # cond_mean/cond_std describe the values the network actually sees.  The
    # spec is written into the output; PrecipDataset reads it back.
    ctf = (
        CondTransform.identity(len(channels))
        if args.no_cond_transform
        else CondTransform.for_channels(channels, eps=args.eps)
    )
    cond = ctf.forward(cond.astype(np.float64), channel_axis=1)
    if not np.isfinite(cond).all():
        raise ValueError("the conditioning transform produced non-finite values")

    cm = np.mean(cond, axis=(0, 2, 3), dtype=np.float64)
    cs = np.std(cond, axis=(0, 2, 3), dtype=np.float64) + 1e-6
    if not np.isfinite(cm).all() or not np.isfinite(cs).all() or np.any(cs <= 0):
        raise ValueError("computed condition statistics are invalid")

    stats = dict(
        precip_transform=tf.to_dict(),
        cond_transform=ctf.to_dict(),
        residual=residual.to_dict(),
        residual_summary=residual_summary,
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
