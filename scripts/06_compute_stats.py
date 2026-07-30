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
        # Accumulated in chunks.  Materialising float64 copies of the full
        # (N, H, W) target and base plus their difference added ~2.2 GB of
        # temporaries to an already 11 GB script and pushed the job over its
        # 16 GB limit.  Running sums keep this O(chunk) instead of O(N).
        count = 0
        sum_d = sum_dd = 0.0
        sum_t = sum_tt = sum_b = sum_bb = sum_tb = 0.0
        min_d, max_d = np.inf, -np.inf
        for start in range(0, tgt.shape[0], 64):
            chunk = tgt[start : start + 64]
            ok = np.isfinite(chunk) & valid[None]
            if not ok.any():
                continue
            target_t = tf.forward(np.where(ok, chunk, 0.0).astype(np.float64))[ok]
            base_t = tf.forward(
                np.clip(
                    cond[start : start + 64, args.era5_tp_index], 0.0, None
                ).astype(np.float64)
            )[ok]
            difference = target_t - base_t
            if not np.isfinite(difference).all():
                raise ValueError("the residual target contains non-finite values")
            count += difference.size
            sum_d += float(difference.sum())
            sum_dd += float((difference**2).sum())
            sum_t += float(target_t.sum())
            sum_tt += float((target_t**2).sum())
            sum_b += float(base_t.sum())
            sum_bb += float((base_t**2).sum())
            sum_tb += float((target_t * base_t).sum())
            min_d = min(min_d, float(difference.min()))
            max_d = max(max_d, float(difference.max()))
        if not count:
            raise ValueError("no valid land pixels for the residual statistics")

        mean_d = sum_d / count
        var_d = max(sum_dd / count - mean_d**2, 0.0)
        residual = ResidualSpec(
            enabled=True,
            mean=float(mean_d),
            std=float(np.sqrt(var_d) + 1e-12),
            base_channel=int(args.era5_tp_index),
        )
        mean_t, mean_b = sum_t / count, sum_b / count
        var_t = max(sum_tt / count - mean_t**2, 0.0)
        var_b = max(sum_bb / count - mean_b**2, 0.0)
        covariance = sum_tb / count - mean_t * mean_b
        correlation = (
            covariance / np.sqrt(var_t * var_b) if var_t > 0 and var_b > 0 else float("nan")
        )
        # The encoding is monotonic in the residual, so the extreme encoded
        # magnitude is reached at one of the two extreme raw values.
        encoded_extremes = [
            abs((value - residual.mean) / residual.std) for value in (min_d, max_d)
        ]
        residual_summary = dict(
            base_channel_name=base_name,
            n_pixels=int(count),
            raw_mean=residual.mean,
            raw_std=residual.std,
            # the network's actual target: standard by construction
            encoded_mean=0.0,
            encoded_std=1.0,
            encoded_abs_max=float(max(encoded_extremes)),
            # how much of the transformed target the base already explains
            base_correlation=float(correlation),
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
    # Transform in place, one channel at a time, staying in float32.
    # ``ctf.forward`` on the whole stack would hold a float64 input AND a float64
    # copy simultaneously -- 8.8 GB for 1500 sampled days.  Per-channel keeps the
    # temporary down to one (N, H, W) slice, and float32 is what the network
    # actually sees, so the statistics describe the real values.
    #
    # NOTE: this mutates ``cond``.  Anything needing the RAW conditioning -- the
    # residual base above -- must run before this point.
    for index, kind in enumerate(ctf.kinds):
        if kind != "none":
            cond[:, index] = ctf.forward_channel(cond[:, index], index)
    if not np.isfinite(cond).all():
        raise ValueError("the conditioning transform produced non-finite values")

    # float64 accumulators over float32 data
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
