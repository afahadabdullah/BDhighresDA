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
        "--cond-transform",
        action="append",
        default=[],
        metavar="CHANNEL=KIND",
        help="override one predictor transform after applying the defaults; "
             "repeatable, with KIND in log1p,sqrt,cbrt,none",
    )
    ap.add_argument(
        "--daily-wetness",
        action="store_true",
        help="store every training day's land-mean rainfall for controlled "
             "wet-day sampling during training",
    )
    ap.add_argument(
        "--residual",
        action="store_true",
        help="parameterise the target as a transformed-space correction to the "
             "selected precipitation base",
    )
    ap.add_argument(
        "--residual-base",
        default="era5_tp",
        choices=["era5_tp", "cpc_precip", "climatology"],
        help="what the residual is taken against. 'climatology' uses the "
             "per-pixel day-of-year CHIRPS mean, which keeps the residual's "
             "well-conditioned target and non-negativity without making the "
             "prior depend on a forecast product.",
    )
    ap.add_argument(
        "--climatology-out",
        default=None,
        help="where to write the (366, H, W) day-of-year climatology "
             "(default: alongside --out as *_climatology.npy)",
    )
    ap.add_argument(
        "--climatology-smooth-days",
        type=int,
        default=15,
        help="circular running-mean window over day-of-year. Raw per-DOY means "
             "from 38 samples are far too noisy to use as a base.",
    )
    ap.add_argument(
        "--era5-tp-index",
        type=int,
        default=0,
        help="index of era5_tp within the conditioning stack (the residual base)",
    )
    ap.add_argument(
        "--residual-base-index",
        type=int,
        default=None,
        help="packed conditioning-channel index for era5_tp or cpc_precip; "
             "defaults to --era5-tp-index for backward compatibility",
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

    # -- day-of-year climatology ------------------------------------------
    # Built from ALL training days, not the 1500-day sample: a per-pixel per-DOY
    # mean has only ~38 contributing days even with the full record, so it needs
    # every one of them plus circular smoothing to be usable as a residual base.
    climatology = None
    if args.residual_base == "climatology":
        print(f"building the day-of-year climatology from {len(idx)} training days",
              flush=True)
        H, W = valid.shape
        totals = np.zeros((366, H, W), np.float64)
        counts = np.zeros(366, np.int64)
        for position, day in enumerate(idx):
            doy = int(
                (time[day].astype("datetime64[D]") - time[day].astype("datetime64[Y]"))
                .astype(int)
            )
            field = np.asarray(z["target"][int(day)], dtype=np.float32)
            totals[doy] += np.nan_to_num(field, nan=0.0)
            counts[doy] += 1
            if position % 2000 == 0:
                print(f"  {position}/{len(idx)}", flush=True)
        counts = np.maximum(counts, 1)
        climatology = (totals / counts[:, None, None]).astype(np.float32)
        # circular running mean over day-of-year
        window = max(1, int(args.climatology_smooth_days))
        if window > 1:
            padded = np.concatenate(
                [climatology[-window:], climatology, climatology[:window]], axis=0
            )
            kernel = np.ones(window, np.float64) / window
            smoothed = np.empty_like(climatology)
            for row in range(H):
                block = padded[:, row, :]
                filtered = np.apply_along_axis(
                    lambda column: np.convolve(column, kernel, mode="same"),
                    0, block,
                )
                smoothed[:, row, :] = filtered[window:-window]
            climatology = smoothed.astype(np.float32)
        climatology_path = Path(
            args.climatology_out
            or str(Path(args.out).with_suffix("")) + "_climatology.npy"
        )
        climatology_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(climatology_path, climatology)
        print(f"wrote {climatology_path}  shape {climatology.shape}  "
              f"mean {float(climatology[:, valid].mean()):.2f} mm/day", flush=True)

    # Residual statistics come from the RAW tp channel, before the conditioning
    # transform is applied below -- the residual base lives in precipitation
    # space (PrecipTransform), not in conditioning space.
    residual = ResidualSpec(enabled=False)
    residual_summary = None
    if args.residual:
        base_index = (
            args.residual_base_index
            if args.residual_base_index is not None
            else args.era5_tp_index
        )
        base_name = "doy climatology"
        if args.residual_base != "climatology":
            if not 0 <= base_index < cond.shape[1]:
                raise ValueError(
                    f"residual base index {base_index} is outside the "
                    f"{cond.shape[1]} conditioning channels"
                )
            base_name = channels[base_index]
            if base_name != args.residual_base:
                raise ValueError(
                    f"residual base index {base_index} selects {base_name!r}, "
                    f"not requested {args.residual_base!r}"
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
            if args.residual_base == "climatology":
                doys = np.array([
                    int((time[int(d)].astype("datetime64[D]")
                         - time[int(d)].astype("datetime64[Y]")).astype(int))
                    for d in sub[start : start + 64]
                ])
                base_raw = climatology[np.minimum(doys, 365)]
            else:
                base_raw = cond[start : start + 64, base_index]
            base_t = tf.forward(np.clip(base_raw, 0.0, None).astype(np.float64))[ok]
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
            base_channel=int(base_index),
            base=args.residual_base,
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
            base=args.residual_base,
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
    if args.cond_transform:
        kinds = list(ctf.kinds)
        allowed = {"log1p", "sqrt", "cbrt", "none"}
        for override in args.cond_transform:
            if "=" not in override:
                ap.error(
                    f"--cond-transform must be CHANNEL=KIND, got {override!r}"
                )
            name, kind = override.split("=", 1)
            if name not in channels:
                ap.error(
                    f"conditioning channel {name!r} is not in the packed store; "
                    f"available: {channels}"
                )
            if kind not in allowed:
                ap.error(
                    f"unknown conditioning transform {kind!r}; "
                    f"choose from {sorted(allowed)}"
                )
            kinds[channels.index(name)] = kind
        ctf = CondTransform(kinds=tuple(kinds), eps=args.eps)
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

    daily_wetness = None
    if args.daily_wetness:
        means = np.empty(len(idx), dtype=np.float32)
        print(
            f"computing daily land-mean rainfall for {len(idx)} training days",
            flush=True,
        )
        for position, day in enumerate(idx):
            field = np.asarray(z["target"][int(day)], dtype=np.float32)
            values = field[valid]
            finite = np.isfinite(values)
            if not finite.any():
                raise ValueError(
                    f"training day {time[int(day)]} has no finite land target"
                )
            means[position] = float(values[finite].mean())
            if position and position % 2000 == 0:
                print(f"  daily wetness {position}/{len(idx)}", flush=True)
        daily_wetness = {
            "time_indices": idx.astype(int).tolist(),
            "land_mean_mm_day": means.astype(float).tolist(),
        }

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
        daily_wetness=daily_wetness,
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    partial.write_text(json.dumps(stats, indent=2) + "\n")
    partial.replace(output)
    # Do not dump the complete daily-wetness vectors to a SLURM log.  They hold
    # roughly 14,000 values and can leave srun draining stdout long after Python
    # exits ("eio_handle_mainloop: Abandoning IO").  The full vectors are already
    # safely stored in the JSON file above; stdout only needs a compact summary.
    printable = {
        key: value
        for key, value in stats.items()
        if key not in {"cond_mean", "cond_std", "daily_wetness"}
    }
    if daily_wetness is not None:
        printable["daily_wetness"] = {
            "n_days": int(len(means)),
            "min_mm_day": float(means.min()),
            "median_mm_day": float(np.median(means)),
            "max_mm_day": float(means.max()),
        }
    print(json.dumps(printable, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
