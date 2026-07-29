#!/usr/bin/env python
"""Fit an IMERG -> CHIRPS quantile map, and an empirical IMERG error model.

Needed only when IMERG is used as an OBSERVATION (``observations.imerg.mode:
assimilate``).  A conditional network can learn to correct IMERG's biases
implicitly; a Gaussian likelihood cannot -- an unbiased observation operator is
an assumption of the DA, not something the sampler can discover.  So if you
assimilate IMERG you must de-bias it first, otherwise you are formally
assimilating a biased observation and the analysis inherits the bias.

    python scripts/07_bias_correct_imerg.py --zarr data/processed/bd_wide.zarr \
        --train-years 2001 2018 --out data/processed/imerg_qm.npz

Method
------
Per grid cell and per season (DJF / MAM / JJAS / ON), an empirical quantile map
from the IMERG CDF to the CHIRPS CDF, fitted on the training years only.  Wet-day
frequency is matched first (frequency adaptation), because IMERG over-detects
light rain over South Asia and a naive quantile map on the full distribution
propagates that drizzle bias into the analysis.

``--fit-error-model`` additionally writes the residual sd of bias-corrected
IMERG against CHIRPS in transformed space, binned by intensity -- feed those
numbers back into ``observations.imerg.sigma_obs`` instead of guessing.
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

SEASONS = [(12, 1, 2), (3, 4, 5), (6, 7, 8, 9), (10, 11)]
SEASON_NAMES = ["DJF", "MAM", "JJAS", "ON"]


def frequency_adapt(src: np.ndarray, dst: np.ndarray, wet_thr: float = 0.1) -> np.ndarray:
    """Set the smallest IMERG values to zero so wet-day frequency matches CHIRPS."""
    n_wet_dst = int(np.isfinite(dst).sum() * np.nanmean(dst >= wet_thr))
    if n_wet_dst <= 0 or not np.isfinite(src).any():
        return np.zeros_like(src)
    finite = src[np.isfinite(src)]
    if len(finite) <= n_wet_dst:
        return src
    cut = np.partition(finite, len(finite) - n_wet_dst)[len(finite) - n_wet_dst]
    out = src.copy()
    out[out < cut] = 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", required=True)
    ap.add_argument("--train-years", nargs=2, type=int, required=True)
    ap.add_argument("--n-quantiles", type=int, default=51)
    ap.add_argument("--wet-threshold", type=float, default=0.1)
    ap.add_argument("--fit-error-model", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    z = zarr.open(args.zarr, mode="r")
    time = np.asarray(z["time"][:]).astype("datetime64[ns]")
    yrs = time.astype("datetime64[Y]").astype(int) + 1970
    months = time.astype("datetime64[M]").astype(int) % 12 + 1
    ic = int(z.attrs["imerg_cond_index"])
    valid = np.asarray(z["valid"][:]) > 0

    train = (yrs >= args.train_years[0]) & (yrs <= args.train_years[1])
    qs = np.linspace(0.0, 1.0, args.n_quantiles)

    H, W = valid.shape
    q_src = np.zeros((len(SEASONS), args.n_quantiles, H, W), np.float32)
    q_dst = np.zeros_like(q_src)
    resid = []

    for si, sm in enumerate(SEASONS):
        sel = np.where(train & np.isin(months, sm))[0]
        if len(sel) == 0:
            continue
        print(f"{SEASON_NAMES[si]}: {len(sel)} days", flush=True)
        chirps = np.stack([np.asarray(z["target"][int(j)]) for j in sel])
        imerg = np.stack([np.asarray(z["cond"][int(j)][ic]) for j in sel])

        for i in range(H):
            for j in range(W):
                if not valid[i, j]:
                    continue
                s = imerg[:, i, j]
                d = chirps[:, i, j]
                s = frequency_adapt(s, d, args.wet_threshold)
                m = np.isfinite(s) & np.isfinite(d)
                if m.sum() < 30:
                    continue
                q_src[si, :, i, j] = np.quantile(s[m], qs)
                q_dst[si, :, i, j] = np.quantile(d[m], qs)

        if args.fit_error_model:
            corr = np.empty_like(imerg)
            for i in range(H):
                for j in range(W):
                    corr[:, i, j] = (
                        np.interp(imerg[:, i, j], q_src[si, :, i, j], q_dst[si, :, i, j])
                        if valid[i, j] else np.nan
                    )
            tf = PrecipTransform(kind="log1p").fit(chirps[np.isfinite(chirps)])
            r = tf.forward(np.nan_to_num(corr)) - tf.forward(np.nan_to_num(chirps))
            r = r[:, valid]
            lvl = np.nan_to_num(chirps[:, valid])
            for lo, hi in [(0, 1), (1, 10), (10, 50), (50, 1e9)]:
                m = (lvl >= lo) & (lvl < hi) & np.isfinite(r)
                if m.sum():
                    resid.append(dict(season=SEASON_NAMES[si], lo=lo, hi=hi,
                                      sigma=float(r[m].std()), n=int(m.sum())))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, q_src=q_src, q_dst=q_dst,
                        seasons=np.array(SEASONS, dtype=object), quantiles=qs)
    print(f"wrote {args.out}")

    if resid:
        p = Path(args.out).with_suffix(".error_model.json")
        p.write_text(json.dumps(resid, indent=2))
        print(f"wrote {p}")
        print("Use these sigmas for observations.imerg.sigma_obs "
              "(they are already in transformed units):")
        for r in resid:
            print(f"  {r['season']:5s} {r['lo']:>5g}-{r['hi']:<6g} mm  sigma={r['sigma']:.3f}")


if __name__ == "__main__":
    main()
