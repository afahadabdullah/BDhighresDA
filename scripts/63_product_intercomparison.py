"""Is daily CHIRPS a usable teacher of daily subgrid placement?

V3-SG learns 0.05-degree structure from CHIRPS because CHIRPS is the only
fine-resolution product available.  That choice is only defensible if daily
CHIRPS carries daily information.  It may not: daily CHIRPS is disaggregated
from a pentad product, so its day-to-day timing comes largely from a model
field rather than from an observation, while its spatial pattern comes from a
climatology blended with cold-cloud duration.  If that is what is happening
here, the model is being asked to reproduce a daily field whose daily component
is partly synthetic, and a near-zero daily pattern correlation is the expected
result rather than a bug.

Two hypotheses produce the same low daily number and are distinguished by how
that number behaves under temporal aggregation:

  registration  CHIRPS is displaced or mis-dated relative to the other
                products.  Correlation stays poor at every aggregation, and a
                spatial transform recovers it at daily resolution.

  timing        CHIRPS daily timing is disaggregated, not observed.  Daily
                correlation is low but rises steeply toward pentad and monthly,
                because the pentad totals *are* observed.

The distinction decides the project.  Under `registration` there is a bug to
fix.  Under `timing` the honest target is CHIRPS structure at the scale CHIRPS
resolves it, the daily placement claim is not supportable from this teacher,
and the design's own fallback -- climatological disaggregation corrected by
IMERG and BMD -- becomes the result.

    python scripts/63_product_intercomparison.py \\
      --target-store data/processed/cpc_v3_subgrid/wide_cpc.zarr \\
      --start 2015-01-01 --end 2020-12-31 \\
      --out-dir data/processed/product_intercomparison
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import zarr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bdhires.data import area_weighted_block_mean  # noqa: E402

WINDOWS = (1, 3, 5, 10, 30)


def mean_window_correlation(left, right, keep, window):
    """Spatial correlation of non-overlapping ``window``-day accumulations."""
    n = (left.shape[0] // window) * window
    if n < window:
        return float("nan"), 0
    def block(field):
        return field[:n].reshape(-1, window, *field.shape[1:]).sum(axis=1)
    a, b = block(left), block(right)
    flat = keep.reshape(-1)
    scores = []
    for x, y in zip(a, b):
        x, y = x.reshape(-1), y.reshape(-1)
        sel = flat & np.isfinite(x) & np.isfinite(y)
        if sel.sum() < 10 or x[sel].std() <= 0 or y[sel].std() <= 0:
            continue
        scores.append(float(np.corrcoef(x[sel], y[sel])[0, 1]))
    return (float(np.mean(scores)) if scores else float("nan")), len(scores)


def _shift(field, cells, axis):
    """Translate along ``axis``, filling the vacated edge with NaN.

    A wrapping roll would compare the far edge of the domain against the near
    edge and needs a margin mask, which silently discards most of a small
    domain.  Filling with NaN lets the correlation's own finite mask drop
    exactly the cells that have no counterpart, and nothing else.
    """
    out = np.full_like(field, np.nan, dtype=np.float64)
    if cells == 0:
        return field.astype(np.float64)
    src = [slice(None)] * field.ndim
    dst = [slice(None)] * field.ndim
    if cells > 0:
        dst[axis], src[axis] = slice(cells, None), slice(None, -cells)
    else:
        dst[axis], src[axis] = slice(None, cells), slice(-cells, None)
    out[tuple(dst)] = field[tuple(src)]
    return out


def spatial_search(left, right, keep, max_cells=4):
    """Best daily correlation over flips and translations, to rule out a bug."""
    variants = {
        "identity": left,
        "lat_flip": left[:, ::-1, :],
        "lon_flip": left[:, :, ::-1],
    }
    for cells in range(1, max_cells + 1):
        for sign in (+1, -1):
            variants[f"lat_shift{sign * cells:+d}"] = _shift(left, sign * cells, -2)
            variants[f"lon_shift{sign * cells:+d}"] = _shift(left, sign * cells, -1)
    return {
        name: mean_window_correlation(np.ascontiguousarray(v), right, keep, 1)[0]
        for name, v in variants.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-store", default=None)
    parser.add_argument(
        "--sample-store", default=None,
        help="resolve --target-store from this archive's source_target_store",
    )
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-shift-cells", type=int, default=4)
    parser.add_argument(
        "--alt-fine-glob", default=None,
        help="candidate replacement fine product (netCDF, lat/lon/time)",
    )
    parser.add_argument("--alt-name", default="ALT")
    parser.add_argument("--alt-var", default=None)
    args = parser.parse_args()

    target_store = args.target_store
    if args.sample_store and not target_store:
        samples = zarr.open_group(args.sample_store, mode="r")
        target_store = samples.attrs.get("source_target_store")
        if not target_store:
            raise ValueError(f"{args.sample_store} does not record source_target_store")
        print(f"resolved target store from samples: {target_store}")
    if not target_store:
        raise ValueError("pass --target-store or --sample-store")

    root = zarr.open_group(target_store, mode="r")
    # This diagnostic only READS raw arrays: no decoding, no reconstruction, no
    # smooth base.  The decoder-contract resolver deliberately refuses
    # superseded schemas, but that rule protects sampling, not a product
    # intercomparison -- and the archives worth interrogating here are exactly
    # the older ones.  Accept any subgrid schema and say which one it is.
    schema = root.attrs.get("schema", "<unset>")
    if not str(schema).startswith("cpc_v3_subgrid_v"):
        raise ValueError(
            f"{target_store} has schema {schema!r}, which is not a V3-SG target archive"
        )
    factor = int(dict(root.attrs.get("subgrid_encoding") or {}).get("factor", 10))
    args.target_store = target_store

    times = np.asarray(root["time"][:], np.int64).astype("datetime64[ns]")
    days = times.astype("datetime64[D]")
    select = np.ones(len(days), bool)
    if args.start:
        select &= days >= np.datetime64(args.start, "D")
    if args.end:
        select &= days <= np.datetime64(args.end, "D")
    index = np.flatnonzero(select)
    if index.size < max(WINDOWS):
        raise ValueError(f"only {index.size} dates selected; need at least {max(WINDOWS)}")

    channels = list(root.attrs.get("coarse_cond_channels") or [])
    candidates = [n for n in channels if "cpc" in n.lower() and "valid" not in n.lower()]
    if "sqrt_cpc_precip" in channels:
        ch = channels.index("sqrt_cpc_precip")
    elif len(candidates) == 1:
        ch = channels.index(candidates[0])
        print(f"note: using coarse conditioning channel {candidates[0]!r}")
    else:
        raise ValueError(
            f"cannot identify the CPC channel in {target_store}.\n"
            f"  available coarse_cond_channels: {channels}\n"
            "  older archives may name it differently; pass a store built by script 56"
        )
    mean = float(root.attrs["coarse_cond_mean"][ch])
    std = float(root.attrs["coarse_cond_std"][ch])

    valid = np.asarray(root["fine_valid"][:], bool)
    area = np.asarray(root["cell_area"][:], np.float32)

    print(f"store   : {args.target_store}  ({schema})")
    print(f"period  : {days[index[0]]} .. {days[index[-1]]}   ({index.size} days)")
    print("reading CHIRPS and CPC on the common 0.5-degree support...", flush=True)

    chirps_coarse, cpc_coarse = [], []
    for start in range(0, index.size, 128):
        chunk = index[start:start + 128]
        fine = np.stack([np.asarray(root["fine_mm"][int(i)], np.float32) for i in chunk])
        block, retained, _ = area_weighted_block_mean(
            torch.from_numpy(fine)[:, None], torch.from_numpy(area),
            torch.from_numpy(valid), factor=factor, valid_area_threshold=0.0,
        )
        chirps_coarse.append(block[:, 0].numpy())
        keep = retained[0, 0].numpy().astype(bool)
        root_cpc = np.stack(
            [np.asarray(root["coarse_cond"][int(i), ch], np.float32) for i in chunk]
        )
        cpc_coarse.append(np.clip(root_cpc * std + mean, 0.0, None) ** 2)
        print(f"  {min(start + 128, index.size)}/{index.size}", flush=True)
    chirps_coarse = np.concatenate(chirps_coarse)
    cpc_coarse = np.concatenate(cpc_coarse)

    results = {
        "store": args.target_store, "schema": schema,
        "period": [str(days[index[0]]), str(days[index[-1]])],
        "n_days": int(index.size),
        "chirps_vs_cpc_by_window": {}, "daily_spatial_search": {},
    }

    print("\nCHIRPS vs CPC on the common 0.5-degree support")
    print(f"  {'accumulation':<14} {'samples':>8}   r")
    curve = []
    for window in WINDOWS:
        score, count = mean_window_correlation(chirps_coarse, cpc_coarse, keep, window)
        results["chirps_vs_cpc_by_window"][str(window)] = score
        curve.append(score)
        label = "daily" if window == 1 else f"{window}-day"
        print(f"  {label:<14} {count:>8d}   {score:6.3f}")

    print("\ndaily spatial-transform search (registration check)")
    search = spatial_search(chirps_coarse, cpc_coarse, keep, args.max_shift_cells)
    results["daily_spatial_search"] = search
    ranked = sorted(search.items(), key=lambda kv: -kv[1] if np.isfinite(kv[1]) else 1.0)
    for name, score in ranked[:5]:
        flag = "  <-- as used" if name == "identity" else ""
        print(f"  {name:<14}            {score:6.3f}{flag}")

    identity = search.get("identity", float("nan"))
    best_name, best_score = ranked[0]
    daily, longest = curve[0], curve[-1]
    registration = best_name != "identity" and best_score > identity + 0.10
    timing = np.isfinite(longest) and np.isfinite(daily) and (longest - daily) > 0.20

    print()
    healthy = (
        not registration and not timing
        and np.isfinite(daily) and daily >= 0.5
    )
    if healthy:
        verdict = "healthy"
        print(
            f"VERDICT: healthy.  Identity wins and daily agreement is already\n"
            f"         {daily:.3f}.  CHIRPS is registered and carries real daily\n"
            "         information; a low model-vs-CHIRPS score is the model's."
        )
    elif registration:
        verdict = "registration"
        print(
            f"VERDICT: registration.  '{best_name}' scores {best_score:.3f} against\n"
            f"         {identity:.3f} as used.  CHIRPS is displaced relative to the\n"
            "         conditioning it is paired with.  Fix the archive; every CHIRPS\n"
            "         number so far, training loss included, used the wrong pixels."
        )
    elif timing:
        verdict = "timing"
        print(
            f"VERDICT: timing.  Daily agreement is {daily:.3f} but {WINDOWS[-1]}-day\n"
            f"         agreement is {longest:.3f}.  The products describe the same\n"
            "         rainfall at the scale CHIRPS actually resolves; its DAILY\n"
            "         component carries little independent information, which is\n"
            "         what pentad disaggregation predicts.\n"
            "         Consequence: daily subgrid PLACEMENT cannot be learned from\n"
            "         this teacher, and a near-zero daily pattern correlation is the\n"
            "         expected result, not a model failure.  Supportable claims are\n"
            "         climatological/multi-day structure from CHIRPS plus daily\n"
            "         correction from IMERG and BMD -- the fallback the design\n"
            "         already predeclares."
        )
    else:
        verdict = "inconclusive"
        print(
            f"VERDICT: inconclusive.  Daily {daily:.3f}, {WINDOWS[-1]}-day {longest:.3f},\n"
            f"         best transform '{best_name}' {best_score:.3f}.  Neither a clean\n"
            "         registration bug nor a clean timing limit.  Widen the period\n"
            "         before drawing a conclusion."
        )
    results["verdict"] = verdict

    # A candidate replacement teacher is judged by the same curve.  The point is
    # not that a product is finer or newer, it is whether its DAILY component
    # carries information that CHIRPS's does not.  A product whose curve is flat
    # -- already high at one day -- observes daily rainfall.  One that rises as
    # steeply as CHIRPS shares the same limitation and buys nothing.
    if args.alt_fine_glob:
        coarse_lat = np.asarray(root["lat"][:], np.float64).reshape(-1, factor).mean(axis=1)
        coarse_lon = np.asarray(root["lon"][:], np.float64).reshape(-1, factor).mean(axis=1)
        half = 0.5 * factor * float(np.mean(np.diff(np.asarray(root["lat"][:], np.float64))))
        import xarray as xr  # only needed for the optional alt-product path

        print(f"\nloading {args.alt_name} from {args.alt_fine_glob} ...", flush=True)
        import glob as _glob

        paths = sorted(_glob.glob(args.alt_fine_glob))
        if not paths:
            direct = Path(args.alt_fine_glob)
            if direct.exists():
                paths = [str(direct)]
            else:
                raise FileNotFoundError(
                    f"--alt-fine-glob {args.alt_fine_glob!r} matched no files.\n"
                    "  Quote the pattern so the shell does not expand it, check the\n"
                    "  directory exists, and confirm the extension (.nc / .nc4).\n"
                    "  The alt-product comparison is optional: omit --alt-fine-glob\n"
                    "  to run the CHIRPS/CPC curve on its own."
                )
        if len(paths) == 1:
            alt = xr.open_dataset(paths[0])
        else:
            # open_mfdataset needs dask; a plain concat keeps this runnable on
            # a login node with a minimal environment.
            alt = xr.concat(
                [xr.open_dataset(path) for path in paths], dim="time"
            ).sortby("time")
        alt = alt.rename({k: v for k, v in
                          (("latitude", "lat"), ("longitude", "lon"), ("valid_time", "time"))
                          if k in alt.dims or k in alt.coords})
        name = args.alt_var or max(
            (v for v in alt.data_vars if {"lat", "lon"} <= set(alt[v].dims)),
            key=lambda v: alt[v].size,
        )
        data = alt[name]
        if bool((data.lat[0] > data.lat[-1]).item()):
            data = data.sortby("lat")
        data = data.sel(
            lat=slice(coarse_lat.min() - half, coarse_lat.max() + half),
            lon=slice(coarse_lon.min() - half, coarse_lon.max() + half),
        )
        if "time" in data.dims and data.time.size > len(days):
            data = data.resample(time="1D").sum()
        alt_days = np.asarray(data.time.values).astype("datetime64[D]")
        common = np.intersect1d(alt_days, days[index])
        if common.size < max(WINDOWS):
            raise ValueError(f"{args.alt_name} overlaps only {common.size} selected days")
        data = data.sel(time=np.isin(alt_days, common))
        # Bin by cell centre onto the 0.5-degree support: 0.04 does not nest in
        # 0.5, so an exact block mean does not exist.  Centre binning with
        # latitude weights is first-order conservative and is more than adequate
        # for a correlation curve.
        edges_lat = np.r_[coarse_lat - half, coarse_lat[-1] + half]
        edges_lon = np.r_[coarse_lon - half, coarse_lon[-1] + half]
        ilat = np.clip(np.digitize(data.lat.values, edges_lat) - 1, 0, len(coarse_lat) - 1)
        ilon = np.clip(np.digitize(data.lon.values, edges_lon) - 1, 0, len(coarse_lon) - 1)
        weight = np.cos(np.deg2rad(data.lat.values))[:, None] * np.ones((1, data.lon.size))
        values = np.nan_to_num(np.asarray(data.values, np.float64), nan=0.0)
        num = np.zeros((values.shape[0], len(coarse_lat), len(coarse_lon)))
        den = np.zeros((len(coarse_lat), len(coarse_lon)))
        np.add.at(den, (ilat[:, None], ilon[None, :]), weight)
        for t in range(values.shape[0]):
            np.add.at(num[t], (ilat[:, None], ilon[None, :]), values[t] * weight)
        alt_coarse = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
        keep_alt = keep & (den > 0)
        mask = np.isin(days[index], common)
        cpc_common = cpc_coarse[mask]
        chirps_common = chirps_coarse[mask]
        print(f"\n{args.alt_name} vs CPC on the common 0.5-degree support "
              f"({common.size} shared days)")
        print(f"  {'accumulation':<14} {'samples':>8}   r")
        alt_curve = []
        for window in WINDOWS:
            score, count = mean_window_correlation(alt_coarse, cpc_common, keep_alt, window)
            alt_curve.append(score)
            results.setdefault("alt_vs_cpc_by_window", {})[str(window)] = score
            label = "daily" if window == 1 else f"{window}-day"
            print(f"  {label:<14} {count:>8d}   {score:6.3f}")
        alt_vs_chirps = mean_window_correlation(alt_coarse, chirps_common, keep_alt, 1)[0]
        results["alt_vs_chirps_daily"] = alt_vs_chirps
        results["alt_name"] = args.alt_name
        rise_chirps = curve[-1] - curve[0]
        rise_alt = alt_curve[-1] - alt_curve[0]
        print(f"\n  {args.alt_name} vs CHIRPS, daily: {alt_vs_chirps:.3f}")
        print(f"  daily->30-day rise:  CHIRPS {rise_chirps:+.3f}   "
              f"{args.alt_name} {rise_alt:+.3f}")
        if alt_curve[0] > curve[0] + 0.10 and rise_alt < rise_chirps - 0.10:
            print(f"  => {args.alt_name} carries daily information CHIRPS does not. "
                  "Worth adopting as the teacher.")
        elif alt_curve[0] <= curve[0] + 0.05:
            print(f"  => {args.alt_name} is no better daily. Switching buys nothing.")
        else:
            print(f"  => partial improvement; weigh against the shared-input risk.")
        axes_alt = alt_curve
    else:
        axes_alt = None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    axes[0].plot(WINDOWS, curve, marker="o", color="#2171b5", label="CHIRPS vs CPC")
    if axes_alt is not None:
        axes[0].plot(WINDOWS, axes_alt, marker="s", color="#d94801",
                     label=f"{args.alt_name} vs CPC")
        axes[0].legend(fontsize=8)
    axes[0].axhline(0.0, color="0.7", lw=0.8)
    axes[0].set_xscale("log")
    axes[0].set_xticks(WINDOWS, [str(w) for w in WINDOWS])
    axes[0].set_ylim(-0.1, 1.0)
    axes[0].set_xlabel("accumulation window (days)")
    axes[0].set_ylabel("mean spatial correlation")
    axes[0].set_title("CHIRPS vs CPC at 0.5°\nrising = CHIRPS daily timing is disaggregated")

    names = [name for name, _ in ranked[:9]][::-1]
    scores = [search[name] for name in names]
    colours = ["#08306b" if name == "identity" else "#c6dbef" for name in names]
    axes[1].barh(range(len(names)), scores, color=colours)
    axes[1].set_yticks(range(len(names)), names)
    axes[1].set_xlabel("daily correlation")
    axes[1].set_title("daily spatial transform search\nidentity should win if registration is right")
    figure.suptitle(f"CHIRPS / CPC intercomparison — verdict: {verdict}")
    figure.tight_layout()
    figure.savefig(out_dir / "product_intercomparison.png", dpi=160)

    (out_dir / "product_intercomparison.json").write_text(
        json.dumps(results, indent=2, sort_keys=True)
    )
    print(f"\nwrote {out_dir / 'product_intercomparison.png'}")
    print(f"wrote {out_dir / 'product_intercomparison.json'}")


if __name__ == "__main__":
    main()
