#!/usr/bin/env python
"""Compare CHIRPS, CPC and raw IMERG against BMD gauges. No network, no GPU.

Why this runs before any more sampler time is spent
---------------------------------------------------
The pooled 2021-2024 evaluation reports, against withheld BMD gauges:

    background   bias +10.30 mm/day
    IMERG only   bias  +9.88 mm/day

but ``scripts/27_fit_imerg_bias_correction.py`` measured raw IMERG against
CHIRPS at only |bias| ~ 0.49 mm/day averaged over the domain. Those two numbers
cannot both be describing the same satellite unless something between them is
responsible. There are exactly three candidates and this script separates them:

1. **CHIRPS is biased against BMD.** The prior's training target disagrees with
   the gauges, so a model that reproduces its target perfectly still scores +10
   at stations. Then the problem is the target or the accumulation window
   (CHIRPS 00-00 UTC versus BMD 03-03 UTC), not the network and not the DA.
2. **IMERG is biased against BMD at STATION LOCATIONS specifically.** The domain
   mean can be near zero while stations -- which are not uniformly placed, and
   which cluster away from the Meghalaya barrier -- sit where the satellite is
   systematically off. Then the observation is locally biased where it matters
   and the quantile map must be conditioned differently.
3. **Neither.** CHIRPS and IMERG both agree with the gauges, and the +10 mm/day
   is manufactured by the generative prior or the guided sampler. Then no amount
   of observation-side work helps and the prior has to be retrained.

Case 1 and case 3 imply completely different papers. This script decides which,
in minutes, on a login node.

It also reports the comparison stratified by intensity and per station, because
a bias that lives entirely in the heavy tail or entirely at three orographic
stations is a different finding from a uniform offset.

Usage
-----
    python scripts/30_observation_space_audit.py \
        --zarr data/processed/bd_wide_cpc.zarr \
        --imerg data/processed/bmd_imerg_eval_2024_may_jun/imerg_aligned_20240501_20240630.nc \
        --stations data/processed/bmd_imerg_eval_2024_may_jun/fold0_bmd.csv \
        --start 2024-05-01 --end 2024-06-30 \
        --out-json data/processed/observation_space_audit_2024.json \
        --out-markdown data/processed/observation_space_audit_2024.md \
        --out-plot data/processed/observation_space_audit_2024.png
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
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
import zarr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bdhires.grids import WIDE, crop_offsets, get_grid  # noqa: E402

MISSING_TOKENS = {-999.0, -99.9, 999.0, 9999.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr", required=True)
    parser.add_argument("--imerg", nargs="*", default=None, help="prepared IMERG NetCDF")
    parser.add_argument("--stations", required=True, help="canonical BMD daily CSV")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--grid", default="bd")
    parser.add_argument("--wet-threshold", type=float, default=1.0)
    parser.add_argument(
        "--max-lag",
        type=int,
        default=2,
        help="scan product-versus-gauge agreement at day lags -MAX_LAG..+MAX_LAG. "
        "The gauge timestamp is the only unambiguous one, so the lag that "
        "maximises correlation reveals each product's true day convention.",
    )
    parser.add_argument(
        "--intensity-bins",
        nargs="+",
        type=float,
        default=[0.0, 1.0, 10.0, 25.0, 50.0, 1e9],
        help="gauge-intensity bin edges (mm/day) for the stratified table",
    )
    parser.add_argument("--out-json", default="data/processed/observation_space_audit.json")
    parser.add_argument(
        "--out-markdown", default="data/processed/observation_space_audit.md"
    )
    parser.add_argument("--out-plot", default="data/processed/observation_space_audit.png")
    return parser.parse_args()


def load_zarr_time(store) -> np.ndarray:
    """Decode the packed Zarr time axis; see scripts/27 for why this is needed."""
    raw = np.asarray(store["time"][:])
    if np.issubdtype(raw.dtype, np.datetime64):
        return raw.astype("datetime64[D]")
    return raw.astype("datetime64[ns]").astype("datetime64[D]")


def bilinear_sample(
    field: np.ndarray, lat_axis: np.ndarray, lon_axis: np.ndarray,
    lat: np.ndarray, lon: np.ndarray,
) -> np.ndarray:
    """Bilinearly sample ``(T, nlat, nlon)`` at station points. Returns (T, S).

    Written in numpy so this audit has no torch dependency and can run anywhere.
    Matches ``BilinearObsOperator`` for in-domain points, which is all that the
    station filter admits.
    """
    field = np.asarray(field, dtype=np.float64)
    n_time, nlat, nlon = field.shape
    res_lat = float(lat_axis[1] - lat_axis[0])
    res_lon = float(lon_axis[1] - lon_axis[0])
    row = (np.asarray(lat, float) - float(lat_axis[0])) / res_lat
    col = (np.asarray(lon, float) - float(lon_axis[0])) / res_lon
    row0 = np.clip(np.floor(row).astype(int), 0, nlat - 1)
    col0 = np.clip(np.floor(col).astype(int), 0, nlon - 1)
    row1 = np.minimum(row0 + 1, nlat - 1)
    col1 = np.minimum(col0 + 1, nlon - 1)
    wy = np.clip(row - row0, 0.0, 1.0)[None, :]
    wx = np.clip(col - col0, 0.0, 1.0)[None, :]
    return (
        (1 - wy) * (1 - wx) * field[:, row0, col0]
        + (1 - wy) * wx * field[:, row0, col1]
        + wy * (1 - wx) * field[:, row1, col0]
        + wy * wx * field[:, row1, col1]
    )


def load_gauges(csv_path: str, dates: np.ndarray, grid) -> tuple[pd.DataFrame, np.ndarray]:
    """Station metadata and a (T, S) gauge matrix, matching bdhires load_stations."""
    frame = pd.read_csv(csv_path)
    frame.columns = [column.strip().lower() for column in frame.columns]
    required = {"station_id", "lat", "lon", "date", "precip_mm"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{csv_path} missing {sorted(required - set(frame.columns))}")
    frame["precip_mm"] = pd.to_numeric(
        frame["precip_mm"].astype(str).str.strip().replace(
            {"T": "0.05", "t": "0.05", "": None, "NA": None}
        ),
        errors="coerce",
    )
    frame.loc[frame["precip_mm"].isin(MISSING_TOKENS), "precip_mm"] = np.nan
    frame.loc[frame["precip_mm"] < 0, "precip_mm"] = np.nan
    frame["date"] = pd.to_datetime(frame["date"]).values.astype("datetime64[D]")

    meta = frame.groupby("station_id")[["lat", "lon"]].first().reset_index()
    lon_min, lat_min, lon_max, lat_max = grid.bbox
    margin = grid.res / 2
    inside = (
        (meta["lon"] > lon_min + margin) & (meta["lon"] < lon_max - margin)
        & (meta["lat"] > lat_min + margin) & (meta["lat"] < lat_max - margin)
    )
    meta = meta[inside].reset_index(drop=True)
    if meta.empty:
        raise ValueError("no stations fall inside the evaluation grid")

    if "name" in frame.columns:
        names = frame.groupby("station_id")["name"].first()
        meta["name"] = meta["station_id"].map(names).astype(str)
    else:
        meta["name"] = meta["station_id"].astype(str)

    pivot = (
        frame.pivot_table(index="date", columns="station_id", values="precip_mm", aggfunc="first")
        .reindex(index=pd.DatetimeIndex(dates), columns=meta["station_id"])
    )
    return meta, pivot.to_numpy(dtype=np.float64)


def score(predicted: np.ndarray, observed: np.ndarray, wet_threshold: float) -> dict:
    finite = np.isfinite(predicted) & np.isfinite(observed)
    if finite.sum() < 2:
        return {"n": int(finite.sum())}
    p, o = predicted[finite], observed[finite]
    difference = p - o
    return {
        "n": int(finite.sum()),
        "bias_mm": float(np.mean(difference)),
        "mae_mm": float(np.mean(np.abs(difference))),
        "rmse_mm": float(np.sqrt(np.mean(difference**2))),
        "correlation": (
            float(np.corrcoef(p, o)[0, 1]) if p.std() > 0 and o.std() > 0 else float("nan")
        ),
        "mean_predicted_mm": float(np.mean(p)),
        "mean_observed_mm": float(np.mean(o)),
        "wet_fraction_predicted": float(np.mean(p >= wet_threshold)),
        "wet_fraction_observed": float(np.mean(o >= wet_threshold)),
    }


def main() -> None:
    args = parse_args()
    grid = get_grid(args.grid)
    store = zarr.open(args.zarr, mode="r")
    zarr_time = load_zarr_time(store)

    wanted = np.arange(
        np.datetime64(args.start, "D"), np.datetime64(args.end, "D") + 1, dtype="datetime64[D]"
    )
    lookup = {value: index for index, value in enumerate(zarr_time)}
    dates = np.array([d for d in wanted if d in lookup], dtype="datetime64[D]")
    if not len(dates):
        raise ValueError(f"{args.zarr} has no dates between {args.start} and {args.end}")
    indices = [lookup[d] for d in dates]
    if len(dates) < len(wanted):
        print(f"[audit] {len(wanted) - len(dates)} requested date(s) absent from the Zarr", flush=True)

    row0, col0 = crop_offsets(WIDE, grid)
    rows, cols = slice(row0, row0 + grid.nlat), slice(col0, col0 + grid.nlon)

    meta, gauges = load_gauges(args.stations, dates, grid)
    lat, lon = meta["lat"].to_numpy(), meta["lon"].to_numpy()
    print(
        f"[audit] {len(dates)} days, {len(meta)} stations, "
        f"{int(np.isfinite(gauges).sum())} station-days with a gauge report",
        flush=True,
    )

    products: dict[str, np.ndarray] = {}

    chirps = np.stack(
        [np.asarray(store["target"][index][rows, cols], np.float32) for index in indices]
    )
    products["CHIRPS (training target)"] = bilinear_sample(chirps, grid.lat, grid.lon, lat, lon)

    cpc_index = store.attrs.get("cpc_precip_cond_index")
    if cpc_index is not None:
        cpc = np.stack(
            [
                np.asarray(store["cond"][index][int(cpc_index)][rows, cols], np.float32)
                for index in indices
            ]
        )
        products["CPC (conditioning input)"] = bilinear_sample(cpc, grid.lat, grid.lon, lat, lon)
    else:
        print("[audit] no cpc_precip_cond_index attribute; skipping CPC", flush=True)

    if args.imerg:
        frames, times = [], []
        for path in sorted(args.imerg):
            with xr.open_dataset(path) as dataset:
                times.append(np.asarray(dataset.time.values).astype("datetime64[D]"))
                frames.append(np.asarray(dataset.precipitation.values, np.float32))
                imerg_lat = np.asarray(dataset.lat.values, float)
                imerg_lon = np.asarray(dataset.lon.values, float)
        imerg_time = np.concatenate(times)
        imerg_field = np.concatenate(frames, axis=0)
        position = {value: index for index, value in enumerate(imerg_time)}
        available = np.array([d in position for d in dates])
        aligned = np.full((len(dates), *imerg_field.shape[1:]), np.nan, np.float32)
        aligned[available] = imerg_field[[position[d] for d in dates[available]]]
        products["IMERG (raw, assimilated)"] = bilinear_sample(
            aligned, imerg_lat, imerg_lon, lat, lon
        )
        if not available.all():
            print(f"[audit] IMERG missing for {int((~available).sum())} day(s)", flush=True)

    # ------------------------------------------------------------- lag scan
    #
    # Do the products and the gauges actually refer to the same 24 hours?
    #
    # IMERG is built on the BMD window: day D is [D-1 03:00 UTC, D 03:00 UTC],
    # so it is ~87% calendar day D-1. CHIRPS daily is 00-00 UTC, i.e. calendar
    # day D. Those two conventions differ by roughly a day, and nothing in the
    # packing code shifts CHIRPS to compensate. If that is a real misalignment
    # it inflates every paired statistic -- including the fitted sigma_obs,
    # which would then be measuring day-to-day rainfall variability rather than
    # retrieval error.
    #
    # A mean bias is nearly blind to a one-day shift because the climatological
    # mean barely changes overnight. Correlation is not. So scan the lag and let
    # the gauges, whose timestamp is unambiguous, arbitrate.
    lags = list(range(-args.max_lag, args.max_lag + 1))
    lag_scan: dict[str, dict] = {}
    for name, values in products.items():
        by_lag = {}
        for lag in lags:
            # lag > 0 compares product day D+lag against gauge day D
            shifted = np.full_like(values, np.nan)
            if lag == 0:
                shifted = values
            elif lag > 0:
                shifted[:-lag] = values[lag:]
            else:
                shifted[-lag:] = values[:lag]
            ok = np.isfinite(shifted) & np.isfinite(gauges)
            if ok.sum() < 30:
                continue
            a, b = shifted[ok], gauges[ok]
            by_lag[lag] = {
                "n": int(ok.sum()),
                "corr": float(np.corrcoef(a, b)[0, 1]),
                "bias_mm": float(np.mean(a - b)),
                "rmse_mm": float(np.sqrt(np.mean((a - b) ** 2))),
            }
        if by_lag:
            best = max(by_lag, key=lambda k: by_lag[k]["corr"])
            lag_scan[name] = {"by_lag": by_lag, "best_lag": best}

    if lag_scan:
        print("\n[audit] lag scan against gauges (correlation; best lag starred)", flush=True)
        header = "  " + "product".ljust(30) + "".join(f"{l:>+9d}" for l in lags)
        print(header, flush=True)
        for name, entry in lag_scan.items():
            cells = ""
            for lag in lags:
                if lag in entry["by_lag"]:
                    star = "*" if lag == entry["best_lag"] else " "
                    cells += f"{entry['by_lag'][lag]['corr']:>8.3f}{star}"
                else:
                    cells += f"{'-':>9}"
            print("  " + name.ljust(30) + cells, flush=True)
        misaligned = {n: e["best_lag"] for n, e in lag_scan.items() if e["best_lag"] != 0}
        if misaligned:
            print(
                "\n[audit] WARNING: best lag is nonzero for "
                + ", ".join(f"{n} ({l:+d} d)" for n, l in misaligned.items())
                + ". These products are compared to the gauges on the wrong day. "
                "Every paired statistic downstream -- correlation, RMSE and the "
                "fitted sigma_obs from script 27 -- is inflated until this is "
                "fixed, because the residual then contains a full day of rainfall "
                "variability on top of the actual error.",
                flush=True,
            )
        else:
            print(
                "\n[audit] all products peak at lag 0: the day labelling is "
                "consistent with the gauges and the paired statistics are sound.",
                flush=True,
            )

    overall = {name: score(values, gauges, args.wet_threshold) for name, values in products.items()}

    edges = np.asarray(args.intensity_bins, float)
    stratified: dict[str, list] = {}
    for name, values in products.items():
        rows_out = []
        for lower, upper in zip(edges[:-1], edges[1:]):
            inside = np.isfinite(gauges) & (gauges >= lower) & (gauges < upper)
            entry = score(
                np.where(inside, values, np.nan), np.where(inside, gauges, np.nan),
                args.wet_threshold,
            )
            entry["bin"] = f"[{lower:g}, {upper:g})" if upper < 1e8 else f">= {lower:g}"
            rows_out.append(entry)
        stratified[name] = rows_out

    per_station: dict[str, list] = {}
    for name, values in products.items():
        entries = []
        for index in range(len(meta)):
            entry = score(values[:, index], gauges[:, index], args.wet_threshold)
            entry.update(
                station=str(meta["name"].iloc[index]),
                lat=float(lat[index]), lon=float(lon[index]),
            )
            entries.append(entry)
        per_station[name] = entries

    # ------------------------------------------------------------------ verdict
    chirps_bias = overall["CHIRPS (training target)"].get("bias_mm", float("nan"))
    imerg_bias = overall.get("IMERG (raw, assimilated)", {}).get("bias_mm", float("nan"))
    verdict = []
    if abs(chirps_bias) > 3.0:
        verdict.append(
            f"CHIRPS itself is {chirps_bias:+.2f} mm/day against these gauges. A prior that "
            "reproduces its training target will inherit most of that, so the reported "
            "analysis bias is substantially a TARGET problem, not a network or DA problem. "
            "Check the CHIRPS 00-00 UTC versus BMD 03-03 UTC accumulation window before "
            "anything else."
        )
    else:
        verdict.append(
            f"CHIRPS agrees with these gauges to {chirps_bias:+.2f} mm/day, so the training "
            "target is sound and the +10.3 mm/day background bias is generated downstream of "
            "it -- by the prior or the sampler, not by the label."
        )
    if np.isfinite(imerg_bias):
        if abs(imerg_bias) > 3.0:
            verdict.append(
                f"Raw IMERG is {imerg_bias:+.2f} mm/day at STATION locations, far from its "
                "near-zero domain-mean bias against CHIRPS. The satellite is locally biased "
                "where the gauges are, so bias correction must be evaluated at stations, not "
                "on the domain mean."
            )
        else:
            verdict.append(
                f"Raw IMERG is only {imerg_bias:+.2f} mm/day at station locations, consistent "
                "with its domain-mean agreement with CHIRPS. The satellite is NOT the source "
                "of the +9.88 mm/day IMERG-only analysis bias -- that bias is manufactured by "
                "the prior or by the guided sampler (prior_temperature 1.25 and the Jensen "
                "effect are the leading suspects). Observation-side work will not fix it."
            )

    report = {
        "lag_scan": lag_scan,
        "scope": {
            "start": str(dates[0]), "end": str(dates[-1]), "n_days": len(dates),
            "n_stations": len(meta),
            "station_days": int(np.isfinite(gauges).sum()),
            "zarr": args.zarr, "stations": args.stations, "imerg": args.imerg,
            "wet_threshold_mm": args.wet_threshold,
        },
        "overall": overall,
        "by_intensity": stratified,
        "per_station": per_station,
        "verdict": verdict,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2, default=float))

    # ---------------------------------------------------------------- markdown
    lines = [
        f"# Observation-space audit — {dates[0]} to {dates[-1]}",
        "",
        f"{len(dates)} days · {len(meta)} stations · "
        f"{int(np.isfinite(gauges).sum())} station-days · everything compared "
        "directly against BMD gauges, no model involved.",
        "",
        "## Against BMD gauges",
        "",
        "| Product | n | Bias | MAE | RMSE | Corr | Mean pred | Mean obs | Wet frac pred/obs |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for name, entry in overall.items():
        if entry.get("n", 0) < 2:
            continue
        lines.append(
            f"| {name} | {entry['n']} | {entry['bias_mm']:+.2f} | {entry['mae_mm']:.2f} | "
            f"{entry['rmse_mm']:.2f} | {entry['correlation']:.3f} | "
            f"{entry['mean_predicted_mm']:.2f} | {entry['mean_observed_mm']:.2f} | "
            f"{entry['wet_fraction_predicted']:.2f} / {entry['wet_fraction_observed']:.2f} |"
        )
    lines += ["", "## Bias by gauge intensity", "",
              "| Product | " + " | ".join(r["bin"] for r in next(iter(stratified.values()))) + " |",
              "|:--|" + "--:|" * len(next(iter(stratified.values())))]
    for name, entries in stratified.items():
        cells = [
            f"{e['bias_mm']:+.1f}" if e.get("n", 0) >= 2 else "—" for e in entries
        ]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "## What this means", ""] + [f"- {line}" for line in verdict] + [""]
    Path(args.out_markdown).write_text("\n".join(lines))
    print("\n".join(lines))

    # ------------------------------------------------------------------ figure
    n_products = len(products)
    figure, axes = plt.subplots(2, max(n_products, 2), figsize=(6 * max(n_products, 2), 10),
                                constrained_layout=True)
    for column, (name, values) in enumerate(products.items()):
        axis = axes[0, column]
        finite = np.isfinite(values) & np.isfinite(gauges)
        axis.scatter(gauges[finite], values[finite], s=8, alpha=0.35, color="#1B4965")
        top = float(max(np.nanmax(gauges[finite]), np.nanmax(values[finite]))) if finite.any() else 1
        axis.plot([0, top], [0, top], color="black", lw=1, ls="--")
        entry = overall[name]
        axis.set_title(
            f"{name}\nbias {entry['bias_mm']:+.2f}  corr {entry['correlation']:.3f}", fontsize=10
        )
        axis.set_xlabel("BMD gauge (mm day$^{-1}$)")
        axis.set_ylabel("Product (mm day$^{-1}$)")
        axis.grid(alpha=0.2)

        axis = axes[1, column]
        station_bias = np.array(
            [e.get("bias_mm", np.nan) for e in per_station[name]], dtype=float
        )
        limit = np.nanmax(np.abs(station_bias)) if np.isfinite(station_bias).any() else 1.0
        scatter = axis.scatter(
            lon, lat, c=station_bias, cmap="RdBu_r", vmin=-limit, vmax=limit,
            s=90, edgecolor="black", linewidth=0.4,
        )
        figure.colorbar(scatter, ax=axis, label="Per-station bias (mm day$^{-1}$)")
        axis.set_title(f"{name} — where the bias lives", fontsize=10)
        axis.set_xlabel("Longitude")
        axis.set_ylabel("Latitude")
        axis.grid(alpha=0.2)
    for column in range(n_products, axes.shape[1]):
        axes[0, column].axis("off")
        axes[1, column].axis("off")
    figure.suptitle(
        f"Observation-space audit against BMD gauges — {dates[0]} to {dates[-1]} "
        "(no model, no assimilation)", fontsize=13,
    )
    Path(args.out_plot).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out_plot, dpi=150)
    print(f"\n[audit] wrote {args.out_json}, {args.out_markdown}, {args.out_plot}", flush=True)


if __name__ == "__main__":
    main()
