#!/usr/bin/env python
"""Gauge-as-truth error budget for CHIRPS, CPC and IMERG.

The problem this exists to fix
------------------------------
Every real-data score in this project so far has been quoted against CHIRPS,
which is not truth.  CHIRPS is a satellite estimate blended with station
reports, it ingests gauges that may include BMD's, and it disagrees with the BMD
network by -59.9 mm/day above 50 mm/day.  Quoting "error vs CHIRPS" bundles
three separate things into one number:

    observed^2  =  model^2  +  representativeness^2  +  reference^2

This script treats the GAUGE as truth -- the only measurement here that is
actually a measurement of rainfall -- and separates the remaining terms two
ways.

1. Spatially, via the variogram.  Station pairs give the spatial structure of
   daily rainfall; integrating the fitted variogram over one grid cell gives the
   expected point-minus-cell difference directly.  See
   ``bdhires.eval.representativeness``.  This calibrates the
   ``representativeness`` term in R and ``obs_sd_for_verification``, which are
   currently guesses (0.25 and 0.10, in transformed units, with nothing behind
   them).

2. Temporally, via aggregation.  Averaging over N days divides any random error
   by N but leaves a persistent offset untouched, so fitting
   ``MSE(N) = systematic + random/N`` across daily, pentad, 10-day and monthly
   means separates the floor from the noise.  The floor is what no amount of
   averaging can remove: real product bias plus the systematic part of the
   point-vs-cell mismatch (a valley gauge whose cell includes a ridge).

What comes out
--------------
* ``representativeness`` and ``obs_sd_for_verification`` in transformed units,
  ready to paste into the DA configs, with the fit they came from.
* A per-product error budget at station scale, daily through monthly.
* The threshold above which CHIRPS-referenced verification is not meaningful.
* Station time series and maps for every product against the gauge.

Alignment
---------
CHIRPS and CPC are 00-00 UTC calendar days; a BMD day D is the 24 h ending
03:00 UTC on D.  Measured over 43,781 station-days, both peak against the gauges
at lag -1, so ``--chirps-day-offset``/``--cpc-day-offset`` default to -1.
Prepared IMERG is already built on the exact BMD window, so it uses lag 0.

Example
-------
    python scripts/35_gauge_truth_error_budget.py \\
        --stations data/processed/bmd_daily_2020_2025.csv \\
        --zarr data/processed/bd_wide_cpc.zarr \\
        --stats data/processed/stats_cpc.json \\
        --start 2020-01-01 --end 2025-12-31 \\
        --out-dir data/processed/error_budget
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bdhires.eval.representativeness import (  # noqa: E402
    aggregation_decomposition,
    empirical_variogram,
    fit_variogram,
    representativeness_sigma,
)
from bdhires.grids import get_grid  # noqa: E402
from bdhires.transforms import PrecipTransform  # noqa: E402

WET_THRESHOLD_MM = 1.0
CATEGORICAL_THRESHOLDS = (1.0, 10.0, 25.0, 50.0)
AGGREGATION_WINDOWS = (1, 5, 10, 30)


# --------------------------------------------------------------------------
# loading


def load_gauges(csv_path: str, start: str, end: str, min_coverage: float):
    """Wide ``(T, S)`` gauge matrix plus station metadata, gauge = truth.

    Deliberately does NOT drop stations near the domain edge the way
    ``bdhires.data.load_stations`` does: this script samples products at station
    points for verification rather than building an assimilation operator, and a
    coastal gauge is one of the more informative places to check a product.
    """
    frame = pd.read_csv(csv_path)
    frame.columns = [c.strip().lower() for c in frame.columns]
    required = {"station_id", "lat", "lon", "date", "precip_mm"}
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"{csv_path} is missing columns: {sorted(missing)}")

    frame["precip_mm"] = pd.to_numeric(
        frame["precip_mm"].astype(str).str.strip().replace(
            {"T": "0.05", "t": "0.05", "": None, "NA": None}
        ),
        errors="coerce",
    )
    frame.loc[frame["precip_mm"] < 0, "precip_mm"] = np.nan
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame[(frame["date"] >= start) & (frame["date"] <= end)]
    if frame.empty:
        raise SystemExit(f"no gauge records between {start} and {end}")

    meta = frame.groupby("station_id")[["lat", "lon"]].first().reset_index()
    if "station" in frame.columns:
        names = frame.groupby("station_id")["station"].first()
        meta["name"] = meta["station_id"].map(names).fillna(meta["station_id"].astype(str))
    else:
        meta["name"] = meta["station_id"].astype(str)

    dates = pd.date_range(start, end, freq="D")
    wide = (
        frame.pivot_table(
            index="date", columns="station_id", values="precip_mm", aggfunc="mean"
        )
        .reindex(dates)
        .reindex(columns=meta["station_id"].values)
    )

    coverage = wide.notna().mean(axis=0).values
    keep = coverage >= min_coverage
    if (~keep).any():
        print(
            f"[gauges] dropping {int((~keep).sum())} station(s) below "
            f"{min_coverage:.0%} coverage",
            flush=True,
        )
    meta = meta.loc[keep].reset_index(drop=True)
    meta["coverage"] = coverage[keep]
    wide = wide.loc[:, meta["station_id"].values]

    print(
        f"[gauges] {len(meta)} stations, {len(dates)} days, "
        f"{int(wide.notna().values.sum()):,} station-days",
        flush=True,
    )
    return dates, meta, wide.to_numpy(np.float64)


def bilinear_sample(field: np.ndarray, grid, lat, lon) -> np.ndarray:
    """Sample ``(T, H, W)`` at station points; NaN-safe, clamped at the edge.

    Written in numpy rather than reusing the torch observation operator so the
    whole script runs on a login node without a GPU environment.  Edge clamping
    matches ``padding_mode="border"`` there, so a coastal station reads the
    nearest cell instead of dropping out.
    """
    field = np.asarray(field, dtype=np.float64)
    n_time = field.shape[0]
    row = (np.asarray(lat, float) - grid.lat[0]) / grid.res
    col = (np.asarray(lon, float) - grid.lon[0]) / grid.res
    row = np.clip(row, 0, grid.nlat - 1.000001)
    col = np.clip(col, 0, grid.nlon - 1.000001)
    r0, c0 = np.floor(row).astype(int), np.floor(col).astype(int)
    r1, c1 = np.minimum(r0 + 1, grid.nlat - 1), np.minimum(c0 + 1, grid.nlon - 1)
    wr, wc = row - r0, col - c0

    out = np.full((n_time, len(row)), np.nan)
    for t in range(n_time):
        plane = field[t]
        corners = np.stack(
            [plane[r0, c0], plane[r0, c1], plane[r1, c0], plane[r1, c1]]
        )
        weights = np.stack(
            [(1 - wr) * (1 - wc), (1 - wr) * wc, wr * (1 - wc), wr * wc]
        )
        weights = np.where(np.isfinite(corners), weights, 0.0)
        total = weights.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[t] = np.nansum(corners * weights, axis=0) / total
        out[t] = np.where(total > 0, out[t], np.nan)
    return out


def load_gridded_products(
    zarr_path: str,
    dates: pd.DatetimeIndex,
    grid,
    meta: pd.DataFrame,
    chirps_offset: int,
    cpc_offset: int,
) -> dict:
    """CHIRPS and CPC sampled at the stations, each on its own day alignment."""
    import zarr

    store = zarr.open(zarr_path, mode="r")
    times = np.asarray(store["time"][:]).astype("datetime64[D]")
    index_of = {np.datetime64(t, "D"): i for i, t in enumerate(times)}

    cond_names = list(store.attrs.get("cond_channels", []))
    cpc_index = cond_names.index("cpc_precip") if "cpc_precip" in cond_names else None
    if cpc_index is None:
        print("[products] no cpc_precip channel in the store; skipping CPC", flush=True)

    # The store is the 256x256 wide domain; the analysis grid is a fixed crop.
    from bdhires.grids import WIDE, crop_offsets

    row0, col0 = crop_offsets(WIDE, grid)
    slices = (slice(row0, row0 + grid.nlat), slice(col0, col0 + grid.nlon))

    out = {}
    for label, offset, reader in (
        ("chirps", chirps_offset, lambda i: np.asarray(store["target"][i])[slices]),
        (
            "cpc",
            cpc_offset,
            None
            if cpc_index is None
            else (lambda i: np.asarray(store["cond"][i][cpc_index])[slices]),
        ),
    ):
        if reader is None:
            continue
        stack = np.full((len(dates), grid.nlat, grid.nlon), np.nan, np.float32)
        found = 0
        for position, date in enumerate(dates):
            key = np.datetime64(date + pd.Timedelta(days=offset), "D")
            if key in index_of:
                stack[position] = reader(index_of[key])
                found += 1
        print(
            f"[products] {label}: {found}/{len(dates)} days at day offset {offset:+d}",
            flush=True,
        )
        out[label] = {
            "grid": stack,
            "at_stations": bilinear_sample(stack, grid, meta["lat"], meta["lon"]),
            "day_offset": offset,
        }
    return out


def load_imerg_product(
    imerg_paths, dates: pd.DatetimeIndex, grid, meta: pd.DataFrame, factor: int
) -> dict | None:
    """Prepared IMERG footprints, upsampled to the fine grid for sampling.

    Accepts SEVERAL files, because IMERG is prepared per evaluation period in
    this project rather than as one continuous archive.  Days are merged onto
    the requested calendar; if two files cover the same day the later file in
    the list wins and the overlap is reported, since silently averaging two
    different preparations of the same day would hide a preparation bug.

    Prepared IMERG is already accumulated on the exact BMD 03-03 UTC window, so
    no day offset applies -- unlike CHIRPS and CPC, which are 00-00 UTC and need
    lag -1.  Footprints are nearest-neighbour expanded rather than interpolated:
    a footprint IS a box average, and smoothing it would invent structure the
    product does not have.
    """
    import xarray as xr

    if isinstance(imerg_paths, (str, Path)):
        imerg_paths = [imerg_paths]
    imerg_paths = [Path(p) for p in imerg_paths]

    stack = np.full((len(dates), grid.nlat, grid.nlon), np.nan, np.float32)
    position_of = {np.datetime64(d, "D"): i for i, d in enumerate(dates)}
    filled = np.zeros(len(dates), dtype=bool)
    factors_seen: set[int] = set()
    total_new = 0
    total_overlap = 0

    for path in imerg_paths:
        with xr.open_dataset(path) as dataset:
            available = np.asarray(dataset["time"].values).astype("datetime64[D]")
            precipitation = np.asarray(dataset["precipitation"].values, np.float32)

        coarse_nlat, coarse_nlon = precipitation.shape[1:]
        file_factor = grid.nlat // coarse_nlat
        if file_factor != factor:
            print(
                f"[products] imerg {path.name}: {coarse_nlat}x{coarse_nlon} implies "
                f"factor {file_factor}, not the requested {factor}; using the file",
                flush=True,
            )
        factors_seen.add(file_factor)

        new = overlap = 0
        for time_index, day in enumerate(available):
            position = position_of.get(np.datetime64(day, "D"))
            if position is None:
                continue
            if filled[position]:
                overlap += 1
            else:
                new += 1
            coarse = precipitation[time_index]
            expanded = np.repeat(
                np.repeat(coarse, file_factor, axis=0), file_factor, axis=1
            )
            stack[position, : expanded.shape[0], : expanded.shape[1]] = expanded[
                : grid.nlat, : grid.nlon
            ]
            filled[position] = True
        total_new += new
        total_overlap += overlap
        print(
            f"[products] imerg {path.name}: {new} new day(s)"
            + (f", {overlap} overlapping (later file wins)" if overlap else ""),
            flush=True,
        )

    if not total_new:
        print("[products] imerg: no overlapping days with the request; skipping",
              flush=True)
        return None
    if len(factors_seen) > 1:
        print(
            f"[products] imerg WARNING: files mix footprint factors {sorted(factors_seen)}. "
            "Representativeness differs by footprint size, so the pooled IMERG score "
            "is an average over two different observation scales.",
            flush=True,
        )
    print(
        f"[products] imerg: {int(filled.sum())}/{len(dates)} days at day offset +0 "
        f"from {len(imerg_paths)} file(s), factor {sorted(factors_seen)}",
        flush=True,
    )
    return {
        "grid": stack,
        "at_stations": bilinear_sample(stack, grid, meta["lat"], meta["lon"]),
        "day_offset": 0,
        "factor": sorted(factors_seen),
        "n_files": len(imerg_paths),
    }


# --------------------------------------------------------------------------
# metrics, all with the GAUGE as truth


def paired_scores(estimate: np.ndarray, gauge: np.ndarray) -> dict:
    """Deterministic scores of one product against the gauge."""
    ok = np.isfinite(estimate) & np.isfinite(gauge)
    if ok.sum() < 2:
        return {"n": int(ok.sum())}
    e, g = estimate[ok], gauge[ok]
    difference = e - g
    out = {
        "n": int(ok.sum()),
        "bias_mm": float(np.mean(difference)),
        "median_bias_mm": float(np.median(difference)),
        "mae_mm": float(np.mean(np.abs(difference))),
        "rmse_mm": float(np.sqrt(np.mean(difference**2))),
        "mse_mm2": float(np.mean(difference**2)),
        "gauge_mean_mm": float(np.mean(g)),
        "product_mean_mm": float(np.mean(e)),
        "correlation": (
            float(np.corrcoef(e, g)[0, 1]) if e.std() > 0 and g.std() > 0 else float("nan")
        ),
        "wet_fraction_product": float(np.mean(e >= WET_THRESHOLD_MM)),
        "wet_fraction_gauge": float(np.mean(g >= WET_THRESHOLD_MM)),
    }
    for threshold in CATEGORICAL_THRESHOLDS:
        hits = float(np.sum((e >= threshold) & (g >= threshold)))
        misses = float(np.sum((e < threshold) & (g >= threshold)))
        false_alarms = float(np.sum((e >= threshold) & (g < threshold)))
        out[f"pod_{threshold:g}"] = hits / (hits + misses) if hits + misses else float("nan")
        out[f"far_{threshold:g}"] = (
            false_alarms / (hits + false_alarms) if hits + false_alarms else float("nan")
        )
        out[f"csi_{threshold:g}"] = (
            hits / (hits + misses + false_alarms)
            if hits + misses + false_alarms
            else float("nan")
        )
        out[f"bias_above_{threshold:g}_mm"] = (
            float(np.mean(difference[g >= threshold]))
            if np.any(g >= threshold)
            else float("nan")
        )
        out[f"n_above_{threshold:g}"] = int(np.sum(g >= threshold))
    return out


def aggregate_time(values: np.ndarray, window: int, min_valid_fraction: float = 0.8):
    """Non-overlapping block means down the time axis of a ``(T, S)`` array."""
    if window == 1:
        return values
    n_blocks = values.shape[0] // window
    if n_blocks == 0:
        return np.empty((0, values.shape[1]))
    trimmed = values[: n_blocks * window].reshape(n_blocks, window, values.shape[1])
    valid = np.isfinite(trimmed).sum(axis=1)
    with np.errstate(invalid="ignore"):
        means = np.nanmean(trimmed, axis=1)
    return np.where(valid >= min_valid_fraction * window, means, np.nan)


def scores_by_window(
    product: np.ndarray, gauge: np.ndarray, windows=AGGREGATION_WINDOWS
) -> dict:
    """Scores at each aggregation window, plus the systematic/random split.

    Both series are aggregated with the SAME block boundaries and the same
    minimum-valid rule, so a block that the gauge cannot support is dropped from
    both rather than silently comparing different days.
    """
    per_window = {}
    for window in windows:
        pair_mask = np.isfinite(product) & np.isfinite(gauge)
        product_masked = np.where(pair_mask, product, np.nan)
        gauge_masked = np.where(pair_mask, gauge, np.nan)
        scores = paired_scores(
            aggregate_time(product_masked, window).ravel(),
            aggregate_time(gauge_masked, window).ravel(),
        )
        per_window[str(window)] = scores

    usable = [
        (int(w), per_window[str(w)]["mse_mm2"])
        for w in windows
        if per_window[str(w)].get("n", 0) > 10 and np.isfinite(per_window[str(w)].get("mse_mm2", np.nan))
    ]
    decomposition = None
    if len(usable) >= 2:
        decomposition = aggregation_decomposition(
            np.array([w for w, _ in usable], float),
            np.array([m for _, m in usable], float),
        )
    return {"by_window": per_window, "decomposition": decomposition}


# --------------------------------------------------------------------------
# the variogram half: representativeness from the gauges alone


def run_variogram(
    gauge: np.ndarray,
    meta: pd.DataFrame,
    transform: PrecipTransform,
    grid,
    max_distance_km: float,
    n_bins: int,
    season_months: np.ndarray | None = None,
    label: str = "all",
) -> dict:
    """Fit the variogram in mm/day AND in transformed units.

    Both are needed and they answer different questions.  The mm/day fit is what
    a reader understands and what the verification tables are in.  The
    transformed fit is what R actually consumes: ``observations.gauges.
    representativeness`` is added in variance to ``sigma_obs`` inside
    ``build_R``, which operates entirely in model space.  Converting a mm/day
    sigma into model space after the fact is wrong under a log transform because
    the mapping is nonlinear -- the variogram has to be computed on the
    transformed values from the start.
    """
    edges = np.linspace(0.0, max_distance_km, n_bins + 1)
    cell_km = grid.res * 111.32       # degrees to km at these latitudes

    out = {"label": label, "cell_km": cell_km, "n_stations": int(len(meta))}
    for space, values in (
        ("mm_per_day", gauge),
        ("transformed", transform.forward(np.nan_to_num(gauge, nan=np.nan))),
    ):
        empirical = empirical_variogram(values, meta["lat"], meta["lon"], edges)
        try:
            fitted = fit_variogram(
                empirical["distance_km"],
                empirical["gamma"],
                empirical["n_pairs"],
                min_separation_km=empirical["min_separation_km"],
                units=space,
            )
        except ValueError as error:
            print(f"[variogram] {label}/{space}: {error}", flush=True)
            continue
        sigma_cell = representativeness_sigma(fitted, cell_km)
        out[space] = {
            "fit": fitted.to_dict(),
            "empirical": {
                "distance_km": empirical["distance_km"].tolist(),
                "gamma": empirical["gamma"].tolist(),
                "n_pairs": empirical["n_pairs"].tolist(),
            },
            "representativeness_sigma_cell": sigma_cell,
            "representativeness_sigma_by_footprint": {
                f"{f * grid.res:.2f}deg": representativeness_sigma(fitted, f * cell_km)
                for f in (1, 2, 6, 8, 10)
            },
        }
        print(
            f"[variogram] {label}/{space}: nugget {fitted.nugget:.4f}, "
            f"sill {fitted.sill:.4f}, range {fitted.range_km:.0f} km, "
            f"nugget fraction {fitted.nugget_fraction:.0%}, "
            f"sigma_rep({cell_km:.1f} km cell) = {sigma_cell:.4f}",
            flush=True,
        )
        if np.isfinite(fitted.min_separation_km):
            print(
                f"[variogram]   closest station pair {fitted.min_separation_km:.1f} km; "
                f"the nugget is an EXTRAPOLATION below that and dominates a "
                f"{cell_km:.1f} km cell",
                flush=True,
            )
            if fitted.range_km <= fitted.min_separation_km * 1.001:
                print(
                    "[variogram]   range sits at the resolution bound: the network "
                    "cannot distinguish this from a pure nugget, so all unresolved "
                    "variance is attributed to the nugget (the conservative choice)",
                    flush=True,
                )
    return out


# --------------------------------------------------------------------------
# plots


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 130,
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


PRODUCT_COLOURS = {
    "chirps": "#c1440e",
    "cpc": "#1f6f8b",
    "imerg": "#4a7c1f",
    "gauge": "#111111",
}


def plot_variogram(variograms: dict, out_path: Path) -> None:
    """Empirical points and fitted curve, with the nugget extrapolation shown."""
    plt = _style()
    spaces = [s for s in ("mm_per_day", "transformed") if s in variograms]
    figure, axes = plt.subplots(1, len(spaces), figsize=(5.2 * len(spaces), 4.0))
    axes = np.atleast_1d(axes)

    for axis, space in zip(axes, spaces):
        block = variograms[space]
        empirical = block["empirical"]
        fit = block["fit"]
        distance = np.array(empirical["distance_km"], float)
        gamma = np.array(empirical["gamma"], float)
        counts = np.array(empirical["n_pairs"], float)
        ok = np.isfinite(gamma)

        sizes = 12 + 40 * (counts[ok] / max(counts[ok].max(), 1)) if ok.any() else 20
        axis.scatter(distance[ok], gamma[ok], s=sizes, color="#333333",
                     zorder=3, label="empirical (size = pair count)")

        h = np.linspace(0, np.nanmax(distance[ok]) * 1.05 if ok.any() else 400, 300)
        curve = fit["nugget"] + fit["sill"] * (1 - np.exp(-h / fit["range_km"]))
        axis.plot(h, curve, color="#c1440e", lw=2,
                  label=f"exponential fit, range {fit['range_km']:.0f} km")

        axis.axhline(fit["nugget"], color="#1f6f8b", ls="--", lw=1.2,
                     label=f"nugget {fit['nugget']:.3f} ({fit['nugget_fraction']:.0%})")
        gap = fit.get("min_separation_km", np.nan)
        if np.isfinite(gap):
            axis.axvspan(0, gap, color="#999999", alpha=0.16, zorder=0)
            axis.text(gap * 0.5, axis.get_ylim()[1] * 0.06, "no data\n(nugget is\nextrapolated)",
                      ha="center", va="bottom", fontsize=7, color="#444444")

        axis.set_xlabel("separation (km)")
        axis.set_ylabel(f"semivariance ({space.replace('_', ' ')})")
        axis.set_title(f"{space.replace('_', ' ')}   "
                       f"$\\sigma_{{rep}}$={block['representativeness_sigma_cell']:.3f}")
        axis.legend(fontsize=7, loc="lower right")
        axis.set_xlim(left=0)
        axis.set_ylim(bottom=0)

    figure.suptitle(
        "Daily rainfall variogram from BMD gauge pairs "
        "— the nugget sets the point-vs-cell error", y=1.0
    )
    figure.tight_layout()
    figure.savefig(out_path, bbox_inches="tight")
    plt.close(figure)


def plot_station_timeseries(
    dates, gauge, products: dict, meta: pd.DataFrame, out_path: Path, n_show: int = 6,
    window: int = 5,
) -> None:
    """Gauge against every product at the wettest stations, daily and smoothed.

    Daily panels show why a single-day gridded-vs-point comparison is hard to
    read; the running mean beside it shows the same data with the random part
    averaged down, which is the visual form of the decomposition.
    """
    plt = _style()
    wetness = np.nanmean(gauge, axis=0)
    order = np.argsort(np.where(np.isfinite(wetness), wetness, -np.inf))[::-1]
    chosen = order[:n_show]

    figure, axes = plt.subplots(len(chosen), 2, figsize=(13, 1.9 * len(chosen)),
                                sharex=True)
    axes = np.atleast_2d(axes)

    def smooth(series):
        return pd.Series(series).rolling(window, min_periods=max(1, window // 2)).mean()

    for row, station in enumerate(chosen):
        for column, transform_fn, title in (
            (0, lambda s: s, "daily"),
            (1, smooth, f"{window}-day running mean"),
        ):
            axis = axes[row, column]
            axis.plot(dates, transform_fn(gauge[:, station]), color=PRODUCT_COLOURS["gauge"],
                      lw=1.1, label="gauge (truth)", zorder=5)
            for name, block in products.items():
                axis.plot(dates, transform_fn(block["at_stations"][:, station]),
                          color=PRODUCT_COLOURS.get(name, "#888888"), lw=0.9,
                          alpha=0.85, label=name.upper())
            if row == 0:
                axis.set_title(title)
            if column == 0:
                axis.set_ylabel(f"{meta['name'].iloc[station]}\nmm/day", fontsize=7)
            if row == 0 and column == 1:
                axis.legend(fontsize=6.5, ncol=4, loc="upper right")

    figure.suptitle("Products against the gauge at the six wettest stations", y=1.0)
    figure.tight_layout()
    figure.savefig(out_path, bbox_inches="tight")
    plt.close(figure)


def plot_error_vs_aggregation(budget: dict, out_path: Path) -> None:
    """RMSE against averaging window, with the systematic floor drawn in.

    The gap between each curve and its own dashed floor is the part of the error
    that averaging removes; the floor itself is what a product would still be
    wrong by given infinite averaging, and is the honest ceiling on how well any
    gridded product can ever match a point gauge.
    """
    plt = _style()
    figure, axis = plt.subplots(figsize=(6.6, 4.4))

    for name, block in budget.items():
        by_window = block["by_window"]
        windows = sorted(int(w) for w in by_window)
        rmse = [by_window[str(w)].get("rmse_mm", np.nan) for w in windows]
        colour = PRODUCT_COLOURS.get(name, "#888888")
        axis.plot(windows, rmse, "o-", color=colour, lw=1.8, label=name.upper())

        decomposition = block.get("decomposition")
        if decomposition and decomposition.get("model_is_valid"):
            floor = decomposition["systematic_rmse"]
            axis.axhline(floor, color=colour, ls="--", lw=1.0, alpha=0.7)
            axis.text(windows[-1], floor, f"  floor {floor:.2f}", color=colour,
                      va="bottom", fontsize=7)

    axis.set_xscale("log")
    axis.set_xticks(list(AGGREGATION_WINDOWS))
    axis.set_xticklabels([f"{w}d" for w in AGGREGATION_WINDOWS])
    axis.set_xlabel("averaging window")
    axis.set_ylabel("RMSE against gauge (mm/day)")
    axis.set_title("Error against gauges falls with averaging — down to a floor\n"
                   "dashed = systematic part that averaging cannot remove",
                   fontsize=9)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(out_path, bbox_inches="tight")
    plt.close(figure)


def plot_station_bias_map(
    products: dict, gauge: np.ndarray, meta: pd.DataFrame, out_path: Path
) -> None:
    """Per-station mean bias, so spatial structure in the error is visible.

    A product that is merely noisy shows a salt-and-pepper map; one with a
    systematic problem shows coherent patches, which is what the aggregation
    floor is measuring numerically.
    """
    plt = _style()
    names = list(products)
    figure, axes = plt.subplots(1, len(names), figsize=(4.3 * len(names), 4.4),
                                sharex=True, sharey=True)
    axes = np.atleast_1d(axes)

    biases = {}
    for name, block in products.items():
        difference = block["at_stations"] - gauge
        with np.errstate(invalid="ignore"):
            biases[name] = np.nanmean(np.where(np.isfinite(difference), difference, np.nan),
                                      axis=0)
    finite = np.concatenate([b[np.isfinite(b)] for b in biases.values()]) if biases else np.array([0.0])
    limit = float(np.nanpercentile(np.abs(finite), 95)) if finite.size else 1.0
    limit = max(limit, 0.5)

    for axis, name in zip(axes, names):
        scatter = axis.scatter(
            meta["lon"], meta["lat"], c=biases[name], cmap="RdBu_r",
            vmin=-limit, vmax=limit, s=90, edgecolor="#222222", linewidth=0.5,
        )
        axis.set_title(f"{name.upper()}  mean bias "
                       f"{np.nanmean(biases[name]):+.2f} mm/day", fontsize=9)
        axis.set_xlabel("longitude")
        axis.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("latitude")
    figure.colorbar(scatter, ax=list(axes), shrink=0.82,
                    label="product − gauge (mm/day)")
    figure.suptitle("Per-station bias: coherent patches mean systematic error, "
                    "speckle means noise", y=1.02, fontsize=10)
    figure.savefig(out_path, bbox_inches="tight")
    plt.close(figure)


def plot_daily_vs_monthly_scatter(budget_inputs: dict, gauge, out_path: Path) -> None:
    """Same products, same stations, daily against monthly means."""
    plt = _style()
    names = list(budget_inputs)
    figure, axes = plt.subplots(2, len(names), figsize=(3.7 * len(names), 7.2))
    axes = np.atleast_2d(axes)

    for column, name in enumerate(names):
        product = budget_inputs[name]
        for row, window in ((0, 1), (1, 30)):
            axis = axes[row, column]
            mask = np.isfinite(product) & np.isfinite(gauge)
            p = aggregate_time(np.where(mask, product, np.nan), window).ravel()
            g = aggregate_time(np.where(mask, gauge, np.nan), window).ravel()
            ok = np.isfinite(p) & np.isfinite(g)
            axis.scatter(g[ok], p[ok], s=6, alpha=0.25,
                         color=PRODUCT_COLOURS.get(name, "#888888"), edgecolor="none")
            top = float(np.nanpercentile(np.concatenate([g[ok], p[ok]]), 99.5)) if ok.any() else 1
            axis.plot([0, top], [0, top], color="#111111", lw=1, ls="--")
            correlation = float(np.corrcoef(g[ok], p[ok])[0, 1]) if ok.sum() > 2 else np.nan
            axis.set_xlim(0, top)
            axis.set_ylim(0, top)
            axis.set_title(f"{name.upper()} {'daily' if window == 1 else 'monthly'}  "
                           f"r={correlation:.2f}", fontsize=8.5)
            axis.set_xlabel("gauge (mm/day)")
            if column == 0:
                axis.set_ylabel("product (mm/day)")
            axis.set_aspect("equal", adjustable="box")

    figure.suptitle("Scatter tightens onto the 1:1 line with averaging; "
                    "what remains is systematic", y=1.0, fontsize=10)
    figure.tight_layout()
    figure.savefig(out_path, bbox_inches="tight")
    plt.close(figure)


def plot_wetday_and_intensity(budget: dict, out_path: Path) -> None:
    """Wet-day frequency and bias stratified by observed intensity."""
    plt = _style()
    figure, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    names = list(budget)

    daily = {n: budget[n]["by_window"]["1"] for n in names}
    gauge_wet = next(iter(daily.values())).get("wet_fraction_gauge", np.nan)
    positions = np.arange(len(names))
    left.bar(positions, [daily[n].get("wet_fraction_product", np.nan) for n in names],
             color=[PRODUCT_COLOURS.get(n, "#888888") for n in names], width=0.6)
    left.axhline(gauge_wet, color="#111111", ls="--", lw=1.4,
                 label=f"gauge {gauge_wet:.3f}")
    left.set_xticks(positions)
    left.set_xticklabels([n.upper() for n in names])
    left.set_ylabel(f"fraction of days >= {WET_THRESHOLD_MM:g} mm")
    left.set_title("Wet-day frequency at station points")
    left.legend(fontsize=8)

    width = 0.8 / max(len(names), 1)
    for offset, name in enumerate(names):
        values = [daily[name].get(f"bias_above_{t:g}_mm", np.nan)
                  for t in CATEGORICAL_THRESHOLDS]
        right.bar(np.arange(len(CATEGORICAL_THRESHOLDS)) + offset * width, values,
                  width=width, label=name.upper(),
                  color=PRODUCT_COLOURS.get(name, "#888888"))
    right.axhline(0, color="#111111", lw=1)
    right.set_xticks(np.arange(len(CATEGORICAL_THRESHOLDS)) + 0.4 - width / 2)
    right.set_xticklabels([f">={t:g}" for t in CATEGORICAL_THRESHOLDS])
    right.set_xlabel("gauge intensity (mm/day)")
    right.set_ylabel("product − gauge (mm/day)")
    right.set_title("Bias grows with intensity: the heavy tail is where\n"
                    "gridded references stop being usable", fontsize=9)
    right.legend(fontsize=8)

    figure.tight_layout()
    figure.savefig(out_path, bbox_inches="tight")
    plt.close(figure)


# --------------------------------------------------------------------------


def print_budget_table(budget: dict) -> None:
    print()
    print("[budget] gauge is truth. Error against gauges by averaging window:")
    print(f"    {'product':10s} {'window':>7s} {'n':>7s} {'bias':>8s} {'MAE':>8s} "
          f"{'RMSE':>8s} {'corr':>6s}")
    for name, block in budget.items():
        for window in AGGREGATION_WINDOWS:
            scores = block["by_window"].get(str(window), {})
            if not scores.get("n"):
                continue
            print(f"    {name:10s} {str(window) + 'd':>7s} {scores['n']:>7d} "
                  f"{scores.get('bias_mm', float('nan')):>+8.2f} "
                  f"{scores.get('mae_mm', float('nan')):>8.2f} "
                  f"{scores.get('rmse_mm', float('nan')):>8.2f} "
                  f"{scores.get('correlation', float('nan')):>6.2f}")

    print()
    print("[budget] systematic / random split from MSE(N) = systematic + random/N:")
    print(f"    {'product':10s} {'floor RMSE':>11s} {'random(1d)':>11s} "
          f"{'R^2':>6s}  interpretation")
    for name, block in budget.items():
        decomposition = block.get("decomposition")
        if not decomposition:
            continue
        if not decomposition.get("model_is_valid"):
            reason = ("error GROWS with averaging"
                      if decomposition.get("error_grows_with_averaging")
                      else "negative intercept")
            print(f"    {name:10s} {'--':>11s} {'--':>11s} {'--':>6s}  "
                  f"UNUSABLE: {reason}")
            continue
        floor = decomposition["systematic_rmse"]
        random = decomposition["random_rmse_daily"]
        share = floor**2 / (floor**2 + random**2) if floor or random else float("nan")
        print(f"    {name:10s} {floor:>11.2f} {random:>11.2f} "
              f"{decomposition['r_squared']:>6.2f}  "
              f"{share:.0%} of daily MSE is irreducible")
    print()
    print("    floor RMSE = what averaging can never remove: real product bias plus")
    print("    the systematic part of point-vs-cell mismatch. It is the ceiling on")
    print("    how well any gridded product can match a point gauge, and therefore")
    print("    a lower bound on 'model error' inferred from a gridded reference.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gauge-as-truth error budget for gridded rainfall products",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--stations", required=True, help="BMD daily CSV from script 05")
    parser.add_argument("--zarr", required=True, help="training store, for CHIRPS and CPC")
    parser.add_argument("--stats", required=True, help="stats JSON, for the transform")
    parser.add_argument(
        "--imerg", nargs="*", default=None,
        help="prepared IMERG NetCDF(s); accepts several files or a shell glob, "
             "since IMERG is prepared per evaluation period here",
    )
    parser.add_argument("--imerg-factor", type=int, default=2)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--grid", default="bd")
    parser.add_argument("--min-coverage", type=float, default=0.5)
    parser.add_argument("--chirps-day-offset", type=int, default=-1,
                        help="CHIRPS is 00-00 UTC; BMD day D ends 03 UTC on D")
    parser.add_argument("--cpc-day-offset", type=int, default=-1)
    parser.add_argument("--max-distance-km", type=float, default=400.0)
    parser.add_argument("--n-bins", type=int, default=24)
    parser.add_argument(
        "--common-sample", action="store_true",
        help="score every product only on days covered by ALL of them, so the "
             "rows of the budget table are comparable (the variogram still uses "
             "the full gauge record, since it involves no product)",
    )
    parser.add_argument("--monsoon-only", action="store_true",
                        help="restrict the variogram to Jun-Sep, whose convective "
                             "structure differs sharply from the pre-monsoon")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = get_grid(args.grid)

    stats = json.loads(Path(args.stats).read_text())
    transform = PrecipTransform(**stats["precip_transform"])
    print(f"[setup] transform {transform.kind} eps={transform.eps} "
          f"mu={transform.mu:.4f} sd={transform.sd:.4f}", flush=True)

    dates, meta, gauge = load_gauges(
        args.stations, args.start, args.end, args.min_coverage
    )

    products = load_gridded_products(
        args.zarr, dates, grid, meta, args.chirps_day_offset, args.cpc_day_offset
    )
    if args.imerg:
        imerg = load_imerg_product(args.imerg, dates, grid, meta, args.imerg_factor)
        if imerg is not None:
            products["imerg"] = imerg
    if not products:
        raise SystemExit("no products loaded; nothing to compare")

    # ---------------------------------------------------------------- variogram
    months = dates.month.to_numpy()
    variogram_gauge = gauge
    label = "all-seasons"
    if args.monsoon_only:
        monsoon = (months >= 6) & (months <= 9)
        variogram_gauge = gauge[monsoon]
        label = "monsoon-JJAS"
        print(f"[variogram] restricted to {int(monsoon.sum())} monsoon days", flush=True)

    variograms = run_variogram(
        variogram_gauge, meta, transform, grid,
        args.max_distance_km, args.n_bins, label=label,
    )

    # ---------------------------------------------------------------- budget
    # IMERG is prepared per evaluation period while CHIRPS and CPC run the whole
    # archive, so without care the table would compare products scored on
    # different years -- and monsoon years differ enough that the ranking could
    # come entirely from the calendar. Report the mismatch always; --common-sample
    # removes it by scoring every product on the days they all cover.
    coverage = {
        name: np.isfinite(block["at_stations"]).any(axis=1)
        for name, block in products.items()
    }
    day_counts = {name: int(mask.sum()) for name, mask in coverage.items()}
    print()
    print("[budget] days available per product: "
          + ", ".join(f"{n}={c}" for n, c in day_counts.items()))
    shared = np.logical_and.reduce(list(coverage.values())) if coverage else None
    if shared is not None and min(day_counts.values()) != max(day_counts.values()):
        print(f"[budget] only {int(shared.sum())} day(s) are covered by ALL products.")
        if not args.common_sample:
            print("[budget] WARNING: scoring each product on its own days. Products "
                  "are NOT directly comparable across rows; pass --common-sample "
                  "to restrict every product to the shared days.")
    if args.common_sample and shared is not None:
        print(f"[budget] --common-sample: restricting all products to "
              f"{int(shared.sum())} shared day(s)", flush=True)
        gauge = np.where(shared[:, None], gauge, np.nan)
        for block in products.values():
            block["at_stations"] = np.where(
                shared[:, None], block["at_stations"], np.nan
            )

    budget = {
        name: scores_by_window(block["at_stations"], gauge)
        for name, block in products.items()
    }
    print_budget_table(budget)

    # ---------------------------------------------------------------- plots
    plot_variogram(variograms, out_dir / "variogram.png")
    plot_station_timeseries(dates, gauge, products, meta,
                            out_dir / "station_timeseries.png")
    plot_error_vs_aggregation(budget, out_dir / "error_vs_aggregation.png")
    plot_station_bias_map(products, gauge, meta, out_dir / "station_bias_map.png")
    plot_daily_vs_monthly_scatter(
        {n: b["at_stations"] for n, b in products.items()}, gauge,
        out_dir / "daily_vs_monthly_scatter.png",
    )
    plot_wetday_and_intensity(budget, out_dir / "wetday_and_intensity.png")

    # ---------------------------------------------------------------- config
    recommendation = {}
    if "transformed" in variograms:
        block = variograms["transformed"]
        sigma_cell = block["representativeness_sigma_cell"]
        recommendation = {
            "observations.gauges.representativeness": round(sigma_cell, 3),
            "calibration.obs_sd_for_verification": round(sigma_cell, 3),
            "units": "transformed (model space), matching build_R",
            "current_config_values": {
                "observations.gauges.representativeness": 0.25,
                "calibration.obs_sd_for_verification": 0.10,
            },
            "basis": block["fit"],
            "by_footprint_size": block["representativeness_sigma_by_footprint"],
            "caveat": (
                "The nugget is extrapolated below the closest station pair "
                f"({block['fit'].get('min_separation_km', float('nan')):.0f} km) and "
                "dominates a 5 km cell. Treat as a founded estimate with real "
                "uncertainty, not a measured constant."
            ),
        }
        print()
        print("[config] recommended, in transformed units:")
        print(f"    observations.gauges.representativeness: {sigma_cell:.3f}"
              f"   (config currently 0.25)")
        print(f"    calibration.obs_sd_for_verification:    {sigma_cell:.3f}"
              f"   (config currently 0.10)")
        print("    larger footprints (a satellite obs is a box, not a point):")
        for key, value in block["representativeness_sigma_by_footprint"].items():
            print(f"        {key:>8s}: {value:.3f}")

    payload = {
        "period": {"start": args.start, "end": args.end, "n_days": len(dates)},
        "stations": {
            "n": int(len(meta)),
            "ids": meta["station_id"].astype(str).tolist(),
            "names": meta["name"].astype(str).tolist(),
            "lat": meta["lat"].tolist(),
            "lon": meta["lon"].tolist(),
            "coverage": meta["coverage"].tolist(),
        },
        "day_offsets": {n: b["day_offset"] for n, b in products.items()},
        "days_per_product": day_counts,
        "common_sample": {
            "applied": bool(args.common_sample),
            "n_shared_days": int(shared.sum()) if shared is not None else 0,
        },
        "variogram": variograms,
        "budget": budget,
        "recommendation": recommendation,
        "note": (
            "Gauge is truth throughout. Products are sampled bilinearly at station "
            "points. The systematic floor from the aggregation fit bounds how well "
            "any gridded product can match a point gauge and is therefore a lower "
            "bound on model error inferred against a gridded reference."
        ),
    }
    out_json = out_dir / "error_budget.json"
    out_json.write_text(json.dumps(payload, indent=2, default=float))
    print()
    print(f"[done] wrote {out_json}")
    for figure in sorted(out_dir.glob("*.png")):
        print(f"[done] wrote {figure}")


if __name__ == "__main__":
    main()
