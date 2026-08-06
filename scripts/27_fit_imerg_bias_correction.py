#!/usr/bin/env python
"""Fit an IMERG -> CHIRPS quantile map from PREPARED BMD-window IMERG files.

Why this exists alongside ``scripts/07_bias_correct_imerg.py``
--------------------------------------------------------------
Script 07 reads an ``imerg`` channel out of the packed Zarr.  ``bd_wide_cpc.zarr``
has no such channel -- its conditioning stack is CPC + ERA5 -- and the IMERG
archive on disk covers 2021-2024 only, not the 1981-2018 training span.  So the
"fit on training years" recipe in METHODOLOGY.md 4.4 is not executable as
written.

This script does the executable version of the same thing:

* the source is the prepared ``imerg_aligned_*.nc`` produced by
  ``scripts/08_prepare_imerg_observations.py``, i.e. exact 03:00-03:00 UTC BMD
  windows on the 0.1 degree lattice that nests in the model grid;
* the target is CHIRPS from the packed Zarr, block-averaged 2x2 to the same
  0.1 degree lattice -- which is exactly the field the block-average observation
  operator compares IMERG against, so the map is fitted on the quantity the
  likelihood actually uses;
* the fit is LEAVE-ONE-YEAR-OUT.  The map applied to year Y is fitted on all
  other available years.  Year Y never sees its own statistics.

Leave-one-year-out is weaker than a 1981-2018 fit -- three monsoon seasons is a
small sample for a per-cell CDF -- so quantiles are pooled over a spatial
neighbourhood before fitting.  Report the pooling radius; it is a real
limitation, not a detail.

Method
------
Per season and per pooled neighbourhood:

1. **Frequency adaptation first.**  IMERG over South Asia over-detects light
   rain.  Zero out the smallest IMERG values until its wet-day frequency matches
   CHIRPS.  Doing this after the quantile map propagates the drizzle bias
   straight through, which is the single most common way this correction is got
   wrong.
2. **Empirical quantile map** from the IMERG CDF to the CHIRPS CDF on the wet
   part of the distribution, evaluated at ``--n-quantiles`` points and applied
   by linear interpolation with constant extrapolation above the top knot.

``--fit-error-model`` additionally writes the residual sd of corrected IMERG
against CHIRPS in TRANSFORMED space, binned by corrected intensity.  Those
numbers are what ``observations.imerg.sigma_obs`` should be set from, instead of
the current guessed 0.35.

Usage
-----
    python scripts/27_fit_imerg_bias_correction.py \
        --imerg data/processed/bmd_imerg_eval_2021_may_sep/imerg_aligned_20210501_20210930.nc \
                data/processed/bmd_imerg_eval_2022_may_sep/imerg_aligned_20220501_20220930.nc \
                data/processed/bmd_imerg_eval_2023_may_sep/imerg_aligned_20230501_20230930.nc \
                data/processed/bmd_imerg_eval_2024_may_jun/imerg_aligned_20240501_20240630.nc \
        --stats data/processed/stats_cpc.json \
        --zarr data/processed/bd_wide_cpc.zarr \
        --grid bd --pool 5 --fit-error-model \
        --out data/processed/imerg_qm_loyo.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bdhires.grids import WIDE, crop_offsets, get_grid  # noqa: E402
from bdhires.transforms import PrecipTransform  # noqa: E402

SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJAS": (6, 7, 8, 9), "ON": (10, 11)}
SEASON_ORDER = ["DJF", "MAM", "JJAS", "ON"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--imerg",
        nargs="+",
        required=True,
        help="prepared BMD-window IMERG NetCDF files from script 08, one or more years",
    )
    parser.add_argument("--zarr", required=True, help="packed Zarr holding the CHIRPS target")
    parser.add_argument("--stats", required=True, help="stats JSON holding precip_transform")
    parser.add_argument("--grid", default="bd")
    parser.add_argument("--factor", type=int, default=2, help="0.05 -> 0.1 degree block factor")
    parser.add_argument(
        "--pool",
        type=int,
        default=5,
        help="pool quantiles over a POOL x POOL neighbourhood of 0.1 degree cells "
        "(5 ~ 50 km). 1 disables pooling and will be noisy with three seasons.",
    )
    parser.add_argument("--n-quantiles", type=int, default=41)
    parser.add_argument("--wet-threshold", type=float, default=0.1, help="mm/day")
    parser.add_argument(
        "--min-samples",
        type=int,
        default=300,
        help="below this many pooled samples the map for that cell/season is identity",
    )
    parser.add_argument("--fit-error-model", action="store_true")
    parser.add_argument(
        "--error-bins",
        nargs="+",
        type=float,
        default=[0.0, 1.0, 5.0, 10.0, 25.0, 50.0, 1e9],
        help="corrected-intensity bin edges (mm/day) for the empirical error model",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", default=None, help="optional JSON diagnostic dump")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Core, kept numpy-only and side-effect free so it can be unit tested directly.
# ---------------------------------------------------------------------------


def wet_frequency(values: np.ndarray, wet_threshold: float) -> float:
    """Fraction of finite samples at or above ``wet_threshold``."""
    finite = values[np.isfinite(values)]
    if not finite.size:
        return float("nan")
    return float(np.mean(finite >= wet_threshold))


def frequency_adaptation_cut(source: np.ndarray, target_wet_fraction: float) -> float:
    """Threshold below which IMERG must be set to zero to match a wet fraction.

    Returns the cut in source units.  Everything strictly below it is drizzle
    that CHIRPS does not report and that must not survive into the quantile map.
    """
    finite = np.sort(source[np.isfinite(source)])
    if not finite.size or not np.isfinite(target_wet_fraction):
        return 0.0
    n_wet = int(round(target_wet_fraction * finite.size))
    n_wet = int(np.clip(n_wet, 0, finite.size))
    if n_wet <= 0:
        return float(finite[-1]) + 1.0  # dry everything
    if n_wet >= finite.size:
        return 0.0
    return float(finite[finite.size - n_wet])


def fit_quantile_map(
    source: np.ndarray,
    target: np.ndarray,
    quantiles: np.ndarray,
    wet_threshold: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit frequency adaptation plus a wet-part quantile map.

    Returns ``(source_knots, target_knots, cut)``.  ``cut`` is the frequency
    adaptation threshold; values below it map to zero.  Both knot arrays are
    monotone non-decreasing so ``apply_quantile_map`` can interpolate directly.
    """
    target_wet = wet_frequency(target, wet_threshold)
    cut = frequency_adaptation_cut(source, target_wet)

    source_wet = source[np.isfinite(source) & (source >= cut) & (source > 0.0)]
    target_wet_values = target[np.isfinite(target) & (target >= wet_threshold)]
    if source_wet.size < 2 or target_wet_values.size < 2:
        # Not enough wet samples to say anything; identity above the cut.
        knots = np.array([0.0, 1.0], dtype=np.float32)
        return knots, knots.copy(), cut

    source_knots = np.quantile(source_wet, quantiles).astype(np.float32)
    target_knots = np.quantile(target_wet_values, quantiles).astype(np.float32)
    # np.quantile is monotone in theory; enforce it in floating point so that
    # np.interp cannot produce a non-monotone correction.
    source_knots = np.maximum.accumulate(source_knots)
    target_knots = np.maximum.accumulate(target_knots)
    return source_knots, target_knots, cut


def apply_quantile_map(
    values: np.ndarray,
    source_knots: np.ndarray,
    target_knots: np.ndarray,
    cut: float,
) -> np.ndarray:
    """Apply a fitted map.  Below ``cut`` the output is exactly zero.

    Above the top knot the correction is extrapolated as a constant RATIO rather
    than a constant offset, so the heaviest events -- which sit above anything in
    a three-season fit -- are scaled rather than clipped.  Clipping the tail of a
    precipitation correction is how you manufacture a dry bias in extremes.
    """
    out = np.zeros_like(values, dtype=np.float32)
    finite = np.isfinite(values)
    wet = finite & (values >= cut) & (values > 0.0)
    if not wet.any():
        out[~finite] = np.nan
        return out

    mapped = np.interp(values[wet], source_knots, target_knots).astype(np.float32)

    top_source = float(source_knots[-1])
    top_target = float(target_knots[-1])
    if top_source > 0.0:
        ratio = top_target / top_source
        above = values[wet] > top_source
        if above.any():
            mapped[above] = (values[wet][above] * ratio).astype(np.float32)

    out[wet] = np.clip(mapped, 0.0, None)
    out[~finite] = np.nan
    return out


def pooled_slice(index: int, pool: int, size: int) -> slice:
    """Symmetric neighbourhood around ``index``, clipped to the array."""
    half = pool // 2
    return slice(max(0, index - half), min(size, index + half + 1))


# ---------------------------------------------------------------------------


def load_imerg_files(paths: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate prepared IMERG files. Returns (time, precip, lat, lon)."""
    times, fields, lat, lon = [], [], None, None
    for path in sorted(paths):
        with xr.open_dataset(path) as dataset:
            if "precipitation" not in dataset:
                raise ValueError(f"{path} has no precipitation variable")
            end_hour = dataset.attrs.get("bmd_accumulation_end_hour_utc")
            if int(end_hour) != 3:
                raise ValueError(
                    f"{path} accumulates to {end_hour}:00 UTC, not the BMD 03:00 window"
                )
            this_lat = np.asarray(dataset.lat.values, np.float64)
            this_lon = np.asarray(dataset.lon.values, np.float64)
            if lat is None:
                lat, lon = this_lat, this_lon
            elif not (np.allclose(lat, this_lat) and np.allclose(lon, this_lon)):
                raise ValueError(f"{path} is on a different lattice from the first file")
            times.append(np.asarray(dataset.time.values).astype("datetime64[D]"))
            fields.append(np.asarray(dataset.precipitation.values, np.float32))
    time = np.concatenate(times)
    precipitation = np.concatenate(fields, axis=0)
    order = np.argsort(time)
    time, precipitation = time[order], precipitation[order]
    unique, counts = np.unique(time, return_counts=True)
    if (counts > 1).any():
        raise ValueError(
            "duplicate dates across the supplied IMERG files: "
            + ", ".join(str(d) for d in unique[counts > 1][:5])
        )
    return time, precipitation, lat.astype(np.float32), lon.astype(np.float32)


def load_zarr_time(store) -> np.ndarray:
    """Decode the packed Zarr's time axis to ``datetime64[D]``.

    ``scripts/04_regrid_and_pack.py`` writes ``time`` as
    ``days.values.astype("datetime64[ns]").view("i8")`` -- i.e. the array on
    disk is a plain int64 VIEW of nanoseconds-since-epoch, not a day-count.
    ``bdhires.data.zarr_dataset.PrecipDataset`` reads it back with
    ``np.asarray(z["time"][:], dtype="datetime64[ns]")``, which reverses that
    view. Casting straight to ``datetime64[D]`` instead -- skipping the
    ``datetime64[ns]`` step -- reinterprets the raw nanosecond integers as
    day-counts and silently produces dates thousands of years off, which is
    why a lookup against real 2021-2024 dates finds nothing. Match the
    production loader exactly.
    """
    raw = np.asarray(store["time"][:])
    if np.issubdtype(raw.dtype, np.datetime64):
        return raw.astype("datetime64[D]")
    return raw.astype("datetime64[ns]").astype("datetime64[D]")


def load_chirps_on_imerg_lattice(
    zarr_path: str, grid_name: str, factor: int, wanted: np.ndarray
) -> np.ndarray:
    """CHIRPS from the Zarr target, cropped to ``grid`` and block-averaged to 0.1 deg."""
    store = zarr.open(zarr_path, mode="r")
    time = load_zarr_time(store)
    grid = get_grid(grid_name)
    row0, col0 = crop_offsets(WIDE, grid)
    rows = slice(row0, row0 + grid.nlat)
    cols = slice(col0, col0 + grid.nlon)

    lookup = {value: index for index, value in enumerate(time)}
    missing = [str(d) for d in wanted if d not in lookup]
    if missing:
        raise ValueError(
            f"{zarr_path} lacks CHIRPS for {len(missing)} requested date(s), "
            f"first few: {missing[:5]}"
        )
    indices = [lookup[d] for d in wanted]

    fine = np.stack(
        [np.asarray(store["target"][index][rows, cols], np.float32) for index in indices]
    )
    n_time, nlat, nlon = fine.shape
    coarse = fine.reshape(n_time, nlat // factor, factor, nlon // factor, factor)
    return np.nanmean(coarse, axis=(2, 4)).astype(np.float32)


def season_of(dates: np.ndarray) -> np.ndarray:
    months = dates.astype("datetime64[M]").astype(int) % 12 + 1
    labels = np.empty(len(dates), dtype=object)
    for name, month_set in SEASONS.items():
        labels[np.isin(months, month_set)] = name
    return labels


def main() -> None:
    args = parse_args()
    if args.pool < 1 or args.pool % 2 == 0:
        raise ValueError("--pool must be a positive odd number")

    stats = json.loads(Path(args.stats).read_text())
    transform = PrecipTransform.from_dict(stats["precip_transform"])

    time, imerg, lat, lon = load_imerg_files(args.imerg)
    chirps = load_chirps_on_imerg_lattice(args.zarr, args.grid, args.factor, time)
    if chirps.shape != imerg.shape:
        raise ValueError(f"shape mismatch: IMERG {imerg.shape} versus CHIRPS {chirps.shape}")

    years = time.astype("datetime64[Y]").astype(int) + 1970
    seasons = season_of(time)
    available_years = sorted(set(int(y) for y in years))
    if len(available_years) < 2:
        raise ValueError(
            "leave-one-year-out needs at least two years of prepared IMERG; "
            f"found {available_years}"
        )
    quantiles = np.linspace(0.0, 1.0, args.n_quantiles)
    nlat, nlon = imerg.shape[1:]

    print(
        f"[fit] {len(time)} days, {available_years[0]}-{available_years[-1]}, "
        f"lattice {nlat}x{nlon}, pooling {args.pool}x{args.pool}, "
        f"{args.n_quantiles} quantiles",
        flush=True,
    )

    payload: dict[str, np.ndarray] = {
        "quantiles": quantiles.astype(np.float32),
        "lat": lat,
        "lon": lon,
        "wet_threshold": np.float32(args.wet_threshold),
        "pool": np.int32(args.pool),
        "factor": np.int32(args.factor),
        "holdout_years": np.asarray(available_years, dtype=np.int32),
        "season_order": np.asarray(SEASON_ORDER, dtype="U4"),
    }
    report: dict = {
        "source_files": list(args.imerg),
        "zarr": args.zarr,
        "years": available_years,
        "pool": args.pool,
        "n_quantiles": args.n_quantiles,
        "wet_threshold": args.wet_threshold,
        "holdouts": {},
    }

    for holdout in available_years:
        fit_mask = years != holdout
        n_fit_years = len(set(int(y) for y in years[fit_mask]))
        source_knots = np.zeros(
            (len(SEASON_ORDER), args.n_quantiles, nlat, nlon), np.float32
        )
        target_knots = np.zeros_like(source_knots)
        cuts = np.zeros((len(SEASON_ORDER), nlat, nlon), np.float32)
        fitted = np.zeros((len(SEASON_ORDER), nlat, nlon), bool)

        for season_index, season in enumerate(SEASON_ORDER):
            select = fit_mask & (seasons == season)
            if not select.any():
                continue
            imerg_season = imerg[select]
            chirps_season = chirps[select]
            for i in range(nlat):
                rows = pooled_slice(i, args.pool, nlat)
                for j in range(nlon):
                    cols = pooled_slice(j, args.pool, nlon)
                    source = imerg_season[:, rows, cols].reshape(-1)
                    target = chirps_season[:, rows, cols].reshape(-1)
                    both = np.isfinite(source) & np.isfinite(target)
                    if int(both.sum()) < args.min_samples:
                        # Identity map: better an uncorrected observation than one
                        # corrected by a CDF estimated from a handful of points.
                        source_knots[season_index, :, i, j] = np.linspace(
                            0.0, 1.0, args.n_quantiles
                        )
                        target_knots[season_index, :, i, j] = np.linspace(
                            0.0, 1.0, args.n_quantiles
                        )
                        continue
                    sk, tk, cut = fit_quantile_map(
                        source[both], target[both], quantiles, args.wet_threshold
                    )
                    if len(sk) != args.n_quantiles:
                        sk = np.interp(
                            quantiles, np.linspace(0, 1, len(sk)), sk
                        ).astype(np.float32)
                        tk = np.interp(
                            quantiles, np.linspace(0, 1, len(tk)), tk
                        ).astype(np.float32)
                    source_knots[season_index, :, i, j] = sk
                    target_knots[season_index, :, i, j] = tk
                    cuts[season_index, i, j] = cut
                    fitted[season_index, i, j] = True
            print(
                f"[fit] holdout {holdout} season {season}: "
                f"{int(select.sum())} days from {n_fit_years} other year(s), "
                f"{int(fitted[season_index].sum())}/{nlat * nlon} cells fitted",
                flush=True,
            )

        prefix = f"y{holdout}"
        payload[f"{prefix}_source_knots"] = source_knots
        payload[f"{prefix}_target_knots"] = target_knots
        payload[f"{prefix}_cut"] = cuts
        payload[f"{prefix}_fitted"] = fitted

        # Verification on the held-out year itself, which is the only honest
        # measure of what the map will do at assimilation time.
        holdout_mask = ~fit_mask
        diagnostics = {"n_fit_years": n_fit_years, "n_holdout_days": int(holdout_mask.sum())}
        if holdout_mask.any():
            corrected = apply_map_to_series(
                imerg[holdout_mask],
                seasons[holdout_mask],
                source_knots,
                target_knots,
                cuts,
            )
            reference = chirps[holdout_mask]
            raw = imerg[holdout_mask]
            both = np.isfinite(reference) & np.isfinite(raw)
            diagnostics.update(
                raw_bias_mm=float(np.mean(raw[both] - reference[both])),
                corrected_bias_mm=float(np.mean(corrected[both] - reference[both])),
                raw_wet_fraction=wet_frequency(raw[both], args.wet_threshold),
                corrected_wet_fraction=wet_frequency(corrected[both], args.wet_threshold),
                chirps_wet_fraction=wet_frequency(reference[both], args.wet_threshold),
                raw_rmse_mm=float(np.sqrt(np.mean((raw[both] - reference[both]) ** 2))),
                corrected_rmse_mm=float(
                    np.sqrt(np.mean((corrected[both] - reference[both]) ** 2))
                ),
            )
            print(
                f"[fit] holdout {holdout}: bias {diagnostics['raw_bias_mm']:+.3f} -> "
                f"{diagnostics['corrected_bias_mm']:+.3f} mm/day, wet fraction "
                f"{diagnostics['raw_wet_fraction']:.3f} -> "
                f"{diagnostics['corrected_wet_fraction']:.3f} "
                f"(CHIRPS {diagnostics['chirps_wet_fraction']:.3f})",
                flush=True,
            )

            if args.fit_error_model:
                residual = transform.forward(corrected[both]) - transform.forward(
                    reference[both]
                )
                intensity = corrected[both]
                edges = np.asarray(args.error_bins, dtype=float)
                sigma, centres, counts = [], [], []
                for lower, upper in zip(edges[:-1], edges[1:]):
                    inside = (intensity >= lower) & (intensity < upper)
                    counts.append(int(inside.sum()))
                    if inside.sum() < 50:
                        sigma.append(float("nan"))
                    else:
                        sigma.append(float(np.std(residual[inside])))
                    centres.append(float(lower))
                diagnostics["error_model"] = {
                    "bin_lower_mm": centres,
                    "sigma_transformed": sigma,
                    "n": counts,
                    "note": (
                        "set observations.imerg.sigma_obs from these, not from a guess; "
                        "the current config value is 0.35"
                    ),
                }
                payload[f"{prefix}_error_sigma"] = np.asarray(sigma, np.float32)
                payload[f"{prefix}_error_bins"] = edges.astype(np.float32)
                readable = ", ".join(
                    f"[{lower:g},{upper:g}) {s:.3f}"
                    for lower, upper, s in zip(edges[:-1], edges[1:], sigma)
                )
                print(f"[fit] holdout {holdout} sigma by intensity: {readable}", flush=True)

        report["holdouts"][str(holdout)] = diagnostics

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)
    print(f"[fit] wrote {out}", flush=True)

    report_path = Path(args.report) if args.report else out.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[fit] wrote {report_path}", flush=True)


def apply_map_to_series(
    values: np.ndarray,
    seasons: np.ndarray,
    source_knots: np.ndarray,
    target_knots: np.ndarray,
    cuts: np.ndarray,
) -> np.ndarray:
    """Apply a fitted leave-one-year-out map to a (T, nlat, nlon) IMERG series."""
    out = np.full_like(values, np.nan, dtype=np.float32)
    for season_index, season in enumerate(SEASON_ORDER):
        select = np.where(seasons == season)[0]
        if not len(select):
            continue
        nlat, nlon = values.shape[1:]
        for i in range(nlat):
            for j in range(nlon):
                out[select, i, j] = apply_quantile_map(
                    values[select, i, j],
                    source_knots[season_index, :, i, j],
                    target_knots[season_index, :, i, j],
                    float(cuts[season_index, i, j]),
                )
    return out


def load_and_apply(
    npz_path: str | Path,
    holdout_year: int,
    values: np.ndarray,
    dates: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Public entry point used by the sweep script.

    Applies the map whose fit EXCLUDED ``holdout_year`` to ``values``.  Raises if
    that year was not held out during fitting, which is the guard that stops a
    map from being applied to the data it was fitted on.
    """
    archive = np.load(npz_path, allow_pickle=False)
    available = [int(y) for y in archive["holdout_years"]]
    if holdout_year not in available:
        raise ValueError(
            f"{npz_path} has no leave-one-out map for {holdout_year}; "
            f"available holdouts are {available}. Applying a map fitted on the "
            "evaluation year would leak CHIRPS into the observation."
        )
    prefix = f"y{holdout_year}"
    corrected = apply_map_to_series(
        np.asarray(values, np.float32),
        season_of(np.asarray(dates).astype("datetime64[D]")),
        archive[f"{prefix}_source_knots"],
        archive[f"{prefix}_target_knots"],
        archive[f"{prefix}_cut"],
    )
    meta = {
        "path": str(npz_path),
        "holdout_year": holdout_year,
        "fitted_cells": int(archive[f"{prefix}_fitted"].sum()),
        "pool": int(archive["pool"]),
        "wet_threshold": float(archive["wet_threshold"]),
    }
    if f"{prefix}_error_sigma" in archive:
        meta["error_sigma_transformed"] = archive[f"{prefix}_error_sigma"].tolist()
        meta["error_bins_mm"] = archive[f"{prefix}_error_bins"].tolist()
    return corrected, meta


if __name__ == "__main__":
    main()
