#!/usr/bin/env python
"""Aggregate a dense gauge network into super-observations, and measure R.

Why this exists
---------------
The frozen ``v2_simul_s04_huber3`` contract assimilates every gauge as an
independent point observation through a bilinear operator, with one scalar
error budget (``observations.gauges.sigma_obs`` plus ``representativeness``)
for the whole network.  That contract was selected on the ~39-station BMD
network, whose median nearest-neighbour separation is near 60 km -- comfortably
more than one analysis grid cell (0.05 deg, ~5.5 km).

Adding BWDB takes the network to ~304 stations with a median nearest-neighbour
separation of 15 km, and 30 pairs closer than two grid cells.  Two things then
stop being true:

1.  Several gauges now share one analysis cell, or sit inside one another's
    guidance-spread kernel.  They are treated as independent measurements of
    the field, so their likelihood gradients add.  The pull at a point grows
    with the number of nearby gauges rather than with the information they
    carry, and where neighbouring gauges disagree the summed gradient is a
    difference of large opposing terms.
2.  ``representativeness`` -- the point-versus-cell mismatch -- is no longer an
    unmeasurable guess.  With multiple gauges inside a cell, the spread of
    those gauges about their own cell mean IS that mismatch, and this script
    measures it in exactly the transformed units the likelihood uses.

Super-obbing is the standard answer to (1): replace the co-located cluster with
one areal observation whose error is the mean's error, not a single gauge's.
It also makes (2) explicit rather than assumed.

What it does
------------
Held-out stations pass through untouched and are never merged, so the withheld
station scoring stays exactly as independent as the source experiment made it.
Only assimilated stations are aggregated, on a regular mesh of ``--cell-deg``.
A cell holding one assimilated station is emitted unchanged; a cell holding
several becomes one ``SOB_`` record at the member centroid carrying the daily
mean of whichever members reported.

The manifest reports the measured error budget both ways: for a single point
gauge (what the current runs assume) and for the emitted super-observations.
Neither is applied here -- this script only writes numbers and a station table.
Choosing to act on them is a ``--set observations.gauges.representativeness=``
on the assimilation job, which stays visible in that job's log.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"station_id", "lat", "lon", "date", "precip_mm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stations", required=True, help="combined long-form daily CSV")
    parser.add_argument(
        "--holdout-ids", default=None,
        help="one station id per line; these are never merged and never dropped",
    )
    parser.add_argument("--out", default=None, help="super-obbed daily CSV")
    parser.add_argument("--report", required=True, help="JSON manifest")
    parser.add_argument(
        "--cell-deg", type=float, default=0.25,
        help="super-observation mesh size in degrees (0.25 deg ~ 28 km)",
    )
    parser.add_argument(
        "--stats", default=None,
        help=(
            "data/processed/stats.json. Required to report the error budget in "
            "the transformed units the likelihood actually uses; without it only "
            "mm-space spreads are written."
        ),
    )
    parser.add_argument(
        "--min-pair-days", type=int, default=3,
        help="ignore cells with fewer co-reporting station-days when measuring R",
    )
    parser.add_argument(
        "--sigma-obs", type=float, default=0.10,
        help=(
            "observations.gauges.sigma_obs from the DA config, in transformed "
            "units. The within-cell spread measures the TOTAL per-gauge error; "
            "representativeness is what is left after removing this term."
        ),
    )
    parser.add_argument(
        "--protect-withheld-km", type=float, default=None,
        help=(
            "leave a cell unaggregated when its centroid would fall within this "
            "distance of a withheld gauge. Aggregation moves an assimilated "
            "location, and a centroid that lands closer to a withheld station "
            "than the source fold allowed would make this arm easier to score "
            "than the arms it is compared against. Pass the fold's "
            "minimum_retained_neighbour_km."
        ),
    )
    parser.add_argument(
        "--fail-under-km", type=float, default=None,
        help=(
            "abort if super-obbing places an assimilated location closer to a "
            "withheld gauge than this. Pass the source fold's "
            "minimum_retained_neighbour_km so the aggregated run cannot be "
            "scored more leniently than the experiment it is compared against."
        ),
    )
    parser.add_argument(
        "--report-only", action="store_true",
        help="measure the error budget and write no station table",
    )
    return parser.parse_args()


def load_transform(stats_path: str | None):
    """Return a callable mm -> transformed units, or None."""
    if stats_path is None:
        return None
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from bdhires.transforms import PrecipTransform  # noqa: E402

    stats = json.loads(Path(stats_path).read_text())
    transform = PrecipTransform.from_dict(stats["precip_transform"])
    return transform.forward


def haversine_km(
    lat0: np.ndarray, lon0: np.ndarray, lat1: np.ndarray, lon1: np.ndarray
) -> np.ndarray:
    """Great-circle distance matrix, shape (len(lat0), len(lat1)), in km."""
    a0, o0 = np.radians(np.asarray(lat0, float))[:, None], np.radians(np.asarray(lon0, float))[:, None]
    a1, o1 = np.radians(np.asarray(lat1, float))[None, :], np.radians(np.asarray(lon1, float))[None, :]
    term = np.sin((a0 - a1) / 2) ** 2 + np.cos(a0) * np.cos(a1) * np.sin((o0 - o1) / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.minimum(1.0, np.sqrt(term)))


def cell_index(lat: np.ndarray, lon: np.ndarray, cell_deg: float) -> np.ndarray:
    """Integer mesh cell for each station, as a single hashable key."""
    row = np.floor(np.asarray(lat, float) / cell_deg).astype(np.int64)
    column = np.floor(np.asarray(lon, float) / cell_deg).astype(np.int64)
    return row * 100_000 + column


def within_cell_spread(
    frame: pd.DataFrame, values: np.ndarray, keys: np.ndarray, min_days: int
) -> dict:
    """Pooled standard deviation of stations about their own cell-day mean.

    This is the point-versus-cell representativeness error, measured rather than
    assumed.  Only cell-days with at least two reporting stations contribute, and
    the pooled estimate uses the (n-1) degrees of freedom each such cell-day
    supplies, so a cell with five gauges is not weighted like a cell with two.
    """
    table = pd.DataFrame(
        {"key": keys, "date": frame["date"].to_numpy(), "value": np.asarray(values, float)}
    ).dropna(subset=["value"])
    grouped = table.groupby(["key", "date"])["value"]
    counts = grouped.transform("size").to_numpy()
    means = grouped.transform("mean").to_numpy()
    multi = counts >= 2
    if multi.sum() < min_days:
        return {"n_cell_days": int(multi.sum()), "pooled_sd": None, "median_members": None}
    residual = table["value"].to_numpy()[multi] - means[multi]
    # Sum of squared residuals over (sum of per-cell-day (n-1)).
    per_cell_day = (
        table[multi].groupby(["key", "date"])["value"].size().to_numpy() - 1
    ).sum()
    pooled = float(np.sqrt(np.sum(residual**2) / max(per_cell_day, 1)))
    return {
        "n_cell_days": int(
            table[multi].groupby(["key", "date"]).ngroups
        ),
        "n_station_days": int(multi.sum()),
        "pooled_sd": pooled,
        "median_members": float(
            np.median(table[multi].groupby(["key", "date"])["value"].size().to_numpy())
        ),
    }


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.stations, parse_dates=["date"])
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise SystemExit(f"{args.stations} lacks {sorted(missing)}")
    frame["station_id"] = frame["station_id"].astype(str)

    holdout: set[str] = set()
    if args.holdout_ids:
        holdout = {
            line.strip()
            for line in Path(args.holdout_ids).read_text().splitlines()
            if line.strip()
        }
        unknown = holdout - set(frame["station_id"])
        if unknown:
            raise SystemExit(
                f"{args.holdout_ids} names {len(unknown)} station(s) absent from "
                f"{args.stations}, e.g. {sorted(unknown)[:5]}"
            )

    assimilated = frame[~frame["station_id"].isin(holdout)].copy()
    withheld = frame[frame["station_id"].isin(holdout)].copy()
    if assimilated.empty:
        raise SystemExit("every station is held out; nothing to aggregate")

    keys = cell_index(assimilated["lat"], assimilated["lon"], args.cell_deg)
    assimilated["cell_key"] = keys

    forward = load_transform(args.stats)
    budget = {
        "cell_deg": args.cell_deg,
        "mm": within_cell_spread(
            assimilated, assimilated["precip_mm"], keys, args.min_pair_days
        ),
    }
    if forward is not None:
        transformed = forward(
            np.nan_to_num(assimilated["precip_mm"].to_numpy(float), nan=np.nan)
        )
        budget["transformed"] = within_cell_spread(
            assimilated, transformed, keys, args.min_pair_days
        )

    members_per_cell = assimilated.groupby("cell_key")["station_id"].nunique()
    size_histogram = members_per_cell.value_counts().sort_index()
    mean_members = float(members_per_cell[members_per_cell > 1].mean()) if (members_per_cell > 1).any() else 1.0

    recommendation = None
    if forward is not None and budget["transformed"]["pooled_sd"] is not None:
        # The spread of gauges about their own cell-day mean estimates the TOTAL
        # per-gauge error standard deviation: measurement/reporting error and
        # point-versus-cell representativeness together, since both are
        # independent between gauges in the same cell. The config splits that
        # budget in two, so representativeness is what remains once the assumed
        # sigma_obs is removed in quadrature.
        total_sd = float(budget["transformed"]["pooled_sd"])
        residual = total_sd**2 - float(args.sigma_obs) ** 2
        implied = float(np.sqrt(residual)) if residual > 0 else 0.0
        superob_total = total_sd / np.sqrt(mean_members)
        superob_residual = superob_total**2 - float(args.sigma_obs) ** 2
        recommendation = {
            "units": "transformed (the units of observations.gauges.*)",
            "assumed_sigma_obs": float(args.sigma_obs),
            "measured_total_gauge_error_sd": round(total_sd, 4),
            "implied_representativeness": round(implied, 4),
            "superob_implied_representativeness": round(
                float(np.sqrt(superob_residual)) if superob_residual > 0 else 0.0, 4
            ),
            "mean_members_in_multi_gauge_cells": round(mean_members, 3),
            "note": (
                "implied_representativeness is what a point gauge should carry at "
                f"{args.cell_deg} deg given the measured spread. The super-ob value "
                "is the same budget after averaging over the cell's members. "
                "Neither is applied by this script; setting one is a "
                "--set observations.gauges.representativeness= on the DA job."
            ),
        }

    manifest = {
        "source_table": str(args.stations),
        "cell_deg": args.cell_deg,
        "stations_in": int(frame["station_id"].nunique()),
        "stations_held_out": int(len(holdout)),
        "stations_assimilated_in": int(assimilated["station_id"].nunique()),
        "occupied_cells": int(len(members_per_cell)),
        "multi_gauge_cells": int((members_per_cell > 1).sum()),
        "cell_size_histogram": {int(k): int(v) for k, v in size_histogram.items()},
        "measured_error_budget": budget,
        "recommended_representativeness": recommendation,
    }

    if not args.report_only:
        if not args.out:
            raise SystemExit("--out is required unless --report-only is given")
        centroids = assimilated.groupby("cell_key")[["lat", "lon"]].mean()
        unaggregated = set(members_per_cell[members_per_cell == 1].index)

        protected: list = []
        if args.protect_withheld_km is not None and not withheld.empty:
            held_sites = withheld.groupby("station_id")[["lat", "lon"]].first()
            candidates = centroids.loc[
                [key for key in centroids.index if key not in unaggregated]
            ]
            if len(candidates):
                distance = haversine_km(
                    candidates["lat"].to_numpy(), candidates["lon"].to_numpy(),
                    held_sites["lat"].to_numpy(), held_sites["lon"].to_numpy(),
                )
                too_close = distance.min(axis=1) < args.protect_withheld_km
                protected = [
                    key for key, flag in zip(candidates.index, too_close) if flag
                ]
                unaggregated.update(protected)

        passthrough = assimilated[assimilated["cell_key"].isin(unaggregated)].drop(
            columns=["cell_key"]
        )
        clustered = assimilated[~assimilated["cell_key"].isin(unaggregated)]
        if clustered.empty:
            raise SystemExit(
                "every multi-gauge cell was protected or singleton; the aggregated "
                "table would be identical to the input. Raise --cell-deg or lower "
                "--protect-withheld-km."
            )
        manifest["cells_left_unaggregated_for_independence"] = len(protected)
        grouped = clustered.groupby(["cell_key", "date"], as_index=False)
        merged = grouped.agg(
            lat=("lat", "mean"),
            lon=("lon", "mean"),
            precip_mm=("precip_mm", "mean"),
            n_members=("precip_mm", "count"),
        )
        # A cell's coordinates must not wander day to day with which members
        # reported, or the observation operator would move under the analysis.
        centroid = clustered.groupby("cell_key")[["lat", "lon"]].mean()
        merged["lat"] = merged["cell_key"].map(centroid["lat"])
        merged["lon"] = merged["cell_key"].map(centroid["lon"])
        merged = merged[merged["n_members"] > 0].copy()
        merged["station_id"] = "SOB_" + merged["cell_key"].astype(str)
        merged["name"] = merged["station_id"]
        merged["source"] = "SUPEROB"
        if "accumulation_end_hour_utc" in frame.columns:
            merged["accumulation_end_hour_utc"] = (
                clustered.groupby("cell_key")["accumulation_end_hour_utc"]
                .mean()
                .reindex(merged["cell_key"])
                .to_numpy()
            )
        columns = [c for c in frame.columns if c in merged.columns]
        output = pd.concat(
            [withheld, passthrough, merged[columns]], ignore_index=True
        ).sort_values(["station_id", "date"])
        if output.duplicated(["station_id", "date"]).any():
            raise SystemExit("super-obbing produced duplicate station-days")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(args.out, index=False)
        manifest["out_table"] = str(args.out)
        manifest["stations_out"] = int(output["station_id"].nunique())
        manifest["superobs_created"] = int(merged["station_id"].nunique())

        # Independence audit: super-obbing moves an assimilated neighbour, so
        # report how close the nearest assimilated location now sits to each
        # withheld gauge. The source experiment already guarantees a retained
        # neighbour within its support radius; this states what that became.
        if not withheld.empty:
            assimilated_out = output[~output["station_id"].isin(holdout)]
            sites = assimilated_out.groupby("station_id")[["lat", "lon"]].first()
            held = withheld.groupby("station_id")[["lat", "lon"]].first()
            distance = haversine_km(
                held["lat"].to_numpy(), held["lon"].to_numpy(),
                sites["lat"].to_numpy(), sites["lon"].to_numpy(),
            )
            nearest = distance.min(axis=1)
            manifest["withheld_to_nearest_assimilated_km"] = {
                "min": float(nearest.min()),
                "median": float(np.median(nearest)),
                "max": float(nearest.max()),
            }
            if args.fail_under_km is not None and nearest.min() < args.fail_under_km:
                closest = held.index[int(np.argmin(nearest))]
                raise SystemExit(
                    f"super-obbing put an assimilated location {nearest.min():.2f} km "
                    f"from withheld station {closest}, inside the source "
                    f"experiment's {args.fail_under_km:.2f} km retained-neighbour "
                    "floor. The aggregated arm would be scored more leniently than "
                    "the arms it is compared against. Increase --cell-deg or "
                    "re-derive the holdout."
                )

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
