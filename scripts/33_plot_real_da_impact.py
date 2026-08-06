#!/usr/bin/env python
"""The OSSE impact figure, redrawn with REAL IMERG and REAL BMD gauges.

Why this exists
---------------
``osse_perfect.npz`` showed 83-95% of land improved when the pseudo-satellite
was noiseless CHIRPS. That established the DA machinery works. It did not
establish anything about real observations, because in that experiment the
satellite WAS the truth.

This draws the same six columns from a ``scripts/15_bmd_month_example.py`` dump,
so the real system can be read the same way -- with one change that matters.

Read the scores against the withheld gauges, not the map
--------------------------------------------------------
In the OSSE, CHIRPS is truth AND the source of the observations, so "error
reduction against CHIRPS" is exactly the right score and 83-95% means what it
says.

Here the observations are IMERG and BMD gauges, and they DISAGREE with CHIRPS:
measured over 43,781 station-days, CHIRPS runs -59.9 mm/day against the gauges
at intensities >= 50 mm/day. An analysis that correctly follows the gauges on a
heavy day therefore moves AWAY from CHIRPS and is painted brown in column F
while actually being more accurate.

So the map answers "where did the increments land, and are there bullseyes
around assimilated stations". The withheld-gauge table underneath answers "did
it help". Do not quote a CHIRPS-based percentage next to the OSSE's 83-95%;
they are not the same measurement.

Usage
-----
    python scripts/33_plot_real_da_impact.py \
        --dump data/processed/real_obs_trusted_may2024.npz \
        --arm combined --days 5 \
        --out-figure data/processed/real_da_impact_trusted.png \
        --out-json data/processed/real_da_impact_trusted.json
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ARMS = {
    "background": ("background", "background_at_stations"),
    "gauges": ("analysis_gauge", "gauge_analysis_at_stations"),
    "imerg": ("analysis_imerg", "imerg_analysis_at_stations"),
    "combined": ("analysis_combined", "combined_analysis_at_stations"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", nargs="+", required=True, help="script 15 .npz")
    parser.add_argument(
        "--arm", default="combined", choices=sorted(set(ARMS) - {"background"})
    )
    parser.add_argument("--days", type=int, default=5, help="wettest N days to draw")
    parser.add_argument("--out-figure", default="data/processed/real_da_impact.png")
    parser.add_argument("--out-json", default="data/processed/real_da_impact.json")
    return parser.parse_args()


def decode_time(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw)
    if np.issubdtype(raw.dtype, np.datetime64):
        return raw.astype("datetime64[D]")
    return raw.astype("datetime64[ns]").astype("datetime64[D]")


def background_lag(archive) -> int:
    """Offset between the background axis and the CHIRPS/analysis axis.

    BACKGROUND_DAY_OFFSET shifts the conditioning, so the dump stores two time
    axes. Indexing them positionally compares fields a day apart -- the mistake
    that once produced a residual correlation of 0.102 where the true value was
    0.376.
    """
    if not {"time", "background_time"}.issubset(archive.files):
        return 0
    main = decode_time(archive["time"])
    back = decode_time(archive["background_time"])
    offsets = (back - main).astype("timedelta64[D]").astype(int)
    unique = set(offsets.tolist())
    return int(next(iter(unique))) if len(unique) == 1 else 0


def score(predicted: np.ndarray, observed: np.ndarray) -> dict:
    ok = np.isfinite(predicted) & np.isfinite(observed)
    if not ok.any():
        return {}
    a, b = predicted[ok], observed[ok]
    return {
        "n": int(ok.sum()),
        "bias_mm": float(np.mean(a - b)),
        "mae_mm": float(np.mean(np.abs(a - b))),
        "rmse_mm": float(np.sqrt(np.mean((a - b) ** 2))),
    }


def station_scores(archive) -> dict:
    """Withheld-gauge scores for every arm. The honest comparison."""
    if "gauge_mm" not in archive.files:
        return {}
    gauges = np.asarray(archive["gauge_mm"], float)
    keep = (
        np.asarray(archive["eval_idx"], int)
        if "eval_idx" in archive.files
        else np.arange(gauges.shape[1])
    )
    observed = gauges[:, keep]
    out = {}
    for name, (_, station_key) in ARMS.items():
        if station_key not in archive.files:
            continue
        members = np.asarray(archive[station_key], float)[:, :, keep]
        out[name] = {
            "mean": score(members.mean(axis=1), observed),
            "median": score(np.median(members, axis=1), observed),
        }
    return out


def main() -> None:
    args = parse_args()
    archive = np.load(args.dump[0], allow_pickle=False)
    grid_key, _ = ARMS[args.arm]
    for required in ("background", grid_key, "chirps"):
        if required not in archive.files:
            raise SystemExit(f"{args.dump[0]} lacks {required!r}")

    lag = background_lag(archive)
    chirps = np.asarray(archive["chirps"], float)
    valid = (
        np.asarray(archive["valid"], bool)
        if "valid" in archive.files
        else np.ones(chirps.shape[1:], bool)
    )
    background = np.asarray(archive["background"], float).mean(axis=1)
    analysis = np.asarray(archive[grid_key], float).mean(axis=1)
    if lag > 0:
        background, analysis = background[:-lag], analysis[:-lag]
        chirps = chirps[lag:]
    elif lag < 0:
        background, analysis = background[-lag:], analysis[-lag:]
        chirps = chirps[:lag]
    times = decode_time(archive["time"])[max(lag, 0) : len(chirps) + max(lag, 0)]
    print(f"[impact] background-to-CHIRPS lag {lag:+d} day(s), {len(chirps)} usable days")

    masked = np.where(valid[None], chirps, np.nan)
    daily_mean = np.nanmean(masked, axis=(1, 2))
    order = np.argsort(-daily_mean)[: args.days]
    order = order[np.argsort(-daily_mean[order])]

    lat = np.asarray(archive["station_lat"], float) if "station_lat" in archive.files else None
    lon = np.asarray(archive["station_lon"], float) if "station_lon" in archive.files else None
    assim = np.asarray(archive["assim_idx"], int) if "assim_idx" in archive.files else np.array([], int)
    held = np.asarray(archive["eval_idx"], int) if "eval_idx" in archive.files else np.array([], int)
    extent = None
    if {"grid_lon", "grid_lat"}.issubset(archive.files):
        glon = np.asarray(archive["grid_lon"], float)
        glat = np.asarray(archive["grid_lat"], float)
        extent = (glon.min(), glon.max(), glat.min(), glat.max())

    titles = [
        "A.  CHIRPS",
        "B.  Background mean (before)",
        f"C.  Analysis mean ({args.arm})",
        "D.  Background error",
        "E.  Analysis error",
        "F.  Error reduction\ngreen = closer to CHIRPS, brown = further",
    ]
    figure, axes = plt.subplots(
        len(order), 6, figsize=(23, 3.4 * len(order)), squeeze=False
    )

    per_day = []
    for row, index in enumerate(order):
        truth = np.where(valid, chirps[index], np.nan)
        before = np.where(valid, background[index], np.nan)
        after = np.where(valid, analysis[index], np.nan)
        error_before = before - truth
        error_after = after - truth
        reduction = np.abs(error_before) - np.abs(error_after)

        finite = np.isfinite(reduction)
        improved = float((reduction[finite] > 0).mean()) if finite.any() else float("nan")
        per_day.append(
            {
                "date": str(times[index]),
                "chirps_mean_mm": float(np.nanmean(truth)),
                "fraction_of_land_closer_to_chirps": improved,
                "background_bias_mm": float(np.nanmean(error_before)),
                "analysis_bias_mm": float(np.nanmean(error_after)),
            }
        )

        rain_max = float(np.nanpercentile(truth, 99.5)) or 1.0
        err_max = float(np.nanpercentile(np.abs(error_before), 99)) or 1.0
        panels = [
            (truth, "viridis", 0, rain_max),
            (before, "viridis", 0, rain_max),
            (after, "viridis", 0, rain_max),
            (error_before, "RdBu_r", -err_max, err_max),
            (error_after, "RdBu_r", -err_max, err_max),
            (reduction, "BrBG", -err_max, err_max),
        ]
        for column, (field, cmap, low, high) in enumerate(panels):
            axis = axes[row][column]
            image = axis.imshow(
                field, origin="lower", extent=extent, cmap=cmap, vmin=low, vmax=high,
                aspect="auto",
            )
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03).ax.tick_params(
                labelsize=7
            )
            if lat is not None and extent is not None:
                if len(assim):
                    axis.scatter(
                        lon[assim], lat[assim], s=26, facecolors="white",
                        edgecolors="black", linewidths=0.7, zorder=5,
                    )
                if len(held):
                    axis.scatter(
                        lon[held], lat[held], s=34, marker="s", facecolors="none",
                        edgecolors="black", linewidths=1.1, zorder=6,
                    )
            if row == 0:
                axis.set_title(titles[column], fontsize=10)
            axis.tick_params(labelsize=7)
            if column == 0:
                axis.set_ylabel(
                    f"{times[index]}\n{np.nanmean(truth):.1f} mm/day\n"
                    f"{improved:.0%} of land closer",
                    fontsize=9,
                )

    stations = station_scores(archive)
    caption = ""
    if stations:
        parts = []
        for name in ("background", "gauges", "imerg", "combined"):
            entry = stations.get(name, {}).get("median")
            if entry:
                parts.append(f"{name} {entry['bias_mm']:+.2f}/{entry['mae_mm']:.2f}")
        caption = "   withheld-gauge median bias/MAE:  " + "   ".join(parts)

    figure.suptitle(
        f"Real IMERG + BMD gauges, arm '{args.arm}' -- {Path(args.dump[0]).name}\n"
        f"{len(assim)} assimilated (circles), {len(held)} withheld (squares)\n"
        "Column F is against CHIRPS, which DISAGREES with the gauges in the heavy tail "
        "(-59.9 mm/day at >=50). Brown can mean the analysis correctly followed a gauge.\n"
        "Judge skill by the withheld-gauge table, not by column F."
        + ("\n" + caption if caption else ""),
        fontsize=11,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.90])
    Path(args.out_figure).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out_figure, dpi=110, bbox_inches="tight")
    plt.close(figure)

    if stations:
        print("\n[impact] withheld-gauge scores (the number to trust):")
        print(f"    {'arm':<12}{'mean bias':>11}{'median bias':>13}{'mean MAE':>11}{'median MAE':>12}")
        for name in ("background", "gauges", "imerg", "combined"):
            entry = stations.get(name)
            if not entry or not entry.get("mean"):
                continue
            print(
                f"    {name:<12}{entry['mean']['bias_mm']:>+11.2f}"
                f"{entry['median']['bias_mm']:>+13.2f}"
                f"{entry['mean']['mae_mm']:>11.2f}{entry['median']['mae_mm']:>12.2f}"
            )
        print(
            "\n[impact] median columns are the spread-independent ones: the mean "
            "carries a Jensen inflation that differs per arm (2-6.6 mm/day measured "
            "on the multi-year run) and is not comparable across arms."
        )

    report = {
        "dump": args.dump[0],
        "arm": args.arm,
        "background_lag_days": lag,
        "per_day": per_day,
        "withheld_gauge_scores": stations,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2))
    print(f"\n[impact] wrote {args.out_figure} and {args.out_json}")


if __name__ == "__main__":
    main()
