#!/usr/bin/env python
"""Paper figures 1-3: the pipeline, the measured observation error, the design.

These are the method and design figures. They depend only on the station
catalogue and the existing schematic, not on any model run, so they can be
built before a single GPU job finishes.

Figure 1  Pipeline schematic. Not redrawn -- the hand-made SVG in docs/figures
          is better than anything matplotlib would produce -- but copied under
          the paper's name so the manuscript has one figure directory, with a
          manifest recording where it came from.

Figure 2  The variogram, and the representativeness error read off it. This is
          the paper's methodological differentiator: most precipitation DA
          papers assume R, this one measures it. The panel shows the Matheron
          estimator, the fitted exponential, and the block dispersion variance
          that converts the fit into a point-versus-cell error.

Figure 3  Station network and the rotated fold assignment. Establishes that
          folds are disjoint, exhaustive and geographically spread, which is
          what makes the withheld-station evaluation defensible.

Every figure writes its numbers to <out-dir>/data/ as CSV plus a provenance
manifest. See src/bdhires/paper/style.py.

Example
-------
    python scripts/46_paper_figures.py \\
        --stations data/stations/bmd_daily.csv \\
        --start 2021-01-01 --end 2024-12-31 \\
        --out-dir docs/paper_figures
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bdhires.bmd import spread_folds  # noqa: E402
from bdhires.eval.representativeness import (  # noqa: E402
    empirical_variogram,
    fit_variogram,
    representativeness_sigma,
)
from bdhires.paper import (  # noqa: E402
    FIGURE_WIDTH_ONE_COLUMN,
    FIGURE_WIDTH_TWO_COLUMN,
    PALETTE,
    save_figure,
    use_paper_style,
)
from bdhires.transforms import PrecipTransform  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stations", default="data/stations/bmd_daily.csv")
    parser.add_argument("--start", default="2021-01-01",
                        help="Evaluation window start. 2021 rather than 2020 "
                             "because the prior's val split is [2019, 2020]; "
                             "2021-2025 is the strict test period.")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--cell-km", type=float, default=5.0,
                        help="Model cell size for the point-to-cell error "
                             "(0.05 deg is ~5.5 km).")
    parser.add_argument("--pipeline-svg", default="docs/figures/pipeline.png")
    parser.add_argument("--out-dir", default="docs/paper_figures")
    parser.add_argument("--skip", default="", help="comma-separated: 1,2,3")
    return parser.parse_args()


def load_gauges(path: str, start: str, end: str):
    """Station set and daily values over the window, as the paper defines it.

    ``load_stations`` is imported here rather than at module scope because
    ``bdhires.data`` reaches ``bdhires.da`` for ``StationSet``, which imports
    torch. Everything else these figures need -- the variogram, the fold
    partition, the transform, the style -- is torch-free, so a deferred import
    lets the module load anywhere and fail only if gauge data is actually
    wanted, with a message that says why.
    """
    try:
        from bdhires.data import load_stations
    except ImportError as error:  # pragma: no cover - environment dependent
        raise SystemExit(
            f"could not import bdhires.data ({error}). Figures 2 and 3 read "
            f"the gauge catalogue, and that import path currently requires "
            f"torch. Run this in the project environment, or pass --skip 2,3 "
            f"to build only the schematic."
        ) from error
    station_path = Path(path)
    if not station_path.is_file():
        raise SystemExit(
            f"station file not found: {station_path}. Run through "
            "slurm/make_paper_figures.sh so the per-station BMD archive is "
            "converted automatically, or pass --stations with a canonical "
            "long-form CSV."
        )
    dates = np.arange(np.datetime64(start, "D"),
                      np.datetime64(end, "D") + np.timedelta64(1, "D"))
    stations, values = load_stations(path, dates, grid=None, min_coverage=0.5)
    if values.shape[1] == 0:
        raise SystemExit(
            f"no BMD station has at least 50% coverage between {start} and "
            f"{end} in {station_path}"
        )
    valid_days = np.any(np.isfinite(values), axis=1)
    if not valid_days.any():
        raise SystemExit(
            f"no valid BMD observations between {start} and {end} in "
            f"{station_path}"
        )
    actual_start = str(dates[np.flatnonzero(valid_days)[0]])
    actual_end = str(dates[np.flatnonzero(valid_days)[-1]])
    print(
        f"[stations] {values.shape[1]} stations; requested {start}..{end}; "
        f"available observations {actual_start}..{actual_end}"
    )
    return stations, values, dates


# ---------------------------------------------------------------- figure 1

def figure_pipeline(args, out_dir: Path) -> None:
    source = Path(args.pipeline_svg)
    if not source.exists():
        print(f"[fig01] SKIP: {source} not found")
        return
    stem = out_dir / f"fig01_pipeline{source.suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, stem)
    # A copied figure still needs a manifest, otherwise it is the one figure in
    # the paper with no recorded origin.
    from bdhires.paper.style import _write_table, _git_commit
    import json
    from datetime import datetime, timezone
    data_dir = out_dir / "data"; data_dir.mkdir(exist_ok=True)
    (data_dir / "fig01_pipeline_manifest.json").write_text(json.dumps({
        "figure": "fig01_pipeline",
        "caption": "Two-phase pipeline: a generative prior trained without "
                   "observations, then assimilation that never retrains it.",
        "files": [stem.name],
        "data": {},
        "sources": [str(source), "docs/figures/pipeline.src.svg"],
        "note": "Hand-drawn schematic, copied rather than generated.",
        "git_commit": _git_commit(),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2) + "\n")
    print(f"[fig01] {stem.name}: copied from {source}")


# ---------------------------------------------------------------- figure 2

def figure_variogram(args, out_dir: Path) -> None:
    plt = use_paper_style()
    stations, values, _ = load_gauges(args.stations, args.start, args.end)
    transform = PrecipTransform()
    transformed = transform.forward(np.nan_to_num(values, nan=np.nan))

    # Bins out to ~400 km: beyond that the domain runs out and pair counts
    # collapse, which is what min_pairs guards against.
    edges = np.concatenate([np.arange(0.0, 120.0, 15.0),
                            np.arange(120.0, 420.0, 40.0)])
    empirical = empirical_variogram(transformed, np.asarray(stations.lat),
                                    np.asarray(stations.lon), edges,
                                    min_pairs=30)
    distance = np.asarray(empirical["distance_km"], float)
    gamma = np.asarray(empirical["gamma"], float)
    pairs = np.asarray(empirical["n_pairs"], float)

    fit = fit_variogram(distance, gamma, pairs, units="transformed")
    sigma_rep = representativeness_sigma(fit, args.cell_km)

    figure, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH_TWO_COLUMN, 2.6))

    axis = axes[0]
    axis.scatter(distance, gamma, s=np.clip(pairs / pairs.max() * 40, 6, 40),
                 color=PALETTE["gauges"], zorder=3,
                 label="Matheron estimator")
    fine = np.linspace(0.0, distance.max(), 300)
    axis.plot(fine, fit(fine), color=PALETTE["combined"],
              label=f"exponential fit (range {fit.range_km:.0f} km)")
    # VariogramFit.sill is the PARTIAL sill: gamma = nugget + sill*(1-exp(-h/r)).
    # Plotting it as a bare line puts it BELOW the nugget whenever the nugget
    # dominates, which reads as an error. The two meaningful references are the
    # nugget and the total sill it asymptotes to.
    axis.axhline(fit.nugget, color=PALETTE["null"], ls=":",
                 label=f"nugget {fit.nugget:.3f}")
    axis.axhline(fit.total_sill, color=PALETTE["truth"], ls="--", lw=0.8,
                 label=f"total sill {fit.total_sill:.3f}")
    axis.set_xlabel("separation (km)")
    axis.set_ylabel(r"$\gamma$ (transformed units$^2$)")
    axis.set_title("(a) Daily rainfall variogram")
    axis.set_ylim(bottom=0.0)
    axis.legend(loc="lower right", frameon=True, framealpha=0.92,
                edgecolor="none")

    axis = axes[1]
    cells = np.linspace(1.0, 60.0, 60)
    sigmas = np.array([representativeness_sigma(fit, c) for c in cells])
    axis.plot(cells, sigmas, color=PALETTE["gauges"])
    axis.axvline(args.cell_km, color=PALETTE["combined"], ls="--", lw=0.9)
    axis.plot([args.cell_km], [sigma_rep], "o", color=PALETTE["combined"], ms=5)
    axis.annotate(rf"$\sigma_{{\rm rep}}$ = {sigma_rep:.3f} at {args.cell_km:.0f} km",
                  xy=(args.cell_km, sigma_rep), xytext=(8, -6),
                  textcoords="offset points", fontsize=7,
                  color=PALETTE["combined"])
    axis.set_xlabel("cell size (km)")
    axis.set_ylabel(r"$\sigma_{\rm rep}$ (transformed)")
    axis.set_title("(b) Point-to-cell representativeness error")
    # From zero, so the reader sees the MAGNITUDE of the error rather than a
    # fourth-decimal wiggle. When the nugget dominates the fit, sigma_rep is
    # nearly flat in cell size, and that flatness is itself the message.
    axis.set_ylim(bottom=0.0, top=float(np.nanmax(sigmas)) * 1.18)

    figure.tight_layout()
    save_figure(
        figure, out_dir, "02", "observation_error",
        data={
            "variogram": {"distance_km": distance, "gamma": gamma,
                          "n_pairs": pairs,
                          "fitted": fit(distance)},
            "sigma_rep_vs_cell": {"cell_km": cells, "sigma_rep": sigmas},
            "fit": [{"nugget": fit.nugget, "sill": fit.sill,
                     "range_km": fit.range_km, "units": "transformed",
                     "cell_km": args.cell_km, "sigma_rep": sigma_rep,
                     "n_stations": int(len(stations.lat)),
                     "window": f"{args.start}..{args.end}"}],
        },
        sources=[args.stations],
        caption="Observation error measured, not assumed. (a) Matheron "
                "variogram of transformed daily rainfall over the BMD network "
                "with a fitted exponential model. (b) Point-to-cell "
                "representativeness error implied by the block dispersion "
                "variance, evaluated at the model cell size.")
    plt.close(figure)
    print(f"    nugget {fit.nugget:.4f}  sill {fit.sill:.4f}  "
          f"range {fit.range_km:.1f} km  sigma_rep {sigma_rep:.4f}")


# ---------------------------------------------------------------- figure 3

def figure_station_folds(args, out_dir: Path) -> None:
    plt = use_paper_style()
    stations, values, dates = load_gauges(args.stations, args.start, args.end)
    lat = np.asarray(stations.lat, float)
    lon = np.asarray(stations.lon, float)
    folds = spread_folds(lat, lon, n_splits=args.folds)

    assignment = np.full(lat.size, -1, int)
    for index, members in enumerate(folds):
        assignment[np.asarray(members, int)] = index
    if (assignment < 0).any():
        raise SystemExit("spread_folds left a station unassigned; folds must "
                         "be exhaustive for the holdout to be valid")

    coverage = np.isfinite(values).mean(axis=0)
    wet = np.nanmean(values >= 1.0, axis=0)

    figure, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH_TWO_COLUMN, 3.2))

    axis = axes[0]
    colours = plt.cm.viridis(np.linspace(0.1, 0.9, args.folds))
    for index in range(args.folds):
        pick = assignment == index
        axis.scatter(lon[pick], lat[pick], s=34, color=colours[index],
                     edgecolor="white", linewidth=0.5, zorder=3,
                     label=f"fold {index} (n={int(pick.sum())})")
    axis.set_xlabel("longitude (°E)")
    axis.set_ylabel("latitude (°N)")
    axis.set_title(f"(a) BMD network, {lat.size} stations, "
                   f"{args.folds} rotated folds")
    axis.legend(loc="upper left", ncol=2, fontsize=6)
    axis.set_aspect("equal", adjustable="datalim")

    axis = axes[1]
    order = np.argsort(coverage)
    axis.barh(np.arange(lat.size), coverage[order] * 100.0,
              color=PALETTE["gauges"], height=0.8)
    axis.set_yticks([])
    axis.set_xlabel("days reporting (%)")
    axis.set_title("(b) Record completeness per station\n"
                   f"{args.start} to {args.end}")
    axis.axvline(50.0, color=PALETTE["warn"], ls="--", lw=0.9)
    # Above the bars, not through them: at 100% coverage every bar spans the
    # label's position and the text becomes unreadable.
    axis.set_ylim(-1.5, lat.size + 1.5)
    axis.text(50.0, lat.size + 0.6, "min_coverage = 50%", fontsize=6,
              color=PALETTE["warn"], ha="center", va="bottom")

    figure.tight_layout()
    save_figure(
        figure, out_dir, "03", "station_folds",
        data={
            "stations": {"station_id": np.asarray(stations.ids, str),
                         "lat": lat, "lon": lon, "fold": assignment,
                         "coverage_fraction": coverage, "wet_fraction": wet},
            "fold_sizes": [{"fold": i, "n_stations": int((assignment == i).sum())}
                           for i in range(args.folds)],
        },
        sources=[args.stations],
        caption="Evaluation design. (a) The BMD gauge network partitioned into "
                "geographically spread, disjoint and exhaustive folds; every "
                "station is withheld exactly once. (b) Reporting completeness "
                "over the evaluation window.")
    plt.close(figure)
    sizes = [int((assignment == i).sum()) for i in range(args.folds)]
    print(f"    fold sizes {sizes}, total {sum(sizes)} of {lat.size} stations")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    for number, builder in (("1", figure_pipeline),
                            ("2", figure_variogram),
                            ("3", figure_station_folds)):
        if number in skip:
            print(f"[fig0{number}] skipped by request")
            continue
        builder(args, out_dir)
    print(f"\n[done] figures and data in {out_dir}/ and {out_dir}/data/")


if __name__ == "__main__":
    main()
