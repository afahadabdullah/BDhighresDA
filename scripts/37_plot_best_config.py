#!/usr/bin/env python
"""Presentation figures for one DA configuration, with gauges as truth.

Distinct from scripts/36, which produces diagnostic small-multiples across every
run.  This makes the figures you would actually show for the configuration you
settled on.

Nothing here is SCORED against CHIRPS.  It is a satellite-gauge blend, the
weakest of the three products against BMD gauges at daily scale (r = 0.56
against CPC 0.76 and IMERG 0.71), and it runs 44 mm/day low where the gauges
report >= 50.  It does appear as one panel among the products, which is a
different thing: seeing how far the products disagree with each other and with
the gauges is most of the argument for not scoring against any of them.

Four figures
------------
``<run>_maps.png``
    One row per day: background mean, analysis mean, and the increment
    (analysis - background), with assimilated gauges as circles and withheld
    gauges as squares, both filled by their OBSERVED value on the same colour
    scale as the fields.  A gauge whose fill matches its surroundings in the
    analysis column but not in the background column is the assimilation
    working, and it can be read directly rather than inferred from a score.

``<run>_products.png``
    CPC, the satellite, CHIRPS, the background and the analysis side by side on
    a single colour scale per day, gauges overlaid on every panel.  The panel
    where the markers vanish into the field is the one that agrees with the
    gauges.

``<run>_product_scatter.png``
    The same products and arms against withheld gauges at station points, on
    shared axes, with bias and MAE per panel.

``<run>_summary.png``
    Withheld-station time series with the background spread band, a scorecard
    over the arms, the rank histogram, and spread against error.

Example
-------
    python scripts/37_plot_best_config.py \\
        --dump data/processed/cpc_test5/T1p0_20240501_20240505.npz \\
        --stats data/processed/stats_cpc.json \\
        --arm combined --sigma-rep 0.410 \\
        --out-dir data/processed/best_config
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bdhires.eval.calibration import rank_histogram  # noqa: E402
from bdhires.eval.metrics import crps_ensemble  # noqa: E402
from bdhires.transforms import PrecipTransform  # noqa: E402

ARM_KEYS = {
    "background": "background_at_stations",
    "gauges": "gauge_analysis_at_stations",
    "satellite": "imerg_analysis_at_stations",
    "combined": "combined_analysis_at_stations",
}
FIELD_KEYS = {
    "background": "background",
    "gauges": "analysis_gauge",
    "satellite": "analysis_imerg",
    "combined": "analysis_combined",
}
# Gridded input products. CHIRPS appears here as one product among several,
# which is a different thing from using it as TRUTH: the point of the panel is
# to show how far the products disagree with each other and with the gauges, and
# that disagreement is most of the argument for not scoring against any of them.
PRODUCT_KEYS = {"cpc": "condition", "satellite": "imerg", "chirps": "chirps"}
ARM_COLOURS = {
    "background": "#8a8a8a",
    "gauges": "#1f6f8b",
    "satellite": "#4a7c1f",
    "combined": "#c1440e",
}


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 140, "savefig.dpi": 140, "font.size": 9,
        "axes.grid": False,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    return plt


def load(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    out = {"name": path.stem}
    for key in (
        "gauge_mm", "assim_idx", "eval_idx", "station_name", "station_lat",
        "station_lon", "grid_lat", "grid_lon", "valid",
    ):
        out[key] = z[key] if key in z else None
    out["time"] = z["time"].astype("datetime64[ns]").astype("datetime64[D]")
    for arm, key in ARM_KEYS.items():
        out[arm] = np.moveaxis(z[key], 1, 0) if key in z else None
    for arm, key in FIELD_KEYS.items():
        out[f"field_{arm}"] = z[key] if key in z else None
    for label, key in PRODUCT_KEYS.items():
        out[f"product_{label}"] = z[key] if key in z else None
    for label, key in (("cpc", "condition_at_stations"),
                       ("satellite_obs", "imerg_at_stations"),
                       ("chirps", "chirps_at_stations")):
        out[f"at_{label}"] = z[key] if key in z else None
    return out


def plot_maps(dump: dict, arm: str, out_path: Path, max_days: int = 6) -> None:
    """Background, analysis and increment per day, with gauges drawn on top.

    Gauges are filled with their OBSERVED value on the field colour scale, so a
    station that stands out against its background but blends into the analysis
    is a visible, unaggregated demonstration that the increment went the right
    way.  Withheld stations are squares: those were never shown to the analysis,
    so they are the ones worth looking at.

    The increment column is symmetric about zero on its own scale, because the
    quantity of interest there is sign and structure rather than amount.
    """
    plt = _style()
    background = dump["field_background"]
    analysis = dump[f"field_{arm}"]
    if background is None or analysis is None:
        print(f"[maps] dump has no gridded field for arm {arm!r}; skipping")
        return

    background_mean = np.nanmean(background, axis=1)
    analysis_mean = np.nanmean(analysis, axis=1)
    increment = analysis_mean - background_mean

    valid = dump["valid"]
    if valid is not None:
        ocean = ~(np.asarray(valid) > 0)
        background_mean = np.where(ocean[None], np.nan, background_mean)
        analysis_mean = np.where(ocean[None], np.nan, analysis_mean)
        increment = np.where(ocean[None], np.nan, increment)

    days = min(max_days, background_mean.shape[0])
    lon, lat = dump["grid_lon"], dump["grid_lat"]
    extent = [lon[0], lon[-1], lat[0], lat[-1]]

    figure, axes = plt.subplots(days, 3, figsize=(11.5, 3.4 * days), squeeze=False)
    assim, evl = dump["assim_idx"], dump["eval_idx"]

    for row in range(days):
        top = float(np.nanpercentile(
            np.concatenate([background_mean[row].ravel(), analysis_mean[row].ravel()]), 99
        ))
        top = max(top, 1.0)
        limit = float(np.nanpercentile(np.abs(increment[row]), 99)) or 1.0
        gauge_today = dump["gauge_mm"][row]

        for column, (field, title, kwargs) in enumerate((
            (background_mean[row], "background mean",
             dict(cmap="viridis", vmin=0, vmax=top)),
            (analysis_mean[row], f"analysis mean ({arm})",
             dict(cmap="viridis", vmin=0, vmax=top)),
            (increment[row], "increment (analysis - background)",
             dict(cmap="RdBu_r", vmin=-limit, vmax=limit)),
        )):
            axis = axes[row][column]
            image = axis.imshow(field, origin="lower", extent=extent,
                                aspect="equal", **kwargs)
            figure.colorbar(image, ax=axis, shrink=0.82,
                            label="mm/day" if column < 2 else "mm/day")

            # Gauges: circles assimilated, squares withheld, filled with the
            # observed value on the SAME scale as the field in columns 0-1.
            for idx, marker, size in ((assim, "o", 34), (evl, "s", 52)):
                if idx is None or not len(idx):
                    continue
                observed = gauge_today[idx]
                ok = np.isfinite(observed)
                if column < 2:
                    axis.scatter(
                        dump["station_lon"][idx][ok], dump["station_lat"][idx][ok],
                        c=observed[ok], cmap="viridis", vmin=0, vmax=top,
                        marker=marker, s=size, edgecolor="#111111", linewidth=0.7,
                        zorder=5,
                    )
                else:
                    axis.scatter(
                        dump["station_lon"][idx][ok], dump["station_lat"][idx][ok],
                        facecolor="none", marker=marker, s=size,
                        edgecolor="#111111", linewidth=0.7, zorder=5,
                    )
            if row == 0:
                axis.set_title(title, fontsize=9.5)
            if column == 0:
                axis.set_ylabel(f"{str(dump['time'][row])}\n"
                                f"gauge mean {np.nanmean(gauge_today):.1f} mm/day",
                                fontsize=8)
            axis.set_xticks([]); axis.set_yticks([])

    figure.suptitle(
        f"{dump['name']} — circles assimilated, squares withheld; gauge fill uses "
        "the field colour scale\nno CHIRPS anywhere: gauges are the only truth here",
        y=1.0, fontsize=10.5,
    )
    figure.tight_layout()
    figure.savefig(out_path, bbox_inches="tight")
    plt.close(figure)


def plot_product_maps(dump: dict, arm: str, out_path: Path, max_days: int = 4) -> None:
    """Every gridded product beside the background and the analysis, one row a day.

    All panels share one colour scale per day so the comparison is honest: a
    product that looks drier here IS drier, not differently normalised.  Gauges
    are drawn on every panel filled with their observed value on that same
    scale, so the panel where the circles blend into their surroundings is the
    field that agrees with the gauges.

    The satellite product is stored on its coarse footprint grid and is expanded
    by nearest neighbour rather than interpolated -- a footprint IS a box
    average, and smoothing it would draw structure the observation does not
    have.
    """
    plt = _style()
    lon, lat = dump["grid_lon"], dump["grid_lat"]
    extent = [lon[0], lon[-1], lat[0], lat[-1]]
    nlat, nlon = len(lat), len(lon)

    panels = []
    for label, key in PRODUCT_KEYS.items():
        field = dump.get(f"product_{label}")
        if field is None or not np.isfinite(field).any():
            continue
        if field.shape[-2:] != (nlat, nlon):
            factor = nlat // field.shape[-2]
            field = np.repeat(np.repeat(field, factor, axis=-2), factor, axis=-1)
            field = field[..., :nlat, :nlon]
            label = f"{label} ({factor * (lat[1] - lat[0]):.2f} deg footprints)"
        panels.append((label, field))

    background = dump["field_background"]
    analysis = dump[f"field_{arm}"]
    if background is not None:
        panels.append(("background mean", np.nanmean(background, axis=1)))
    if analysis is not None:
        panels.append((f"analysis ({arm})", np.nanmean(analysis, axis=1)))
    if not panels:
        print("[products] nothing gridded to plot; skipping")
        return

    valid = dump["valid"]
    ocean = ~(np.asarray(valid) > 0) if valid is not None else None
    if ocean is not None:
        panels = [(name, np.where(ocean[None], np.nan, f)) for name, f in panels]

    days = min(max_days, min(f.shape[0] for _, f in panels))
    figure, axes = plt.subplots(days, len(panels),
                                figsize=(2.9 * len(panels), 2.9 * days), squeeze=False)
    # A per-row colorbar rules out tight_layout, so the spacing is set directly.
    figure.subplots_adjust(hspace=0.06, wspace=0.04)

    for row in range(days):
        stack = np.concatenate([f[row].ravel() for _, f in panels])
        top = max(float(np.nanpercentile(stack, 99)), 1.0)
        gauge_today = dump["gauge_mm"][row]
        for column, (name, field) in enumerate(panels):
            axis = axes[row][column]
            image = axis.imshow(field[row], origin="lower", extent=extent,
                                aspect="equal", cmap="viridis", vmin=0, vmax=top)
            for idx, marker, size in ((dump["assim_idx"], "o", 26),
                                      (dump["eval_idx"], "s", 44)):
                if idx is None or not len(idx):
                    continue
                observed = gauge_today[idx]
                ok = np.isfinite(observed)
                axis.scatter(dump["station_lon"][idx][ok], dump["station_lat"][idx][ok],
                             c=observed[ok], cmap="viridis", vmin=0, vmax=top,
                             marker=marker, s=size, edgecolor="#111111",
                             linewidth=0.6, zorder=5)
            if row == 0:
                axis.set_title(name, fontsize=8.5)
            if column == 0:
                axis.set_ylabel(f"{str(dump['time'][row])}\n"
                                f"gauge mean {np.nanmean(gauge_today):.1f} mm/day",
                                fontsize=7.5)
            axis.set_xticks([]); axis.set_yticks([])
        figure.colorbar(image, ax=axes[row].tolist(), shrink=0.85, label="mm/day")

    figure.suptitle(
        f"{dump['name']} — all products on one scale per day; circles assimilated, "
        "squares withheld,\ngauges filled with their OBSERVED value: the panel where "
        "markers blend in is the field that agrees with them",
        y=1.0, fontsize=10,
    )
    figure.savefig(out_path, bbox_inches="tight")
    plt.close(figure)


def plot_product_scatter(dump: dict, arm: str, out_path: Path) -> None:
    """Each product and each arm against the gauges, at station points.

    Same axes throughout, so the panels are directly comparable, and the 1:1
    line is the only thing a product should sit on.
    """
    plt = _style()
    evl = dump["eval_idx"]
    truth = dump["gauge_mm"][:, evl].ravel()

    series = {}
    for label in ("cpc", "satellite_obs", "chirps"):
        values = dump.get(f"at_{label}")
        if values is not None and np.isfinite(values).any():
            series[label] = values[:, evl].ravel()
    for name in ("background", arm):
        block = dump.get(name)
        if block is not None:
            series[f"{name} (mean)"] = np.nanmean(block[:, :, evl], axis=0).ravel()
    if not series:
        return

    figure, axes = plt.subplots(1, len(series), figsize=(3.1 * len(series), 3.4),
                                squeeze=False)
    finite = np.concatenate([v[np.isfinite(v)] for v in series.values()] + [truth[np.isfinite(truth)]])
    top = float(np.nanpercentile(finite, 99)) or 1.0

    for axis, (label, values) in zip(axes[0], series.items()):
        ok = np.isfinite(values) & np.isfinite(truth)
        axis.scatter(truth[ok], values[ok], s=26, alpha=0.7,
                     color=ARM_COLOURS.get(label.split()[0], "#777777"),
                     edgecolor="none")
        axis.plot([0, top], [0, top], "k--", lw=1.2)
        bias = float(np.mean(values[ok] - truth[ok])) if ok.any() else float("nan")
        mae = float(np.mean(np.abs(values[ok] - truth[ok]))) if ok.any() else float("nan")
        axis.set_title(f"{label}\nbias {bias:+.2f}  MAE {mae:.2f}", fontsize=8.5)
        axis.set_xlabel("gauge (mm/day)")
        axis.set_xlim(0, top); axis.set_ylim(0, top)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.25)
    axes[0][0].set_ylabel("estimate (mm/day)")
    figure.suptitle("Withheld gauges on the x-axis throughout — the only truth here",
                    y=1.02, fontsize=10)
    figure.tight_layout()
    figure.savefig(out_path, bbox_inches="tight")
    plt.close(figure)


def plot_summary(
    dump: dict, arm: str, transform: PrecipTransform, sigma_rep: float,
    out_path: Path, seed: int = 0,
) -> None:
    """Time series, scorecard, rank histogram and spread-versus-error."""
    plt = _style()
    evl = dump["eval_idx"]
    dates = dump["time"]
    arms = [a for a in ("background", "gauges", "satellite", "combined")
            if dump.get(a) is not None]

    n_station = len(evl)
    columns = min(4, n_station)
    rows = int(np.ceil(n_station / columns))
    figure = plt.figure(figsize=(3.5 * columns, 2.5 * rows + 7.0))
    grid = figure.add_gridspec(rows + 2, columns, height_ratios=[1] * rows + [1.5, 1.5],
                               hspace=0.55, wspace=0.3)

    for position, station in enumerate(evl):
        axis = figure.add_subplot(grid[position // columns, position % columns])
        band = dump["background"][:, :, station]
        low, high = np.nanpercentile(band, [5, 95], axis=0)
        axis.fill_between(dates, low, high, color=ARM_COLOURS["background"],
                          alpha=0.25, lw=0, label="background 5-95%")
        for name in arms:
            if name == "background":
                continue
            axis.plot(dates, np.nanmean(dump[name][:, :, station], axis=0), "o-",
                      color=ARM_COLOURS[name], ms=3, lw=1.4, label=name)
        axis.plot(dates, dump["gauge_mm"][:, station], "k-", lw=2.2,
                  label="gauge (truth)", zorder=6)
        axis.plot(dates, dump["gauge_mm"][:, station], "ko", ms=4, zorder=7)
        axis.set_title(str(dump["station_name"][station]), fontsize=8.5)
        axis.tick_params(axis="x", rotation=45, labelsize=6.5)
        axis.grid(alpha=0.25)
        if position == 0:
            axis.legend(fontsize=6, loc="upper left")

    truth = dump["gauge_mm"][:, evl].ravel()

    # --- scorecard
    axis = figure.add_subplot(grid[rows, :])
    metrics = {}
    for name in arms:
        members = dump[name][:, :, evl].reshape(dump[name].shape[0], -1)
        ok = np.isfinite(truth) & np.all(np.isfinite(members), axis=0)
        mean = members[:, ok].mean(axis=0)
        metrics[name] = {
            "CRPS": crps_ensemble(members[:, ok], truth[ok]),
            "mean MAE": float(np.mean(np.abs(mean - truth[ok]))),
            "median MAE": float(np.median(np.abs(mean - truth[ok]))),
        }
    labels = list(metrics["background"])
    width = 0.8 / len(arms)
    for offset, name in enumerate(arms):
        axis.bar(np.arange(len(labels)) + offset * width,
                 [metrics[name][m] for m in labels], width=width,
                 color=ARM_COLOURS[name], label=name)
        for position, m in enumerate(labels):
            axis.text(position + offset * width, metrics[name][m],
                      f"{metrics[name][m]:.2f}", ha="center", va="bottom", fontsize=6.5)
    axis.set_xticks(np.arange(len(labels)) + 0.4 - width / 2)
    axis.set_xticklabels(labels)
    axis.set_ylabel("mm/day  (CRPS in mm/day)")
    axis.set_title("Withheld gauges only — lower is better", fontsize=9.5)
    axis.legend(fontsize=7.5, ncol=len(arms))
    axis.grid(alpha=0.25, axis="y")

    members = dump[arm][:, :, evl].reshape(dump[arm].shape[0], -1)
    ok = np.isfinite(truth) & np.all(np.isfinite(members), axis=0)

    # --- rank histogram
    axis = figure.add_subplot(grid[rows + 1, : max(columns // 2, 1)])
    histogram = rank_histogram(members[:, ok], truth[ok])
    axis.bar(np.arange(len(histogram)), histogram / max(histogram.sum(), 1),
             color=ARM_COLOURS[arm])
    axis.axhline(1.0 / len(histogram), color="#111111", ls="--", lw=1.2,
                 label="flat = calibrated")
    axis.set_title(f"rank histogram, {arm}\nU-shape = under-dispersed, "
                   "dome = over-dispersed", fontsize=8.5)
    axis.set_xticks([])
    axis.legend(fontsize=7)
    axis.grid(alpha=0.25, axis="y")

    # --- spread versus error, binned
    axis = figure.add_subplot(grid[rows + 1, max(columns // 2, 1):])
    spread = members[:, ok].std(axis=0, ddof=1)
    error = np.abs(members[:, ok].mean(axis=0) - truth[ok])
    if spread.size > 8:
        edges = np.quantile(spread, np.linspace(0, 1, 6))
        edges = np.unique(edges)
        centres, errors = [], []
        for low, high in zip(edges[:-1], edges[1:]):
            inside = (spread >= low) & (spread <= high)
            if inside.sum() < 2:
                continue
            centres.append(float(np.mean(spread[inside])))
            errors.append(float(np.sqrt(np.mean(error[inside] ** 2))))
        axis.plot(centres, errors, "o-", color=ARM_COLOURS[arm], lw=1.8)
        top = max(max(centres, default=1), max(errors, default=1)) * 1.15
        axis.plot([0, top], [0, top], "k--", lw=1.2, label="perfect (spread = error)")
        axis.set_xlim(0, top); axis.set_ylim(0, top)
        axis.legend(fontsize=7)
    axis.set_xlabel("ensemble spread (mm/day)")
    axis.set_ylabel("RMSE (mm/day)")
    axis.set_title("Spread against error — above the line is over-confident",
                   fontsize=8.5)
    axis.grid(alpha=0.25)

    figure.suptitle(f"{dump['name']} — gauge truth, arm '{arm}'", y=0.995,
                    fontsize=11)
    figure.savefig(out_path, bbox_inches="tight")
    plt.close(figure)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Presentation figures for one DA configuration (gauge truth)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dump", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--arm", default="combined",
                        choices=sorted(ARM_KEYS))
    parser.add_argument("--sigma-rep", type=float, default=0.410)
    parser.add_argument("--max-days", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = json.loads(Path(args.stats).read_text())
    transform = PrecipTransform(**stats["precip_transform"])

    dump = load(Path(args.dump))
    if dump.get(args.arm) is None:
        raise SystemExit(f"{args.dump} has no arm {args.arm!r}")
    print(f"[setup] {dump['name']}: {len(dump['time'])} day(s), "
          f"{len(dump['assim_idx'])} assimilated / {len(dump['eval_idx'])} withheld")

    maps_path = out_dir / f"{dump['name']}_maps.png"
    summary_path = out_dir / f"{dump['name']}_summary.png"
    plot_maps(dump, args.arm, maps_path, max_days=args.max_days)
    plot_product_maps(dump, args.arm, out_dir / f"{dump['name']}_products.png",
                      max_days=min(args.max_days, 4))
    plot_product_scatter(dump, args.arm,
                         out_dir / f"{dump['name']}_product_scatter.png")
    metrics = plot_summary(dump, args.arm, transform, args.sigma_rep,
                           summary_path, seed=args.seed)

    print()
    print(f"[scores] withheld gauges, arm '{args.arm}':")
    for name, block in metrics.items():
        print(f"    {name:12s} " + "  ".join(f"{k}={v:.3f}" for k, v in block.items()))
    (out_dir / f"{dump['name']}_scores.json").write_text(
        json.dumps(metrics, indent=2, default=float)
    )
    print()
    for figure in sorted(out_dir.glob(f"{dump['name']}*")):
        print(f"[done] wrote {figure}")


if __name__ == "__main__":
    main()
