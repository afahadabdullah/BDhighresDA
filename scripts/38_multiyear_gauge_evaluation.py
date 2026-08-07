#!/usr/bin/env python
"""Multi-year gauge-truth evaluation across every product and DA arm.

Pools any number of script-15 dumps -- rotated folds, several seasons, several
years -- and evaluates them at daily through monthly aggregation with the GAUGE
as truth throughout.  CHIRPS is one of the products being judged, never the
reference.

Why pooling the folds is the point
----------------------------------
Each fold withholds a different fifth of the network, so pooling the five folds
of a season gives every station a turn as a withheld station.  That is the only
way to get a withheld-gauge sample large enough to separate configurations: the
5-day, single-fold experiments that drove the earlier tuning had 35 station-days
of which 5 were wet, which cannot resolve the differences that were being read
from it.

Aggregation and the point-versus-area problem
---------------------------------------------
Scores are computed at 1, 5, 10 and 30 days.  Averaging removes the random part
of the point-vs-cell mismatch roughly as 1/N but leaves the systematic part
untouched, so ``MSE(N) = systematic + random/N`` splits them -- see
``bdhires.eval.representativeness``.  The systematic floor bounds how well any
gridded field can ever match a point gauge, and it is the number that says
whether a difference between two arms is real or below the noise.

Figures
-------
``daily_vs_monthly.png``   every product and arm against gauges at both scales
``score_matrix.png``       heatmaps of bias / MAE / correlation, arm x window
``aggregation_curve.png``  RMSE against averaging window with the fitted floor
``seasonal_cycle.png``     monthly means, all products, gauges in black
``station_map.png``        per-station bias for every product and arm
``taylor.png``             standard deviation against correlation, gauge-centred

Example
-------
    python scripts/38_multiyear_gauge_evaluation.py \\
        --dumps 'data/processed/bmd_imerg_eval_202*_may_sep/*fold*.npz' \\
        --stats data/processed/stats_cpc.json \\
        --out-dir data/processed/multiyear_gauge
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bdhires.eval.metrics import crps_ensemble  # noqa: E402
from bdhires.eval.representativeness import aggregation_decomposition  # noqa: E402

ARM_KEYS = {
    "background": "background_at_stations",
    "gauges": "gauge_analysis_at_stations",
    "satellite": "imerg_analysis_at_stations",
    "combined": "combined_analysis_at_stations",
}
PRODUCT_KEYS = {
    "CPC": "condition_at_stations",
    "IMERG": "imerg_at_stations",
    "CHIRPS": "chirps_at_stations",
}
COLOURS = {
    "background": "#8a8a8a", "gauges": "#1f6f8b",
    "satellite": "#4a7c1f", "combined": "#c1440e",
    "CPC": "#3b7ea1", "IMERG": "#6b9e3f", "CHIRPS": "#d4762a",
}
WINDOWS = (1, 5, 10, 30)
WET_MM = 1.0


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 140, "savefig.dpi": 140, "font.size": 9,
        "axes.grid": True, "grid.alpha": 0.25,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    return plt


def collect(paths: list[Path]) -> dict:
    """Pool withheld-station series from every dump onto one time axis.

    Only WITHHELD stations are taken from each dump.  Pooling the assimilated
    ones would be circular: the analysis was fitted to them, and on this system
    it fits them almost exactly (assimilated median error 0.00 mm/day).

    Ensembles are reduced to their mean here.  Members cannot be pooled across
    dumps -- different folds and seasons have different ensembles -- but the
    mean, the truth and the products are all per-station-day and pool cleanly.
    CRPS is therefore computed per dump and averaged, weighted by sample count.
    """
    records = {"time": [], "station": [], "truth": []}
    for name in list(ARM_KEYS) + list(PRODUCT_KEYS):
        records[name] = []
    crps_parts: dict[str, list[tuple[float, int]]] = {a: [] for a in ARM_KEYS}
    n_dumps = 0

    for path in paths:
        try:
            z = np.load(path, allow_pickle=False)
        except Exception as error:                                # noqa: BLE001
            print(f"[skip] {path.name}: {error}", flush=True)
            continue
        if "eval_idx" not in z or "gauge_mm" not in z:
            print(f"[skip] {path.name}: not a script-15 dump", flush=True)
            continue
        evl = z["eval_idx"]
        if not len(evl):
            continue
        times = z["time"].astype("datetime64[ns]").astype("datetime64[D]")
        names = z["station_name"][evl] if "station_name" in z else evl.astype(str)
        truth = z["gauge_mm"][:, evl]
        n_time = truth.shape[0]

        records["time"].append(np.repeat(times, len(evl)))
        records["station"].append(np.tile(np.asarray(names, dtype="U32"), n_time))
        records["truth"].append(truth.ravel())

        for arm, key in ARM_KEYS.items():
            if key not in z:
                records[arm].append(np.full(truth.size, np.nan))
                continue
            block = np.moveaxis(z[key], 1, 0)[:, :, evl]          # (M, T, S)
            records[arm].append(block.mean(axis=0).ravel())
            flat = block.reshape(block.shape[0], -1)
            ok = np.isfinite(truth.ravel()) & np.all(np.isfinite(flat), axis=0)
            if ok.sum() > 2:
                crps_parts[arm].append(
                    (crps_ensemble(flat[:, ok], truth.ravel()[ok]), int(ok.sum()))
                )
        for product, key in PRODUCT_KEYS.items():
            records[product].append(
                z[key][:, evl].ravel() if key in z else np.full(truth.size, np.nan)
            )
        n_dumps += 1

    if not n_dumps:
        raise SystemExit("no usable dumps")
    pooled = {k: np.concatenate(v) for k, v in records.items()}
    crps = {}
    for arm, parts in crps_parts.items():
        if not parts:
            continue
        weight = sum(n for _, n in parts)
        crps[arm] = sum(c * n for c, n in parts) / weight
    print(f"[collect] {n_dumps} dump(s), {pooled['truth'].size:,} withheld "
          f"station-days, {int(np.isfinite(pooled['truth']).sum()):,} with a gauge "
          f"value, {int((pooled['truth'] >= WET_MM).sum()):,} wet",
          flush=True)
    return {"pooled": pooled, "crps": crps, "n_dumps": n_dumps}


def aggregate(pooled: dict, name: str, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Block-mean a series and the truth together, per station.

    Aggregating a pooled vector without regard to station would average across
    the domain rather than in time.  Blocks are formed within each station and
    require the same days to be present in both series.
    """
    values, truth = pooled[name], pooled["truth"]
    both = np.isfinite(values) & np.isfinite(truth)
    if window == 1:
        return values[both], truth[both]
    out_v, out_t = [], []
    for station in np.unique(pooled["station"]):
        pick = (pooled["station"] == station) & both
        if pick.sum() < window:
            continue
        order = np.argsort(pooled["time"][pick])
        v = values[pick][order]
        t = truth[pick][order]
        blocks = len(v) // window
        out_v.append(v[: blocks * window].reshape(blocks, window).mean(axis=1))
        out_t.append(t[: blocks * window].reshape(blocks, window).mean(axis=1))
    if not out_v:
        return np.array([]), np.array([])
    return np.concatenate(out_v), np.concatenate(out_t)


def score(values: np.ndarray, truth: np.ndarray) -> dict:
    if values.size < 3:
        return {"n": int(values.size)}
    difference = values - truth
    wet = truth >= WET_MM
    return {
        "n": int(values.size),
        "n_wet": int(wet.sum()),
        "bias_mm": float(np.mean(difference)),
        "median_bias_mm": float(np.median(difference)),
        "mae_mm": float(np.mean(np.abs(difference))),
        "wet_mae_mm": float(np.mean(np.abs(difference[wet]))) if wet.any() else np.nan,
        "rmse_mm": float(np.sqrt(np.mean(difference**2))),
        "mse_mm2": float(np.mean(difference**2)),
        "correlation": float(np.corrcoef(values, truth)[0, 1])
        if values.std() > 0 and truth.std() > 0 else np.nan,
        "sd_ratio": float(values.std() / truth.std()) if truth.std() > 0 else np.nan,
    }


def evaluate(pooled: dict) -> dict:
    names = [n for n in list(PRODUCT_KEYS) + list(ARM_KEYS)
             if np.isfinite(pooled[n]).any()]
    out = {}
    for name in names:
        by_window = {}
        for window in WINDOWS:
            values, truth = aggregate(pooled, name, window)
            by_window[str(window)] = score(values, truth)
        usable = [(w, by_window[str(w)]["mse_mm2"]) for w in WINDOWS
                  if by_window[str(w)].get("n", 0) > 10
                  and np.isfinite(by_window[str(w)].get("mse_mm2", np.nan))]
        decomposition = (
            aggregation_decomposition(np.array([w for w, _ in usable], float),
                                      np.array([m for _, m in usable], float))
            if len(usable) >= 2 else None
        )
        out[name] = {"by_window": by_window, "decomposition": decomposition}
    return out


# --------------------------------------------------------------------------
# figures


def plot_daily_vs_monthly(pooled: dict, results: dict, out_path: Path) -> None:
    """Every product and arm against gauges, daily on top, monthly below."""
    plt = _style()
    names = list(results)
    figure, axes = plt.subplots(2, len(names), figsize=(2.7 * len(names), 6.4),
                                squeeze=False)
    for column, name in enumerate(names):
        for row, window in ((0, 1), (1, 30)):
            axis = axes[row][column]
            values, truth = aggregate(pooled, name, window)
            if values.size < 3:
                axis.axis("off"); continue
            top = float(np.nanpercentile(np.concatenate([values, truth]), 99.5))
            axis.scatter(truth, values, s=4, alpha=0.18,
                         color=COLOURS.get(name, "#777777"), edgecolor="none")
            axis.plot([0, top], [0, top], "k--", lw=1.1)
            s = results[name]["by_window"][str(window)]
            axis.set_title(f"{name} {'daily' if window == 1 else 'monthly'}\n"
                           f"r={s['correlation']:.2f}  bias={s['bias_mm']:+.2f}",
                           fontsize=8)
            axis.set_xlim(0, top); axis.set_ylim(0, top)
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlabel("gauge (mm/day)", fontsize=8)
            if column == 0:
                axis.set_ylabel("estimate (mm/day)", fontsize=8)
    figure.suptitle("Withheld gauges are truth on every x-axis; scatter tightens "
                    "with averaging and what remains is systematic", y=1.0)
    figure.tight_layout()
    figure.savefig(out_path, bbox_inches="tight")
    plt.close(figure)


def plot_score_matrix(results: dict, out_path: Path) -> None:
    """Heatmaps of bias, MAE and correlation over product x averaging window.

    Rows are ordered by daily MAE so the reading order is the ranking. Products
    and DA arms share the axis deliberately: the question that matters is
    whether assimilation beats simply taking a product off the shelf.
    """
    plt = _style()
    names = sorted(results, key=lambda n: results[n]["by_window"]["1"].get(
        "mae_mm", np.inf))
    panels = (("bias_mm", "bias (mm/day)", "RdBu_r", True),
              ("mae_mm", "MAE (mm/day)", "viridis_r", False),
              ("correlation", "correlation", "viridis", False))
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 0.52 * len(names) + 2.6))

    for axis, (metric, title, cmap, diverging) in zip(axes, panels):
        matrix = np.array([[results[n]["by_window"][str(w)].get(metric, np.nan)
                            for w in WINDOWS] for n in names], float)
        if diverging:
            limit = np.nanmax(np.abs(matrix)) or 1.0
            image = axis.imshow(matrix, cmap=cmap, vmin=-limit, vmax=limit,
                                aspect="auto")
        else:
            image = axis.imshow(matrix, cmap=cmap, aspect="auto")
        axis.set_xticks(range(len(WINDOWS)))
        axis.set_xticklabels([f"{w}d" for w in WINDOWS])
        axis.set_yticks(range(len(names)))
        axis.set_yticklabels(names, fontsize=8)
        axis.set_title(title, fontsize=9.5)
        axis.grid(False)
        for i in range(len(names)):
            for j in range(len(WINDOWS)):
                if np.isfinite(matrix[i, j]):
                    axis.text(j, i, f"{matrix[i, j]:.2f}", ha="center",
                              va="center", fontsize=6.5, color="#111111")
        figure.colorbar(image, ax=axis, shrink=0.85)

    figure.suptitle("Gauge truth. Rows sorted by daily MAE — products and DA arms "
                    "on the same axis on purpose", y=1.02)
    figure.tight_layout()
    figure.savefig(out_path, bbox_inches="tight")
    plt.close(figure)


def plot_aggregation_curve(results: dict, out_path: Path) -> None:
    """RMSE against averaging window, with each series' systematic floor."""
    plt = _style()
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for name, block in results.items():
        rmse = [block["by_window"][str(w)].get("rmse_mm", np.nan) for w in WINDOWS]
        colour = COLOURS.get(name, "#777777")
        style = "--" if name in PRODUCT_KEYS else "-"
        axis.plot(WINDOWS, rmse, style, marker="o", color=colour, lw=1.8, label=name)
        decomposition = block.get("decomposition")
        if decomposition and decomposition.get("model_is_valid"):
            axis.axhline(decomposition["systematic_rmse"], color=colour,
                         ls=":", lw=0.9, alpha=0.6)
    axis.set_xscale("log")
    axis.set_xticks(list(WINDOWS))
    axis.set_xticklabels([f"{w}d" for w in WINDOWS])
    axis.set_xlabel("averaging window")
    axis.set_ylabel("RMSE against gauges (mm/day)")
    axis.set_title("Dashed = raw product, solid = DA arm, dotted = systematic floor\n"
                   "the floor is what averaging can never remove", fontsize=9.5)
    axis.legend(fontsize=7.5, ncol=2)
    figure.tight_layout()
    figure.savefig(out_path, bbox_inches="tight")
    plt.close(figure)


def plot_seasonal_cycle(pooled: dict, out_path: Path) -> None:
    """Monthly domain-mean series over the whole record, gauges in black."""
    plt = _style()
    months = pooled["time"].astype("datetime64[M]")
    unique = np.unique(months)
    figure, axis = plt.subplots(figsize=(11.5, 4.4))
    names = [n for n in list(PRODUCT_KEYS) + list(ARM_KEYS)
             if np.isfinite(pooled[n]).any()]
    for name in names:
        series = [np.nanmean(pooled[name][months == m]) for m in unique]
        axis.plot(unique.astype("datetime64[D]"), series, "o-",
                  color=COLOURS.get(name, "#777777"), lw=1.5, ms=3, label=name,
                  ls="--" if name in PRODUCT_KEYS else "-")
    truth = [np.nanmean(pooled["truth"][months == m]) for m in unique]
    axis.plot(unique.astype("datetime64[D]"), truth, "k-", lw=2.6,
              label="gauge (truth)", zorder=6)
    axis.set_ylabel("monthly mean rainfall (mm/day)")
    axis.set_title("Monthly means at withheld stations across the record")
    axis.legend(fontsize=8, ncol=4)
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(out_path, bbox_inches="tight")
    plt.close(figure)


def plot_station_map(pooled: dict, stations: dict, out_path: Path) -> None:
    """Per-station mean bias for every product and arm on a shared scale."""
    plt = _style()
    names = [n for n in list(PRODUCT_KEYS) + list(ARM_KEYS)
             if np.isfinite(pooled[n]).any()]
    labels = np.array(sorted(set(pooled["station"]) & set(stations)))
    if not len(labels):
        print("[map] no station coordinates available; skipping")
        return
    lat = np.array([stations[s][0] for s in labels])
    lon = np.array([stations[s][1] for s in labels])

    biases = {}
    for name in names:
        values = []
        for station in labels:
            pick = (pooled["station"] == station)
            difference = pooled[name][pick] - pooled["truth"][pick]
            values.append(np.nanmean(difference) if np.isfinite(difference).any()
                          else np.nan)
        biases[name] = np.array(values)
    finite = np.concatenate([b[np.isfinite(b)] for b in biases.values()])
    limit = max(float(np.nanpercentile(np.abs(finite), 95)), 0.5)

    columns = min(4, len(names))
    rows = int(np.ceil(len(names) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(3.5 * columns, 3.4 * rows),
                                squeeze=False, sharex=True, sharey=True)
    for position, name in enumerate(names):
        axis = axes[position // columns][position % columns]
        scatter = axis.scatter(lon, lat, c=biases[name], cmap="RdBu_r",
                               vmin=-limit, vmax=limit, s=80,
                               edgecolor="#222222", linewidth=0.5)
        axis.set_title(f"{name}  mean {np.nanmean(biases[name]):+.2f} mm/day",
                       fontsize=9)
        axis.set_aspect("equal", adjustable="box")
    for extra in range(len(names), rows * columns):
        axes[extra // columns][extra % columns].axis("off")
    figure.colorbar(scatter, ax=axes.ravel().tolist(), shrink=0.7,
                    label="estimate - gauge (mm/day)")
    figure.suptitle("Per-station bias: coherent patches are systematic, "
                    "speckle is noise", y=1.01)
    figure.savefig(out_path, bbox_inches="tight")
    plt.close(figure)


def plot_taylor(results: dict, out_path: Path) -> None:
    """Standard-deviation ratio against correlation, gauge at (1, 1).

    A Taylor-style summary without the polar machinery: the distance from the
    gauge point combines amplitude and pattern error, which is what a single
    ranking number usually hides.
    """
    plt = _style()
    figure, axis = plt.subplots(figsize=(6.6, 5.4))
    for name, block in results.items():
        daily = block["by_window"]["1"]
        monthly = block["by_window"]["30"]
        for scores, marker, size, alpha in ((daily, "o", 70, 0.9),
                                            (monthly, "^", 70, 0.55)):
            if not np.isfinite(scores.get("correlation", np.nan)):
                continue
            axis.scatter(scores["correlation"], scores["sd_ratio"], marker=marker,
                         s=size, alpha=alpha, color=COLOURS.get(name, "#777777"),
                         edgecolor="#222222", linewidth=0.5,
                         label=name if marker == "o" else None)
    axis.scatter([1.0], [1.0], marker="*", s=320, color="#111111",
                 label="gauge (truth)", zorder=6)
    axis.axhline(1.0, color="#111111", lw=0.8, ls=":")
    axis.set_xlabel("correlation with gauges")
    axis.set_ylabel("sd(estimate) / sd(gauge)")
    axis.set_title("circles daily, triangles monthly — closer to the star is better")
    axis.legend(fontsize=7.5, ncol=2, loc="lower left")
    figure.tight_layout()
    figure.savefig(out_path, bbox_inches="tight")
    plt.close(figure)


# --------------------------------------------------------------------------


def print_table(results: dict, crps: dict) -> None:
    print()
    print("[scores] GAUGE IS TRUTH, withheld stations pooled over every dump.")
    print(f"    {'series':12s} {'win':>4s} {'n':>8s} {'nwet':>7s} {'bias':>8s} "
          f"{'MAE':>7s} {'wetMAE':>7s} {'RMSE':>7s} {'corr':>5s}")
    for name in sorted(results, key=lambda n: results[n]["by_window"]["1"].get(
            "mae_mm", np.inf)):
        for window in WINDOWS:
            s = results[name]["by_window"][str(window)]
            if not s.get("n"):
                continue
            print(f"    {name:12s} {str(window) + 'd':>4s} {s['n']:>8,d} "
                  f"{s.get('n_wet', 0):>7,d} {s['bias_mm']:>+8.2f} "
                  f"{s['mae_mm']:>7.2f} {s.get('wet_mae_mm', np.nan):>7.2f} "
                  f"{s['rmse_mm']:>7.2f} {s['correlation']:>5.2f}")
    if crps:
        print()
        print("    CRPS (ensemble arms only, per-dump then sample-weighted):")
        for arm, value in sorted(crps.items(), key=lambda kv: kv[1]):
            print(f"      {arm:12s} {value:.3f}")

    print()
    print("[floor] MSE(N) = systematic + random/N. The floor is what averaging")
    print("    cannot remove, and bounds how close ANY gridded field can get to a")
    print("    point gauge -- differences smaller than it are not measurable.")
    print(f"    {'series':12s} {'floor RMSE':>11s} {'random(1d)':>11s} {'R2':>6s}")
    for name, block in results.items():
        d = block.get("decomposition")
        if not d:
            continue
        if not d.get("model_is_valid"):
            print(f"    {name:12s} {'--':>11s} {'--':>11s} {'--':>6s}  UNUSABLE")
            continue
        print(f"    {name:12s} {d['systematic_rmse']:>11.2f} "
              f"{d['random_rmse_daily']:>11.2f} {d['r_squared']:>6.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-year gauge-truth evaluation of products and DA arms",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dumps", nargs="+", required=True)
    parser.add_argument("--stats", default=None, help="unused; accepted for symmetry")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    paths = sorted({Path(p) for pattern in args.dumps
                    for p in (glob.glob(pattern) or [pattern])})
    paths = [p for p in paths if p.suffix == ".npz" and p.exists()]
    if not paths:
        raise SystemExit(f"no NPZ dumps matched {args.dumps}")
    print(f"[setup] {len(paths)} dump(s)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    collected = collect(paths)
    pooled, crps = collected["pooled"], collected["crps"]

    stations = {}
    for path in paths:
        z = np.load(path, allow_pickle=False)
        if "station_name" in z and "station_lat" in z:
            for name, lat, lon in zip(z["station_name"], z["station_lat"],
                                      z["station_lon"]):
                stations.setdefault(str(name), (float(lat), float(lon)))

    results = evaluate(pooled)
    print_table(results, crps)

    plot_daily_vs_monthly(pooled, results, out_dir / "daily_vs_monthly.png")
    plot_score_matrix(results, out_dir / "score_matrix.png")
    plot_aggregation_curve(results, out_dir / "aggregation_curve.png")
    plot_seasonal_cycle(pooled, out_dir / "seasonal_cycle.png")
    plot_station_map(pooled, stations, out_dir / "station_map.png")
    plot_taylor(results, out_dir / "taylor.png")

    payload = {
        "n_dumps": collected["n_dumps"],
        "n_station_days": int(pooled["truth"].size),
        "n_wet_station_days": int((pooled["truth"] >= WET_MM).sum()),
        "crps": crps,
        "results": results,
        "note": "Gauge is truth throughout. CHIRPS is evaluated as a product and "
                "is never used as a reference.",
    }
    (out_dir / "multiyear_gauge.json").write_text(
        json.dumps(payload, indent=2, default=float))
    print()
    print(f"[done] wrote {out_dir / 'multiyear_gauge.json'}")
    for figure in sorted(out_dir.glob("*.png")):
        print(f"[done] wrote {figure}")


if __name__ == "__main__":
    main()
