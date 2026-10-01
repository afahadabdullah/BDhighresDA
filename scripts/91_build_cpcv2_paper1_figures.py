#!/usr/bin/env python3
"""Build manuscript figures from the copied, completed Paper 1 evaluation.

No model inference or new scientific verification is performed. Inputs remain
unchanged; the manifest records content hashes and derived temporal summaries.
Run: python scripts/91_build_cpcv2_paper1_figures.py
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bdhires.country import DEFAULT_BOUNDARY, read_boundary, grid_mask, map_layer
from matplotlib.colors import TwoSlopeNorm

FINAL = "dense_s6_bwdb_r4"
METHODS = ["background", FINAL]
NAMES = {"background": "Background", FINAL: "SURMA-Flow", "chirps": "CHIRPS",
         "imerg": "IMERG (0.4°)", "cpc": "CPC (same day)"}
COLORS = {"background": "#64748b", FINAL: "#007f78", "chirps": "#b07c24",
          "imerg": "#8056a3", "cpc": "#3479ae"}


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def number(row, key):
    return float(row[key]) if row[key] else np.nan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("paper1_cpcv2/bangladesh/superob-final"))
    parser.add_argument("--output", type=Path, default=Path("manuscript/figures"))
    parser.add_argument("--boundary-geojson", type=Path, default=DEFAULT_BOUNDARY)
    args = parser.parse_args()
    country, region = read_boundary(args.boundary_geojson)
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 11, "axes.labelsize": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.dpi": 220})
    inputs = [args.input / "daily_withheld_scores.csv",
              args.input / "temporal_withheld_scores.csv",
              args.input / "gridded/long_term_withheld_product_matrix.csv",
              args.input / "gridded/withheld_gauge_subgrid_anomalies.csv",
              args.input / "gridded/data/fig05_subgrid_case_fields.csv",
              args.input / "paper1_evaluation.json"]
    daily, temporal, products, anomalies, fields = [read_csv(p) for p in inputs[:5]]
    evaluation = json.loads(inputs[5].read_text())
    if evaluation["status"] not in ("withheld_evaluation_complete", "withheld_summary_recomputed"):
        raise ValueError("Paper figures require a completed withheld evaluation")
    if evaluation.get("evaluation_region", {}).get("sha256") != region["sha256"]:
        raise ValueError("figures require Bangladesh-only scores with the same boundary identity")
    n = evaluation["counts"]["scored_station_days"]
    assert all(int(r["pooled_daily_n"]) == n for r in products)
    station_path = args.input / "gridded/long_term_withheld_station_scores.csv"
    station_products = read_csv(station_path)
    inputs.append(station_path)
    source_order = ["background", FINAL, "chirps", "imerg", "cpc"]
    station_counts = [{r["station_id"]: int(r["n"]) for r in station_products if r["source"] == source}
                      for source in source_order]
    if not all(counts == station_counts[0] for counts in station_counts) or sum(station_counts[0].values()) != n:
        raise ValueError("product/network scores require identical station counts for all methods")
    strata_path = args.input / "gridded/withheld_product_strata.csv"
    strata = read_csv(strata_path) if strata_path.is_file() and strata_path.stat().st_size else []
    if strata:
        for group, label in {(r["group"], r["label"]) for r in strata}:
            subset = [r for r in strata if (r["group"], r["label"]) == (group, label)]
            if ({r["source"] for r in subset} != set(source_order) or len(subset) != 5
                    or len({int(r["n"]) for r in subset}) != 1
                    or any(int(r["matched_daily_n"]) != n for r in subset)):
                raise ValueError("stratified product panels require all five methods on the same sample")
        inputs.append(strata_path)
    for name in ("fig02_daily_skill_calibration", "fig03_intensity", "fig04_product_comparison", "fig05_temporal", "fig06_spatial_case", "fig07_subgrid", "fig08_product_intensity", "fig09_product_temporal"):
        for extension in ("pdf", "png"):
            (args.output / (name + "." + extension)).unlink(missing_ok=True)
    inputs.append(args.boundary_geojson)
    manifest = {"inputs": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                           for p in inputs], "figures": [], "temporal_summary": [], "evaluation_region": region, "pending": evaluation.get("pending", [])}
    manifest["primary_verification"] = "original withheld Bangladesh gauges; common station-day samples across products"
    manifest["supporting_figures"] = ["fig03_intensity", "fig05_temporal", "fig06_spatial_case", "fig07_subgrid"]

    def save(fig, name):
        for extension in ("pdf", "png"):
            fig.savefig(args.output / f"{name}.{extension}", bbox_inches="tight")
        plt.close(fig)
        manifest["figures"].append(name)

    def selected(group, label, method):
        return next(r for r in daily if (r["group"], r["label"], r["method"])
                    == (group, label, method))

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6), layout="constrained")
    groups = [("pooled", "all"), ("source", "BMD"), ("source", "BWDB")]
    for ax, metric, title in zip(axes.flat[:2], ["crps", "rmse"],
                                ["a  Bangladesh daily fair CRPS", "b  Bangladesh daily RMSE"]):
        x = np.arange(3)
        for i, method in enumerate(METHODS):
            ax.bar(x + (i - .5) * .34, [number(selected(g, label, method), metric) for g, label in groups],
                   .34, color=COLORS[method], label=NAMES[method])
        ax.set(xticks=x, xticklabels=["Pooled", "BMD", "BWDB"], ylabel="mm/day", title=title)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    groups = [("pooled", "all"), ("source", "BMD"), ("source", "BWDB")]
    for ax, metric, title, target in zip(axes.flat[2:], ["coverage_90", "spread_skill"],
                                        ["c  Central 90% interval coverage", "d  Spread / RMSE"], [.9, 1.]):
        x = np.arange(3)
        for i, method in enumerate(METHODS):
            ax.bar(x + (i - .5) * .34, [number(selected(g, l, method), metric) for g, l in groups],
                   .34, color=COLORS[method], label=NAMES[method])
        ax.axhline(target, ls="--", color="#a84638", lw=1.2)
        ax.set(xticks=x, xticklabels=["Pooled", "BMD", "BWDB"], ylim=(0, 1.10), title=title)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    axes[0, 0].legend(frameon=False, fontsize=9)
    axes[0, 0].set_ylim(0, 11.5)
    save(fig, "fig02_daily_skill_calibration")

    if any(r["group"] == "intensity" for r in daily):
        bins = ["[0,1)", "[1,10)", "[10,25)", "[25,50)", "[50,100)", "[100,inf)"]
        labels = ["<1", "1–10", "10–25", "25–50", "50–100", "≥100"]
        counts = [int(selected("intensity", b, FINAL)["n"]) for b in bins]
        fig, axes = plt.subplots(2, 2, figsize=(10, 6.7), layout="constrained")
        for ax, metric, title in zip(axes.flat, ["crps", "rmse", "bias", "coverage_90"],
                                    ["a  Fair CRPS", "b  Ensemble-mean RMSE", "c  Ensemble-mean bias", "d  90% interval coverage"]):
            for method in METHODS:
                ax.plot(np.arange(6), [number(selected("intensity", b, method), metric) for b in bins],
                        marker="o", color=COLORS[method], label=NAMES[method], lw=1.8)
            ax.set(xticks=np.arange(6), xticklabels=labels, title=title,
                   xlabel="Observed daily rainfall (mm/day)", ylabel="Fraction" if metric == "coverage_90" else "mm/day")
            ax.grid(alpha=.18)
            if metric == "bias":
                ax.axhline(0, color="black", lw=.6)
            if metric == "coverage_90":
                ax.axhline(.9, ls="--", color="#a84638", lw=1.)
                ax.set_ylim(0, 1.05)
        axes[0, 0].legend(frameon=False)
        fig.suptitle("Station-days per bin: " + "; ".join(f"{l}: {n:,}" for l, n in zip(labels, counts)), fontsize=10)
        save(fig, "fig03_intensity")
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), layout="constrained")
    for col, network in enumerate(("Pooled", "BMD", "BWDB")):
        selected = [r for r in station_products if network == "Pooled" or
                    (r["station_id"].startswith("BWDB_") == (network == "BWDB"))]
        count = sum(int(r["n"]) for r in selected if r["source"] == FINAL)
        for row, metric in enumerate(("rmse_mm", "mae_mm")):
            ax = axes[row, col]
            values = []
            for source in source_order:
                group = [r for r in selected if r["source"] == source]
                power = 2 if row == 0 else 1
                values.append((sum(int(r["n"]) * number(r, metric) ** power for r in group) / count) ** (1 / power))
            ax.barh(np.arange(5), values, color=[COLORS[s] for s in source_order])
            ax.set(yticks=np.arange(5), yticklabels=[NAMES[s] for s in source_order],
                   xlabel="mm/day", title=f"{'abcdef'[row * 3 + col]}  {network} {'RMSE' if row == 0 else 'MAE'} · n={count:,}")
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=.18)
            ax.set_axisbelow(True)
    save(fig, "fig04_product_comparison")

    # Optional all-product panels require daily arrays, not reconstructed bins.
    for group, labels, name, titles in (
            ("intensity", ["[0,1)", "[1,10)", "[10,25)", "[25,50)", "[50,100)", "[100,inf)"],
             "fig08_product_intensity", ["<1", "1-10", "10-25", "25-50", "50-100", ">=100"]),
            ("temporal", ["monthly", "may_sep"], "fig09_product_temporal", ["Monthly mean", "May-September mean"])):
        available = [r for r in strata if r["group"] == group]
        if not available:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout="constrained")
        for ax, metric, title in zip(axes, ("rmse_mm", "mae_mm"), ("RMSE", "MAE")):
            for source in source_order:
                values = [next((number(r, metric) for r in available if r["source"] == source and r["label"] == label), np.nan)
                          for label in labels]
                ax.plot(np.arange(len(labels)), values, "o-", label=NAMES[source], color=COLORS[source])
            ax.set(xticks=np.arange(len(labels)), xticklabels=titles, ylabel="mm/day", title=title,
                   xlabel="Observed rainfall (mm/day)" if group == "intensity" else "Aggregation of daily means")
            ax.grid(alpha=.18)
        counts = [next((int(r["n"]) for r in available if r["source"] == FINAL and r["label"] == label), 0) for label in labels]
        fig.suptitle("Common withheld-gauge sample: " + "; ".join(f"{label}: {count:,}" for label, count in zip(titles, counts)), fontsize=9)
        axes[0].legend(frameon=False, fontsize=8)
        save(fig, name)

    fig, axes = plt.subplots(2, 2, figsize=(9, 7), layout="constrained")
    for col, scale in enumerate(["monthly", "may_sep"]):
        for row, (obs_key, pred_key) in enumerate([
                ("observed_mean_mm_day", "predicted_mean_mm_day"),
                ("observed_daily_temporal_sd_mm", "predicted_daily_temporal_sd_mm")]):
            ax = axes[row, col]
            for method in METHODS:
                rows = [r for r in temporal if r["scale"] == scale and r["method"] == method]
                obs = np.array([number(r, obs_key) for r in rows])
                pred = np.array([number(r, pred_key) for r in rows])
                ax.scatter(obs, pred, s=12, alpha=.35, color=COLORS[method], label=NAMES[method], rasterized=True)
                if row == 0:
                    error = pred - obs
                    manifest["temporal_summary"].append({"scale": scale, "method": method, "n": len(rows),
                        "n_periods": len({r["period"] for r in rows}), "rmse": float(np.sqrt(np.mean(error ** 2))),
                        "mae": float(np.mean(np.abs(error))), "bias": float(np.mean(error))})
            end = np.ceil(max(number(r, key) for r in temporal if r["scale"] == scale
                              for key in (obs_key, pred_key)) / 10) * 10
            ax.plot([0, end], [0, end], ls="--", color="black", lw=.8)
            ax.set(xlim=(0, end), ylim=(0, end), aspect="equal", xlabel="Gauge (mm/day)",
                   ylabel="Ensemble mean (mm/day)", title=f"{'abcd'[row * 2 + col]}  " +
                   ("Monthly" if scale == "monthly" else "May–September") + (" mean" if row == 0 else " daily variability"))
            ax.grid(alpha=.15)
    axes[0, 0].legend(frameon=False, loc="upper left", fontsize=9)
    save(fig, "fig05_temporal")

    lats = np.array(sorted({number(r, "lat") for r in fields}))
    lons = np.array(sorted({number(r, "lon") for r in fields}))
    lat_index, lon_index = {v: i for i, v in enumerate(lats)}, {v: i for i, v in enumerate(lons)}
    sources = ["background", FINAL, "chirps", "imerg", "cpc"]
    arrays = {(s, k): np.full((len(lats), len(lons)), np.nan) for s in sources for k in ("full_mm", "subgrid_mm")}
    for r in fields:
        source = r["source"].lower().split()[0]
        source = FINAL if source == FINAL else source
        for k in ("full_mm", "subgrid_mm"):
            arrays[(source, k)][lat_index[number(r, "lat")], lon_index[number(r, "lon")]] = number(r, k)
    extent = [lons[0] - .025, lons[-1] + .025, lats[0] - .025, lats[-1] + .025]
    fig, axes = plt.subplots(2, 5, figsize=(12.8, 6.8), layout="constrained")
    inside = grid_mask(lats, lons, country)
    arrays = {key: np.where(inside, value, np.nan) for key, value in arrays.items()}
    full_max = max(float(np.nanmax(arrays[s, "full_mm"])) for s in sources)
    residual_max = max(float(np.nanmax(np.abs(arrays[s, "subgrid_mm"]))) for s in sources)
    for col, source in enumerate(sources):
        top = map_layer(axes[0, col], arrays[source, "full_mm"], lats, lons, country,
                        vmin=0, vmax=full_max, cmap="YlGnBu")
        bottom = map_layer(axes[1, col], arrays[source, "subgrid_mm"], lats, lons, country,
                           norm=TwoSlopeNorm(0, -residual_max, residual_max), cmap="RdBu_r")
        axes[0, col].set_title("CPC\n(conditioning)" if source == "cpc" else NAMES[source], fontsize=15)
        for row in (0, 1):
            axes[row, col].set(xticks=[89, 91, 93], yticks=[21, 23, 25], xlabel="Longitude (°E)" if row else "")
            axes[row, col].tick_params(labelsize=13)
            axes[row, col].xaxis.label.set_size(14)
            if col:
                axes[row, col].set_yticklabels([])
            else:
                axes[row, col].set_ylabel("Latitude (°N)", fontsize=14)
    cb1 = fig.colorbar(top, ax=axes[0, :], shrink=.82, pad=.015, label="Daily precipitation (mm/day)")
    cb2 = fig.colorbar(bottom, ax=axes[1, :], shrink=.82, pad=.015, label="Residual below 0.4° (mm/day)")
    for cb in (cb1, cb2):
        cb.ax.tick_params(labelsize=13)
        cb.ax.yaxis.label.set_size(14)
    fig.suptitle("Bangladesh · 29 May 2024 · ensemble means and reference fields", fontsize=17)
    save(fig, "fig06_spatial_case")

    if anomalies:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), layout="constrained")
        for ax, metric, title, multiplier in zip(axes, ["correlation", "mse_skill_vs_no_subgrid"],
                                                ["a  Gauge-anomaly correlation", "b  MSE skill against zero residual"], [1, 100]):
            x = np.arange(3)
            for i, method in enumerate(METHODS):
                rows = [next(r for r in anomalies if r["method"] == method and r["coarse_baseline"] == source)
                        for source in ["chirps", "imerg", "cpc"]]
                ax.bar(x + (i - .5) * .34, [number(r, metric) * multiplier for r in rows], .34,
                       color=COLORS[method], label=NAMES[method])
            ax.set(xticks=x, xticklabels=["CHIRPS", "IMERG", "CPC"], xlabel="Coarse baseline",
                   ylabel="r" if multiplier == 1 else "%", title=title)
            ax.grid(axis="y", alpha=.18)
            ax.set_axisbelow(True)
        axes[0].legend(frameon=False, fontsize=9)
        save(fig, "fig07_subgrid")
    (args.output / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["temporal_summary"], indent=2))
    print(f"Wrote {len(manifest['figures'])} figures to {args.output}")


if __name__ == "__main__":
    main()
