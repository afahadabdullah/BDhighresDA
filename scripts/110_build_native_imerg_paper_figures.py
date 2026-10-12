#!/usr/bin/env python3
"""Regenerate Paper 1 product comparisons with native 0.1-degree IMERG.

First run script 96 with --run --native-imerg PATH ... . This renderer uses
its matched six-product scores, plus existing compact gridded paper exports.
Writes a separate figure folder; the 0.4-degree assimilation diagram and
ensemble-only figures do not need regeneration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
A = "dense_s6_bwdb_r4"
ORDER = [A, "imerg_native", "cpc", "chirps", "background"]
LABEL = {A: "SURMA-Flow", "imerg_native": "IMERG (0.1°)", "cpc": "CPC",
         "chirps": "CHIRPS", "background": "Background"}
COLOR = dict(zip(ORDER, ["#0B7A75", "#7B5EA7", "#2F6FB0", "#C8862A", "#8A94A6"]))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_scores(evidence):
    """Require the current scoring manifest, including hashes of native exports."""
    manifest = json.loads((evidence / "evidence_manifest.json").read_text())
    if manifest.get("status") != "matched_product_scores_complete" or not manifest.get("native_imerg_sources"):
        raise ValueError("run script 96 with --run --native-imerg before rendering native comparisons")
    outputs = {r["path"]: r["sha256"] for r in manifest["outputs"]}
    for name in ("native_imerg_comparison.csv", "native_imerg_station_scores.csv", "native_imerg_grids.npz"):
        if outputs.get(name) != digest(evidence / name):
            raise ValueError(f"{name}: missing or mismatched current-run output hash")
    rows = pd.read_csv(evidence / "native_imerg_comparison.csv")
    stations = pd.read_csv(evidence / "native_imerg_station_scores.csv")
    pooled = rows[rows.group == "pooled"].set_index("source")
    if not set(ORDER).issubset(pooled.index) or pooled.loc[ORDER, "n"].nunique() != 1:
        raise ValueError("native comparison requires matching product station-days")
    return rows, stations, manifest


def country_path(boundary):
    geometry = json.loads(boundary.read_text())
    verts, codes = [], []
    for feature in geometry["features"]:
        g = feature["geometry"]
        polygons = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for polygon in polygons:
            for ring in polygon:
                r = np.asarray(ring)[:, :2]
                verts.extend(r.tolist())
                codes.extend([MPath.MOVETO] + [MPath.LINETO] * (len(r) - 2) + [MPath.CLOSEPOLY])
    return MPath(np.asarray(verts), codes)


def map_field(ax, border, field=None, lat=None, lon=None, vmax=30):
    mesh = None
    if field is not None:
        mesh = ax.pcolormesh(lon, lat, field, cmap="YlGnBu", vmin=0, vmax=vmax,
                             shading="nearest", rasterized=True)
        mesh.set_clip_path(PathPatch(border, transform=ax.transData))
    ax.add_patch(PathPatch(border, facecolor="none", edgecolor="#1F2933", lw=.6, zorder=5))
    ax.set(xlim=(87.9, 92.8), ylim=(20.5, 26.75), xticks=[], yticks=[])
    ax.set_aspect(1 / np.cos(np.deg2rad(23.7)))
    for spine in ax.spines.values():
        spine.set_visible(False)
    return mesh


def build(args):
    rows, stations, evidence_manifest = load_scores(args.evidence)
    if digest(args.boundary) != evidence_manifest["evaluation_region"]["sha256"]:
        raise ValueError("map boundary differs from the scored Bangladesh region")
    border = country_path(args.boundary)
    with np.load(args.evidence / "native_imerg_grids.npz", allow_pickle=False) as f:
        native = {k: f[k] for k in f.files}
    for coord in ("lat", "lon"):
        if not np.allclose(np.diff(native[coord]), .1, atol=1e-5, rtol=0):
            raise ValueError("native IMERG maps require a 0.1-degree grid")
    grid_path = args.gridded / "temporal_mean_variability_grids.npz"
    with np.load(grid_path, allow_pickle=False) as f:
        grids = {k: f[k] for k in f.files}
    if not np.array_equal(native["seasonal_year"], grids["seasonal_year"]):
        raise ValueError("native and existing seasonal maps must use identical complete years")
    case_path = args.gridded / "data/fig05_subgrid_case_fields.csv"
    case = pd.read_csv(case_path)
    case_days = case.date.unique()
    if len(case_days) != 1 or str(native.get("case_date", "")) != str(case_days[0])[:10]:
        raise ValueError("native case date differs from the existing map; rerun script 96 with --native-case-date")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8,
                         "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    generated, numbers = [], {}

    def save(fig, name):
        for suffix in ("pdf", "png"):
            path = args.out_dir / f"{name}.{suffix}"
            fig.savefig(path, bbox_inches="tight", dpi=220)
            generated.append({"path": path.name, "sha256": digest(path)})
        plt.close(fig)
        print("wrote", name)

    # Figure 2: pooled and network-specific errors on the six-way intersection.
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 3.9), layout="constrained")
    for row, metric in enumerate(("rmse_mm", "mae_mm")):
        for col, (group, label) in enumerate((("pooled", "all"), ("network", "BMD"), ("network", "BWDB"))):
            scores = rows[(rows.group == group) & (rows.label == label)].set_index("source").loc[ORDER]
            ax = axes[row, col]; y = np.arange(5)[::-1]
            ax.barh(y, scores[metric], color=[COLOR[k] for k in ORDER], height=.65)
            for yi, value in zip(y, scores[metric]):
                ax.text(value + .3, yi, f"{value:.1f}", va="center", fontsize=7)
            ax.set(yticks=y, yticklabels=[LABEL[k] for k in ORDER] if col == 0 else [],
                   xlabel=f"{metric[:-3].upper()} (mm/day)", xlim=(0, scores[metric].max() * 1.22),
                   title=f"{'All withheld gauges' if col == 0 else label} · n = {int(scores.n.iloc[0]):,}")
    save(fig, "fig02_products")

    # Figure 3: finite, paired station RMSE differences and strict win counts.
    piv = stations.pivot(index="station_id", columns="source", values="rmse_mm")
    coord = stations.drop_duplicates("station_id").set_index("station_id")[["lat", "lon"]]
    piv = piv.join(coord)
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 3.15), layout="constrained")
    for i, source in enumerate((A, "imerg_native", "cpc", "chirps")):
        ax = axes[i]; map_field(ax, border)
        values = piv[A] if i == 0 else piv[A] - piv[source]
        mesh = ax.scatter(piv.lon, piv.lat, c=values, s=13, cmap="YlGnBu" if i == 0 else "BrBG_r",
                          vmin=5 if i == 0 else -12, vmax=30 if i == 0 else 12, edgecolors="#1F2933", lw=.25)
        ax.set_title("SURMA-Flow RMSE" if i == 0 else f"minus {LABEL[source]}")
        if i:
            win = int((values < 0).sum()); numbers.setdefault("station_wins", {})[source] = win
            ax.text(.5, -.04, f"better at {win} of {values.notna().sum()} gauges",
                    transform=ax.transAxes, ha="center", fontsize=7)
        if i in (0, 3):
            fig.colorbar(mesh, ax=ax, orientation="horizontal", shrink=.8, label="mm/day")
    save(fig, "fig03_station_maps")

    # Figure 4: native-resolution product scores in each evaluated year.
    fig, ax = plt.subplots(figsize=(7.2, 2.75), layout="constrained")
    for source in ORDER:
        g = rows[(rows.group == "year") & (rows.source == source)].sort_values("label")
        ax.plot(g.label.astype(str), g.rmse_mm, "o-", color=COLOR[source], label=LABEL[source])
    ax.set(xlabel="Year", ylabel="Daily RMSE (mm/day)")
    ax.legend(frameon=False, ncol=3, fontsize=7); ax.grid(alpha=.2)
    save(fig, "fig04_native_product_years")

    # Figures 5 and 8 retain each product's own grid; native IMERG is never upsampled.
    for name, is_case, vmax in (("fig05_seasonal_maps", False, 30), ("fig08_case", True, 120)):
        fig, axes = plt.subplots(1, 5, figsize=(7.2, 2.85), layout="constrained")
        means = {}
        for ax, source in zip(axes, ORDER):
            if source == "imerg_native":
                field = native["case_mm" if is_case else "seasonal_mean"]
                lat, lon = native["lat"], native["lon"]
            elif is_case:
                label = {A: A, "background": "background", "cpc": "CPC (conditioning)", "chirps": "CHIRPS (reference)"}[source]
                selected = case[case.source == label]
                grid = selected.pivot(index="lat", columns="lon", values="full_mm").sort_index().sort_index(axis=1)
                field, lat, lon = grid.values, grid.index.values, grid.columns.values
            else:
                field, lat, lon = grids[f"seasonal_mean__{source}"], grids["lat"], grids["lon"]
            mesh = map_field(ax, border, field, lat, lon, vmax)
            ax.set_title(LABEL[source], color=COLOR[source], fontweight="bold")
            if not is_case:
                xx, yy = np.meshgrid(lon, lat)
                inside = border.contains_points(np.column_stack([xx.ravel(), yy.ravel()])).reshape(field.shape)
                valid = inside & np.isfinite(field); weights = np.cos(np.deg2rad(yy))
                means[source] = float(np.sum(field[valid] * weights[valid]) / weights[valid].sum())
                ax.text(.5, -.03, f"country mean {means[source]:.1f}", transform=ax.transAxes, ha="center", fontsize=7)
        title = f"Rainfall, {case_days[0]}" if is_case else "May–September mean, " + ", ".join(map(str, native["seasonal_year"]))
        fig.suptitle(title, fontsize=9)
        fig.colorbar(mesh, ax=axes, shrink=.65, label="Rainfall (mm/day)", extend="max")
        if not is_case:
            numbers["country_mean_mm_day"] = means
        save(fig, name)

    # Figure 6: monthly and complete-season station mean scores.
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), layout="constrained")
    for ax, scale, title in zip(axes, ("monthly", "may_sep"), ("Monthly means", "Complete May–September means")):
        g = rows[(rows.group == "temporal") & (rows.label == scale)].set_index("source").loc[ORDER]
        y = np.arange(5)[::-1]
        ax.barh(y, g.rmse_mm, color=[COLOR[k] for k in ORDER])
        ax.set(yticks=y, yticklabels=[LABEL[k] for k in ORDER], xlabel="RMSE (mm/day)", title=f"{title} · n = {int(g.n.iloc[0]):,}")
    save(fig, "fig06_native_time_scales")

    bins = ["[0,1)", "[1,10)", "[10,25)", "[25,50)", "[50,100)", "[100,inf)"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), layout="constrained")
    for ax, metric in zip(axes, ("rmse_mm", "mae_mm", "bias_mm")):
        for source in ORDER:
            g = rows[(rows.group == "intensity") & (rows.source == source)].set_index("label").reindex(bins)
            ax.plot(range(6), g[metric], "o-", color=COLOR[source], label=LABEL[source], ms=3)
        ax.set(xticks=range(6), xticklabels=["<1", "1–10", "10–25", "25–50", "50–100", "≥100"],
               xlabel="Observed rain (mm/day)", ylabel=metric[:-3].upper() + " (mm/day)")
        ax.tick_params(axis="x", labelsize=6); ax.grid(alpha=.2)
    axes[0].legend(frameon=False, fontsize=6)
    save(fig, "fig08_product_intensity")

    pooled = rows[rows.group == "pooled"].set_index("source")
    numbers["headline"] = {k: {metric: float(pooled.loc[k, metric]) if pd.notna(pooled.loc[k, metric]) else None for metric in
                               ("n", "rmse_mm", "mae_mm", "bias_mm", "correlation")} for k in ORDER}
    numbers["comparison_counts"] = evidence_manifest["native_comparison_counts"]
    numbers["imerg_comparison_degrees"] = .1
    numbers["imerg_assimilation_degrees"] = .4
    (args.out_dir / "native_comparison_numbers.json").write_text(json.dumps(numbers, indent=2, allow_nan=False))
    inputs = [args.evidence / "evidence_manifest.json", args.evidence / "native_imerg_comparison.csv",
              args.evidence / "native_imerg_station_scores.csv", args.evidence / "native_imerg_grids.npz",
              grid_path, case_path, args.boundary]
    manifest = {"inputs": [{"path": str(p), "sha256": digest(p)} for p in inputs], "outputs": generated,
                "comparison_counts": numbers["comparison_counts"],
                "note": "0.1-degree IMERG product comparisons; assimilation remains 0.4 degrees. Update manuscript claims using the new scores. Yearly and temporal figures are separate replacements for the corresponding product panels."}
    (args.out_dir / "native_figure_manifest.json").write_text(json.dumps(manifest, indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence", type=Path, required=True, help="script-96 output containing current native exports")
    p.add_argument("--gridded", type=Path, required=True, help="existing Bangladesh paper gridded exports")
    p.add_argument("--boundary", type=Path, default=ROOT / "configs/geography/geoBoundaries-BGD-ADM0.geojson")
    p.add_argument("--out-dir", type=Path, default=ROOT / "manuscript/figures_native_imerg")
    args = p.parse_args(argv)
    try:
        build(args)
    except (ValueError, KeyError, FileNotFoundError) as error:
        print(f"[native-paper] {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
