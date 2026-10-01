#!/usr/bin/env python
"""Figures for the restructured SURMA-Flow Paper 1 manuscript.

CPU post-processing only: every panel is drawn from exports that already exist
in the repository. No training, sampling or re-scoring of ensembles happens here.

    python scripts/95_build_paper1_restructured_figures.py

Inputs (all relative to the repository root)
    paper1_cpcv2/bangladesh/superob-final/   Bangladesh-only scores (132 sites, 29,401 pairs)
    paper1_cpcv2/superob-final/              archive-domain scores (133 sites, 29,615 pairs)
    configs/geography/geoBoundaries-BGD-ADM0.geojson
    data/stations/data_2020_2025/Stations.csv, output/bwdb_analysis/bwdb_stations_with_distance.csv

Outputs: manuscript/figures/fig0X_*.pdf/.png and manuscript/figures/restructured_numbers.json
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path as MPath

ROOT = Path(__file__).resolve().parents[1]
BD = ROOT / "paper1_cpcv2/bangladesh/superob-final"
WD = ROOT / "paper1_cpcv2/superob-final"
OUT = ROOT / "manuscript/figures"
OUT.mkdir(parents=True, exist_ok=True)

A = "dense_s6_bwdb_r4"
C = {"surma": "#0B7A75", "background": "#8A94A6", "chirps": "#C8862A",
     "imerg": "#7B5EA7", "cpc": "#2F6FB0", "ink": "#1F2933", "muted": "#6B7785",
     "grid": "#E6E9EE", "accent": "#B4413C"}
KEY = {A: "surma", "background": "background", "chirps": "chirps", "imerg": "imerg", "cpc": "cpc"}
LAB = {A: "SURMA-Flow", "background": "Background (prior)", "chirps": "CHIRPS",
       "imerg": "IMERG (0.4°)", "cpc": "CPC (same day)"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "Inter", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.edgecolor": C["muted"],
    "axes.labelcolor": C["ink"], "text.color": C["ink"], "xtick.color": C["muted"],
    "ytick.color": C["muted"], "axes.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42, "savefig.dpi": 300,
})
RAIN = plt.get_cmap("YlGnBu")


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=220)
    plt.close(fig)
    print("wrote", name)


def ttl(ax, letter, title, **kw):
    """Panel letter (bold) followed by a left-aligned title, both in offset points."""
    ax.annotate(letter, (0, 1), (0, 5), xycoords="axes fraction", textcoords="offset points", fontweight="bold", fontsize=9, va="bottom")
    ax.annotate(title, (0, 1), (11, 5), xycoords="axes fraction", textcoords="offset points", fontsize=kw.pop("fontsize", 8), va="bottom", **kw)


# ---------------------------------------------------------------- geography
def country_path():
    g = json.loads((ROOT / "configs/geography/geoBoundaries-BGD-ADM0.geojson").read_text())
    verts, codes = [], []
    for poly in g["features"][0]["geometry"]["coordinates"]:
        for ring in poly:
            r = np.asarray(ring)[:, :2]
            verts.extend(r.tolist())
            codes.extend([MPath.MOVETO] + [MPath.LINETO] * (len(r) - 2) + [MPath.CLOSEPOLY])
    return MPath(np.asarray(verts), codes)


BGD = country_path()
EXT = (87.9, 92.8, 20.5, 26.75)


def bd_map(ax, field=None, lat=None, lon=None, vmin=0, vmax=None, cmap=RAIN, lw=0.6, frame=False):
    """Draw a field clipped to Bangladesh; returns the mesh (or None)."""
    mesh = None
    if field is not None:
        mesh = ax.pcolormesh(lon, lat, field, cmap=cmap, vmin=vmin, vmax=vmax, shading="nearest",
                             rasterized=True)
        mesh.set_clip_path(PathPatch(BGD, transform=ax.transData))
    ax.add_patch(PathPatch(BGD, facecolor="none", edgecolor=C["ink"], lw=lw, zorder=5))
    ax.set_xlim(EXT[0], EXT[1]); ax.set_ylim(EXT[2], EXT[3])
    ax.set_aspect(1 / np.cos(np.deg2rad(23.7)))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(frame)
    return mesh


def block_mean(f, k):
    """Block-average on k x k cells, keeping the fine-grid shape (edge blocks may be partial)."""
    out = np.full_like(f, np.nan, dtype=float)
    for i in range(0, f.shape[0], k):
        for j in range(0, f.shape[1], k):
            blk = f[i:i + k, j:j + k]
            if np.isfinite(blk).any():
                out[i:i + k, j:j + k] = np.nanmean(blk)
    return out


def inside(lat, lon):
    return BGD.contains_points(np.column_stack([lon, lat]))


# ---------------------------------------------------------------- data
daily = pd.read_csv(BD / "daily_withheld_scores.csv")
wide = pd.read_csv(WD / "daily_withheld_scores.csv")
prod = pd.read_csv(BD / "gridded/long_term_withheld_product_matrix.csv").set_index("source")
wprod = pd.read_csv(WD / "gridded/long_term_withheld_product_matrix.csv").set_index("source")
stn = pd.read_csv(BD / "gridded/long_term_withheld_station_scores.csv")
temp = pd.read_csv(BD / "temporal_withheld_scores.csv")
audit = pd.read_csv(BD / "station_country_audit.csv")
audit = audit[audit.inside_bangladesh]
grids = np.load(WD / "gridded/temporal_mean_variability_grids.npz", allow_pickle=True)
LAT, LON = grids["lat"], grids["lon"]
ORDER = [A, "imerg", "cpc", "chirps", "background"]
numbers = {}


# ================================================================ Figure 1
def fig01():
    H = 78.5
    fig = plt.figure(figsize=(7.3, 7.3 * H / 100))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, H); ax.axis("off")
    blue, red = "#2A5B84", "#9A3B4B"

    def band(y0, y1, col, face, head, sub, sx=38.6):
        ax.add_patch(FancyBboxPatch((0.6, y0), 98.8, y1 - y0, boxstyle="round,pad=0,rounding_size=1.4",
                                    fc=face, ec=col, lw=0.8, ls=(0, (4, 3))))
        ax.add_patch(FancyBboxPatch((0.6, y1 - 2.3), 36, 4.6, boxstyle="round,pad=0,rounding_size=2.2",
                                    fc=col, ec="none", zorder=3))
        ax.text(2.6, y1, head, color="white", fontsize=8.6, fontweight="bold", va="center", zorder=4)
        ax.text(sx, y1 + 0.1, sub, color=col, fontsize=7.4, va="center", style="italic", zorder=4,
                bbox=dict(fc="white", ec="none", pad=1.5))

    band(43.5, 72.6, blue, "#F4F8FC", "1  TRAIN ONCE  ·  1981–2018", "learn a prior for 5 km rainfall given the large-scale state")

    band(1.0, 36.6, red, "#FCF6F6", "2  ANALYSE EVERY DAY", "assimilate observations while sampling · no retraining", sx=53.5)

    def card(x, y, w, h, ec, fc="white"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0", fc=fc, ec=ec, lw=0.8, zorder=2))

    def arrow(p, q, col=C["ink"], style="-|>", ls="-", rad=0.0, lw=1.2):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=9, lw=lw, color=col, ls=ls,
                                     connectionstyle=f"arc3,rad={rad}", zorder=6, shrinkA=0, shrinkB=0))

    def inset(x, y, w, h):
        return fig.add_axes([x / 100, y / H, w / 100, h / H])

    sm = {k: np.array(grids[f"seasonal_mean__{k}"], dtype=float) for k in ["cpc", "chirps", "imerg", A]}
    vmax = 26

    def mapcard(x, y, title, sub, field, ec, note=None, pts=None, w=17.5, h=22.5):
        card(x, y, w, h, ec)
        ax.text(x + 1.0, y + h - 1.9, title, fontsize=8, fontweight="bold", va="center", zorder=7)
        ax.text(x + 1.0, y + h - 4.3, sub, fontsize=6.5, color=C["muted"], va="center", zorder=7)
        a = inset(x + 1.2, y + (3.0 if note else 0.8), w - 2.4, h - (9.0 if note else 6.6))
        bd_map(a, field, LAT, LON, 0, vmax, lw=0.5)
        if pts is not None:
            for lat, lon, kw in pts:
                a.scatter(lon, lat, zorder=8, **kw)
        if note:
            ax.text(x + w / 2, y + 1.6, note, fontsize=6.3, color=C["muted"], ha="center", va="center", zorder=7)
        return a

    # ---- Phase 1
    mapcard(2.2, 46.2, "Coarse predictors", "CPC 0.5° rain + ERA5 state", block_mean(sm["cpc"], 10), blue, "≈55 km cells")
    ax.text(20.6, 66.6, "+ ERA5: TCWV, CAPE,\n   u10, v10, MSLP\n+ elevation, land mask\n+ day of year", fontsize=6.3,
            color=C["muted"], va="top", linespacing=1.35)
    # U-Net card
    card(36.0, 46.2, 27.5, 22.5, blue, "#EAF1F8")
    ax.text(37.0, 66.8, "Rectified-flow U-Net", fontsize=8, fontweight="bold", va="center", zorder=7)
    ax.text(37.0, 64.4, r"velocity $u_\theta(x_\tau,\tau,c)$", fontsize=6.8, color=C["muted"], va="center", zorder=7)
    xs = [39.0, 42.4, 45.8, 51.2, 54.6, 58.0]; hs = [8.4, 6.2, 4, 4, 6.2, 8.4]; yc = 56.0
    for x, hgt in zip(xs, hs):
        ax.add_patch(Rectangle((x, yc - hgt / 2), 2.4, hgt, fc="#86A9D6", ec=blue, lw=0.6, zorder=4))
    ax.add_patch(plt.Circle((49.75, yc), 1.25, fc="#7B5EA7", ec="none", zorder=4))
    for i in range(5):
        arrow((xs[i] + 2.4, yc), (xs[i + 1], yc) if i != 2 else (48.5, yc), blue, "-", lw=0.6)
    ax.plot([51.0, 51.2], [yc, yc], color=blue, lw=0.6, zorder=4)
    for (i, j), r in zip([(0, 5), (1, 4)], [-0.26, -0.3]):
        arrow((xs[i] + 1.2, yc + hs[i] / 2), (xs[j] + 1.2, yc + hs[j] / 2), blue, "-", (0, (2, 2)), r, 0.6)
    ax.text(49.75, 48.4, r"noise $x_0$ → rain residual $x_1$" "\n" r"$x_\tau=\tau x_1+(1-\tau)x_0$", fontsize=6.3,
            color=C["muted"], ha="center", va="center", linespacing=1.4, zorder=7)
    mapcard(66.5, 46.2, "Training target", "CHIRPS 0.05° daily rain", sm["chirps"], "#4E8B57", "≈5 km cells")
    arrow((31.5, 56.6), (36.0, 56.6), blue); ax.text(33.7, 58.0, "c", fontsize=7, color=blue, ha="center", style="italic")
    arrow((66.5, 56.6), (63.5, 56.6), "#4E8B57"); ax.text(65.0, 58.0, r"$x_1$", fontsize=7, color="#4E8B57", ha="center")
    # training note
    card(86.0, 46.2, 12.2, 22.5, "#C8862A", "#FFF8EC")
    ax.text(87.0, 67.2, "No observation\nenters training", fontsize=7.4, fontweight="bold", color="#8A5A12",
            va="top", linespacing=1.2, zorder=7)
    ax.text(87.0, 61.6, "Gauges and\nsatellite rain are\nused only in\nphase 2, so the\nobserving network\ncan change with\nno retraining.",
            fontsize=6.3, color=C["ink"], va="top", linespacing=1.38, zorder=7)

    # frozen weights link
    arrow((49.75, 46.2), (49.75, 31.6), C["ink"], ls=(0, (3, 2)), lw=1.0)
    ax.text(48.6, 40.6, "frozen weights θ", fontsize=6.8, color=C["ink"], va="center", ha="right", style="italic")

    # ---- Phase 2
    bmd = pd.read_csv(ROOT / "data/stations/data_2020_2025/Stations.csv")
    bw = pd.read_csv(ROOT / "output/bwdb_analysis/bwdb_stations_with_distance.csv")
    bmd = bmd[inside(bmd.Latitude.values, bmd.Longitude.values)]
    bw = bw[inside(bw.Latitude.values, bw.Longitude.values)]
    pts = [(bw.Latitude, bw.Longitude, dict(s=2.2, c=C["imerg"], lw=0, alpha=0.75)),
           (bmd.Latitude, bmd.Longitude, dict(s=7, c=C["accent"], lw=0.3, ec="white", marker="s"))]
    mapcard(2.2, 4.0, "Rain gauges", "     BMD       BWDB", None, red, "0.25° super-observations", pts=pts, w=15.3, h=27.6)
    ax.scatter([3.9], [27.3], s=9, c=C["accent"], marker="s", lw=0, zorder=8); ax.scatter([9.9], [27.3], s=6, c=C["imerg"], lw=0, zorder=8)
    mapcard(18.6, 4.0, "Satellite", "IMERG, 0.4° means", block_mean(sm["imerg"], 8), red, "area-mean operator", w=15.3, h=27.6)

    card(37.0, 4.0, 25.5, 27.6, "#C8862A", "#FFFBF3")
    ax.text(38.0, 29.7, "Guided sampling", fontsize=8, fontweight="bold", va="center", zorder=7)
    ax.text(38.0, 27.3, r"50 steps,  $\tau: 0 \rightarrow 1$,  30 members", fontsize=6.5, color=C["muted"], va="center", zorder=7)
    a = inset(38.6, 13.4, 22.3, 12.0)
    t = np.linspace(0, 1, 60)
    rng = np.random.default_rng(3)
    for k, col in enumerate([C["accent"], C["chirps"], C["surma"], C["imerg"], C["cpc"]]):
        y0 = 1.6 - 0.8 * k; y1 = 0.45 - 0.22 * k + rng.normal(0, 0.04)
        a.plot(t, y0 + (y1 - y0) * t ** 0.8, color=col, lw=1.0)
    for v in (0.25, 0.5, 0.75):
        a.axvline(v, color=red, lw=0.5, ls=(0, (2, 2)))
    a.set_xticks([]); a.set_yticks([]); a.set_xlim(0, 1)
    for s in a.spines.values():
        s.set_visible(True); s.set_color(C["grid"])
    ax.text(38.6, 12.2, "noise", fontsize=6, color=C["muted"], va="center"); ax.text(60.9, 12.2, "5 km rain", fontsize=6, color=C["muted"], va="center", ha="right")
    ax.text(49.75, 8.3, r"each step:  $dx = (u_\theta + $" + r"$\Delta u_{\rm obs}$" + r"$)\,d\tau$", fontsize=7.2, ha="center", va="center", zorder=7)
    ax.text(49.75, 5.6, "prior velocity + observation nudge", fontsize=6.2, color=C["muted"], ha="center", va="center", zorder=7)
    arrow((33.9, 17.8), (37.0, 17.8), red)
    ax.text(35.4, 19.4, "y", fontsize=7, color=red, ha="center", style="italic")

    # analysis stack
    for off in (1.4, 0.7):
        card(65.0 + off, 4.0 + off - 0.7, 16.2, 27.6, "#4E8B57", "#F1F7F2")
    mapcard(65.0, 3.3, "Analysis ensemble", "SURMA-Flow, 0.05° daily", sm[A], "#4E8B57", "30 members per day", w=16.2, h=27.6)
    arrow((62.5, 17.8), (65.0, 17.8), C["ink"])
    wb = audit[audit.station_id.str.startswith("BWDB")]; wm = audit[audit.station_id.str.startswith("BMD")]
    pts = [(wb.lat, wb.lon, dict(s=4.5, c=C["surma"], lw=0.25, ec="white")),
           (wm.lat, wm.lon, dict(s=9, c=C["accent"], lw=0.3, ec="white", marker="s"))]
    mapcard(84.9, 4.0, "Verification", "132 withheld gauges", None, C["surma"], "never assimilated", pts=pts, w=13.4, h=27.6)
    arrow((82.7, 17.8), (84.9, 17.8), C["ink"])

    cax = fig.add_axes([0.80, 76.6 / H, 0.17, 0.009])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, vmax), cmap=RAIN), cax=cax, orientation="horizontal")
    cb.set_ticks([0, 13, 26]); cb.ax.tick_params(labelsize=5.5, length=1.5, pad=1); cb.outline.set_linewidth(0.4)
    ax.text(79.0, 76.9, "Map shading: May–Sep mean rainfall, 2021 and 2023 (mm/day)", fontsize=6.0, color=C["muted"], ha="right", va="center")
    save(fig, "fig01_overview")


# ================================================================ Figure 2
def fig02():
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 3.9), gridspec_kw={"width_ratios": [1.5, 1, 1], "wspace": 0.1, "hspace": 0.85})
    src = stn.merge(audit[["station_id"]], on="station_id")
    src["net"] = np.where(src.station_id.str.startswith("BMD"), "BMD", "BWDB")

    def pooled(g, col):
        if col == "rmse_mm":
            return np.sqrt((g[col] ** 2 * g.n).sum() / g.n.sum())
        return (g[col] * g.n).sum() / g.n.sum()

    for r, (col, name) in enumerate([("rmse_mm", "RMSE"), ("mae_mm", "MAE")]):
        for c, net in enumerate(["All withheld gauges", "BMD", "BWDB"]):
            ax = axes[r, c]
            g0 = src if c == 0 else src[src.net == net]
            vals = {m: pooled(g0[g0.source == m], col) for m in ORDER}
            n = int(g0[g0.source == A].n.sum())
            y = np.arange(len(ORDER))[::-1]
            ax.barh(y, [vals[m] for m in ORDER], color=[C[KEY[m]] if m == A else "#C9CED6" for m in ORDER], height=0.66)
            for yi, m in zip(y, ORDER):
                ax.plot([vals[m]] * 2, [yi - 0.33, yi + 0.33], color=C[KEY[m]], lw=2.2, solid_capstyle="butt")
                txt = f"{vals[m]:.1f}"
                if m != A:
                    txt += f"   −{100 * (1 - vals[A] / vals[m]):.0f}%"
                ax.text(vals[m] + 0.4, yi, txt if c == 0 else f"{vals[m]:.1f}", va="center", fontsize=6.8,
                        color=C["ink"] if m == A else C["muted"], fontweight="bold" if m == A else "normal")
            ax.set_yticks(y); ax.set_yticklabels([LAB[m] for m in ORDER] if c == 0 else [])
            ax.tick_params(axis="y", length=0)
            if c == 0:
                ax.get_yticklabels()[0].set_fontweight("bold"); ax.get_yticklabels()[0].set_color(C["surma"])
            ax.set_xlim(0, (27 if r == 0 else 17) * (1.3 if c == 0 else 1.15))
            ax.xaxis.grid(True, color=C["grid"], lw=0.5); ax.set_axisbelow(True)
            ax.spines["left"].set_visible(False)
            ax.set_xlabel(f"Daily {name} (mm/day)")
            ttl(ax, "abcdef"[r * 3 + c], f"{net} · n = {n:,}")
            numbers[f"fig02_{name}_{net}"] = {m: round(float(vals[m]), 3) for m in ORDER}
    save(fig, "fig02_products")


# ================================================================ Figure 3
def fig03():
    piv = stn.pivot(index="station_id", columns="source", values="rmse_mm").join(audit.set_index("station_id")[["lat", "lon"]])
    fig = plt.figure(figsize=(7.2, 3.15))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.02], wspace=0.04, left=0.005, right=0.905, top=0.86, bottom=0.14)
    ax = fig.add_subplot(gs[0, 0]); bd_map(ax)
    sc0 = ax.scatter(piv.lon, piv.lat, c=piv[A], s=13, cmap="YlGnBu", vmin=5, vmax=30, ec=C["ink"], lw=0.25, zorder=6)
    ttl(ax, "a", "SURMA-Flow daily RMSE")
    cax = fig.add_axes([0.035, 0.09, 0.17, 0.022]); cb = fig.colorbar(sc0, cax=cax, orientation="horizontal", extend="both")
    cb.set_label("mm/day", fontsize=7, labelpad=1); cb.ax.tick_params(labelsize=6.5, length=2); cb.outline.set_linewidth(0.4)
    win = {}
    for i, o in enumerate(["imerg", "cpc", "chirps"]):
        ax = fig.add_subplot(gs[0, i + 1]); bd_map(ax)
        d = piv[A] - piv[o]
        sc = ax.scatter(piv.lon, piv.lat, c=d, s=13, cmap="BrBG_r", vmin=-12, vmax=12, ec=C["ink"], lw=0.25, zorder=6)
        win[o] = int((d < 0).sum())
        ttl(ax, "bcd"[i], f"minus {LAB[o]}")
        ax.text(0.5, -0.04, f"better at {win[o]} of {len(d)} gauges", transform=ax.transAxes, ha="center", va="top", fontsize=7.4, color=C["ink"])
    cax = fig.add_axes([0.945, 0.27, 0.012, 0.46]); cb = fig.colorbar(sc, cax=cax, extend="both")
    cb.set_label("RMSE difference (mm/day)", fontsize=7); cb.ax.tick_params(labelsize=6.5, length=2); cb.outline.set_linewidth(0.4)
    cb.ax.text(0.5, 1.09, "product\nbetter", transform=cb.ax.transAxes, fontsize=6.2, color="#8C5A1A", va="bottom", ha="center", linespacing=1.1)
    cb.ax.text(0.5, -0.09, "SURMA-Flow\nbetter", transform=cb.ax.transAxes, fontsize=6.2, color=C["surma"], va="top", ha="center", linespacing=1.1)
    numbers["fig03_station_wins_rmse"] = win | {"n": int(len(piv)), "lowest_of_all": piv[list(KEY)].idxmin(axis=1).value_counts().to_dict()}
    save(fig, "fig03_station_maps")


# ================================================================ Figure 4
def fig04():
    mo = temp[temp.scale == "monthly"].copy()
    g = mo.groupby(["period", "method"]).mean_error_mm_day.apply(lambda e: np.sqrt((e ** 2).mean())).unstack()
    fig = plt.figure(figsize=(7.2, 2.75))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.9, 1, 1], wspace=0.36, left=0.065, right=0.995, top=0.86, bottom=0.2)
    ax = fig.add_subplot(gs[0, 0])
    months = pd.period_range("2021-05", "2024-09", freq="M")
    wet = [m for m in months if m.month in (5, 6, 7, 8, 9)]
    x = {str(m): i + 0.9 * (m.year - 2021) for i, m in enumerate(wet)}
    for m, k in [("background", "background"), (A, "surma")]:
        for yr in (2021, 2022, 2023, 2024):
            keys = [p for p in g.index if p.startswith(str(yr))]
            ax.plot([x[p] for p in keys], g.loc[keys, m], "-o", color=C[k], ms=3.2, lw=1.3, label=LAB[m] if yr == 2021 else None)
    ax.scatter([x["2022-05"]], [0.6], marker="x", s=18, color=C["muted"], lw=0.9, zorder=5)
    ax.text(x["2022-05"] + 0.35, 0.65, "May 2022 (tuning month) excluded", fontsize=6.2, color=C["muted"], va="center")
    for p in ("2024-07", "2024-08", "2024-09"):
        ax.add_patch(Rectangle((x[p] - 0.5, 0), 1.0, 11.6, fc="#F3F4F6", ec="none", zorder=0))
    ax.text(x["2024-08"], 6.3, "to be\ngenerated", fontsize=6.2, color=C["muted"], ha="center", va="center")
    ticks = [p for p in x if p[-2:] in ("05", "07", "09")]
    ax.set_xticks([x[p] for p in ticks]); ax.set_xticklabels([{"05": "May", "07": "Jul", "09": "Sep"}[p[-2:]] for p in ticks], fontsize=6.6)
    for yr in (2021, 2022, 2023, 2024):
        ax.text(x[f"{yr}-07"], -1.75, str(yr), ha="center", va="top", fontsize=7.5, fontweight="bold", color=C["ink"])
    ax.set_ylim(0, 12.5); ax.set_ylabel("RMSE of monthly mean (mm/day)")
    ax.yaxis.grid(True, color=C["grid"], lw=0.5); ax.set_axisbelow(True)
    ax.set_ylim(0, 13.6); ax.legend(frameon=False, loc="upper left", ncol=2, bbox_to_anchor=(0.0, 1.03), handlelength=1.6, columnspacing=1.0)
    ttl(ax, "a", "Every scored month, 2021–2024")
    numbers["fig04_monthly_rmse"] = {p: {m: round(float(g.loc[p, m]), 3) for m in g.columns} for p in g.index}

    # daily scores by year (2021, 2022 exact; 2023-24 = Bangladesh pooled minus 2021 and 2022)
    yr = {}
    for m in ["background", A]:
        p = daily[(daily.group == "pooled") & (daily.method == m)].iloc[0]
        a = wide[(wide.group == "period") & (wide.method == m)].set_index("label")
        k = ["2021_may_sep", "2022_may_sep"]; rest = p.n - a.loc[k, "n"].sum()
        crps = [a.loc[k[0], "crps"], a.loc[k[1], "crps"], (p.crps * p.n - (a.loc[k, "crps"] * a.loc[k, "n"]).sum()) / rest, p.crps]
        rmse = [a.loc[k[0], "rmse"], a.loc[k[1], "rmse"], np.sqrt((p.rmse ** 2 * p.n - (a.loc[k, "rmse"] ** 2 * a.loc[k, "n"]).sum()) / rest), p.rmse]
        cov = [a.loc[k[0], "coverage_90"], a.loc[k[1], "coverage_90"], (p.coverage_90 * p.n - (a.loc[k, "coverage_90"] * a.loc[k, "n"]).sum()) / rest, p.coverage_90]
        mae = [a.loc[k[0], "mae"], a.loc[k[1], "mae"], (p.mae * p.n - (a.loc[k, "mae"] * a.loc[k, "n"]).sum()) / rest, p.mae]
        yr[m] = dict(crps=crps, rmse=rmse, cov=cov, mae=mae, n=[int(a.loc[k[0], "n"]), int(a.loc[k[1], "n"]), int(rest), int(p.n)])
    numbers["fig04_by_year"] = {m: {k: [round(float(v), 3) for v in vals] for k, vals in d.items()} for m, d in yr.items()}
    labs = ["2021", "2022", "2023–\n2024", "All"]
    for j, (key, name) in enumerate([("crps", "Fair CRPS"), ("rmse", "RMSE")]):
        ax = fig.add_subplot(gs[0, j + 1]); xx = np.arange(4)
        ax.bar(xx - 0.19, yr["background"][key], 0.36, color=C["background"])
        ax.bar(xx + 0.19, yr[A][key], 0.36, color=C["surma"])
        for i in range(4):
            red = 100 * (yr[A][key][i] / yr["background"][key][i] - 1)
            ax.text(xx[i] + 0.19, yr[A][key][i] + (0.15 if key == "crps" else 0.35), f"−{abs(red):.0f}%", ha="center", fontsize=6.2, color=C["surma"], fontweight="bold")
        ax.axvline(2.5, color=C["grid"], lw=0.8)
        ax.set_xticks(xx); ax.set_xticklabels(labs, fontsize=7); ax.set_ylabel(f"Daily {name} (mm/day)")
        ax.yaxis.grid(True, color=C["grid"], lw=0.5); ax.set_axisbelow(True)
        ttl(ax, "bc"[j], f"Daily {name.replace('Fair ', '')} by year"); ax.set_ylabel(f"{name} (mm/day)"); ax.set_ylim(0, max(yr["background"][key]) * 1.08)
    save(fig, "fig04_full_period")


# ================================================================ Figure 5
def fig05():
    fig = plt.figure(figsize=(7.2, 2.85))
    gs = fig.add_gridspec(1, 6, width_ratios=[1, 1, 1, 1, 1, 0.05], wspace=0.03, left=0.005, right=0.94, top=0.88, bottom=0.12)
    order = [A, "imerg", "cpc", "chirps", "background"]
    gm = {}
    for i, k in enumerate(order):
        ax = fig.add_subplot(gs[0, i])
        f = np.array(grids[f"seasonal_mean__{k}"], dtype=float)
        if k == "cpc":
            f[f <= 0] = np.nan
        m = bd_map(ax, f, LAT, LON, 0, 30)
        lon2, lat2 = np.meshgrid(LON, LAT); ins = inside(lat2.ravel(), lon2.ravel()).reshape(f.shape)
        w = np.cos(np.deg2rad(lat2))
        ok = ins & np.isfinite(f); gm[k] = float((f[ok] * w[ok]).sum() / w[ok].sum())
        ttl(ax, "abcde"[i], {A: "SURMA-Flow", "imerg": "IMERG (0.4°)", "cpc": "CPC", "chirps": "CHIRPS", "background": "Background"}[k], color=C[KEY[k]], fontweight="bold")
        ax.text(0.5, -0.03, f"country mean {gm[k]:.1f}", transform=ax.transAxes, ha="center", va="top", fontsize=7.2)
    cax = fig.add_axes([0.95, 0.2, 0.012, 0.6]); cb = fig.colorbar(m, cax=cax, extend="max")
    cb.set_label("May–Sep mean rainfall (mm/day)", fontsize=7); cb.ax.tick_params(labelsize=6.5, length=2); cb.outline.set_linewidth(0.4)
    numbers["fig05_country_mean_mm_day"] = {k: round(v, 2) for k, v in gm.items()}
    save(fig, "fig05_seasonal_maps")


# ================================================================ Figure 6
def fig06():
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.2), gridspec_kw={"wspace": 0.5, "left": 0.06, "right": 0.995, "top": 0.84, "bottom": 0.17})
    # a: long-term station mean error by product
    ax = axes[0]; vals = {}
    for m in ORDER:
        g = stn[stn.source == m]; e = g.long_term_predicted_mm - g.long_term_observed_mm
        vals[m] = (float(np.sqrt((e ** 2).mean())), float(e.mean()), float(np.corrcoef(g.long_term_predicted_mm, g.long_term_observed_mm)[0, 1]))
    y = np.arange(5)[::-1]
    ax.barh(y, [vals[m][0] for m in ORDER], color=[C[KEY[m]] for m in ORDER], height=0.62)
    for yi, m in zip(y, ORDER):
        ax.text(vals[m][0] + 0.15, yi, f"{vals[m][0]:.1f}", va="center", fontsize=6.8)
    ax.set_yticks(y); ax.set_yticklabels(["SURMA-Flow", "IMERG", "CPC", "CHIRPS", "Background"], fontsize=6.8); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 10); ax.set_xlabel("RMSE (mm/day)"); ax.spines["left"].set_visible(False)
    ttl(ax, "a", "Multi-year mean"); ax.set_box_aspect(1)
    numbers["fig06_long_term_station_mean"] = {m: {"rmse": round(v[0], 3), "bias": round(v[1], 3), "r": round(v[2], 3)} for m, v in vals.items()}
    out = {}
    for ax, sc, lim, name, let in [(axes[1], "monthly", 65, "Monthly means", "b"), (axes[2], "may_sep", 38, "Seasonal means", "c")]:
        for m, k in [("background", "background"), (A, "surma")]:
            g = temp[(temp.scale == sc) & (temp.method == m)]
            ax.scatter(g.observed_mean_mm_day, g.predicted_mean_mm_day, s=4 if sc == "monthly" else 8, color=C[k], alpha=0.55, lw=0, rasterized=True)
            out[f"{sc}_{m}"] = {"n": int(len(g)), "rmse": round(float(np.sqrt((g.mean_error_mm_day ** 2).mean())), 3),
                                "mae": round(float(g.mean_error_mm_day.abs().mean()), 3), "bias": round(float(g.mean_error_mm_day.mean()), 3),
                                "r": round(float(np.corrcoef(g.observed_mean_mm_day, g.predicted_mean_mm_day)[0, 1]), 3)}
        ax.plot([0, lim], [0, lim], color=C["ink"], lw=0.6, ls="--"); ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
        ax.set_xlabel("Gauge (mm/day)"); ax.set_ylabel("Estimate (mm/day)", labelpad=1)
        ttl(ax, let, name)
    ax = axes[3]
    for m, k in [("background", "background"), (A, "surma")]:
        g = temp[(temp.scale == "monthly") & (temp.method == m)]
        ax.scatter(g.observed_daily_temporal_sd_mm, g.predicted_daily_temporal_sd_mm, s=4, color=C[k], alpha=0.55, lw=0, rasterized=True,
                   label="SURMA-Flow" if m == A else "Background")
        out[f"sd_ratio_{m}"] = round(float(g.predicted_daily_temporal_sd_mm.mean() / g.observed_daily_temporal_sd_mm.mean()), 3)
    ax.plot([0, 80], [0, 80], color=C["ink"], lw=0.6, ls="--"); ax.set_xlim(0, 80); ax.set_ylim(0, 80); ax.set_aspect("equal")
    ax.set_xlabel("Gauge (mm/day)"); ax.set_ylabel("Ensemble mean (mm/day)", labelpad=1)
    ttl(ax, "d", "Day-to-day SD")
    ax.legend(frameon=False, loc="upper left", fontsize=6.4, handletextpad=0.1, borderaxespad=0.1, markerscale=2)
    numbers["fig06_temporal"] = out
    save(fig, "fig06_time_scales")


# ================================================================ Figure 7
def fig07():
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.25), gridspec_kw={"wspace": 0.45, "left": 0.06, "right": 0.995, "top": 0.84, "bottom": 0.2})
    grp = [("pooled", "all", "All"), ("source", "BMD", "BMD"), ("source", "BWDB", "BWDB")]
    get = lambda g, l, m, c: float(daily[(daily.group == g) & (daily.label == l) & (daily.method == m)][c].iloc[0])
    x = np.arange(3)
    for ax, col, name, ref, let in [(axes[0], "crps", "Fair CRPS (mm/day)", None, "a"), (axes[1], "coverage_90", "90% coverage", 0.9, "b"),
                                   (axes[2], "spread_skill", "Spread / RMSE", 1.0, "c")]:
        b = [get(g, l, "background", col) for g, l, _ in grp]; s = [get(g, l, A, col) for g, l, _ in grp]
        ax.bar(x - 0.19, b, 0.36, color=C["background"], label="Background"); ax.bar(x + 0.19, s, 0.36, color=C["surma"], label="SURMA-Flow")
        if ref:
            ax.axhline(ref, color=C["accent"], lw=0.9, ls="--"); ax.text(2.48, ref + 0.015, "ideal", color=C["accent"], fontsize=6.4, ha="right", va="bottom")
            ax.set_ylim(0, 1.12)
        ax.set_xticks(x); ax.set_xticklabels([g[2] for g in grp]); ax.yaxis.grid(True, color=C["grid"], lw=0.5); ax.set_axisbelow(True)
        ttl(ax, let, name)
    axes[0].legend(frameon=False, fontsize=6.4, loc="upper left", bbox_to_anchor=(-0.02, 1.02), handlelength=1.0, ncol=1); axes[0].set_ylim(0, 12.2)
    # d: intensity dependence (archive-domain export, 133 sites)
    ax = axes[3]; it = wide[wide.group == "intensity"]
    bins = ["[0,1)", "[1,10)", "[10,25)", "[25,50)", "[50,100)", "[100,inf)"]; xl = ["<1", "1–10", "10–25", "25–50", "50–100", "≥100"]
    for m, k in [("background", "background"), (A, "surma")]:
        v = [float(it[(it.label == b) & (it.method == m)].bias.iloc[0]) for b in bins]
        ax.plot(range(6), v, "-o", color=C[k], ms=3, lw=1.3)
    ax.axhline(0, color=C["ink"], lw=0.6); ax.set_xticks(range(6)); ax.set_xticklabels(xl, fontsize=6, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_xlabel("Observed rain (mm/day)", labelpad=1); ax.yaxis.grid(True, color=C["grid"], lw=0.5)
    ttl(ax, "d", "Bias (mm/day)")
    numbers["fig07_intensity_archive_domain"] = {b: {m: {c: round(float(it[(it.label == b) & (it.method == m)][c].iloc[0]), 3) for c in ["n", "crps", "mae", "bias", "rmse", "coverage_90"]}
                                                    for m in ["background", A]} for b in bins}
    save(fig, "fig07_calibration_intensity")


# ================================================================ Figure 8
def fig08():
    f = pd.read_csv(BD / "gridded/data/fig05_subgrid_case_fields.csv")
    names = [(A, A), ("background", "background"), ("IMERG S04 (assimilated)", "imerg"), ("CPC (conditioning)", "cpc"), ("CHIRPS (reference)", "chirps")]
    lat = np.sort(f.lat.unique()); lon = np.sort(f.lon.unique())
    fig = plt.figure(figsize=(7.2, 2.8))
    gs = fig.add_gridspec(1, 5, wspace=0.03, left=0.005, right=0.94, top=0.88, bottom=0.05)
    for i, (src, k) in enumerate(names):
        g = f[f.source == src].pivot(index="lat", columns="lon", values="full_mm").reindex(index=lat, columns=lon).values
        ax = fig.add_subplot(gs[0, i]); m = bd_map(ax, g, lat, lon, 0, 120)
        name = {A: "SURMA-Flow", "background": "Background", "imerg": "IMERG (0.4°)", "cpc": "CPC (conditioning)", "chirps": "CHIRPS"}[k]
        ttl(ax, "abcde"[i], name, color=C[KEY[k]], fontweight="bold")
    cax = fig.add_axes([0.95, 0.2, 0.012, 0.55]); cb = fig.colorbar(m, cax=cax, extend="max")
    cb.set_label("Rainfall, 29 May 2024 (mm/day)", fontsize=7); cb.ax.tick_params(labelsize=6.5, length=2); cb.outline.set_linewidth(0.4)
    save(fig, "fig08_case")


if __name__ == "__main__":
    for f in (fig01, fig02, fig03, fig04, fig05, fig06, fig07, fig08):
        f()
    # headline numbers used in the text
    p = prod; s = p.loc[A]
    numbers["headline"] = {
        "n": int(s.pooled_daily_n),
        "products": {k: {"rmse": round(float(p.loc[k, "pooled_daily_rmse_mm"]), 3), "mae": round(float(p.loc[k, "pooled_daily_mae_mm"]), 3),
                         "bias": round(float(p.loc[k, "pooled_daily_bias_mm"]), 3),
                         "rmse_reduction_pct": round(100 * (1 - float(s.pooled_daily_rmse_mm) / float(p.loc[k, "pooled_daily_rmse_mm"])), 1),
                         "mae_reduction_pct": round(100 * (1 - float(s.pooled_daily_mae_mm) / float(p.loc[k, "pooled_daily_mae_mm"])), 1)} for k in p.index},
        "archive_domain_pooled_r": {k: round(float(wprod.loc[k, "pooled_daily_correlation"]), 3) for k in wprod.index},
        "median_station_r": {k: round(float(v), 3) for k, v in stn.pivot(index="station_id", columns="source", values="correlation").median().items()},
    }
    (OUT / "restructured_numbers.json").write_text(json.dumps(numbers, indent=1, default=str))
    print(json.dumps(numbers["headline"], indent=1))
