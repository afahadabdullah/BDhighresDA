#!/usr/bin/env python
"""Assemble every OSSE arm into the complete paper evidence set.

This is the single entry point between "the experiments have run" and "the
manuscript can be written".  It reads one directory per arm -- each containing
``osse_report.json`` from ``10_osse.py`` and ``downscaling.json`` /
``downscaling_curves.npz`` from ``22_evaluate_osse_downscaling.py`` -- and emits
tables, figures, tidy data and a written summary that are consistent with one
another by construction.

Emitted artifacts
-----------------
  osse_paper_summary.json     every number, one machine-readable file
  osse_paper_metrics.csv      tidy long format: arm, claim, metric, value
  osse_paper_curves.npz       spectra / FSS / ladders, all arms, for replotting
  table_ablation.tex          the ablation matrix, booktabs
  table_downscaling.tex       the downscaling-claim table
  fig_ablation_matrix.png     arm x metric heatmap
  fig_claims.png              claim A and B skill across arms
  fig_spectra.png             spectra, ratio and effective resolution
  fig_scale_ladder.png        error by aggregation scale, and FSS skillful scale
  RESULTS.md                  narrative summary keyed to the tables

Design rule
-----------
Nothing is computed here that was not already computed by script 22.  This
script only selects, arranges and renders, so a number in the manuscript can
always be traced to exactly one upstream JSON.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

# Arms are rendered in this order when present.  Order encodes the argument of
# the results section: establish capacity, then degrade toward realism.
PREFERRED_ORDER = [
    "gauges_exact_bmd",
    "satellite_exact_bmd",
    "simultaneous_exact_bmd",
    "simultaneous_perfect_40",
    "gauges_realistic_40",
    "satellite_realistic_40",
    "simultaneous_realistic_40",
    "simultaneous_dense_uncorrected_40",
    "simultaneous_realistic_20",
    "simultaneous_realistic_80",
]

PRETTY = {
    "gauges_exact_bmd": "BMD pseudo-gauges only",
    "satellite_exact_bmd": "Exact 0.1° footprints only",
    "simultaneous_exact_bmd": "Simultaneous (primary)",
    "simultaneous_perfect_40": "Perfect obs (upper bound)",
    "gauges_realistic_40": "Gauges only",
    "satellite_realistic_40": "Pseudo-IMERG only",
    "simultaneous_realistic_40": "Simultaneous (primary)",
    "simultaneous_dense_uncorrected_40": "Dense, no thinning",
    "simultaneous_realistic_20": "Simultaneous, 20 gauges",
    "simultaneous_realistic_80": "Simultaneous, 80 gauges",
}

# (json path, column label, objective)
#
# The objective is "max", "min", or a float TARGET.  Calibration statistics are
# targets, not maxima: a spread/skill ratio of 1.4 is over-dispersive and worse
# than 1.0, and a member energy ratio of 1.5 means members are too rough.
# Treating either as higher-is-better would rank the least calibrated arm first.
ABLATION_COLUMNS = [
    ("withheld_background.crps_mm", "CRPS bg", "min"),
    ("withheld_analysis.crps_mm", "CRPS an", "min"),
    ("withheld_improvement_crps_mm", "CRPS gain %", "max"),
    ("withheld_analysis.spread_skill", "Spread/skill", 1.0),
    ("withheld_analysis.coverage_90", "Cover90", 0.90),
    ("field_improvement_crps_mm", "Field gain %", "max"),
]

DOWNSCALING_COLUMNS = [
    ("claim_a_downscaling_gain.background.mse_skill", "A: MSE skill", "max"),
    ("claim_a_downscaling_gain.background.crps_skill", "A: CRPS skill", "max"),
    ("claim_b_sub_footprint_gain.background.mse_skill", "B: bg MSE skill", "max"),
    ("claim_b_sub_footprint_gain.analysis.mse_skill", "B: an MSE skill", "max"),
    ("claim_b_sub_footprint_gain.analysis.crps_skill", "B: an CRPS skill", "max"),
    ("claim_b_sub_footprint_gain.analysis.correlation", "B: an corr", "max"),
    ("claim_b_sub_footprint_gain.analysis.mean_member_energy_ratio",
     "B: member energy", 1.0),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", default="data/processed/osse_paper",
                   help="Directory holding one subdirectory per arm")
    p.add_argument("--arm", action="append", default=None, metavar="DIR_OR_LABEL=DIR",
                   help="Explicit arm; repeat. Default: autodiscover under --root")
    p.add_argument("--out-dir", default=None,
                   help="Where artifacts are written (default: --root/paper)")
    p.add_argument("--primary", default="simultaneous_realistic_40",
                   help="Arm treated as the headline configuration")
    p.add_argument("--skip-figures", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def dig(payload: dict, dotted: str, default=np.nan):
    """Fetch a dotted path from nested dicts without raising."""
    node = payload
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node if node is not None else default


def discover_arms(root: Path, explicit: list[str] | None) -> list[tuple[str, Path]]:
    if explicit:
        arms = []
        for item in explicit:
            label, _, path = item.partition("=")
            arms.append((label, Path(path or label)))
        return arms
    if not root.is_dir():
        raise SystemExit(f"no such directory: {root}")
    found = {d.name: d for d in sorted(root.iterdir())
             if d.is_dir() and (d / "downscaling.json").exists()}
    ordered = [(name, found.pop(name)) for name in PREFERRED_ORDER if name in found]
    ordered += sorted(found.items())
    if not ordered:
        raise SystemExit(
            f"{root} contains no arm directory with downscaling.json -- run "
            f"slurm/submit_osse_paper.sh first"
        )
    return ordered


def load_arm(label: str, directory: Path) -> dict:
    """One arm: DA scores, downscaling claims, and its plotting curves."""
    downscaling = json.loads((directory / "downscaling.json").read_text())

    osse: dict = {}
    report_path = directory / "osse_report.json"
    if report_path.exists():
        payload = json.loads(report_path.read_text())
        results = payload.get("results", [])
        # 10_osse.py may sweep several networks; take the row matching the dump
        # so the DA scores and the downscaling claims describe the same run.
        wanted = str(downscaling.get("network", ""))
        matched = [r for r in results if str(r.get("network", "")) == wanted]
        osse = (matched or results or [{}])[-1]

    curves: dict[str, np.ndarray] = {}
    curve_path = directory / "downscaling_curves.npz"
    if curve_path.exists():
        with np.load(curve_path) as data:
            curves = {k: data[k] for k in data.files}

    return {"label": label, "pretty": PRETTY.get(label, label.replace("_", " ")),
            "directory": str(directory), "osse": osse, "downscaling": downscaling,
            "curves": curves}


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def _goodness(values: np.ndarray, objective) -> np.ndarray:
    """Map raw values onto a higher-is-better axis for the given objective."""
    values = np.asarray(values, dtype=float)
    if objective == "max":
        return values
    if objective == "min":
        return -values
    return -np.abs(values - float(objective))   # numeric target


def _best_row(values: np.ndarray, objective) -> int:
    goodness = _goodness(values, objective)
    if np.all(np.isnan(goodness)):
        return -1
    return int(np.nanargmax(goodness))


def build_matrix(arms: list[dict], columns, source: str) -> np.ndarray:
    matrix = np.full((len(arms), len(columns)), np.nan)
    for row, arm in enumerate(arms):
        for col, (path, _, _) in enumerate(columns):
            try:
                matrix[row, col] = float(dig(arm[source], path))
            except (TypeError, ValueError):
                matrix[row, col] = np.nan
    return matrix


def write_latex(path: Path, arms: list[dict], columns, matrix: np.ndarray,
                caption: str, label: str) -> None:
    """Booktabs table with the best entry per column bolded."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table*}[t]", r"\centering",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\small",
        r"\begin{tabular}{l" + "r" * len(columns) + "}", r"\toprule",
        "Arm & " + " & ".join(c[1] for c in columns) + r" \\", r"\midrule",
    ]
    best = [_best_row(matrix[:, col], objective)
            for col, (_, _, objective) in enumerate(columns)]
    for row, arm in enumerate(arms):
        cells = []
        for col in range(len(columns)):
            value = matrix[row, col]
            text = "--" if not np.isfinite(value) else f"{value:.3f}"
            if row == best[col] and np.isfinite(value):
                text = rf"\textbf{{{text}}}"
            cells.append(text)
        lines.append(f"{arm['pretty']} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    path.write_text("\n".join(lines))


def write_tidy_csv(path: Path, arms: list[dict]) -> None:
    """Long format: one row per (arm, claim, metric) so any plot is a groupby."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for arm in arms:
        for source, columns in (("osse", ABLATION_COLUMNS),
                                ("downscaling", DOWNSCALING_COLUMNS)):
            for dotted, pretty, _ in columns:
                value = dig(arm[source], dotted)
                rows.append({"arm": arm["label"], "arm_label": arm["pretty"],
                             "source": source, "metric_path": dotted,
                             "metric": pretty,
                             "value": "" if value is None or (
                                 isinstance(value, float) and not np.isfinite(value)
                             ) else value})
        # stratifications, so intensity and year plots need no extra parsing
        for band, payload in dig(arm["downscaling"], "by_intensity", {}).items():
            for member in ("background", "analysis"):
                rows.append({"arm": arm["label"], "arm_label": arm["pretty"],
                             "source": "by_intensity",
                             "metric_path": f"{band}.{member}.mse_skill",
                             "metric": f"subgrid MSE skill ({member})",
                             "value": dig(payload, f"{member}.mse_skill")})
        for year, payload in dig(arm["downscaling"], "by_year", {}).items():
            for member in ("background", "analysis"):
                rows.append({"arm": arm["label"], "arm_label": arm["pretty"],
                             "source": "by_year",
                             "metric_path": f"{year}.{member}.mse_skill",
                             "metric": f"subgrid MSE skill ({member})",
                             "value": dig(payload, f"{member}.mse_skill")})
        for name, value in dig(arm["downscaling"],
                               "structure.spectra.effective_resolution_km", {}).items():
            rows.append({"arm": arm["label"], "arm_label": arm["pretty"],
                         "source": "spectra", "metric_path": f"effective_resolution.{name}",
                         "metric": f"effective resolution km ({name})", "value": value})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["arm", "arm_label", "source", "metric_path",
                                "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def write_combined_curves(path: Path, arms: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for arm in arms:
        for key, value in arm["curves"].items():
            payload[f"{arm['label']}::{key}"] = value
    if payload:
        np.savez_compressed(path, **payload)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _normalise(matrix: np.ndarray, columns) -> np.ndarray:
    """Per-column 0-1 shading with the good end always at 1."""
    shaded = np.full_like(matrix, np.nan)
    for col, (_, _, objective) in enumerate(columns):
        values = _goodness(matrix[:, col], objective)
        finite = values[np.isfinite(values)]
        if finite.size < 2 or np.ptp(finite) == 0:
            shaded[:, col] = 0.5
            continue
        shaded[:, col] = (values - finite.min()) / np.ptp(finite)
    return shaded


def figure_matrix(path: Path, arms, columns, matrix, title) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(1.55 * len(columns) + 4, 0.62 * len(arms) + 2.4),
                           constrained_layout=True)
    ax.imshow(_normalise(matrix, columns), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels([c[1] for c in columns], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels([a["pretty"] for a in arms], fontsize=8)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            ax.text(col, row, "--" if not np.isfinite(value) else f"{value:.3f}",
                    ha="center", va="center", fontsize=7.5)
    ax.set_title(title, fontsize=10)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def figure_claims(path: Path, arms: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    y = np.arange(len(arms))
    labels = [a["pretty"] for a in arms]

    a_values = [dig(a["downscaling"],
                    "claim_a_downscaling_gain.background.mse_skill") for a in arms]
    axes[0].barh(y, a_values, color="#2F5C8F")
    axes[0].axvline(0, color="k", lw=1)
    axes[0].set(yticks=y, yticklabels=labels, xlabel="MSE skill vs coarse input",
                title="A.  Downscaling gain (background only)")
    axes[0].invert_yaxis(); axes[0].grid(alpha=0.3, axis="x")

    bg = [dig(a["downscaling"], "claim_b_sub_footprint_gain.background.mse_skill")
          for a in arms]
    an = [dig(a["downscaling"], "claim_b_sub_footprint_gain.analysis.mse_skill")
          for a in arms]
    axes[1].barh(y - 0.2, bg, 0.4, label="background", color="#0F7A6B")
    axes[1].barh(y + 0.2, an, 0.4, label="analysis", color="#B0512E")
    axes[1].axvline(0, color="k", lw=1)
    axes[1].set(yticks=y, yticklabels=labels,
                xlabel="MSE skill vs perfect-footprint null",
                title="B.  Sub-footprint gain")
    axes[1].invert_yaxis(); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3, axis="x")

    fig.suptitle("Claim A isolates the prior; claim B measures structure below "
                 "the satellite footprint", fontsize=10)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def figure_spectra(path: Path, arms: list[dict], primary: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    head = next((a for a in arms if a["label"] == primary), arms[0])
    curves = head["curves"]

    wl = curves.get("spectra_wavelength_km")
    if wl is not None and wl.size:
        axes[0].loglog(wl, curves["spectra_truth_power"], "k-", lw=2, label="CHIRPS truth")
        for name in ("coarse_input", "background_member", "analysis_member"):
            key = f"spectra_power__{name}"
            if key in curves:
                axes[0].loglog(wl, curves[key], lw=1.4, label=name.replace("_", " "))
        axes[0].invert_xaxis()
        axes[0].set(xlabel="wavelength (km)", ylabel="power",
                    title=f"A.  Spectra -- {head['pretty']}")
        axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3, which="both")

        axes[1].axhline(1, color="k", lw=1)
        axes[1].axhspan(0.5, 2.0, color="grey", alpha=0.15)
        for name in ("coarse_input", "background_member", "analysis_member"):
            key = f"spectra_ratio__{name}"
            if key in curves:
                axes[1].semilogx(wl, curves[key], lw=1.4, label=name.replace("_", " "))
        axes[1].invert_xaxis(); axes[1].set_yscale("log")
        axes[1].set(xlabel="wavelength (km)", ylabel="model / truth power",
                    title="B.  Spectral ratio (band = factor of two)")
        axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3, which="both")

    y = np.arange(len(arms))
    for offset, name, colour in ((-0.2, "background_member", "#0F7A6B"),
                                 (0.2, "analysis_member", "#B0512E")):
        values = [dig(a["downscaling"],
                      f"structure.spectra.effective_resolution_km.{name}")
                  for a in arms]
        axes[2].barh(y + offset, values, 0.4, label=name.replace("_", " "), color=colour)
    axes[2].set(yticks=y, yticklabels=[a["pretty"] for a in arms],
                xlabel="effective resolution (km; smaller is better)",
                title="C.  Wavelength down to which power is right")
    axes[2].invert_yaxis(); axes[2].legend(fontsize=7); axes[2].grid(alpha=0.3, axis="x")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def figure_scale_ladder(path: Path, arms: list[dict], primary: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    head = next((a for a in arms if a["label"] == primary), arms[0])
    curves = head["curves"]

    degrees = curves.get("ladder_degrees")
    if degrees is not None and degrees.size > 1:
        for name in ("coarse_input", "background", "analysis"):
            key = f"ladder_residual__{name}__rmse_mm"
            if key in curves:
                axes[0].plot(degrees[1:], curves[key][1:], marker="o",
                             label=name.replace("_", " "))
        axes[0].set_xscale("log")
        axes[0].set(xlabel="aggregation scale (deg)",
                    ylabel="RMSE of the component below (mm day$^{-1}$)",
                    title=f"A.  Sub-scale error -- {head['pretty']}")
        axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)

        for name in ("coarse_input", "background", "analysis"):
            key = f"ladder_aggregated__{name}__correlation"
            if key in curves:
                axes[1].plot(degrees, curves[key], marker="s",
                             label=name.replace("_", " "))
        axes[1].set_xscale("log")
        axes[1].set(xlabel="aggregation scale (deg)", ylabel="correlation",
                    title="B.  Skill of the component at and above each scale")
        axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)

    windows = curves.get("fss_windows__analysis_mean")
    if windows is not None and windows.size:
        for threshold in ("1", "10", "25"):
            key = f"fss__analysis_mean__{threshold}"
            if key in curves:
                axes[2].plot(windows, curves[key], marker="^",
                             label=f"{threshold} mm day$^{{-1}}$")
        axes[2].set_xscale("log")
        axes[2].set(xlabel="neighbourhood width (fine cells)", ylabel="FSS",
                    title="C.  Fractions skill score, analysis mean")
        axes[2].legend(fontsize=7); axes[2].grid(alpha=0.3)
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Narrative
# ---------------------------------------------------------------------------

def write_results_markdown(path: Path, arms: list[dict], primary: str,
                           ablation: np.ndarray, downscaling: np.ndarray) -> None:
    """A prose summary keyed to the tables, safe to paste into the manuscript."""
    head = next((a for a in arms if a["label"] == primary), arms[0])
    d = head["downscaling"]
    o = head["osse"]

    def fmt(value, digits=3):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return "n/a"
        return "n/a" if not np.isfinite(value) else f"{value:.{digits}f}"

    a_skill = dig(d, "claim_a_downscaling_gain.background.mse_skill")
    b_bg = dig(d, "claim_b_sub_footprint_gain.background.mse_skill")
    b_an = dig(d, "claim_b_sub_footprint_gain.analysis.mse_skill")
    energy = dig(d, "claim_b_sub_footprint_gain.analysis.mean_member_energy_ratio")
    eff_member = dig(d, "structure.spectra.effective_resolution_km.analysis_member")
    eff_coarse = dig(d, "structure.spectra.effective_resolution_km.coarse_input")

    lines = [
        "# OSSE results summary",
        "",
        f"Generated from `{head['directory']}` and {len(arms)} matched arms. "
        "CHIRPS is the 0.05-degree nature run; pseudo-gauges and exact nested "
        "0.1-degree pseudo-IMERG footprints are drawn from it, so every score "
        "below is measured against the same truth that generated the "
        "observations.",
        "",
        "## Headline numbers (primary arm: "
        f"{head['pretty']}, {d.get('n_days', 'n/a')} days, "
        f"{d.get('n_members', 'n/a')} members)",
        "",
        "| Claim | Null model | Field scored | MSE skill |",
        "|---|---|---|---|",
        f"| A. Downscaling gain | coarse conditioning field | background | "
        f"{fmt(a_skill)} |",
        f"| B. Sub-footprint gain | truth's own footprint mean | background | "
        f"{fmt(b_bg)} |",
        f"| B. Sub-footprint gain | truth's own footprint mean | analysis | "
        f"{fmt(b_an)} |",
        "",
        "### How to read these",
        "",
        "Claim A is scored on the **background only**. The background never sees "
        "an observation, so a positive number here is attributable to the "
        "generative prior and to nothing else. Scoring the analysis against this "
        "null would let assimilated observations masquerade as downscaling skill.",
        "",
        "Claim B is scored against a null that is **handed the exact 0.1-degree "
        "block means of the truth** and has precisely zero structure inside each "
        "footprint. In the satellite-only arm, beating it demonstrates located "
        "skill at scales unresolved by every assimilated observation. In gauge "
        "and simultaneous arms, point gauges can constrain this component "
        "locally, so those results are evidence of sub-footprint performance but "
        "not observation-free recovery.",
        "",
        "## Texture realism",
        "",
        f"- Member-to-truth subgrid energy ratio: **{fmt(energy)}** "
        "(1.0 is correct; well above 1 means members are too rough, "
        "below 1 means they are too smooth).",
        f"- Effective resolution of an analysis member: **{fmt(eff_member, 0)} km**, "
        f"against **{fmt(eff_coarse, 0)} km** for the coarse input. This is the "
        "wavelength down to which spectral power stays within a factor of two of "
        "CHIRPS; features finer than this are damped or invented.",
        "",
        "Energy ratio and spread--skill answer different questions and must be "
        "reported together. A member can be too rough while the ensemble is "
        "simultaneously too narrow -- the two defects have opposite fixes, so a "
        "single 'calibration' verdict would hide both.",
        "",
        "## Assimilation scores (withheld pseudo-stations)",
        "",
        f"- Background CRPS {fmt(dig(o, 'withheld_background.crps_mm'), 2)} -> "
        f"analysis {fmt(dig(o, 'withheld_analysis.crps_mm'), 2)} mm/day "
        f"({fmt(dig(o, 'withheld_improvement_crps_mm'), 1)}%)",
        f"- Analysis spread/skill {fmt(dig(o, 'withheld_analysis.spread_skill'), 2)}, "
        f"90% coverage {fmt(dig(o, 'withheld_analysis.coverage_90'), 3)}",
        "",
        "## Ablations",
        "",
        "See `table_ablation.tex` and `fig_ablation_matrix.png`. Arms:",
        "",
    ]
    for arm in arms:
        lines.append(
            f"- **{arm['pretty']}** -- withheld CRPS "
            f"{fmt(dig(arm['osse'], 'withheld_background.crps_mm'), 2)} -> "
            f"{fmt(dig(arm['osse'], 'withheld_analysis.crps_mm'), 2)}; "
            f"claim B (analysis) {fmt(dig(arm['downscaling'], 'claim_b_sub_footprint_gain.analysis.mse_skill'))}"
        )

    by_year = dig(head["downscaling"], "by_year", {})
    if by_year:
        lines += ["", "## Stability across years", "",
                  "| Year | Days | Background | Analysis |", "|---|---|---|---|"]
        for year, payload in sorted(by_year.items()):
            lines.append(
                f"| {year} | {payload.get('n_days', 'n/a')} | "
                f"{fmt(dig(payload, 'background.mse_skill'))} | "
                f"{fmt(dig(payload, 'analysis.mse_skill'))} |")
        lines += ["", "A conclusion that holds in one year and not the others is "
                      "a sampling artifact, not a result."]

    by_intensity = dig(head["downscaling"], "by_intensity", {})
    if by_intensity:
        lines += ["", "## Stability across rainfall intensity", "",
                  "| CHIRPS band (mm/day) | Background | Analysis |", "|---|---|---|"]
        for band, payload in by_intensity.items():
            lines.append(f"| {band} | {fmt(dig(payload, 'background.mse_skill'))} | "
                         f"{fmt(dig(payload, 'analysis.mse_skill'))} |")

    lines += [
        "",
        "## Caveats that belong in the manuscript",
        "",
        "1. CHIRPS supplies both the nature truth and the pseudo-observations, so "
        "this OSSE omits every real satellite and gauge bias and all product "
        "mismatch. It bounds what is achievable; it does not predict real skill.",
        "2. Withheld pseudo-gauges are not fully independent when dense "
        "pseudo-satellite footprints are also assimilated -- they test "
        "sub-footprint allocation at unseen point locations, which is a narrower "
        "claim than independent validation.",
        "3. Spectra are computed on zero-filled, Hann-tapered fields. Gap filling "
        "adds a little spurious power near the coast; interpolating instead would "
        "have suppressed exactly the high-wavenumber power being measured, which "
        "is the error that would flatter the model.",
        "",
        "## Files",
        "",
        "| File | Contents |",
        "|---|---|",
        "| `osse_paper_summary.json` | every number, machine readable |",
        "| `osse_paper_metrics.csv` | tidy long format for any plot |",
        "| `osse_paper_curves.npz` | spectra, FSS, ladders, all arms |",
        "| `table_ablation.tex` | ablation matrix, booktabs |",
        "| `table_downscaling.tex` | downscaling-claim table |",
        "| `fig_*.png` | manuscript figure drafts |",
        "| `<arm>/spatial_fields.nc` | georeferenced maps per arm |",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "paper"
    out_dir.mkdir(parents=True, exist_ok=True)

    arms = [load_arm(label, directory)
            for label, directory in discover_arms(root, args.arm)]
    print(f"loaded {len(arms)} arm(s): {', '.join(a['label'] for a in arms)}")

    ablation = build_matrix(arms, ABLATION_COLUMNS, "osse")
    downscaling = build_matrix(arms, DOWNSCALING_COLUMNS, "downscaling")

    write_latex(out_dir / "table_ablation.tex", arms, ABLATION_COLUMNS, ablation,
                "OSSE ablation matrix at withheld pseudo-stations. CHIRPS is the "
                "nature run; every arm shares dates, seeds and spatial holdout "
                "except where the arm name says otherwise. Best per column in bold.",
                "tab:osse-ablation")
    write_latex(out_dir / "table_downscaling.tex", arms, DOWNSCALING_COLUMNS,
                downscaling,
                "Scale-separated downscaling skill. Claim A is scored on the "
                "background against the coarse conditioning field; claim B is "
                "scored against a null given the exact footprint means of the "
                "truth. Member energy ratio near one indicates realistic "
                "sub-footprint variance.",
                "tab:osse-downscaling")
    write_tidy_csv(out_dir / "osse_paper_metrics.csv", arms)
    write_combined_curves(out_dir / "osse_paper_curves.npz", arms)

    summary = {
        "arms": [
            {"label": a["label"], "pretty": a["pretty"], "directory": a["directory"],
             "osse": a["osse"], "downscaling": a["downscaling"]}
            for a in arms
        ],
        "primary": args.primary,
        "ablation_columns": [c[1] for c in ABLATION_COLUMNS],
        "downscaling_columns": [c[1] for c in DOWNSCALING_COLUMNS],
    }
    (out_dir / "osse_paper_summary.json").write_text(
        json.dumps(summary, indent=2, default=float) + "\n")

    if not args.skip_figures:
        figure_matrix(out_dir / "fig_ablation_matrix.png", arms, ABLATION_COLUMNS,
                      ablation, "OSSE ablations at withheld pseudo-stations")
        figure_matrix(out_dir / "fig_downscaling_matrix.png", arms,
                      DOWNSCALING_COLUMNS, downscaling,
                      "Scale-separated downscaling skill")
        figure_claims(out_dir / "fig_claims.png", arms)
        figure_spectra(out_dir / "fig_spectra.png", arms, args.primary)
        figure_scale_ladder(out_dir / "fig_scale_ladder.png", arms, args.primary)

    write_results_markdown(out_dir / "RESULTS.md", arms, args.primary,
                           ablation, downscaling)

    print(f"\nwrote {out_dir}/")
    for item in sorted(out_dir.iterdir()):
        print(f"  {item.name}")


if __name__ == "__main__":
    main()
