#!/usr/bin/env python
"""Paper figures 4-7 and Table 1: the OSSE.

``scripts/24_osse_paper_suite.py`` already computes these quantities and knows
the on-disk schema of an OSSE arm. This script does NOT reimplement that: it
imports ``load_arm``, ``dig`` and the column definitions from it, so the schema
lives in exactly one place. What it adds is paper styling, stable figure
numbering, and -- the part 24 does not do -- a CSV of the numbers behind every
panel plus a provenance manifest.

Figure 4  Claim A (downscaling gain over the coarse input) and Claim B
          (sub-footprint gain over a null handed perfect footprint means).
          Claim B is the load-bearing one: its null is deliberately unfair,
          because beating "perfect coarse information" is the only way to show
          the prior allocated rain correctly INSIDE footprints it was told only
          the average of.

Figure 5  Radially averaged power spectra against the nature run, with the
          effective resolution each arm reaches.

Figure 6  The OSSE scale ladder. The real-data version of this result is
          confounded by product circularity -- pattern correlation against
          IMERG when IMERG is what is assimilated. Here truth is known, so it
          is clean.

Figure 7  Observation value: what gauges, satellite, and both together buy,
          across network densities.

Table 1   Arm x metric matrix, as LaTeX.

Example
-------
    python scripts/47_osse_paper_figures.py \\
        --root data/processed/osse_paper \\
        --primary simultaneous_realistic_40 \\
        --out-dir docs/paper_figures
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bdhires.paper import (  # noqa: E402
    FIGURE_WIDTH_TWO_COLUMN,
    PALETTE,
    save_figure,
    use_paper_style,
)


def load_suite():
    """Import scripts/24 as a module despite the leading digit in its name.

    Reusing its loaders rather than re-reading the JSON keeps one definition of
    the arm schema. Guessing at that schema is how a summary script silently
    reports the wrong field.
    """
    path = ROOT / "scripts" / "24_osse_paper_suite.py"
    if not path.exists():
        raise SystemExit(f"missing {path}; this script reuses its loaders")
    spec = importlib.util.spec_from_file_location("osse_paper_suite", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["osse_paper_suite"] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="data/processed/osse_paper")
    parser.add_argument("--arm", action="append", default=None,
                        metavar="LABEL=DIR")
    parser.add_argument("--primary", default="simultaneous_realistic_40",
                        help="arm used for the single-arm panels (5 and 6)")
    parser.add_argument("--out-dir", default="docs/paper_figures")
    parser.add_argument("--skip", default="", help="comma-separated: 4,5,6,7")
    return parser.parse_args()


def _series(arm: dict, *names):
    """First present curve among ``names``, as a float array, else None."""
    for name in names:
        value = arm.get("curves", {}).get(name)
        if value is not None and np.size(value):
            return np.asarray(value, float)
    return None


# ---------------------------------------------------------------- figure 4

def figure_claims(suite, arms, out_dir: Path, sources) -> None:
    plt = use_paper_style()
    dig = suite.dig

    # Dotted paths taken from suite.DOWNSCALING_COLUMNS rather than invented.
    # An earlier draft guessed "claim_a.skill_score", which does not exist and
    # would have produced an empty figure against real output while passing a
    # synthetic test built from the same guess.
    labels, rows = [], []
    for arm in arms:
        d = arm["downscaling"]
        rows.append({
            "arm": arm["label"],
            "claim_a_mse_skill":
                dig(d, "claim_a_downscaling_gain.background.mse_skill"),
            "claim_a_crps_skill":
                dig(d, "claim_a_downscaling_gain.background.crps_skill"),
            "claim_b_mse_skill":
                dig(d, "claim_b_sub_footprint_gain.analysis.mse_skill"),
            "claim_b_crps_skill":
                dig(d, "claim_b_sub_footprint_gain.analysis.crps_skill"),
            "claim_b_correlation":
                dig(d, "claim_b_sub_footprint_gain.analysis.correlation"),
            "claim_b_background_mse_skill":
                dig(d, "claim_b_sub_footprint_gain.background.mse_skill"),
        })
        labels.append(arm["pretty"])

    skill_a = np.array([r["claim_a_mse_skill"] for r in rows], float)
    skill_b = np.array([r["claim_b_mse_skill"] for r in rows], float)
    if not np.isfinite(skill_a).any() and not np.isfinite(skill_b).any():
        print("[fig04] SKIP: no claim_a/claim_b skill scores in any arm. "
              "Run scripts/22_evaluate_osse_downscaling.py first.")
        plt.close("all")
        return

    y = np.arange(len(labels))
    figure, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH_TWO_COLUMN,
                                               0.32 * len(labels) + 1.9),
                                sharey=True)
    for axis, skill, title, note in (
        (axes[0], skill_a, "(a) Claim A: downscaling gain",
         "vs 0.5° coarse input"),
        (axes[1], skill_b, "(b) Claim B: sub-footprint gain",
         "vs perfect 0.1° footprint means"),
    ):
        colours = [PALETTE["accent"] if np.isfinite(v) and v > 0
                   else PALETTE["warn"] for v in skill]
        axis.barh(y, skill, color=colours, height=0.72)
        axis.axvline(0.0, color=PALETTE["truth"], lw=1.0)
        axis.set_xlabel("skill score (1 = perfect, 0 = no better than null)")
        axis.set_title(f"{title}\n{note}")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=6.5)
    axes[0].invert_yaxis()

    figure.tight_layout()
    save_figure(figure, out_dir, "04", "osse_claims",
                data={"claims": rows}, sources=sources,
                caption="OSSE downscaling claims against their nulls. Claim A "
                        "scores the background only, so no assimilated "
                        "observation can be counted as downscaling skill. "
                        "Claim B's null is handed perfect footprint-mean "
                        "information and zero sub-footprint structure.")
    plt.close(figure)


# ---------------------------------------------------------------- figure 5

def figure_spectra(suite, arms, primary: str, out_dir: Path, sources) -> None:
    plt = use_paper_style()
    chosen = [a for a in arms if a["label"] == primary] or arms[:1]
    arm = chosen[0]

    wavelength = _series(arm, "wavelength_km", "wavelengths_km", "scale_km")
    if wavelength is None:
        print("[fig05] SKIP: no spectral curves in "
              f"{arm['directory']}/downscaling_curves.npz")
        plt.close("all")
        return

    series = {
        "truth": _series(arm, "psd_truth", "truth_psd"),
        "background": _series(arm, "psd_background", "background_psd"),
        "analysis": _series(arm, "psd_analysis", "analysis_psd"),
        "coarse input": _series(arm, "psd_coarse", "coarse_psd", "psd_base"),
    }
    series = {k: v for k, v in series.items() if v is not None}
    if len(series) < 2:
        print(f"[fig05] SKIP: only {list(series)} available; need truth plus "
              "at least one field")
        plt.close("all")
        return

    figure, axis = plt.subplots(figsize=(FIGURE_WIDTH_TWO_COLUMN * 0.55, 2.9))
    colours = {"truth": PALETTE["truth"], "background": PALETTE["background"],
               "analysis": PALETTE["combined"], "coarse input": PALETTE["null"]}
    styles = {"truth": "-", "background": "--", "analysis": "-",
              "coarse input": ":"}
    for name, values in series.items():
        axis.loglog(wavelength, values, styles.get(name, "-"),
                    color=colours.get(name, None), label=name)
    axis.invert_xaxis()
    axis.set_xlabel("wavelength (km)")
    axis.set_ylabel("radially averaged power")
    axis.set_title(f"(a) Spectra against the nature run\n{arm['pretty']}")
    axis.legend()

    figure.tight_layout()
    save_figure(figure, out_dir, "05", "osse_spectra",
                data={"psd": {"wavelength_km": wavelength, **series}},
                sources=sources,
                caption="Radially averaged power spectra. The coarse input has "
                        "no variance below its own resolution; the prior "
                        "synthesises it, and the analysis retains it.")
    plt.close(figure)


# ---------------------------------------------------------------- figure 6

def figure_scale_ladder(suite, arms, out_dir: Path, sources) -> None:
    plt = use_paper_style()
    dig = suite.dig
    rows = []
    for arm in arms:
        factor = dig(arm["downscaling"], "satellite_factor",
                     dig(arm["osse"], "satellite_factor", np.nan))
        if not np.isfinite(factor):
            continue
        rows.append({
            "arm": arm["label"],
            "satellite_factor": float(factor),
            "footprint_deg": float(factor) * 0.05,
            "crps_mm": dig(arm["osse"], "withheld_analysis.crps_mm"),
            "crps_background_mm": dig(arm["osse"], "withheld_background.crps_mm"),
            "claim_b_skill": dig(arm["downscaling"],
                                 "claim_b_sub_footprint_gain.analysis.mse_skill"),
        })
    if len(rows) < 2:
        print("[fig06] SKIP: fewer than two arms report satellite_factor. "
              "Run slurm/submit_osse_footprint_ablation.sh for the ladder.")
        plt.close("all")
        return

    rows.sort(key=lambda r: r["footprint_deg"])
    x = np.array([r["footprint_deg"] for r in rows])
    figure, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH_TWO_COLUMN, 2.6))
    axes[0].plot(x, [r["crps_mm"] for r in rows], "o-",
                 color=PALETTE["combined"])
    axes[0].set_xlabel("observation footprint (°)")
    axes[0].set_ylabel("CRPS at withheld gauges (mm day$^{-1}$)")
    axes[0].set_title("(a) Point skill vs observation scale")
    axes[1].plot(x, [r["claim_b_skill"] for r in rows], "o-",
                 color=PALETTE["accent"])
    axes[1].axhline(0.0, color=PALETTE["truth"], lw=0.9)
    axes[1].set_xlabel("observation footprint (°)")
    axes[1].set_ylabel("Claim B skill score")
    axes[1].set_title("(b) Sub-footprint structure vs observation scale")

    figure.tight_layout()
    save_figure(figure, out_dir, "06", "osse_scale_ladder",
                data={"ladder": rows}, sources=sources,
                caption="Observation scale in the OSSE, where truth is known "
                        "and the product circularity affecting the real-data "
                        "version is absent.")
    plt.close(figure)


# ---------------------------------------------------------------- figure 7

def figure_observation_value(suite, arms, out_dir: Path, sources) -> None:
    plt = use_paper_style()
    dig = suite.dig
    rows = []
    for arm in arms:
        rows.append({
            "arm": arm["label"],
            "pretty": arm["pretty"],
            "network": dig(arm["downscaling"], "network",
                           dig(arm["osse"], "network", np.nan)),
            "crps_background": dig(arm["osse"], "withheld_background.crps_mm"),
            "crps_analysis": dig(arm["osse"], "withheld_analysis.crps_mm"),
            "crps_gain_percent": dig(arm["osse"], "withheld_improvement_crps_mm"),
            "spread_skill": dig(arm["osse"], "withheld_analysis.spread_skill"),
            "coverage_90": dig(arm["osse"], "withheld_analysis.coverage_90"),
        })
    finite = [r for r in rows if np.isfinite(r["crps_analysis"])]
    if not finite:
        print("[fig07] SKIP: no arm reports crps_mm")
        plt.close("all")
        return

    labels = [r["pretty"] for r in finite]
    y = np.arange(len(finite))
    figure, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH_TWO_COLUMN,
                                               0.32 * len(finite) + 1.9))
    axis = axes[0]
    axis.barh(y - 0.2, [r["crps_background"] for r in finite], height=0.38,
              color=PALETTE["background"], label="background")
    axis.barh(y + 0.2, [r["crps_analysis"] for r in finite], height=0.38,
              color=PALETTE["combined"], label="analysis")
    axis.set_yticks(y); axis.set_yticklabels(labels, fontsize=6.5)
    axis.invert_yaxis(); axis.legend()
    axis.set_xlabel("CRPS at withheld gauges (mm day$^{-1}$)")
    axis.set_title("(a) What the observations buy")

    axis = axes[1]
    ratio = np.array([r["spread_skill"] for r in finite], float)
    axis.barh(y, ratio, color=PALETTE["gauges"], height=0.72)
    axis.axvline(1.0, color=PALETTE["truth"], lw=1.0, ls="--")
    axis.set_yticks(y); axis.set_yticklabels([])
    axis.invert_yaxis()
    axis.set_xlabel("spread / skill  (1 = calibrated)")
    axis.set_title("(b) Ensemble calibration")

    figure.tight_layout()
    save_figure(figure, out_dir, "07", "osse_observation_value",
                data={"arms": rows}, sources=sources,
                caption="Observation value and calibration by arm. A "
                        "spread-skill ratio below one indicates an "
                        "under-dispersive ensemble.")
    plt.close(figure)


# ------------------------------------------------------------------ table 1

def table_matrix(suite, arms, out_dir: Path) -> None:
    columns = suite.ABLATION_COLUMNS
    matrix = suite.build_matrix(arms, columns, "osse")
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "table01_osse_matrix.tex"
    suite.write_latex(target, arms, columns, matrix,
                      "OSSE arm comparison. Best value in each column in bold.",
                      "tab:osse-matrix")
    rows = []
    for arm, line in zip(arms, matrix):
        row = {"arm": arm["label"]}
        row.update({c[0] if isinstance(c, (tuple, list)) else str(c): v
                    for c, v in zip(columns, line)})
        rows.append(row)
    from bdhires.paper.style import _write_table
    data_dir = out_dir / "data"; data_dir.mkdir(exist_ok=True)
    _write_table(data_dir / "table01_osse_matrix.csv", rows)
    print(f"[table01] {target.name} (+ CSV)")


def main() -> None:
    args = parse_args()
    suite = load_suite()
    out_dir = Path(args.out_dir)
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    discovered = suite.discover_arms(Path(args.root), args.arm)
    arms = [suite.load_arm(label, directory) for label, directory in discovered]
    sources = [a["directory"] for a in arms]
    print(f"[setup] {len(arms)} OSSE arm(s) from {args.root}")
    for arm in arms:
        print(f"    {arm['label']}")

    if "4" not in skip:
        figure_claims(suite, arms, out_dir, sources)
    if "5" not in skip:
        figure_spectra(suite, arms, args.primary, out_dir, sources)
    if "6" not in skip:
        figure_scale_ladder(suite, arms, out_dir, sources)
    if "7" not in skip:
        figure_observation_value(suite, arms, out_dir, sources)
    table_matrix(suite, arms, out_dir)
    print(f"\n[done] figures and data in {out_dir}/ and {out_dir}/data/")


if __name__ == "__main__":
    main()
