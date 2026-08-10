#!/usr/bin/env python
"""One matrix: every ingestion arm, every metric, one verdict.

``scripts/42`` answers the question for ONE arm at a time and writes one JSON
per arm. With eleven configurations and four arms that is a lot of files and no
single place to look, which is how a screen ends up being read off point
estimates again. This consolidates them.

Rows are configurations. Columns are, in the order they should be read:

1. ``n_wet`` -- the sample. Below ~50 nothing below it means anything.
2. **CRPS by arm** -- background, gauges, satellite, combined. The background
   column is the honest baseline: any arm that does not beat it has added
   nothing at all.
3. **combined - gauges** with its 95% interval. This is the experiment's actual
   question, and both arms come from the same dump file, so the pairing is
   exact. An interval containing zero means the satellite changed nothing that
   this sample can detect.
4. **Structure** -- best pattern correlation against CHIRPS/IMERG/CPC, and
   whether the wet-area fraction falls inside the envelope the three products
   span. A field outside all three is implausible whatever its point scores.

The verdict follows the rule fixed before the runs: an arm counts as better
only if its interval excludes zero, and among those that pass, structure
decides. If nothing passes, that is reported as the result rather than
dressed up -- "the satellite does not help at these settings" is a finding.

Example
-------
    python scripts/45_ingestion_matrix.py \\
        --selections 'data/processed/ing2022_selection_*/config_selection.json' \\
        --out-dir data/processed/ing2022_matrix
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np

ARM_ORDER = ["background", "gauges", "satellite", "combined"]


def load_selections(patterns: list[str]) -> dict:
    """``{arm: parsed json}`` from scripts/42 outputs."""
    paths = sorted({Path(p) for pat in patterns
                    for p in (glob.glob(pat) or [pat])})
    paths = [p for p in paths if p.exists()]
    if not paths:
        raise SystemExit(f"no scripts/42 output matched {patterns}")
    out = {}
    for path in paths:
        blob = json.loads(path.read_text())
        arm = blob.get("arm")
        if arm is None:
            print(f"[skip] {path}: no 'arm' field; not a scripts/42 output")
            continue
        if arm in out:
            print(f"[warn] two selections for arm {arm!r}; keeping {path}")
        out[arm] = blob
    if not out:
        raise SystemExit("no usable selection files")
    return out


def build_rows(selections: dict) -> list[dict]:
    """One row per configuration, pulling each arm's numbers into line."""
    configs = sorted({name for blob in selections.values()
                      for name in blob.get("summary", {})})
    contrast = {}
    for blob in selections.values():
        for c in blob.get("arm_comparisons", []):
            # keyed by (arm, vs_arm) so combined-gauges and gauges-background
            # can coexist without overwriting each other
            contrast[(c["config"], c["arm"], c["vs_arm"])] = c

    rows = []
    for name in configs:
        row = {"config": name}
        for arm in ARM_ORDER:
            gauge = selections.get(arm, {}).get("summary", {}).get(name, {}).get("gauge")
            row[f"crps_{arm}"] = gauge["crps"] if gauge else float("nan")
            if arm == "combined" and gauge:
                row.update({"n": gauge["n"], "n_wet": gauge["n_wet"],
                            "n_folds": gauge["n_folds"], "members": gauge["members"],
                            "mae": gauge["mae"], "wet_mae": gauge["wet_mae"],
                            "bias": gauge["bias"], "corr": gauge["correlation"]})
        row.setdefault("n", 0); row.setdefault("n_wet", 0)
        for key in ("n_folds", "members", "mae", "wet_mae", "bias", "corr"):
            row.setdefault(key, float("nan"))

        c = contrast.get((name, "combined", "gauges"))
        row["delta"] = c["difference"] if c else float("nan")
        row["ci_low"] = c["ci_low"] if c else float("nan")
        row["ci_high"] = c["ci_high"] if c else float("nan")
        row["significant"] = bool(c["significant"]) if c else False
        row["helps"] = bool(c and c["significant"] and c["difference"] < 0)
        row["hurts"] = bool(c and c["significant"] and c["difference"] > 0)

        spatial = (selections.get("combined", {}).get("summary", {})
                   .get(name, {}).get("spatial", {}) or {})
        row["pattern"] = spatial.get("pattern_correlation_best", float("nan"))
        row["wet_area"] = spatial.get("wet_area", float("nan"))
        row["wet_inside"] = spatial.get("wet_area_inside")
        row["pattern_by_product"] = spatial.get("pattern_correlation", {})
        rows.append(row)

    rows.sort(key=lambda r: (np.isnan(r["crps_combined"]), r["crps_combined"]))
    return rows


def render_markdown(rows: list[dict], selections: dict) -> str:
    reference = next(iter(selections.values())).get("reference", "?")
    lines = [
        "# Ingestion experiment: configuration matrix",
        "",
        f"Reference configuration: `{reference}`. Gauges are truth for value; "
        "CHIRPS/IMERG/CPC bound plausible structure.",
        "",
        "`delta` is combined minus gauges, paired exactly (same dump, same "
        "withheld stations, same seeds). Negative means the satellite helped. "
        "**Read the interval, not the point estimate.**",
        "",
        "| config | n_wet | CRPS bg | CRPS gauge | CRPS sat | CRPS comb | "
        "delta (95% CI) | verdict | wetMAE | bias | patt | wet area |",
        "|---|--:|--:|--:|--:|--:|:--|:--|--:|--:|--:|:--|",
    ]
    for r in rows:
        if r["helps"]:
            verdict = "**satellite helps**"
        elif r["hurts"]:
            verdict = "satellite HURTS"
        elif np.isnan(r["delta"]):
            verdict = "-"
        else:
            verdict = "no effect"
        inside = {True: "inside", False: "OUTSIDE", None: "?"}[r["wet_inside"]]
        lines.append(
            f"| `{r['config']}` | {r['n_wet']:,} | "
            f"{r['crps_background']:.2f} | {r['crps_gauges']:.2f} | "
            f"{r['crps_satellite']:.2f} | {r['crps_combined']:.2f} | "
            f"{r['delta']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | "
            f"{verdict} | {r['wet_mae']:.2f} | {r['bias']:+.2f} | "
            f"{r['pattern']:.2f} | {r['wet_area']:.3f} {inside} |"
        )
    return "\n".join(lines) + "\n"


def verdict_block(rows: list[dict]) -> list[str]:
    out = []
    sample = max((r["n_wet"] for r in rows), default=0)
    if sample < 50:
        out.append(f"SAMPLE TOO SMALL: {sample} wet station-days. Nothing below "
                   f"this line is interpretable; extend the window.")
    helped = [r for r in rows if r["helps"]]
    hurt = [r for r in rows if r["hurts"]]
    if helped:
        best = min(helped, key=lambda r: r["crps_combined"])
        out.append(f"Satellite significantly HELPS in {len(helped)} arm(s): "
                   f"{', '.join(r['config'] for r in helped)}")
        out.append(f"  Best by CRPS among them: {best['config']} "
                   f"({best['crps_combined']:.3f}, delta {best['delta']:+.3f})")
        structural = [r for r in helped if r["wet_inside"]]
        if structural:
            out.append(f"  Of those, structurally plausible (wet area inside the "
                       f"product envelope): "
                       f"{', '.join(r['config'] for r in structural)}")
        else:
            out.append("  NONE of them has a wet-area fraction inside the product "
                       "envelope; the point skill is not backed by plausible "
                       "structure.")
    if hurt:
        out.append(f"Satellite significantly HURTS in {len(hurt)} arm(s): "
                   f"{', '.join(r['config'] for r in hurt)}")
    if not helped and not hurt:
        out.append("No arm makes the satellite distinguishable from gauges alone.")
        out.append("  On this sample, IMERG assimilation neither helps nor hurts "
                   "at any setting tried.")
        out.append("  That is a result. Report it as one; do not rank the arms "
                   "on point estimates.")
    # Only judge against the background where it was actually computed; a NaN
    # column would otherwise fire this alarm on every run that omitted it.
    comparable = [r for r in rows if np.isfinite(r["crps_background"])
                  and np.isfinite(r["crps_combined"])]
    if not comparable:
        out.append("No background column: run scripts/42 with --arm background "
                   "to get the baseline every arm should be beating.")
    elif not [r for r in comparable if r["crps_combined"] < r["crps_background"]]:
        out.append("WARNING: no arm beats the BACKGROUND. The assimilation is "
                   "not adding information at all, which is a bigger problem "
                   "than any choice between arms.")
    return out


def plot_matrix(rows: list[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 140, "font.size": 9,
                         "axes.grid": True, "grid.alpha": 0.25,
                         "axes.spines.top": False, "axes.spines.right": False})
    names = [r["config"] for r in rows]
    y = np.arange(len(rows))
    figure, axes = plt.subplots(1, 3, figsize=(16.5, 0.42 * len(rows) + 3.4))

    axis = axes[0]
    width = 0.2
    for i, (arm, colour) in enumerate([("background", "#999999"),
                                       ("gauges", "#1f6f8b"),
                                       ("satellite", "#e0a458"),
                                       ("combined", "#c1440e")]):
        axis.barh(y + i * width, [r[f"crps_{arm}"] for r in rows],
                  height=width, color=colour, label=arm)
    axis.set_yticks(y + 1.5 * width); axis.set_yticklabels(names, fontsize=7.5)
    axis.invert_yaxis(); axis.legend(fontsize=7)
    axis.set_xlabel("CRPS at withheld gauges (mm/day)")
    axis.set_title("Point skill by arm — gauges are truth", fontsize=9.5)

    axis = axes[1]
    delta = np.array([r["delta"] for r in rows])
    lo = delta - np.array([r["ci_low"] for r in rows])
    hi = np.array([r["ci_high"] for r in rows]) - delta
    axis.errorbar(delta, y, xerr=[lo, hi], fmt="none", ecolor="#555555", capsize=3)
    for i, r in enumerate(rows):
        colour = "#1a7f37" if r["helps"] else ("#c1440e" if r["hurts"] else "#999999")
        axis.plot(delta[i], y[i], "o", ms=8, color=colour)
    axis.axvline(0.0, color="#111111", lw=1.4, ls="--")
    axis.set_yticks(y); axis.set_yticklabels(names, fontsize=7.5)
    axis.invert_yaxis()
    axis.set_xlabel("CRPS(combined) - CRPS(gauges), mm/day")
    axis.set_title("Does the satellite help?\ngreen better, red worse, "
                   "grey indistinguishable", fontsize=9.5)

    axis = axes[2]
    axis.barh(y, [r["pattern"] for r in rows], color="#1f6f8b")
    for i, r in enumerate(rows):
        if r["wet_inside"] is False:
            axis.text(0.01, y[i], " wet area OUTSIDE envelope", va="center",
                      fontsize=6.5, color="#c1440e")
    axis.set_yticks(y); axis.set_yticklabels(names, fontsize=7.5)
    axis.invert_yaxis(); axis.set_xlabel("best pattern correlation vs products")
    axis.set_title("Structure — products bound plausibility", fontsize=9.5)

    figure.suptitle("Ingestion experiment: how should IMERG and BMD be combined?",
                    y=1.0)
    figure.tight_layout()
    figure.savefig(out_path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consolidate scripts/42 outputs into one matrix",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--selections", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    selections = load_selections(args.selections)
    missing = [a for a in ARM_ORDER if a not in selections]
    if missing:
        print(f"[warn] no selection for arm(s) {missing}; their columns will be "
              f"blank. Run scripts/42 with --arm for each.")
    rows = build_rows(selections)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    markdown = render_markdown(rows, selections)
    (out_dir / "ingestion_matrix.md").write_text(markdown)
    print(markdown)

    fields = ["config", "n", "n_wet", "n_folds", "members",
              *[f"crps_{a}" for a in ARM_ORDER],
              "mae", "wet_mae", "bias", "corr",
              "delta", "ci_low", "ci_high", "significant", "helps", "hurts",
              "pattern", "wet_area", "wet_inside"]
    with (out_dir / "ingestion_matrix.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print("[verdict]")
    for line in verdict_block(rows):
        print(f"    {line}")

    plot_matrix(rows, out_dir / "ingestion_matrix.png")
    (out_dir / "ingestion_matrix.json").write_text(
        json.dumps({"rows": rows, "arms": sorted(selections)}, indent=2,
                   default=float))
    print()
    for name in ("ingestion_matrix.md", "ingestion_matrix.csv",
                 "ingestion_matrix.png", "ingestion_matrix.json"):
        print(f"[done] wrote {out_dir / name}")


if __name__ == "__main__":
    main()
