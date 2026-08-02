#!/usr/bin/env python3
"""Compare completed five-day BMD evaluation JSON files across IMERG R cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLOURS = {
    "Background": "#7D8597",
    "Gauges only": "#0077B6",
    "IMERG only": "#F4A261",
    "Simultaneous": "#D1495B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluations", nargs="+", required=True)
    parser.add_argument(
        "--out-json", default="data/processed/bmd_imerg_5day_method_selection.json"
    )
    parser.add_argument(
        "--out-plot", default="data/processed/bmd_imerg_5day_method_selection.png"
    )
    return parser.parse_args()


def finite_or_none(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def main() -> None:
    args = parse_args()
    cases = []
    for path in args.evaluations:
        payload = json.loads(Path(path).read_text())
        scope = payload["scope"]
        extra = finite_or_none(scope.get("imerg_extra_r_multiplier"))
        total = finite_or_none(scope.get("imerg_total_r_inflation"))
        label = (
            f"R×{extra:g}\n(total {total:.1f}×)"
            if extra is not None and total is not None
            else Path(path).stem
        )
        cases.append(
            {
                "path": path,
                "label": label,
                "extra_r": extra,
                "total_r": total,
                "evaluation": payload,
            }
        )
    cases.sort(key=lambda item: item["extra_r"] if item["extra_r"] is not None else 1e9)
    methods = [
        name
        for name in COLOURS
        if all(name in case["evaluation"]["probabilistic_methods"] for case in cases)
    ]
    if not methods:
        raise ValueError("evaluation files share no probabilistic methods")

    labels = [case["label"] for case in cases]
    x = np.arange(len(cases))

    def values(method, key):
        return np.array(
            [case["evaluation"]["probabilistic_methods"][method].get(key, np.nan) for case in cases],
            dtype=float,
        )

    figure, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    for method in methods:
        axes[0, 0].plot(x, values(method, "crps_mm"), marker="o", color=COLOURS[method], label=method)
        axes[0, 1].plot(x, 100 * values(method, "crps_skill_vs_background"), marker="o", color=COLOURS[method], label=method)
        axes[0, 2].plot(x, values(method, "coverage_90"), marker="o", color=COLOURS[method], label=method)
        axes[1, 0].plot(x, values(method, "bias_mm"), marker="o", color=COLOURS[method], label=method)

    axes[0, 0].set_title("A. Withheld-BMD CRPS — lower is better")
    axes[0, 0].set_ylabel("CRPS (mm day$^{-1}$)")
    axes[0, 1].axhline(0, color="black", lw=1)
    axes[0, 1].set_title("B. CRPSS versus background")
    axes[0, 1].set_ylabel("CRPS improvement (%)")
    axes[0, 2].axhline(0.90, color="black", ls="--", lw=1, label="nominal 0.90")
    axes[0, 2].set_title("C. Ensemble 90% coverage")
    axes[0, 2].set_ylabel("Empirical coverage")
    axes[1, 0].axhline(0, color="black", lw=1)
    axes[1, 0].set_title("D. Withheld-BMD mean bias")
    axes[1, 0].set_ylabel("Bias (mm day$^{-1}$)")
    for axis in (axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0]):
        axis.set_xticks(x)
        axis.set_xticklabels(labels)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)

    threshold = "10"
    brier = np.array(
        [
            [
                case["evaluation"]["probabilistic_methods"][method]["thresholds"][threshold]["brier_score"]
                for case in cases
            ]
            for method in methods
        ],
        dtype=float,
    )
    image = axes[1, 1].imshow(brier, cmap="magma", aspect="auto")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels)
    axes[1, 1].set_yticks(np.arange(len(methods)))
    axes[1, 1].set_yticklabels(methods)
    for row in range(len(methods)):
        for column in range(len(cases)):
            axes[1, 1].text(column, row, f"{brier[row, column]:.3f}", ha="center", va="center", color="white", fontsize=8)
    figure.colorbar(image, ax=axes[1, 1], label="Brier score")
    axes[1, 1].set_title("E. Probability skill for ≥10 mm")

    gauge = values("Gauges only", "crps_mm")
    fused_names = [name for name in ("Simultaneous",) if name in methods]
    fused_values = np.stack([values(name, "crps_mm") for name in fused_names])
    best_position = np.argmin(fused_values, axis=0)
    best_fused = np.min(fused_values, axis=0)
    best_fused_name = [fused_names[index] for index in best_position]
    difference = best_fused - gauge
    axes[1, 2].bar(x, difference, color=[COLOURS[name] for name in best_fused_name])
    axes[1, 2].axhline(0, color="black", lw=1)
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(labels)
    axes[1, 2].set_ylabel("Best fused CRPS − gauges-only CRPS")
    axes[1, 2].set_title("F. Fusion gate — negative passes")
    axes[1, 2].grid(axis="y", alpha=0.2)
    for index, value in enumerate(difference):
        axes[1, 2].text(index, value, f"{best_fused_name[index]}\n{value:+.2f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=8)

    passing = np.where(difference < 0)[0]
    if len(passing):
        selected_index = int(passing[np.argmin(best_fused[passing])])
        recommendation = {
            "method": best_fused_name[selected_index],
            "case": cases[selected_index]["label"].replace("\n", " "),
            "reason": "lowest fused withheld-BMD CRPS among cases that beat gauges only",
        }
    else:
        selected_index = int(np.argmin(gauge))
        recommendation = {
            "method": "Gauges only",
            "case": cases[selected_index]["label"].replace("\n", " "),
            "reason": "no fused method beats gauges-only CRPS in this five-day gate",
        }

    summary = {
        "primary_reference": "withheld BMD gauges",
        "cases": cases,
        "fusion_gate": [
            {
                "case": cases[index]["label"].replace("\n", " "),
                "best_fused_method": best_fused_name[index],
                "best_fused_crps_mm": float(best_fused[index]),
                "gauges_only_crps_mm": float(gauge[index]),
                "difference_mm": float(difference[index]),
                "passes": bool(difference[index] < 0),
            }
            for index in range(len(cases))
        ],
        "provisional_recommendation": recommendation,
        "warning": "Five days and one station split are insufficient for final method selection.",
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_plot).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    figure.suptitle(
        "Five-day IMERG-weight sensitivity selected only with withheld BMD gauges\n"
        "CHIRPS is excluded from method ranking; negative fusion-gate values are required",
        fontsize=15,
    )
    figure.savefig(args.out_plot, dpi=180)
    plt.close(figure)
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_plot}")
    print("provisional recommendation:", recommendation["method"], recommendation["case"])


if __name__ == "__main__":
    main()
