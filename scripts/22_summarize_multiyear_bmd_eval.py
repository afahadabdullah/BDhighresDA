#!/usr/bin/env python3
"""Pool evaluation metrics across multi-year real BMD + IMERG runs (2021-2024).

Usage:
    python scripts/22_summarize_multiyear_bmd_eval.py \
        --summaries data/processed/bmd_imerg_eval_2021_may_sep/rotated_summary.json \
                    data/processed/bmd_imerg_eval_2022_may_sep/rotated_summary.json \
                    data/processed/bmd_imerg_eval_2023_may_sep/rotated_summary.json \
                    data/processed/bmd_imerg_eval_2024_may_jun/rotated_summary.json \
        --out-json data/processed/bmd_imerg_2021_2024_pooled_summary.json \
        --out-markdown data/processed/bmd_imerg_2021_2024_pooled_summary.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = ("Background", "Gauges only", "IMERG only", "Simultaneous")
METHOD_COLOURS = {
    "Background": "#7D8597",
    "Gauges only": "#0077B6",
    "IMERG only": "#F4A261",
    "Simultaneous": "#D1495B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summaries", nargs="+", required=True, help="List of rotated_summary.json files")
    parser.add_argument("--out-json", default="data/processed/bmd_imerg_2021_2024_pooled_summary.json")
    parser.add_argument("--out-markdown", default="data/processed/bmd_imerg_2021_2024_pooled_summary.md")
    parser.add_argument("--out-plot", default="data/processed/bmd_imerg_2021_2024_pooled_summary.png")
    return parser.parse_args()


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return float(np.average(values[valid], weights=weights[valid])) if valid.any() else float("nan")


def main() -> None:
    args = parse_args()
    runs = []
    run_labels = []
    for filepath in args.summaries:
        path = Path(filepath)
        if not path.is_file():
            print(f"Warning: summary file not found, skipping: {filepath}")
            continue
        data = json.loads(path.read_text())
        years = data["scope"]["years"]
        if len(years) == 1:
            label = str(years[0])
        else:
            label = f"{years[0]}-{years[-1]}"
        runs.append({"path": str(path), "label": label, "data": data})
        run_labels.append(label)

    if not runs:
        raise ValueError("No valid summary JSON files provided.")

    total_station_days = sum(run["data"]["scope"]["total_withheld_station_days"] for run in runs)
    weights = np.array([run["data"]["scope"]["total_withheld_station_days"] for run in runs], dtype=float)

    pooled_results = {}
    for method in METHODS:
        crps_list = np.array([run["data"]["methods"][method]["crps_mm"] for run in runs], dtype=float)
        rmse_list = np.array([run["data"]["methods"][method]["rmse_mm"] for run in runs], dtype=float)
        mae_list = np.array([run["data"]["methods"][method]["mae_mm"] for run in runs], dtype=float)
        bias_list = np.array([run["data"]["methods"][method]["bias_mm"] for run in runs], dtype=float)

        corrs = np.clip(
            np.array([run["data"]["methods"][method]["correlation_fisher_pooled"] for run in runs], dtype=float),
            -0.999999, 0.999999
        )
        fisher_z = np.arctanh(corrs)
        fisher_w = np.maximum(weights - 3.0, 1.0)
        pooled_corr = float(np.tanh(weighted_mean(fisher_z, fisher_w)))

        brier_scores = {}
        for thresh in ("1", "10", "25", "50"):
            b_list = np.array(
                [run["data"]["methods"][method]["brier_score"].get(thresh, np.nan) for run in runs],
                dtype=float
            )
            brier_scores[thresh] = weighted_mean(b_list, weights)

        pooled_results[method] = {
            "crps_mm": weighted_mean(crps_list, weights),
            "rmse_mm": float(np.sqrt(weighted_mean(rmse_list**2, weights))),
            "mae_mm": weighted_mean(mae_list, weights),
            "bias_mm": weighted_mean(bias_list, weights),
            "correlation": pooled_corr,
            "brier_score": brier_scores,
        }

    summary_payload = {
        "runs": [r["path"] for r in runs],
        "years": sorted(list(set(y for r in runs for y in r["data"]["scope"]["years"]))),
        "total_station_days": total_station_days,
        "pooled_methods": pooled_results,
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(summary_payload, indent=2) + "\n")

    # Generate Markdown Table
    rows = []
    for m in METHODS:
        p = pooled_results[m]
        rows.append({
            "Method": m,
            "CRPS (mm)": f"{p['crps_mm']:.3f}",
            "RMSE (mm)": f"{p['rmse_mm']:.3f}",
            "MAE (mm)": f"{p['mae_mm']:.3f}",
            "Bias (mm)": f"{p['bias_mm']:.3f}",
            "Correlation": f"{p['correlation']:.3f}",
            "BS (>1mm)": f"{p['brier_score']['1']:.4f}",
            "BS (>10mm)": f"{p['brier_score']['10']:.4f}",
            "BS (>25mm)": f"{p['brier_score']['25']:.4f}",
            "BS (>50mm)": f"{p['brier_score']['50']:.4f}",
        })

    df = pd.DataFrame(rows)
    md_content = f"# Pooled Real BMD + IMERG Multi-Year Evaluation Summary (2021–2024)\n\n"
    md_content += f"- **Years Included**: {', '.join(str(y) for y in summary_payload['years'])}\n"
    md_content += f"- **Total Withheld Station-Days**: {total_station_days:,}\n\n"
    md_content += df.to_markdown(index=False) + "\n"

    Path(args.out_markdown).write_text(md_content)

    # Plot Multi-Year Summary Figure
    categories = run_labels + ["Pooled"]
    n_cat = len(categories)
    x = np.arange(n_cat)
    width = 0.18

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

    # Panel A: CRPS by Year & Pooled
    for idx, method in enumerate(METHODS):
        vals = [run["data"]["methods"][method]["crps_mm"] for run in runs] + [pooled_results[method]["crps_mm"]]
        axes[0, 0].bar(x + (idx - 1.5) * width, vals, width, color=METHOD_COLOURS[method], label=method)
    axes[0, 0].set_xticks(x, categories)
    axes[0, 0].set_ylabel("CRPS (mm day$^{-1}$)")
    axes[0, 0].set_title("A. CRPS by Year & Multi-Year Pooled")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(alpha=0.2)

    # Panel B: Deterministic Metrics (RMSE & MAE - Pooled)
    b_x = np.arange(len(METHODS))
    b_width = 0.35
    rmse_vals = [pooled_results[m]["rmse_mm"] for m in METHODS]
    mae_vals = [pooled_results[m]["mae_mm"] for m in METHODS]
    axes[0, 1].bar(b_x - b_width / 2, rmse_vals, b_width, label="RMSE", color="#2A9D8F")
    axes[0, 1].bar(b_x + b_width / 2, mae_vals, b_width, label="MAE", color="#E76F51")
    axes[0, 1].set_xticks(b_x, METHODS, rotation=15)
    axes[0, 1].set_ylabel("Error (mm day$^{-1}$)")
    axes[0, 1].set_title("B. Pooled RMSE & MAE by Method")
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(alpha=0.2)

    # Panel C: Correlation by Year & Pooled
    for idx, method in enumerate(METHODS):
        vals = [run["data"]["methods"][method]["correlation_fisher_pooled"] for run in runs] + [pooled_results[method]["correlation"]]
        axes[0, 2].plot(x, vals, marker="o", lw=2, color=METHOD_COLOURS[method], label=method)
    axes[0, 2].set_xticks(x, categories)
    axes[0, 2].set_ylabel("Fisher-Pooled Correlation")
    axes[0, 2].set_title("C. Correlation across Evaluation Years")
    axes[0, 2].legend(fontsize=8)
    axes[0, 2].grid(alpha=0.2)

    # Panel D: Heavy Rainfall Brier Score (Pooled)
    thresholds = ("1", "10", "25", "50")
    t_x = np.arange(len(thresholds))
    for idx, method in enumerate(METHODS):
        b_vals = [pooled_results[method]["brier_score"][t] for t in thresholds]
        axes[1, 0].bar(t_x + (idx - 1.5) * width, b_vals, width, color=METHOD_COLOURS[method], label=method)
    axes[1, 0].set_xticks(t_x, [f"≥{t} mm" for t in thresholds])
    axes[1, 0].set_ylabel("Brier Score (lower is better)")
    axes[1, 0].set_title("D. Pooled Heavy Precipitation Brier Score")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.2)

    # Panel E: Simultaneous DA CRPS Skill Gain (%) vs Benchmarks
    bg_crps = np.array([run["data"]["methods"]["Background"]["crps_mm"] for run in runs] + [pooled_results["Background"]["crps_mm"]])
    sim_crps = np.array([run["data"]["methods"]["Simultaneous"]["crps_mm"] for run in runs] + [pooled_results["Simultaneous"]["crps_mm"]])
    gauges_crps = np.array([run["data"]["methods"]["Gauges only"]["crps_mm"] for run in runs] + [pooled_results["Gauges only"]["crps_mm"]])
    imerg_crps = np.array([run["data"]["methods"]["IMERG only"]["crps_mm"] for run in runs] + [pooled_results["IMERG only"]["crps_mm"]])

    gain_vs_bg = (1.0 - sim_crps / bg_crps) * 100.0
    gain_vs_gauges = (1.0 - sim_crps / gauges_crps) * 100.0
    gain_vs_imerg = (1.0 - sim_crps / imerg_crps) * 100.0

    axes[1, 1].bar(x - width, gain_vs_bg, width, label="vs Background", color="#7D8597")
    axes[1, 1].bar(x, gain_vs_gauges, width, label="vs Gauges Only", color="#0077B6")
    axes[1, 1].bar(x + width, gain_vs_imerg, width, label="vs IMERG Only", color="#F4A261")
    axes[1, 1].axhline(0, color="black", lw=1)
    axes[1, 1].set_xticks(x, categories)
    axes[1, 1].set_ylabel("CRPS Improvement (%)")
    axes[1, 1].set_title("E. Simultaneous DA Skill Gain (%)")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.2)

    # Panel F: Withheld Station-Days per Year
    st_days = [run["data"]["scope"]["total_withheld_station_days"] for run in runs] + [total_station_days]
    bars = axes[1, 2].bar(x, st_days, color="#264653")
    axes[1, 2].set_xticks(x, categories)
    axes[1, 2].set_ylabel("Withheld Station-Days")
    axes[1, 2].set_title("F. Evaluation Scope & Sample Sizes")
    for bar in bars:
        height = bar.get_height()
        axes[1, 2].text(bar.get_x() + bar.get_width() / 2.0, height * 1.01, f"{height:,}", ha="center", va="bottom", fontsize=8)
    axes[1, 2].grid(alpha=0.2)

    fig.suptitle("Pooled Real BMD + IMERG Multi-Year Evaluation Summary (2021–2024)", fontsize=15)
    Path(args.out_plot).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_plot, dpi=200)
    plt.close(fig)

    print(f"Wrote pooled JSON: {args.out_json}")
    print(f"Wrote pooled Markdown: {args.out_markdown}")
    print(f"Wrote pooled Figure: {args.out_plot}")
    print("\n" + md_content)


if __name__ == "__main__":
    main()

