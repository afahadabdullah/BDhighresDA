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

import numpy as np
import pandas as pd


METHODS = ("Background", "Gauges only", "IMERG only", "Simultaneous")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summaries", nargs="+", required=True, help="List of rotated_summary.json files")
    parser.add_argument("--out-json", default="data/processed/bmd_imerg_2021_2024_pooled_summary.json")
    parser.add_argument("--out-markdown", default="data/processed/bmd_imerg_2021_2024_pooled_summary.md")
    return parser.parse_args()


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return float(np.average(values[valid], weights=weights[valid])) if valid.any() else float("nan")


def main() -> None:
    args = parse_args()
    runs = []
    for filepath in args.summaries:
        path = Path(filepath)
        if not path.is_file():
            print(f"Warning: summary file not found, skipping: {filepath}")
            continue
        data = json.loads(path.read_text())
        runs.append({"path": str(path), "data": data})

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

    Path(args.out-json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out-json).write_text(json.dumps(summary_payload, indent=2) + "\n")

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

    Path(args.out-markdown).write_text(md_content)

    print(f"Wrote pooled JSON: {args.out-json}")
    print(f"Wrote pooled Markdown: {args.out-markdown}")
    print("\n" + md_content)


if __name__ == "__main__":
    main()
