#!/usr/bin/env python3
"""Summarize matched BMD spatial holdout folds using withheld gauges only."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = ("Background", "Gauges only", "IMERG only", "Simultaneous")
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
        "--out-json", default="data/processed/bmd_imerg_offset_m1_rotated_summary.json"
    )
    parser.add_argument(
        "--out-plot", default="data/processed/bmd_imerg_offset_m1_rotated_summary.png"
    )
    return parser.parse_args()


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return float(np.average(values[valid], weights=weights[valid])) if valid.any() else float("nan")


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def main() -> None:
    args = parse_args()
    folds = []
    for filename in args.evaluations:
        payload = json.loads(Path(filename).read_text())
        scope = payload["scope"]
        methods = payload["probabilistic_methods"]
        missing = set(METHODS) - set(methods)
        if missing:
            raise ValueError(f"{filename} lacks active methods: {sorted(missing)}")
        folds.append(
            {
                "file": filename,
                "fold": int(scope["holdout_fold"]),
                "folds": int(scope["holdout_folds"]),
                "station_days": int(scope["station_days"]),
                "station_count": int(scope["assimilated_stations"])
                + int(scope["withheld_stations"]),
                "dates": scope["dates"],
                "background_day_offset": int(scope["background_day_offset"]),
                "withheld_ids": [str(value) for value in scope["withheld_station_ids"]],
                "withheld_names": scope["withheld_station_names"],
                "methods": methods,
            }
        )
    folds.sort(key=lambda item: item["fold"])
    expected = folds[0]["folds"]
    if len(folds) != expected or [item["fold"] for item in folds] != list(range(expected)):
        raise ValueError(f"need exactly folds 0..{expected - 1}; received {[item['fold'] for item in folds]}")
    if any(item["folds"] != expected for item in folds):
        raise ValueError("evaluation files disagree on holdout-fold count")
    if any(item["dates"] != folds[0]["dates"] for item in folds):
        raise ValueError("evaluation files do not use identical observation dates")
    if any(item["background_day_offset"] != -1 for item in folds):
        raise ValueError("rotated timing gate requires background_day_offset=-1")
    withheld_ids = [station for item in folds for station in item["withheld_ids"]]
    if len(withheld_ids) != len(set(withheld_ids)):
        raise ValueError("a BMD station is withheld in more than one fold")
    station_count = folds[0]["station_count"]
    if any(item["station_count"] != station_count for item in folds):
        raise ValueError("evaluation files disagree on total station count")
    if len(withheld_ids) != station_count:
        raise ValueError(
            f"rotated folds withhold {len(withheld_ids)} unique stations; expected {station_count}"
        )

    weights = np.array([item["station_days"] for item in folds], dtype=float)
    fold_labels = [f"Fold {item['fold'] + 1}" for item in folds]

    def metric(method: str, key: str) -> np.ndarray:
        return np.array([item["methods"][method].get(key, np.nan) for item in folds], dtype=float)

    aggregate = {}
    for method in METHODS:
        rmse = metric(method, "rmse_mm")
        correlation = np.clip(metric(method, "correlation"), -0.999999, 0.999999)
        fisher_weight = np.maximum(weights - 3.0, 1.0)
        aggregate[method] = {
            "crps_mm": weighted_mean(metric(method, "crps_mm"), weights),
            "rmse_mm": float(np.sqrt(weighted_mean(rmse**2, weights))),
            "mae_mm": weighted_mean(metric(method, "mae_mm"), weights),
            "bias_mm": weighted_mean(metric(method, "bias_mm"), weights),
            "correlation_fisher_pooled": float(
                np.tanh(weighted_mean(np.arctanh(correlation), fisher_weight))
            ),
            "spread_skill_fold_mean": weighted_mean(metric(method, "spread_skill"), weights),
            "coverage_90": weighted_mean(metric(method, "coverage_90"), weights),
            "fold_crps_mean_mm": float(np.mean(metric(method, "crps_mm"))),
            "fold_crps_sd_mm": float(np.std(metric(method, "crps_mm"), ddof=1)),
            "brier_score": {
                threshold: weighted_mean(
                    np.array(
                        [
                            item["methods"][method]["thresholds"][threshold]["brier_score"]
                            for item in folds
                        ],
                        dtype=float,
                    ),
                    weights,
                )
                for threshold in ("1", "10", "25", "50")
            },
        }

    gauge_crps = metric("Gauges only", "crps_mm")
    simultaneous_crps = metric("Simultaneous", "crps_mm")
    fold_difference = simultaneous_crps - gauge_crps
    pooled_difference = aggregate["Simultaneous"]["crps_mm"] - aggregate["Gauges only"]["crps_mm"]
    folds_won = int(np.sum(fold_difference < 0))
    required_wins = math.ceil(expected / 2)
    passes = bool(pooled_difference < 0 and folds_won >= required_wins)
    recommendation = "Simultaneous" if passes else "Gauges only"

    summary = {
        "experiment": "offset-minus-one five-day rotated BMD spatial holdout",
        "primary_reference": "BMD gauges withheld once each across disjoint folds",
        "observation_dates": folds[0]["dates"],
        "background_day_offset": -1,
        "fold_count": expected,
        "withheld_station_count": len(withheld_ids),
        "station_days": int(weights.sum()),
        "active_methods": list(METHODS),
        "retired_method": "IMERG -> gauges sequential update",
        "folds": [
            {
                "fold": item["fold"],
                "station_days": item["station_days"],
                "withheld_ids": item["withheld_ids"],
                "withheld_names": item["withheld_names"],
                "crps_mm": {
                    method: float(item["methods"][method]["crps_mm"])
                    for method in METHODS
                },
            }
            for item in folds
        ],
        "aggregate_metrics": aggregate,
        "fusion_gate": {
            "simultaneous_minus_gauges_crps_by_fold_mm": fold_difference.tolist(),
            "pooled_difference_mm": float(pooled_difference),
            "simultaneous_fold_wins": folds_won,
            "required_fold_wins": required_wins,
            "passes": passes,
            "rule": "pooled CRPS must be lower and simultaneous must win at least half the folds",
        },
        "provisional_recommendation": recommendation,
        "caveat": "Five 2018 days are still inside checkpoint training; extend the winning configuration to a longer period and an excluded year.",
    }

    x = np.arange(expected)
    figure, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    for method in METHODS:
        axes[0, 0].plot(x, metric(method, "crps_mm"), marker="o", color=COLOURS[method], label=method)
    axes[0, 0].set_title("A. Withheld-BMD CRPS by spatial fold")
    axes[0, 0].set_ylabel("CRPS (mm day$^{-1}$)")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].bar(x, fold_difference, color=np.where(fold_difference < 0, "#2A9D8F", "#D1495B"))
    axes[0, 1].axhline(0, color="black", lw=1)
    axes[0, 1].set_title("B. Fusion gate by fold")
    axes[0, 1].set_ylabel("Simultaneous − gauges CRPS")

    aggregate_crps = [aggregate[method]["crps_mm"] for method in METHODS]
    aggregate_sd = [aggregate[method]["fold_crps_sd_mm"] for method in METHODS]
    axes[0, 2].bar(METHODS, aggregate_crps, yerr=aggregate_sd, color=[COLOURS[m] for m in METHODS], capsize=4)
    axes[0, 2].set_title("C. Pooled CRPS with fold variability")
    axes[0, 2].set_ylabel("CRPS (mm day$^{-1}$)")
    axes[0, 2].tick_params(axis="x", rotation=25)

    deterministic_keys = ("rmse_mm", "mae_mm", "bias_mm", "correlation_fisher_pooled")
    matrix = np.array([[aggregate[method][key] for key in deterministic_keys] for method in METHODS])
    image = axes[1, 0].imshow(matrix, cmap="Blues", aspect="auto")
    axes[1, 0].set_xticks(np.arange(4), ["RMSE", "MAE", "Bias", "Corr"])
    axes[1, 0].set_yticks(np.arange(4), METHODS)
    for row in range(4):
        for column in range(4):
            axes[1, 0].text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center")
    figure.colorbar(image, ax=axes[1, 0], fraction=0.05)
    axes[1, 0].set_title("D. Pooled deterministic metrics")

    for method in METHODS:
        axes[1, 1].plot(x, metric(method, "coverage_90"), marker="o", color=COLOURS[method], label=method)
    axes[1, 1].axhline(0.9, color="black", ls="--", lw=1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title("E. 90% interval coverage by fold")
    axes[1, 1].set_ylabel("Empirical coverage")

    thresholds = ("1", "10", "25")
    width = 0.18
    for position, method in enumerate(METHODS):
        values = [aggregate[method]["brier_score"][threshold] for threshold in thresholds]
        axes[1, 2].bar(np.arange(len(thresholds)) + (position - 1.5) * width, values, width, color=COLOURS[method], label=method)
    axes[1, 2].set_xticks(np.arange(len(thresholds)), [f"≥{value} mm" for value in thresholds])
    axes[1, 2].set_title("F. Pooled event Brier score")
    axes[1, 2].set_ylabel("Brier score (lower is better)")
    axes[1, 2].legend(fontsize=7)

    for axis in axes.flat:
        axis.grid(alpha=0.2)
    for axis in (axes[0, 0], axes[0, 1], axes[1, 1]):
        axis.set_xticks(x, fold_labels, rotation=20)
    figure.suptitle(
        "Offset −1 DA method gate across disjoint spatial BMD folds\n"
        f"Every station withheld once; recommendation: {recommendation}",
        fontsize=15,
    )
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_plot).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(
        json.dumps(json_safe(summary), indent=2, allow_nan=False) + "\n"
    )
    figure.savefig(args.out_plot, dpi=180)
    plt.close(figure)
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_plot}")
    print(
        f"fusion gate: simultaneous - gauges = {pooled_difference:+.3f} mm; "
        f"fold wins {folds_won}/{expected}; recommendation {recommendation}"
    )


if __name__ == "__main__":
    main()
