#!/usr/bin/env python3
"""Rank the matched V7 native/S04 IMERG ingestion arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


METRICS = ("crps_mm", "mae_mm", "bias_mm", "rmse_mm", "spread_mm", "spread_skill")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="v7_two_stage_real.json")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-markdown", required=True)
    parser.add_argument("--out-plot", required=True)
    return parser.parse_args()


def finite_float(value) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def build_summary(results: dict) -> dict:
    arms = results["arms"]
    simultaneous = [name for name in arms if name == "da_sim" or name.startswith("da_sim_")]
    if not simultaneous:
        raise ValueError("result contains no simultaneous IMERG + gauge arms")
    r_map = results.get("arm_imerg_r", {})
    stream_map = results.get("arm_imerg_stream", {})
    gamma_map = results.get("arm_guidance_gamma", {})
    huber_map = results.get("arm_huber_delta", {})
    default_r = float(results["imerg_r_multiplier"])

    rows = []
    for name in simultaneous:
        mean = arms[name]["mean"]
        stream = stream_map.get(name, "native")
        daily = [finite_float(day["withheld"].get("crps_mm", np.nan))
                 for day in arms[name]["days"]]
        rows.append({
            "arm": name,
            "support_deg": 0.4 if stream == "s04" else 0.1,
            "imerg_stream": stream,
            "r_multiplier": float(r_map.get(name, default_r)),
            "gamma": float(gamma_map.get(name, 1.0e-3)),
            "loss": ("L2" if huber_map.get(name, 3.0) is None
                     else f"Huber-{float(huber_map.get(name, 3.0)):g}"),
            "metrics": {metric: finite_float(mean.get(metric, np.nan)) for metric in METRICS},
            "daily_crps_mm": daily,
            "pattern_r": {
                key: finite_float(value)
                for key, value in arms[name].get("pattern_r", {}).items()
            },
        })
    rows.sort(key=lambda row: row["metrics"]["crps_mm"])
    best = rows[0]
    tolerance = 0.01 * best["metrics"]["crps_mm"]
    co_winners = [
        row["arm"] for row in rows
        if row["metrics"]["crps_mm"] - best["metrics"]["crps_mm"] <= tolerance
    ]
    gauges = arms.get("da_meso", {}).get("mean", {})
    background = arms.get("background", {}).get("mean", {})
    winners_by_support = {
        f"{support:.1f}": min(
            (row for row in rows if row["support_deg"] == support),
            key=lambda row: row["metrics"]["crps_mm"],
        )["arm"]
        for support in sorted({row["support_deg"] for row in rows})
    }
    gauges_crps = finite_float(gauges.get("crps_mm", np.nan))
    overall_winner = (
        "da_meso" if gauges_crps is not None
        and gauges_crps < best["metrics"]["crps_mm"] else best["arm"]
    )
    return {
        "scope": {
            "model_dates": results.get("model_dates"),
            "gauge_dates": results.get("gauge_dates"),
            "members": results.get("members"),
            "n_steps": results.get("n_steps"),
            "checkpoints": results.get("checkpoints"),
            "selection_rule": (
                "lowest mean daily withheld-gauge CRPS; arms within 1% of the "
                "minimum are retained as practically tied on this short window"
            ),
        },
        "winner": best["arm"],
        "overall_da_winner": overall_winner,
        "winner_by_support_deg": winners_by_support,
        "co_winners_within_1pct": co_winners,
        "winner_minus_gauges_crps_mm": (
            best["metrics"]["crps_mm"] - gauges_crps
            if gauges_crps is not None else None
        ),
        "gauges_only": {metric: finite_float(gauges.get(metric, np.nan)) for metric in METRICS},
        "background": {metric: finite_float(background.get(metric, np.nan)) for metric in METRICS},
        "ranking": rows,
    }


def markdown(summary: dict) -> str:
    lines = [
        "# V7 IMERG ingestion-scale sweep",
        "",
        f"Best simultaneous arm by withheld-gauge CRPS: `{summary['winner']}`.",
        f"Overall DA winner including gauges-only: `{summary['overall_da_winner']}`.",
        "Best native 0.1° arm: "
        f"`{summary['winner_by_support_deg'].get('0.1', 'not run')}`; "
        "best S04 0.4° arm: "
        f"`{summary['winner_by_support_deg'].get('0.4', 'not run')}`.",
        "",
        "Practically tied within 1%: "
        + ", ".join(f"`{name}`" for name in summary["co_winners_within_1pct"])
        + ".",
        "",
        "| rank | arm | support | R multiplier | gamma | loss | CRPS | MAE | bias | RMSE | spread/RMSE |",
        "|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(summary["ranking"], 1):
        metric = row["metrics"]
        lines.append(
            f"| {rank} | `{row['arm']}` | {row['support_deg']:.1f}° | "
            f"{row['r_multiplier']:.3g} | {row['gamma']:.4g} | {row['loss']} | "
            f"{metric['crps_mm']:.3f} | {metric['mae_mm']:.3f} | "
            f"{metric['bias_mm']:+.3f} | {metric['rmse_mm']:.3f} | "
            f"{metric['spread_skill']:.2f} |"
        )
    delta = summary["winner_minus_gauges_crps_mm"]
    lines.extend([
        "",
        (f"Winning simultaneous arm minus gauges-only CRPS: {delta:+.3f} mm."
         if delta is not None else "Gauges-only CRPS was unavailable."),
        "",
        "Selection caveat: " + summary["scope"]["selection_rule"] + ".",
        "",
    ])
    return "\n".join(lines)


def plot(summary: dict, path: Path) -> None:
    rows = summary["ranking"]
    labels = [row["arm"].replace("da_sim_", "") for row in rows]
    colours = ["#D55E00" if row["support_deg"] == 0.4 else "#0072B2" for row in rows]
    figure, axes = plt.subplots(1, 3, figsize=(16, max(5, 0.48 * len(rows))))
    y = np.arange(len(rows))
    for axis, key, title in (
        (axes[0], "crps_mm", "CRPS (lower is better)"),
        (axes[1], "rmse_mm", "RMSE (lower is better)"),
        (axes[2], "spread_skill", "Spread / RMSE (target 1)"),
    ):
        axis.barh(y, [row["metrics"][key] for row in rows], color=colours)
        axis.set_yticks(y, labels if axis is axes[0] else [])
        axis.invert_yaxis()
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.25)
    axes[2].axvline(1.0, color="black", ls="--", lw=1)
    figure.suptitle("V7-only native 0.1° vs S04 0.4° IMERG ingestion sweep")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    summary = build_summary(json.loads(Path(args.input).read_text()))
    json_path = Path(args.out_json)
    markdown_path = Path(args.out_markdown)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2, allow_nan=False))
    markdown_path.write_text(markdown(summary))
    plot(summary, Path(args.out_plot))
    print(f"[winner] {summary['winner']}")
    print(f"[co-winners within 1%] {', '.join(summary['co_winners_within_1pct'])}")
    print(f"[done] {markdown_path}")


if __name__ == "__main__":
    main()
