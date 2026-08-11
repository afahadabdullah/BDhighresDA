#!/usr/bin/env python
"""Summarize the matched CPC-v2 gauge/IMERG/simultaneous ingestion triplet.

Every comparison is paired by day, station, fold, prior seed and ensemble
member.  Uncertainty is bootstrapped in circular day blocks so spatially
correlated stations from one weather day are never treated as independent.
Positive CRPS differences mean the candidate named before ``_vs_`` is better.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "_v2_gauge_summary", ROOT / "scripts" / "49_summarize_v2_gauge_sweep.py"
)
_summary = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_summary)

GAUGES = "guided_s6_t100"
IMERG = "v2_imerg_s04_t100"
SIMULTANEOUS = "v2_simultaneous_s04_t100"
EXPECTED = ["background", GAUGES, IMERG, SIMULTANEOUS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dumps", nargs="+", required=True)
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--block-days", type=int, default=3)
    parser.add_argument("--n-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=202209)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-markdown", required=True)
    parser.add_argument("--out-plot", required=True)
    return parser.parse_args()


def comparison(
    samples: dict[str, dict], candidate: str, reference: str,
    block_days: int, n_resamples: int, seed: int,
) -> dict:
    """Positive difference means ``candidate`` has lower CRPS."""
    result = _summary.circular_block_bootstrap(
        samples[reference]["crps"] - samples[candidate]["crps"],
        block_days, n_resamples, seed,
    )
    result.update({"candidate": candidate, "reference": reference})
    return result


def upsample_footprints(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Nearest-block expansion used only for same-scale pattern diagnostics."""
    if values.ndim != 3:
        raise ValueError(f"expected (day, y, x) IMERG, got {values.shape}")
    fy, fx = shape[0] // values.shape[1], shape[1] // values.shape[2]
    if fy * values.shape[1] != shape[0] or fx * values.shape[2] != shape[1]:
        raise ValueError(f"IMERG shape {values.shape[1:]} does not nest in {shape}")
    return np.repeat(np.repeat(values, fy, axis=1), fx, axis=2)


def imerg_pattern_correlation(folds: list[dict], name: str) -> float | None:
    values = []
    for item in folds:
        dump = item["dump"]
        if "raw_imerg_mm" not in dump:
            continue
        field = np.asarray(dump[f"meanfield_{name}"], float)
        reference = upsample_footprints(
            np.asarray(dump["raw_imerg_mm"], float), field.shape[-2:]
        )
        value = _summary.daily_pattern_correlation(
            field, reference, np.asarray(dump["valid"], bool)
        )
        if np.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else None


def verdict(direct: dict) -> str:
    if direct["ci_low"] > 0:
        return "satellite_helps"
    if direct["ci_high"] < 0:
        return "satellite_hurts"
    return "unresolved"


def fmt_interval(result: dict) -> str:
    return (
        f"{result['difference']:+.3f} "
        f"[{result['ci_low']:+.3f}, {result['ci_high']:+.3f}]"
    )


def main() -> None:
    args = parse_args()
    folds = _summary.validate_and_load(
        [Path(path) for path in args.dumps],
        [Path(path) for path in args.reports],
    )
    names = folds[0]["dump"]["variant_names"].astype(str).tolist()
    if names != EXPECTED:
        raise ValueError(f"expected matched triplet {EXPECTED}, got {names}")

    metrics, samples = {}, {}
    for name in names:
        metrics[name], samples[name] = _summary.pooled_variant(folds, name)
        metrics[name]["imerg_pattern_correlation"] = imerg_pattern_correlation(
            folds, name
        )

    pair_specs = [
        ("gauges_vs_background", GAUGES, "background"),
        ("imerg_vs_background", IMERG, "background"),
        ("simultaneous_vs_background", SIMULTANEOUS, "background"),
        ("simultaneous_vs_gauges", SIMULTANEOUS, GAUGES),
        ("simultaneous_vs_imerg", SIMULTANEOUS, IMERG),
    ]
    comparisons = {
        key: comparison(
            samples, candidate, reference, args.block_days,
            args.n_resamples, args.seed + index * 10_000,
        )
        for index, (key, candidate, reference) in enumerate(pair_specs)
    }
    direct = comparisons["simultaneous_vs_gauges"]
    decision = verdict(direct)
    ordered = sorted(names, key=lambda name: metrics[name]["crps"])
    scope = folds[0]["report"]["scope"]

    lines = [
        "# CPC-v2 BMD/IMERG ingestion triplet",
        "",
        f"- Period: **{scope['start']} to {scope['end']}**",
        f"- Members: **{scope['members']}**; five disjoint BMD spatial folds",
        "- IMERG: **S04, 0.4-degree footprints, stride 1**, raw V07B",
        "- Simultaneous gradient: spread gauges by 6 cells; do not re-spread IMERG",
        f"- Day-block bootstrap: **{args.block_days} days**, "
        f"{args.n_resamples:,} resamples",
        "",
        "| Method | CRPS | MAE dry/wet | Bias | Corr | Cov90 | CPC r | IMERG r | Wet area |",
        "|:--|--:|:--|--:|--:|--:|--:|--:|--:|",
    ]
    for name in ordered:
        metric = metrics[name]
        imerg_r = metric["imerg_pattern_correlation"]
        lines.append(
            f"| `{name}` | {metric['crps']:.3f} | "
            f"{metric['dry_mae']:.2f}/{metric['wet_mae']:.2f} | "
            f"{metric['bias']:+.2f} | {metric['correlation']:.3f} | "
            f"{metric['coverage_90']:.2f} | "
            f"{metric['cpc_pattern_correlation']:.3f} | "
            f"{'—' if imerg_r is None else f'{imerg_r:.3f}'} | "
            f"{metric['wet_area']:.3f} |"
        )
    lines += [
        "",
        "## Paired CRPS tests",
        "",
        *(
            f"- `{key}`: {fmt_interval(value)} mm/day"
            for key, value in comparisons.items()
        ),
        "",
        "Positive values favour the method before `_vs_`.",
        "",
        "## Primary decision",
        "",
        f"**{decision.replace('_', ' ').title()}**: simultaneous minus gauges-only "
        f"is represented as CRPS(gauges) − CRPS(simultaneous) = "
        f"{fmt_interval(direct)} mm/day.",
    ]

    output = {
        "scope": {
            "start": scope["start"], "end": scope["end"],
            "members": scope["members"], "folds": len(folds),
            "checkpoint": scope["checkpoint"],
            "group": scope.get("group"),
            "imerg_configuration": {
                "tag": "S04", "footprint_degrees": 0.4,
                "factor_model_cells": 8, "stride": 1,
                "error_corr_cells": 0.75, "bias_correction": False,
            },
            "gauge_configuration": {
                "spread_cells": 6.0, "prior_temperature": 1.0,
            },
            "block_days": args.block_days,
            "n_resamples": args.n_resamples,
        },
        "ranked_by_crps": ordered,
        "metrics": metrics,
        "comparisons": comparisons,
        "primary_verdict": decision,
    }

    out_json = Path(args.out_json)
    out_markdown = Path(args.out_markdown)
    out_plot = Path(args.out_plot)
    for path in (out_json, out_markdown, out_plot):
        path.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    out_markdown.write_text("\n".join(lines) + "\n")

    display_names = [
        "background", "gauges only", "IMERG only", "simultaneous"
    ]
    positions = np.arange(len(names))
    figure, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    axes[0, 0].barh(positions, [metrics[n]["crps"] for n in names], color="#C1440E")
    axes[0, 0].set_yticks(positions, display_names); axes[0, 0].invert_yaxis()
    axes[0, 0].set_xlabel("CRPS (mm/day)"); axes[0, 0].set_title("A. Withheld gauges")

    pair_names = [key for key, _, _ in pair_specs]
    centres = np.array([comparisons[key]["difference"] for key in pair_names])
    lows = np.array([comparisons[key]["ci_low"] for key in pair_names])
    highs = np.array([comparisons[key]["ci_high"] for key in pair_names])
    pair_positions = np.arange(len(pair_names))
    axes[0, 1].errorbar(
        centres, pair_positions,
        xerr=np.vstack([centres - lows, highs - centres]),
        fmt="o", color="#1B4965", capsize=3,
    )
    axes[0, 1].axvline(0, color="black", ls="--")
    axes[0, 1].set_yticks(pair_positions, [name.replace("_", " ") for name in pair_names], fontsize=8)
    axes[0, 1].invert_yaxis(); axes[0, 1].set_xlabel("CRPS(reference) − CRPS(candidate)")
    axes[0, 1].set_title("B. Paired day-block uncertainty")

    width = 0.38
    axes[0, 2].barh(positions - width / 2, [metrics[n]["dry_mae"] for n in names],
                    height=width, label="dry", color="#E9C46A")
    axes[0, 2].barh(positions + width / 2, [metrics[n]["wet_mae"] for n in names],
                    height=width, label="wet", color="#2A9D8F")
    axes[0, 2].set_yticks(positions, display_names); axes[0, 2].invert_yaxis()
    axes[0, 2].set_xlabel("MAE (mm/day)"); axes[0, 2].legend()
    axes[0, 2].set_title("C. Dry/wet trade-off")

    axes[1, 0].barh(positions, [metrics[n]["bias"] for n in names], color="#D1495B")
    axes[1, 0].axvline(0, color="black")
    axes[1, 0].set_yticks(positions, display_names); axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlabel("Bias (mm/day)"); axes[1, 0].set_title("D. Mean bias")

    axes[1, 1].barh(positions, [metrics[n]["correlation"] for n in names], color="#457B9D")
    axes[1, 1].set_yticks(positions, display_names); axes[1, 1].invert_yaxis()
    axes[1, 1].set_xlabel("Correlation"); axes[1, 1].set_title("E. Gauge correlation")

    axes[1, 2].barh(positions - width / 2,
                    [metrics[n]["cpc_pattern_correlation"] for n in names],
                    height=width, label="CPC", color="#F4A261")
    axes[1, 2].barh(positions + width / 2,
                    [metrics[n]["imerg_pattern_correlation"] for n in names],
                    height=width, label="IMERG", color="#6A4C93")
    axes[1, 2].set_yticks(positions, display_names); axes[1, 2].invert_yaxis()
    axes[1, 2].set_xlabel("Mean daily spatial correlation"); axes[1, 2].legend()
    axes[1, 2].set_title("F. Product pattern agreement")

    figure.suptitle("CPC-v2: gauges, IMERG S04, and simultaneous ingestion")
    figure.savefig(out_plot, dpi=160)
    plt.close(figure)

    print("\n".join(lines))
    print(f"\n[done] wrote {out_json}")
    print(f"[done] wrote {out_markdown}")
    print(f"[done] wrote {out_plot}")


if __name__ == "__main__":
    main()
