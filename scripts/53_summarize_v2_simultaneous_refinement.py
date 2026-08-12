#!/usr/bin/env python
"""Rank new CPC-v2 simultaneous arms against the completed S04 controls.

The refinement folds contain only new simultaneous configurations (plus the
automatically generated background).  The gauges-only and operational
simultaneous controls are read from the completed ingestion-triplet folds, so
the expensive controls are not rerun.  Cross-file checks enforce identical
dates, stations, holdouts, checkpoint, ensemble size, seeds and background.

CRPS uncertainty uses paired circular day blocks.  Positive differences mean
the candidate named before ``_vs_`` has lower CRPS than the reference.
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
_summary_spec = importlib.util.spec_from_file_location(
    "_v2_gauge_summary", ROOT / "scripts" / "49_summarize_v2_gauge_sweep.py"
)
_summary = importlib.util.module_from_spec(_summary_spec)
_summary_spec.loader.exec_module(_summary)


BACKGROUND = "background"
GAUGES = "guided_s6_g010_t100"
IMERG = "v2_imerg_s04_t100"
CURRENT = "v2_simultaneous_s04_t100"
REFERENCE_NAMES = [
    BACKGROUND,
    GAUGES,
    IMERG,
    CURRENT,
]
NEW_ARMS = [
    "v2_simul_s04_iw050",
    "v2_simul_s04_iw075",
    "v2_simul_s04_ig003",
    "v2_simul_s04_ig010",
    "v2_simul_s04_gw125",
    "v2_simul_s04_huber3",
    "v2_simul_s04_gap050",
    "v2_simul_s04_gap100",
    "v2_simul_s04_nc0_n025",
    "v2_simul_s04_nc0_n050",
    "v2_simul_s04_nc0_n100",
    "v2_simul_s04_n100",
]
CANDIDATE_NAMES = [BACKGROUND, *NEW_ARMS]
ODE_ARMS = {
    25: "v2_simul_s04_nc0_n025",
    50: "v2_simul_s04_nc0_n050",
    100: "v2_simul_s04_nc0_n100",
}
OPERATIONAL_N100 = "v2_simul_s04_n100"
ALL_METHODS = [BACKGROUND, GAUGES, IMERG, CURRENT, *NEW_ARMS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dumps", nargs="+", required=True)
    parser.add_argument("--reference-reports", nargs="+", required=True)
    parser.add_argument("--candidate-dumps", nargs="+", required=True)
    parser.add_argument("--candidate-reports", nargs="+", required=True)
    parser.add_argument("--block-days", type=int, default=3)
    parser.add_argument("--n-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=202210)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-markdown", required=True)
    parser.add_argument("--out-plot", required=True)
    parser.add_argument("--fold-plot-dir", required=True)
    return parser.parse_args()


def _names(folds: list[dict]) -> list[str]:
    return folds[0]["dump"]["variant_names"].astype(str).tolist()


def validate_cross_run_pairing(
    reference: list[dict], candidates: list[dict]
) -> None:
    """Require candidate and control folds to be exchangeable member by member."""
    if _names(reference) != REFERENCE_NAMES:
        raise ValueError(
            f"expected reference variants {REFERENCE_NAMES}, got {_names(reference)}"
        )
    if _names(candidates) != CANDIDATE_NAMES:
        raise ValueError(
            f"expected candidate variants {CANDIDATE_NAMES}, got {_names(candidates)}"
        )
    if len(reference) != len(candidates):
        raise ValueError("reference and candidate runs contain different fold counts")

    for ref, cand in zip(reference, candidates):
        if ref["fold"] != cand["fold"]:
            raise ValueError("reference and candidate fold numbers do not align")
        ref_scope, cand_scope = ref["report"]["scope"], cand["report"]["scope"]
        for key in (
            "start", "end", "members", "checkpoint", "checkpoint_data",
            "checkpoint_stats", "background_day_offset", "seed",
            "holdout_folds", "holdout_fold", "precip_transform",
            "config_overrides",
        ):
            if ref_scope.get(key) != cand_scope.get(key):
                raise ValueError(
                    f"fold {ref['fold']} differs on {key}: "
                    f"{ref_scope.get(key)!r} versus {cand_scope.get(key)!r}"
                )
        for key in ("times", "station_ids", "eval_idx", "assim_idx", "valid"):
            if not np.array_equal(ref["dump"][key], cand["dump"][key]):
                raise ValueError(f"fold {ref['fold']} differs on {key}")
        for key in (
            "station_lat", "station_lon", "grid_lat", "grid_lon", "gauge_mm",
            "condition", "chirps", "raw_imerg_mm", "station_background",
            "meanfield_background",
        ):
            if not np.allclose(
                ref["dump"][key], cand["dump"][key],
                rtol=0.0, atol=1.0e-6, equal_nan=True,
            ):
                raise ValueError(
                    f"fold {ref['fold']} paired inputs differ in {key}; paired "
                    "comparisons would mix code, seed, or sampler changes"
                )


def source_for(
    name: str, reference: list[dict], candidates: list[dict]
) -> list[dict]:
    return reference if name in {BACKGROUND, GAUGES, IMERG, CURRENT} else candidates


def comparison(
    samples: dict[str, dict], candidate: str, reference: str,
    block_days: int, n_resamples: int, seed: int,
) -> dict:
    result = _summary.circular_block_bootstrap(
        samples[reference]["crps"] - samples[candidate]["crps"],
        block_days, n_resamples, seed,
    )
    result.update({"candidate": candidate, "reference": reference})
    return result


def upsample_footprints(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    fy, fx = shape[0] // values.shape[1], shape[1] // values.shape[2]
    if fy * values.shape[1] != shape[0] or fx * values.shape[2] != shape[1]:
        raise ValueError(f"IMERG shape {values.shape[1:]} does not nest in {shape}")
    return np.repeat(np.repeat(values, fy, axis=1), fx, axis=2)


def imerg_pattern_correlation(folds: list[dict], name: str) -> float | None:
    values = []
    for item in folds:
        dump = item["dump"]
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


def method_spec(folds: list[dict], name: str) -> dict:
    return folds[0]["report"]["variants"][name]["spec"]


def effective_sampler(folds: list[dict], name: str) -> dict:
    scope = folds[0]["report"]["scope"]
    spec = method_spec(folds, name)
    n_steps = spec.get("n_steps")
    n_corrections = spec.get("n_corrections")
    if name == BACKGROUND:
        # The completed reference reports predate explicit sampler metadata,
        # but configs/da.yaml has always used 50/0 for this v2 background.
        n_steps = int(scope.get("background_sampler_n_steps", 50))
        n_corrections = int(scope.get("background_sampler_n_corrections", 0))
    if n_steps is None:
        n_steps = int(scope.get("analysis_sampler_n_steps", 50))
    if n_corrections is None:
        n_corrections = int(scope.get("analysis_sampler_n_corrections", 2))
    heun = bool(scope.get("analysis_sampler_heun", True))
    integration_evaluations = 2 * n_steps - 1 if heun else n_steps
    corrector_evaluations = n_corrections * (n_steps - 1)
    return {
        "n_steps": int(n_steps),
        "n_corrections_per_level": int(n_corrections),
        "heun": heun,
        "integration_guidance_evaluations": int(integration_evaluations),
        "corrector_guidance_evaluations": int(corrector_evaluations),
        "total_guidance_evaluations": int(
            integration_evaluations + corrector_evaluations
        ),
    }


def corrector_stability(folds: list[dict], name: str) -> dict:
    sampler = effective_sampler(folds, name)
    if sampler["n_corrections_per_level"] == 0:
        return {
            "enabled": False,
            "member_steps": 0,
            "capped_member_steps": 0,
            "capped_fraction": 0.0,
            "max_raw_step": None,
            "max_applied_step": None,
        }
    entries = []
    for item in folds:
        entries.extend(
            item["report"].get("sampler_diagnostics", {}).get(name, [])
        )
    correctors = [entry.get("corrector", {}) for entry in entries]
    correctors = [entry for entry in correctors if entry.get("member_steps")]
    if not correctors:
        raise ValueError(f"{name}: correctors are enabled but diagnostics are absent")
    member_steps = sum(int(entry["member_steps"]) for entry in correctors)
    capped = sum(int(entry["capped_member_steps"]) for entry in correctors)
    return {
        "enabled": True,
        "member_steps": member_steps,
        "capped_member_steps": capped,
        "capped_fraction": capped / member_steps,
        "max_raw_step": max(float(entry["max_raw_step"]) for entry in correctors),
        "max_applied_step": max(
            float(entry["max_applied_step"]) for entry in correctors
        ),
    }


def fmt_interval(result: dict) -> str:
    return (
        f"{result['difference']:+.3f} "
        f"[{result['ci_low']:+.3f}, {result['ci_high']:+.3f}]"
    )


def fmt_metric(value: float | None, digits: int = 2, signed: bool = False) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    sign = "+" if signed else ""
    return format(float(value), f"{sign}.{digits}f")


def plot_metric(value: float | None) -> float:
    return float(value) if value is not None and np.isfinite(value) else np.nan


def time_mean(field: np.ndarray) -> np.ndarray:
    finite = np.isfinite(field)
    count = finite.sum(axis=0)
    total = np.where(finite, field, 0.0).sum(axis=0)
    return np.divide(total, count, out=np.full(total.shape, np.nan), where=count > 0)


def fold_item(folds: list[dict], fold: int) -> dict:
    return next(item for item in folds if item["fold"] == fold)


def plot_fold_diagnostics(
    fold: int,
    reference: list[dict],
    candidates: list[dict],
    best_new: str,
    out_path: Path,
) -> None:
    selected = [BACKGROUND, GAUGES, CURRENT, best_new]
    items = {
        name: fold_item(source_for(name, reference, candidates), fold)
        for name in selected
    }
    dumps = {name: item["dump"] for name, item in items.items()}
    control_dump = dumps[BACKGROUND]
    valid = np.asarray(control_dump["valid"], bool)
    means = {
        name: time_mean(np.asarray(dumps[name][f"meanfield_{name}"], float))
        for name in selected
    }
    fold_metrics = {
        name: _summary.pooled_variant([items[name]], name)[0]
        for name in selected
    }
    display = {
        BACKGROUND: "background",
        GAUGES: "gauges s6/g0.01",
        CURRENT: "current simultaneous",
        best_new: best_new.replace("v2_simul_s04_", ""),
    }
    rain = np.concatenate([means[name][valid] for name in selected])
    rain_top = max(1.0, float(np.nanpercentile(rain, 99.0)))
    increments = {
        CURRENT: means[CURRENT] - means[BACKGROUND],
        best_new: means[best_new] - means[BACKGROUND],
        "difference": means[best_new] - means[CURRENT],
    }
    inc = np.concatenate([np.abs(value[valid]) for value in increments.values()])
    inc_top = max(0.25, float(np.nanpercentile(inc, 99.0)))

    grid_lat = np.asarray(control_dump["grid_lat"], float)
    grid_lon = np.asarray(control_dump["grid_lon"], float)
    extent = [grid_lon[0], grid_lon[-1], grid_lat[0], grid_lat[-1]]
    station_lat = np.asarray(control_dump["station_lat"], float)
    station_lon = np.asarray(control_dump["station_lon"], float)
    assim_idx = np.asarray(control_dump["assim_idx"], int)
    eval_idx = np.asarray(control_dump["eval_idx"], int)

    figure, axes = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)
    for column, name in enumerate(selected):
        axis = axes[0, column]
        image = axis.imshow(
            np.where(valid, means[name], np.nan), origin="lower", extent=extent,
            cmap="viridis", vmin=0.0, vmax=rain_top, aspect="auto",
        )
        axis.scatter(station_lon[assim_idx], station_lat[assim_idx], s=6, c="black")
        axis.scatter(
            station_lon[eval_idx], station_lat[eval_idx], s=22,
            facecolors="none", edgecolors="cyan", linewidths=0.8,
        )
        metric = fold_metrics[name]
        axis.set_title(
            f"{display[name]}\nCRPS {metric['crps']:.2f}; bias {metric['bias']:+.2f}"
        )
        axis.set_xlabel("longitude")
        if column == 0:
            axis.set_ylabel("latitude")
    figure.colorbar(image, ax=axes[0, :], label="10-day mean rain (mm/day)", shrink=0.8)

    positions = np.arange(len(selected))
    axes[1, 0].barh(
        positions, [fold_metrics[name]["crps"] for name in selected], color="#C1440E"
    )
    axes[1, 0].set_yticks(positions, [display[name] for name in selected], fontsize=8)
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlabel("withheld-gauge CRPS")
    axes[1, 0].set_title("Fold point skill")

    increment_specs = [
        (CURRENT, "current − background"),
        (best_new, "best new − background"),
        ("difference", "best new − current"),
    ]
    for column, (key, title) in enumerate(increment_specs, start=1):
        axis = axes[1, column]
        inc_image = axis.imshow(
            np.where(valid, increments[key], np.nan), origin="lower", extent=extent,
            cmap="RdBu_r", vmin=-inc_top, vmax=inc_top, aspect="auto",
        )
        axis.scatter(station_lon[assim_idx], station_lat[assim_idx], s=6, c="black")
        axis.scatter(
            station_lon[eval_idx], station_lat[eval_idx], s=22,
            facecolors="none", edgecolors="cyan", linewidths=0.8,
        )
        axis.set_title(title)
        axis.set_xlabel("longitude")
    figure.colorbar(
        inc_image, ax=axes[1, 1:], label="mean increment (mm/day)", shrink=0.8
    )
    withheld = np.asarray(control_dump["station_ids"]).astype(str)[eval_idx]
    figure.suptitle(
        f"CPC-v2 simultaneous refinement — fold {fold}; withheld: "
        + ", ".join(withheld.tolist()),
        fontsize=12,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    reference = _summary.validate_and_load(
        [Path(path) for path in args.reference_dumps],
        [Path(path) for path in args.reference_reports],
    )
    candidates = _summary.validate_and_load(
        [Path(path) for path in args.candidate_dumps],
        [Path(path) for path in args.candidate_reports],
    )
    validate_cross_run_pairing(reference, candidates)

    metrics, samples, sampler_specs = {}, {}, {}
    for name in ALL_METHODS:
        folds = source_for(name, reference, candidates)
        metrics[name], samples[name] = _summary.pooled_variant(folds, name)
        metrics[name]["imerg_pattern_correlation"] = imerg_pattern_correlation(
            folds, name
        )
        metrics[name]["corrector_stability"] = corrector_stability(folds, name)
        sampler_specs[name] = effective_sampler(folds, name)

    vs_current = {
        name: comparison(
            samples, name, CURRENT, args.block_days, args.n_resamples,
            args.seed + index * 10_000,
        )
        for index, name in enumerate(ALL_METHODS)
    }
    vs_gauges = {
        name: comparison(
            samples, name, GAUGES, args.block_days, args.n_resamples,
            args.seed + 500_000 + index * 10_000,
        )
        for index, name in enumerate([CURRENT, *NEW_ARMS])
    }
    ode_comparisons = {
        "n025_vs_n050": comparison(
            samples, ODE_ARMS[25], ODE_ARMS[50], args.block_days,
            args.n_resamples, args.seed + 800_000,
        ),
        "n100_vs_n050": comparison(
            samples, ODE_ARMS[100], ODE_ARMS[50], args.block_days,
            args.n_resamples, args.seed + 810_000,
        ),
        "correctors2_vs_correctors0_n050": comparison(
            samples, CURRENT, ODE_ARMS[50], args.block_days,
            args.n_resamples, args.seed + 820_000,
        ),
        "n100c2_vs_n050c2": comparison(
            samples, OPERATIONAL_N100, CURRENT, args.block_days,
            args.n_resamples, args.seed + 830_000,
        ),
    }

    ordered = sorted(ALL_METHODS, key=lambda name: metrics[name]["crps"])
    current_bias_limit = abs(metrics[CURRENT]["bias"]) + 0.5
    eligible = [
        name for name in NEW_ARMS
        if vs_current[name]["difference"] > 0.0
        and abs(metrics[name]["bias"]) <= current_bias_limit
        and metrics[name]["corrector_stability"]["capped_fraction"] <= 0.01
    ]
    promoted = sorted(eligible, key=lambda name: metrics[name]["crps"])[:2]
    best_new = min(NEW_ARMS, key=lambda name: metrics[name]["crps"])

    ode_detected = any(
        ode_comparisons[key]["ci_low"] > 0.0
        or ode_comparisons[key]["ci_high"] < 0.0
        for key in ("n025_vs_n050", "n100_vs_n050")
    )
    scope = reference[0]["report"]["scope"]
    lines = [
        "# CPC-v2 simultaneous DA refinement tournament",
        "",
        f"- Period: **{scope['start']} to {scope['end']}**; five matched folds",
        f"- **{scope['members']} members**; every BMD station withheld exactly once",
        "- Controls are reused from the corrected S04 triplet; candidate/background "
        "pairing is checked numerically",
        f"- Paired circular bootstrap: **{args.block_days}-day blocks**, "
        f"{args.n_resamples:,} resamples",
        "",
        "| Method | CRPS | Δ vs current (95% CI) | MAE dry/wet | Bias | Corr | "
        "Cov90 | CPC r | IMERG r | Wet area |",
        "|:--|--:|:--|:--|--:|--:|--:|--:|--:|--:|",
    ]
    for name in ordered:
        metric = metrics[name]
        delta = vs_current[name]
        sign = "+" if delta["ci_low"] > 0 else (
            "−" if delta["ci_high"] < 0 else "~"
        )
        lines.append(
            f"| `{name}` | {metric['crps']:.3f} | {fmt_interval(delta)} {sign} | "
            f"{fmt_metric(metric['dry_mae'])}/{fmt_metric(metric['wet_mae'])} | "
            f"{fmt_metric(metric['bias'], signed=True)} | "
            f"{fmt_metric(metric['correlation'], 3)} | "
            f"{fmt_metric(metric['coverage_90'])} | "
            f"{fmt_metric(metric['cpc_pattern_correlation'], 3)} | "
            f"{fmt_metric(metric['imerg_pattern_correlation'], 3)} | "
            f"{fmt_metric(metric['wet_area'], 3)} |"
        )
    lines += [
        "",
        "+ = beats the current simultaneous arm across the whole interval; "
        "− = worse; ~ = unresolved. Positive differences favour the candidate.",
        "",
        "## ODE-step test",
        "",
        "The production DA uses **50 Heun steps and two Langevin correctors at each "
        "of 49 interior levels**: 99 integration plus 98 corrector guidance "
        "evaluations, 197 total. The 100-step operational arm uses 397 total. "
        "The isolated convergence arms turn correctors off:",
        "",
        "| Arm | ODE steps | Correctors/level | Integration evals | Corrector evals | Total | CRPS |",
        "|:--|--:|--:|--:|--:|--:|--:|",
    ]
    for name in (CURRENT, OPERATIONAL_N100):
        spec = sampler_specs[name]
        lines.append(
            f"| `{name}` | {spec['n_steps']} | "
            f"{spec['n_corrections_per_level']} | "
            f"{spec['integration_guidance_evaluations']} | "
            f"{spec['corrector_guidance_evaluations']} | "
            f"{spec['total_guidance_evaluations']} | {metrics[name]['crps']:.3f} |"
        )
    for steps, name in ODE_ARMS.items():
        spec = sampler_specs[name]
        lines.append(
            f"| `{name}` | {steps} | {spec['n_corrections_per_level']} | "
            f"{spec['integration_guidance_evaluations']} | "
            f"{spec['corrector_guidance_evaluations']} | "
            f"{spec['total_guidance_evaluations']} | {metrics[name]['crps']:.3f} |"
        )
    lines += [
        "",
        f"- `n025_vs_n050`: {fmt_interval(ode_comparisons['n025_vs_n050'])} mm/day",
        f"- `n100_vs_n050`: {fmt_interval(ode_comparisons['n100_vs_n050'])} mm/day",
        "- `correctors2_vs_correctors0_n050`: "
        f"{fmt_interval(ode_comparisons['correctors2_vs_correctors0_n050'])} mm/day",
        "- `n100c2_vs_n050c2`: "
        f"{fmt_interval(ode_comparisons['n100c2_vs_n050c2'])} mm/day",
        "",
        (
            "A step-count sensitivity was detected in this screen."
            if ode_detected else
            "No step-count sensitivity was resolved in this ten-day screen; this "
            "supports using the cheapest member of the convergence-equivalent set, "
            "but is not a multi-year convergence proof."
        ),
        "",
        "## Promotion decision",
        "",
        "Promotion requires central CRPS improvement over the current simultaneous "
        "arm, absolute bias no more than 0.5 mm/day worse, and no more than 1% "
        "capped corrector member-steps. Promote at most two:",
        "",
        *(f"- `{name}`" for name in promoted),
    ]
    if not promoted:
        lines.append("- None; retain the corrected operational S04 simultaneous arm.")
    lines += [
        "",
        "Ten days are a screening experiment. A promoted arm still needs a longer "
        "pre-registered evaluation before claiming improvement.",
    ]

    out_json = Path(args.out_json)
    out_markdown = Path(args.out_markdown)
    out_plot = Path(args.out_plot)
    fold_plot_dir = Path(args.fold_plot_dir)
    fold_paths = [
        fold_plot_dir / f"fold{fold}_diagnostics.png" for fold in range(len(reference))
    ]
    output = {
        "scope": {
            "start": scope["start"], "end": scope["end"],
            "members": scope["members"], "folds": len(reference),
            "checkpoint": scope["checkpoint"],
            "reference_group": reference[0]["report"]["scope"].get("group"),
            "candidate_group": candidates[0]["report"]["scope"].get("group"),
            "block_days": args.block_days, "n_resamples": args.n_resamples,
        },
        "ranked_by_crps": ordered,
        "metrics": metrics,
        "sampler_costs": sampler_specs,
        "vs_current_simultaneous": vs_current,
        "vs_selected_gauges": vs_gauges,
        "ode_step_test": {
            "no_corrector_arms": {
                str(steps): name for steps, name in ODE_ARMS.items()
            },
            "operational_arms": {
                "50": CURRENT,
                "100": OPERATIONAL_N100,
            },
            "comparisons": ode_comparisons,
            "step_sensitivity_detected": ode_detected,
        },
        "promotion": {
            "bias_tolerance_mm": 0.5,
            "max_corrector_capped_fraction": 0.01,
            "eligible": eligible,
            "promoted": promoted,
        },
        "artifacts": {
            "pooled_plot": str(out_plot),
            "fold_plots": [str(path) for path in fold_paths],
        },
    }
    for path in (out_json, out_markdown, out_plot, fold_plot_dir):
        path.parent.mkdir(parents=True, exist_ok=True)
    fold_plot_dir.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    out_markdown.write_text("\n".join(lines) + "\n")

    for fold, path in enumerate(fold_paths):
        plot_fold_diagnostics(fold, reference, candidates, best_new, path)

    positions = np.arange(len(ordered))
    labels = [name.replace("v2_simul_s04_", "") for name in ordered]
    figure, axes = plt.subplots(2, 3, figsize=(21, 12), constrained_layout=True)
    axes[0, 0].barh(positions, [metrics[name]["crps"] for name in ordered], color="#C1440E")
    axes[0, 0].set_yticks(positions, labels, fontsize=8)
    axes[0, 0].invert_yaxis(); axes[0, 0].set_xlabel("CRPS (mm/day)")
    axes[0, 0].set_title("A. Withheld-gauge probabilistic skill")

    centres = np.array([vs_current[name]["difference"] for name in ordered])
    lows = np.array([vs_current[name]["ci_low"] for name in ordered])
    highs = np.array([vs_current[name]["ci_high"] for name in ordered])
    axes[0, 1].errorbar(
        centres, positions, xerr=np.vstack([centres - lows, highs - centres]),
        fmt="o", color="#1B4965", capsize=3,
    )
    axes[0, 1].axvline(0.0, color="black", ls="--")
    axes[0, 1].set_yticks(positions, labels, fontsize=8); axes[0, 1].invert_yaxis()
    axes[0, 1].set_xlabel("CRPS(current) − CRPS(candidate)")
    axes[0, 1].set_title("B. Paired day-block uncertainty")

    width = 0.38
    axes[0, 2].barh(
        positions - width / 2,
        [plot_metric(metrics[name]["dry_mae"]) for name in ordered],
        height=width, label="dry", color="#E9C46A",
    )
    axes[0, 2].barh(
        positions + width / 2,
        [plot_metric(metrics[name]["wet_mae"]) for name in ordered],
        height=width, label="wet", color="#2A9D8F",
    )
    axes[0, 2].set_yticks(positions, labels, fontsize=8); axes[0, 2].invert_yaxis()
    axes[0, 2].set_xlabel("MAE (mm/day)"); axes[0, 2].legend()
    axes[0, 2].set_title("C. Dry/wet trade-off")

    axes[1, 0].barh(positions, [metrics[name]["bias"] for name in ordered], color="#D1495B")
    axes[1, 0].axvline(0.0, color="black")
    axes[1, 0].set_yticks(positions, labels, fontsize=8); axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlabel("Bias (mm/day)"); axes[1, 0].set_title("D. Mean bias")

    axes[1, 1].barh(
        positions - width / 2,
        [plot_metric(metrics[name]["correlation"]) for name in ordered],
        height=width, label="gauge r", color="#457B9D",
    )
    axes[1, 1].barh(
        positions + width / 2,
        [plot_metric(metrics[name]["imerg_pattern_correlation"]) for name in ordered],
        height=width, label="IMERG r", color="#6A4C93",
    )
    axes[1, 1].set_yticks(positions, labels, fontsize=8); axes[1, 1].invert_yaxis()
    axes[1, 1].set_xlabel("Correlation"); axes[1, 1].legend()
    axes[1, 1].set_title("E. Gauge skill and product structure")

    ode_steps = np.array(sorted(ODE_ARMS))
    ode_crps = np.array([metrics[ODE_ARMS[step]]["crps"] for step in ode_steps])
    axes[1, 2].plot(ode_steps, ode_crps, marker="o", label="no correctors")
    axes[1, 2].scatter(
        [50], [metrics[CURRENT]["crps"]], marker="s", s=60,
        label="2 correctors/level", color="#C1440E",
    )
    axes[1, 2].scatter(
        [100], [metrics[OPERATIONAL_N100]["crps"]], marker="s", s=60,
        color="#C1440E",
    )
    axes[1, 2].set_xscale("log", base=2)
    axes[1, 2].set_xticks(ode_steps, [str(value) for value in ode_steps])
    axes[1, 2].set_xlabel("Heun ODE steps"); axes[1, 2].set_ylabel("CRPS (mm/day)")
    axes[1, 2].legend(); axes[1, 2].set_title("F. ODE/corrector convergence screen")

    figure.suptitle("CPC-v2 simultaneous S04 DA refinement tournament")
    figure.savefig(out_plot, dpi=160)
    plt.close(figure)

    print("\n".join(lines))
    print(f"\n[done] wrote {out_json}")
    print(f"[done] wrote {out_markdown}")
    print(f"[done] wrote {out_plot}")
    print(f"[done] wrote {len(fold_paths)} fold plots under {fold_plot_dir}")


if __name__ == "__main__":
    main()
