#!/usr/bin/env python
"""Pool five v2 gauges-only DA folds and select methods without pseudoreplication.

The sweep in :mod:`scripts/28_simultaneous_method_sweep.py` writes one NPZ and
report per spatial fold.  Each fold withholds a disjoint part of the BMD network,
so pooling all five gives every station one turn as independent verification.

Uncertainty is blocked over *days*: all stations observing the same weather day
stay together in every bootstrap draw.  This replaces the station-day bootstrap
used by script 42, which treated spatially correlated gauges as independent and
made ten days look like 380 independent events.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


WET_MM = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dumps", nargs="+", required=True)
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--baseline", default="background")
    parser.add_argument("--current", default="guided_s0_t125")
    parser.add_argument("--block-days", type=int, default=3)
    parser.add_argument("--n-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=202208)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-markdown", required=True)
    parser.add_argument("--out-plot", required=True)
    return parser.parse_args()


def crps_per_sample(members: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Fair CRPS for a ``(day, member, station)`` ensemble."""
    members = np.asarray(members, dtype=float)
    observed = np.asarray(observed, dtype=float)
    count = members.shape[1]
    output = np.full(observed.shape, np.nan)
    finite = np.isfinite(observed) & np.all(np.isfinite(members), axis=1)
    if not finite.any():
        return output
    selected = np.moveaxis(members, 1, 0)[:, finite]
    truth = observed[finite]
    first = np.mean(np.abs(selected - truth[None]), axis=0)
    ordered = np.sort(selected, axis=0)
    weights = (2 * np.arange(1, count + 1) - count - 1)[:, None]
    pair = np.sum(weights * ordered, axis=0) / (count * (count - 1))
    output[finite] = first - pair
    return output


def circular_block_bootstrap(
    difference: np.ndarray,
    block_days: int,
    n_resamples: int,
    seed: int,
) -> dict:
    """Bootstrap a paired ``(day, station)`` difference in circular day blocks."""
    difference = np.asarray(difference, dtype=float)
    days = difference.shape[0]
    block_days = max(1, min(int(block_days), days))
    blocks = int(np.ceil(days / block_days))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, days, size=(n_resamples, blocks))
    offsets = np.arange(block_days)
    estimates = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        selected = (
            (starts[index][:, None] + offsets[None]) % days
        ).reshape(-1)[:days]
        sample = difference[selected]
        estimates[index] = np.nanmean(sample)
    low, high = np.nanpercentile(estimates, [2.5, 97.5])
    mean = float(np.nanmean(difference))
    return {
        "difference": mean,
        "ci_low": float(low),
        "ci_high": float(high),
        "block_days": block_days,
        "n_resamples": int(n_resamples),
        "significant": bool(low > 0 or high < 0),
    }


def daily_pattern_correlation(
    analysis: np.ndarray, reference: np.ndarray, valid: np.ndarray
) -> float:
    values = []
    for first, second in zip(analysis, reference):
        keep = valid & np.isfinite(first) & np.isfinite(second)
        if keep.sum() > 10 and first[keep].std() > 0 and second[keep].std() > 0:
            values.append(float(np.corrcoef(first[keep], second[keep])[0, 1]))
    return float(np.mean(values)) if values else float("nan")


def validate_and_load(dump_paths: list[Path], report_paths: list[Path]) -> list[dict]:
    if len(dump_paths) != len(report_paths):
        raise ValueError("--dumps and --reports must contain the same number of files")
    folds = []
    for dump_path, report_path in zip(dump_paths, report_paths):
        if not dump_path.is_file() or not report_path.is_file():
            raise FileNotFoundError(f"missing pair: {dump_path}, {report_path}")
        dump = np.load(dump_path, allow_pickle=False)
        report = json.loads(report_path.read_text())
        fold = int(report["scope"]["holdout_fold"])
        folds.append({"fold": fold, "dump": dump, "report": report,
                      "dump_path": str(dump_path), "report_path": str(report_path)})
    folds.sort(key=lambda item: item["fold"])

    expected = int(folds[0]["report"]["scope"]["holdout_folds"])
    if [item["fold"] for item in folds] != list(range(expected)):
        raise ValueError(
            f"need exactly folds 0..{expected - 1}; got {[f['fold'] for f in folds]}"
        )
    times = folds[0]["dump"]["times"].astype(str)
    stations = folds[0]["dump"]["station_ids"].astype(str)
    variants = folds[0]["dump"]["variant_names"].astype(str)
    checkpoint = folds[0]["report"]["scope"]["checkpoint"]
    group = folds[0]["report"]["scope"].get("group")
    withheld = []
    for item in folds:
        dump = item["dump"]
        scope = item["report"]["scope"]
        if int(scope["holdout_folds"]) != expected:
            raise ValueError("reports disagree on holdout-fold count")
        if scope["checkpoint"] != checkpoint or scope.get("group") != group:
            raise ValueError("fold reports disagree on checkpoint or sweep group")
        if not np.array_equal(dump["times"].astype(str), times):
            raise ValueError("folds do not use identical dates")
        if not np.array_equal(dump["station_ids"].astype(str), stations):
            raise ValueError("folds do not use identical station order")
        if not np.array_equal(dump["variant_names"].astype(str), variants):
            raise ValueError("folds do not contain identical variants")
        withheld.extend(stations[dump["eval_idx"]].tolist())
    if len(withheld) != len(set(withheld)) or set(withheld) != set(stations.tolist()):
        raise ValueError("rotated folds must withhold every station exactly once")
    return folds


def pooled_variant(folds: list[dict], name: str) -> tuple[dict, dict]:
    """Return scalar metrics and day-by-station arrays for one method."""
    crps_blocks, truth_blocks, mean_blocks = [], [], []
    median_blocks, spread_blocks, coverage_blocks = [], [], []
    cpc_correlation, chirps_correlation, wet_area = [], [], []
    locality = []

    for item in folds:
        dump = item["dump"]
        eval_idx = dump["eval_idx"]
        observed = np.asarray(dump["gauge_mm"][:, eval_idx], float)
        members = np.asarray(dump[f"station_{name}"][:, :, eval_idx], float)
        mean = np.nanmean(members, axis=1)
        median = np.nanmedian(members, axis=1)
        spread = np.nanvar(members, axis=1, ddof=1)
        low, high = np.nanquantile(members, [0.05, 0.95], axis=1)
        coverage = ((observed >= low) & (observed <= high)).astype(float)
        coverage[~np.isfinite(observed)] = np.nan

        crps_blocks.append(crps_per_sample(members, observed))
        truth_blocks.append(observed)
        mean_blocks.append(mean)
        median_blocks.append(median)
        spread_blocks.append(spread)
        coverage_blocks.append(coverage)

        field = np.asarray(dump[f"meanfield_{name}"], float)
        valid = np.asarray(dump["valid"], bool)
        wet_area.append(float(np.nanmean(field[:, valid] >= WET_MM)))
        if "condition" in dump:
            cpc_correlation.append(
                daily_pattern_correlation(field, np.asarray(dump["condition"]), valid)
            )
        if "chirps" in dump:
            chirps_correlation.append(
                daily_pattern_correlation(field, np.asarray(dump["chirps"]), valid)
            )
        ratio = item["report"]["variants"][name]["increment_locality"].get(
            "locality_ratio"
        )
        if ratio is not None and np.isfinite(ratio):
            locality.append(float(ratio))

    per_sample = {
        "crps": np.concatenate(crps_blocks, axis=1),
        "truth": np.concatenate(truth_blocks, axis=1),
        "mean": np.concatenate(mean_blocks, axis=1),
        "median": np.concatenate(median_blocks, axis=1),
        "spread_variance": np.concatenate(spread_blocks, axis=1),
        "coverage": np.concatenate(coverage_blocks, axis=1),
    }
    truth = per_sample["truth"]
    estimate = per_sample["mean"]
    keep = np.isfinite(truth) & np.isfinite(estimate)
    wet = keep & (truth >= WET_MM)
    dry = keep & (truth < WET_MM)
    difference = estimate - truth
    rmse = float(np.sqrt(np.nanmean(difference[keep] ** 2)))
    spread = float(np.sqrt(np.nanmean(per_sample["spread_variance"][keep])))
    metrics = {
        "n": int(keep.sum()),
        "n_wet": int(wet.sum()),
        "crps": float(np.nanmean(per_sample["crps"])),
        "mae": float(np.nanmean(np.abs(difference[keep]))),
        "wet_mae": float(np.nanmean(np.abs(difference[wet]))) if wet.any() else None,
        "dry_mae": float(np.nanmean(np.abs(difference[dry]))) if dry.any() else None,
        "bias": float(np.nanmean(difference[keep])),
        "correlation": (
            float(np.corrcoef(estimate[keep], truth[keep])[0, 1])
            if estimate[keep].std() > 0 and truth[keep].std() > 0 else None
        ),
        "rmse": rmse,
        "spread": spread,
        "spread_skill": spread / rmse if rmse else None,
        "coverage_90": float(np.nanmean(per_sample["coverage"])),
        "median_bias": float(np.nanmean(per_sample["median"][keep] - truth[keep])),
        "wet_area": float(np.nanmean(wet_area)),
        "cpc_pattern_correlation": (
            float(np.nanmean(cpc_correlation)) if cpc_correlation else None
        ),
        "chirps_pattern_correlation": (
            float(np.nanmean(chirps_correlation)) if chirps_correlation else None
        ),
        "locality_ratio": float(np.nanmean(locality)) if locality else None,
    }
    return metrics, per_sample


def main() -> None:
    args = parse_args()
    folds = validate_and_load(
        [Path(path) for path in args.dumps], [Path(path) for path in args.reports]
    )
    names = folds[0]["dump"]["variant_names"].astype(str).tolist()
    for required in (args.baseline, args.current):
        if required not in names:
            raise ValueError(f"required method {required!r} not in {names}")

    metrics, samples = {}, {}
    for name in names:
        metrics[name], samples[name] = pooled_variant(folds, name)

    comparisons = {}
    current_comparisons = {}
    for name in names:
        # Positive means the candidate is better than the named reference.
        comparisons[name] = circular_block_bootstrap(
            samples[args.baseline]["crps"] - samples[name]["crps"],
            args.block_days, args.n_resamples, args.seed,
        )
        current_comparisons[name] = circular_block_bootstrap(
            samples[args.current]["crps"] - samples[name]["crps"],
            args.block_days, args.n_resamples, args.seed + 10_000,
        )

    ordered = sorted(names, key=lambda name: metrics[name]["crps"])
    baseline_bias = abs(metrics[args.baseline]["bias"])
    promoted = [
        name for name in ordered
        if name != args.baseline
        and comparisons[name]["difference"] > 0
        and abs(metrics[name]["bias"]) <= baseline_bias + 0.5
    ][:2]

    scope = folds[0]["report"]["scope"]
    lines = [
        "# CPC-v2 gauges-only DA method tournament",
        "",
        f"- Period: **{scope['start']} to {scope['end']}**",
        f"- Five disjoint spatial folds; every BMD station withheld exactly once",
        f"- Members: **{scope['members']}**; reference: `{args.baseline}`",
        f"- Paired circular bootstrap: **{args.block_days}-day blocks**, "
        f"{args.n_resamples:,} resamples; stations from one day stay together",
        "",
        "| Method | CRPS | Δ vs background (95% CI) | MAE dry/wet | Bias | Corr | "
        "Cov90 | CPC r | Wet area | Locality |",
        "|:--|--:|:--|:--|--:|--:|--:|--:|--:|--:|",
    ]
    for name in ordered:
        metric = metrics[name]
        comparison = comparisons[name]
        locality_text = (
            "—" if metric["locality_ratio"] is None
            else f"{metric['locality_ratio']:.2f}"
        )
        verdict = "+" if comparison["ci_low"] > 0 else (
            "−" if comparison["ci_high"] < 0 else "~"
        )
        lines.append(
            f"| `{name}` | {metric['crps']:.3f} | "
            f"{comparison['difference']:+.3f} "
            f"[{comparison['ci_low']:+.3f}, {comparison['ci_high']:+.3f}] {verdict} | "
            f"{metric['dry_mae']:.2f}/{metric['wet_mae']:.2f} | "
            f"{metric['bias']:+.2f} | {metric['correlation']:.3f} | "
            f"{metric['coverage_90']:.2f} | "
            f"{metric['cpc_pattern_correlation']:.3f} | {metric['wet_area']:.3f} | "
            f"{locality_text} |"
        )
    lines += [
        "",
        "+ = beats the background with the whole block-bootstrap interval above zero; "
        "− = worse; ~ = unresolved.",
        "",
        "## Promotion decision",
        "",
        "Promote at most two methods whose central CRPS improves on background and "
        "whose absolute bias is no more than 0.5 mm/day worse:",
        "",
        *(f"- `{name}`" for name in promoted),
    ]
    if not promoted:
        lines.append("- None. Retain the v2 background until a method clears the gates.")
    lines += [
        "",
        f"The current operational comparator is `{args.current}`. Exact paired "
        "differences against it are stored in the JSON under `vs_current`.",
    ]

    output = {
        "scope": {
            "start": scope["start"], "end": scope["end"],
            "members": scope["members"], "folds": len(folds),
            "checkpoint": scope["checkpoint"],
            "block_days": args.block_days, "n_resamples": args.n_resamples,
            "baseline": args.baseline, "current": args.current,
        },
        "ranked_by_crps": ordered,
        "metrics": metrics,
        "vs_background": comparisons,
        "vs_current": current_comparisons,
        "promoted": promoted,
    }
    for path in (Path(args.out_json), Path(args.out_markdown), Path(args.out_plot)):
        path.parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    Path(args.out_markdown).write_text("\n".join(lines) + "\n")

    positions = np.arange(len(ordered))
    figure, axes = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True)
    axes[0, 0].barh(positions, [metrics[n]["crps"] for n in ordered], color="#C1440E")
    axes[0, 0].set_yticks(positions, ordered, fontsize=8)
    axes[0, 0].invert_yaxis(); axes[0, 0].set_xlabel("CRPS (mm/day)")
    axes[0, 0].set_title("A. Withheld-gauge probabilistic skill")

    centres = np.array([comparisons[n]["difference"] for n in ordered])
    lows = np.array([comparisons[n]["ci_low"] for n in ordered])
    highs = np.array([comparisons[n]["ci_high"] for n in ordered])
    axes[0, 1].errorbar(
        centres, positions, xerr=np.vstack([centres - lows, highs - centres]),
        fmt="o", color="#1B4965", capsize=3,
    )
    axes[0, 1].axvline(0, color="black", ls="--")
    axes[0, 1].set_yticks(positions, ordered, fontsize=8); axes[0, 1].invert_yaxis()
    axes[0, 1].set_xlabel("CRPS(background) − CRPS(method)")
    axes[0, 1].set_title("B. Paired day-block uncertainty")

    width = 0.38
    axes[0, 2].barh(positions - width / 2, [metrics[n]["dry_mae"] for n in ordered],
                    height=width, label="dry", color="#E9C46A")
    axes[0, 2].barh(positions + width / 2, [metrics[n]["wet_mae"] for n in ordered],
                    height=width, label="wet", color="#2A9D8F")
    axes[0, 2].set_yticks(positions, ordered, fontsize=8); axes[0, 2].invert_yaxis()
    axes[0, 2].set_xlabel("MAE (mm/day)"); axes[0, 2].legend()
    axes[0, 2].set_title("C. Dry/wet trade-off")

    axes[1, 0].barh(positions, [metrics[n]["bias"] for n in ordered], color="#D1495B")
    axes[1, 0].axvline(0, color="black")
    axes[1, 0].set_yticks(positions, ordered, fontsize=8); axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlabel("Bias (mm/day)"); axes[1, 0].set_title("D. Mean bias")

    axes[1, 1].barh(positions - width / 2,
                    [metrics[n]["cpc_pattern_correlation"] for n in ordered],
                    height=width, label="CPC r", color="#F4A261")
    axes[1, 1].barh(positions + width / 2, [metrics[n]["wet_area"] for n in ordered],
                    height=width, label="wet area", color="#457B9D")
    axes[1, 1].set_yticks(positions, ordered, fontsize=8); axes[1, 1].invert_yaxis()
    axes[1, 1].legend(); axes[1, 1].set_title("E. Spatial plausibility")

    locality_names = [name for name in ordered if metrics[name]["locality_ratio"] is not None]
    locality_pos = np.arange(len(locality_names))
    axes[1, 2].barh(locality_pos, [metrics[n]["locality_ratio"] for n in locality_names],
                    color="#6A4C93")
    axes[1, 2].axvline(1.0, color="black", ls="--")
    axes[1, 2].set_yticks(locality_pos, locality_names, fontsize=8); axes[1, 2].invert_yaxis()
    axes[1, 2].set_xlabel("Near/far increment ratio")
    axes[1, 2].set_title("F. Locality; high values indicate bullseyes")

    figure.suptitle("CPC-v2 gauges-only DA method tournament")
    figure.savefig(args.out_plot, dpi=160)
    plt.close(figure)

    print("\n".join(lines))
    print(f"\n[done] wrote {args.out_json}")
    print(f"[done] wrote {args.out_markdown}")
    print(f"[done] wrote {args.out_plot}")


if __name__ == "__main__":
    main()
