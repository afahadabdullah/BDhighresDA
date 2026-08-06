#!/usr/bin/env python
"""Rank the simultaneous-DA variants from script 28, honestly.

Three things this prints that the existing summaries do not:

1. **A paired circular block bootstrap on delta-CRPS** against the gauges-only
   arm, blocked over days because daily rainfall is autocorrelated.  The pooled
   2021-2024 result put simultaneous ahead of gauges-only by 2.6% with no
   interval attached; on a five-day window the interval will be very wide, and
   printing it is the point.  A variant whose interval straddles zero has not
   beaten gauges-only, however good its central estimate looks.

2. **The Jensen gap** -- mean-based bias minus median-based bias.  Prior
   tempering broadens the ensemble in transformed space, and inverting a convex
   transform then pushes the mm-space ensemble MEAN up without moving the median.
   A large gap means the arm's bias is a reporting artefact, not a physical one.

3. **The increment-locality curve** -- mean |analysis - background| binned by
   distance to the nearest assimilated gauge.  This is the quantitative version
   of the station bullseyes in the intercomparison figure.  Falling curves are
   discs around gauges; flat curves are increments spread along meteorology.

Usage
-----
    python scripts/29_summarize_method_sweep.py \
        --dump data/processed/sweep_may2024.npz \
        --report data/processed/sweep_may2024.json \
        --baseline gauges_only \
        --out-markdown data/processed/sweep_may2024_ranking.md \
        --out-plot data/processed/sweep_may2024_ranking.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", required=True, help="npz written by script 28")
    parser.add_argument("--report", required=True, help="json written by script 28")
    parser.add_argument("--baseline", default="gauges_only")
    parser.add_argument("--block-days", type=int, default=3)
    parser.add_argument("--n-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-markdown", default="data/processed/sweep_ranking.md")
    parser.add_argument("--out-plot", default="data/processed/sweep_ranking.png")
    parser.add_argument("--out-json", default=None)
    return parser.parse_args()


def crps_per_sample(members: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Fair (unbiased) CRPS for every (day, station) pair.

    ``members`` is (T, M, S) and ``observed`` is (T, S).  Returns (T, S) with NaN
    where the observation is missing.  The second term uses the M(M-1)
    denominator, i.e. the fair estimator, so ensembles of different effective
    size are not rewarded for being small.
    """
    members = np.asarray(members, dtype=float)
    observed = np.asarray(observed, dtype=float)
    n_members = members.shape[1]
    out = np.full(observed.shape, np.nan)
    finite = np.isfinite(observed) & np.all(np.isfinite(members), axis=1)
    if not finite.any():
        return out
    selected = np.moveaxis(members, 1, 0)[:, finite]        # (M, N)
    truth = observed[finite]                                # (N,)
    term1 = np.mean(np.abs(selected - truth[None, :]), axis=0)
    # sum_{i<j} |x_i - x_j| in O(M log M): in the sorted sample x_(k) is the
    # larger element of (k-1) pairs and the smaller of (M-k), so it carries a
    # weight of (2k - M - 1).
    ordered = np.sort(selected, axis=0)
    weights = (2 * np.arange(1, n_members + 1) - n_members - 1)[:, None]
    pair_sum = np.sum(weights * ordered, axis=0)
    term2 = pair_sum / (n_members * (n_members - 1))
    # Identical to bdhires.eval.crps_ensemble, which uses the O(M^2) form; the
    # per-sample values here must average to that function's scalar output, and
    # tests/test_method_sweep.py asserts exactly that.
    out[finite] = term1 - term2
    return out


def circular_block_bootstrap(
    difference: np.ndarray,
    block_days: int,
    n_resamples: int,
    seed: int,
) -> tuple[float, float, float]:
    """95% interval for the mean of a (T, S) paired difference, blocked in time.

    Stations within a day are kept together, because two gauges 60 km apart on
    the same day are not independent samples.  Blocks wrap around the window
    (circular) so every day has equal resampling probability.
    """
    difference = np.asarray(difference, dtype=float)
    n_days = difference.shape[0]
    block_days = max(1, min(block_days, n_days))
    n_blocks = int(np.ceil(n_days / block_days))
    rng = np.random.default_rng(seed)

    observed_mean = float(np.nanmean(difference))
    starts = rng.integers(0, n_days, size=(n_resamples, n_blocks))
    offsets = np.arange(block_days)
    means = np.empty(n_resamples)
    for index in range(n_resamples):
        days = ((starts[index][:, None] + offsets[None, :]).reshape(-1) % n_days)[:n_days]
        sample = difference[days]
        means[index] = np.nanmean(sample) if np.isfinite(sample).any() else np.nan
    low, high = np.nanpercentile(means, [2.5, 97.5])
    return observed_mean, float(low), float(high)


def main() -> None:
    args = parse_args()
    dump = np.load(args.dump, allow_pickle=False)
    report = json.loads(Path(args.report).read_text())
    scope = report["scope"]
    variants = [str(name) for name in dump["variant_names"]]
    eval_idx = dump["eval_idx"]
    observed = dump["gauge_mm"][:, eval_idx]
    n_days = observed.shape[0]

    if args.baseline not in variants:
        raise ValueError(f"baseline {args.baseline!r} not in {variants}")

    per_sample = {
        name: crps_per_sample(dump[f"station_{name}"][:, :, eval_idx], observed)
        for name in variants
    }
    baseline = per_sample[args.baseline]

    rows = []
    for name in variants:
        entry = report["variants"][name]
        difference = baseline - per_sample[name]      # positive = variant is better
        mean, low, high = circular_block_bootstrap(
            difference, args.block_days, args.n_resamples, args.seed
        )
        locality = entry.get("increment_locality", {})
        rows.append(
            {
                "variant": name,
                "crps_mm": entry.get("crps_mm", float("nan")),
                "delta_crps_vs_baseline_mm": mean,
                "delta_crps_ci95": [low, high],
                "beats_baseline": bool(low > 0),
                "distinguishable": bool(low > 0 or high < 0),
                "mean_bias_mm": entry.get("mean_bias_mm", float("nan")),
                "median_bias_mm": entry.get("median_bias_mm", float("nan")),
                "jensen_gap_mm": entry.get("jensen_gap_mm", float("nan")),
                "mean_mae_mm": entry.get("mean_mae_mm", float("nan")),
                "mean_correlation": entry.get("mean_correlation", float("nan")),
                "coverage_90": entry.get("coverage_90", float("nan")),
                "spread_skill": entry.get("spread_skill", float("nan")),
                "wet_day_fraction": entry.get("wet_day_fraction", float("nan")),
                "locality_ratio": locality.get("locality_ratio"),
                "note": entry.get("spec", {}).get("note", ""),
            }
        )

    ordered = sorted(rows, key=lambda row: row["crps_mm"])

    lines = [
        f"# Simultaneous-DA method sweep — {scope['start']} to {scope['end']}",
        "",
        f"- Days: **{scope['n_days']}**, members: **{scope['members']}**, "
        f"withheld station-days: **{scope['withheld_station_days']}**",
        f"- Holdout fold {scope['holdout_fold'] + 1}/{scope['holdout_folds']}; "
        f"{scope['n_assimilated_stations']} assimilated, "
        f"{scope['n_withheld_stations']} withheld",
        f"- Baseline for ΔCRPS: `{args.baseline}`; paired circular block bootstrap, "
        f"block {args.block_days} d, {args.n_resamples:,} resamples",
        "",
        f"> {scope['caveat']}",
        "",
    ]
    if n_days < 15:
        lines += [
            f"> **Sample-size warning.** {n_days} days supports roughly "
            f"{max(1, n_days // args.block_days)} independent blocks. The ΔCRPS "
            "intervals below are wide by construction and are reported so that "
            "small central estimates are not mistaken for results.",
            "",
        ]

    lines += [
        "| Variant | CRPS | ΔCRPS vs baseline (95% CI) | Bias (mean) | Bias (median) "
        "| Jensen gap | MAE | Corr | Cov90 | Wet frac | Locality |",
        "|:--|--:|:--|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for row in ordered:
        low, high = row["delta_crps_ci95"]
        verdict = "**+**" if row["beats_baseline"] else ("−" if high < 0 else "~")
        locality = row["locality_ratio"]
        lines.append(
            f"| `{row['variant']}` | {row['crps_mm']:.3f} | "
            f"{row['delta_crps_vs_baseline_mm']:+.3f} "
            f"[{low:+.3f}, {high:+.3f}] {verdict} | "
            f"{row['mean_bias_mm']:+.2f} | {row['median_bias_mm']:+.2f} | "
            f"{row['jensen_gap_mm']:+.2f} | {row['mean_mae_mm']:.2f} | "
            f"{row['mean_correlation']:.3f} | {row['coverage_90']:.2f} | "
            f"{row['wet_day_fraction']:.2f} | "
            f"{'—' if locality is None else f'{locality:.2f}'} |"
        )

    lines += [
        "",
        "`+` beats the baseline with the whole interval above zero, `−` is worse "
        "with the whole interval below zero, `~` is unresolved at this sample size.",
        "",
        "**Locality** is mean |analysis − background| in the 0–25 km bin divided by "
        "the 150–250 km bin. Values near 1 mean gauge information is spread along "
        "meteorological structure; large values mean discs around stations.",
        "",
        "**Jensen gap** is mean-based bias minus median-based bias. A large positive "
        "gap says the arm's wet bias is an artefact of inverting a convex transform "
        "over a tempered ensemble, not a physical over-prediction.",
        "",
    ]

    promoted = [
        row["variant"]
        for row in ordered
        if row["variant"] != args.baseline
        and row["delta_crps_vs_baseline_mm"] > 0
        and abs(row["mean_bias_mm"]) <= abs(
            next(r["mean_bias_mm"] for r in rows if r["variant"] == args.baseline)
        )
        + 0.5
    ]
    lines += [
        "## Promotion gate",
        "",
        "A variant earns a full 2021–2024 run only if it improves central ΔCRPS "
        "**and** does not worsen absolute bias against the baseline by more than "
        "0.5 mm/day. On this window that set is:",
        "",
        ("- " + "\n- ".join(f"`{name}`" for name in promoted)) if promoted else "- (none)",
        "",
        "Promote at most two. Everything else is a five-day coincidence until a "
        "longer window says otherwise.",
        "",
    ]

    markdown_path = Path(args.out_markdown)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[summary] wrote {markdown_path}", flush=True)

    if args.out_json:
        Path(args.out_json).write_text(
            json.dumps({"scope": scope, "baseline": args.baseline, "rows": rows}, indent=2)
        )
        print(f"[summary] wrote {args.out_json}", flush=True)

    # ------------------------------------------------------------------ figure
    figure, axes = plt.subplots(2, 3, figsize=(19, 10), constrained_layout=True)
    names = [row["variant"] for row in ordered]
    positions = np.arange(len(names))
    palette = plt.get_cmap("tab20")(np.linspace(0, 1, max(len(names), 2)))

    axes[0, 0].barh(positions, [row["crps_mm"] for row in ordered], color=palette)
    axes[0, 0].set_yticks(positions)
    axes[0, 0].set_yticklabels(names, fontsize=8)
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_xlabel("CRPS (mm day$^{-1}$)")
    axes[0, 0].set_title("A. Withheld-BMD CRPS — lower is better")

    centres = np.array([row["delta_crps_vs_baseline_mm"] for row in ordered])
    lows = np.array([row["delta_crps_ci95"][0] for row in ordered])
    highs = np.array([row["delta_crps_ci95"][1] for row in ordered])
    axes[0, 1].errorbar(
        centres,
        positions,
        xerr=np.vstack([centres - lows, highs - centres]),
        fmt="o",
        color="#1B4965",
        ecolor="#5FA8D3",
        capsize=3,
    )
    axes[0, 1].axvline(0, color="black", lw=1)
    axes[0, 1].set_yticks(positions)
    axes[0, 1].set_yticklabels(names, fontsize=8)
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_xlabel(f"CRPS({args.baseline}) − CRPS(variant)  (mm day$^{{-1}}$)")
    axes[0, 1].set_title(
        f"B. ΔCRPS vs {args.baseline}\nintervals crossing zero are unresolved"
    )

    width = 0.38
    axes[0, 2].barh(
        positions - width / 2,
        [row["mean_bias_mm"] for row in ordered],
        height=width,
        color="#D1495B",
        label="mean estimator",
    )
    axes[0, 2].barh(
        positions + width / 2,
        [row["median_bias_mm"] for row in ordered],
        height=width,
        color="#00798C",
        label="median estimator",
    )
    axes[0, 2].axvline(0, color="black", lw=1)
    axes[0, 2].set_yticks(positions)
    axes[0, 2].set_yticklabels(names, fontsize=8)
    axes[0, 2].invert_yaxis()
    axes[0, 2].set_xlabel("Bias (mm day$^{-1}$)")
    axes[0, 2].set_title("C. Bias, and how much of it is Jensen")
    axes[0, 2].legend(fontsize=8)

    for index, name in enumerate(names):
        locality = report["variants"][name]["increment_locality"]
        edges = np.asarray(locality["edges_km"], dtype=float)
        values = np.array(
            [np.nan if v is None else v for v in locality["mean_abs_increment_mm"]],
            dtype=float,
        )
        centres_km = 0.5 * (edges[:-1] + np.minimum(edges[1:], 300.0))
        axes[1, 0].plot(
            centres_km, values, marker="o", color=palette[index], label=name, lw=1.4
        )
    axes[1, 0].set_xlabel("Distance to nearest assimilated gauge (km)")
    axes[1, 0].set_ylabel("Mean |analysis − background| (mm day$^{-1}$)")
    axes[1, 0].set_title("D. Increment locality — falling curve = station bullseyes")
    axes[1, 0].legend(fontsize=6, ncol=2)
    axes[1, 0].grid(alpha=0.2)

    axes[1, 1].scatter(
        [row["coverage_90"] for row in ordered],
        [row["spread_skill"] for row in ordered],
        c=palette,
        s=60,
    )
    for index, row in enumerate(ordered):
        axes[1, 1].annotate(
            row["variant"],
            (row["coverage_90"], row["spread_skill"]),
            fontsize=6,
            xytext=(3, 3),
            textcoords="offset points",
        )
    axes[1, 1].axvline(0.90, color="black", ls="--", lw=1)
    axes[1, 1].axhline(1.0, color="black", ls="--", lw=1)
    axes[1, 1].set_xlabel("Empirical 90% coverage")
    axes[1, 1].set_ylabel("Spread / skill")
    axes[1, 1].set_title("E. Calibration — dashed lines are nominal")
    axes[1, 1].grid(alpha=0.2)

    axes[1, 2].barh(
        positions,
        [row["wet_day_fraction"] for row in ordered],
        color=palette,
    )
    axes[1, 2].set_yticks(positions)
    axes[1, 2].set_yticklabels(names, fontsize=8)
    axes[1, 2].invert_yaxis()
    axes[1, 2].set_xlabel("Fraction of land cells with ensemble-mean ≥ 1 mm")
    axes[1, 2].set_title("F. Wet-day frequency — the prior's drizzle problem")

    figure.suptitle(
        f"Simultaneous-DA method sweep — {scope['start']} to {scope['end']}, "
        f"{scope['n_days']} days, {scope['withheld_station_days']} withheld station-days "
        "(screening run, not a skill evaluation)",
        fontsize=13,
    )
    plot_path = Path(args.out_plot)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=150)
    print(f"[summary] wrote {plot_path}", flush=True)


if __name__ == "__main__":
    main()
