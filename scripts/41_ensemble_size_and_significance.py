#!/usr/bin/env python
"""How many members are needed, and is the gap between two configs real?

Two questions this project has repeatedly answered by eye and got wrong.

**Ensemble size.** Running 50 members instead of 30 costs 67% more GPU time. It
is worth it only if CRPS is still improving at 30. The fair (unbiased) CRPS of
Zamo and Naveau is constructed so that its expectation does not depend on
ensemble size, so a flat curve here is the expected result and means members
beyond that point buy nothing in the mean. What DOES keep improving with size is
the sampling precision of the estimate, and that is visible as the shrinking
confidence band rather than as a falling curve. Distinguishing those two is the
point of the figure.

**Significance.** The 5-day experiments ranked configurations on CRPS gaps of
0.08 mm/day and a wetMAE difference that later reversed sign on a larger sample.
A paired bootstrap over station-days gives a confidence interval on the
DIFFERENCE between two configurations, which is the quantity actually being
claimed. Pairing is never worse than not pairing, and how much it helps depends
on how correlated the two arms are sample by sample: measured on synthetic arms
that share their background and member noise, as real DA arms do, the paired
interval is about 1.7x tighter (per-sample CRPS correlation 0.64). On arms that
share nothing it makes no difference at all. Do not expect a dramatic gain.

Outputs a table with, for every pair of runs, the CRPS difference and its 95%
interval, and a verdict of significant or not.

Example
-------
    python scripts/41_ensemble_size_and_significance.py \\
        --dumps 'data/processed/imerg_screen_may2024/*.npz' \\
        --arm combined --out-dir data/processed/ensemble_size
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bdhires.eval.metrics import crps_ensemble  # noqa: E402

ARM_KEYS = {
    "background": "background_at_stations",
    "gauges": "gauge_analysis_at_stations",
    "satellite": "imerg_analysis_at_stations",
    "combined": "combined_analysis_at_stations",
}


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 140, "savefig.dpi": 140, "font.size": 9,
        "axes.grid": True, "grid.alpha": 0.25,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    return plt


def load_pairs(path: Path, arm: str):
    """Withheld-station ensemble ``(M, N)`` and truth ``(N,)`` from one dump."""
    z = np.load(path, allow_pickle=False)
    key = ARM_KEYS.get(arm)
    if key is None or key not in z or "eval_idx" not in z:
        return None, None
    evl = z["eval_idx"]
    block = np.moveaxis(z[key], 1, 0)[:, :, evl]        # (M, T, S)
    members = block.reshape(block.shape[0], -1)
    truth = z["gauge_mm"][:, evl].ravel()
    ok = np.isfinite(truth) & np.all(np.isfinite(members), axis=0)
    return members[:, ok], truth[ok]


def crps_pointwise(members: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Fair CRPS per sample, so it can be bootstrapped over samples.

    ``bdhires.eval.metrics.crps_ensemble`` returns the mean; the bootstrap needs
    the per-sample values, so the same estimator is written out elementwise here.
    The second term is the fair correction: dividing the pairwise member spread
    by ``M(M-1)`` rather than ``M^2`` removes the finite-ensemble bias, which is
    exactly what makes CRPS comparable ACROSS ensemble sizes.
    """
    m = members.shape[0]
    if m < 2:
        return np.abs(members[0] - truth)
    first = np.abs(members - truth[None]).mean(axis=0)
    spread = np.abs(members[:, None] - members[None, :]).sum(axis=(0, 1))
    return first - spread / (2 * m * (m - 1))


def crps_vs_members(members, truth, sizes, n_draws=40, seed=0, ci=95.0) -> dict:
    """Fair CRPS against ensemble size, resampling members without replacement."""
    rng = np.random.default_rng(seed)
    total = members.shape[0]
    out = []
    for size in sizes:
        if size > total:
            continue
        draws = []
        repeats = 1 if size == total else n_draws
        for _ in range(repeats):
            pick = rng.choice(total, size=size, replace=False)
            draws.append(float(np.mean(crps_pointwise(members[pick], truth))))
        low, high = np.percentile(draws, [(100 - ci) / 2, 100 - (100 - ci) / 2]) \
            if len(draws) > 1 else (draws[0], draws[0])
        out.append({"members": int(size), "crps": float(np.mean(draws)),
                    "low": float(low), "high": float(high), "n_draws": len(draws)})
    return {"curve": out, "n_samples": int(truth.size), "max_members": int(total)}


def paired_bootstrap(a_members, a_truth, b_members, b_truth,
                     n_boot=2000, seed=0, ci=95.0) -> dict:
    """Confidence interval on the CRPS DIFFERENCE between two configurations.

    Resampling is over station-days, and the SAME resampled indices are applied
    to both configurations. That pairing is what gives the test power: the arms
    see identical days, so whatever they have in common cancels in the
    difference: var(a - b) = var(a) + var(b) - 2 cov(a, b), and the pairing buys
    exactly the covariance term.

    That term is only large when the arms are genuinely correlated. Verified on
    synthetic data: arms sharing their background and member noise (per-sample
    CRPS correlation 0.64, which is the realistic case since DA arms share a
    prior, seeds and days) give an interval 1.7x tighter than the unpaired one;
    arms built independently (correlation -0.03) give no improvement whatever.
    Pairing is still the right default, since it cannot be worse, but the gain
    is conditional rather than automatic.

    Returns the difference a - b, so a negative value means ``a`` is better.
    """
    if a_truth.size != b_truth.size or not np.allclose(a_truth, b_truth):
        return {"comparable": False,
                "reason": "the two runs were not scored on the same station-days"}
    per_a = crps_pointwise(a_members, a_truth)
    per_b = crps_pointwise(b_members, b_truth)
    difference = per_a - per_b
    rng = np.random.default_rng(seed)
    n = difference.size
    draws = np.array([difference[rng.integers(0, n, n)].mean()
                      for _ in range(n_boot)])
    low, high = np.percentile(draws, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return {
        "comparable": True,
        "crps_a": float(per_a.mean()),
        "crps_b": float(per_b.mean()),
        "difference": float(difference.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "significant": bool(low > 0 or high < 0),
        "n_samples": int(n),
        "n_boot": int(n_boot),
    }


def unique_labels(paths: list[Path]) -> dict:
    """Map each path to a label that is unique ACROSS DIRECTORIES.

    Keying by ``path.stem`` alone silently loses data: every configuration's
    output directory contains a file called ``..._fold0.npz``, so globbing over
    several configurations collapses them into one entry and the last read wins.
    That happened -- a four-configuration screen reported four entries named
    fold0..fold3, which were folds of whichever configuration happened to be
    read last, and the comparison that was wanted never took place.

    The parent directory name is prefixed only where stems actually collide, so
    single-directory use keeps its short labels.
    """
    stems = [p.stem for p in paths]
    clashing = {s for s in stems if stems.count(s) > 1}
    labels = {}
    for path in paths:
        if path.stem in clashing:
            labels[path] = f"{path.parent.name}/{path.stem}"
        else:
            labels[path] = path.stem
    return labels


# --------------------------------------------------------------------------


def plot_all(curves: dict, pairs: list, out_path: Path) -> None:
    plt = _style()
    figure, (left, right) = plt.subplots(1, 2, figsize=(13.0, 4.8),
                                         gridspec_kw={"width_ratios": [1, 1.2]})

    for name, block in curves.items():
        c = block["curve"]
        if not c:
            continue
        m = [r["members"] for r in c]
        v = [r["crps"] for r in c]
        left.plot(m, v, "o-", lw=1.8, label=name)
        left.fill_between(m, [r["low"] for r in c], [r["high"] for r in c],
                          alpha=0.18, lw=0)
    left.set_xlabel("ensemble members")
    left.set_ylabel("fair CRPS against withheld gauges (mm/day)")
    left.set_title("Fair CRPS is size-unbiased by construction:\n"
                   "a FLAT curve means extra members buy nothing in the mean,\n"
                   "and the narrowing band is the precision they do buy",
                   fontsize=8.5)
    left.legend(fontsize=7)

    if pairs:
        labels = [f"{a}\n- {b}" for a, b, _ in pairs]
        diff = [r["difference"] for _, _, r in pairs]
        low = [r["difference"] - r["ci_low"] for _, _, r in pairs]
        high = [r["ci_high"] - r["difference"] for _, _, r in pairs]
        colours = ["#c1440e" if r["significant"] else "#999999"
                   for _, _, r in pairs]
        y = np.arange(len(pairs))
        right.errorbar(diff, y, xerr=[low, high], fmt="o", ms=6,
                       ecolor="#555555", capsize=3, linestyle="none")
        for i, c in enumerate(colours):
            right.plot(diff[i], y[i], "o", color=c, ms=8, zorder=5)
        right.axvline(0.0, color="#111111", lw=1.4, ls="--")
        right.set_yticks(y); right.set_yticklabels(labels, fontsize=6.5)
        right.set_xlabel("CRPS difference (mm/day), negative favours the first")
        right.set_title("Paired bootstrap over station-days, 95% interval\n"
                        "orange = interval excludes zero; grey = not "
                        "distinguishable", fontsize=8.5)
    figure.tight_layout()
    figure.savefig(out_path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ensemble-size convergence and paired significance testing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dumps", nargs="+", required=True)
    parser.add_argument("--arm", default="combined", choices=sorted(ARM_KEYS))
    parser.add_argument("--sizes", nargs="+", type=int,
                        default=[2, 4, 8, 12, 16, 24, 32, 40, 50])
    parser.add_argument("--n-draws", type=int, default=40)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    paths = sorted({Path(p) for pat in args.dumps for p in (glob.glob(pat) or [pat])})
    paths = [p for p in paths if p.suffix == ".npz" and p.exists()]
    if not paths:
        raise SystemExit(f"no NPZ dumps matched {args.dumps}")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    labels = unique_labels(paths)
    loaded, curves = {}, {}
    for path in paths:
        members, truth = load_pairs(path, args.arm)
        if members is None or truth is None or truth.size < 10:
            print(f"[skip] {path.name}: no usable arm {args.arm!r}")
            continue
        loaded[labels[path]] = (members, truth)
        curves[labels[path]] = crps_vs_members(
            members, truth, args.sizes, args.n_draws, args.seed)
        print(f"[run] {labels[path]}: {members.shape[0]} members, "
              f"{truth.size} withheld station-days")

    if not curves:
        raise SystemExit("nothing to evaluate")

    print()
    print("[members] fair CRPS against ensemble size (95% over member draws):")
    for name, block in curves.items():
        print(f"  {name}")
        for r in block["curve"]:
            print(f"    m={r['members']:>3d}  CRPS {r['crps']:6.3f}  "
                  f"[{r['low']:6.3f}, {r['high']:6.3f}]")
        c = block["curve"]
        if len(c) > 1:
            change = (c[-1]["crps"] - c[0]["crps"]) / max(abs(c[0]["crps"]), 1e-9)
            narrowing = (c[0]["high"] - c[0]["low"]) - (c[-1]["high"] - c[-1]["low"])
            print(f"    mean changes {change:+.1%} from m={c[0]['members']} to "
                  f"m={c[-1]['members']}; interval narrows by {narrowing:.3f} mm/day")

    pairs = []
    for a, b in itertools.combinations(sorted(loaded), 2):
        result = paired_bootstrap(*loaded[a], *loaded[b],
                                  n_boot=args.n_boot, seed=args.seed)
        if result.get("comparable"):
            pairs.append((a, b, result))
        else:
            print(f"[pair] {a} vs {b}: {result['reason']}")

    if pairs:
        print()
        print("[significance] paired bootstrap over station-days, 95% interval.")
        print("    Negative difference favours the FIRST run.")
        print(f"    {'comparison':52s} {'diff':>7s} {'95% interval':>18s}  verdict")
        for a, b, r in sorted(pairs, key=lambda p: p[2]["difference"]):
            verdict = "SIGNIFICANT" if r["significant"] else "not distinguishable"
            print(f"    {a[:24]:24s} vs {b[:24]:24s} {r['difference']:>+7.3f} "
                  f"[{r['ci_low']:>+7.3f},{r['ci_high']:>+7.3f}]  {verdict}")
        print()
        print("    An interval spanning zero means the configurations cannot be")
        print("    separated on this sample, however different the point estimates")
        print("    look. Ranking them anyway is what produced the 5-day results")
        print("    that later reversed.")

    plot_all(curves, pairs, out_dir / "ensemble_size_and_significance.png")
    (out_dir / "ensemble_size_and_significance.json").write_text(json.dumps(
        {"arm": args.arm, "curves": curves,
         "pairs": [{"a": a, "b": b, **r} for a, b, r in pairs]},
        indent=2, default=float))
    print()
    print(f"[done] wrote {out_dir / 'ensemble_size_and_significance.json'}")
    print(f"[done] wrote {out_dir / 'ensemble_size_and_significance.png'}")


if __name__ == "__main__":
    main()
