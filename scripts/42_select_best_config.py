#!/usr/bin/env python
"""Compare DA configurations and name the best one. One table, one decision.

This is the script the configuration sweep needed. Scripts 36, 38, 40 and 41
each answer part of the question -- point skill, aggregation, structure,
significance -- and none of them ranks configurations or produces a verdict.
This does, on both criteria at once:

**Gauges are the truth for VALUE.** CRPS, mean and wet-day absolute error, bias
and correlation at WITHHELD stations, pooled over every fold of a configuration
so each station takes a turn being withheld.

**The products bound plausible STRUCTURE.** Spatial pattern correlation against
CHIRPS, IMERG and CPC, and wet-area fraction against the envelope the three of
them span. No single product is treated as correct -- they disagree too much for
that -- but a field that is unlike all three is placing rain where none of them
do, and a field with a wildly wrong wet-area fraction is implausible whatever
its point scores say.

**Differences are tested, not eyeballed.** Configurations are compared by a
paired bootstrap over station-days, matched FOLD BY FOLD so that both arms are
scored on identical withheld stations, and the per-sample differences are then
pooled. Earlier attempts to pair across configurations failed because different
folds withhold different stations; pairing within fold and pooling afterwards is
what makes the comparison valid. A configuration only counts as better if the
interval on the difference excludes zero.

Configurations are grouped by the parent directory of each dump, which is how
the submission wrappers separate them.

Example
-------
    python scripts/42_select_best_config.py \\
        --dumps 'data/processed/bmd_imerg_eval_screen_*/*.npz' \\
        --arm combined --reference s3r1 \\
        --out-dir data/processed/config_selection
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

WET_MM = 1.0
ARM_KEYS = {
    "background": "background_at_stations",
    "gauges": "gauge_analysis_at_stations",
    "satellite": "imerg_analysis_at_stations",
    "combined": "combined_analysis_at_stations",
}
ARM_FIELDS = {
    "background": "background",
    "gauges": "analysis_gauge",
    "satellite": "analysis_imerg",
    "combined": "analysis_combined",
}
PRODUCTS = {"CHIRPS": "chirps", "CPC": "condition", "IMERG": "imerg"}


def config_name(path: Path) -> str:
    """Configuration label from the dump's directory."""
    name = path.parent.name
    for prefix in ("bmd_imerg_eval_", "cpc_satellite_", "screen_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name or path.parent.name


def fold_key(path: Path) -> str:
    """Fold identifier, so the same fold can be matched across configurations."""
    match = re.search(r"fold[_-]?(\d+)", path.stem)
    return match.group(1) if match else path.stem


def crps_pointwise(members: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Fair CRPS per sample (Zamo and Naveau), so it can be pooled and bootstrapped."""
    m = members.shape[0]
    if m < 2:
        return np.abs(members[0] - truth)
    first = np.abs(members - truth[None]).mean(axis=0)
    spread = np.abs(members[:, None] - members[None, :]).sum(axis=(0, 1))
    return first - spread / (2 * m * (m - 1))


def read_dump(path: Path, arm: str) -> dict | None:
    """Withheld-station ensemble, truth and gridded fields from one fold."""
    z = np.load(path, allow_pickle=False)
    key = ARM_KEYS.get(arm)
    if key is None or key not in z or "eval_idx" not in z:
        return None
    evl = z["eval_idx"]
    block = np.moveaxis(z[key], 1, 0)[:, :, evl]
    members = block.reshape(block.shape[0], -1)
    truth = z["gauge_mm"][:, evl].ravel()
    ok = np.isfinite(truth) & np.all(np.isfinite(members), axis=0)

    out = {"members": members[:, ok], "truth": truth[ok],
           "n_members": int(members.shape[0])}

    field_key = ARM_FIELDS.get(arm)
    if field_key in z:
        analysis = np.nanmean(z[field_key], axis=1)
        valid = np.asarray(z["valid"]) > 0 if "valid" in z else \
            np.ones(analysis.shape[-2:], bool)
        nlat, nlon = analysis.shape[-2:]
        products = {}
        for label, k in PRODUCTS.items():
            if k not in z:
                continue
            f = np.asarray(z[k], float)
            if f.shape[-2:] != (nlat, nlon):
                factor = nlat // f.shape[-2]
                f = np.repeat(np.repeat(f, factor, axis=-2), factor, axis=-1)
                f = f[..., :nlat, :nlon]
            if f.shape[0] == analysis.shape[0] and np.isfinite(f).any():
                products[label] = f
        out.update({"analysis": analysis, "valid": valid, "products": products})
    return out


def spatial_metrics(folds: list) -> dict:
    """Pattern correlation with each product, and wet-area against their envelope."""
    correlations = defaultdict(list)
    analysis_wet, product_wet = [], defaultdict(list)
    for f in folds:
        if "analysis" not in f:
            continue
        valid = f["valid"]
        analysis_wet.append(float(np.nanmean(f["analysis"][:, valid] >= WET_MM)))
        for label, field in f["products"].items():
            product_wet[label].append(
                float(np.nanmean(field[:, valid] >= WET_MM)))
            for a, b in zip(f["analysis"], field):
                u, v = a[valid], b[valid]
                ok = np.isfinite(u) & np.isfinite(v)
                if ok.sum() > 10 and u[ok].std() > 0 and v[ok].std() > 0:
                    correlations[label].append(
                        float(np.corrcoef(u[ok], v[ok])[0, 1]))
    if not analysis_wet:
        return {}
    wet = float(np.mean(analysis_wet))
    envelope = [float(np.mean(v)) for v in product_wet.values()]
    return {
        "pattern_correlation": {k: float(np.mean(v)) for k, v in correlations.items()},
        "pattern_correlation_best": max(
            (float(np.mean(v)) for v in correlations.values()), default=float("nan")),
        "wet_area": wet,
        "wet_area_envelope": [min(envelope), max(envelope)] if envelope else None,
        "wet_area_inside": bool(envelope and min(envelope) <= wet <= max(envelope)),
    }


def gauge_metrics(folds: list) -> dict:
    """Point scores at withheld gauges, pooled over folds."""
    per_sample = np.concatenate([crps_pointwise(f["members"], f["truth"])
                                 for f in folds])
    mean = np.concatenate([f["members"].mean(axis=0) for f in folds])
    truth = np.concatenate([f["truth"] for f in folds])
    difference = mean - truth
    wet = truth >= WET_MM
    return {
        "n": int(truth.size),
        "n_wet": int(wet.sum()),
        "n_folds": len(folds),
        "members": folds[0]["n_members"],
        "crps": float(per_sample.mean()),
        "mae": float(np.mean(np.abs(difference))),
        "wet_mae": float(np.mean(np.abs(difference[wet]))) if wet.any() else float("nan"),
        "bias": float(np.mean(difference)),
        "correlation": float(np.corrcoef(mean, truth)[0, 1])
        if mean.std() > 0 and truth.std() > 0 else float("nan"),
    }


def paired_difference(a_folds: dict, b_folds: dict, n_boot=2000, seed=0,
                      ci=95.0) -> dict:
    """CRPS difference a - b, paired WITHIN each fold then pooled.

    Pairing has to happen fold by fold: fold 0 of one configuration and fold 3
    of another withhold different stations and are not comparable. Matching on
    the fold identifier and concatenating the per-sample differences afterwards
    keeps every comparison on identical withheld station-days while still using
    the whole sample.
    """
    shared = sorted(set(a_folds) & set(b_folds))
    pieces = []
    for key in shared:
        a, b = a_folds[key], b_folds[key]
        if a["truth"].size != b["truth"].size or not np.allclose(a["truth"], b["truth"]):
            continue
        pieces.append(crps_pointwise(a["members"], a["truth"])
                      - crps_pointwise(b["members"], b["truth"]))
    if not pieces:
        return {"comparable": False,
                "reason": "no fold withheld the same stations in both runs"}
    difference = np.concatenate(pieces)
    rng = np.random.default_rng(seed)
    n = difference.size
    draws = np.array([difference[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    low, high = np.percentile(draws, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return {"comparable": True, "n_folds": len(pieces), "n_samples": int(n),
            "difference": float(difference.mean()),
            "ci_low": float(low), "ci_high": float(high),
            "significant": bool(low > 0 or high < 0)}


# --------------------------------------------------------------------------


def plot_selection(summary: dict, comparisons: list, out_path: Path) -> None:
    plt = __import__("matplotlib")
    plt.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 140, "font.size": 9,
                         "axes.grid": True, "grid.alpha": 0.25,
                         "axes.spines.top": False, "axes.spines.right": False})
    names = list(summary)
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    axis = axes[0]
    y = np.arange(len(names))
    axis.barh(y, [summary[n]["gauge"]["crps"] for n in names], color="#c1440e")
    axis.set_yticks(y); axis.set_yticklabels(names, fontsize=7.5)
    axis.invert_yaxis()
    axis.set_xlabel("CRPS at withheld gauges (mm/day)")
    axis.set_title("Point skill (gauges are truth)", fontsize=9.5)

    axis = axes[1]
    products = sorted({p for n in names
                       for p in summary[n].get("spatial", {}).get(
                           "pattern_correlation", {})})
    width = 0.8 / max(len(products), 1)
    for i, p in enumerate(products):
        axis.barh(y + i * width,
                  [summary[n].get("spatial", {}).get(
                      "pattern_correlation", {}).get(p, np.nan) for n in names],
                  height=width, label=p)
    axis.set_yticks(y + 0.4 - width / 2); axis.set_yticklabels(names, fontsize=7.5)
    axis.invert_yaxis()
    axis.set_xlabel("mean daily spatial correlation")
    axis.set_title("Structure: pattern agreement with products", fontsize=9.5)
    axis.legend(fontsize=7)

    axis = axes[2]
    if comparisons:
        labels = [f"{c['a']}\n- {c['b']}" for c in comparisons]
        diff = [c["difference"] for c in comparisons]
        lo = [c["difference"] - c["ci_low"] for c in comparisons]
        hi = [c["ci_high"] - c["difference"] for c in comparisons]
        yy = np.arange(len(comparisons))
        axis.errorbar(diff, yy, xerr=[lo, hi], fmt="none", ecolor="#555555",
                      capsize=3)
        for i, c in enumerate(comparisons):
            axis.plot(diff[i], yy[i], "o", ms=8,
                      color="#c1440e" if c["significant"] else "#999999")
        axis.axvline(0.0, color="#111111", lw=1.4, ls="--")
        axis.set_yticks(yy); axis.set_yticklabels(labels, fontsize=6)
        axis.invert_yaxis()
        axis.set_xlabel("CRPS difference (mm/day), negative favours the first")
        axis.set_title("Paired within fold, 95% interval\n"
                       "grey = not distinguishable", fontsize=9.5)
    figure.suptitle("Configuration selection: gauges for value, products for "
                    "structure", y=1.02)
    figure.tight_layout()
    figure.savefig(out_path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank DA configurations on gauge skill and spatial structure",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dumps", nargs="+", required=True)
    parser.add_argument("--arm", default="combined", choices=sorted(ARM_KEYS))
    parser.add_argument("--reference", default=None,
                        help="configuration every other is compared against; "
                             "default is the one with the best CRPS")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    paths = sorted({Path(p) for pat in args.dumps for p in (glob.glob(pat) or [pat])})
    paths = [p for p in paths if p.suffix == ".npz" and p.exists()]
    if not paths:
        raise SystemExit(f"no NPZ dumps matched {args.dumps}")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    by_config = defaultdict(dict)
    for path in paths:
        block = read_dump(path, args.arm)
        if block is None or block["truth"].size < 5:
            print(f"[skip] {path.parent.name}/{path.name}: no usable arm "
                  f"{args.arm!r}")
            continue
        by_config[config_name(path)][fold_key(path)] = block

    if len(by_config) < 1:
        raise SystemExit("nothing to compare")
    print(f"[setup] {len(by_config)} configuration(s), arm {args.arm!r}")
    for name, folds in sorted(by_config.items()):
        total = sum(f["truth"].size for f in folds.values())
        print(f"    {name:24s} {len(folds)} fold(s), {total:,} withheld "
              f"station-days, {folds[next(iter(folds))]['n_members']} members")
    if len(by_config) == 1:
        print("\n    NOTE: only one configuration was found. If several were "
              "expected,\n    check the glob spans their directories -- each "
              "configuration writes to\n    its own.")

    summary = {}
    for name, folds in by_config.items():
        ordered = list(folds.values())
        summary[name] = {"gauge": gauge_metrics(ordered),
                         "spatial": spatial_metrics(ordered)}

    ranked = sorted(summary, key=lambda n: summary[n]["gauge"]["crps"])
    reference = args.reference if args.reference in summary else ranked[0]

    print()
    print("[table] gauges are TRUTH for value; products bound plausible STRUCTURE.")
    print(f"    {'config':22s} {'n':>6s} {'nwet':>5s} {'CRPS':>6s} {'MAE':>6s} "
          f"{'wetMAE':>7s} {'bias':>7s} {'corr':>5s} | {'patt':>5s} {'wetarea':>8s}")
    for name in ranked:
        g = summary[name]["gauge"]; s = summary[name].get("spatial", {})
        inside = s.get("wet_area_inside")
        flag = "in " if inside else ("OUT" if inside is not None else "  ?")
        print(f"    {name:22s} {g['n']:>6,d} {g['n_wet']:>5,d} {g['crps']:>6.2f} "
              f"{g['mae']:>6.2f} {g['wet_mae']:>7.2f} {g['bias']:>+7.2f} "
              f"{g['correlation']:>5.2f} | "
              f"{s.get('pattern_correlation_best', float('nan')):>5.2f} "
              f"{s.get('wet_area', float('nan')):>5.3f}{flag:>3s}")

    comparisons = []
    for name in ranked:
        if name == reference:
            continue
        result = paired_difference(by_config[name], by_config[reference],
                                   n_boot=args.n_boot, seed=args.seed)
        if result.get("comparable"):
            comparisons.append({"a": name, "b": reference, **result})
        else:
            print(f"[pair] {name} vs {reference}: {result['reason']}")

    print()
    print(f"[significance] every configuration against '{reference}', paired "
          f"within fold.")
    print("    Negative difference favours the first named configuration.")
    for c in comparisons:
        verdict = "SIGNIFICANT" if c["significant"] else "not distinguishable"
        print(f"    {c['a']:22s} {c['difference']:>+7.3f} "
              f"[{c['ci_low']:>+7.3f},{c['ci_high']:>+7.3f}]  "
              f"{c['n_folds']} fold(s), n={c['n_samples']:,}  {verdict}")

    beaten = [c["a"] for c in comparisons
              if c["significant"] and c["difference"] < 0]
    print()
    if beaten:
        best = min(beaten, key=lambda n: summary[n]["gauge"]["crps"])
        print(f"[verdict] BEST: {best} -- significantly better than "
              f"'{reference}' on withheld gauges.")
    else:
        print(f"[verdict] No configuration significantly beats '{reference}'.")
        print("    The lowest CRPS is "
              f"'{ranked[0]}', but the interval on its difference includes zero,")
        print("    so on this sample the configurations are NOT separable. "
              "Choosing on")
        print("    the point estimate alone is what produced the earlier "
              "rankings that")
        print("    later reversed. Either lengthen the window or accept that "
              "these")
        print("    settings do not matter as much as assumed.")
    structural = [n for n in ranked
                  if summary[n].get("spatial", {}).get("wet_area_inside")]
    if structural:
        print(f"    Structurally plausible (wet area inside the product "
              f"envelope): {', '.join(structural)}")
    else:
        print("    WARNING: no configuration has a wet-area fraction inside the "
              "product envelope.")

    plot_selection(summary, comparisons, out_dir / "config_selection.png")
    (out_dir / "config_selection.json").write_text(json.dumps(
        {"arm": args.arm, "reference": reference, "summary": summary,
         "comparisons": comparisons, "ranked_by_crps": ranked},
        indent=2, default=float))
    print()
    print(f"[done] wrote {out_dir / 'config_selection.json'}")
    print(f"[done] wrote {out_dir / 'config_selection.png'}")


if __name__ == "__main__":
    main()
