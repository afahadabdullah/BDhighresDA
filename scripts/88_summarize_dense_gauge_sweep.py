#!/usr/bin/env python
"""Compare dense-network gauge DA methods on one short window.

Read this before reading the table
----------------------------------
Five days cannot resolve a small CRPS difference, and this script says so in
its own output rather than leaving the reader to assume otherwise.  What five
days CAN resolve, and what this experiment is actually for, is:

* a field that is visibly speckled versus one that is not (``speckle``),
* an increment pinned to gauges versus one that propagates (``locality_ratio``),
* whether the guidance norm cap is binding (``clip_fraction``), which decides
  whether the run is even executing the method its config describes,
* a bias or a wet-day frequency that moves by more than a millimetre.

Every arm inside one profile shares the checkpoint, the background draw, the
day seeds, the observation perturbations and the withheld stations, so a
within-profile difference is the method.  Profiles differ only by the config
override named in their label, so a between-profile difference at the same arm
is that override.

Scores are computed at withheld stations only, joined by station id, so a
profile that changes the assimilated station table (super-obbing) is still
scored against exactly the same withheld gauges as every other profile.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", action="append", required=True, metavar="LABEL=PREFIX",
        help=(
            "one profile per flag, as label=path/prefix; the script reads "
            "PREFIX.npz and PREFIX.json"
        ),
    )
    parser.add_argument("--station-summary", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--reference-arm", default="prod_huber3",
        help="arm every other arm is compared against, within each profile",
    )
    parser.add_argument(
        "--reference-profile", default=None,
        help="profile holding the global control (default: the first --profile)",
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20220501)
    parser.add_argument("--no-figure", action="store_true")
    return parser.parse_args()


# ------------------------------------------------------------------ scoring
def fair_crps(members: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Fair (unbiased) ensemble CRPS, matching scripts 83/84."""
    members, truth = np.asarray(members, float), np.asarray(truth, float)
    result = np.full(len(truth), np.nan)
    valid = np.isfinite(truth) & np.all(np.isfinite(members), axis=1)
    if not valid.any():
        return result
    selected, observed = members[valid], truth[valid]
    count = selected.shape[1]
    first = np.mean(np.abs(selected - observed[:, None]), axis=1)
    ordered = np.sort(selected, axis=1)
    pair = np.sum(
        ordered * (2 * np.arange(1, count + 1) - count - 1)[None, :], axis=1
    )
    result[valid] = first - pair / (count * (count - 1))
    return result


def score(members: np.ndarray, truth: np.ndarray) -> dict:
    valid = np.isfinite(truth) & np.all(np.isfinite(members), axis=1)
    members, truth = members[valid], truth[valid]
    if not len(truth):
        return {"n": 0}
    mean = members.mean(axis=1)
    error = mean - truth
    return {
        "n": int(len(truth)),
        "mae_mm": float(np.mean(np.abs(error))),
        "rmse_mm": float(np.sqrt(np.mean(error**2))),
        "bias_mm": float(np.mean(error)),
        "crps_mm": float(np.nanmean(fair_crps(members, truth))),
        "correlation": (
            float(np.corrcoef(mean, truth)[0, 1])
            if mean.std() and truth.std()
            else None
        ),
    }


# ------------------------------------------------------------- field shape
def speckle(field: np.ndarray, valid: np.ndarray) -> float:
    """Mean |discrete Laplacian| over interior land, normalised by the mean.

    This is the quantitative form of "the map looks spotty".  A point-gauge
    likelihood that paints each station's own reporting error into the field
    raises it; a field whose structure is meteorological does not.  It is
    reported alongside the same statistic for CHIRPS on the same days, because
    the number only means something relative to what an observed field scores.
    """
    field = np.asarray(field, float)
    interior = (
        valid
        & np.roll(valid, 1, 0) & np.roll(valid, -1, 0)
        & np.roll(valid, 1, 1) & np.roll(valid, -1, 1)
    )
    interior[0, :] = interior[-1, :] = interior[:, 0] = interior[:, -1] = False
    days = field if field.ndim == 3 else field[None]
    total = []
    for day in days:
        filled = np.nan_to_num(day, nan=0.0)
        laplacian = (
            np.roll(filled, 1, 0) + np.roll(filled, -1, 0)
            + np.roll(filled, 1, 1) + np.roll(filled, -1, 1)
            - 4.0 * filled
        )
        magnitude = np.mean(np.abs(laplacian[interior]))
        scale = np.mean(filled[valid])
        total.append(magnitude / scale if scale > 1e-6 else np.nan)
    return float(np.nanmean(total))


def locality_ratio(
    analysis: np.ndarray, background: np.ndarray, distance_km: np.ndarray,
    valid: np.ndarray, edges=(0.0, 10.0, 25.0, 50.0, 100.0, 1.0e6),
) -> dict:
    """Time-mean |analysis - background| binned by distance to nearest gauge.

    A method that carries gauge information along meteorological structure gives
    a nearly flat curve; a method that grows discs around gauges gives one that
    falls steeply.  ``ratio`` is the innermost bin over the outermost populated
    bin.  Mirrors ``increment_locality`` in scripts/28.
    """
    increment = np.abs(np.nanmean(analysis - background, axis=0))
    edges = np.asarray(edges, float)
    values, counts = [], []
    for lower, upper in zip(edges[:-1], edges[1:]):
        inside = valid & (distance_km >= lower) & (distance_km < upper)
        counts.append(int(inside.sum()))
        values.append(float(np.nanmean(increment[inside])) if inside.any() else np.nan)
    populated = [v for v, c in zip(values, counts) if c > 0 and np.isfinite(v)]
    return {
        "bins_mm": [None if not np.isfinite(v) else round(v, 4) for v in values],
        "bin_counts": counts,
        "bin_edges_km": edges[:-1].tolist() + ["inf"],
        "ratio": (
            float(populated[0] / populated[-1])
            if len(populated) >= 2 and populated[-1] > 1e-9
            else None
        ),
    }


# --------------------------------------------------------------- reporting
def clip_summary(report: dict, arm: str) -> dict:
    """Fraction of guided member-steps whose gradient hit ``guidance.clip_norm``.

    A value near 1 means the analysis is running against the cap on essentially
    every step: the increment has stopped scaling with the innovation and the
    method being executed is not the one the config describes.
    """
    entries = report.get("sampler_diagnostics", {}).get(arm, [])
    calls = members = clipped = 0
    norm_sum = 0.0
    norm_max = 0.0
    components = None
    clip_norm = None
    for entry in entries:
        record = (entry or {}).get("guidance")
        if not record:
            continue
        calls += record.get("calls", 0)
        members += record.get("members", 0)
        clipped += record.get("clipped_members", 0)
        norm_sum += record.get("pre_clip_norm_sum", 0.0)
        norm_max = max(norm_max, record.get("pre_clip_norm_max", 0.0))
        clip_norm = record.get("clip_norm", clip_norm)
        sums = record.get("component_norm_sums")
        if sums:
            components = (
                [a + b for a, b in zip(components, sums)] if components else list(sums)
            )
    if not members:
        return {"available": False}
    result = {
        "available": True,
        "clip_norm": clip_norm,
        "clip_fraction": clipped / members,
        "mean_pre_clip_norm": norm_sum / members,
        "max_pre_clip_norm": norm_max,
    }
    if components:
        # Component order follows CompositeObsOperator: gauges, then IMERG.
        labels = ["gauge", "imerg"][: len(components)]
        total = sum(components) or 1.0
        result["component_gradient_share"] = {
            label: round(value / total, 4) for label, value in zip(labels, components)
        }
    return result


def paired_bootstrap(
    per_day_reference: np.ndarray, per_day_arm: np.ndarray, draws: int, rng
) -> dict:
    """Day-block bootstrap of the paired CRPS difference (arm minus reference).

    Blocks are whole days because station-days within a day are not independent.
    With a five-day window there are five blocks, so the interval is wide by
    construction; that is the honest answer, not a defect to tune away.
    """
    difference = per_day_arm - per_day_reference
    finite = np.isfinite(difference)
    if finite.sum() < 2:
        return {"mean": None, "ci95": None, "n_days": int(finite.sum())}
    values = difference[finite]
    index = rng.integers(0, len(values), size=(draws, len(values)))
    samples = values[index].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
        "n_days": int(len(values)),
        "per_day": [round(float(v), 4) for v in difference],
    }


def load_profile(label: str, prefix: str, source_map: dict) -> dict:
    dump = np.load(f"{prefix}.npz", allow_pickle=False)
    report = json.loads(Path(f"{prefix}.json").read_text())
    ids = dump["station_ids"].astype(str)
    eval_idx = np.asarray(dump["eval_idx"], int)
    eval_ids = ids[eval_idx]
    times = dump["times"].astype("datetime64[D]")
    arms = dump["variant_names"].astype(str).tolist()
    truth = np.asarray(dump["gauge_mm"][:, eval_idx], float)  # (T, S_eval)
    ensembles = {
        arm: np.asarray(dump[f"station_{arm}"][:, :, eval_idx], float)  # (T, M, S)
        for arm in arms
    }
    fields = {arm: np.asarray(dump[f"meanfield_{arm}"], float) for arm in arms}
    payload = {
        "label": label,
        "prefix": prefix,
        "arms": arms,
        "times": times,
        "eval_ids": eval_ids,
        "truth": truth,
        "ensembles": ensembles,
        "fields": fields,
        "valid": np.asarray(dump["valid"], bool),
        "distance_km": np.asarray(dump["distance_km"], float),
        "chirps": np.asarray(dump["chirps"], float),
        "report": report,
        "n_assimilated": int(len(np.asarray(dump["assim_idx"], int))),
    }
    dump.close()
    unknown = sorted(set(eval_ids) - set(source_map))
    if unknown:
        raise SystemExit(
            f"{label}: {len(unknown)} withheld station(s) missing from the station "
            f"summary, e.g. {unknown[:5]}"
        )
    return payload


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    summary_table = pd.read_csv(args.station_summary)
    source_map = (
        summary_table.set_index("station_id")["source"].astype(str).to_dict()
    )
    manifest = json.loads(Path(args.manifest).read_text())

    profiles = []
    for item in args.profile:
        if "=" not in item:
            raise SystemExit(f"--profile expects label=prefix, got {item!r}")
        label, prefix = item.split("=", 1)
        profiles.append(load_profile(label.strip(), prefix.strip(), source_map))

    # Every profile must score the same withheld gauges on the same days, or the
    # comparison is between different experiments.
    reference = profiles[0]
    for profile in profiles[1:]:
        if not np.array_equal(profile["times"], reference["times"]):
            raise SystemExit(f"{profile['label']}: dates differ from {reference['label']}")
        if set(profile["eval_ids"]) != set(reference["eval_ids"]):
            raise SystemExit(
                f"{profile['label']}: withheld station set differs from "
                f"{reference['label']}; the profiles are not comparable"
            )

    reference_profile = args.reference_profile or profiles[0]["label"]
    by_label = {profile["label"]: profile for profile in profiles}
    if reference_profile not in by_label:
        raise SystemExit(f"--reference-profile {reference_profile!r} was not supplied")

    def per_day_crps(profile: dict, arm: str) -> np.ndarray:
        """Mean CRPS on each day, for the day-block bootstrap."""
        truth = profile["truth"]
        ensemble = profile["ensembles"][arm]
        out = np.full(truth.shape[0], np.nan)
        for day in range(truth.shape[0]):
            values = fair_crps(ensemble[day].T, truth[day])
            if np.isfinite(values).any():
                out[day] = float(np.nanmean(values))
        return out

    global_control_profile = by_label[reference_profile]
    if args.reference_arm not in global_control_profile["arms"]:
        raise SystemExit(
            f"reference arm {args.reference_arm!r} absent from profile "
            f"{reference_profile!r}"
        )
    global_control = per_day_crps(global_control_profile, args.reference_arm)

    sources = np.array([source_map[i] for i in reference["eval_ids"]])
    results: dict = {
        "period": {
            "start": str(reference["times"][0]),
            "end": str(reference["times"][-1]),
            "days": int(len(reference["times"])),
        },
        "withheld_stations": int(len(reference["eval_ids"])),
        "withheld_by_source": {
            str(k): int(v) for k, v in pd.Series(sources).value_counts().items()
        },
        "holdout_design": manifest.get("analysis_selection", {}),
        "reference": {"profile": reference_profile, "arm": args.reference_arm},
        "profiles": {},
        "interpretation": (
            "A five-day window resolves speckle, increment locality, clip "
            "saturation and bias. It does not resolve small CRPS differences: "
            "read the bootstrap interval before calling any CRPS gap real."
        ),
    }

    rows = []
    for profile in profiles:
        label = profile["label"]
        background = profile["fields"].get("background")
        chirps_speckle = speckle(profile["chirps"], profile["valid"])
        control = (
            per_day_crps(profile, args.reference_arm)
            if args.reference_arm in profile["arms"]
            else None
        )
        entry = {
            "prefix": profile["prefix"],
            "assimilated_stations": profile["n_assimilated"],
            "config_overrides": profile["report"].get("config_overrides", []),
            "chirps_speckle": round(chirps_speckle, 4),
            "arms": {},
        }
        for arm in profile["arms"]:
            flat_truth = profile["truth"].reshape(-1)
            flat_members = np.moveaxis(profile["ensembles"][arm], 1, 2).reshape(
                -1, profile["ensembles"][arm].shape[1]
            )
            flat_source = np.tile(sources, profile["truth"].shape[0])
            arm_scores = {
                "pooled": score(flat_members, flat_truth),
                "BMD": score(
                    flat_members[flat_source == "BMD"], flat_truth[flat_source == "BMD"]
                ),
                "BWDB": score(
                    flat_members[flat_source == "BWDB"],
                    flat_truth[flat_source == "BWDB"],
                ),
            }
            field_stats = {
                "speckle": round(speckle(profile["fields"][arm], profile["valid"]), 4),
                "speckle_over_chirps": (
                    round(
                        speckle(profile["fields"][arm], profile["valid"]) / chirps_speckle,
                        3,
                    )
                    if chirps_speckle and np.isfinite(chirps_speckle)
                    else None
                ),
            }
            if background is not None and arm != "background":
                field_stats["locality"] = locality_ratio(
                    profile["fields"][arm], background,
                    profile["distance_km"], profile["valid"],
                )
            arm_days = per_day_crps(profile, arm)
            entry["arms"][arm] = {
                "scores": arm_scores,
                "field": field_stats,
                "guidance": clip_summary(profile["report"], arm),
                "vs_global_control": paired_bootstrap(
                    global_control, arm_days, args.bootstrap, rng
                ),
                "vs_profile_control": (
                    paired_bootstrap(control, arm_days, args.bootstrap, rng)
                    if control is not None
                    else None
                ),
            }
            rows.append(
                {
                    "profile": label,
                    "arm": arm,
                    "pool_crps": arm_scores["pooled"].get("crps_mm"),
                    "pool_rmse": arm_scores["pooled"].get("rmse_mm"),
                    "pool_bias": arm_scores["pooled"].get("bias_mm"),
                    "bmd_crps": arm_scores["BMD"].get("crps_mm"),
                    "bwdb_crps": arm_scores["BWDB"].get("crps_mm"),
                    "speckle_ratio": field_stats["speckle_over_chirps"],
                    "locality_ratio": (
                        field_stats.get("locality", {}).get("ratio")
                        if arm != "background"
                        else None
                    ),
                    "clip_fraction": entry["arms"][arm]["guidance"].get("clip_fraction"),
                    "d_crps_vs_control": entry["arms"][arm]["vs_global_control"]["mean"],
                    "d_crps_ci_low": (
                        entry["arms"][arm]["vs_global_control"]["ci95"] or [None, None]
                    )[0],
                    "d_crps_ci_high": (
                        entry["arms"][arm]["vs_global_control"]["ci95"] or [None, None]
                    )[1],
                }
            )
        results["profiles"][label] = entry

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dense_gauge_sweep.json").write_text(json.dumps(results, indent=2) + "\n")
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "dense_gauge_sweep.csv", index=False)

    ranked = frame.dropna(subset=["pool_crps"]).sort_values("pool_crps")
    lines = [
        "# Dense BMD+BWDB gauge DA method sweep",
        "",
        f"Period {results['period']['start']} to {results['period']['end']} "
        f"({results['period']['days']} days). "
        f"{results['withheld_stations']} withheld stations "
        f"({results['withheld_by_source']}).",
        "",
        f"Control: `{args.reference_arm}` in profile `{reference_profile}` "
        "(the frozen production contract). `d CRPS` is arm minus that control; "
        "negative is better.",
        "",
        "**" + results["interpretation"] + "**",
        "",
        "| Profile | Arm | Pool CRPS | d CRPS | 95% CI | BMD CRPS | BWDB CRPS | Bias | Speckle/CHIRPS | Locality | Clip frac |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]

    def fmt(value, digits=3):
        return "—" if value is None or (isinstance(value, float) and not np.isfinite(value)) else f"{value:.{digits}f}"

    for _, row in ranked.iterrows():
        interval = (
            f"[{fmt(row['d_crps_ci_low'])}, {fmt(row['d_crps_ci_high'])}]"
            if row["d_crps_ci_low"] is not None
            else "—"
        )
        lines.append(
            f"| {row['profile']} | `{row['arm']}` | {fmt(row['pool_crps'])} | "
            f"{fmt(row['d_crps_vs_control'])} | {interval} | {fmt(row['bmd_crps'])} | "
            f"{fmt(row['bwdb_crps'])} | {fmt(row['pool_bias'], 2)} | "
            f"{fmt(row['speckle_ratio'], 2)} | {fmt(row['locality_ratio'], 2)} | "
            f"{fmt(row['clip_fraction'], 2)} |"
        )

    lines += [
        "",
        "## How to read the diagnostic columns",
        "",
        "* **Speckle/CHIRPS** — mean |Laplacian| of the analysis mean field over "
        "its own mean, divided by the same statistic for CHIRPS on the same days. "
        "1.0 means the analysis is as smooth as an observed field; values well "
        "above 1 are the number behind \"the map looks spotty\".",
        "* **Locality** — time-mean |analysis − background| in the 0–10 km "
        "distance-to-gauge bin over the outermost bin. Large means the increment "
        "is pinned to gauges and does not propagate.",
        "* **Clip frac** — fraction of guided member-steps whose likelihood "
        "gradient hit `guidance.clip_norm`. Near 1 means the increment has "
        "stopped scaling with the innovation and the run is not executing the "
        "method its config names.",
        "",
        "## Per-profile guidance-gradient balance",
        "",
        "| Profile | Arm | Mean pre-clip norm | Clip norm | Gauge share | IMERG share |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for label, entry in results["profiles"].items():
        for arm, payload in entry["arms"].items():
            guidance = payload["guidance"]
            if not guidance.get("available"):
                continue
            share = guidance.get("component_gradient_share", {})
            lines.append(
                f"| {label} | `{arm}` | {fmt(guidance['mean_pre_clip_norm'], 1)} | "
                f"{fmt(guidance.get('clip_norm'), 0)} | "
                f"{fmt(share.get('gauge'), 3)} | {fmt(share.get('imerg'), 3)} |"
            )

    (out_dir / "dense_gauge_sweep.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    if not args.no_figure:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as error:  # pragma: no cover
            print(f"[summary] figure skipped: {error}")
            return
        plot = ranked.dropna(subset=["speckle_ratio"])
        figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        labels = [f"{r.profile}/{r.arm}" for r in plot.itertuples()]
        axes[0].barh(labels, plot["pool_crps"], color="#4C78A8")
        axes[0].set_xlabel("pooled CRPS at withheld gauges (mm/day)")
        axes[0].invert_yaxis()
        axes[0].set_title("Skill (lower is better)")
        axes[1].scatter(
            plot["speckle_ratio"], plot["pool_crps"], c="#E45756", zorder=3
        )
        for row in plot.itertuples():
            axes[1].annotate(
                f"{row.profile}/{row.arm}",
                (row.speckle_ratio, row.pool_crps),
                fontsize=7, xytext=(4, 3), textcoords="offset points",
            )
        axes[1].axvline(1.0, color="0.4", linestyle="--", linewidth=1)
        axes[1].set_xlabel("field roughness / CHIRPS roughness")
        axes[1].set_ylabel("pooled CRPS (mm/day)")
        axes[1].set_title("Skill against field smoothness")
        axes[1].grid(alpha=0.3)
        figure.suptitle(
            f"Dense BMD+BWDB gauge DA sweep, {results['period']['start']} to "
            f"{results['period']['end']}"
        )
        figure.tight_layout()
        figure.savefig(out_dir / "dense_gauge_sweep.png", dpi=150)
        print(f"[summary] wrote {out_dir / 'dense_gauge_sweep.png'}")


if __name__ == "__main__":
    main()
