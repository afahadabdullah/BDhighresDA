#!/usr/bin/env python3
"""Rank the four frozen/latest V7 checkpoint pairs on the matched May 3 run.

All runs must contain the same dates, observations, station split, ensemble
size, sampling horizon, seed, and ``da_sim_r81`` arm.  The comparison uses fair
CRPS at the exact same withheld stations.  A station bootstrap is included as
a descriptive one-day uncertainty check; stations are spatially dependent, so
it must not be interpreted as a confirmatory confidence interval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


LABELS = ("frozen_frozen", "latest_frozen", "frozen_latest", "latest_latest")
ARM = "da_sim_r81"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True,
                        help="tournament root containing results/<label>")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-markdown", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20220503)
    return parser.parse_args()


def digest(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def fair_crps_per_station(members: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Fair CRPS for ``(member, station)`` preserving the station axis."""
    members = np.asarray(members, float)
    observed = np.asarray(observed, float)
    output = np.full(observed.shape, np.nan)
    keep = np.isfinite(observed) & np.all(np.isfinite(members), axis=0)
    if not keep.any():
        return output
    selected = members[:, keep]
    truth = observed[keep]
    count = selected.shape[0]
    first = np.mean(np.abs(selected - truth[None]), axis=0)
    if count == 1:
        output[keep] = first
        return output
    ordered = np.sort(selected, axis=0)
    weights = (2 * np.arange(1, count + 1) - count - 1)[:, None]
    spread = np.sum(weights * ordered, axis=0) / (count * (count - 1))
    output[keep] = first - spread
    return output


def same_values(first: np.ndarray, second: np.ndarray) -> bool:
    first, second = np.asarray(first), np.asarray(second)
    return (
        first.shape == second.shape
        and np.array_equal(np.isfinite(first), np.isfinite(second))
        and np.allclose(first[np.isfinite(first)], second[np.isfinite(second)], atol=1e-6)
    )


def load_run(root: Path, label: str) -> dict:
    directory = root / "results" / label
    report_path = directory / "v7_two_stage_real.json"
    station_path = directory / "station_ensembles.npz"
    if not report_path.is_file() or not station_path.is_file():
        raise FileNotFoundError(f"{label} lacks report or station dump in {directory}")
    report = json.loads(report_path.read_text())
    if ARM not in report.get("arms", {}):
        raise ValueError(f"{label} report lacks {ARM}")
    with np.load(station_path, allow_pickle=False) as archive:
        required = (
            "times", "model_times", "station_ids", "eval_idx", "assim_idx",
            "observed_mm", f"station_{ARM}",
        )
        missing = [name for name in required if name not in archive]
        if missing:
            raise ValueError(f"{station_path} lacks {missing}")
        held = np.asarray(archive["eval_idx"], int)
        ensemble = np.asarray(archive[f"station_{ARM}"], float)
        observed = np.asarray(archive["observed_mm"], float)
        if ensemble.shape[0] != 1 or observed.shape[0] != 1:
            raise ValueError(f"{label} is not a one-day experiment")
        # Slice the day first.  ``ensemble[0, :, held]`` invokes NumPy advanced
        # indexing and moves the station axis in front, producing
        # (station, member) rather than the required (member, station).
        station_crps = fair_crps_per_station(ensemble[0][:, held], observed[0, held])
        arrays = {
            name: np.asarray(archive[name])
            for name in ("times", "model_times", "station_ids", "eval_idx", "assim_idx")
        }
        arrays["observed_mm"] = observed
    checkpoint = {}
    for stage in ("meso", "allocation"):
        info = report["checkpoints"][stage]
        checkpoint[stage] = {
            "source": info["source"],
            "frozen": info["frozen"],
            "sha256": digest(info["frozen"]),
            "epoch": info.get("epoch"),
            "best_val": info.get("best_val"),
            "weights": info.get("weights"),
        }
    return {
        "label": label,
        "directory": str(directory),
        "report": report,
        "arrays": arrays,
        "station_crps": station_crps,
        "checkpoints": checkpoint,
    }


def validate_matched(runs: dict[str, dict]) -> None:
    reference = runs["frozen_frozen"]
    fixed_report_keys = (
        "model_dates", "gauge_dates", "members", "n_steps", "seed",
        "observations", "gauge_day_offset", "imerg_day_offset",
        "meso_gauge_sigma_transformed", "meso_gauge_representativeness",
        "fine_gauge_sigma_mm",
    )
    for label, run in runs.items():
        for key in fixed_report_keys:
            if run["report"].get(key) != reference["report"].get(key):
                raise ValueError(
                    f"{label} differs from frozen_frozen on {key}: "
                    f"{run['report'].get(key)!r} vs {reference['report'].get(key)!r}"
                )
        for key in ("times", "model_times", "station_ids", "eval_idx", "assim_idx"):
            if not np.array_equal(run["arrays"][key], reference["arrays"][key]):
                raise ValueError(f"{label} differs from frozen_frozen on {key}")
        if not same_values(run["arrays"]["observed_mm"], reference["arrays"]["observed_mm"]):
            raise ValueError(f"{label} BMD observations differ from frozen_frozen")
        multiplier = run["report"].get("arm_imerg_r", {}).get(ARM)
        if float(multiplier) != 81.0:
            raise ValueError(f"{label} {ARM} has R multiplier {multiplier}, not 81")


def paired_bootstrap(
    candidate: np.ndarray,
    reference: np.ndarray,
    resamples: int,
    seed: int,
) -> dict:
    """Bootstrap candidate-reference CRPS over the one day's held-out stations."""
    candidate, reference = np.asarray(candidate, float), np.asarray(reference, float)
    keep = np.isfinite(candidate) & np.isfinite(reference)
    difference = candidate[keep] - reference[keep]
    if not difference.size:
        return {"n": 0}
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(difference), size=(resamples, len(difference)))
    estimates = difference[index].mean(axis=1)
    low, high = np.percentile(estimates, [2.5, 97.5])
    return {
        "n": int(len(difference)),
        "candidate_minus_reference_crps_mm": float(difference.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "resamples": int(resamples),
        "unit": "withheld station",
        "warning": "descriptive only; stations are spatially dependent and there is one day",
    }


def score(root: Path, resamples: int, seed: int) -> dict:
    runs = {label: load_run(root, label) for label in LABELS}
    validate_matched(runs)
    reference = runs["frozen_frozen"]
    rows = []
    for index, label in enumerate(LABELS):
        run = runs[label]
        metrics = run["report"]["arms"][ARM]["mean"]
        rows.append({
            "label": label,
            "crps_mm": float(metrics["crps_mm"]),
            "mae_mm": float(metrics["mae_mm"]),
            "bias_mm": float(metrics["bias_mm"]),
            "rmse_mm": float(metrics["rmse_mm"]),
            "spread_mm": float(metrics["spread_mm"]),
            "spread_skill": float(metrics["spread_skill"]),
            "paired_vs_frozen_frozen": paired_bootstrap(
                run["station_crps"], reference["station_crps"],
                resamples, seed + 1000 * index,
            ),
            "checkpoints": run["checkpoints"],
        })
    winner = min(rows, key=lambda row: row["crps_mm"])
    tolerance = max(0.05, 0.01 * winner["crps_mm"])
    ties = [row["label"] for row in rows if row["crps_mm"] - winner["crps_mm"] <= tolerance]
    by_label = {row["label"]: row for row in rows}
    base = by_label["frozen_frozen"]["crps_mm"]
    meso = by_label["latest_frozen"]["crps_mm"] - base
    allocation = by_label["frozen_latest"]["crps_mm"] - base
    combined = by_label["latest_latest"]["crps_mm"] - base
    return {
        "arm": ARM,
        "scope": {
            "model_dates": reference["report"]["model_dates"],
            "gauge_dates": reference["report"]["gauge_dates"],
            "members": reference["report"]["members"],
            "n_steps": reference["report"]["n_steps"],
            "seed": reference["report"].get("seed"),
            "withheld_stations": int(np.isfinite(reference["station_crps"]).sum()),
        },
        "rows": rows,
        "nominal_winner": winner["label"],
        "practical_tie_tolerance_mm": tolerance,
        "practical_ties": ties,
        "factorial_crps_effects_mm": {
            "latest_meso_with_frozen_allocation": meso,
            "latest_allocation_with_frozen_meso": allocation,
            "latest_pair": combined,
            "interaction": combined - meso - allocation,
        },
        "interpretation": (
            "This is a locked one-day checkpoint screen, not final model selection. "
            "Confirm the nominal winner over an untouched multi-day window before replacing production."
        ),
    }


def markdown(result: dict) -> str:
    lines = [
        "# V7 frozen-versus-latest checkpoint tournament",
        "",
        f"Primary arm: `{result['arm']}`.",
        "",
        "| Pair | Meso epoch | Allocation epoch | CRPS | MAE | Bias | RMSE | Spread/RMSE | ΔCRPS vs frozen |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for row in result["rows"]:
        delta = row["paired_vs_frozen_frozen"].get(
            "candidate_minus_reference_crps_mm", float("nan")
        )
        lines.append(
            f"| `{row['label']}` | {row['checkpoints']['meso']['epoch']} | "
            f"{row['checkpoints']['allocation']['epoch']} | {row['crps_mm']:.3f} | "
            f"{row['mae_mm']:.3f} | {row['bias_mm']:+.3f} | {row['rmse_mm']:.3f} | "
            f"{row['spread_skill']:.2f} | {delta:+.3f} |"
        )
    effects = result["factorial_crps_effects_mm"]
    lines += [
        "",
        f"Nominal one-day winner: **`{result['nominal_winner']}`**.",
        f"Practical ties (within {result['practical_tie_tolerance_mm']:.3f} mm): "
        + ", ".join(f"`{label}`" for label in result["practical_ties"]) + ".",
        "",
        "## Factorial attribution",
        "",
        f"- Latest meso effect with frozen allocation: {effects['latest_meso_with_frozen_allocation']:+.3f} mm CRPS.",
        f"- Latest allocation effect with frozen meso: {effects['latest_allocation_with_frozen_meso']:+.3f} mm CRPS.",
        f"- Latest complete-pair effect: {effects['latest_pair']:+.3f} mm CRPS.",
        f"- Meso/allocation interaction: {effects['interaction']:+.3f} mm CRPS.",
        "",
        "Negative effects favour the latest checkpoint.",
        "",
        "## Decision rule",
        "",
        result["interpretation"],
        "The paired intervals resample the eleven withheld stations and are descriptive only; "
        "one day cannot represent weather-regime uncertainty.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    result = score(Path(args.root), args.bootstrap_resamples, args.seed)
    out_json, out_markdown = Path(args.out_json), Path(args.out_markdown)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_markdown.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2))
    out_markdown.write_text(markdown(result))
    print(markdown(result), flush=True)
    print(f"[checkpoint tournament] wrote {out_markdown}", flush=True)


if __name__ == "__main__":
    main()
