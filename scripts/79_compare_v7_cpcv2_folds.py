#!/usr/bin/env python3
"""Pool five matched June folds into one V7-R81 versus CPCv2 comparison.

Each station is scored only in the fold where it was withheld.  CPCv2 dumps may
cover a longer season; ``73_compare_v7_cpcv2_day.py`` first subsets them to the
V7 observation dates and audits dates, coordinates, observations, station pool,
withheld IDs and ensemble shape.  This script then pools the raw station-day
samples and bootstraps the 30 daily paired CRPS differences.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "_v7_cpcv2_compare", ROOT / "scripts" / "73_compare_v7_cpcv2_day.py"
)
COMPARE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = COMPARE
SPEC.loader.exec_module(COMPARE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v7-dumps", nargs=5, required=True)
    parser.add_argument("--cpcv2-dumps", nargs=5, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=202306)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-markdown", required=True)
    return parser.parse_args()


def sample_arrays(members: np.ndarray, observed: np.ndarray):
    ensemble = np.moveaxis(np.asarray(members, float), 1, 0)  # M,T,S
    truth = np.asarray(observed, float)
    keep = np.isfinite(truth) & np.all(np.isfinite(ensemble), axis=0)
    selected = ensemble[:, keep]
    target = truth[keep]
    if selected.shape[0] < 2 or not selected.shape[1]:
        raise ValueError("each fold needs finite withheld samples and >=2 members")
    count = selected.shape[0]
    first = np.mean(np.abs(selected - target[None]), axis=0)
    second = np.abs(selected[:, None] - selected[None, :]).sum(axis=(0, 1))
    fair = first - second / (2.0 * count * (count - 1))
    day_index = np.broadcast_to(
        np.arange(truth.shape[0])[:, None], truth.shape
    )[keep]
    return selected, target, fair, day_index


def pooled_score(members: np.ndarray, truth: np.ndarray) -> dict:
    mean = members.mean(axis=0)
    rmse = float(np.sqrt(np.mean((mean - truth) ** 2)))
    spread = float(np.sqrt(np.mean(members.var(axis=0, ddof=1))))
    count = members.shape[0]
    fair = np.mean(np.abs(members - truth[None]), axis=0)
    fair -= np.abs(members[:, None] - members[None, :]).sum(axis=(0, 1)) / (
        2.0 * count * (count - 1)
    )
    low, high = np.quantile(members, [0.05, 0.95], axis=0)
    return {
        "station_days": int(len(truth)),
        "crps_mm": float(fair.mean()),
        "mae_mm": float(np.mean(np.abs(mean - truth))),
        "bias_mm": float(np.mean(mean - truth)),
        "rmse_mm": rmse,
        "spread_mm": spread,
        "spread_skill": float(spread / rmse) if rmse else None,
        "coverage_90": float(np.mean((truth >= low) & (truth <= high))),
    }


def compare_folds(v7_paths: list[Path], cpc_paths: list[Path],
                  resamples: int, seed: int) -> dict:
    pools: dict[str, dict[str, list[np.ndarray]]] = {}
    all_eval_ids: list[str] = []
    station_pool: set[str] | None = None
    dates: np.ndarray | None = None
    comparison_specs = None

    for fold, (v7_path, cpc_path) in enumerate(zip(v7_paths, cpc_paths)):
        with np.load(v7_path, allow_pickle=False) as v7, np.load(
            cpc_path, allow_pickle=False
        ) as cpc:
            aligned = COMPARE._align_cpc_to_v7(v7, cpc, v7_path, cpc_path)
            fold_dates, ids, eval_idx, observed, members, specs = aligned
            if dates is None:
                dates = fold_dates
                station_pool = set(ids.tolist())
                comparison_specs = specs
            elif not np.array_equal(dates, fold_dates):
                raise ValueError(f"fold {fold} has different dates")
            elif station_pool != set(ids.tolist()):
                raise ValueError(f"fold {fold} has a different station pool")
            elif comparison_specs != specs:
                raise ValueError(f"fold {fold} exposes different comparison arms")

            eval_ids = ids[eval_idx].astype(str).tolist()
            overlap = sorted(set(eval_ids) & set(all_eval_ids))
            if overlap:
                raise ValueError(f"stations withheld in more than one fold: {overlap}")
            all_eval_ids.extend(eval_ids)

            for label in specs:
                target = observed[:, eval_idx]
                v7_selected, truth, v7_fair, day_index = sample_arrays(
                    members[f"v7_{label}"][:, :, eval_idx], target
                )
                cpc_selected, cpc_truth, cpc_fair, cpc_day_index = sample_arrays(
                    members[f"cpcv2_{label}"][:, :, eval_idx], target
                )
                if not np.array_equal(truth, cpc_truth) or not np.array_equal(
                    day_index, cpc_day_index
                ):
                    raise RuntimeError(f"fold {fold} {label}: paired samples differ")
                bucket = pools.setdefault(label, {
                    "v7_members": [], "cpc_members": [], "truth": [],
                    "v7_fair": [], "cpc_fair": [], "day_index": [],
                })
                bucket["v7_members"].append(v7_selected)
                bucket["cpc_members"].append(cpc_selected)
                bucket["truth"].append(truth)
                bucket["v7_fair"].append(v7_fair)
                bucket["cpc_fair"].append(cpc_fair)
                bucket["day_index"].append(day_index)

    if station_pool != set(all_eval_ids):
        missing = sorted(station_pool - set(all_eval_ids))
        raise ValueError(f"five folds do not withhold every station exactly once: {missing}")

    rng = np.random.default_rng(seed)
    results = {}
    for label, bucket in pools.items():
        combined = {key: np.concatenate(value, axis=1 if key.endswith("members") else 0)
                    for key, value in bucket.items()}
        v7_score = pooled_score(combined["v7_members"], combined["truth"])
        cpc_score = pooled_score(combined["cpc_members"], combined["truth"])
        paired = combined["v7_fair"] - combined["cpc_fair"]
        daily = np.asarray([
            paired[combined["day_index"] == day].mean()
            for day in range(len(dates))
            if np.any(combined["day_index"] == day)
        ])
        if not len(daily):
            raise ValueError(f"{label}: no day has a finite paired CRPS sample")
        draws = daily[rng.integers(0, len(daily), size=(resamples, len(daily)))].mean(axis=1)
        delta = float(paired.mean())
        results[label] = {
            "v7_arm": comparison_specs[label][0],
            "cpcv2_arm": comparison_specs[label][1],
            "v7": v7_score,
            "cpcv2": cpc_score,
            "v7_minus_cpcv2_crps_mm": delta,
            "paired_daily_crps_delta_ci95_mm": [
                float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))
            ],
            "crps_winner": "v7" if delta < 0 else "cpcv2" if delta > 0 else "tie",
        }

    return {
        "scope": {
            "start": str(dates[0]), "end": str(dates[-1]),
            "days": int(len(dates)), "folds": 5,
            "station_pool": len(station_pool),
            "stations_scored_exactly_once": len(all_eval_ids),
            "bootstrap_unit": "day", "bootstrap_resamples": resamples,
            "audit": (
                "CPCv2 was subset to the V7 dates; each fold matched station "
                "coordinates, BMD values, withheld IDs and member count"
            ),
        },
        "comparisons": results,
    }


def markdown(report: dict) -> str:
    lines = [
        "# June 2023 V7 R81 versus CPCv2",
        "",
        report["scope"]["audit"] + ".",
        "",
        "| V7 arm | CPCv2 arm | V7 CRPS | CPCv2 CRPS | V7-CPCv2 | day-bootstrap 95% CI | winner |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for result in report["comparisons"].values():
        low, high = result["paired_daily_crps_delta_ci95_mm"]
        lines.append(
            f"| `{result['v7_arm']}` | `{result['cpcv2_arm']}` | "
            f"{result['v7']['crps_mm']:.3f} | {result['cpcv2']['crps_mm']:.3f} | "
            f"{result['v7_minus_cpcv2_crps_mm']:+.3f} | [{low:+.3f}, {high:+.3f}] | "
            f"{result['crps_winner']} |"
        )
    lines.extend([
        "",
        "Negative V7-CPCv2 CRPS favors V7. The interval resamples whole days, "
        "so stations observed on the same weather day remain paired.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.bootstrap_resamples < 100:
        raise SystemExit("--bootstrap-resamples must be at least 100")
    report = compare_folds(
        [Path(value) for value in args.v7_dumps],
        [Path(value) for value in args.cpcv2_dumps],
        args.bootstrap_resamples,
        args.seed,
    )
    json_path, markdown_path = Path(args.out_json), Path(args.out_markdown)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False))
    markdown_path.write_text(markdown(report))
    print(markdown(report))
    print(f"wrote {json_path}")
    print(f"wrote {markdown_path}")


if __name__ == "__main__":
    main()
