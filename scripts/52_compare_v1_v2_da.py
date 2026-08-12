#!/usr/bin/env python
"""Compare existing v1 DA with the corrected CPC-v2 DA on matched BMD folds.

The two experiments were written by different drivers and therefore use
different NPZ key names.  This adapter aligns them by withheld station ID and
date, verifies that the five-fold partitions and observations are identical,
and then applies the same fair CRPS and circular day-block bootstrap used by
the CPC-v2 tournament summaries.

Positive paired differences mean that the named candidate has lower CRPS than
the reference.  Stations from one weather day always remain in the same
bootstrap draw; they are not treated as independent replicates.
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

V2_GAUGES = "guided_s6_g010_t100"
V2_SIMULTANEOUS = "v2_simultaneous_s04_t100"

METHOD_ORDER = [
    "v1_background",
    "v1_gauges",
    "v1_simultaneous_s04",
    "v2_background",
    "v2_gauges_s6_g010",
    "v2_simultaneous_s04",
]
DISPLAY = {
    "v1_background": "v1 background",
    "v1_gauges": "v1 gauges",
    "v1_simultaneous_s04": "v1 simultaneous S04",
    "v2_background": "v2 background",
    "v2_gauges_s6_g010": "v2 gauges s6 g0.01",
    "v2_simultaneous_s04": "v2 simultaneous S04",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-raw-dumps", nargs="+", required=True)
    parser.add_argument("--v1-s04-dumps", nargs="+", required=True)
    parser.add_argument("--v2-dumps", nargs="+", required=True)
    parser.add_argument("--block-days", type=int, default=3)
    parser.add_argument("--n-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=202210)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-markdown", required=True)
    parser.add_argument("--out-plot", required=True)
    return parser.parse_args()


def station_ids(dump: np.lib.npyio.NpzFile) -> np.ndarray:
    key = "station_ids" if "station_ids" in dump else "station_id"
    values = np.asarray(dump[key]).astype(str)
    if len(values) != len(set(values.tolist())):
        raise ValueError("station IDs are not unique")
    return values


def dates(dump: np.lib.npyio.NpzFile) -> np.ndarray:
    if "times" in dump:
        return np.asarray(dump["times"]).astype("datetime64[D]")
    raw = np.asarray(dump["time"])
    if np.issubdtype(raw.dtype, np.integer):
        return raw.astype("datetime64[ns]").astype("datetime64[D]")
    return raw.astype("datetime64[D]")


def load_folds(paths: list[Path], schema: str) -> list[dict]:
    """Load a five-fold collection without assuming corresponding fold labels."""
    folds = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        dump = np.load(path, allow_pickle=False)
        ids = station_ids(dump)
        eval_idx = np.asarray(dump["eval_idx"], dtype=int)
        withheld = tuple(sorted(ids[eval_idx].tolist()))
        if schema == "v1":
            members = {
                "background": np.asarray(dump["background_at_stations"], float),
                "gauges": np.asarray(dump["gauge_analysis_at_stations"], float),
                "simultaneous": np.asarray(
                    dump["combined_analysis_at_stations"], float
                ),
            }
        elif schema == "v2":
            members = {
                "background": np.asarray(dump["station_background"], float),
                "gauges": np.asarray(dump[f"station_{V2_GAUGES}"], float),
                "simultaneous": np.asarray(
                    dump[f"station_{V2_SIMULTANEOUS}"], float
                ),
            }
        else:  # pragma: no cover - caller controls this
            raise ValueError(f"unknown schema {schema!r}")
        truth = np.asarray(dump["gauge_mm"], float)
        expected_shape = (len(dates(dump)), len(ids))
        if truth.shape != expected_shape:
            raise ValueError(
                f"{path}: gauge_mm shape {truth.shape} != {expected_shape}"
            )
        for name, values in members.items():
            if (
                values.ndim != 3
                or values.shape[0] != truth.shape[0]
                or values.shape[2] != len(ids)
            ):
                raise ValueError(f"{path}: invalid {name} station ensemble {values.shape}")
        folds.append(
            {
                "path": str(path),
                "ids": ids,
                "withheld": withheld,
                "dates": dates(dump),
                "truth": truth,
                "members": members,
            }
        )
    if len(folds) != 5:
        raise ValueError(f"expected five folds, got {len(folds)} for {schema}")
    return folds


def partition_map(folds: list[dict], label: str) -> dict[tuple[str, ...], dict]:
    mapping = {}
    all_ids = []
    for fold in folds:
        key = fold["withheld"]
        if key in mapping:
            raise ValueError(f"{label}: duplicate withheld fold {key}")
        mapping[key] = fold
        all_ids.extend(key)
    first_station_set = set(folds[0]["ids"].tolist())
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != first_station_set:
        raise ValueError(
            f"{label}: five folds must withhold every station exactly once"
        )
    for fold in folds[1:]:
        if set(fold["ids"].tolist()) != first_station_set:
            raise ValueError(f"{label}: folds do not use the same station pool")
    return mapping


def match_fold_partitions(collections: dict[str, list[dict]]) -> list[tuple[str, ...]]:
    """Require the same fold partition, while allowing fold-number permutation."""
    maps = {name: partition_map(folds, name) for name, folds in collections.items()}
    reference_name = next(iter(maps))
    reference = set(maps[reference_name])
    for name, mapping in maps.items():
        if set(mapping) != reference:
            raise ValueError(
                f"{name}: withheld station groups differ from {reference_name}; "
                "a five-fold mean would not be the same DA experiment"
            )
    return sorted(reference)


def pool_collection(
    folds: list[dict], partitions: list[tuple[str, ...]], canonical_ids: list[str]
) -> dict:
    """Pool exhaustive folds into one day-by-station verification array."""
    mapping = partition_map(folds, "pool")
    first_dates = folds[0]["dates"]
    member_counts = {
        values.shape[1]
        for fold in folds
        for values in fold["members"].values()
    }
    if len(member_counts) != 1:
        raise ValueError(f"ensemble sizes differ within a collection: {member_counts}")
    members_count = member_counts.pop()
    truth = np.full((len(first_dates), len(canonical_ids)), np.nan)
    members = {
        name: np.full((len(first_dates), members_count, len(canonical_ids)), np.nan)
        for name in folds[0]["members"]
    }
    target = {station: index for index, station in enumerate(canonical_ids)}

    for partition in partitions:
        fold = mapping[partition]
        if not np.array_equal(fold["dates"], first_dates):
            raise ValueError(f"{fold['path']}: dates differ across folds")
        source = {station: index for index, station in enumerate(fold["ids"])}
        for station in partition:
            source_index = source[station]
            target_index = target[station]
            truth[:, target_index] = fold["truth"][:, source_index]
            for name in members:
                members[name][:, :, target_index] = fold["members"][name][
                    :, :, source_index
                ]
    return {
        "dates": first_dates,
        "ids": np.asarray(canonical_ids),
        "truth": truth,
        "members": members,
        "n_members": members_count,
    }


def require_same_truth(reference: dict, candidate: dict, label: str) -> None:
    if not np.array_equal(reference["dates"], candidate["dates"]):
        raise ValueError(f"{label}: observation dates differ")
    if not np.array_equal(reference["ids"], candidate["ids"]):
        raise ValueError(f"{label}: station IDs differ after alignment")
    first, second = reference["truth"], candidate["truth"]
    if not np.array_equal(np.isfinite(first), np.isfinite(second)):
        raise ValueError(f"{label}: missing-observation masks differ")
    finite = np.isfinite(first)
    if not np.allclose(first[finite], second[finite], rtol=0.0, atol=1.0e-5):
        maximum = float(np.max(np.abs(first[finite] - second[finite])))
        raise ValueError(f"{label}: BMD observations differ (max {maximum:g} mm/day)")


def exact_control(first: np.ndarray, second: np.ndarray) -> dict:
    finite = np.isfinite(first) & np.isfinite(second)
    same_mask = np.array_equal(np.isfinite(first), np.isfinite(second))
    maximum = (
        float(np.max(np.abs(first[finite] - second[finite])))
        if finite.any()
        else None
    )
    return {
        "same_finite_mask": same_mask,
        "bit_identical": bool(same_mask and np.array_equal(first[finite], second[finite])),
        "max_abs_difference_mm": maximum,
    }


def method_metrics(members: np.ndarray, truth: np.ndarray) -> tuple[dict, np.ndarray]:
    crps = _summary.crps_per_sample(members, truth)
    mean = np.nanmean(members, axis=1)
    low, high = np.nanquantile(members, [0.05, 0.95], axis=1)
    keep = np.isfinite(truth) & np.isfinite(mean) & np.isfinite(crps)
    if not keep.any():
        raise ValueError("method has no finite withheld observations")
    wet = keep & (truth >= 1.0)
    dry = keep & (truth < 1.0)
    difference = mean - truth
    return {
        "n": int(keep.sum()),
        "n_wet": int(wet.sum()),
        "crps": float(np.nanmean(crps)),
        "mae": float(np.mean(np.abs(difference[keep]))),
        "dry_mae": float(np.mean(np.abs(difference[dry]))) if dry.any() else None,
        "wet_mae": float(np.mean(np.abs(difference[wet]))) if wet.any() else None,
        "bias": float(np.mean(difference[keep])),
        "correlation": (
            float(np.corrcoef(mean[keep], truth[keep])[0, 1])
            if mean[keep].std() > 0 and truth[keep].std() > 0 else None
        ),
        "coverage_90": float(
            np.mean((truth[keep] >= low[keep]) & (truth[keep] <= high[keep]))
        ),
    }, crps


def paired(
    crps: dict[str, np.ndarray], candidate: str, reference: str,
    block_days: int, n_resamples: int, seed: int,
) -> dict:
    result = _summary.circular_block_bootstrap(
        crps[reference] - crps[candidate], block_days, n_resamples, seed
    )
    result.update({"candidate": candidate, "reference": reference})
    return result


def verdict(result: dict) -> str:
    if result["ci_low"] > 0:
        return "candidate_beats_reference"
    if result["ci_high"] < 0:
        return "candidate_worse_than_reference"
    return "unresolved"


def fmt(result: dict) -> str:
    return (
        f"{result['difference']:+.3f} "
        f"[{result['ci_low']:+.3f}, {result['ci_high']:+.3f}]"
    )


def main() -> None:
    args = parse_args()
    collections = {
        "v1_raw": load_folds([Path(path) for path in args.v1_raw_dumps], "v1"),
        "v1_s04": load_folds([Path(path) for path in args.v1_s04_dumps], "v1"),
        "v2": load_folds([Path(path) for path in args.v2_dumps], "v2"),
    }
    partitions = match_fold_partitions(collections)
    canonical_ids = sorted(collections["v1_raw"][0]["ids"].tolist())
    pooled = {
        name: pool_collection(folds, partitions, canonical_ids)
        for name, folds in collections.items()
    }
    require_same_truth(pooled["v1_raw"], pooled["v1_s04"], "v1 RAW vs S04")
    require_same_truth(pooled["v1_raw"], pooled["v2"], "v1 vs v2")
    if len({value["n_members"] for value in pooled.values()}) != 1:
        raise ValueError("v1 and v2 ensemble sizes differ")

    raw = pooled["v1_raw"]["members"]
    s04 = pooled["v1_s04"]["members"]
    v2 = pooled["v2"]["members"]
    methods = {
        "v1_background": raw["background"],
        "v1_gauges": raw["gauges"],
        "v1_simultaneous_s04": s04["simultaneous"],
        "v2_background": v2["background"],
        "v2_gauges_s6_g010": v2["gauges"],
        "v2_simultaneous_s04": v2["simultaneous"],
    }
    truth = pooled["v1_raw"]["truth"]
    metrics, crps = {}, {}
    for name, members in methods.items():
        metrics[name], crps[name] = method_metrics(members, truth)

    pair_specs = [
        ("v2_background_vs_v1_background", "v2_background", "v1_background"),
        ("v2_gauges_vs_v1_gauges", "v2_gauges_s6_g010", "v1_gauges"),
        (
            "v2_simultaneous_vs_v1_simultaneous",
            "v2_simultaneous_s04", "v1_simultaneous_s04",
        ),
        ("v2_simultaneous_vs_v1_gauges", "v2_simultaneous_s04", "v1_gauges"),
        ("v2_gauges_vs_v2_background", "v2_gauges_s6_g010", "v2_background"),
        (
            "v2_simultaneous_vs_v2_gauges",
            "v2_simultaneous_s04", "v2_gauges_s6_g010",
        ),
    ]
    comparisons = {
        key: paired(
            crps, candidate, reference, args.block_days,
            args.n_resamples, args.seed + index * 10_000,
        )
        for index, (key, candidate, reference) in enumerate(pair_specs)
    }
    headline = comparisons["v2_simultaneous_vs_v1_gauges"]
    matched_method = comparisons["v2_simultaneous_vs_v1_simultaneous"]

    controls = {
        "v1_background_raw_vs_s04": exact_control(
            raw["background"], s04["background"]
        ),
        "v1_gauges_raw_vs_s04": exact_control(raw["gauges"], s04["gauges"]),
    }
    scope = {
        "start": str(pooled["v1_raw"]["dates"][0]),
        "end": str(pooled["v1_raw"]["dates"][-1]),
        "members": pooled["v1_raw"]["n_members"],
        "stations": len(canonical_ids),
        "folds": 5,
        "block_days": args.block_days,
        "n_resamples": args.n_resamples,
    }

    lines = [
        "# Matched v1 versus CPC-v2 real-BMD DA",
        "",
        f"- Period: **{scope['start']} to {scope['end']}**",
        f"- **{scope['stations']} stations**, each withheld exactly once in five matched folds",
        f"- **{scope['members']} members**; paired circular bootstrap in "
        f"{scope['block_days']}-day blocks ({scope['n_resamples']:,} resamples)",
        "- v1 and v2 were aligned by withheld station ID and date; "
        "fold-number permutation is allowed",
        "- This is a performance comparison: v2 uses its selected gauge "
        "spread/gamma, while v1 retains its previously reported DA settings",
        "",
        "| Method | CRPS | MAE dry/wet | Bias | Corr | Cov90 |",
        "|:--|--:|:--|--:|--:|--:|",
    ]
    for name in sorted(METHOD_ORDER, key=lambda value: metrics[value]["crps"]):
        value = metrics[name]
        lines.append(
            f"| `{name}` | {value['crps']:.3f} | "
            f"{value['dry_mae']:.2f}/{value['wet_mae']:.2f} | "
            f"{value['bias']:+.2f} | {value['correlation']:.3f} | "
            f"{value['coverage_90']:.2f} |"
        )
    lines += [
        "",
        "## Paired CRPS comparisons",
        "",
        *(
            f"- `{key}`: {fmt(value)} mm/day — {verdict(value).replace('_', ' ')}"
            for key, value in comparisons.items()
        ),
        "",
        "Positive values favour the candidate named before `_vs_`.",
        "",
        "## Answer",
        "",
        f"- Corrected v2 simultaneous versus the previously reported v1 gauge DA: "
        f"**{fmt(headline)} mm/day — {verdict(headline).replace('_', ' ')}**.",
        f"- S04 simultaneous versus the same S04 method under v1: "
        f"**{fmt(matched_method)} mm/day — {verdict(matched_method).replace('_', ' ')}**.",
    ]

    output = {
        "scope": scope,
        "sources": {
            name: [fold["path"] for fold in folds]
            for name, folds in collections.items()
        },
        "matched_withheld_partitions": [list(value) for value in partitions],
        "metrics": metrics,
        "comparisons": comparisons,
        "headline": {
            "question": "Does corrected v2 simultaneous DA beat existing v1 gauge DA?",
            "comparison": "v2_simultaneous_vs_v1_gauges",
            "verdict": verdict(headline),
        },
        "same_method": {
            "question": "Does v2 S04 simultaneous DA beat v1 S04 simultaneous DA?",
            "comparison": "v2_simultaneous_vs_v1_simultaneous",
            "verdict": verdict(matched_method),
        },
        "v1_shared_run_controls": controls,
    }

    out_json = Path(args.out_json)
    out_markdown = Path(args.out_markdown)
    out_plot = Path(args.out_plot)
    for path in (out_json, out_markdown, out_plot):
        path.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    out_markdown.write_text("\n".join(lines) + "\n")

    positions = np.arange(len(METHOD_ORDER))
    labels = [DISPLAY[name] for name in METHOD_ORDER]
    colors = ["#8C564B"] * 3 + ["#1F77B4"] * 3
    figure, axes = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True)
    axes[0, 0].barh(
        positions, [metrics[name]["crps"] for name in METHOD_ORDER], color=colors
    )
    axes[0, 0].set_yticks(positions, labels, fontsize=8)
    axes[0, 0].invert_yaxis(); axes[0, 0].set_xlabel("CRPS (mm/day)")
    axes[0, 0].set_title("A. Withheld-gauge probabilistic skill")

    pair_names = [key for key, _, _ in pair_specs]
    centres = np.asarray([comparisons[key]["difference"] for key in pair_names])
    lows = np.asarray([comparisons[key]["ci_low"] for key in pair_names])
    highs = np.asarray([comparisons[key]["ci_high"] for key in pair_names])
    pair_positions = np.arange(len(pair_names))
    axes[0, 1].errorbar(
        centres, pair_positions,
        xerr=np.vstack([centres - lows, highs - centres]),
        fmt="o", color="#1B4965", capsize=3,
    )
    axes[0, 1].axvline(0, color="black", ls="--")
    axes[0, 1].set_yticks(
        pair_positions, [name.replace("_", " ") for name in pair_names], fontsize=7
    )
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_xlabel("CRPS(reference) − CRPS(candidate)")
    axes[0, 1].set_title("B. Paired day-block uncertainty")

    width = 0.38
    axes[0, 2].barh(
        positions - width / 2, [metrics[name]["dry_mae"] for name in METHOD_ORDER],
        height=width, label="dry", color="#E9C46A",
    )
    axes[0, 2].barh(
        positions + width / 2, [metrics[name]["wet_mae"] for name in METHOD_ORDER],
        height=width, label="wet", color="#2A9D8F",
    )
    axes[0, 2].set_yticks(positions, labels, fontsize=8); axes[0, 2].invert_yaxis()
    axes[0, 2].set_xlabel("MAE (mm/day)"); axes[0, 2].legend()
    axes[0, 2].set_title("C. Dry/wet trade-off")

    axes[1, 0].barh(
        positions, [metrics[name]["bias"] for name in METHOD_ORDER], color=colors
    )
    axes[1, 0].axvline(0, color="black")
    axes[1, 0].set_yticks(positions, labels, fontsize=8); axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlabel("Bias (mm/day)"); axes[1, 0].set_title("D. Mean bias")

    axes[1, 1].barh(
        positions, [metrics[name]["correlation"] for name in METHOD_ORDER], color=colors
    )
    axes[1, 1].set_yticks(positions, labels, fontsize=8); axes[1, 1].invert_yaxis()
    axes[1, 1].set_xlabel("Correlation"); axes[1, 1].set_title("E. Gauge correlation")

    axes[1, 2].barh(
        positions, [metrics[name]["coverage_90"] for name in METHOD_ORDER], color=colors
    )
    axes[1, 2].axvline(0.90, color="black", ls="--", label="nominal 0.90")
    axes[1, 2].set_yticks(positions, labels, fontsize=8); axes[1, 2].invert_yaxis()
    axes[1, 2].set_xlabel("Empirical 90% coverage"); axes[1, 2].legend()
    axes[1, 2].set_title("F. Ensemble reliability")

    figure.suptitle("Matched v1 versus CPC-v2 real-BMD data assimilation")
    figure.savefig(out_plot, dpi=160)
    plt.close(figure)

    print("\n".join(lines))
    print(f"\n[done] wrote {out_json}")
    print(f"[done] wrote {out_markdown}")
    print(f"[done] wrote {out_plot}")


if __name__ == "__main__":
    main()
