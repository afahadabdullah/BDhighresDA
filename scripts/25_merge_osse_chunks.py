#!/usr/bin/env python
"""Merge month-sized OSSE dumps and reports into one auditable arm.

The full paper OSSE is split by observation arm, year and JJA month so GPU
sampling can run in parallel. This script verifies that all static metadata are
identical, rejects duplicate dates, concatenates only known day-axis arrays,
and combines the per-day-averaged metrics with exact day-count weights.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DAY_ARRAYS = {
    "background",
    "analysis",
    "truth",
    "coarse_base_mm",
    "obs_transformed",
    "truth_at_stations",
    "pseudo_satellite_mm",
    "pseudo_satellite_truth_mm",
    "days",
}

# These scalars are QC maxima accumulated within each month, not invariant run
# metadata.  The merged value must be the worst (largest) discrepancy over all
# chunks.  Averaging would weaken the exact-observation audit; requiring equality
# incorrectly rejects sound chunks whose round-off maxima differ.
MAX_DIAGNOSTICS = {
    "exact_gauge_max_abs_error_transformed",
    "exact_satellite_max_abs_error_mm",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chunks-root", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def same_value(left: np.ndarray, right: np.ndarray) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if np.issubdtype(left.dtype, np.number):
        return bool(np.allclose(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def finite_max(values) -> float:
    finite = []
    for value in values:
        if value is None:
            continue
        scalar = float(np.asarray(value).item())
        if np.isfinite(scalar):
            finite.append(scalar)
    return max(finite) if finite else float("nan")


def weighted_merge(values: list, weights: np.ndarray):
    """Recursively merge report values that represent per-day means."""
    present = [(value, weight) for value, weight in zip(values, weights) if value is not None]
    if not present:
        return None
    first = present[0][0]
    if isinstance(first, dict):
        keys = {key for value, _ in present if isinstance(value, dict) for key in value}
        return {
            key: weighted_merge(
                [value.get(key) if isinstance(value, dict) else None for value in values],
                weights,
            )
            for key in sorted(keys)
        }
    if isinstance(first, bool):
        if any(value != first for value, _ in present):
            raise ValueError(f"inconsistent boolean report values: {present}")
        return first
    if isinstance(first, (int, float)) and not isinstance(first, bool):
        finite = [
            (float(value), float(weight))
            for value, weight in present
            if np.isfinite(float(value))
        ]
        if not finite:
            return float("nan")
        numerator = sum(value * weight for value, weight in finite)
        denominator = sum(weight for _, weight in finite)
        return numerator / denominator
    if any(value != first for value, _ in present):
        raise ValueError(f"inconsistent report values: {[value for value, _ in present]}")
    return first


def main() -> None:
    args = parse_args()
    chunks_root = Path(args.chunks_root) / args.arm
    chunk_dirs = sorted(path.parent for path in chunks_root.glob("*/ensemble.npz"))
    if not chunk_dirs:
        raise SystemExit(f"no ensemble chunks found below {chunks_root}")

    stores = [np.load(directory / "ensemble.npz", allow_pickle=False) for directory in chunk_dirs]
    try:
        expected_keys = set(stores[0].files)
        for directory, store in zip(chunk_dirs[1:], stores[1:]):
            if set(store.files) != expected_keys:
                raise ValueError(
                    f"{directory}: dump keys differ from first chunk; "
                    f"missing={sorted(expected_keys - set(store.files))}, "
                    f"extra={sorted(set(store.files) - expected_keys)}"
                )

        for key in sorted(expected_keys - DAY_ARRAYS - MAX_DIAGNOSTICS):
            reference = stores[0][key]
            for directory, store in zip(chunk_dirs[1:], stores[1:]):
                if not same_value(reference, store[key]):
                    raise ValueError(f"{directory}: static metadata {key!r} differs")

        merged: dict[str, np.ndarray] = {}
        for key in sorted(expected_keys):
            if key in DAY_ARRAYS:
                merged[key] = np.concatenate([store[key] for store in stores], axis=0)
            elif key in MAX_DIAGNOSTICS:
                merged[key] = np.asarray(
                    finite_max([store[key] for store in stores]),
                    dtype=stores[0][key].dtype,
                )
            else:
                merged[key] = stores[0][key]

        dates = merged["days"].astype(str)
        if len(np.unique(dates)) != len(dates):
            duplicate = sorted({date for date in dates if np.count_nonzero(dates == date) > 1})
            raise ValueError(f"duplicate OSSE dates across chunks: {duplicate}")
        order = np.argsort(dates)
        for key in DAY_ARRAYS & set(merged):
            merged[key] = merged[key][order]

        expected_days = 4 * (30 + 31 + 31)
        if len(dates) != expected_days:
            raise ValueError(
                f"expected every JJA day in 2021-2024 ({expected_days}), "
                f"found {len(dates)}"
            )

        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / "ensemble.npz", **merged)
    finally:
        for store in stores:
            store.close()

    reports = [json.loads((directory / "osse_report.json").read_text()) for directory in chunk_dirs]
    weights = np.asarray([len(report["days"]) for report in reports], dtype=float)
    rows = []
    for report in reports:
        if len(report.get("results", [])) != 1:
            raise ValueError("each chunk must contain exactly one OSSE result row")
        rows.append(report["results"][0])
    result = weighted_merge(rows, weights)
    for key in MAX_DIAGNOSTICS:
        values = [row.get(key) for row in rows]
        if any(value is not None for value in values):
            result[key] = finite_max(values)

    for scope in ("withheld", "assimilated", "field"):
        for metric in ("rmse_mm", "crps_mm", "mae_mm"):
            background = result.get(f"{scope}_background", {}).get(metric)
            analysis = result.get(f"{scope}_analysis", {}).get(metric)
            key = f"{scope}_improvement_{metric}"
            result[key] = (
                100.0 * (background - analysis) / background
                if background is not None
                and analysis is not None
                and np.isfinite(background)
                and background != 0
                else float("nan")
            )

    report = dict(reports[0])
    report["days"] = sorted(date for item in reports for date in item["days"])
    report["requested_months"] = [6, 7, 8]
    report["day_selection"] = "all JJA days; merged from year-month chunks"
    report["chunks"] = [str(directory) for directory in chunk_dirs]
    report["results"] = [result]
    (Path(args.out_dir) / "osse_report.json").write_text(
        json.dumps(report, indent=2, default=float) + "\n"
    )

    print(
        f"merged {len(chunk_dirs)} chunks / {len(report['days'])} days for "
        f"{args.arm} -> {args.out_dir}"
    )


if __name__ == "__main__":
    main()
