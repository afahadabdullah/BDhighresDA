#!/usr/bin/env python3
"""Audit a V7 replay against its reference JSON and frozen checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


METRICS = ("crps_mm", "mae_mm", "bias_mm", "rmse_mm", "spread_mm", "spread_skill")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-markdown", required=True)
    return parser.parse_args()


def digest(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def number(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def checkpoint_audit(reference: dict, candidate: dict) -> dict:
    audit: dict[str, dict] = {}
    for label in ("meso", "allocation"):
        first = reference["checkpoints"][label]["frozen"]
        second = candidate["checkpoints"][label]["frozen"]
        first_hash, second_hash = digest(first), digest(second)
        audit[label] = {
            "reference": first,
            "candidate": second,
            "reference_sha256": first_hash,
            "candidate_sha256": second_hash,
            "identical": first_hash == second_hash,
        }
    return audit


def compare(reference: dict, candidate: dict) -> dict:
    settings = {}
    for key in ("members", "n_steps", "observations", "gauge_day_offset", "imerg_day_offset"):
        settings[key] = {
            "reference": reference.get(key),
            "candidate": candidate.get(key),
            "identical": reference.get(key) == candidate.get(key),
        }
    arms = {}
    for arm in sorted(set(reference["arms"]) & set(candidate["arms"])):
        expected = reference["arms"][arm].get("mean", {})
        actual = candidate["arms"][arm].get("mean", {})
        metrics = {}
        for metric in METRICS:
            before, after = number(expected.get(metric)), number(actual.get(metric))
            metrics[metric] = {
                "reference": before,
                "candidate": after,
                "candidate_minus_reference": (
                    after - before if before is not None and after is not None else None
                ),
            }
        arms[arm] = metrics
    return {"settings": settings, "arms": arms}


def markdown(report: dict) -> str:
    lines = ["# V7 May 3 replay audit", ""]
    lines.append("## Frozen checkpoints")
    lines.append("")
    lines.extend([
        "| Stage | Identical SHA-256 |",
        "|---|---|",
    ])
    for label, value in report["checkpoints"].items():
        lines.append(f"| {label} | {'yes' if value['identical'] else 'NO'} |")
    lines.extend(["", "## Withheld-gauge metrics", ""])
    lines.extend([
        "| Arm | Reference CRPS | Replay CRPS | Δ CRPS | Reference MAE | Replay MAE |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for arm, metrics in report["comparison"]["arms"].items():
        crps, mae = metrics["crps_mm"], metrics["mae_mm"]
        lines.append(
            f"| {arm} | {crps['reference']:.6f} | {crps['candidate']:.6f} | "
            f"{crps['candidate_minus_reference']:+.6f} | {mae['reference']:.6f} | "
            f"{mae['candidate']:.6f} |"
        )
    lines.extend([
        "",
        "A non-zero difference with identical checkpoints/settings is residual GPU numerical nondeterminism; "
        "a checkpoint mismatch is not a reproduction test.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    reference_path, candidate_path = Path(args.reference), Path(args.candidate)
    reference = json.loads(reference_path.read_text())
    candidate = json.loads(candidate_path.read_text())
    report = {
        "reference": str(reference_path),
        "candidate": str(candidate_path),
        "checkpoints": checkpoint_audit(reference, candidate),
        "comparison": compare(reference, candidate),
    }
    out_json, out_markdown = Path(args.out_json), Path(args.out_markdown)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_markdown.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2))
    out_markdown.write_text(markdown(report))
    print(f"[reproduction] wrote {out_json}")
    print(f"[reproduction] wrote {out_markdown}")


if __name__ == "__main__":
    main()
