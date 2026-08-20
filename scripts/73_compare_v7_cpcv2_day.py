#!/usr/bin/env python3
"""Compare matched-window V7 and CPCv2 DA analyses at BMD gauges.

The two model families have different checkpoints and different satellite
footprint resolutions, but this comparison locks everything that can otherwise
make a one-day result accidental: BMD observation date, station set, withheld
IDs, ensemble size, and sampling horizon.  It compares the frozen winners in
two like-for-like rows:

* gauges only: V7 ``da_meso`` vs CPCv2 ``guided_s6_g010_t100``;
* simultaneous: V7 ``da_sim`` at the selected IMERG R multiplier 9 vs CPCv2
  ``v2_simul_s04_ig010``.

When the V7 dump also contains ``da_sim_r*`` calibration arms, each is scored
against that same CPCv2 simultaneous ensemble.  This keeps dates, observations,
members, and random draw fixed while testing only V7's satellite likelihood
weight.

The CPCv2 runner writes ``gauge_mm``; the V7 diagnostic writes ``observed_mm``
when called with ``--station-dump``.  This program refuses to compare summary
numbers alone: it verifies the raw observations and selected holdout IDs first.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


COMPARISONS = {
    "gauges_only": ("da_meso", "guided_s6_g010_t100"),
    "simultaneous": ("da_sim", "v2_simul_s04_ig010"),
}


def _available_comparisons(v7: np.lib.npyio.NpzFile) -> dict[str, tuple[str, str]]:
    """Base winners plus any optional V7 simultaneous R-calibration arms."""
    comparisons = dict(COMPARISONS)
    if "arm_names" in v7:
        ordered = np.asarray(v7["arm_names"], dtype=str).tolist()
        arms = [
            arm for arm in ordered
            if arm.startswith("da_sim_") and f"station_{arm}" in v7
        ]
    else:
        arms = sorted(
            key[len("station_"):] for key in v7.files
            if key.startswith("station_da_sim_")
        )
    for arm in arms:
        suffix = arm.removeprefix("da_sim_")
        comparisons[f"simultaneous_{suffix}"] = (arm, "v2_simul_s04_ig010")
    return comparisons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v7-dump", required=True)
    parser.add_argument("--cpcv2-dump", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-markdown", required=True)
    return parser.parse_args()


def _required(dump: np.lib.npyio.NpzFile, path: Path, *keys: str) -> None:
    missing = [key for key in keys if key not in dump]
    if missing:
        raise ValueError(f"{path}: missing required arrays {missing}")


def _fair_crps_per_sample(members: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Fair CRPS for each finite station-day, matching ``crps_ensemble``."""
    if members.ndim != 3 or observed.ndim != 2:
        raise ValueError("members must be T,M,S and observations T,S")
    ensemble = np.moveaxis(members, 1, 0)  # M,T,S
    keep = np.isfinite(observed) & np.all(np.isfinite(ensemble), axis=0)
    if not keep.any():
        return np.empty(0, dtype=np.float64)
    selected = ensemble[:, keep]
    truth = observed[keep]
    n_members = selected.shape[0]
    if n_members < 2:
        raise ValueError("fair CRPS needs at least two ensemble members")
    term1 = np.abs(selected - truth[None]).mean(axis=0)
    term2 = np.abs(selected[:, None] - selected[None, :]).sum(axis=(0, 1))
    term2 /= 2.0 * n_members * (n_members - 1)
    return term1 - term2


def score(members: np.ndarray, observed: np.ndarray) -> dict[str, float | int]:
    """Station-day metrics with the exact fair-CRPS convention used by V7."""
    ensemble = np.moveaxis(members, 1, 0)  # M,T,S
    keep = np.isfinite(observed) & np.all(np.isfinite(ensemble), axis=0)
    if not keep.any():
        raise ValueError("no finite withheld station-days to compare")
    selected = ensemble[:, keep]
    truth = observed[keep]
    mean = selected.mean(axis=0)
    spread = float(np.sqrt(np.mean(selected.var(axis=0, ddof=1))))
    rmse = float(np.sqrt(np.mean((mean - truth) ** 2)))
    n_members = selected.shape[0]
    fair_crps = np.abs(selected - truth[None]).mean(axis=0)
    fair_crps -= (
        np.abs(selected[:, None] - selected[None, :]).sum(axis=(0, 1))
        / (2.0 * n_members * (n_members - 1))
    )
    low, high = np.quantile(selected, [0.05, 0.95], axis=0)
    return {
        "station_days": int(keep.sum()),
        "crps_mm": float(np.mean(fair_crps)),
        "mae_mm": float(np.mean(np.abs(mean - truth))),
        "bias_mm": float(np.mean(mean - truth)),
        "rmse_mm": rmse,
        "spread_mm": spread,
        "spread_skill": float(spread / rmse) if rmse else float("nan"),
        "coverage_90": float(np.mean((truth >= low) & (truth <= high))),
    }


def _align_cpc_to_v7(
    v7: np.lib.npyio.NpzFile,
    cpc: np.lib.npyio.NpzFile,
    v7_path: Path,
    cpc_path: Path,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    dict[str, np.ndarray], dict[str, tuple[str, str]],
]:
    _required(
        v7, v7_path, "times", "model_times", "station_ids", "station_lat",
        "station_lon", "eval_idx", "observed_mm",
    )
    _required(
        cpc, cpc_path, "times", "model_times", "station_ids", "station_lat",
        "station_lon", "eval_idx", "gauge_mm",
    )

    v7_times = np.asarray(v7["times"]).astype("datetime64[D]")
    cpc_times = np.asarray(cpc["times"]).astype("datetime64[D]")
    if not np.array_equal(v7_times, cpc_times):
        raise ValueError(
            "observation dates differ: V7 "
            f"{v7_times.astype(str).tolist()} vs CPCv2 {cpc_times.astype(str).tolist()}"
        )
    v7_model_times = np.asarray(v7["model_times"]).astype("datetime64[D]")
    cpc_model_times = np.asarray(cpc["model_times"]).astype("datetime64[D]")
    if not np.array_equal(v7_model_times, cpc_model_times):
        raise ValueError(
            "model dates differ: V7 "
            f"{v7_model_times.astype(str).tolist()} vs CPCv2 "
            f"{cpc_model_times.astype(str).tolist()}"
        )

    v7_ids = np.asarray(v7["station_ids"]).astype(str)
    cpc_ids = np.asarray(cpc["station_ids"]).astype(str)
    if len(v7_ids) != len(set(v7_ids.tolist())) or len(cpc_ids) != len(set(cpc_ids.tolist())):
        raise ValueError("station IDs must be unique in both dumps")
    if set(v7_ids.tolist()) != set(cpc_ids.tolist()):
        raise ValueError("station pools differ after coverage filtering")
    cpc_order = np.asarray([np.where(cpc_ids == station)[0][0] for station in v7_ids])
    for coordinate in ("station_lat", "station_lon"):
        v7_coord = np.asarray(v7[coordinate], dtype=float)
        cpc_coord = np.asarray(cpc[coordinate], dtype=float)[cpc_order]
        if not np.allclose(v7_coord, cpc_coord, rtol=0.0, atol=1.0e-6):
            raise ValueError(f"station {coordinate[8:]} values differ")

    v7_eval_ids = set(v7_ids[np.asarray(v7["eval_idx"], dtype=int)].tolist())
    cpc_eval_ids = set(cpc_ids[np.asarray(cpc["eval_idx"], dtype=int)].tolist())
    if v7_eval_ids != cpc_eval_ids:
        raise ValueError(
            "withheld station IDs differ: "
            f"V7={sorted(v7_eval_ids)}, CPCv2={sorted(cpc_eval_ids)}"
        )
    eval_idx = np.flatnonzero(np.isin(v7_ids, sorted(v7_eval_ids)))

    observed = np.asarray(v7["observed_mm"], dtype=np.float64)
    cpc_observed = np.asarray(cpc["gauge_mm"], dtype=np.float64)[:, cpc_order]
    if observed.shape != cpc_observed.shape:
        raise ValueError(
            f"observation shape differs: V7 {observed.shape}, CPCv2 {cpc_observed.shape}"
        )
    same_finite = np.array_equal(np.isfinite(observed), np.isfinite(cpc_observed))
    finite = np.isfinite(observed)
    if not same_finite or not np.allclose(observed[finite], cpc_observed[finite], atol=1e-5, rtol=0.0):
        maximum = float(np.max(np.abs(observed[finite] - cpc_observed[finite]))) if finite.any() else float("nan")
        raise ValueError(
            "BMD values differ between V7 and CPCv2 "
            f"(same finite mask={same_finite}, max |difference|={maximum:g} mm)"
        )

    comparisons = _available_comparisons(v7)
    members: dict[str, np.ndarray] = {}
    for label, (v7_arm, cpc_arm) in comparisons.items():
        v7_key, cpc_key = f"station_{v7_arm}", f"station_{cpc_arm}"
        _required(v7, v7_path, v7_key)
        _required(cpc, cpc_path, cpc_key)
        v7_members = np.asarray(v7[v7_key], dtype=np.float64)
        cpc_members = np.asarray(cpc[cpc_key], dtype=np.float64)[:, :, cpc_order]
        if v7_members.shape != cpc_members.shape:
            raise ValueError(
                f"{label}: ensemble shape differs: V7 {v7_members.shape}, "
                f"CPCv2 {cpc_members.shape}"
            )
        if v7_members.shape[:1] != observed.shape[:1] or v7_members.shape[2:] != observed.shape[1:]:
            raise ValueError(f"{label}: ensemble dimensions do not match observations")
        if v7_members.shape[1] < 2:
            raise ValueError(f"{label}: comparison needs at least two members")
        members[f"v7_{label}"] = v7_members
        members[f"cpcv2_{label}"] = cpc_members

    return v7_times, v7_ids, eval_idx, observed, members, comparisons


def compare_dumps(v7_path: Path, cpc_path: Path) -> dict:
    """Load, audit, and score the two raw outputs on identical station-days."""
    with np.load(v7_path, allow_pickle=False) as v7, np.load(cpc_path, allow_pickle=False) as cpc:
        times, ids, eval_idx, observed, members, comparisons = _align_cpc_to_v7(
            v7, cpc, v7_path, cpc_path
        )
        results: dict[str, dict] = {}
        for label in comparisons:
            v7_score = score(members[f"v7_{label}"][:, :, eval_idx], observed[:, eval_idx])
            cpc_score = score(
                members[f"cpcv2_{label}"][:, :, eval_idx], observed[:, eval_idx]
            )
            v7_crps = _fair_crps_per_sample(
                members[f"v7_{label}"][:, :, eval_idx], observed[:, eval_idx]
            )
            cpc_crps = _fair_crps_per_sample(
                members[f"cpcv2_{label}"][:, :, eval_idx], observed[:, eval_idx]
            )
            if v7_crps.shape != cpc_crps.shape:
                raise RuntimeError(f"{label}: paired CRPS shapes disagree")
            delta = {
                key: float(v7_score[key] - cpc_score[key])
                for key in ("crps_mm", "mae_mm", "bias_mm", "rmse_mm", "spread_mm", "spread_skill")
            }
            delta["mean_paired_crps_mm"] = float(np.mean(v7_crps - cpc_crps))
            results[label] = {
                "v7_arm": comparisons[label][0],
                "cpcv2_arm": comparisons[label][1],
                "v7": v7_score,
                "cpcv2": cpc_score,
                "v7_minus_cpcv2": delta,
                "crps_winner": (
                    "v7" if delta["crps_mm"] < 0.0
                    else "cpcv2" if delta["crps_mm"] > 0.0 else "tie"
                ),
            }

    return {
        "scope": {
            "observation_dates": times.astype(str).tolist(),
            "members": int(next(iter(members.values())).shape[1]),
            "station_pool": int(len(ids)),
            "withheld_station_ids": ids[eval_idx].astype(str).tolist(),
            "withheld_station_days": int(np.isfinite(observed[:, eval_idx]).sum()),
            "v7_dump": str(v7_path),
            "cpcv2_dump": str(cpc_path),
            "audit": (
                "matching model/BMD dates, station coordinates, BMD values, "
                "station pool, and withheld IDs verified"
            ),
            "caveat": (
                "This is a matched evaluation window, not a significance test. V7 uses its "
                "native 0.1-degree IMERG stage-A stream, while CPCv2 retains its "
                "frozen winning S04 (0.4-degree) stream."
            ),
        },
        "comparisons": results,
    }


def markdown(report: dict) -> str:
    lines = [
        "# Matched-window V7 vs CPCv2 comparison",
        "",
        report["scope"]["audit"] + ".",
        "",
        "| DA mode | V7 arm | CPCv2 arm | V7 CRPS | CPCv2 CRPS | V7 − CPCv2 | CRPS winner |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for label, values in report["comparisons"].items():
        delta = values["v7_minus_cpcv2"]["crps_mm"]
        lines.append(
            f"| {label.replace('_', ' ')} | `{values['v7_arm']}` | "
            f"`{values['cpcv2_arm']}` | {values['v7']['crps_mm']:.3f} | "
            f"{values['cpcv2']['crps_mm']:.3f} | {delta:+.3f} | "
            f"{values['crps_winner']} |"
        )
    lines.extend(["", f"Caveat: {report['scope']['caveat']}", ""])
    return "\n".join(lines)


def json_ready(value):
    """Convert diagnostic NaNs to JSON ``null`` without hiding valid scores."""
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def main() -> None:
    args = parse_args()
    report = compare_dumps(Path(args.v7_dump), Path(args.cpcv2_dump))
    json_path = Path(args.out_json)
    markdown_path = Path(args.out_markdown)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(json_ready(report), indent=2, allow_nan=False))
    markdown_path.write_text(markdown(report))
    for label, values in report["comparisons"].items():
        delta = values["v7_minus_cpcv2"]["crps_mm"]
        print(
            f"{label:13s} V7 {values['v7']['crps_mm']:.3f}  "
            f"CPCv2 {values['cpcv2']['crps_mm']:.3f}  "
            f"V7-CPCv2 {delta:+.3f} mm  -> {values['crps_winner']}",
            flush=True,
        )
    print(f"[done] {json_path}", flush=True)
    print(f"[done] {markdown_path}", flush=True)


if __name__ == "__main__":
    main()
