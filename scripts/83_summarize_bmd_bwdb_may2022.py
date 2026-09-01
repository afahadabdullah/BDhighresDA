#!/usr/bin/env python
"""Summarize the three constrained BMD+BWDB May 2022 CPC-v2 folds.

This deliberately does not require exhaustive cross-validation: the experiment
uses three disjoint 20% holdouts and leaves isolated gauges assimilated but
unscored.  It validates that limited scope against the preparation manifest
before calculating pooled, BMD-only, and BWDB-only deterministic metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dumps", nargs="+", required=True)
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--station-summary", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def fair_crps(members: np.ndarray, truth: np.ndarray) -> np.ndarray:
    members, truth = np.asarray(members, float), np.asarray(truth, float)
    result = np.full(len(truth), np.nan)
    valid = np.isfinite(truth) & np.all(np.isfinite(members), axis=1)
    if not valid.any():
        return result
    selected, observed = members[valid], truth[valid]
    count = selected.shape[1]
    first = np.mean(np.abs(selected - observed[:, None]), axis=1)
    ordered = np.sort(selected, axis=1)
    pair = np.sum(ordered * (2 * np.arange(1, count + 1) - count - 1)[None, :], axis=1)
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
        "rmse_mm": float(np.sqrt(np.mean(error ** 2))),
        "bias_mm": float(np.mean(error)),
        "crps_mm": float(np.nanmean(fair_crps(members, truth))),
        "correlation": float(np.corrcoef(mean, truth)[0, 1]) if mean.std() and truth.std() else None,
    }


def main() -> None:
    args = parse_args()
    if len(args.dumps) != len(args.reports):
        raise ValueError("--dumps and --reports must have equal length")
    manifest = json.loads(Path(args.manifest).read_text())
    selection = manifest["analysis_selection"]
    if len(args.dumps) != int(selection["holdout_folds"]):
        raise ValueError("number of fold dumps disagrees with manifest")
    if selection["maximum_retained_neighbour_km"] > selection["support_radius_km"] + 1e-9:
        raise ValueError("manifest violates its retained-neighbour radius")
    source_map = pd.read_csv(args.station_summary).set_index("station_id")["source"].astype(str).to_dict()

    seen, observations = set(), []
    members_by_method: dict[str, list[np.ndarray]] = {}
    names: list[str] | None = None
    times_ref: np.ndarray | None = None
    for dump_name, report_name in zip(args.dumps, args.reports):
        report = json.loads(Path(report_name).read_text())
        scope = report["scope"]
        if int(scope["holdout_folds"]) != int(selection["holdout_folds"]):
            raise ValueError(f"{report_name}: wrong holdout fold count")
        dump = np.load(dump_name, allow_pickle=False)
        ids = dump["station_ids"].astype(str)
        eval_ids = ids[np.asarray(dump["eval_idx"], int)]
        if len(eval_ids) != int(round(selection["analysis_stations"] * selection["holdout_fraction_each_fold"])):
            raise ValueError(f"{dump_name}: fold is not the requested 20% size")
        if seen.intersection(eval_ids):
            raise ValueError("held-out station appears in more than one fold")
        seen.update(eval_ids)
        times = dump["times"].astype("datetime64[D]")
        if times_ref is None:
            times_ref = times
            names = dump["variant_names"].astype(str).tolist()
            for name in names:
                members_by_method[name] = []
        elif not np.array_equal(times, times_ref) or dump["variant_names"].astype(str).tolist() != names:
            raise ValueError("fold dates or method names differ")
        truth = np.asarray(dump["gauge_mm"][:, dump["eval_idx"]], float)
        station = np.tile(eval_ids, len(times))
        date = np.repeat(times.astype(str), len(eval_ids))
        observations.append(pd.DataFrame({"date": date, "station_id": station, "truth_mm": truth.reshape(-1)}))
        for name in names or []:
            ensemble = np.asarray(dump[f"station_{name}"][:, :, dump["eval_idx"]], float)
            members_by_method[name].append(np.moveaxis(ensemble, 1, 2).reshape(-1, ensemble.shape[1]))
        dump.close()
    table = pd.concat(observations, ignore_index=True)
    table["source"] = table["station_id"].map(source_map)
    if table["source"].isna().any():
        raise ValueError("a held-out station has no source classification")
    truth = table["truth_mm"].to_numpy(float)
    result = {
        "scope": selection,
        "temporal_support_note": manifest["temporal_support_note"],
        "period": manifest["period"],
        "stations_scored": int(len(seen)),
        "station_days_scored": int(table["truth_mm"].notna().sum()),
        "methods": {},
    }
    for name, pieces in members_by_method.items():
        ensemble = np.concatenate(pieces)
        result["methods"][name] = {
            "pooled": score(ensemble, truth),
            "BMD": score(ensemble[table["source"].to_numpy() == "BMD"], truth[table["source"].to_numpy() == "BMD"]),
            "BWDB": score(ensemble[table["source"].to_numpy() == "BWDB"], truth[table["source"].to_numpy() == "BWDB"]),
        }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "may2022_bmd_bwdb_scores.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# May 2022 combined BMD/BWDB CPC-v2 test", "",
        f"Three disjoint 20% folds scored {result['stations_scored']} stations; every held-out station retained an assimilated neighbour within {selection['support_radius_km']:.0f} km.",
        "", "| Method | Pool RMSE | BMD RMSE | BWDB RMSE | Pool CRPS | BMD CRPS | BWDB CRPS |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in result["methods"].items():
        pooled, bmd, bwdb = metrics["pooled"], metrics["BMD"], metrics["BWDB"]
        lines.append(
            f"| {name} | {pooled.get('rmse_mm', float('nan')):.3f} | {bmd.get('rmse_mm', float('nan')):.3f} | {bwdb.get('rmse_mm', float('nan')):.3f} | {pooled.get('crps_mm', float('nan')):.3f} | {bmd.get('crps_mm', float('nan')):.3f} | {bwdb.get('crps_mm', float('nan')):.3f} |"
        )
    lines.extend(["", "## Temporal scope", "", manifest["temporal_support_note"], ""])
    (out_dir / "may2022_bmd_bwdb_scores.md").write_text("\n".join(lines))
    print(json.dumps({"out_dir": str(out_dir), "stations_scored": result["stations_scored"]}, indent=2))


if __name__ == "__main__":
    main()
