#!/usr/bin/env python
"""Summarize the frozen Huber3 BMD+BWDB 2021-2024 evaluation/production run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


BACKGROUND = "background"
WINNER = "v2_simul_s04_huber3"
METHODS = [BACKGROUND, WINNER]
PERIODS = {
    ("2021-05-01", "2021-09-30"): "2021_may_sep",
    ("2022-05-01", "2022-09-30"): "2022_may_sep",
    ("2023-05-01", "2023-09-30"): "2023_may_sep",
    ("2024-05-01", "2024-06-30"): "2024_may_jun",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dumps", nargs="+", required=True)
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--station-summaries", nargs="+", required=True)
    parser.add_argument("--zarr-stores", nargs="+", required=True)
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
    weights = 2 * np.arange(1, count + 1) - count - 1
    result[valid] = first - np.sum(ordered * weights[None, :], axis=1) / (count * (count - 1))
    return result


def metrics(members: np.ndarray, truth: np.ndarray) -> dict:
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
        "correlation": float(np.corrcoef(mean, truth)[0, 1]) if mean.std() and truth.std() else None,
    }


def score_groups(members: np.ndarray, truth: np.ndarray, source: np.ndarray) -> dict:
    return {
        "pooled": metrics(members, truth),
        "BMD": metrics(members[source == "BMD"], truth[source == "BMD"]),
        "BWDB": metrics(members[source == "BWDB"], truth[source == "BWDB"]),
    }


def main() -> None:
    args = parse_args()
    collections = [args.dumps, args.reports, args.manifests, args.station_summaries, args.zarr_stores]
    if any(len(values) != len(PERIODS) for values in collections):
        raise ValueError("each input collection must contain exactly four period files/stores")

    period_results: dict[str, dict] = {}
    pooled_truth, pooled_source = [], []
    pooled_members = {method: [] for method in METHODS}
    production_catalog = []
    for dump_name, report_name, manifest_name, summary_name, zarr_name in zip(*collections):
        for required in (dump_name, report_name, manifest_name, summary_name):
            if not Path(required).is_file():
                raise FileNotFoundError(required)
        if not Path(zarr_name).is_dir():
            raise FileNotFoundError(zarr_name)
        report = json.loads(Path(report_name).read_text())
        manifest = json.loads(Path(manifest_name).read_text())
        scope = report["scope"]
        period_key = (scope["start"], scope["end"])
        if period_key not in PERIODS:
            raise ValueError(f"unexpected period {period_key}")
        label = PERIODS[period_key]
        selection = manifest["analysis_selection"]
        if selection["holdout_folds"] != 1 or selection["holdout_fraction_each_fold"] != 0.2:
            raise ValueError(f"{label}: expected one 20% holdout")
        if selection["support_radius_km"] != 15.0 or selection["maximum_retained_neighbour_km"] > 15.0:
            raise ValueError(f"{label}: retained-neighbour constraint is not <=15 km")
        if scope.get("assimilate_all_stations"):
            raise ValueError(f"{report_name}: evaluation report assimilates all stations")

        dump = np.load(dump_name, allow_pickle=False)
        method_names = dump["variant_names"].astype(str).tolist()
        if method_names != METHODS:
            raise ValueError(f"{dump_name}: expected only {METHODS}, got {method_names}")
        station_ids = dump["station_ids"].astype(str)
        eval_idx = np.asarray(dump["eval_idx"], int)
        expected = int(round(selection["analysis_stations"] * 0.2))
        if len(eval_idx) != expected:
            raise ValueError(f"{label}: expected {expected} withheld stations, got {len(eval_idx)}")
        source_map = pd.read_csv(summary_name).set_index("station_id")["source"].astype(str).to_dict()
        eval_ids = station_ids[eval_idx]
        station_source = np.asarray([source_map.get(station) for station in eval_ids])
        if np.any(pd.isna(station_source)):
            raise ValueError(f"{label}: held-out station is missing source metadata")
        truth_matrix = np.asarray(dump["gauge_mm"][:, eval_idx], float)
        truth = truth_matrix.reshape(-1)
        source = np.tile(station_source, truth_matrix.shape[0])
        period_entry = {
            "period": manifest["period"],
            "selection": selection,
            "withheld_stations": int(len(eval_idx)),
            "withheld_by_source": {
                "BMD": int(np.sum(station_source == "BMD")),
                "BWDB": int(np.sum(station_source == "BWDB")),
            },
            "methods": {},
            "temporal_support_note": manifest["temporal_support_note"],
        }
        for method in METHODS:
            ensemble = np.asarray(dump[f"station_{method}"][:, :, eval_idx], float)
            ensemble = np.moveaxis(ensemble, 1, 2).reshape(-1, ensemble.shape[1])
            period_entry["methods"][method] = score_groups(ensemble, truth, source)
            pooled_members[method].append(ensemble)
        pooled_truth.append(truth)
        pooled_source.append(source)
        production_catalog.append({"period": label, "zarr": zarr_name})
        period_results[label] = period_entry
        dump.close()

    truth = np.concatenate(pooled_truth)
    source = np.concatenate(pooled_source)
    aggregate = {
        method: score_groups(np.concatenate(pooled_members[method]), truth, source)
        for method in METHODS
    }
    result = {
        "experiment": "combined BMD+BWDB CPC-v2 Huber3 winner, 2021-2024",
        "methods": METHODS,
        "periods": period_results,
        "aggregate": aggregate,
        "production_zarr": production_catalog,
        "scope_note": (
            "Withheld files provide independent gauge evaluation. All-station Zarr stores are production "
            "analyses and are catalogued here but are not used to compute gauge skill."
        ),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "huber3_2021_2024_scores.json").write_text(json.dumps(result, indent=2) + "\n")

    lines = [
        "# Combined BMD/BWDB CPC-v2 Huber3 production evaluation, 2021–2024", "",
        "Each seasonal evaluation withholds a deterministic random 20% of stations, with at least one assimilated station within 15 km. The matched all-station run writes the production ensemble fields.", "",
        "| Period | Method | Pool RMSE | BMD RMSE | BWDB RMSE | Pool CRPS | BMD CRPS | BWDB CRPS |", "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    table_periods = list(period_results) + ["all_periods"]
    for label in table_periods:
        block = aggregate if label == "all_periods" else period_results[label]["methods"]
        for method in METHODS:
            values = block[method]
            lines.append(
                f"| {label} | {method} | {values['pooled']['rmse_mm']:.3f} | {values['BMD']['rmse_mm']:.3f} | {values['BWDB']['rmse_mm']:.3f} | {values['pooled']['crps_mm']:.3f} | {values['BMD']['crps_mm']:.3f} | {values['BWDB']['crps_mm']:.3f} |"
            )
    lines.extend(["", "All-station Zarr stores are analysis products, not independent gauge verification.", ""])
    (out_dir / "huber3_2021_2024_scores.md").write_text("\n".join(lines))

    labels = list(period_results) + ["All"]
    positions = np.arange(len(labels))
    width = 0.36
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for axis, metric, title in zip(axes, ("crps_mm", "rmse_mm"), ("Fair CRPS", "Ensemble-mean RMSE")):
        for offset, method, color in ((-width / 2, BACKGROUND, "#8c8c8c"), (width / 2, WINNER, "#1769aa")):
            values = [period_results[label]["methods"][method]["pooled"][metric] for label in period_results]
            values.append(aggregate[method]["pooled"][metric])
            axis.bar(positions + offset, values, width, label=method, color=color)
        axis.set_xticks(positions, ["2021", "2022", "2023", "2024", "All"])
        axis.set_ylabel("mm")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    figure.suptitle("CPC-v2 combined BMD/BWDB 15-km constrained holdout")
    figure.savefig(out_dir / "huber3_2021_2024_scores.png", dpi=180)
    plt.close(figure)
    print(json.dumps({"out_dir": str(out_dir), "periods": list(period_results)}, indent=2))


if __name__ == "__main__":
    main()
