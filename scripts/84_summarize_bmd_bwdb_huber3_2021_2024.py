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
WET_MM = 1.0
SELECTION_START = np.datetime64("2022-05-01", "D")
SELECTION_END = np.datetime64("2022-05-31", "D")
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
    parser.add_argument("--winner", default=None, help="Name of candidate method (default: auto-detected or v2_simul_s04_huber3)")
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
    wet = truth >= WET_MM
    low, high = np.quantile(members, [0.05, 0.95], axis=1)
    rmse = float(np.sqrt(np.mean(error**2)))
    spread = float(np.sqrt(np.mean(np.var(members, axis=1, ddof=1))))
    return {
        "n": int(len(truth)),
        "n_wet": int(wet.sum()),
        "mae_mm": float(np.mean(np.abs(error))),
        "dry_mae_mm": float(np.mean(np.abs(error[~wet]))) if (~wet).any() else None,
        "wet_mae_mm": float(np.mean(np.abs(error[wet]))) if wet.any() else None,
        "rmse_mm": rmse,
        "bias_mm": float(np.mean(error)),
        "crps_mm": float(np.nanmean(fair_crps(members, truth))),
        "correlation": float(np.corrcoef(mean, truth)[0, 1]) if mean.std() and truth.std() else None,
        "spread_mm": spread,
        "spread_skill_ratio": spread / rmse if rmse else None,
        "coverage_90": float(np.mean((truth >= low) & (truth <= high))),
    }


def score_groups(members: np.ndarray, truth: np.ndarray, source: np.ndarray) -> dict:
    return {
        "pooled": metrics(members, truth),
        "BMD": metrics(members[source == "BMD"], truth[source == "BMD"]),
        "BWDB": metrics(members[source == "BWDB"], truth[source == "BWDB"]),
    }


def monthly_samples(data: dict) -> dict:
    """Aggregate to station-month means when >=80% of requested days are valid."""
    months = data["date"].astype("datetime64[M]")
    output = {"truth": [], "source": []}
    output["members"] = {method: [] for method in METHODS}
    for month in np.unique(months):
        for station in np.unique(data["station"][months == month]):
            choose = (months == month) & (data["station"] == station)
            requested_days = len(np.unique(data["date"][choose]))
            valid = np.isfinite(data["truth"][choose])
            for method in METHODS:
                valid &= np.all(np.isfinite(data["members"][method][choose]), axis=1)
            if valid.sum() < int(np.ceil(0.8 * requested_days)):
                continue
            output["truth"].append(float(np.mean(data["truth"][choose][valid])))
            output["source"].append(str(data["source"][choose][0]))
            for method in METHODS:
                output["members"][method].append(
                    np.mean(data["members"][method][choose][valid], axis=0)
                )
    return {
        "truth": np.asarray(output["truth"], float),
        "source": np.asarray(output["source"], str),
        "members": {method: np.asarray(values, float) for method, values in output["members"].items()},
    }


def select_data(data: dict, exclude_selection: bool) -> dict:
    mask = np.ones(len(data["truth"]), dtype=bool)
    if exclude_selection:
        mask &= ~((data["date"] >= SELECTION_START) & (data["date"] <= SELECTION_END))
    return {
        "date": data["date"][mask], "station": data["station"][mask],
        "source": data["source"][mask], "truth": data["truth"][mask],
        "members": {method: data["members"][method][mask] for method in METHODS},
    }


def score_scopes(data: dict, exclude_selection: bool = True) -> dict:
    data = select_data(data, exclude_selection)
    monthly = monthly_samples(data)
    return {
        "daily": {method: score_groups(data["members"][method], data["truth"], data["source"]) for method in METHODS},
        "monthly": {method: score_groups(monthly["members"][method], monthly["truth"], monthly["source"]) for method in METHODS},
    }


def paired_daily_crps(data: dict, block_days: int, n_resamples: int, seed: int) -> dict:
    """Paired circular day-block uncertainty for Huber3 relative to background."""
    data = select_data(data, exclude_selection=True)
    candidate = fair_crps(data["members"][WINNER], data["truth"])
    reference = fair_crps(data["members"][BACKGROUND], data["truth"])
    dates = np.unique(data["date"])
    difference = np.asarray([
        np.nanmean((reference - candidate)[data["date"] == date]) for date in dates
    ])
    split = np.where(np.diff(dates).astype("timedelta64[D]").astype(int) > 1)[0] + 1
    segments = [segment for segment in np.split(difference, split) if len(segment)]
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_resamples, float)
    for index in range(n_resamples):
        samples = []
        for segment in segments:
            width = max(1, min(block_days, len(segment)))
            starts = rng.integers(0, len(segment), size=int(np.ceil(len(segment) / width)))
            take = (starts[:, None] + np.arange(width)[None, :]).reshape(-1)[:len(segment)] % len(segment)
            samples.append(segment[take])
        estimates[index] = np.nanmean(np.concatenate(samples))
    low, high = np.nanpercentile(estimates, [2.5, 97.5])
    return {
        "candidate": WINNER, "reference": BACKGROUND,
        "crps_improvement_mm": float(np.nanmean(difference)),
        "ci_low": float(low), "ci_high": float(high),
        "significant": bool(low > 0 or high < 0), "block_days": int(block_days),
        "n_resamples": int(n_resamples), "n_days": int(len(dates)),
    }


def validate_zarr_stores(paths: list[str]) -> list[dict]:
    import xarray as xr

    output = []
    for name in paths:
        path = Path(name)
        if not path.is_dir():
            raise FileNotFoundError(path)
        dataset = xr.open_zarr(path, consolidated=True)
        scope = dataset.attrs.get("scope", {})
        methods = dataset.method.values.astype(str).tolist()
        if not dataset.attrs.get("complete") or not scope.get("assimilate_all_stations"):
            raise ValueError(f"{path}: incomplete or not all-station")
        if methods != METHODS:
            raise ValueError(f"{path}: expected methods {METHODS}, got {methods}")
        if not bool(np.asarray(dataset.assimilated_station).all()):
            raise ValueError(f"{path}: has non-assimilated production stations")
        output.append({
            "path": str(path), "start": scope["start"], "end": scope["end"],
            "days": int(dataset.sizes["time"]), "members": int(dataset.sizes["member"]),
            "methods": methods, "variables": sorted(dataset.data_vars),
        })
        dataset.close()
    return sorted(output, key=lambda value: value["start"])


def plot_fold_diagnostic(label: str, data: dict, out_dir: Path) -> str:
    data = select_data(data, exclude_selection=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    scores = {method: metrics(data["members"][method], data["truth"]) for method in METHODS}
    daily = {}
    for method in METHODS:
        crps = fair_crps(data["members"][method], data["truth"])
        dates = np.unique(data["date"])
        daily[method] = np.asarray([np.nanmean(crps[data["date"] == date]) for date in dates])
    positions = np.arange(len(METHODS))
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    labels = ["background", "Huber3"]
    axes[0, 0].barh(positions, [scores[m]["crps_mm"] for m in METHODS], color="#C1440E")
    axes[0, 0].set_yticks(positions, labels); axes[0, 0].invert_yaxis(); axes[0, 0].set_title("A. Held-out CRPS")
    axes[0, 1].barh(positions, [scores[m]["bias_mm"] for m in METHODS], color="#D1495B")
    axes[0, 1].axvline(0, color="black", lw=1); axes[0, 1].set_yticks(positions, labels); axes[0, 1].invert_yaxis(); axes[0, 1].set_title("B. Bias")
    width = 0.38
    axes[1, 0].barh(positions - width / 2, [scores[m]["correlation"] for m in METHODS], height=width, label="correlation", color="#457B9D")
    axes[1, 0].barh(positions + width / 2, [scores[m]["coverage_90"] for m in METHODS], height=width, label="coverage 90", color="#6A4C93")
    axes[1, 0].set_xlim(0, 1); axes[1, 0].set_yticks(positions, labels); axes[1, 0].invert_yaxis(); axes[1, 0].legend(); axes[1, 0].set_title("C. Correlation and coverage")
    dates = np.unique(data["date"])
    for method, text in zip(METHODS, labels):
        axes[1, 1].plot(dates, daily[method], label=text, lw=1)
    axes[1, 1].legend(); axes[1, 1].set_title("D. Daily held-out CRPS"); axes[1, 1].tick_params(axis="x", rotation=20)
    figure.suptitle(f"{label}: single constrained 20% holdout (15 km retained-neighbour rule; May 2022 excluded)")
    output = out_dir / f"{label}_fold0_diagnostics.png"
    figure.savefig(output, dpi=150); plt.close(figure)
    return str(output)


def main() -> None:
    args = parse_args()
    collections = [args.dumps, args.reports, args.manifests, args.station_summaries, args.zarr_stores]
    if any(len(values) != len(PERIODS) for values in collections):
        raise ValueError("each input collection must contain exactly four period files/stores")

    global WINNER, METHODS
    if args.winner:
        WINNER = args.winner
    else:
        first_dump = np.load(args.dumps[0], allow_pickle=False)
        cand_methods = [m for m in first_dump["variant_names"].astype(str).tolist() if m != BACKGROUND]
        if cand_methods:
            WINNER = cand_methods[0]
    METHODS = [BACKGROUND, WINNER]

    period_results: dict[str, dict] = {}
    period_data: dict[str, dict] = {}
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
        dates = np.repeat(dump["times"].astype("datetime64[D]"), len(eval_idx))
        stations = np.tile(eval_ids, truth_matrix.shape[0])
        data = {"date": dates, "station": stations, "source": source, "truth": truth, "members": {}}
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
            data["members"][method] = ensemble
            pooled_members[method].append(ensemble)
        period_entry["methods"] = score_scopes(data)["daily"]
        period_entry["monthly_methods"] = score_scopes(data)["monthly"]
        pooled_truth.append(truth)
        pooled_source.append(source)
        production_catalog.append({"period": label, "zarr": zarr_name})
        period_results[label] = period_entry
        period_data[label] = data
        dump.close()

    truth = np.concatenate(pooled_truth)
    source = np.concatenate(pooled_source)
    pooled_data = {
        "date": np.concatenate([period_data[label]["date"] for label in period_results]),
        "station": np.concatenate([period_data[label]["station"] for label in period_results]),
        "source": source, "truth": truth,
        "members": {method: np.concatenate(pooled_members[method]) for method in METHODS},
    }
    aggregate = score_scopes(pooled_data)
    paired = paired_daily_crps(pooled_data, block_days=3, n_resamples=10_000, seed=202420)
    zarr_catalog = validate_zarr_stores(args.zarr_stores)
    result = {
        "experiment": "combined BMD+BWDB CPC-v2 Huber3 winner, 2021-2024",
        "methods": METHODS,
        "periods": period_results,
        "aggregate": aggregate,
        "paired_daily_crps": paired,
        "production_zarr": zarr_catalog,
        "scope_note": (
            "Withheld files provide independent gauge evaluation. All-station Zarr stores are production "
            "analyses and are catalogued here but are not used to compute gauge skill."
        ),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "huber3_2021_2024_scores.json").write_text(json.dumps(result, indent=2) + "\n")
    (out_dir / "gridded_catalog.json").write_text(json.dumps(zarr_catalog, indent=2) + "\n")

    lines = [
        "# Combined BMD/BWDB CPC-v2 Huber3 production evaluation, 2021–2024", "",
        "Each seasonal evaluation withholds a deterministic random 20% of stations, with at least one assimilated station within 15 km. The matched all-station run writes the production ensemble fields.", "",
        "| Period | Method | Pool RMSE | BMD RMSE | BWDB RMSE | Pool CRPS | BMD CRPS | BWDB CRPS |", "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    table_periods = list(period_results) + ["all_periods"]
    for label in table_periods:
        block = aggregate["daily"] if label == "all_periods" else period_results[label]["methods"]
        for method in METHODS:
            values = block[method]
            lines.append(
                f"| {label} | {method} | {values['pooled']['rmse_mm']:.3f} | {values['BMD']['rmse_mm']:.3f} | {values['BWDB']['rmse_mm']:.3f} | {values['pooled']['crps_mm']:.3f} | {values['BMD']['crps_mm']:.3f} | {values['BWDB']['crps_mm']:.3f} |"
            )
    lines.extend([
        "", "## Paired daily CRPS", "",
        f"Huber3 improvement over background: **{paired['crps_improvement_mm']:+.3f} mm/day** "
        f"(95% circular 3-day block interval [{paired['ci_low']:+.3f}, {paired['ci_high']:+.3f}]).",
        "", "All-station Zarr stores are analysis products, not independent gauge verification.", "",
    ])
    (out_dir / "huber3_2021_2024_scores.md").write_text("\n".join(lines))

    labels = list(period_results) + ["All"]
    positions = np.arange(len(labels))
    width = 0.36
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for axis, metric, title in zip(axes, ("crps_mm", "rmse_mm"), ("Fair CRPS", "Ensemble-mean RMSE")):
        for offset, method, color in ((-width / 2, BACKGROUND, "#8c8c8c"), (width / 2, WINNER, "#1769aa")):
            values = [period_results[label]["methods"][method]["pooled"][metric] for label in period_results]
            values.append(aggregate["daily"][method]["pooled"][metric])
            axis.bar(positions + offset, values, width, label=method, color=color)
        axis.set_xticks(positions, ["2021", "2022", "2023", "2024", "All"])
        axis.set_ylabel("mm")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    figure.suptitle("CPC-v2 combined BMD/BWDB 15-km constrained holdout")
    figure.savefig(out_dir / "huber3_2021_2024_scores.png", dpi=180)
    plt.close(figure)
    fold_plots = [plot_fold_diagnostic(label, period_data[label], out_dir / "fold_plots") for label in period_results]
    result["fold_plots"] = fold_plots
    (out_dir / "huber3_2021_2024_scores.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out_dir": str(out_dir), "periods": list(period_results), "fold_plots": fold_plots}, indent=2))


if __name__ == "__main__":
    main()
