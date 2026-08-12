#!/usr/bin/env python
"""Pool the frozen 2021-2024 CPC-v2 DA confirmation experiment.

Five spatial folds are pooled so every eligible BMD station is verified exactly
once per requested period. The primary confirmatory daily analysis excludes
2022-05-01..10, which selected ``ig010`` and ``huber3``. Monthly confirmatory
scores exclude all of May 2022 because a May mean containing tuning days is not
independent. Results using every requested day are retained as descriptive
secondary output.

The four all-station Zarr shards are validated and catalogued, but never used
for gauge skill: every station in those stores entered the likelihood.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


BACKGROUND = "background"
GAUGES = "guided_s6_g010_t100"
CURRENT = "v2_simultaneous_s04_t100"
PRIMARY = "v2_simul_s04_ig010"
CHALLENGER = "v2_simul_s04_huber3"
METHODS = [BACKGROUND, GAUGES, CURRENT, PRIMARY, CHALLENGER]
PERIODS = {
    "2021_may_sep": ("2021-05-01", "2021-09-30"),
    "2022_may_sep": ("2022-05-01", "2022-09-30"),
    "2023_may_sep": ("2023-05-01", "2023-09-30"),
    "2024_may_jun": ("2024-05-01", "2024-06-30"),
}
SELECTION_START = np.datetime64("2022-05-01", "D")
SELECTION_END = np.datetime64("2022-05-10", "D")
WET_MM = 1.0


@dataclass
class FoldFile:
    period: str
    fold: int
    dump: np.lib.npyio.NpzFile
    report: dict
    dump_path: str
    report_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dumps", nargs="+", required=True)
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--zarr-stores", nargs="+", required=True)
    parser.add_argument("--block-days", type=int, default=3)
    parser.add_argument("--n-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=202210)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def period_for(start: str, end: str) -> str:
    matches = [name for name, bounds in PERIODS.items() if bounds == (start, end)]
    if len(matches) != 1:
        raise ValueError(f"unexpected period {start}..{end}; expected {PERIODS}")
    return matches[0]


def load_and_validate(dump_paths: list[Path], report_paths: list[Path]) -> list[FoldFile]:
    if len(dump_paths) != len(report_paths):
        raise ValueError("--dumps and --reports require equal counts")
    files = []
    for dump_path, report_path in zip(dump_paths, report_paths):
        if not dump_path.is_file() or not report_path.is_file():
            raise FileNotFoundError(f"missing pair {dump_path}, {report_path}")
        dump = np.load(dump_path, allow_pickle=False)
        report = json.loads(report_path.read_text())
        scope = report["scope"]
        period = period_for(scope["start"], scope["end"])
        files.append(
            FoldFile(
                period=period,
                fold=int(scope["holdout_fold"]),
                dump=dump,
                report=report,
                dump_path=str(dump_path),
                report_path=str(report_path),
            )
        )
    files.sort(key=lambda item: (item.period, item.fold))
    if len(files) != len(PERIODS) * 5:
        raise ValueError(f"expected {len(PERIODS) * 5} fold files, got {len(files)}")

    reference_scope = files[0].report["scope"]
    immutable = (
        "members", "checkpoint", "checkpoint_data", "checkpoint_stats",
        "background_day_offset", "seed", "group", "precip_transform",
        "config_overrides", "holdout_folds",
    )
    for period in PERIODS:
        block = [item for item in files if item.period == period]
        if [item.fold for item in block] != list(range(5)):
            raise ValueError(f"{period}: need folds 0..4")
        times = block[0].dump["times"].astype(str)
        station_ids = block[0].dump["station_ids"].astype(str)
        variants = block[0].dump["variant_names"].astype(str).tolist()
        if variants != METHODS:
            raise ValueError(f"{period}: expected methods {METHODS}, got {variants}")
        start, end = PERIODS[period]
        expected_times = np.arange(
            np.datetime64(start, "D"),
            np.datetime64(end, "D") + np.timedelta64(1, "D"),
        ).astype(str)
        if not np.array_equal(times, expected_times):
            raise ValueError(
                f"{period}: CV dates are incomplete or out of order; got "
                f"{len(times)}, expected {len(expected_times)} exact days"
            )
        withheld = []
        for item in block:
            scope = item.report["scope"]
            if scope.get("assimilate_all_stations"):
                raise ValueError(f"{item.report_path}: CV file assimilates all stations")
            for key in immutable:
                if scope.get(key) != reference_scope.get(key):
                    raise ValueError(f"{item.report_path}: differs on {key}")
            if not np.array_equal(item.dump["times"].astype(str), times):
                raise ValueError(f"{period}: fold dates differ")
            if not np.array_equal(item.dump["station_ids"].astype(str), station_ids):
                raise ValueError(f"{period}: fold station order differs")
            if int(scope["n_days"]) != len(expected_times):
                raise ValueError(f"{item.report_path}: scope n_days is inconsistent")
            withheld.extend(station_ids[item.dump["eval_idx"]].tolist())
        if len(withheld) != len(set(withheld)) or set(withheld) != set(station_ids):
            raise ValueError(f"{period}: every station must be withheld exactly once")
    return files


def fair_crps(members: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Fair ensemble CRPS for members ``(sample,member)``."""
    members = np.asarray(members, float)
    truth = np.asarray(truth, float)
    count = members.shape[1]
    result = np.full(truth.shape, np.nan)
    finite = np.isfinite(truth) & np.all(np.isfinite(members), axis=1)
    if not finite.any():
        return result
    selected = members[finite]
    first = np.mean(np.abs(selected - truth[finite, None]), axis=1)
    ordered = np.sort(selected, axis=1)
    weights = 2 * np.arange(1, count + 1) - count - 1
    pair = np.sum(ordered * weights[None, :], axis=1) / (count * (count - 1))
    result[finite] = first - pair
    return result


def collect(files: list[FoldFile]) -> dict:
    dates, stations, truth = [], [], []
    members = {name: [] for name in METHODS}
    station_coordinates: dict[str, tuple[float, float]] = {}
    sources = []
    for item in files:
        dump = item.dump
        eval_idx = np.asarray(dump["eval_idx"], int)
        times = dump["times"].astype("datetime64[D]")
        ids = dump["station_ids"].astype(str)
        observed = np.asarray(dump["gauge_mm"][:, eval_idx], float)
        dates.append(np.repeat(times, len(eval_idx)))
        stations.append(np.tile(ids[eval_idx], len(times)))
        truth.append(observed.reshape(-1))
        for name in METHODS:
            ensemble = np.asarray(dump[f"station_{name}"][:, :, eval_idx], float)
            members[name].append(np.moveaxis(ensemble, 1, 2).reshape(-1, ensemble.shape[1]))
        for index, station in enumerate(ids):
            coordinate = (
                float(dump["station_lat"][index]),
                float(dump["station_lon"][index]),
            )
            if station in station_coordinates and station_coordinates[station] != coordinate:
                raise ValueError(f"station {station} coordinates change across files")
            station_coordinates[station] = coordinate
        sources.append({
            "period": item.period,
            "fold": item.fold,
            "dump": item.dump_path,
            "report": item.report_path,
        })

    output = {
        "date": np.concatenate(dates),
        "station": np.concatenate(stations),
        "truth": np.concatenate(truth),
        "members": {name: np.concatenate(parts) for name, parts in members.items()},
        "station_coordinates": station_coordinates,
        "sources": sources,
    }
    order = np.lexsort((output["station"], output["date"]))
    for key in ("date", "station", "truth"):
        output[key] = output[key][order]
    for name in METHODS:
        output["members"][name] = output["members"][name][order]
    keys = np.asarray(
        [f"{date}|{station}" for date, station in zip(output["date"], output["station"])]
    )
    if len(keys) != len(np.unique(keys)):
        raise ValueError("a date/station pair appears in more than one withheld fold")
    return output


def metrics(members: np.ndarray, truth: np.ndarray) -> dict:
    members = np.asarray(members, float)
    truth = np.asarray(truth, float)
    finite = np.isfinite(truth) & np.all(np.isfinite(members), axis=1)
    members = members[finite]
    truth = truth[finite]
    if not len(truth):
        return {"n": 0}
    mean = members.mean(axis=1)
    difference = mean - truth
    wet = truth >= WET_MM
    low, high = np.quantile(members, [0.05, 0.95], axis=1)
    rmse = float(np.sqrt(np.mean(difference**2)))
    spread = float(np.sqrt(np.mean(np.var(members, axis=1, ddof=1))))
    return {
        "n": int(len(truth)),
        "n_wet": int(wet.sum()),
        "crps": float(np.mean(fair_crps(members, truth))),
        "mae": float(np.mean(np.abs(difference))),
        "dry_mae": float(np.mean(np.abs(difference[~wet]))) if (~wet).any() else None,
        "wet_mae": float(np.mean(np.abs(difference[wet]))) if wet.any() else None,
        "bias": float(np.mean(difference)),
        "rmse": rmse,
        "correlation": (
            float(np.corrcoef(mean, truth)[0, 1])
            if mean.std() > 0 and truth.std() > 0 else None
        ),
        "spread": spread,
        "spread_skill": spread / rmse if rmse else None,
        "coverage_90": float(np.mean((truth >= low) & (truth <= high))),
    }


def daily_mask(data: dict, confirmatory: bool) -> np.ndarray:
    finite = np.isfinite(data["truth"])
    if not confirmatory:
        return finite
    selected = (data["date"] >= SELECTION_START) & (data["date"] <= SELECTION_END)
    return finite & ~selected


def monthly_samples(data: dict, confirmatory: bool) -> dict:
    """Aggregate station ensembles to requested calendar-month means."""
    dates = data["date"]
    months = dates.astype("datetime64[M]")
    output_truth, output_month, output_station = [], [], []
    output_members = {name: [] for name in METHODS}
    for month in np.unique(months):
        if confirmatory and month == np.datetime64("2022-05", "M"):
            continue
        requested_days = len(np.unique(dates[months == month]))
        required = int(np.ceil(0.8 * requested_days))
        for station in np.unique(data["station"][months == month]):
            choose = (months == month) & (data["station"] == station)
            finite = np.isfinite(data["truth"][choose])
            for name in METHODS:
                finite &= np.all(np.isfinite(data["members"][name][choose]), axis=1)
            if finite.sum() < required:
                continue
            output_truth.append(float(np.mean(data["truth"][choose][finite])))
            output_month.append(month)
            output_station.append(station)
            for name in METHODS:
                output_members[name].append(
                    np.mean(data["members"][name][choose][finite], axis=0)
                )
    return {
        "month": np.asarray(output_month, dtype="datetime64[M]"),
        "station": np.asarray(output_station),
        "truth": np.asarray(output_truth, float),
        "members": {
            name: np.asarray(values, float) for name, values in output_members.items()
        },
    }


def score_scope(data: dict, confirmatory: bool) -> dict:
    mask = daily_mask(data, confirmatory)
    monthly = monthly_samples(data, confirmatory)
    return {
        "daily": {
            name: metrics(data["members"][name][mask], data["truth"][mask])
            for name in METHODS
        },
        "monthly": {
            name: metrics(monthly["members"][name], monthly["truth"])
            for name in METHODS
        },
        "n_dates": int(len(np.unique(data["date"][mask]))),
        "start": str(np.min(data["date"][mask])),
        "end": str(np.max(data["date"][mask])),
        "excluded_selection_dates": bool(confirmatory),
        "excluded_monthly_months": ["2022-05"] if confirmatory else [],
    }


def daily_crps_difference(
    data: dict, candidate: str, reference: str, confirmatory: bool
) -> tuple[np.ndarray, np.ndarray]:
    mask = daily_mask(data, confirmatory)
    candidate_crps = fair_crps(data["members"][candidate], data["truth"])
    reference_crps = fair_crps(data["members"][reference], data["truth"])
    dates = np.unique(data["date"][mask])
    difference = np.asarray([
        np.nanmean((reference_crps - candidate_crps)[mask & (data["date"] == date)])
        for date in dates
    ])
    return dates, difference


def segmented_block_bootstrap(
    dates: np.ndarray,
    difference: np.ndarray,
    block_days: int,
    n_resamples: int,
    seed: int,
) -> dict:
    """Paired circular day-block bootstrap that cannot cross seasonal gaps."""
    dates = np.asarray(dates).astype("datetime64[D]")
    difference = np.asarray(difference, float)
    split = np.where(np.diff(dates).astype("timedelta64[D]").astype(int) > 1)[0] + 1
    segments = [segment for segment in np.split(difference, split) if len(segment)]
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_resamples, float)
    for sample_index in range(n_resamples):
        sampled = []
        for segment in segments:
            length = len(segment)
            width = max(1, min(block_days, length))
            blocks = int(np.ceil(length / width))
            starts = rng.integers(0, length, size=blocks)
            indices = (
                starts[:, None] + np.arange(width)[None, :]
            ).reshape(-1)[:length] % length
            sampled.append(segment[indices])
        estimates[sample_index] = np.nanmean(np.concatenate(sampled))
    low, high = np.nanpercentile(estimates, [2.5, 97.5])
    return {
        "difference": float(np.nanmean(difference)),
        "ci_low": float(low),
        "ci_high": float(high),
        "block_days": int(block_days),
        "n_resamples": int(n_resamples),
        "n_days": int(len(difference)),
        "n_segments": int(len(segments)),
        "significant": bool(low > 0 or high < 0),
    }


def comparisons(
    data: dict, confirmatory: bool, block_days: int, n_resamples: int, seed: int
) -> dict:
    pairs = [
        ("current_vs_gauges", CURRENT, GAUGES),
        ("ig010_vs_gauges", PRIMARY, GAUGES),
        ("huber3_vs_gauges", CHALLENGER, GAUGES),
        ("ig010_vs_current", PRIMARY, CURRENT),
        ("huber3_vs_current", CHALLENGER, CURRENT),
        ("huber3_vs_ig010", CHALLENGER, PRIMARY),
    ]
    output = {}
    for index, (label, candidate, reference) in enumerate(pairs):
        dates, difference = daily_crps_difference(
            data, candidate, reference, confirmatory
        )
        result = segmented_block_bootstrap(
            dates, difference, block_days, n_resamples, seed + index * 10_000
        )
        result.update(candidate=candidate, reference=reference)
        output[label] = result
    return output


def validate_zarr_stores(paths: list[Path]) -> list[dict]:
    import xarray as xr

    if len(paths) != len(PERIODS):
        raise ValueError(f"expected {len(PERIODS)} Zarr stores, got {len(paths)}")
    entries = []
    seen_periods = set()
    for path in paths:
        if not path.is_dir():
            raise FileNotFoundError(path)
        dataset = xr.open_zarr(path, consolidated=True)
        scope = dataset.attrs.get("scope", {})
        period = period_for(scope.get("start"), scope.get("end"))
        if period in seen_periods:
            raise ValueError(f"duplicate Zarr period {period}")
        seen_periods.add(period)
        if not dataset.attrs.get("complete"):
            raise ValueError(f"{path} is not marked complete")
        if dataset.attrs.get("schema") != "bdhires.physical_ensemble.v1":
            raise ValueError(f"{path} has unknown schema")
        if not scope.get("assimilate_all_stations"):
            raise ValueError(f"{path} is not an all-station production analysis")
        if dataset.method.values.astype(str).tolist() != METHODS:
            raise ValueError(f"{path} methods do not match frozen set")
        if not bool(dataset.assimilated_station.values.all()):
            raise ValueError(f"{path} leaves production stations unassimilated")
        expected_days = (
            np.datetime64(PERIODS[period][1], "D")
            - np.datetime64(PERIODS[period][0], "D")
        ).astype(int) + 1
        if dataset.sizes["time"] != expected_days:
            raise ValueError(f"{path} has {dataset.sizes['time']} days, expected {expected_days}")
        expected_times = np.arange(
            np.datetime64(PERIODS[period][0], "D"),
            np.datetime64(PERIODS[period][1], "D") + np.timedelta64(1, "D"),
        )
        actual_times = np.asarray(dataset.time.values).astype("datetime64[D]")
        if not np.array_equal(actual_times, expected_times):
            raise ValueError(f"{path} dates are incomplete, duplicated, or out of order")
        entries.append({
            "period": period,
            "path": str(path),
            "start": scope["start"],
            "end": scope["end"],
            "days": int(dataset.sizes["time"]),
            "members": int(dataset.sizes["member"]),
            "methods": dataset.method.values.astype(str).tolist(),
            "variables": sorted(dataset.data_vars),
            "dimensions": {name: int(value) for name, value in dataset.sizes.items()},
        })
        dataset.close()
    return sorted(entries, key=lambda item: item["start"])


def fmt(value: float | None, digits: int = 3, signed: bool = False) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return format(value, f"{'+' if signed else ''}.{digits}f")


def report_lines(primary_scores: dict, all_scores: dict, paired: dict) -> list[str]:
    daily = primary_scores["daily"]
    monthly = primary_scores["monthly"]
    block_days = next(iter(paired.values()))["block_days"]
    ordered = sorted(METHODS, key=lambda name: daily[name]["crps"])
    lines = [
        "# Frozen CPC-v2 DA confirmation, 2021–2024",
        "",
        "- Requested archive: May–September 2021–2023 and May–June 2024",
        "- Primary daily confirmation excludes **2022-05-01 through 2022-05-10**",
        "- Primary monthly confirmation excludes **May 2022**",
        f"- Primary daily sample: **{primary_scores['n_dates']} days**, "
        f"{daily[BACKGROUND]['n']:,} withheld station-days",
        "- Five matched spatial folds per period; every eligible station withheld once",
        f"- Paired uncertainty: circular {block_days}-day blocks within each contiguous season",
        "",
        "## Primary daily withheld-gauge scores",
        "",
        "| Method | CRPS | MAE dry/wet | Bias | Corr | Cov90 | Spread/skill |",
        "|:--|--:|:--|--:|--:|--:|--:|",
    ]
    for name in ordered:
        score = daily[name]
        lines.append(
            f"| `{name}` | {score['crps']:.3f} | "
            f"{fmt(score['dry_mae'], 2)}/{fmt(score['wet_mae'], 2)} | "
            f"{fmt(score['bias'], 2, True)} | {fmt(score['correlation'])} | "
            f"{fmt(score['coverage_90'], 2)} | {fmt(score['spread_skill'], 2)} |"
        )
    lines += [
        "",
        "## Primary paired CRPS comparisons",
        "",
    ]
    for label, result in paired.items():
        verdict = (
            "candidate wins" if result["ci_low"] > 0 else
            "candidate loses" if result["ci_high"] < 0 else "unresolved"
        )
        lines.append(
            f"- `{label}`: {result['difference']:+.3f} "
            f"[{result['ci_low']:+.3f}, {result['ci_high']:+.3f}] mm/day — {verdict}"
        )
    lines += [
        "",
        "Positive values favour the candidate named before `_vs_`.",
        "",
        "## Primary monthly withheld-gauge scores",
        "",
        "| Method | CRPS | MAE | Bias | Corr | Cov90 | n station-months |",
        "|:--|--:|--:|--:|--:|--:|--:|",
    ]
    for name in sorted(METHODS, key=lambda value: monthly[value]["crps"]):
        score = monthly[name]
        lines.append(
            f"| `{name}` | {score['crps']:.3f} | {score['mae']:.3f} | "
            f"{fmt(score['bias'], 2, True)} | {fmt(score['correlation'])} | "
            f"{fmt(score['coverage_90'], 2)} | {score['n']:,} |"
        )
    lines += [
        "",
        "## Descriptive all-requested-data check",
        "",
        "These values include the ten selection days and are not the confirmatory endpoint:",
        "",
        "| Method | Daily CRPS | Daily bias | Daily corr |",
        "|:--|--:|--:|--:|",
    ]
    for name in sorted(METHODS, key=lambda value: all_scores["daily"][value]["crps"]):
        score = all_scores["daily"][name]
        lines.append(
            f"| `{name}` | {score['crps']:.3f} | "
            f"{fmt(score['bias'], 2, True)} | {fmt(score['correlation'])} |"
        )
    return lines


def plot_summary(scores: dict, paired: dict, out_path: Path) -> None:
    daily, monthly = scores["daily"], scores["monthly"]
    ordered = sorted(METHODS, key=lambda name: daily[name]["crps"])
    labels = [name.replace("v2_simul_s04_", "").replace("v2_", "") for name in ordered]
    positions = np.arange(len(ordered))
    figure, axes = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True)
    axes[0, 0].barh(positions, [daily[name]["crps"] for name in ordered], color="#C1440E")
    axes[0, 0].set_yticks(positions, labels, fontsize=8); axes[0, 0].invert_yaxis()
    axes[0, 0].set_xlabel("daily CRPS (mm/day)"); axes[0, 0].set_title("A. Withheld gauges")

    pair_names = list(paired)
    centres = np.asarray([paired[name]["difference"] for name in pair_names])
    lows = np.asarray([paired[name]["ci_low"] for name in pair_names])
    highs = np.asarray([paired[name]["ci_high"] for name in pair_names])
    pair_pos = np.arange(len(pair_names))
    axes[0, 1].errorbar(
        centres, pair_pos, xerr=np.vstack([centres - lows, highs - centres]),
        fmt="o", capsize=3, color="#1B4965",
    )
    axes[0, 1].axvline(0, color="black", ls="--")
    axes[0, 1].set_yticks(pair_pos, [name.replace("_", " ") for name in pair_names], fontsize=8)
    axes[0, 1].invert_yaxis(); axes[0, 1].set_xlabel("CRPS(reference) − CRPS(candidate)")
    axes[0, 1].set_title("B. Confirmatory paired intervals")

    width = 0.38
    axes[0, 2].barh(
        positions - width / 2, [daily[name]["dry_mae"] for name in ordered],
        height=width, label="dry", color="#E9C46A",
    )
    axes[0, 2].barh(
        positions + width / 2, [daily[name]["wet_mae"] for name in ordered],
        height=width, label="wet", color="#2A9D8F",
    )
    axes[0, 2].set_yticks(positions, labels, fontsize=8); axes[0, 2].invert_yaxis()
    axes[0, 2].set_xlabel("daily MAE (mm/day)"); axes[0, 2].legend()
    axes[0, 2].set_title("C. Dry/wet trade-off")

    axes[1, 0].barh(positions, [daily[name]["bias"] for name in ordered], color="#D1495B")
    axes[1, 0].axvline(0, color="black")
    axes[1, 0].set_yticks(positions, labels, fontsize=8); axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlabel("daily bias (mm/day)"); axes[1, 0].set_title("D. Mean bias")

    axes[1, 1].barh(
        positions - width / 2, [daily[name]["correlation"] for name in ordered],
        height=width, label="daily", color="#457B9D",
    )
    axes[1, 1].barh(
        positions + width / 2, [monthly[name]["correlation"] for name in ordered],
        height=width, label="monthly", color="#6A4C93",
    )
    axes[1, 1].set_yticks(positions, labels, fontsize=8); axes[1, 1].invert_yaxis()
    axes[1, 1].set_xlabel("correlation"); axes[1, 1].legend()
    axes[1, 1].set_title("E. Daily and monthly correlation")

    axes[1, 2].barh(positions, [monthly[name]["crps"] for name in ordered], color="#6A4C93")
    axes[1, 2].set_yticks(positions, labels, fontsize=8); axes[1, 2].invert_yaxis()
    axes[1, 2].set_xlabel("monthly CRPS (mm/day)")
    axes[1, 2].set_title("F. Monthly mean verification")
    figure.suptitle("Frozen CPC-v2 DA confirmation (selection dates excluded)")
    figure.savefig(out_path, dpi=160)
    plt.close(figure)


def plot_fold_diagnostics(files: list[FoldFile], out_dir: Path) -> list[str]:
    """Write one traceable held-out diagnostic for each period/fold."""
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    short = {
        BACKGROUND: "background",
        GAUGES: "gauges",
        CURRENT: "current",
        PRIMARY: "ig010",
        CHALLENGER: "huber3",
    }
    for item in files:
        dump = item.dump
        eval_idx = np.asarray(dump["eval_idx"], int)
        observed = np.asarray(dump["gauge_mm"][:, eval_idx], float)
        scores, daily_crps = {}, {}
        for name in METHODS:
            ensemble = np.asarray(dump[f"station_{name}"][:, :, eval_idx], float)
            flat = np.moveaxis(ensemble, 1, 2).reshape(-1, ensemble.shape[1])
            scores[name] = metrics(flat, observed.reshape(-1))
            crps = np.full(observed.shape, np.nan)
            for station_index in range(observed.shape[1]):
                crps[:, station_index] = fair_crps(
                    ensemble[:, :, station_index], observed[:, station_index]
                )
            counts = np.isfinite(crps).sum(axis=1)
            daily_crps[name] = np.divide(
                np.nansum(crps, axis=1),
                counts,
                out=np.full(len(crps), np.nan),
                where=counts > 0,
            )

        positions = np.arange(len(METHODS))
        labels = [short[name] for name in METHODS]
        figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
        axes[0, 0].barh(positions, [scores[name]["crps"] for name in METHODS], color="#C1440E")
        axes[0, 0].set_yticks(positions, labels); axes[0, 0].invert_yaxis()
        axes[0, 0].set_xlabel("CRPS (mm/day)"); axes[0, 0].set_title("A. Fold CRPS")

        axes[0, 1].barh(positions, [scores[name]["bias"] for name in METHODS], color="#D1495B")
        axes[0, 1].axvline(0, color="black", lw=1)
        axes[0, 1].set_yticks(positions, labels); axes[0, 1].invert_yaxis()
        axes[0, 1].set_xlabel("bias (mm/day)"); axes[0, 1].set_title("B. Fold bias")

        width = 0.38
        axes[1, 0].barh(
            positions - width / 2,
            [scores[name]["correlation"] for name in METHODS],
            height=width, label="correlation", color="#457B9D",
        )
        axes[1, 0].barh(
            positions + width / 2,
            [scores[name]["coverage_90"] for name in METHODS],
            height=width, label="coverage 90", color="#6A4C93",
        )
        axes[1, 0].set_yticks(positions, labels); axes[1, 0].invert_yaxis()
        axes[1, 0].set_xlim(0, 1); axes[1, 0].legend(fontsize=8)
        axes[1, 0].set_title("C. Correlation and coverage")

        dates = dump["times"].astype("datetime64[D]")
        for name in METHODS:
            axes[1, 1].plot(dates, daily_crps[name], label=short[name], lw=1)
        axes[1, 1].set_ylabel("withheld CRPS (mm/day)")
        axes[1, 1].set_title("D. Daily fold behavior")
        axes[1, 1].legend(ncol=2, fontsize=8)
        axes[1, 1].tick_params(axis="x", rotation=20)

        station_names = dump["station_ids"].astype(str)[eval_idx].tolist()
        figure.suptitle(
            f"{item.period}, fold {item.fold}: withheld " + ", ".join(station_names)
        )
        output = out_dir / f"{item.period}_fold{item.fold}_diagnostics.png"
        figure.savefig(output, dpi=150)
        plt.close(figure)
        outputs.append(str(output))
    return outputs


def main() -> None:
    args = parse_args()
    files = load_and_validate(
        [Path(path) for path in args.dumps],
        [Path(path) for path in args.reports],
    )
    data = collect(files)
    primary_scores = score_scope(data, confirmatory=True)
    all_scores = score_scope(data, confirmatory=False)
    paired = comparisons(
        data, True, args.block_days, args.n_resamples, args.seed
    )
    all_paired = comparisons(
        data, False, args.block_days, args.n_resamples, args.seed + 500_000
    )
    zarr_catalog = validate_zarr_stores([Path(path) for path in args.zarr_stores])

    lines = report_lines(primary_scores, all_scores, paired)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "design": {
            "requested_periods": PERIODS,
            "methods": METHODS,
            "primary_candidate": PRIMARY,
            "secondary_candidate": CHALLENGER,
            "selection_dates_excluded_daily": [str(SELECTION_START), str(SELECTION_END)],
            "selection_months_excluded_monthly": ["2022-05"],
            "block_days": args.block_days,
            "n_resamples": args.n_resamples,
            "note": (
                "All requested dates remain in the gridded Zarr archive. The "
                "primary verification excludes configuration-selection data."
            ),
        },
        "primary_confirmatory": primary_scores,
        "primary_paired_crps": paired,
        "all_requested_descriptive": all_scores,
        "all_requested_paired_crps": all_paired,
        "station_coordinates": {
            name: {"lat": value[0], "lon": value[1]}
            for name, value in data["station_coordinates"].items()
        },
        "cv_sources": data["sources"],
        "gridded_zarr": zarr_catalog,
    }
    (out_dir / "confirmatory_selection.md").write_text("\n".join(lines) + "\n")
    (out_dir / "gridded_catalog.json").write_text(
        json.dumps(zarr_catalog, indent=2, allow_nan=False) + "\n"
    )
    plot_summary(primary_scores, paired, out_dir / "confirmatory_selection.png")
    fold_plots = plot_fold_diagnostics(files, out_dir / "fold_plots")
    payload["fold_plots"] = fold_plots
    (out_dir / "confirmatory_selection.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n"
    )
    print("\n".join(lines))
    print(f"\n[done] wrote {out_dir / 'confirmatory_selection.json'}")
    print(f"[done] wrote {out_dir / 'confirmatory_selection.md'}")
    print(f"[done] wrote {out_dir / 'confirmatory_selection.png'}")
    print(f"[done] wrote {out_dir / 'gridded_catalog.json'}")
    print(f"[done] wrote {len(fold_plots)} diagnostics under {out_dir / 'fold_plots'}")


if __name__ == "__main__":
    main()
