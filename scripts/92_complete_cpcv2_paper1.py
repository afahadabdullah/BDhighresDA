#!/usr/bin/env python3
"""Generate pending Paper 1 artifacts from actual archives; never run inference.

The default archive layout is the same as script 90. Missing inputs leave
manuscript slots pending. Malformed or incompatible available inputs fail.
See docs/PAPER1_COMPLETION.md for optional-input schemas and HPC commands.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bdhires.country import DEFAULT_BOUNDARY, read_boundary, points_inside, map_axes
from bdhires.paper_evidence import paired_intervals
FINAL = "dense_s6_bwdb_r4"
SLOTS = {
    "country_intervals": ["tab_country_intervals.tex", "paired_country_crps.csv"],
    "network": ["fig_network.pdf", "network_geometry.csv"],
    "calibration": ["fig_calibration.pdf", "tab_thresholds.tex", "threshold_scores.csv",
                    "reliability.csv", "rank_histogram.csv", "coverage_curve.csv"],
    "interpolation": ["tab_interpolation.tex", "interpolation_scores.csv",
                      "interpolation_station_days.csv", "paired_idw_intervals.csv",
                      "idw_intensity_scores.csv", "tab_idw_paired.tex", "tab_idw_intensity.tex"],
    "selection": ["tab_selection.tex", "selection_scores.csv"],
    "training": ["fig_training.pdf", "training_curve.csv"],
    "compute": ["tab_compute.tex", "compute_summary.csv"],
    "robustness": ["tab_robustness.tex", "robustness_scores.csv"],
}


def module90():
    spec = importlib.util.spec_from_file_location("_paper1", ROOT / "scripts/90_evaluate_cpcv2_paper1.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tex_escape(value):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
                    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(c, c) for c in str(value))


def table(path, headers, rows, note=""):
    text = [r"\begin{tabular}{" + "l" + "r" * (len(headers) - 1) + "}", r"\toprule",
            " & ".join(tex_escape(h) for h in headers) + r"\\", r"\midrule"]
    text += [" & ".join(tex_escape(v) for v in row) + r"\\" for row in rows]
    text += [r"\bottomrule", r"\end{tabular}"]
    if note:
        text += [r"\par\medskip\begin{minipage}{.96\linewidth}\footnotesize " + tex_escape(note)
                 + r"\end{minipage}"]
    path.write_text("\n".join(text) + "\n")


def distance_km(a, b):
    """Pairwise haversine distances for (lat, lon) arrays."""
    a, b = np.radians(a), np.radians(b)
    d = a[:, None, :] - b[None, :, :]
    h = np.sin(d[..., 0] / 2) ** 2 + np.cos(a[:, None, 0]) * np.cos(b[None, :, 0]) * np.sin(d[..., 1] / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def idw(distances, observations):
    """Daily inverse-distance squared interpolation, excluding missing reports.

    The caller passes retained gauges only. A coincident retained gauge takes
    precedence; multiple coincident reports are averaged. No finite reports
    yield NaN rather than zero rainfall.
    """
    finite = np.isfinite(observations)
    weights = np.where(finite[None, :], 1 / np.maximum(distances, .001) ** 2, 0.)
    coincident = (distances <= .001) & finite[None, :]
    weights[coincident.any(axis=1)] = coincident[coincident.any(axis=1)]
    total = weights.sum(axis=1)
    return np.divide(weights @ np.where(finite, observations, 0.), total,
                     out=np.full(len(distances), np.nan), where=total > 0)


def deterministic(prediction, truth):
    error = prediction - truth
    return {"n": len(truth), "rmse": float(np.sqrt(np.mean(error ** 2))),
            "mae": float(np.mean(np.abs(error))), "bias": float(np.mean(error)),
            "correlation": (float(np.corrcoef(prediction, truth)[0, 1])
                            if prediction.std() > 0 and truth.std() > 0 else None)}


def calibration(members, truth, rng):
    """Randomized ranks, central coverage, raw Brier scores and reliability.

    Ties (including dry-day zero masses) are randomized uniformly among all
    admissible discrete ranks. Reliability bins include both endpoints.
    """
    count = members.shape[1]
    lower = (members < truth[:, None]).sum(axis=1)
    ties = (members == truth[:, None]).sum(axis=1)
    ranks = lower + rng.integers(0, ties + 1)
    histogram = np.bincount(ranks, minlength=count + 1)
    coverage = []
    for level in (.5, .8, .9, .95):
        lo, hi = np.quantile(members, [(1 - level) / 2, (1 + level) / 2], axis=1)
        coverage.append({"nominal": level, "empirical": float(np.mean((truth >= lo) & (truth <= hi)))})
    scores, reliability = [], []
    for threshold in (1, 10, 25, 50, 100):
        event = truth >= threshold
        probability = (members >= threshold).mean(axis=1)
        forecast = members.mean(axis=1) >= threshold
        tp = np.sum(forecast & event)
        denominator = np.sum(forecast | event)
        scores.append({"threshold_mm_day": threshold, "n": len(truth), "events": int(event.sum()),
                       "brier": float(np.mean((probability - event) ** 2)),
                       "mean_csi": float(tp / denominator) if denominator else None})
        bins = np.minimum((probability * 10).astype(int), 9)
        for bin_index in range(10):
            select = bins == bin_index
            reliability.append({"threshold_mm_day": threshold, "bin": bin_index,
                                "n": int(select.sum()),
                                "mean_probability": float(probability[select].mean()) if select.any() else None,
                                "event_frequency": float(event[select].mean()) if select.any() else None})
    return histogram, coverage, scores, reliability


def plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "savefig.dpi": 180,
                         "axes.spines.top": False, "axes.spines.right": False})
    return plt


def independent_mask(data, profile):
    selected = ((data["date"] >= np.datetime64(profile["selection_start"])) &
                (data["date"] <= np.datetime64(profile["selection_end"])))
    keep = np.isfinite(data["truth"]) & ~selected
    for values in data["members"].values():
        keep &= np.isfinite(values).all(axis=1)
    if not keep.any():
        raise ValueError("no independent finite station-days")
    return keep


def build_calibration(data, keep, out, seed):
    plt = plotting()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    thresholds, reliability, ranks, coverages = [], [], [], []
    colors = ["#64748b", "#007f78"]
    for method, color in zip(("background", FINAL), colors):
        label = "Background" if method == "background" else "SURMA-Flow"
        # Resetting the random generator gives a reproducible tie treatment.
        histogram, coverage, scores, rel = calibration(data["members"][method][keep], data["truth"][keep], np.random.default_rng(seed))
        axes[0, 0].plot(np.arange(len(histogram)), histogram / histogram.sum(), label=label, color=color)
        axes[0, 1].plot([r["nominal"] for r in coverage], [r["empirical"] for r in coverage], "o-", label=label, color=color)
        for ax, threshold in zip(axes[1], (25, 50)):
            rows = [r for r in rel if r["threshold_mm_day"] == threshold and r["n"] >= 20]
            ax.plot([r["mean_probability"] for r in rows], [r["event_frequency"] for r in rows], "o-", color=color)
            ax.set(title=f"Rainfall ≥ {threshold} mm/day", xlabel="Predicted probability", ylabel="Observed frequency")
        thresholds += [{"method": method, **r} for r in scores]
        reliability += [{"method": method, **r} for r in rel]
        ranks += [{"method": method, "rank": i, "count": int(n), "fraction": float(n / histogram.sum())} for i, n in enumerate(histogram)]
        coverages += [{"method": method, **r} for r in coverage]
    axes[0, 0].axhline(1 / len(histogram), ls="--", color="black", lw=.8)
    axes[0, 0].set(title="Randomized rank histogram", xlabel="Observation rank", ylabel="Fraction")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].set(title="Central-interval coverage", xlabel="Nominal coverage", ylabel="Empirical coverage")
    for ax in (axes[0, 1], *axes[1]):
        ax.plot([0, 1], [0, 1], "--", color="black", lw=.8)
        ax.set(xlim=(0, 1), ylim=(0, 1))
    fig.savefig(out / "fig_calibration.pdf", bbox_inches="tight")
    plt.close(fig)
    for filename, rows in (("threshold_scores.csv", thresholds), ("reliability.csv", reliability),
                           ("rank_histogram.csv", ranks), ("coverage_curve.csv", coverages)):
        write_csv(out / filename, rows)
    rows = []
    for threshold in (1, 10, 25, 50, 100):
        a, b = [next(r for r in thresholds if r["method"] == m and r["threshold_mm_day"] == threshold) for m in ("background", FINAL)]
        rows.append([threshold, a["events"], f'{a["brier"]:.4f}', f'{b["brier"]:.4f}',
                     "--" if a["mean_csi"] is None else f'{a["mean_csi"]:.3f}',
                     "--" if b["mean_csi"] is None else f'{b["mean_csi"]:.3f}'])
    table(out / "tab_thresholds.tex", ["Threshold", "Events", "BS bg.", "BS analysis", "CSI bg.", "CSI analysis"], rows,
          f"Matched station-days: {int(keep.sum()):,}. Brier scores use uncorrected finite-ensemble exceedance probabilities; CSI uses the ensemble mean.")


def raw_station_data(path):
    values, coords = {}, {}
    for row in read_csv(path):
        station = row["station_id"]
        if station.startswith("SOB_"):
            raise ValueError("IDW requires original retained stations, not super-observation records")
        coordinate = (float(row["lat"]), float(row["lon"]))
        if not np.isfinite(coordinate).all():
            raise ValueError(f"invalid station coordinate: {station}")
        if station in coords and not np.allclose(coords[station], coordinate, rtol=0, atol=1e-6):
            raise ValueError(f"station coordinates change: {station}")
        coords[station] = coordinate
        key = (str(np.datetime64(row["date"], "D")), station)
        if key in values:
            raise ValueError(f"duplicate original station-day: {key}")
        raw = row["precip_mm"].strip()
        try:
            value = .05 if raw.lower() == "t" else float(raw)
        except ValueError:
            value = np.nan
        values[key] = value if value >= 0 and value != 999 else np.nan
    return coords, values


def build_network_idw(data, keep, root, periods, out, country=None,
                      seed=20261001, resamples=10000, block_days=(3, 7)):
    plt = plotting()
    fig, axes = plt.subplots(1, len(periods), figsize=(12, 4.8), layout="constrained", squeeze=False)
    predictions = np.full(len(data["truth"]), np.nan)
    geometry = []
    for ax, period in zip(axes[0], periods):
        coords, values = raw_station_data(root / "stations" / period / "combined_daily.csv")
        if country is not None:
            coords = {s: p for s, p in coords.items() if points_inside(*p, country)}
        with np.load(root / "evaluation" / f"{period}.npz", allow_pickle=False) as dump:
            ids = dump["station_ids"].astype(str)
            indices = dump["eval_idx"]
            if country is not None:
                indices = indices[points_inside(dump["station_lat"][indices], dump["station_lon"][indices], country)]
            withheld = ids[indices].tolist()
            for i in indices:
                if ids[i] not in coords or not np.allclose(coords[ids[i]], [dump["station_lat"][i], dump["station_lon"][i]], rtol=0, atol=1e-5):
                    raise ValueError("original and evaluation station coordinates differ")
        retained = sorted(set(coords) - set(withheld))
        if not retained:
            raise ValueError("no retained original stations")
        distances = distance_km(np.array([coords[s] for s in withheld]), np.array([coords[s] for s in retained]))
        for i, station in enumerate(withheld):
            geometry.append({"period": period, "station_id": station, "role": "withheld",
                             "lat": coords[station][0], "lon": coords[station][1],
                             "nearest_retained_km": float(distances[i].min())})
        for station in retained:
            geometry.append({"period": period, "station_id": station, "role": "retained",
                             "lat": coords[station][0], "lon": coords[station][1], "nearest_retained_km": ""})
        period_mask = (data["period"] == period) & keep
        for day in np.unique(data["date"][period_mask]):
            raw = np.array([values.get((str(day), s), np.nan) for s in retained])
            interpolated = dict(zip(withheld, idw(distances, raw)))
            for index in np.flatnonzero(period_mask & (data["date"] == day)):
                original = values.get((str(day), data["station"][index]), np.nan)
                if not np.isfinite(original) or not np.isclose(original, data["truth"][index], rtol=0, atol=1e-4):
                    raise ValueError("original CSV and archived withheld rainfall differ")
                predictions[index] = interpolated[data["station"][index]]
        for role, marker, color in ((retained, ".", "#64748b"), (withheld, "o", "#b44f42")):
            points = np.array([coords[s] for s in role])
            ax.scatter(points[:, 1], points[:, 0], s=13, marker=marker, color=color,
                       label="Retained" if role is retained else "Withheld")
        ax.set(xlim=(87.6, 94), ylim=(20.3, 26.7), title=period[:4], xlabel="Longitude (°E)", ylabel="Latitude (°N)")
        ax.set_aspect(1 / np.cos(np.radians(23.5)))
        ax.grid(alpha=.15)
        if country is not None:
            map_axes(ax, country)
    axes[0, 0].legend(frameon=False)
    fig.savefig(out / "fig_network.pdf", bbox_inches="tight")
    plt.close(fig)
    write_csv(out / "network_geometry.csv", geometry)
    common = keep & np.isfinite(predictions)
    if not common.any():
        raise ValueError("no finite IDW comparison sample")
    rows = [{"method": method, **deterministic(pred, data["truth"][common])} for method, pred in
            [("background", data["members"]["background"][common].mean(axis=1)),
             (FINAL, data["members"][FINAL][common].mean(axis=1)), ("IDW p=2", predictions[common])]]
    write_csv(out / "interpolation_scores.csv", rows)
    write_csv(out / "interpolation_station_days.csv", [
        {"date": str(data["date"][i]), "station_id": data["station"][i], "period": data["period"][i],
         "truth_mm": data["truth"][i], "idw_mm": predictions[i],
         "background_mean_mm": data["members"]["background"][i].mean(),
         "analysis_mean_mm": data["members"][FINAL][i].mean()} for i in np.flatnonzero(common)])
    build_idw_comparisons(read_csv(out / "interpolation_station_days.csv"), out,
                          seed, resamples, block_days)
    table(out / "tab_interpolation.tex", ["Method", "n", "RMSE", "MAE", "Bias", "r"],
          [[r["method"], r["n"], f'{r["rmse"]:.3f}', f'{r["mae"]:.3f}', f'{r["bias"]:.3f}',
            "--" if r["correlation"] is None else f'{r["correlation"]:.3f}'] for r in rows],
          f"Fixed p=2 IDW uses retained original daily reports only. Common sample: {int(common.sum()):,}; excluded for unavailable IDW: {int(keep.sum() - common.sum()):,}.")


def build_idw_comparisons(rows, out, seed=20261001, resamples=10000, block_days=(3, 7)):
    """Paired mean-versus-IDW evidence from the identical independent sample."""
    if not rows or any(not all(k in row for k in ("date", "station_id", "period",
                          "truth_mm", "idw_mm", "analysis_mean_mm")) for row in rows):
        raise ValueError("IDW daily export needs analysis_mean_mm; rerun script 92 from original arrays")
    keys = [(r["date"], r["station_id"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate IDW station-days")
    dates = np.asarray([r["date"] for r in rows], dtype="datetime64[D]")
    if np.any((dates >= np.datetime64("2022-05-01")) & (dates <= np.datetime64("2022-05-31"))):
        raise ValueError("IDW comparisons must exclude the May 2022 selection month")
    truth = np.asarray([float(r["truth_mm"]) for r in rows])
    predictions = {FINAL: np.asarray([float(r["analysis_mean_mm"]) for r in rows]),
                   "IDW p=2": np.asarray([float(r["idw_mm"]) for r in rows])}
    if not np.isfinite(truth).all() or not all(np.isfinite(p).all() for p in predictions.values()):
        raise ValueError("IDW export must contain common finite samples only")
    paired = paired_intervals({"date": dates, "truth": truth}, predictions,
                              block_days, resamples, seed)
    write_csv(out / "paired_idw_intervals.csv", paired)
    scores = []
    for low, high, label in ((0, 1, "[0,1)"), (1, 10, "[1,10)"), (10, 25, "[10,25)"),
                             (25, 50, "[25,50)"), (50, 100, "[50,100)"), (100, np.inf, "[100,inf)")):
        keep = (truth >= low) & (truth < high)
        if not keep.any():
            continue
        model, baseline = [deterministic(predictions[m][keep], truth[keep]) for m in (FINAL, "IDW p=2")]
        for method, score in ((FINAL, model), ("IDW p=2", baseline)):
            scores.append({"intensity": label, "method": method, **score,
                           "rmse_gain_mm_day": baseline["rmse"] - model["rmse"],
                           "mae_gain_mm_day": baseline["mae"] - model["mae"]})
    write_csv(out / "idw_intensity_scores.csv", scores)
    table(out / "tab_idw_paired.tex", ["Metric", "Block", "Gain", "95% low", "95% high"],
          [[r["metric"], r["block_days"], *[f'{r[k]:.3f}' for k in ("gain_mm_day", "ci_low", "ci_high")]] for r in paired],
          "Positive gain favours SURMA-Flow over retained-gauge IDW p=2. Identical original withheld station-days; whole-day blocks stay within date-gap segments. Pooled RMSE is recomputed per resample. Temporal uncertainty conditional on this holdout; pointwise intervals.")
    table(out / "tab_idw_intensity.tex", ["Rain bin", "n", "RMSE IDW", "RMSE model", "RMSE gain", "MAE gain"],
          [[r["intensity"], r["n"], f'{r["rmse"] + r["rmse_gain_mm_day"]:.3f}',
            f'{r["rmse"]:.3f}', f'{r["rmse_gain_mm_day"]:.3f}', f'{r["mae_gain_mm_day"]:.3f}']
           for r in scores if r["method"] == FINAL],
          "Bins use observed rainfall (mm/day). Positive gain favours SURMA-Flow. Each bin uses the same IDW/model station-days; rare-bin counts are reported. These are point estimates, not simultaneous intensity-bin confidence intervals.")


def build_selection(path, contract, out, scorer, record, country=None):
    specification = json.loads(path.read_text())
    profiles = specification["profiles"]
    if len(profiles) < 2:
        raise ValueError("selection comparison requires at least two recorded profiles")
    reference, arrays, rows = None, [], []
    labels = [p["label"] for p in profiles]
    if len(set(labels)) != len(labels):
        raise ValueError("selection labels must be unique")
    for item in profiles:
        prefix = Path(item["prefix"])
        dump_path, report_path = Path(str(prefix) + ".npz"), Path(str(prefix) + ".json")
        record(dump_path); record(report_path)
        report = json.loads(report_path.read_text())
        scope = report["scope"]
        if scope.get("assimilate_all_stations") is not False:
            raise ValueError("selection scores require withheld stations")
        if tuple(Path(scope["checkpoint"]).parts[-2:]) != ("prior_h100_cpc_v2", "best.pt"):
            raise ValueError("selection comparison must use CPCv2")
        with np.load(dump_path, allow_pickle=False) as dump:
            dates = dump["times"].astype("datetime64[D]")
            profile = contract["profiles"]["superob-final"]
            if not len(dates) or np.any(dates < np.datetime64(profile["selection_start"])) or np.any(dates > np.datetime64(profile["selection_end"])) or len(set(dates)) != len(dates):
                raise ValueError("selection comparison must stay within excluded May 2022")
            ids = dump["station_ids"].astype(str)
            indices = dump["eval_idx"]
            if indices.dtype.kind not in "iu" or indices.ndim != 1 or not len(indices) or len(set(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= len(ids)) or len(set(ids)) != len(ids):
                raise ValueError("invalid selection station identifiers or withheld indices")
            if set(indices) & set(dump["assim_idx"]) or any(s.startswith("SOB_") for s in ids[indices]):
                raise ValueError("selection holdout leaks into assimilation or uses synthetic gauges")
            if set(scope.get("withheld_station_ids", [])) != set(ids[indices]) or scope.get("seed") is None or scope.get("checkpoint_stats") is None:
                raise ValueError("selection metadata must identify withheld gauges, statistics and seed")
            if country is not None:
                indices = indices[points_inside(dump["station_lat"][indices], dump["station_lon"][indices], country)]
                if not len(indices):
                    raise ValueError("no selection gauges inside Bangladesh")
            order = indices[np.argsort(ids[indices])]
            day_order = np.argsort(dates)
            truth = dump["gauge_mm"][day_order][:, order].reshape(-1)
            members = np.moveaxis(dump[f'station_{item["method"]}'][day_order][:, :, order], 1, 2).reshape(len(truth), -1)
            signature = (dates[day_order].astype(str).tolist(), ids[order].tolist(),
                         scope.get("checkpoint_stats"), scope.get("seed"), members.shape[1], scope.get("background_day_offset"),
                         scope.get("checkpoint_data"), scope.get("precip_transform"),
                         scope.get("analysis_sampler_n_steps"), scope.get("analysis_sampler_n_corrections"), scope.get("analysis_sampler_heun"))
            if members.shape[1] != contract["members"]:
                raise ValueError("selection member count differs from the frozen model contract")
        if reference is None:
            reference = (signature, truth)
        elif signature != reference[0] or not np.array_equal(truth, reference[1], equal_nan=True):
            raise ValueError("selection profiles do not share dates, original gauges, truth, statistics, seeds and ensemble size")
        arrays.append(members)
    common = np.isfinite(reference[1])
    for array in arrays:
        common &= np.isfinite(array).all(axis=1)
    if not common.any():
        raise ValueError("empty common selection sample")
    for item, array in zip(profiles, arrays):
        scores = scorer.metrics(array[common], reference[1][common])
        rows.append({"profile": item["label"], "method": item["method"],
                     "start": min(reference[0][0]), "end": max(reference[0][0]), "days": len(reference[0][0]),
                     "n": int(common.sum()), "crps": scores["crps"], "rmse": scores["rmse"], "coverage_90": scores["coverage_90"]})
    write_csv(out / "selection_scores.csv", rows)
    table(out / "tab_selection.tex", ["Profile", "Days", "n", "CRPS", "RMSE", "Coverage"],
          [[r["profile"], r["days"], r["n"], f'{r["crps"]:.3f}', f'{r["rmse"]:.3f}', f'{r["coverage_90"]:.3f}'] for r in rows],
          f"Selection-only sample: {rows[0]['start']} through {rows[0]['end']}. These scores are excluded from confirmatory verification. Profiles changing several settings are not single-factor causal ablations.")


def build_training(path, out):
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [{"epoch": int(r["epoch"]) + 1, "step": int(r["step"]), "validation_crps_mm": float(r["mean_crps_mm"]),
             "cases": len(r["cases"]), "members": int(r["members"])} for r in records]
    if not rows or not all(np.isfinite(r["validation_crps_mm"]) and r["cases"] > 0 and r["members"] > 1 for r in rows):
        raise ValueError("invalid CPCv2 monitor history")
    if len({r["epoch"] for r in rows}) != len(rows):
        raise ValueError("duplicate history epochs: resolve resumed or mixed runs first")
    cases = [[(c["date"], c.get("quantile")) for c in r["cases"]] for r in records]
    if any(c != cases[0] for c in cases) or len({r["members"] for r in rows}) != 1:
        raise ValueError("validation cases or ensemble size change across history")
    if any(not (2019 <= int(c[0][:4]) <= 2020) for c in cases[0]):
        raise ValueError("CPCv2 validation history must use 2019–2020 dates")
    rows.sort(key=lambda r: r["epoch"])
    write_csv(out / "training_curve.csv", rows)
    plt = plotting()
    fig, ax = plt.subplots(figsize=(7, 3.5), layout="constrained")
    ax.plot([r["epoch"] for r in rows], [r["validation_crps_mm"] for r in rows], "o-", color="#007f78")
    ax.set(xlabel="Completed training epoch", ylabel="Validation fair CRPS (mm/day)")
    ax.grid(alpha=.2)
    fig.savefig(out / "fig_training.pdf", bbox_inches="tight")
    plt.close(fig)


def build_compute(path, contract, out):
    payload = json.loads(path.read_text())
    if payload["checkpoint_sha256"] != "a04a3d9ae9109f905e06c32bfd55252daf1229d17c98b404e265064b89f210ea":
        raise ValueError("compute record does not identify the evaluated checkpoint")
    rows = payload["measurements"]
    if set(r["stage"] for r in rows) != {"training", "background", "analysis"} or len(rows) != 3:
        raise ValueError("provide one measurement each for training, background and analysis")
    output = []
    for r in rows:
        if not r["record_source"] or not r["hardware"] or int(r["gpus"]) < 1 or not np.isfinite(float(r["wall_seconds"])) or float(r["wall_seconds"]) <= 0:
            raise ValueError("compute measurements need real source, hardware, GPU count and positive elapsed time")
        if r["stage"] != "training" and (int(r["days"]) < 1 or int(r["members"]) != contract["members"] or int(r["steps"]) != 50 or int(r["correctors"]) != (2 if r["stage"] == "analysis" else 0) or r["grid"] != [128, 128]):
            raise ValueError("sampling benchmark differs from the evaluated contract")
        output.append({"stage": r["stage"], "hardware": r["hardware"], "gpus": int(r["gpus"]),
                       "wall_hours": float(r["wall_seconds"]) / 3600,
                       "gpu_hours": float(r["wall_seconds"]) * int(r["gpus"]) / 3600,
                       "seconds_per_ensemble_day": float(r["wall_seconds"]) / int(r["days"]) if r["stage"] != "training" else None,
                       "record_source": r["record_source"]})
    write_csv(out / "compute_summary.csv", output)
    table(out / "tab_compute.tex", ["Stage", "Hardware", "GPUs", "Wall h", "GPU h", "s/day"],
          [[r["stage"], r["hardware"], r["gpus"], f'{r["wall_hours"]:.2f}', f'{r["gpu_hours"]:.2f}',
            "--" if r["seconds_per_ensemble_day"] is None else f'{r["seconds_per_ensemble_day"]:.1f}'] for r in output],
          "Sampling time is for one 30-member, 128 × 128 daily ensemble, including guidance. Training duration must come from the complete training-job record.")


def build_robustness(paths, out, record, boundary_sha256=None):
    rows = []
    for path in paths:
        report_path, scores_path = path / "paper1_evaluation.json", path / "daily_withheld_scores.csv"
        record(report_path); record(scores_path)
        report = json.loads(report_path.read_text())
        if report["status"] != "withheld_evaluation_complete":
            raise ValueError("robustness input must be a completed audited evaluation")
        if boundary_sha256 is not None and report.get("evaluation_region", {}).get("sha256") != boundary_sha256:
            raise ValueError("robustness summaries must use the same Bangladesh boundary")
        profile = report["contract"]["profiles"][report["profile"]]
        scores = [r for r in read_csv(scores_path) if r["group"] == "pooled" and r["label"] == "all"
                  and r["method"] in ("background", profile["method"])]
        if len(scores) != 2:
            raise ValueError("robustness input missing background or fixed analysis")
        for r in scores:
            rows.append({"experiment": report["profile"], "layout": profile["layout"], "folds": profile["folds"],
                         "method": r["method"], "n": int(r["n"]), "crps": float(r["crps"]),
                         "rmse": float(r["rmse"]), "coverage_90": float(r["coverage_90"]),
                         "selection_start": profile["selection_start"], "selection_end": profile["selection_end"]})
    if len({(r["experiment"], r["layout"]) for r in rows}) < 2:
        raise ValueError("robustness summary requires distinct experiments")
    write_csv(out / "robustness_scores.csv", rows)
    table(out / "tab_robustness.tex", ["Experiment / method", "Folds", "n", "CRPS", "RMSE", "Coverage"],
          [[r["experiment"] + (" / bg." if r["method"] == "background" else " / analysis"), r["folds"], r["n"],
            f'{r["crps"]:.3f}', f'{r["rmse"]:.3f}', f'{r["coverage_90"]:.3f}'] for r in rows],
          "Different networks, likelihoods and selection exclusions are reported separately; these rows are not a paired causal comparison or pooled robustness estimate.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=ROOT / "configs/paper1_cpcv2_final.json")
    parser.add_argument("--root", type=Path, help="original archive root, not the copied score-export directory")
    parser.add_argument("--output", type=Path, default=ROOT / "manuscript/additional")
    parser.add_argument("--selection", type=Path, help="JSON list of selection profile prefixes and method keys")
    parser.add_argument("--history", type=Path, help="CPCv2 validation monitor/history.jsonl")
    parser.add_argument("--compute", type=Path, help="measured timing JSON, schema in completion documentation")
    parser.add_argument("--robustness", type=Path, nargs="+", help="completed script-90 evaluation directories, at least two")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--require-complete", action="store_true", help="exit 2 when any slot remains pending")
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--block-days", type=int, nargs="+", default=[3, 7])
    parser.add_argument("--boundary-geojson", type=Path, default=DEFAULT_BOUNDARY)
    args = parser.parse_args(argv)
    if args.bootstrap < 1 or any(n < 1 for n in args.block_days):
        parser.error("bootstrap and block widths must be positive")
    contract = json.loads(args.contract.read_text())
    profile = contract["profiles"]["superob-final"]
    periods = list(contract["periods"])
    root = args.root or ROOT / profile["root"]
    required = module90().inventory(root, periods, profile)["required"]
    raw = [root / "stations" / p / "combined_daily.csv" for p in periods]
    archive_ready = all(r["present"] for r in required)
    paths = {"network": [Path(r["path"]) for r in required] + raw,
             "country_intervals": [Path(r["path"]) for r in required],
             "interpolation": [Path(r["path"]) for r in required] + raw,
             "calibration": [Path(r["path"]) for r in required],
             "selection": [args.selection] if args.selection else [],
             "training": [args.history] if args.history else [],
             "compute": [args.compute] if args.compute else [],
             "robustness": [p / "paper1_evaluation.json" for p in args.robustness or []]
                           + [p / "daily_withheld_scores.csv" for p in args.robustness or []]}
    status = {name: {"status": "ready" if inputs and all(p.is_file() for p in inputs) else "pending",
                     "missing": [str(p) for p in inputs if not p.is_file()],
                     "needs": "See docs/PAPER1_COMPLETION.md"} for name, inputs in paths.items()}
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"audit_only": args.audit_only, "archive_root": str(root), "slots": status, "inputs": [],
                "selection_exclusion": [profile["selection_start"], profile["selection_end"]]}

    def record(path):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        manifest["inputs"].append({"path": str(path.resolve()), "sha256": digest.hexdigest()})

    record(args.contract)
    country, region = read_boundary(args.boundary_geojson)
    record(args.boundary_geojson)
    manifest["evaluation_region"] = region
    destination = args.output
    temporary = None
    if not args.audit_only:
        # Only this script's managed files are removed, preventing stale fills
        # when a later run has missing data. Authored prose is never changed.
        for files in SLOTS.values():
            for filename in files:
                (args.output / filename).unlink(missing_ok=True)
        # All fills are staged until every available input validates. A bad
        # optional record cannot leave a partially validated manuscript fill.
        temporary = tempfile.TemporaryDirectory(prefix="paper1-completion-", dir=args.output.parent)
        args.output = Path(temporary.name)
        if archive_ready:
            mod = module90()
            data, _, _, warnings = mod.load_samples(root, periods, contract, profile, ["background", FINAL])
            data, region = mod.country_samples(data, args.boundary_geojson)
            manifest["evaluation_region"] = region
            keep = independent_mask(data, profile)
            manifest["warnings"] = warnings
            manifest["scored_station_days"] = int(keep.sum())
            for item in required:
                record(Path(item["path"]))
            build_calibration(data, keep, args.output, args.seed)
            status["calibration"]["status"] = "generated"
            _, paired, _, _ = mod.score_samples(data, ["background", FINAL], profile,
                                               mod.scoring_module(), args.block_days, args.bootstrap, args.seed)
            write_csv(args.output / "paired_country_crps.csv", paired)
            table(args.output / "tab_country_intervals.tex", ["Block days", "Gain", "95% lower", "95% upper"],
                  [[r["block_days"], f'{r["difference"]:.3f}', f'{r["ci_low"]:.3f}', f'{r["ci_high"]:.3f}'] for r in paired],
                  f"Bangladesh-only equal-day fair CRPS differences; {args.bootstrap:,} resamples, no blocks cross gaps or the excluded selection month. Units: mm/day.")
            status["country_intervals"]["status"] = "generated"
            if all(p.is_file() for p in raw):
                for p in raw:
                    record(p)
                build_network_idw(data, keep, root, periods, args.output, country,
                                  args.seed, args.bootstrap, args.block_days)
                status["network"]["status"] = status["interpolation"]["status"] = "generated"
        for slot, path, build in (("training", args.history, build_training), ("compute", args.compute, None),
                                  ("selection", args.selection, None)):
            if status[slot]["status"] != "ready":
                continue
            record(path)
            if slot == "compute":
                build_compute(path, contract, args.output)
            elif slot == "selection":
                build_selection(path, contract, args.output, module90().scoring_module(), record, country)
            else:
                build(path, args.output)
            status[slot]["status"] = "generated"
        if status["robustness"]["status"] == "ready":
            build_robustness(args.robustness, args.output, record, region["sha256"])
            status["robustness"]["status"] = "generated"
    manifest["outputs"] = []
    for name, files in SLOTS.items():
        if status[name]["status"] == "generated":
            for filename in files:
                path = args.output / filename
                manifest["outputs"].append({"path": filename, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    if temporary is not None:
        for output in manifest["outputs"]:
            (args.output / output["path"]).replace(destination / output["path"])
        temporary.cleanup()
    (destination / "completion_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for slot, item in status.items():
        print(f"[paper1-completion] {slot}: {item['status']}")
    return 2 if args.require_complete and any(r["status"] not in ("generated", "ready") for r in status.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
