#!/usr/bin/env python3
"""Evaluate real-observation BMD DA using withheld gauges as the reference.

This script intentionally separates three questions:

1. Primary method selection uses only BMD stations withheld from assimilation.
2. Ensemble calibration and rainfall-event skill are verified at those stations.
3. CHIRPS, IMERG and CPC maps are shown only as non-independent spatial
   intercomparisons; none is labelled as truth.

All calculations use an already completed NPZ, so plots can be regenerated on
a CPU node without repeating the expensive generative assimilation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


THRESHOLDS_MM = (1.0, 10.0, 25.0, 50.0)
METHOD_COLOURS = {
    "Background": "#7D8597",
    "Gauges only": "#0077B6",
    "IMERG only": "#F4A261",
    "Simultaneous": "#D1495B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", default="data/processed/bmd_may2018_example.npz")
    parser.add_argument("--report", default="data/processed/bmd_may2018_example.json")
    parser.add_argument(
        "--out-diagnostics", default="data/processed/bmd_may2018_diagnostics.png"
    )
    parser.add_argument(
        "--out-events", default="data/processed/bmd_may2018_events.png"
    )
    parser.add_argument(
        "--out-station-comparison",
        default="data/processed/bmd_may2018_station_comparison.png",
    )
    parser.add_argument("--out-spatial", default="data/processed/bmd_may2018_spatial.png")
    parser.add_argument(
        "--out-intercomparison",
        "--out-chirps-spatial",
        dest="out_intercomparison",
        default="data/processed/bmd_may2018_intercomparison.png",
    )
    parser.add_argument(
        "--out-evaluation", default="data/processed/bmd_may2018_evaluation.json"
    )
    return parser.parse_args()


def open_dump(path: str):
    try:
        data = np.load(path, allow_pickle=False)
        _ = data["station_name"]
        return data
    except ValueError as exc:
        if "Object arrays cannot be loaded" not in str(exc):
            raise
        data.close()
        print("WARNING: loading legacy locally-generated object-string labels")
        return np.load(path, allow_pickle=True)


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    return value


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def crps_values(ensemble: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """CRPS at every observation for an ensemble shaped ``(member, ...)``."""

    ensemble = np.asarray(ensemble, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    if ensemble.shape[1:] != observed.shape:
        raise ValueError(f"ensemble {ensemble.shape} does not match truth {observed.shape}")
    valid = np.isfinite(observed) & np.all(np.isfinite(ensemble), axis=0)
    output = np.full(observed.shape, np.nan, dtype=np.float64)
    if not valid.any():
        return output
    members = ensemble[:, valid]
    truth = observed[valid]
    member_count = members.shape[0]
    first = np.mean(np.abs(members - truth[None]), axis=0)
    # Match bdhires.eval.crps_ensemble exactly: the fair/unbiased finite-
    # ensemble estimator divides the pair term by R(R-1), not R^2.
    second = np.sum(
        np.abs(members[:, None, :] - members[None, :, :]), axis=(0, 1)
    ) / (2.0 * member_count * (member_count - 1))
    output[valid] = first - second
    return output


def reliability(probability: np.ndarray, event: np.ndarray) -> list[dict]:
    bins = np.linspace(0.0, 1.0, 6)
    output = []
    for index, (lower, upper) in enumerate(zip(bins[:-1], bins[1:])):
        selected = (probability >= lower) & (
            (probability <= upper) if index == len(bins) - 2 else (probability < upper)
        )
        output.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "n": int(selected.sum()),
                "forecast_probability": (
                    float(np.mean(probability[selected])) if selected.any() else float("nan")
                ),
                "observed_frequency": (
                    float(np.mean(event[selected])) if selected.any() else float("nan")
                ),
            }
        )
    return output


def threshold_metrics(
    ensemble: np.ndarray, observed: np.ndarray, threshold: float
) -> dict:
    valid = np.isfinite(observed) & np.all(np.isfinite(ensemble), axis=0)
    if not valid.any():
        return {"n": 0}
    members = ensemble[:, valid]
    truth = observed[valid]
    probability = np.mean(members >= threshold, axis=0)
    event = truth >= threshold
    forecast_event = probability >= 0.5
    hits = int(np.sum(forecast_event & event))
    misses = int(np.sum(~forecast_event & event))
    false_alarms = int(np.sum(forecast_event & ~event))
    correct_negatives = int(np.sum(~forecast_event & ~event))
    return {
        "n": int(valid.sum()),
        "observed_events": int(event.sum()),
        "brier_score": float(np.mean((probability - event.astype(float)) ** 2)),
        "probability_mean": float(np.mean(probability)),
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "correct_negatives": correct_negatives,
        "pod": float(hits / (hits + misses)) if hits + misses else float("nan"),
        "far": (
            float(false_alarms / (hits + false_alarms))
            if hits + false_alarms
            else float("nan")
        ),
        "csi": (
            float(hits / (hits + misses + false_alarms))
            if hits + misses + false_alarms
            else float("nan")
        ),
        "reliability": reliability(probability, event),
    }


def ensemble_metrics(ensemble: np.ndarray, observed: np.ndarray) -> dict:
    valid = np.isfinite(observed) & np.all(np.isfinite(ensemble), axis=0)
    if not valid.any():
        return {"n": 0}
    members = ensemble[:, valid].astype(np.float64)
    truth = observed[valid].astype(np.float64)
    mean = np.mean(members, axis=0)
    difference = mean - truth
    rmse = float(np.sqrt(np.mean(difference**2)))
    spread = float(np.sqrt(np.mean(np.var(members, axis=0, ddof=1))) )
    coverage = {}
    interval_width = {}
    for level in (0.50, 0.80, 0.90):
        tail = (1.0 - level) / 2.0
        low, high = np.quantile(members, [tail, 1.0 - tail], axis=0)
        key = str(int(round(level * 100)))
        coverage[key] = float(np.mean((truth >= low) & (truth <= high)))
        interval_width[key] = float(np.mean(high - low))
    return {
        "n": int(valid.sum()),
        "rmse_mm": rmse,
        "mae_mm": float(np.mean(np.abs(difference))),
        "bias_mm": float(np.mean(difference)),
        "crps_mm": float(np.nanmean(crps_values(ensemble, observed))),
        "correlation": safe_correlation(mean, truth),
        "spread_mm": spread,
        "spread_skill": float(spread / rmse) if rmse else float("nan"),
        "coverage": coverage,
        "coverage_90": coverage["90"],
        "interval_width_mm": interval_width,
        "thresholds": {
            f"{threshold:g}": threshold_metrics(ensemble, observed, threshold)
            for threshold in THRESHOLDS_MM
        },
    }


def deterministic_metrics(predicted: np.ndarray, observed: np.ndarray) -> dict:
    valid = np.isfinite(predicted) & np.isfinite(observed)
    if not valid.any():
        return {"n": 0}
    predicted = np.asarray(predicted)[valid].astype(np.float64)
    truth = np.asarray(observed)[valid].astype(np.float64)
    difference = predicted - truth
    return {
        "n": int(valid.sum()),
        "rmse_mm": float(np.sqrt(np.mean(difference**2))),
        "mae_mm": float(np.mean(np.abs(difference))),
        "bias_mm": float(np.mean(difference)),
        "correlation": safe_correlation(predicted, truth),
    }


def idw_predictions(
    gauge: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    assim_idx: np.ndarray,
    eval_idx: np.ndarray,
    power: float = 2.0,
) -> np.ndarray:
    """Cross-validated inverse-distance gauge baseline at withheld stations."""

    prediction = np.full((len(gauge), len(eval_idx)), np.nan, dtype=np.float64)
    mean_latitude = np.deg2rad(float(np.mean(lat)))
    for target_position, target in enumerate(eval_idx):
        dx = (lon[assim_idx] - lon[target]) * np.cos(mean_latitude)
        dy = lat[assim_idx] - lat[target]
        distance = np.maximum(np.sqrt(dx**2 + dy**2), 1e-6)
        base_weight = distance ** (-power)
        for day in range(len(gauge)):
            available = np.isfinite(gauge[day, assim_idx])
            if available.any():
                weight = base_weight[available]
                prediction[day, target_position] = np.sum(
                    weight * gauge[day, assim_idx[available]]
                ) / np.sum(weight)
    return prediction


def rank_histogram(ensemble: np.ndarray, observed: np.ndarray) -> np.ndarray:
    valid = np.isfinite(observed) & np.all(np.isfinite(ensemble), axis=0)
    ranks = (ensemble[:, valid] < observed[valid][None]).sum(axis=0)
    return np.bincount(ranks, minlength=ensemble.shape[0] + 1)


def add_station_markers(axis, lon, lat, assim_idx, eval_idx, labels=False):
    axis.scatter(
        lon[assim_idx], lat[assim_idx], s=24, c="black", edgecolors="white",
        linewidths=0.35, label="assimilated", zorder=5,
    )
    axis.scatter(
        lon[eval_idx], lat[eval_idx], s=52, facecolors="none", edgecolors="#00D5FF",
        linewidths=1.5, label="withheld", zorder=6,
    )
    if labels:
        for index in eval_idx:
            axis.annotate(
                str(index + 1), (lon[index], lat[index]), xytext=(3, 3),
                textcoords="offset points", fontsize=6, color="#007C91",
            )


def add_rain_markers(axis, lon, lat, gauge_day, assim_idx, eval_idx, maximum):
    common = dict(c=gauge_day, cmap="viridis", vmin=0, vmax=maximum, zorder=6)
    axis.scatter(
        lon[assim_idx], lat[assim_idx], s=18, marker="o", edgecolors="black",
        linewidths=0.4, **{key: value[assim_idx] if key == "c" else value for key, value in common.items()},
    )
    axis.scatter(
        lon[eval_idx], lat[eval_idx], s=38, marker="o", edgecolors="#00D5FF",
        linewidths=1.0, **{key: value[eval_idx] if key == "c" else value for key, value in common.items()},
    )


def matrix_plot(axis, values, row_labels, column_labels, title, fmt=".2f"):
    values = np.asarray(values, dtype=float)
    colour = np.empty_like(values)
    for column in range(values.shape[1]):
        finite = values[np.isfinite(values[:, column]), column]
        if not len(finite) or np.max(finite) == np.min(finite):
            colour[:, column] = 0.5
        else:
            colour[:, column] = (
                values[:, column] - np.min(finite)
            ) / (np.max(finite) - np.min(finite))
    axis.imshow(colour, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(np.arange(len(column_labels)))
    axis.set_xticklabels(column_labels, rotation=35, ha="right")
    axis.set_yticks(np.arange(len(row_labels)))
    axis.set_yticklabels(row_labels)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            label = format(value, fmt) if np.isfinite(value) else "—"
            axis.text(column, row, label, ha="center", va="center", fontsize=8)
    axis.set_title(title)


def block_smooth(field: np.ndarray, factor: int = 4) -> np.ndarray:
    height, width = field.shape[-2:]
    if height % factor or width % factor:
        return field
    coarse = field.reshape(
        *field.shape[:-2], height // factor, factor, width // factor, factor
    ).mean(axis=(-3, -1))
    return np.repeat(np.repeat(coarse, factor, axis=-2), factor, axis=-1)


def main() -> None:
    args = parse_args()
    data = open_dump(args.dump)
    report = json.loads(Path(args.report).read_text())
    time = data["time"].astype("datetime64[ns]").astype("datetime64[D]")
    names = data["station_name"].astype(str)
    lat, lon = data["station_lat"], data["station_lon"]
    assim_idx, eval_idx = data["assim_idx"], data["eval_idx"]
    gauge = np.asarray(data["gauge_mm"], dtype=np.float64)
    observed = gauge[:, eval_idx]
    has_imerg = bool(report.get("scope", {}).get("imerg_assimilated", False))
    background_day_offset = int(
        report.get("scope", {}).get("background_day_offset", 0)
    )
    background_dates = report.get("scope", {}).get("background_dates", [])
    imerg_error = report.get("observation_error", {}).get("imerg") or {}
    network = report.get("network", {})
    control_text = (
        "IMERG window: previous-day 03:00 to selected-day 03:00 UTC; "
        f"checkpoint background D{background_day_offset:+d}; "
        f"stride {imerg_error.get('footprint_stride', '?')}; "
        f"total IMERG R inflation {imerg_error.get('correlation_variance_inflation', np.nan):.1f}×"
        if has_imerg and imerg_error
        else "gauge-only configuration"
    )

    def station_ensemble(key, fallback="analysis_at_stations"):
        values = data[key] if key in data.files else data[fallback]
        return np.moveaxis(np.asarray(values)[:, :, eval_idx], 1, 0)

    methods = {
        "Background": station_ensemble("background_at_stations"),
        "Gauges only": station_ensemble("gauge_analysis_at_stations"),
    }
    if has_imerg:
        methods.update(
            {
                "IMERG only": station_ensemble("imerg_analysis_at_stations"),
                "Simultaneous": station_ensemble("combined_analysis_at_stations"),
            }
        )
    method_scores = {
        name: ensemble_metrics(ensemble, observed) for name, ensemble in methods.items()
    }
    background_crps = method_scores["Background"]["crps_mm"]
    for score in method_scores.values():
        score["crps_skill_vs_background"] = (
            float(1.0 - score["crps_mm"] / background_crps)
            if background_crps > 0
            else float("nan")
        )

    idw = idw_predictions(gauge, lat, lon, assim_idx, eval_idx)
    deterministic = {
        "IDW gauges": deterministic_metrics(idw, observed),
        "CPC condition": deterministic_metrics(
            np.asarray(data["condition_at_stations"])[:, eval_idx], observed
        ),
        "CHIRPS context": deterministic_metrics(
            np.asarray(data["chirps_at_stations"])[:, eval_idx], observed
        ),
    }
    if has_imerg:
        deterministic["Raw IMERG"] = deterministic_metrics(
            np.asarray(data["imerg_at_stations"])[:, eval_idx], observed
        )

    # Put raw gridded products, a simple station interpolation, the model
    # background, and every DA arm onto the identical withheld station-days.
    # This is the direct apples-to-apples deterministic product comparison.
    station_predictions = {
        "CPC": np.asarray(data["condition_at_stations"])[:, eval_idx],
        "Raw IMERG": np.asarray(data["imerg_at_stations"])[:, eval_idx],
        "CHIRPS": np.asarray(data["chirps_at_stations"])[:, eval_idx],
        "IDW gauges": idw,
        **{name: np.mean(ensemble, axis=0) for name, ensemble in methods.items()},
    }
    if not has_imerg:
        station_predictions.pop("Raw IMERG")
    station_comparison_scores = {
        name: deterministic_metrics(prediction, observed)
        for name, prediction in station_predictions.items()
    }

    daily_crps = {
        name: np.nanmean(crps_values(ensemble, observed), axis=1)
        for name, ensemble in methods.items()
    }
    station_crps = {
        name: np.nanmean(crps_values(ensemble, observed), axis=0)
        for name, ensemble in methods.items()
    }
    winner = min(method_scores, key=lambda name: method_scores[name]["crps_mm"])
    fused = [name for name in ("Simultaneous",) if name in methods]
    best_fused = min(fused, key=lambda name: method_scores[name]["crps_mm"]) if fused else None
    gauge_crps = method_scores["Gauges only"]["crps_mm"]
    evaluation = {
        "scope": {
            "dump": args.dump,
            "source_report": args.report,
            "dates": [str(value) for value in time],
            "background_day_offset": background_day_offset,
            "background_dates": background_dates,
            "holdout_fold": network.get("holdout_fold", 0),
            "holdout_folds": network.get("holdout_folds", 1),
            "withheld_station_ids": network.get("withheld_ids", []),
            "withheld_station_names": network.get("withheld_names", []),
            "station_days": int(np.isfinite(observed).sum()),
            "assimilated_stations": int(len(assim_idx)),
            "withheld_stations": int(len(eval_idx)),
            "primary_reference": "BMD stations withheld from every assimilation arm",
            "imerg_total_r_inflation": imerg_error.get("correlation_variance_inflation"),
            "imerg_extra_r_multiplier": imerg_error.get("extra_r_multiplier"),
        },
        "probabilistic_methods": method_scores,
        "deterministic_context": deterministic,
        "station_collocated_comparison": {
            "reference": "the identical BMD station-days withheld from DA",
            "scores": station_comparison_scores,
            "note": (
                "DA rows use their ensemble mean here; use probabilistic_methods "
                "for CRPS and ensemble calibration."
            ),
        },
        "daily_crps_mm": {name: value.tolist() for name, value in daily_crps.items()},
        "withheld_station_crps_mm": {
            name: {
                station_name: float(value[index])
                for index, station_name in enumerate(names[eval_idx])
            }
            for name, value in station_crps.items()
        },
        "selection": {
            "lowest_withheld_bmd_crps": winner,
            "best_fused_method": best_fused,
            "best_fused_beats_gauge_only": (
                bool(method_scores[best_fused]["crps_mm"] < gauge_crps)
                if best_fused is not None
                else None
            ),
            "decision_rule": (
                "A fused DA method must beat gauges-only CRPS at withheld BMD stations, "
                "retain calibrated intervals, and avoid event degradation."
            ),
        },
        "caveats": [
            (
                f"{len(time)} days and one spatial holdout are a process gate, not a "
                "final skill estimate; final selection uses rotated spatial folds."
            ),
            (
                f"{str(time[0])} to {str(time[-1])} lies inside the checkpoint training "
                "period and is not independent temporal validation."
                if bool(report.get("scope", {}).get("in_checkpoint_training_period", False))
                else f"{str(time[0])} to {str(time[-1])} is outside the checkpoint training "
                "period; independence still requires all tuning to be frozen beforehand."
            ),
            "IMERG Final has monthly GPCC gauge adjustment; BMD/GPCC overlap must be audited.",
            "IMERG uses exact BMD 03:00-03:00 UTC accumulation windows.",
            (
                "The complete checkpoint item is offset together: CPC, ERA5 state "
                "means, residual base and seasonal encoding. Observation-date CHIRPS "
                "is unchanged and remains contextual only."
            ),
            "The retired sequential IMERG-to-gauges arm is excluded from selection.",
            "CPC and CHIRPS are calendar-labelled products shown only as contextual intercomparisons.",
        ],
    }
    Path(args.out_evaluation).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_evaluation).write_text(
        json.dumps(json_safe(evaluation), indent=2, allow_nan=False) + "\n"
    )

    # ---------------- primary withheld-BMD scorecard ----------------
    method_names = list(methods)
    figure, axes = plt.subplots(3, 3, figsize=(18, 15), constrained_layout=True)
    axis = axes[0, 0]
    add_station_markers(axis, lon, lat, assim_idx, eval_idx, labels=True)
    axis.set_xlabel("Longitude (°E)")
    axis.set_ylabel("Latitude (°N)")
    axis.set_title("A. BMD spatial holdout")
    axis.legend(loc="lower left", fontsize=8)
    axis.grid(alpha=0.2)

    keys = ["crps_mm", "rmse_mm", "mae_mm", "bias_mm", "correlation", "spread_skill", "coverage_90"]
    columns = ["CRPS", "RMSE", "MAE", "Bias", "Corr", "Spread/skill", "Cover90"]
    matrix_plot(
        axes[0, 1],
        [[method_scores[name].get(key, np.nan) for key in keys] for name in method_names],
        method_names,
        columns,
        "B. Primary withheld-BMD metrics\n(shading is neutral; read values)",
    )

    axis = axes[0, 2]
    skill = [method_scores[name]["crps_skill_vs_background"] for name in method_names]
    axis.barh(method_names, skill, color=[METHOD_COLOURS[name] for name in method_names])
    axis.axvline(0, color="black", lw=1)
    axis.set_xlabel("CRPSS relative to background")
    axis.set_title("C. Probabilistic added value")
    for row, value in enumerate(skill):
        axis.text(value, row, f" {100 * value:+.1f}%", va="center", fontsize=8)
    axis.grid(axis="x", alpha=0.2)

    daily_matrix = np.stack([daily_crps[name] for name in method_names])
    axis = axes[1, 0]
    image = axis.imshow(daily_matrix, cmap="magma", aspect="auto")
    axis.set_yticks(np.arange(len(method_names)))
    axis.set_yticklabels(method_names)
    axis.set_xticks(np.arange(len(time)))
    axis.set_xticklabels([str(value)[5:] for value in time], rotation=35)
    for row in range(len(method_names)):
        for column in range(len(time)):
            axis.text(column, row, f"{daily_matrix[row, column]:.1f}", ha="center", va="center", fontsize=7, color="white")
    figure.colorbar(image, ax=axis, label="CRPS (mm day$^{-1}$)")
    axis.set_title("D. CRPS by day")

    station_matrix = np.stack([station_crps[name] for name in method_names])
    axis = axes[1, 1]
    image = axis.imshow(station_matrix, cmap="magma", aspect="auto")
    axis.set_yticks(np.arange(len(method_names)))
    axis.set_yticklabels(method_names)
    axis.set_xticks(np.arange(len(eval_idx)))
    axis.set_xticklabels(names[eval_idx], rotation=45, ha="right", fontsize=8)
    for row in range(len(method_names)):
        for column in range(len(eval_idx)):
            axis.text(column, row, f"{station_matrix[row, column]:.1f}", ha="center", va="center", fontsize=7, color="white")
    figure.colorbar(image, ax=axis, label="CRPS (mm day$^{-1}$)")
    axis.set_title("E. CRPS by withheld station")

    axis = axes[1, 2]
    valid_obs = np.isfinite(observed)
    limit_values = [observed[valid_obs], idw[valid_obs]]
    for name, ensemble in methods.items():
        mean = np.mean(ensemble, axis=0)
        limit_values.append(mean[valid_obs])
        axis.scatter(
            observed[valid_obs], mean[valid_obs], s=20, alpha=0.55,
            color=METHOD_COLOURS[name], label=name,
        )
    axis.scatter(observed[valid_obs], idw[valid_obs], s=30, marker="x", color="black", label="IDW gauges")
    limit = max(10.0, float(np.nanpercentile(np.concatenate(limit_values), 99)))
    axis.plot([0, limit], [0, limit], "k--", lw=1)
    axis.set(xlim=(0, limit), ylim=(0, limit), xlabel="Withheld BMD (mm day$^{-1}$)", ylabel="Prediction (mm day$^{-1}$)")
    axis.set_title("F. Station-space comparison")
    axis.legend(fontsize=7)

    axis = axes[2, 0]
    ranks = np.arange(next(iter(methods.values())).shape[0] + 1)
    width = 0.8 / len(method_names)
    for position, (name, ensemble) in enumerate(methods.items()):
        counts = rank_histogram(ensemble, observed)
        axis.bar(
            ranks - 0.4 + width / 2 + position * width,
            counts / max(1, counts.sum()), width,
            color=METHOD_COLOURS[name], alpha=0.9, label=name,
        )
    axis.axhline(1 / len(ranks), color="black", ls="--", lw=1, label="flat")
    axis.set(xlabel="Ensemble rank", ylabel="Relative frequency")
    axis.set_title("G. Rank histogram")
    axis.legend(fontsize=7)

    axis = axes[2, 1]
    for name in method_names:
        brier = [method_scores[name]["thresholds"][f"{value:g}"]["brier_score"] for value in THRESHOLDS_MM]
        axis.plot(THRESHOLDS_MM, brier, marker="o", color=METHOD_COLOURS[name], label=name)
    axis.set(xlabel="Rain threshold (mm day$^{-1}$)", ylabel="Brier score")
    axis.set_title("H. Occurrence and heavy-rain probability")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=7)

    axis = axes[2, 2]
    nominal = np.array([0.50, 0.80, 0.90])
    axis.plot(nominal, nominal, "k--", label="ideal")
    for name in method_names:
        empirical = [method_scores[name]["coverage"][str(int(value * 100))] for value in nominal]
        axis.plot(nominal, empirical, marker="o", color=METHOD_COLOURS[name], label=name)
    axis.set(xlim=(0.45, 0.95), ylim=(0, 1), xlabel="Nominal interval coverage", ylabel="Empirical coverage")
    axis.set_title("I. Ensemble interval calibration")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=7)

    in_training_period = bool(
        report.get("scope", {}).get("in_checkpoint_training_period", False)
    )
    temporal_label = (
        "inside checkpoint training; process test only"
        if in_training_period
        else "outside checkpoint training; tuning must remain frozen"
    )
    figure.suptitle(
        "DA process gate: primary evaluation against withheld BMD gauges\n"
        f"{str(time[0])} to {str(time[-1])}; {len(time)} days; "
        f"{len(assim_idx)} assimilated; {len(eval_idx)} withheld; "
        f"n={np.isfinite(observed).sum()} station-days; {temporal_label}\n{control_text}",
        fontsize=15,
    )
    Path(args.out_diagnostics).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out_diagnostics, dpi=180)
    plt.close(figure)

    # ---------------- rainfall-event and baseline suite ----------------
    figure, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    reliability_thresholds = THRESHOLDS_MM[:3]
    for axis, threshold, panel in zip(axes[0], reliability_thresholds, "ABC"):
        axis.plot([0, 1], [0, 1], "k--", label="ideal")
        for name in method_names:
            rows = method_scores[name]["thresholds"][f"{threshold:g}"]["reliability"]
            x = [row["forecast_probability"] for row in rows if row["n"]]
            y = [row["observed_frequency"] for row in rows if row["n"]]
            axis.plot(x, y, marker="o", color=METHOD_COLOURS[name], label=name)
        events = method_scores["Background"]["thresholds"][f"{threshold:g}"]["observed_events"]
        axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Forecast probability", ylabel="Observed frequency")
        axis.set_title(f"{panel}. Reliability ≥{threshold:g} mm ({events} events)")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)

    axis = axes[1, 0]
    brier_skill = []
    for name in method_names:
        values = []
        for threshold in THRESHOLDS_MM:
            key = f"{threshold:g}"
            reference = method_scores["Background"]["thresholds"][key]["brier_score"]
            score = method_scores[name]["thresholds"][key]["brier_score"]
            values.append(1.0 - score / reference if reference > 0 else np.nan)
        brier_skill.append(values)
    matrix_plot(axis, brier_skill, method_names, [f"≥{value:g}" for value in THRESHOLDS_MM], "D. Brier skill versus background")

    axis = axes[1, 1]
    csi = [
        [method_scores[name]["thresholds"][f"{value:g}"]["csi"] for value in THRESHOLDS_MM]
        for name in method_names
    ]
    matrix_plot(axis, csi, method_names, [f"≥{value:g}" for value in THRESHOLDS_MM], "E. CSI at probability ≥0.5")

    context_names = list(deterministic)
    context_keys = ["rmse_mm", "mae_mm", "bias_mm", "correlation"]
    matrix_plot(
        axes[1, 2],
        [[deterministic[name].get(key, np.nan) for key in context_keys] for name in context_names],
        context_names,
        ["RMSE", "MAE", "Bias", "Corr"],
        "F. Deterministic context\n(CHIRPS is not a target)",
    )
    figure.suptitle(
        "Withheld-BMD rainfall-event verification and simple baselines\n"
        "Small event counts make this diagnostic exploratory; final selection needs rotated folds",
        fontsize=15,
    )
    Path(args.out_events).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out_events, dpi=180)
    plt.close(figure)

    # ---------------- all products collocated to withheld BMD stations ----------------
    comparison_names = list(station_predictions)
    comparison_colours = {
        "CPC": "#8D99AE",
        "Raw IMERG": "#E9C46A",
        "CHIRPS": "#8338EC",
        "IDW gauges": "#222222",
        **METHOD_COLOURS,
    }
    daily_mae = np.array(
        [
            np.nanmean(np.abs(station_predictions[name] - observed), axis=1)
            for name in comparison_names
        ]
    )
    station_mae = np.array(
        [
            np.nanmean(np.abs(station_predictions[name] - observed), axis=0)
            for name in comparison_names
        ]
    )
    figure, axes = plt.subplots(2, 2, figsize=(18, 11), constrained_layout=True)
    comparison_keys = ["rmse_mm", "mae_mm", "bias_mm", "correlation"]
    matrix_plot(
        axes[0, 0],
        [
            [station_comparison_scores[name].get(key, np.nan) for key in comparison_keys]
            for name in comparison_names
        ],
        comparison_names,
        ["RMSE", "MAE", "Bias", "Corr"],
        "A. All products versus withheld BMD\n(DA rows are ensemble means)",
    )

    axis = axes[0, 1]
    image = axis.imshow(daily_mae, cmap="magma", aspect="auto")
    axis.set_yticks(np.arange(len(comparison_names)))
    axis.set_yticklabels(comparison_names)
    axis.set_xticks(np.arange(len(time)))
    axis.set_xticklabels([str(value)[5:] for value in time], rotation=35)
    for row in range(len(comparison_names)):
        for column in range(len(time)):
            axis.text(column, row, f"{daily_mae[row, column]:.1f}", ha="center", va="center", fontsize=7, color="white")
    figure.colorbar(image, ax=axis, label="MAE (mm day$^{-1}$)")
    axis.set_title("B. Absolute error by day")

    axis = axes[1, 0]
    image = axis.imshow(station_mae, cmap="magma", aspect="auto")
    axis.set_yticks(np.arange(len(comparison_names)))
    axis.set_yticklabels(comparison_names)
    axis.set_xticks(np.arange(len(eval_idx)))
    axis.set_xticklabels(names[eval_idx], rotation=45, ha="right")
    for row in range(len(comparison_names)):
        for column in range(len(eval_idx)):
            axis.text(column, row, f"{station_mae[row, column]:.1f}", ha="center", va="center", fontsize=7, color="white")
    figure.colorbar(image, ax=axis, label="MAE (mm day$^{-1}$)")
    axis.set_title("C. Absolute error by withheld station")

    axis = axes[1, 1]
    valid = np.isfinite(observed)
    plot_values = [observed[valid]]
    marker_cycle = ["o", "s", "^", "x", "D", "P", "v", "*", ">"]
    for marker, name in zip(marker_cycle, comparison_names):
        prediction = station_predictions[name]
        pair = valid & np.isfinite(prediction)
        plot_values.append(prediction[pair])
        axis.scatter(
            observed[pair], prediction[pair], s=24, alpha=0.45,
            marker=marker, color=comparison_colours[name], label=name,
        )
    scatter_limit = max(10.0, float(np.nanpercentile(np.concatenate(plot_values), 99)))
    axis.plot([0, scatter_limit], [0, scatter_limit], "k--", lw=1)
    axis.set(
        xlim=(0, scatter_limit), ylim=(0, scatter_limit),
        xlabel="Withheld BMD observation (mm day$^{-1}$)",
        ylabel="Collocated product value (mm day$^{-1}$)",
    )
    axis.set_title("D. Same station-days for every product")
    axis.legend(fontsize=7, ncol=2)
    figure.suptitle(
        "Direct station-collocated rainfall-product comparison\n"
        "BMD is the reference; CPC, IMERG, CHIRPS, IDW, background and all DA methods "
        "use exactly the same withheld station-days",
        fontsize=15,
    )
    Path(args.out_station_comparison).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out_station_comparison, dpi=180)
    plt.close(figure)

    # ---------------- spatial cases: qualitative, station-anchored ----------------
    chirps, condition = np.asarray(data["chirps"]), np.asarray(data["condition"])
    background = np.asarray(data["background"])
    analysis_gauge = np.asarray(data["analysis_gauge"] if "analysis_gauge" in data.files else data["analysis"])
    analysis_imerg = np.asarray(data["analysis_imerg"] if "analysis_imerg" in data.files else data["analysis"])
    analysis_combined = np.asarray(data["analysis_combined"] if "analysis_combined" in data.files else data["analysis"])
    analysis = analysis_combined if has_imerg else analysis_gauge
    imerg = np.asarray(data["imerg"]) if "imerg" in data.files else None
    valid_grid = np.asarray(data["valid"]).astype(bool)

    def ensemble_mean(values):
        return np.where(valid_grid[None], np.nanmean(values, axis=1), np.nan)

    background_mean_field = ensemble_mean(background)
    gauge_mean_field = ensemble_mean(analysis_gauge)
    imerg_mean_field = ensemble_mean(analysis_imerg)
    combined_mean_field = ensemble_mean(analysis_combined)
    analysis_mean_field = ensemble_mean(analysis)
    analysis_spread = np.where(valid_grid[None], np.nanstd(analysis, axis=1, ddof=1), np.nan)
    network_total = np.nansum(gauge, axis=1)
    selected_days = list(
        dict.fromkeys(
            [
                int(np.argmax(network_total)),
                int(np.argmax(np.nanmax(gauge, axis=1))),
                int(np.argmin(np.abs(network_total - np.nanmedian(network_total)))),
            ]
        )
    )
    for candidate in np.argsort(network_total)[::-1]:
        if len(selected_days) >= min(3, len(time)):
            break
        if int(candidate) not in selected_days:
            selected_days.append(int(candidate))

    grid_lat, grid_lon = data["grid_lat"], data["grid_lon"]
    extent = [grid_lon[0], grid_lon[-1], grid_lat[0], grid_lat[-1]]
    n_columns = 9 if has_imerg else 7
    figure, axes = plt.subplots(
        len(selected_days), n_columns,
        figsize=(3.2 * n_columns, 3.4 * len(selected_days)), constrained_layout=True,
    )
    if len(selected_days) == 1:
        axes = axes[None, :]
    column_titles = (
        [
            "CPC input", "IMERG observation", "CHIRPS context\n(not truth)",
            "Background mean", "Gauges-only mean", "IMERG-only mean",
            "Simultaneous mean", "Simultaneous − background", "Simultaneous spread",
        ]
        if has_imerg
        else [
            "CPC input", "CHIRPS context\n(not truth)", "Background mean",
            "Gauges-only mean", "Analysis mean", "Analysis − background", "Analysis spread",
        ]
    )
    for row, day in enumerate(selected_days):
        precipitation_fields = [condition[day]]
        if has_imerg:
            precipitation_fields.extend(
                [
                    imerg[day], chirps[day], background_mean_field[day], gauge_mean_field[day],
                    imerg_mean_field[day], combined_mean_field[day],
                ]
            )
        else:
            precipitation_fields.extend(
                [chirps[day], background_mean_field[day], gauge_mean_field[day], analysis_mean_field[day]]
            )
        rain_max = max(10.0, max(float(np.nanpercentile(value, 99)) for value in precipitation_fields))
        increment = analysis_mean_field[day] - background_mean_field[day]
        increment_limit = max(1.0, float(np.nanpercentile(np.abs(increment), 99)))
        spread_limit = max(1.0, float(np.nanpercentile(analysis_spread[day], 99)))
        images = []
        for column, field in enumerate(precipitation_fields):
            images.append(
                axes[row, column].imshow(field, origin="lower", extent=extent, cmap="viridis", vmin=0, vmax=rain_max)
            )
        images.append(
            axes[row, len(precipitation_fields)].imshow(
                increment, origin="lower", extent=extent, cmap="RdBu_r", vmin=-increment_limit, vmax=increment_limit
            )
        )
        images.append(
            axes[row, len(precipitation_fields) + 1].imshow(
                analysis_spread[day], origin="lower", extent=extent, cmap="magma", vmin=0, vmax=spread_limit
            )
        )
        method_columns = range(3, 7) if has_imerg else range(2, 5)
        for column in method_columns:
            add_rain_markers(axes[row, column], lon, lat, gauge[day], assim_idx, eval_idx, rain_max)
        axes[row, 0].set_ylabel(f"{time[day]}\nBMD mean {np.nanmean(gauge[day]):.1f} mm", fontsize=9)
        for column, axis in enumerate(axes[row]):
            if row == 0:
                axis.set_title(column_titles[column], fontsize=9)
            axis.set_xlim(extent[0], extent[1])
            axis.set_ylim(extent[2], extent[3])
            axis.tick_params(labelsize=7)
            figure.colorbar(images[column], ax=axis, orientation="horizontal", fraction=0.045, pad=0.04)
    figure.suptitle(
        f"Selected DA spatial cases from {str(time[0])} to {str(time[-1])}, "
        "anchored by BMD station observations\n"
        "Filled circles are assimilated gauges; cyan-edged circles are withheld; maps are qualitative\n"
        + control_text,
        fontsize=14,
    )
    Path(args.out_spatial).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out_spatial, dpi=180)
    plt.close(figure)

    # ---------------- non-independent gridded product intercomparison ----------------
    if imerg is not None:
        # Derive the upsample from the shapes. A hardcoded 2 is right only for
        # 0.1 deg IMERG; the scale-ladder arms assimilate 0.4 and 0.8 deg
        # footprints, and factor 8 then produced a 32x32 field that could not
        # broadcast against the 128x128 valid mask -- after the assimilation had
        # already succeeded and written its NPZ.
        scale_lat = max(1, -(-valid_grid.shape[0] // imerg.shape[1]))
        scale_lon = max(1, -(-valid_grid.shape[1] // imerg.shape[2]))
        imerg_fine = np.repeat(np.repeat(imerg, scale_lat, axis=1), scale_lon, axis=2)
        imerg_fine = imerg_fine[:, : valid_grid.shape[0], : valid_grid.shape[1]]
    else:
        imerg_fine = np.full_like(condition, np.nan)
    product_daily = [
        ("CPC input", condition),
        ("IMERG", imerg_fine),
        ("CHIRPS", chirps),
        ("Background", background_mean_field),
        ("Gauges only", gauge_mean_field),
        ("IMERG only", imerg_mean_field),
        ("Simultaneous", combined_mean_field),
    ]
    if not has_imerg:
        product_daily = [item for item in product_daily if "IMERG" not in item[0]]
    product_daily = [
        (name, np.where(valid_grid[None], np.asarray(value), np.nan))
        for name, value in product_daily
    ]
    period_mean = [np.nanmean(value, axis=0) for _, value in product_daily]
    wet_frequency = [np.nanmean(value >= 1.0, axis=0) for _, value in product_daily]
    texture = [value - block_smooth(np.nan_to_num(value, nan=0.0), factor=4) for value in period_mean]
    rain_limit = max(10.0, float(np.nanpercentile(np.stack(period_mean), 99)))
    texture_limit = max(1.0, float(np.nanpercentile(np.abs(np.stack(texture)), 98)))
    row_fields = [period_mean, wet_frequency, texture]
    row_specs = [
        ("viridis", 0.0, rain_limit, f"{len(time)}-day mean (mm day$^{{-1}}$)"),
        ("Blues", 0.0, 1.0, "Wet-day frequency (≥1 mm)"),
        ("RdBu_r", -texture_limit, texture_limit, "Fine-scale departure from 0.2° mean"),
    ]
    figure, axes = plt.subplots(3, len(product_daily), figsize=(3.1 * len(product_daily), 10.5), constrained_layout=True)
    for row, (cmap, lower, upper, label) in enumerate(row_specs):
        for column, (name, _) in enumerate(product_daily):
            field = np.where(valid_grid, row_fields[row][column], np.nan)
            image = axes[row, column].imshow(field, origin="lower", extent=extent, cmap=cmap, vmin=lower, vmax=upper)
            if row == 0:
                axes[row, column].set_title(name, fontsize=10)
            if column == 0:
                axes[row, column].set_ylabel(label, fontsize=9)
            axes[row, column].tick_params(labelsize=7)
            figure.colorbar(image, ax=axes[row, column], orientation="horizontal", fraction=0.045, pad=0.04)
    figure.suptitle(
        "Non-independent gridded-product intercomparison — not method validation\n"
        "CHIRPS was the model training target; IMERG Final and CHIRPS are gauge-informed; "
        "method ranking comes only from withheld BMD gauges",
        fontsize=14,
    )
    Path(args.out_intercomparison).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out_intercomparison, dpi=180)
    plt.close(figure)
    data.close()

    print(f"wrote {args.out_evaluation}")
    print(f"wrote {args.out_diagnostics}")
    print(f"wrote {args.out_events}")
    print(f"wrote {args.out_station_comparison}")
    print(f"wrote {args.out_spatial}")
    print(f"wrote {args.out_intercomparison}")
    print(f"primary withheld-BMD winner: {winner}")
    if best_fused:
        print(
            f"best fused: {best_fused}; beats gauges only: "
            f"{evaluation['selection']['best_fused_beats_gauge_only']}"
        )


if __name__ == "__main__":
    main()
