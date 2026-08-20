#!/usr/bin/env python3
"""Plot matched V7/CPCv2 DA skill and 0.05-degree subgrid diagnostics.

The station dumps provide the only fair verification: both families are scored
against the same BMD reports at the same withheld stations.  The V7 map dump
and CPCv2 mean fields are then put on V7's stage-B geographic window.  This is
important: a map can show whether either model retains sub-0.1-degree texture,
but it is not gridded truth when running with real observations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COMPARISONS = {
    "gauges_only": ("da_meso", "guided_s6_g010_t100"),
    "simultaneous": ("da_sim", "v2_simul_s04_ig010"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v7-dump", required=True)
    parser.add_argument("--cpcv2-dump", required=True)
    parser.add_argument("--v7-map-dump", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def as_days(values: np.ndarray) -> np.ndarray:
    return np.asarray(values).astype("datetime64[D]")


def required(archive: np.lib.npyio.NpzFile, path: Path, *keys: str) -> None:
    missing = [key for key in keys if key not in archive]
    if missing:
        raise ValueError(f"{path} lacks required arrays {missing}")


def fair_crps(members: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Fair CRPS per finite station, for one day (members, stations)."""
    keep = np.isfinite(observed) & np.all(np.isfinite(members), axis=0)
    if not keep.any():
        return np.empty(0, dtype=float)
    selected = members[:, keep]
    truth = observed[keep]
    n_members = selected.shape[0]
    if n_members < 2:
        raise ValueError("fair CRPS needs at least two ensemble members")
    return (
        np.abs(selected - truth[None]).mean(axis=0)
        - np.abs(selected[:, None] - selected[None, :]).sum(axis=(0, 1))
        / (2.0 * n_members * (n_members - 1))
    )


def daily_score(members: np.ndarray, observed: np.ndarray, held: np.ndarray) -> dict:
    """Per-day CRPS/MAE/RMSE on the locked held-out stations."""
    result = {"crps_mm": [], "mae_mm": [], "rmse_mm": [], "n": []}
    for day in range(members.shape[0]):
        ensemble = members[day][:, held]
        truth = observed[day, held]
        keep = np.isfinite(truth) & np.all(np.isfinite(ensemble), axis=0)
        result["n"].append(int(keep.sum()))
        if not keep.any():
            for key in ("crps_mm", "mae_mm", "rmse_mm"):
                result[key].append(float("nan"))
            continue
        mean = ensemble[:, keep].mean(axis=0)
        result["crps_mm"].append(float(fair_crps(ensemble, truth).mean()))
        result["mae_mm"].append(float(np.abs(mean - truth[keep]).mean()))
        result["rmse_mm"].append(float(np.sqrt(np.mean((mean - truth[keep]) ** 2))))
    return result


def aligned_station_data(v7_path: Path, cpc_path: Path) -> tuple[np.ndarray, dict, dict]:
    """Return matched daily scores and station geometry, refusing split drift."""
    with np.load(v7_path, allow_pickle=False) as v7, np.load(cpc_path, allow_pickle=False) as cpc:
        required(v7, v7_path, "times", "station_ids", "eval_idx", "observed_mm",
                 "station_lat", "station_lon")
        required(cpc, cpc_path, "times", "station_ids", "eval_idx", "gauge_mm")
        times = as_days(v7["times"])
        if not np.array_equal(times, as_days(cpc["times"])):
            raise ValueError("V7 and CPCv2 BMD dates differ")
        v7_ids = np.asarray(v7["station_ids"], dtype=str)
        cpc_ids = np.asarray(cpc["station_ids"], dtype=str)
        if len(v7_ids) != len(set(v7_ids)) or set(v7_ids) != set(cpc_ids):
            raise ValueError("station pools do not match")
        cpc_order = np.asarray([np.where(cpc_ids == name)[0][0] for name in v7_ids])
        held_ids = set(v7_ids[np.asarray(v7["eval_idx"], int)])
        cpc_held = set(cpc_ids[np.asarray(cpc["eval_idx"], int)])
        if held_ids != cpc_held:
            raise ValueError("withheld station IDs differ")
        held = np.flatnonzero(np.isin(v7_ids, sorted(held_ids)))
        observed = np.asarray(v7["observed_mm"], float)
        cpc_observed = np.asarray(cpc["gauge_mm"], float)[:, cpc_order]
        same = np.isfinite(observed) == np.isfinite(cpc_observed)
        finite = np.isfinite(observed)
        if not same.all() or not np.allclose(observed[finite], cpc_observed[finite], atol=1e-5):
            raise ValueError("BMD values differ between V7 and CPCv2")

        scores: dict[str, dict] = {}
        for label, (v7_arm, cpc_arm) in COMPARISONS.items():
            v7_key, cpc_key = f"station_{v7_arm}", f"station_{cpc_arm}"
            required(v7, v7_path, v7_key)
            required(cpc, cpc_path, cpc_key)
            v7_members = np.asarray(v7[v7_key], float)
            cpc_members = np.asarray(cpc[cpc_key], float)[:, :, cpc_order]
            if v7_members.shape != cpc_members.shape:
                raise ValueError(f"{label}: ensemble shapes differ")
            scores[label] = {
                "v7": daily_score(v7_members, observed, held),
                "cpcv2": daily_score(cpc_members, observed, held),
            }
        geometry = {
            "station_lat": np.asarray(v7["station_lat"], float),
            "station_lon": np.asarray(v7["station_lon"], float),
            "assim_idx": np.asarray(v7["assim_idx"], int),
            "eval_idx": held,
        }
    return times, scores, geometry


def crop_to_target(field: np.ndarray, valid: np.ndarray, lat: np.ndarray,
                   lon: np.ndarray, target_lat: np.ndarray,
                   target_lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Crop a CPCv2 field to V7's exact 0.05-degree stage-B coordinates."""
    row = np.asarray([np.argmin(np.abs(lat - value)) for value in target_lat])
    col = np.asarray([np.argmin(np.abs(lon - value)) for value in target_lon])
    if (not np.allclose(lat[row], target_lat, atol=1e-6, rtol=0.0)
            or not np.allclose(lon[col], target_lon, atol=1e-6, rtol=0.0)):
        raise ValueError("CPCv2 grid does not contain V7's stage-B 0.05-degree window")
    cropped = field[:, row][:, :, col]
    if valid.ndim == 2:
        valid = np.broadcast_to(valid, field.shape)
    return cropped, valid[:, row][:, :, col].astype(bool)


def map_data(v7_map_path: Path, cpc_path: Path, times: np.ndarray) -> tuple[dict, dict]:
    """Load model maps, align dates, and crop CPCv2 to V7's product window."""
    with np.load(v7_map_path, allow_pickle=False) as v7, np.load(cpc_path, allow_pickle=False) as cpc:
        required(v7, v7_map_path, "times", "grid_lat", "grid_lon", "valid",
                 "meanfield_background", "meanfield_da_meso", "meanfield_da_sim")
        required(cpc, cpc_path, "times", "grid_lat", "grid_lon", "valid",
                 "meanfield_background", "meanfield_guided_s6_g010_t100",
                 "meanfield_v2_simul_s04_ig010")
        if not np.array_equal(as_days(v7["times"]), times) or not np.array_equal(
            as_days(cpc["times"]), times
        ):
            raise ValueError("map and station dump dates differ")
        target_lat, target_lon = np.asarray(v7["grid_lat"], float), np.asarray(v7["grid_lon"], float)
        v7_valid = np.asarray(v7["valid"], bool)
        v7_maps = {
            "background": np.asarray(v7["meanfield_background"], float),
            "gauges_only": np.asarray(v7["meanfield_da_meso"], float),
            "simultaneous": np.asarray(v7["meanfield_da_sim"], float),
        }
        cpc_valid = np.asarray(cpc["valid"], bool)
        cpc_native = {
            "background": np.asarray(cpc["meanfield_background"], float),
            "gauges_only": np.asarray(cpc["meanfield_guided_s6_g010_t100"], float),
            "simultaneous": np.asarray(cpc["meanfield_v2_simul_s04_ig010"], float),
        }
        cpc_maps = {}
        cpc_valid_crop = None
        for name, field in cpc_native.items():
            cropped, valid = crop_to_target(
                field, cpc_valid, np.asarray(cpc["grid_lat"], float),
                np.asarray(cpc["grid_lon"], float), target_lat, target_lon
            )
            cpc_maps[name] = cropped
            cpc_valid_crop = valid
        for name, field in v7_maps.items():
            if field.shape != cpc_maps[name].shape:
                raise ValueError(f"{name}: V7/CPCv2 map shapes differ after crop")
        keep = v7_valid & cpc_valid_crop
    return ({"lat": target_lat, "lon": target_lon, "valid": keep, "fields": v7_maps},
            {"lat": target_lat, "lon": target_lon, "valid": keep, "fields": cpc_maps})


def subgrid_sd(field: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Within-0.1-degree standard deviation, expanded back to 0.05-degree cells."""
    if field.shape != valid.shape or field.shape[0] % 2 or field.shape[1] % 2:
        raise ValueError("subgrid diagnostic needs an even 0.05-degree grid")
    blocks = field.reshape(field.shape[0] // 2, 2, field.shape[1] // 2, 2)
    keep = valid.reshape(valid.shape[0] // 2, 2, valid.shape[1] // 2, 2)
    complete = keep.all(axis=(1, 3))
    values = np.where(keep, blocks, np.nan)
    with np.errstate(invalid="ignore"):
        spread = np.nanstd(values, axis=(1, 3))
    spread[~complete] = np.nan
    return np.repeat(np.repeat(spread, 2, axis=0), 2, axis=1)


def finite_span(*arrays: np.ndarray, percentile: float = 99.0) -> float:
    values = np.concatenate([array[np.isfinite(array)] for array in arrays])
    return float(np.percentile(np.abs(values), percentile)) if values.size else 1.0


def plot_skill(times: np.ndarray, scores: dict, out_path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
    for row, label in enumerate(COMPARISONS):
        for column, metric in enumerate(("crps_mm", "mae_mm")):
            axis = axes[row, column]
            for model, colour in (("v7", "tab:blue"), ("cpcv2", "tab:orange")):
                axis.plot(times, scores[label][model][metric], marker="o", lw=1.8,
                          color=colour, label="V7" if model == "v7" else "CPCv2")
            axis.set_title(f"{label.replace('_', ' ')}: {metric.replace('_mm', '').upper()} (lower is better)")
            axis.grid(alpha=0.3)
            if row == 0 and column == 0:
                axis.legend()
            if column == 0:
                axis.set_ylabel("mm/day")
    figure.suptitle("Matched withheld-BMD DA skill by BMD observation day", fontsize=13)
    figure.autofmt_xdate()
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(out_path, dpi=150)
    plt.close(figure)


def plot_subgrid_timeseries(times: np.ndarray, v7: dict, cpc: dict, out_path: Path) -> dict:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharex=True)
    summary: dict[str, dict] = {}
    for label in COMPARISONS:
        summary[label] = {}
        for model, payload, colour in (("v7", v7, "tab:blue"), ("cpcv2", cpc, "tab:orange")):
            means, p95s = [], []
            for day in range(len(times)):
                texture = subgrid_sd(payload["fields"][label][day], payload["valid"][day])
                finite = texture[np.isfinite(texture)]
                means.append(float(finite.mean()) if finite.size else float("nan"))
                p95s.append(float(np.percentile(finite, 95)) if finite.size else float("nan"))
            summary[label][model] = {"mean_within_0p1_sd_mm": means, "p95_within_0p1_sd_mm": p95s}
            axes[0].plot(times, means, marker="o", lw=1.8, color=colour,
                         label=f"{label.replace('_', ' ')} — {model}")
            axes[1].plot(times, p95s, marker="o", lw=1.8, color=colour,
                         label=f"{label.replace('_', ' ')} — {model}")
    axes[0].set_title("Mean within-0.1° SD")
    axes[1].set_title("95th-percentile within-0.1° SD")
    for axis in axes:
        axis.set_ylabel("mm/day")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7)
    figure.suptitle("Retained 0.05° subgrid variability (descriptive, not a skill score)", fontsize=12)
    figure.autofmt_xdate()
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    return summary


def plot_day_maps(times: np.ndarray, v7: dict, cpc: dict, stations: dict, out_dir: Path) -> None:
    extent = (v7["lon"].min() - 0.025, v7["lon"].max() + 0.025,
              v7["lat"].min() - 0.025, v7["lat"].max() + 0.025)
    for day, date in enumerate(times):
        valid = v7["valid"][day]
        cpc_gauge, v7_gauge = cpc["fields"]["gauges_only"][day], v7["fields"]["gauges_only"][day]
        cpc_sim, v7_sim = cpc["fields"]["simultaneous"][day], v7["fields"]["simultaneous"][day]
        cpc_inc = cpc_sim - cpc["fields"]["background"][day]
        v7_inc = v7_sim - v7["fields"]["background"][day]
        cpc_tex, v7_tex = subgrid_sd(cpc_sim, valid), subgrid_sd(v7_sim, valid)
        rain_max = finite_span(cpc_gauge, v7_gauge, cpc_sim, v7_sim, percentile=99.5)
        inc_max = finite_span(cpc_inc, v7_inc)
        tex_max = finite_span(cpc_tex, v7_tex)
        diff_rain = finite_span(v7_gauge - cpc_gauge, v7_sim - cpc_sim)
        diff_inc = finite_span(v7_inc - cpc_inc)
        diff_tex = finite_span(v7_tex - cpc_tex)
        rows = [
            ("CPCv2", [cpc_gauge, cpc_sim, cpc_inc, cpc_tex]),
            ("V7", [v7_gauge, v7_sim, v7_inc, v7_tex]),
            ("V7 − CPCv2", [v7_gauge - cpc_gauge, v7_sim - cpc_sim,
                              v7_inc - cpc_inc, v7_tex - cpc_tex]),
        ]
        titles = ["gauge-only DA mean", "simultaneous DA mean", "simultaneous increment", "sim 0.1° within-cell SD"]
        figure, axes = plt.subplots(3, 4, figsize=(16, 11), squeeze=False)
        for row, (model, fields) in enumerate(rows):
            for column, field in enumerate(fields):
                shown = np.where(valid, field, np.nan)
                if row < 2 and column < 2:
                    cmap, low, high = "turbo", 0.0, rain_max
                elif row < 2 and column == 2:
                    cmap, low, high = "RdBu_r", -inc_max, inc_max
                elif row < 2:
                    cmap, low, high = "magma", 0.0, tex_max
                elif column < 2:
                    cmap, low, high = "RdBu_r", -diff_rain, diff_rain
                elif column == 2:
                    cmap, low, high = "RdBu_r", -diff_inc, diff_inc
                else:
                    cmap, low, high = "RdBu_r", -diff_tex, diff_tex
                axis = axes[row, column]
                image = axis.imshow(shown, origin="lower", extent=extent, cmap=cmap,
                                    vmin=low, vmax=high, aspect="auto")
                figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03, label="mm/day")
                if row < 2:
                    axis.scatter(stations["station_lon"][stations["assim_idx"]],
                                 stations["station_lat"][stations["assim_idx"]],
                                 s=14, facecolors="none", edgecolors="white", linewidths=0.6)
                    axis.scatter(stations["station_lon"][stations["eval_idx"]],
                                 stations["station_lat"][stations["eval_idx"]],
                                 s=20, marker="^", facecolors="none", edgecolors="magenta", linewidths=0.8)
                if row == 0:
                    axis.set_title(titles[column], fontsize=10)
                if column == 0:
                    axis.set_ylabel(model, fontsize=10)
                axis.tick_params(labelsize=7)
        figure.suptitle(
            f"Matched 0.05° subgrid diagnostics — BMD day {date}\n"
            "circles: assimilated gauges; triangles: withheld gauges; maps are descriptive, BMD scores are authoritative",
            fontsize=12,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.95))
        figure.savefig(out_dir / f"subgrid_maps_{str(date)}.png", dpi=140)
        plt.close(figure)


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    times, scores, stations = aligned_station_data(Path(args.v7_dump), Path(args.cpcv2_dump))
    v7, cpc = map_data(Path(args.v7_map_dump), Path(args.cpcv2_dump), times)
    plot_skill(times, scores, out_dir / "da_skill_timeseries.png")
    texture = plot_subgrid_timeseries(times, v7, cpc, out_dir / "subgrid_variability_timeseries.png")
    plot_day_maps(times, v7, cpc, stations, out_dir)
    summary = {
        "bmd_observation_dates": times.astype(str).tolist(),
        "daily_withheld_scores": scores,
        "subgrid_variability": texture,
        "map_window": {
            "lat": [float(v7["lat"][0]), float(v7["lat"][-1])],
            "lon": [float(v7["lon"][0]), float(v7["lon"][-1])],
            "resolution_deg": float(np.median(np.diff(v7["lat"]))),
        },
        "interpretation": (
            "CRPS and MAE are verified only at locked withheld BMD gauges. "
            "The 0.05-degree maps and within-0.1-degree standard deviations show "
            "retained spatial structure, not gridded truth under real observations."
        ),
    }
    (out_dir / "diagnostics_summary.json").write_text(json.dumps(json_ready(summary), indent=2))
    print(f"[diagnostics] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
