#!/usr/bin/env python3
"""Evaluate and plot the corrected v4 short-window DA pilot.

Independent gauge skill uses only ``station_withheld`` and the folded gauge
arms.  Spatial maps use the separate all-gauge arms.  CHIRPS, CPC and IMERG
scores are explicitly product-pattern agreement, not independent truth.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import zarr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bdhires.da import AreaWeightedBlockObsOperator, BilinearObsOperator  # noqa: E402
from bdhires.data import (  # noqa: E402
    SUBGRID_SCHEMA,
    SubgridEncoding,
    area_weighted_block_mean,
    encoding_metadata,
)
from bdhires.grids import Grid  # noqa: E402


POINT_METHODS = (
    "background",
    "gauges_withheld",
    "imerg_only",
    "simultaneous_withheld",
)
MAP_METHODS = (
    "background",
    "gauges_all",
    "imerg_only",
    "simultaneous_all",
)
DISPLAY = {
    "background": "background",
    "gauges_withheld": "gauges (withheld)",
    "imerg_only": "IMERG only",
    "simultaneous_withheld": "simultaneous (withheld)",
    "gauges_all": "gauges (all)",
    "simultaneous_all": "simultaneous (all)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-store", required=True)
    parser.add_argument("--sample-store", required=True)
    parser.add_argument("--gridded-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--wet-threshold-mm", type=float, default=1.0)
    parser.add_argument("--plot-days", type=int, default=5)
    return parser.parse_args()


def ensemble_station_metrics(
    prediction: np.ndarray,
    observation: np.ndarray,
    wet_threshold: float = 1.0,
) -> dict:
    """Score (time,member,station) members against (time,station) observations."""
    valid = np.isfinite(observation) & np.all(np.isfinite(prediction), axis=1)
    if not valid.any():
        return {"n": 0}
    members = prediction.transpose(1, 0, 2)[:, valid]
    truth = observation[valid]
    mean = members.mean(axis=0)
    difference = mean - truth
    first = np.mean(np.abs(members - truth[None]), axis=0)
    second = 0.5 * np.mean(
        np.abs(members[:, None] - members[None, :]), axis=(0, 1)
    )
    low, high = np.quantile(members, [0.05, 0.95], axis=0)
    dry = truth < wet_threshold
    wet = ~dry

    def subset_mae(mask) -> float | None:
        return float(np.mean(np.abs(difference[mask]))) if mask.any() else None

    correlation = None
    if mean.std() > 0.0 and truth.std() > 0.0:
        correlation = float(np.corrcoef(mean, truth)[0, 1])
    return {
        "n": int(valid.sum()),
        "crps_mm_day": float(np.mean(first - second)),
        "rmse_mm_day": float(np.sqrt(np.mean(difference**2))),
        "mae_mm_day": float(np.mean(np.abs(difference))),
        "dry_mae_mm_day": subset_mae(dry),
        "wet_mae_mm_day": subset_mae(wet),
        "bias_mm_day": float(np.mean(difference)),
        "correlation": correlation,
        "coverage90": float(np.mean((truth >= low) & (truth <= high))),
    }


def station_samples(
    fields: np.ndarray,
    grid: Grid,
    lat: np.ndarray,
    lon: np.ndarray,
) -> np.ndarray:
    time, members, height, width = fields.shape
    operator = BilinearObsOperator(grid, lat, lon)
    flat = torch.from_numpy(fields.reshape(time * members, 1, height, width))
    with torch.no_grad():
        sampled = operator(flat)[:, 0].numpy()
    return sampled.reshape(time, members, len(lat))


def block_anomaly(
    values: np.ndarray,
    area: np.ndarray,
    valid: np.ndarray,
    factor: int,
) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(values, np.float32))[:, None]
    coarse, retained, _ = area_weighted_block_mean(
        tensor,
        torch.from_numpy(area),
        torch.from_numpy(valid),
        factor=factor,
        valid_area_threshold=0.0,
    )
    expanded = coarse.repeat_interleave(factor, -2).repeat_interleave(factor, -1)
    retained_fine = retained.repeat_interleave(factor, -2).repeat_interleave(factor, -1)
    anomaly = torch.where(retained_fine, tensor - expanded, torch.nan)
    return anomaly[:, 0].numpy()


def mean_daily_correlation(
    candidate: np.ndarray,
    reference: np.ndarray,
    valid: np.ndarray,
) -> float | None:
    values = []
    for left, right in zip(candidate, reference):
        keep = valid & np.isfinite(left) & np.isfinite(right)
        if keep.sum() < 3:
            continue
        x, y = left[keep], right[keep]
        if x.std() > 0.0 and y.std() > 0.0:
            values.append(float(np.corrcoef(x, y)[0, 1]))
    return float(np.mean(values)) if values else None


def coarse_product_correlation(
    means: dict[str, np.ndarray],
    reference: np.ndarray,
    area: np.ndarray,
    valid: np.ndarray,
    factor: int,
) -> dict[str, float | None]:
    out = {}
    for name, field in means.items():
        coarse, retained, _ = area_weighted_block_mean(
            torch.from_numpy(field)[:, None],
            torch.from_numpy(area),
            torch.from_numpy(valid),
            factor=factor,
            valid_area_threshold=0.0,
        )
        candidate = coarse[:, 0].numpy()
        out[name] = mean_daily_correlation(
            candidate, reference, retained[0, 0].numpy()
        )
    return out


def imerg_product_correlation(
    means: dict[str, np.ndarray],
    reference: np.ndarray,
    area: np.ndarray,
    valid: np.ndarray,
    factor: int,
    crop: tuple[int, int, int, int],
) -> dict[str, float | None]:
    operator = AreaWeightedBlockObsOperator(
        factor, area, valid=valid, min_valid_frac=0.5, crop=crop
    )
    keep = operator.valid_mask().numpy().astype(bool)
    ref = reference.reshape(reference.shape[0], -1)
    out = {}
    for name, field in means.items():
        with torch.no_grad():
            candidate = operator(torch.from_numpy(field)[:, None])[:, 0].numpy()
        daily = []
        for left, right in zip(candidate, ref):
            selected = keep & np.isfinite(left) & np.isfinite(right)
            if selected.sum() >= 3 and left[selected].std() > 0.0 and right[selected].std() > 0.0:
                daily.append(float(np.corrcoef(left[selected], right[selected])[0, 1]))
        out[name] = float(np.mean(daily)) if daily else None
    return out


def finite_or_none(value):
    if isinstance(value, dict):
        return {key: finite_or_none(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_or_none(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def plot_value(value) -> float:
    """Convert an optional JSON metric to a Matplotlib-safe scalar."""
    return np.nan if value is None else float(value)


def metric_text(value, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def summarize_sampler_diagnostics(raw: dict, methods: tuple[str, ...]) -> dict:
    """Keep the terminal fit and decoder screens visible beside skill scores."""
    summary = {}
    for name in methods:
        daily = list(raw.get(name, {}).get("daily", []))

        def values(key: str) -> list[float]:
            return [float(item[key]) for item in daily if key in item]

        bias = values("terminal_oa_bias_sigma")
        rmse = values("terminal_oa_rmse_sigma")
        maximum = values("terminal_oa_max_abs_sigma")
        fallback = values("reconstruction_fallback_fraction")
        clipped = values("reconstruction_clipped_intensity_fraction")
        bounds = [
            bool(item["soft_hard_bounds_pass"])
            for item in daily
            if "soft_hard_bounds_pass" in item
        ]
        summary[name] = {
            "days": len(daily),
            "terminal_oa_bias_sigma_mean": float(np.mean(bias)) if bias else None,
            "terminal_oa_rmse_sigma_mean": float(np.mean(rmse)) if rmse else None,
            "terminal_oa_max_abs_sigma": max(maximum) if maximum else None,
            "soft_hard_bounds_all_pass": all(bounds) if bounds else None,
            "decoder_fallback_fraction_max": max(fallback) if fallback else None,
            "decoder_clipped_intensity_fraction_max": max(clipped) if clipped else None,
        }
    return summary


def plot_matrix(payload: dict, output: Path) -> None:
    point = payload["withheld_gauges"]
    structure = payload["spatial_product_agreement"]
    gridded = payload["gridded_chirps_agreement"]
    point_positions = np.arange(len(POINT_METHODS))
    map_positions = np.arange(len(MAP_METHODS))
    point_labels = [DISPLAY[name] for name in POINT_METHODS]
    map_labels = [DISPLAY[name] for name in MAP_METHODS]
    figure, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

    axes[0, 0].barh(
        point_positions,
        [point[name]["crps_mm_day"] for name in POINT_METHODS],
        color="#C1440E",
    )
    axes[0, 0].set_yticks(point_positions, point_labels)
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_xlabel("CRPS (mm/day)")
    axes[0, 0].set_title("A. Independently withheld BMD gauges")

    rmse = [point[name]["rmse_mm_day"] for name in POINT_METHODS]
    bias = [point[name]["bias_mm_day"] for name in POINT_METHODS]
    axes[0, 1].barh(point_positions, rmse, color="#457B9D", label="RMSE")
    axes[0, 1].scatter(bias, point_positions, color="#D1495B", marker="D", label="bias")
    axes[0, 1].axvline(0.0, color="black", linewidth=0.8)
    axes[0, 1].set_yticks(point_positions, point_labels)
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_xlabel("mm/day")
    axes[0, 1].legend()
    axes[0, 1].set_title("B. Withheld error and bias")

    width = 0.38
    axes[0, 2].barh(
        point_positions - width / 2,
        [plot_value(point[name]["dry_mae_mm_day"]) for name in POINT_METHODS],
        height=width,
        color="#E9C46A",
        label="dry",
    )
    axes[0, 2].barh(
        point_positions + width / 2,
        [plot_value(point[name]["wet_mae_mm_day"]) for name in POINT_METHODS],
        height=width,
        color="#2A9D8F",
        label="wet",
    )
    axes[0, 2].set_yticks(point_positions, point_labels)
    axes[0, 2].invert_yaxis()
    axes[0, 2].set_xlabel("MAE (mm/day)")
    axes[0, 2].legend()
    axes[0, 2].set_title("C. Dry/wet trade-off")

    axes[1, 0].barh(
        map_positions - width / 2,
        [plot_value(structure[name]["chirps_mean_pattern_r"]) for name in MAP_METHODS],
        height=width,
        color="#4C78A8",
        label="full field",
    )
    axes[1, 0].barh(
        map_positions + width / 2,
        [plot_value(structure[name]["chirps_subgrid_pattern_r"]) for name in MAP_METHODS],
        height=width,
        color="#9C6ADE",
        label="within 0.5°",
    )
    axes[1, 0].set_yticks(map_positions, map_labels)
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlim(-0.2, 1.0)
    axes[1, 0].set_xlabel("mean daily correlation")
    axes[1, 0].legend()
    axes[1, 0].set_title("D. CHIRPS pattern agreement (not truth)")

    axes[1, 1].barh(
        map_positions - width / 2,
        [gridded[name]["crps_mm_day"] for name in MAP_METHODS],
        height=width,
        color="#F4A261",
        label="field",
    )
    axes[1, 1].barh(
        map_positions + width / 2,
        [gridded[name]["subgrid_anomaly_crps_mm_day"] for name in MAP_METHODS],
        height=width,
        color="#6A4C93",
        label="within 0.5°",
    )
    axes[1, 1].set_yticks(map_positions, map_labels)
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_xlabel("CRPS against CHIRPS (mm/day)")
    axes[1, 1].legend()
    axes[1, 1].set_title("E. Gridded product agreement")

    axes[1, 2].barh(
        map_positions - width / 2,
        [plot_value(structure[name]["cpc_pattern_r"]) for name in MAP_METHODS],
        height=width,
        color="#59A14F",
        label="CPC 0.5°",
    )
    axes[1, 2].barh(
        map_positions + width / 2,
        [plot_value(structure[name]["imerg_pattern_r"]) for name in MAP_METHODS],
        height=width,
        color="#EDC948",
        label="IMERG S04",
    )
    axes[1, 2].set_yticks(map_positions, map_labels)
    axes[1, 2].invert_yaxis()
    axes[1, 2].set_xlim(-0.2, 1.0)
    axes[1, 2].set_xlabel("mean daily correlation")
    axes[1, 2].legend()
    axes[1, 2].set_title("F. Conditioning/observation pattern agreement")

    figure.suptitle(
        "Corrected CPC-V3-SG/v4 DA pilot (short window; not configuration selection)",
        fontsize=14,
    )
    figure.savefig(output, dpi=170)
    plt.close(figure)


def plot_maps(
    output: Path,
    days: np.ndarray,
    grid: Grid,
    valid: np.ndarray,
    cpc: np.ndarray,
    chirps: np.ndarray,
    means: dict[str, np.ndarray],
    plot_days: int,
) -> None:
    count = min(len(days), plot_days)
    cpc_fine = np.repeat(np.repeat(cpc, 10, axis=-2), 10, axis=-1)
    extent = [grid.lon_min, grid.lon_max, grid.lat_min, grid.lat_max]
    figure, axes = plt.subplots(count, 6, figsize=(22, 3.5 * count), squeeze=False)
    for row in range(count):
        values = np.concatenate(
            [
                cpc_fine[row][valid],
                chirps[row][valid],
                *(means[name][row][valid] for name in MAP_METHODS),
            ]
        )
        vmax = max(1.0, float(np.nanpercentile(values, 98.0)))
        panels = [
            (cpc_fine[row], "CPC input (0.5°)"),
            (chirps[row], "CHIRPS training product"),
            (means["background"][row], "background"),
            (means["gauges_all"][row], "gauges: all BMD"),
            (means["imerg_only"][row], "IMERG S04 only"),
            (means["simultaneous_all"][row], "simultaneous: all BMD + IMERG"),
        ]
        for column, (field, title) in enumerate(panels):
            artist = axes[row, column].imshow(
                np.where(valid, field, np.nan),
                origin="lower",
                extent=extent,
                cmap="YlGnBu",
                vmin=0.0,
                vmax=vmax,
                aspect="auto",
            )
            axes[row, column].set_title(title)
            axes[row, column].set_xlabel("longitude")
            if column == 0:
                axes[row, column].set_ylabel(f"{days[row]}\nlatitude")
            figure.colorbar(artist, ax=axes[row, column], fraction=0.046, pad=0.02)
    figure.suptitle("Corrected v4 daily ensemble means", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(output, dpi=160)
    plt.close(figure)


def plot_subgrid_maps(
    output: Path,
    days: np.ndarray,
    grid: Grid,
    valid: np.ndarray,
    area: np.ndarray,
    factor: int,
    chirps: np.ndarray,
    means: dict[str, np.ndarray],
    plot_days: int,
) -> None:
    count = min(len(days), plot_days)
    target = block_anomaly(chirps, area, valid, factor)
    anomaly = {name: block_anomaly(field, area, valid, factor) for name, field in means.items()}
    extent = [grid.lon_min, grid.lon_max, grid.lat_min, grid.lat_max]
    figure, axes = plt.subplots(count, 6, figsize=(22, 3.5 * count), squeeze=False)
    for row in range(count):
        absolute = np.concatenate(
            [
                np.abs(target[row][valid & np.isfinite(target[row])]),
                *(np.abs(anomaly[name][row][valid]) for name in MAP_METHODS),
            ]
        )
        limit = max(0.5, float(np.nanpercentile(absolute, 98.0)))
        panels = [
            (target[row], "CHIRPS within-0.5° anomaly"),
            (anomaly["background"][row], "background anomaly"),
            (anomaly["gauges_all"][row], "gauges-all anomaly"),
            (anomaly["imerg_only"][row], "IMERG-only anomaly"),
            (anomaly["simultaneous_all"][row], "simultaneous-all anomaly"),
            (
                anomaly["simultaneous_all"][row] - anomaly["background"][row],
                "simultaneous − background anomaly",
            ),
        ]
        for column, (field, title) in enumerate(panels):
            artist = axes[row, column].imshow(
                np.where(valid, field, np.nan),
                origin="lower",
                extent=extent,
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
                aspect="auto",
            )
            axes[row, column].set_title(title)
            axes[row, column].set_xlabel("longitude")
            if column == 0:
                axes[row, column].set_ylabel(f"{days[row]}\nlatitude")
            figure.colorbar(artist, ax=axes[row, column], fraction=0.046, pad=0.02)
    figure.suptitle(
        "Corrected v4 structure below each area-weighted 0.5° block mean",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    target = zarr.open_group(args.target_store, mode="r")
    samples = zarr.open_group(args.sample_store, mode="r")
    if target.attrs.get("schema") != SUBGRID_SCHEMA or not target.attrs.get("complete", False):
        raise ValueError(f"target must be a completed {SUBGRID_SCHEMA} archive")
    if samples.attrs.get("schema") != "cpc_v3_hierarchical_samples_v3":
        raise ValueError("sample store was not written by the audited hierarchical writer")
    if not samples.attrs.get("complete", False) or not samples.attrs.get(
        "diagnostic_complete", False
    ):
        raise ValueError("sample diagnostic is incomplete")
    if samples.attrs.get("archive_uses_likelihood_hard_decoder") is not True:
        raise ValueError("sample store lacks a passing hard-decoder round-trip audit")
    for name in set(POINT_METHODS + MAP_METHODS):
        if name not in samples:
            raise ValueError(f"sample store is missing required method {name}")
    encoding = SubgridEncoding.from_mapping(target.attrs["subgrid_encoding"])
    sample_encoding = SubgridEncoding.from_mapping(samples.attrs["subgrid_encoding"])
    encoding.validate()
    sample_encoding.validate()
    if encoding_metadata(encoding) != encoding_metadata(sample_encoding):
        raise ValueError("target and samples use different v4 encodings")

    times = np.asarray(samples["time"][:], np.int64).astype("datetime64[ns]")
    if times.size == 0 or len(np.unique(times)) != len(times):
        raise ValueError("sample time axis must be non-empty and unique")
    if times.size > 1 and not np.all(np.diff(times.astype(np.int64)) > 0):
        raise ValueError("sample time axis must be strictly increasing")
    days = times.astype("datetime64[D]")
    lat = np.asarray(samples["lat"][:], np.float32)
    lon = np.asarray(samples["lon"][:], np.float32)
    if len(lat) < 2 or len(lon) < 2:
        raise ValueError("sample grid must contain at least two cells per dimension")
    resolution = float(np.median(np.diff(lon)))
    if not np.allclose(np.diff(lon), resolution, atol=1.0e-6) or not np.allclose(
        np.diff(lat), resolution, atol=1.0e-6
    ):
        raise ValueError("sample latitude/longitude coordinates are not one regular grid")
    grid = Grid(
        "v4_da_evaluation",
        float(lon[0] - resolution / 2.0),
        float(lat[0] - resolution / 2.0),
        len(lon),
        len(lat),
        resolution,
    )
    valid = np.asarray(samples["valid"][:], bool)
    area = np.asarray(samples["cell_area"][:], np.float32)
    chirps = np.asarray(samples["context_chirps_mm"][:], np.float32)
    cpc = np.asarray(samples["context_cpc_mm"][:], np.float32)
    imerg = np.asarray(samples["context_imerg_mm"][:], np.float32)
    station_lat = np.asarray(samples["station_lat"][:], np.float32)
    station_lon = np.asarray(samples["station_lon"][:], np.float32)
    observations = np.asarray(samples["station_value_mm"][:], np.float32)
    withheld = np.asarray(samples["station_withheld"][:], bool)
    assimilated = ~withheld
    if not withheld.any() or not assimilated.any():
        raise ValueError("diagnostic station roles are empty")

    fields = {
        name: np.asarray(samples[name][:], np.float32)
        for name in set(POINT_METHODS + MAP_METHODS)
    }
    predictions = {
        name: station_samples(field, grid, station_lat, station_lon)
        for name, field in fields.items()
    }
    withheld_metrics = {
        name: ensemble_station_metrics(
            predictions[name][:, :, withheld],
            observations[:, withheld],
            args.wet_threshold_mm,
        )
        for name in POINT_METHODS
    }
    empty_point_methods = [
        name for name, values in withheld_metrics.items() if values.get("n", 0) == 0
    ]
    if empty_point_methods:
        raise ValueError(
            f"no independently withheld gauge pairs for methods {empty_point_methods}"
        )
    assimilated_fit = {
        name: ensemble_station_metrics(
            predictions[name][:, :, assimilated],
            observations[:, assimilated],
            args.wet_threshold_mm,
        )
        for name in POINT_METHODS
    }
    all_station_fit = {
        name: ensemble_station_metrics(
            predictions[name], observations, args.wet_threshold_mm
        )
        for name in ("gauges_all", "simultaneous_all")
    }

    means = {name: field.mean(axis=1) for name, field in fields.items() if name in MAP_METHODS}
    target_anomaly = block_anomaly(chirps, area, valid, encoding.factor)
    candidate_anomaly = {
        name: block_anomaly(field, area, valid, encoding.factor)
        for name, field in means.items()
    }
    chirps_full = {
        name: mean_daily_correlation(field, chirps, valid)
        for name, field in means.items()
    }
    chirps_subgrid = {
        name: mean_daily_correlation(field, target_anomaly, valid)
        for name, field in candidate_anomaly.items()
    }
    cpc_corr = coarse_product_correlation(
        means, cpc, area, valid, encoding.factor
    )
    imerg_crop = tuple(int(value) for value in samples.attrs["imerg_canvas_crop"])
    imerg_corr = imerg_product_correlation(
        means,
        imerg,
        area,
        valid,
        int(samples.attrs["imerg_factor"]),
        imerg_crop,
    )
    structure = {
        name: {
            "chirps_mean_pattern_r": chirps_full[name],
            "chirps_subgrid_pattern_r": chirps_subgrid[name],
            "cpc_pattern_r": cpc_corr[name],
            "imerg_pattern_r": imerg_corr[name],
        }
        for name in MAP_METHODS
    }

    gridded_payload = json.loads(Path(args.gridded_json).read_text())
    if gridded_payload.get("schema") != "cpc_v3_subgrid_evaluation_v1":
        raise ValueError("--gridded-json does not use the script-58 evaluation schema")
    if gridded_payload.get("sample_store") != args.sample_store:
        raise ValueError("--gridded-json was generated from a different sample store")
    if gridded_payload.get("target_store") != args.target_store:
        raise ValueError("--gridded-json was generated from a different target store")
    gridded_all = gridded_payload["results"]
    missing_gridded = [name for name in MAP_METHODS if name not in gridded_all]
    if missing_gridded:
        raise ValueError(f"--gridded-json lacks methods {missing_gridded}")
    gridded = {name: gridded_all[name] for name in MAP_METHODS}
    authority = gridded_all.get("authority", {})
    diagnostic_summary = summarize_sampler_diagnostics(
        dict(samples.attrs.get("sampler_diagnostics", {})), POINT_METHODS + MAP_METHODS
    )
    payload = finite_or_none(
        {
            "schema": "cpc_v3_subgrid_v4_da_pilot_evaluation_v1",
            "warning": (
                "Short-window diagnostic only. DA settings are preliminary; CHIRPS, "
                "CPC and IMERG scores are product agreement, not independent truth."
            ),
            "sample_store": args.sample_store,
            "target_store": args.target_store,
            "dates": [str(days[0]), str(days[-1])],
            "members": int(samples["member"].shape[0]),
            "withheld_gauges": withheld_metrics,
            "assimilated_fold_fit": assimilated_fit,
            "all_station_fit_in_sample": all_station_fit,
            "spatial_product_agreement": structure,
            "gridded_chirps_agreement": gridded,
            "physical_authority_vs_background": authority,
            "sampler_terminal_and_decoder_screen": diagnostic_summary,
        }
    )

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "v4_da_test_metrics.json"
    markdown_path = output / "v4_da_test_metrics.md"
    matrix_path = output / "v4_da_test_matrix.png"
    maps_path = output / "v4_da_test_daily_maps.png"
    subgrid_path = output / "v4_da_test_subgrid_maps.png"
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")

    lines = [
        "# Corrected CPC-V3-SG/v4 DA pilot",
        "",
        f"- Dates: **{days[0]} to {days[-1]}**",
        f"- Ensemble: **{payload['members']} members**",
        "- This is a short diagnostic, not configuration selection or confirmation.",
        "- Gauge skill below is independently withheld; spatial maps use separate all-gauge arms.",
        "- CHIRPS/CPC/IMERG metrics are product agreement, not independent truth.",
        "",
        "## Independently withheld BMD gauges",
        "",
        "| Method | CRPS | RMSE | Bias | r | Cov90 | dry/wet MAE |",
        "|:--|--:|--:|--:|--:|--:|:--|",
    ]
    for name in POINT_METHODS:
        value = payload["withheld_gauges"][name]
        dry_text = metric_text(value["dry_mae_mm_day"], 2)
        wet_text = metric_text(value["wet_mae_mm_day"], 2)
        lines.append(
            f"| `{name}` | {value['crps_mm_day']:.3f} | "
            f"{value['rmse_mm_day']:.3f} | {value['bias_mm_day']:+.3f} | "
            f"{metric_text(value['correlation'])} | "
            f"{value['coverage90']:.2f} | "
            f"{dry_text}/{wet_text} |"
        )
    lines += [
        "",
        "## Spatial product agreement",
        "",
        "| Method | CHIRPS field r | CHIRPS subgrid r | CPC r | IMERG r | field CRPS | subgrid CRPS |",
        "|:--|--:|--:|--:|--:|--:|--:|",
    ]
    for name in MAP_METHODS:
        value = payload["spatial_product_agreement"][name]
        grid_value = payload["gridded_chirps_agreement"][name]
        lines.append(
            f"| `{name}` | {metric_text(value['chirps_mean_pattern_r'])} | "
            f"{metric_text(value['chirps_subgrid_pattern_r'])} | "
            f"{metric_text(value['cpc_pattern_r'])} | "
            f"{metric_text(value['imerg_pattern_r'])} | "
            f"{grid_value['crps_mm_day']:.3f} | "
            f"{grid_value['subgrid_anomaly_crps_mm_day']:.3f} |"
        )
    lines += [
        "",
        "## Terminal likelihood and decoder screen",
        "",
        "| Method | O−A bias (σ) | O−A RMSE (σ) | max |O−A| (σ) | soft/hard pass | decoder fallback max | clipped allocation max |",
        "|:--|--:|--:|--:|:--:|--:|--:|",
    ]
    for name in POINT_METHODS + tuple(
        value for value in MAP_METHODS if value not in POINT_METHODS
    ):
        value = payload["sampler_terminal_and_decoder_screen"][name]
        passed = value["soft_hard_bounds_all_pass"]
        pass_text = "—" if passed is None else ("yes" if passed else "NO")
        lines.append(
            f"| `{name}` | {metric_text(value['terminal_oa_bias_sigma_mean'])} | "
            f"{metric_text(value['terminal_oa_rmse_sigma_mean'])} | "
            f"{metric_text(value['terminal_oa_max_abs_sigma'])} | {pass_text} | "
            f"{metric_text(value['decoder_fallback_fraction_max'])} | "
            f"{metric_text(value['decoder_clipped_intensity_fraction_max'])} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n")
    plot_matrix(payload, matrix_path)
    plot_maps(maps_path, days, grid, valid, cpc, chirps, means, args.plot_days)
    plot_subgrid_maps(
        subgrid_path,
        days,
        grid,
        valid,
        area,
        encoding.factor,
        chirps,
        means,
        args.plot_days,
    )
    print("\n".join(lines))
    for path in (json_path, markdown_path, matrix_path, maps_path, subgrid_path):
        print(f"[done] wrote {path}")


if __name__ == "__main__":
    main()
