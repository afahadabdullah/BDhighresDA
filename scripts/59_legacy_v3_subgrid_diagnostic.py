#!/usr/bin/env python3
"""Five-day visual diagnostic for the completed, pre-v4 V3-SG checkpoint.

This script exists only to inspect an already completed legacy experiment.  It
requires the old ``cpc_v3_subgrid_v3`` target/checkpoint schema, writes into a
clearly labelled legacy directory, and is intentionally rejected by the v4
evaluation contract.  It is not a way to promote the old model to a corrected
result.

The two matched-noise arms are:

* ``background``: unguided joint V3-SG samples;
* ``gauges_only``: the same initial noise guided by assimilated BMD gauges.

A deterministic spatial holdout is never assimilated and is scored separately.
CHIRPS is plotted as the training product, not asserted to be truth.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bdhires.bmd import spread_holdout  # noqa: E402
from bdhires.da import (  # noqa: E402
    BilinearObsOperator,
    GuidanceConfig,
    HierarchicalObservations,
    HierarchicalSamplerConfig,
    perturb_observations,
    sample_hierarchical,
)
from bdhires.data import (  # noqa: E402
    SubgridEncoding,
    aligned_production_canvas,
    area_weighted_block_mean,
    encoding_metadata,
    load_stations,
)
from bdhires.grids import BD_CPC, WIDE_CPC, Grid  # noqa: E402
from bdhires.models import (  # noqa: E402
    AllocationFlow,
    CoarseHurdleFlow,
    CoupledSubgridFlow,
    HierarchicalState,
    select_weights,
)
from bdhires.zarr_output import write_hierarchical_sample_zarr  # noqa: E402


LEGACY_SCHEMA = "cpc_v3_subgrid_v3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target-store",
        default="data/processed/cpc_v3_subgrid/wide_cpc.zarr",
    )
    parser.add_argument(
        "--checkpoint",
        default="runs/prior_h100_cpc_v3_subgrid/joint/best.pt",
    )
    parser.add_argument("--stations", required=True, help="canonical BMD daily CSV")
    parser.add_argument("--start", default="2022-05-01")
    parser.add_argument("--end", default="2022-05-05")
    parser.add_argument(
        "--background-day-offset",
        type=int,
        default=-1,
        help="condition-date offset relative to the BMD observation date",
    )
    parser.add_argument("--members", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=25)
    parser.add_argument("--canvas", type=int, default=160)
    parser.add_argument("--withhold", type=float, default=0.20)
    parser.add_argument("--min-coverage", type=float, default=0.80)
    parser.add_argument("--gauge-sigma-mm", type=float, default=3.0)
    parser.add_argument("--guidance-gamma", type=float, default=1.0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--guidance-clip-norm", type=float, default=100.0)
    parser.add_argument("--huber-delta", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=202205)
    parser.add_argument(
        "--out-dir",
        default="data/processed/v3_legacy_diagnostic/may2022_5day",
    )
    return parser.parse_args()


def _require_legacy_contract(root, checkpoint: dict) -> SubgridEncoding:
    target_schema = root.attrs.get("schema")
    checkpoint_schema = checkpoint.get("schema")
    if target_schema != LEGACY_SCHEMA or checkpoint_schema != LEGACY_SCHEMA:
        raise ValueError(
            "this diagnostic accepts only the completed legacy v3 target and "
            f"checkpoint; got target={target_schema!r}, checkpoint={checkpoint_schema!r}"
        )
    if not root.attrs.get("complete", False):
        raise ValueError("legacy target store is not marked complete")
    if checkpoint.get("stage") != "joint":
        raise ValueError("legacy checkpoint must be a completed joint-stage checkpoint")
    if "subgrid_encoding" not in root.attrs or "subgrid_encoding" not in checkpoint:
        raise ValueError("legacy target/checkpoint lacks frozen subgrid encoding metadata")
    target_encoding = SubgridEncoding.from_mapping(root.attrs["subgrid_encoding"])
    checkpoint_encoding = SubgridEncoding.from_mapping(checkpoint["subgrid_encoding"])
    target_encoding.validate()
    if encoding_metadata(target_encoding) != encoding_metadata(checkpoint_encoding):
        raise ValueError("legacy target and checkpoint use different subgrid encodings")
    return target_encoding


def _build_joint_model(checkpoint: dict, root, device: torch.device):
    config = checkpoint.get("config")
    if not isinstance(config, dict) or config.get("stage") != "joint":
        raise ValueError("legacy checkpoint lacks its resolved joint training config")
    model_config = config["model"]
    crop = int(config["data"]["crop"])
    factor = int(config["data"].get("factor", 10))
    coarse_channels = int(root["coarse_cond"].shape[1])
    fine_channels = int(root["fine_cond"].shape[1])
    coarse = CoarseHurdleFlow(
        coarse_channels,
        image_size=crop // factor,
        **model_config["coarse"],
    )
    allocation = AllocationFlow(
        fine_channels,
        image_size=crop,
        **model_config["allocation"],
    )
    model = CoupledSubgridFlow(
        coarse,
        allocation,
        clean_context_probability=float(
            config["train"].get("clean_context_probability", 0.0)
        ),
    )
    model.load_state_dict(select_weights(checkpoint), strict=True)
    return model.to(device).eval(), config


def _date_indices(root, start: str, end: str, offset: int):
    observation_days = np.arange(
        np.datetime64(start, "D"), np.datetime64(end, "D") + np.timedelta64(1, "D")
    )
    if observation_days.size == 0:
        raise ValueError("requested diagnostic period is empty")
    source_days = np.asarray(root["time"][:], np.int64).astype("datetime64[ns]").astype(
        "datetime64[D]"
    )
    if len(np.unique(source_days)) != len(source_days):
        raise ValueError("legacy target store contains duplicate dates")
    lookup = {day: index for index, day in enumerate(source_days)}
    missing_observation = [day for day in observation_days if day not in lookup]
    condition_days = observation_days + np.timedelta64(offset, "D")
    missing_condition = [day for day in condition_days if day not in lookup]
    if missing_observation or missing_condition:
        raise ValueError(
            "legacy target lacks requested dates: observation="
            f"{missing_observation}, conditioning={missing_condition}"
        )
    observation_index = np.asarray([lookup[day] for day in observation_days], np.int64)
    condition_index = np.asarray([lookup[day] for day in condition_days], np.int64)
    return observation_days, condition_days, observation_index, condition_index


def _canvas_grid(canvas_slice: tuple[slice, slice]) -> Grid:
    rows, columns = canvas_slice
    return Grid(
        name="legacy_v3_bd_canvas",
        lon_min=WIDE_CPC.lon_min + int(columns.start) * WIDE_CPC.res,
        lat_min=WIDE_CPC.lat_min + int(rows.start) * WIDE_CPC.res,
        nlon=int(columns.stop) - int(columns.start),
        nlat=int(rows.stop) - int(rows.start),
        res=WIDE_CPC.res,
    )


def _initial_noise(
    members: int,
    fine_shape: tuple[int, int],
    factor: int,
    seed: int,
    device: torch.device,
) -> HierarchicalState:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    height, width = fine_shape
    return HierarchicalState(
        torch.randn(
            members, 2, height // factor, width // factor,
            generator=generator, device=device,
        ),
        torch.randn(
            members, 2, height, width, generator=generator, device=device,
        ),
    )


def _clone_state(state: HierarchicalState) -> HierarchicalState:
    return HierarchicalState(state.coarse.clone(), state.allocation.clone())


def _append_array(group, name: str, values: np.ndarray, dimensions: tuple[str, ...], chunks=None):
    values = np.asarray(values)
    array = group.create_dataset(
        name,
        data=values,
        shape=values.shape,
        chunks=chunks or values.shape,
        fill_value=None,
        overwrite=False,
    )
    array.attrs["_ARRAY_DIMENSIONS"] = list(dimensions)
    return array


def _station_samples(
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


def _ensemble_metrics(prediction: np.ndarray, observation: np.ndarray) -> dict:
    # prediction: time, member, station; observation: time, station
    valid = np.isfinite(observation) & np.all(np.isfinite(prediction), axis=1)
    if not valid.any():
        return {"n": 0}
    ensemble = prediction.transpose(1, 0, 2)[:, valid]
    truth = observation[valid]
    mean = ensemble.mean(axis=0)
    difference = mean - truth
    crps_first = np.mean(np.abs(ensemble - truth[None]), axis=0)
    crps_second = 0.5 * np.mean(
        np.abs(ensemble[:, None] - ensemble[None, :]), axis=(0, 1)
    )
    correlation = (
        float(np.corrcoef(mean, truth)[0, 1])
        if mean.std() > 0.0 and truth.std() > 0.0
        else None
    )
    low, high = np.quantile(ensemble, [0.05, 0.95], axis=0)
    return {
        "n": int(valid.sum()),
        "crps_mm": float(np.mean(crps_first - crps_second)),
        "rmse_mm": float(np.sqrt(np.mean(difference**2))),
        "mae_mm": float(np.mean(np.abs(difference))),
        "bias_mm": float(np.mean(difference)),
        "correlation": correlation,
        "coverage90": float(np.mean((truth >= low) & (truth <= high))),
    }


def _subgrid_component(
    field: np.ndarray,
    area: np.ndarray,
    valid: np.ndarray,
    factor: int,
) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(field, np.float32))[:, None]
    coarse, retained, _ = area_weighted_block_mean(
        tensor,
        torch.from_numpy(area),
        torch.from_numpy(valid),
        factor=factor,
        valid_area_threshold=0.0,
    )
    upscaled = coarse.repeat_interleave(factor, -2).repeat_interleave(factor, -1)
    retained_fine = retained.repeat_interleave(factor, -2).repeat_interleave(factor, -1)
    anomaly = tensor - upscaled
    anomaly = torch.where(retained_fine, anomaly, torch.nan)
    return anomaly[:, 0].numpy()


def _pattern_metrics(
    fields: dict[str, np.ndarray],
    chirps: np.ndarray,
    area: np.ndarray,
    valid: np.ndarray,
    factor: int,
) -> dict:
    target = _subgrid_component(chirps, area, valid, factor)
    out = {}
    for method, ensemble in fields.items():
        candidate = _subgrid_component(ensemble.mean(axis=1), area, valid, factor)
        daily = []
        for index in range(len(chirps)):
            keep = valid & np.isfinite(target[index]) & np.isfinite(candidate[index])
            left, right = candidate[index][keep], target[index][keep]
            correlation = (
                float(np.corrcoef(left, right)[0, 1])
                if left.std() > 0.0 and right.std() > 0.0
                else None
            )
            daily.append(
                {
                    "correlation_with_chirps": correlation,
                    "sd_ratio_to_chirps": (
                        float(left.std() / right.std()) if right.std() > 0.0 else None
                    ),
                    "rmse_mm": float(np.sqrt(np.mean((left - right) ** 2))),
                }
            )
        correlations = [
            entry["correlation_with_chirps"] for entry in daily
            if entry["correlation_with_chirps"] is not None
        ]
        out[method] = {
            "interpretation": "pattern agreement with CHIRPS; CHIRPS is not gauge truth",
            "daily": daily,
            "mean_correlation_with_chirps": (
                float(np.mean(correlations)) if correlations else None
            ),
        }
    return out


def _plot_daily(
    output: Path,
    days: np.ndarray,
    grid: Grid,
    valid: np.ndarray,
    cpc: np.ndarray,
    chirps: np.ndarray,
    fields: dict[str, np.ndarray],
    station_lat: np.ndarray,
    station_lon: np.ndarray,
    assimilated: np.ndarray,
    withheld: np.ndarray,
) -> None:
    background = fields["background"].mean(axis=1)
    analysis = fields["gauges_only"].mean(axis=1)
    cpc_fine = np.repeat(np.repeat(cpc, 10, axis=-2), 10, axis=-1)
    extent = [
        grid.lon_min, grid.lon_max, grid.lat_min, grid.lat_max,
    ]
    figure, axes = plt.subplots(len(days), 5, figsize=(19, 3.4 * len(days)), squeeze=False)
    for row, day in enumerate(days):
        comparison = np.concatenate(
            [
                cpc_fine[row][valid], chirps[row][valid],
                background[row][valid], analysis[row][valid],
            ]
        )
        vmax = max(1.0, float(np.nanpercentile(comparison, 98.0)))
        panels = [
            (cpc_fine[row], "CPC input (0.5 degree)"),
            (chirps[row], "CHIRPS training product"),
            (background[row], "legacy background mean"),
            (analysis[row], "legacy gauges-only mean"),
        ]
        for column, (values, title) in enumerate(panels):
            image = np.where(valid, values, np.nan)
            artist = axes[row, column].imshow(
                image, origin="lower", extent=extent, cmap="YlGnBu", vmin=0.0, vmax=vmax
            )
            axes[row, column].set_title(title)
            if column >= 2:
                axes[row, column].scatter(
                    station_lon[assimilated], station_lat[assimilated],
                    s=12, facecolors="none", edgecolors="black", linewidths=0.7,
                    label="assimilated" if row == 0 and column == 3 else None,
                )
                axes[row, column].scatter(
                    station_lon[withheld], station_lat[withheld],
                    s=18, marker="x", color="magenta", linewidths=0.8,
                    label="withheld" if row == 0 and column == 3 else None,
                )
            figure.colorbar(artist, ax=axes[row, column], fraction=0.046, pad=0.02)
        increment = np.where(valid, analysis[row] - background[row], np.nan)
        limit = max(1.0, float(np.nanpercentile(np.abs(increment[valid]), 98.0)))
        artist = axes[row, 4].imshow(
            increment, origin="lower", extent=extent, cmap="RdBu", vmin=-limit, vmax=limit
        )
        axes[row, 4].set_title("DA increment")
        axes[row, 4].scatter(
            station_lon[assimilated], station_lat[assimilated],
            s=12, facecolors="none", edgecolors="black", linewidths=0.7,
        )
        axes[row, 4].scatter(
            station_lon[withheld], station_lat[withheld],
            s=18, marker="x", color="magenta", linewidths=0.8,
        )
        figure.colorbar(artist, ax=axes[row, 4], fraction=0.046, pad=0.02)
        axes[row, 0].set_ylabel(str(day))
        for axis in axes[row]:
            axis.set_xlabel("longitude")
            axis.set_ylabel(axis.get_ylabel() or "latitude")
    axes[0, 3].legend(loc="upper right", fontsize=8)
    figure.suptitle(
        "Legacy pre-v4 V3-SG diagnostic only: matched background and BMD gauge DA",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_subgrid(
    output: Path,
    days: np.ndarray,
    grid: Grid,
    valid: np.ndarray,
    area: np.ndarray,
    factor: int,
    chirps: np.ndarray,
    fields: dict[str, np.ndarray],
) -> None:
    target = _subgrid_component(chirps, area, valid, factor)
    background = _subgrid_component(fields["background"].mean(axis=1), area, valid, factor)
    analysis = _subgrid_component(fields["gauges_only"].mean(axis=1), area, valid, factor)
    increment = analysis - background
    spread = fields["gauges_only"].std(axis=1, ddof=1)
    extent = [grid.lon_min, grid.lon_max, grid.lat_min, grid.lat_max]
    figure, axes = plt.subplots(len(days), 5, figsize=(19, 3.4 * len(days)), squeeze=False)
    for row, day in enumerate(days):
        combined = np.concatenate(
            [
                np.abs(target[row][valid & np.isfinite(target[row])]),
                np.abs(background[row][valid & np.isfinite(background[row])]),
                np.abs(analysis[row][valid & np.isfinite(analysis[row])]),
            ]
        )
        limit = max(0.5, float(np.nanpercentile(combined, 98.0)))
        panels = [
            (target[row], "CHIRPS subgrid anomaly"),
            (background[row], "background subgrid anomaly"),
            (analysis[row], "analysis subgrid anomaly"),
            (increment[row], "DA change in subgrid anomaly"),
        ]
        for column, (values, title) in enumerate(panels):
            artist = axes[row, column].imshow(
                np.where(valid, values, np.nan),
                origin="lower", extent=extent, cmap="RdBu", vmin=-limit, vmax=limit,
            )
            axes[row, column].set_title(title)
            figure.colorbar(artist, ax=axes[row, column], fraction=0.046, pad=0.02)
        spread_limit = max(0.5, float(np.nanpercentile(spread[row][valid], 98.0)))
        artist = axes[row, 4].imshow(
            np.where(valid, spread[row], np.nan),
            origin="lower", extent=extent, cmap="magma", vmin=0.0, vmax=spread_limit,
        )
        axes[row, 4].set_title("analysis ensemble SD")
        figure.colorbar(artist, ax=axes[row, 4], fraction=0.046, pad=0.02)
        axes[row, 0].set_ylabel(str(day))
        for axis in axes[row]:
            axis.set_xlabel("longitude")
            axis.set_ylabel(axis.get_ylabel() or "latitude")
    figure.suptitle(
        "Legacy pre-v4 V3-SG: structure below each 0.5-degree block mean",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.members < 2:
        raise ValueError("--members must be at least 2 for an ensemble diagnostic")
    if args.n_steps <= 0:
        raise ValueError("--n-steps must be positive")
    if not 0.0 < args.withhold < 1.0:
        raise ValueError("--withhold must lie strictly between 0 and 1")
    if args.gauge_sigma_mm <= 0.0:
        raise ValueError("--gauge-sigma-mm must be positive")

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "legacy_v3_may2022_5day.zarr"
    report_path = output_dir / "legacy_v3_may2022_5day.json"
    figure_path = output_dir / "legacy_v3_may2022_5day.png"
    subgrid_figure_path = output_dir / "legacy_v3_may2022_5day_subgrid.png"
    for path in (archive_path, report_path, figure_path, subgrid_figure_path):
        if path.exists():
            raise FileExistsError(
                f"refusing to overwrite existing diagnostic output {path}; "
                "choose a different --out-dir"
            )

    root = zarr.open_group(args.target_store, mode="r")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    encoding = _require_legacy_contract(root, checkpoint)
    if encoding.factor != 10:
        raise ValueError(f"legacy diagnostic expects factor 10, got {encoding.factor}")

    days, condition_days, target_index, condition_index = _date_indices(
        root, args.start, args.end, args.background_day_offset
    )
    canvas_slice, core_slice = aligned_production_canvas(
        WIDE_CPC, BD_CPC, canvas=args.canvas, factor=encoding.factor
    )
    rows, columns = canvas_slice
    coarse_slice = (
        slice(rows.start // encoding.factor, rows.stop // encoding.factor),
        slice(columns.start // encoding.factor, columns.stop // encoding.factor),
    )
    grid = _canvas_grid(canvas_slice)
    valid = np.asarray(root["fine_valid"][rows, columns], bool)
    coarse_valid = np.asarray(root["coarse_valid"][coarse_slice], bool)
    area = np.asarray(root["cell_area"][rows, columns], np.float32)
    lat = np.asarray(root["lat"][rows], np.float32)
    lon = np.asarray(root["lon"][columns], np.float32)
    chirps = np.asarray(root["fine_mm"].oindex[target_index, rows, columns], np.float32)

    stations, gauge_mm = load_stations(
        args.stations, days, grid=grid, min_coverage=args.min_coverage
    )
    if len(stations) < 5:
        raise ValueError(f"only {len(stations)} BMD stations remain after filtering")
    n_withheld = max(1, min(len(stations) - 1, int(round(args.withhold * len(stations)))))
    withheld = spread_holdout(stations.lat, stations.lon, n_withheld)
    assimilated = np.setdiff1d(np.arange(len(stations)), withheld)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("the 160-cell joint diagnostic requires an allocated GPU")
    model, training_config = _build_joint_model(checkpoint, root, device)
    sampler_config = HierarchicalSamplerConfig(
        n_steps=args.n_steps,
        heun=True,
        n_corrections=0,
        occurrence_temperature=float(
            training_config.get("sampling", {}).get("occurrence_temperature", 1.0)
        ),
    )
    guidance = GuidanceConfig(
        gamma=args.guidance_gamma,
        scale=args.guidance_scale,
        t_start=0.10,
        t_end=0.999,
        clip_norm=args.guidance_clip_norm,
        huber_delta=args.huber_delta,
    )
    gauge_operator = BilinearObsOperator(
        grid, stations.lat[assimilated], stations.lon[assimilated]
    ).to(device)
    gauge_variance = torch.full(
        (len(assimilated),), args.gauge_sigma_mm**2,
        dtype=torch.float32, device=device,
    )

    methods = ("background", "gauges_only")
    fine_shape = grid.shape
    coarse_shape = (grid.nlat // encoding.factor, grid.nlon // encoding.factor)
    fields = {
        name: np.empty((len(days), args.members, *fine_shape), np.float32)
        for name in methods
    }
    coarse_states = {
        name: np.empty((len(days), args.members, 2, *coarse_shape), np.float32)
        for name in methods
    }
    allocation_states = {
        name: np.empty((len(days), args.members, 2, *fine_shape), np.float32)
        for name in methods
    }
    diagnostics = {name: {"daily": []} for name in methods}
    cpc = np.empty((len(days), *coarse_shape), np.float32)

    coarse_names = list(root.attrs["coarse_cond_channels"])
    if "cpc_precip" not in coarse_names:
        raise ValueError("legacy target has no cpc_precip conditioning channel")
    cpc_channel = coarse_names.index("cpc_precip")
    coarse_mean = np.asarray(root.attrs["coarse_cond_mean"], np.float32)
    coarse_std = np.asarray(root.attrs["coarse_cond_std"], np.float32)

    for position, (day, source_index) in enumerate(zip(days, condition_index)):
        coarse_cond_np = np.asarray(
            root["coarse_cond"][int(source_index), :, coarse_slice[0], coarse_slice[1]],
            np.float32,
        )
        fine_cond_np = np.asarray(
            root["fine_cond"][int(source_index), :, rows, columns], np.float32
        )
        cpc[position] = (
            coarse_cond_np[cpc_channel] * coarse_std[cpc_channel]
            + coarse_mean[cpc_channel]
        )
        coarse_cond = torch.from_numpy(coarse_cond_np[None]).to(device)
        fine_cond = torch.from_numpy(fine_cond_np[None]).to(device)
        initial = _initial_noise(
            args.members, fine_shape, encoding.factor,
            args.seed + int((day - days[0]) / np.timedelta64(1, "D")), device,
        )
        background = sample_hierarchical(
            model,
            coarse_cond,
            fine_cond,
            (args.members, 2, *coarse_shape),
            (args.members, 2, *fine_shape),
            torch.from_numpy(coarse_valid).to(device),
            torch.from_numpy(valid).to(device),
            torch.from_numpy(area).to(device),
            encoding,
            config=sampler_config,
            initial_noise=_clone_state(initial),
        )

        day_observation = gauge_mm[position, assimilated]
        perturbed = perturb_observations(
            day_observation,
            np.full(len(assimilated), args.gauge_sigma_mm**2, np.float32),
            args.members,
            seed=args.seed + 1_000_000 + position,
        ).astype(np.float32)
        observations = HierarchicalObservations(
            gauge_operator,
            torch.from_numpy(perturbed[:, None]).to(device),
            gauge_variance,
            guidance,
        )
        analysis = sample_hierarchical(
            model,
            coarse_cond,
            fine_cond,
            (args.members, 2, *coarse_shape),
            (args.members, 2, *fine_shape),
            torch.from_numpy(coarse_valid).to(device),
            torch.from_numpy(valid).to(device),
            torch.from_numpy(area).to(device),
            encoding,
            observations=observations,
            config=sampler_config,
            initial_noise=_clone_state(initial),
        )
        for name, sample in (("background", background), ("gauges_only", analysis)):
            fields[name][position] = sample.precipitation[:, 0].cpu().numpy()
            coarse_states[name][position] = sample.state.coarse.cpu().numpy()
            allocation_states[name][position] = sample.state.allocation.cpu().numpy()
            diagnostics[name]["daily"].append(sample.diagnostics)
        print(
            f"[legacy-v3] {day}: background/analysis complete "
            f"({args.members} members, {args.n_steps} Heun steps)",
            flush=True,
        )

    method_specs = {
        "background": {
            "role": "legacy pre-v4 unguided diagnostic",
            "matched_noise": True,
        },
        "gauges_only": {
            "role": "legacy pre-v4 BMD gauge-guided diagnostic",
            "tuning_status": "preliminary, not selected for V3 physical-space DA",
            "matched_noise": True,
            "gauge_sigma_mm": args.gauge_sigma_mm,
            "guidance_gamma": args.guidance_gamma,
            "guidance_scale": args.guidance_scale,
            "guidance_clip_norm": args.guidance_clip_norm,
            "huber_delta": args.huber_delta,
        },
    }
    write_hierarchical_sample_zarr(
        archive_path,
        fields=fields,
        coarse_states=coarse_states,
        allocation_states=allocation_states,
        selected_times=days,
        lat=lat,
        lon=lon,
        valid=valid,
        coarse_valid=coarse_valid,
        cell_area=area,
        encoding=encoding,
        diagnostics=diagnostics,
        method_specs=method_specs,
        target_crop=(rows.start, rows.stop, columns.start, columns.stop),
    )
    archive = zarr.open_group(archive_path, mode="a")
    archive.attrs.update(
        diagnostic_complete=False,
        diagnostic_only=True,
        legacy_pre_v4=True,
        source_target_schema=LEGACY_SCHEMA,
        source_checkpoint_schema=LEGACY_SCHEMA,
        source_checkpoint=str(args.checkpoint),
        source_target_store=str(args.target_store),
        condition_day_offset=int(args.background_day_offset),
        evaluator_warning=(
            "Do not pool with v4. This model was trained before the corrected "
            "coarse-occurrence and hurdle-loss contracts."
        ),
    )
    _append_array(
        archive, "context_chirps_mm", chirps,
        ("time", "lat", "lon"), chunks=(1, *fine_shape),
    )
    _append_array(
        archive, "context_cpc_mm", cpc,
        ("time", "coarse_lat", "coarse_lon"), chunks=(1, *coarse_shape),
    )
    _append_array(
        archive, "station_lat", stations.lat.astype(np.float32), ("station",)
    )
    _append_array(
        archive, "station_lon", stations.lon.astype(np.float32), ("station",)
    )
    _append_array(
        archive, "station_value_mm", gauge_mm.astype(np.float32),
        ("time", "station"),
    )
    station_role = np.zeros(len(stations), np.int8)
    station_role[assimilated] = 1
    _append_array(archive, "station_role", station_role, ("station",))
    # Publish the appended context atomically at the metadata level.  If a
    # later score/plot fails, consolidated readers see diagnostic_complete=False.
    zarr.consolidate_metadata(str(archive_path))

    station_predictions = {
        method: _station_samples(values, grid, stations.lat, stations.lon)
        for method, values in fields.items()
    }
    gauge_scores = {}
    for method, prediction in station_predictions.items():
        gauge_scores[method] = {
            "assimilated": _ensemble_metrics(
                prediction[:, :, assimilated], gauge_mm[:, assimilated]
            ),
            "withheld": _ensemble_metrics(
                prediction[:, :, withheld], gauge_mm[:, withheld]
            ),
        }
    pattern = _pattern_metrics(fields, chirps, area, valid, encoding.factor)
    _plot_daily(
        figure_path, days, grid, valid, cpc, chirps, fields,
        stations.lat, stations.lon, assimilated, withheld,
    )
    _plot_subgrid(
        subgrid_figure_path, days, grid, valid, area, encoding.factor,
        chirps, fields,
    )
    report = {
        "status": "diagnostic_only_legacy_pre_v4",
        "warning": (
            "The old checkpoint is scientifically superseded. These maps are useful "
            "for visual diagnosis only and must not be pooled with corrected v4. "
            "The physical-space gauge likelihood is a preliminary diagnostic, not "
            "a selected V3 DA configuration."
        ),
        "checkpoint": str(args.checkpoint),
        "target_store": str(args.target_store),
        "target_schema": LEGACY_SCHEMA,
        "dates": [str(day) for day in days],
        "condition_dates": [str(day) for day in condition_days],
        "members": args.members,
        "n_steps": args.n_steps,
        "canvas": {
            "grid": grid.name,
            "shape": list(grid.shape),
            "wide_crop": [rows.start, rows.stop, columns.start, columns.stop],
            "bd_cpc_core_inside_canvas": [
                core_slice[0].start, core_slice[0].stop,
                core_slice[1].start, core_slice[1].stop,
            ],
        },
        "stations": {
            "total": len(stations),
            "assimilated": int(len(assimilated)),
            "withheld": int(len(withheld)),
            "holdout_design": (
                "deterministic farthest-point spread_holdout; intentionally "
                "adversarial and used here only as a small independent diagnostic"
            ),
            "assimilated_ids": [str(stations.ids[index]) for index in assimilated],
            "withheld_ids": [str(stations.ids[index]) for index in withheld],
        },
        "gauge_scores": gauge_scores,
        "subgrid_pattern_agreement": pattern,
        "outputs": {
            "archive": str(archive_path),
            "figure": str(figure_path),
            "subgrid_figure": str(subgrid_figure_path),
            "report": str(report_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    archive.attrs["diagnostic_complete"] = True
    zarr.consolidate_metadata(str(archive_path))
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
