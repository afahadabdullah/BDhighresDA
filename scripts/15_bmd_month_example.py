#!/usr/bin/env python
"""One-month real-BMD gauge assimilation and withheld-station evaluation.

This is a process-validation experiment, not a final independent skill claim.
The default May 2018 period is inside the CPC checkpoint's training years, but
the withheld BMD gauges are independent observations and therefore reveal
whether real gauge ingestion, guidance, and station-space verification work.

IMERG is intentionally excluded from this first real-data gate.  It isolates
the BMD contribution and avoids pretending that the current packed Zarr store
contains a real, bias-corrected IMERG observation field.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bdhires.bmd import spread_holdout  # noqa: E402
from bdhires.da import (  # noqa: E402
    BilinearObsOperator,
    GuidanceConfig,
    PhysicalBilinearObsOperator,
    SamplerConfig,
    build_R,
    perturb_observations,
)
from bdhires.da.sampler import assimilate as run_assim  # noqa: E402
from bdhires.data import DatasetConfig, PrecipDataset, load_stations  # noqa: E402
from bdhires.eval import crps_ensemble  # noqa: E402
from bdhires.grids import WIDE, crop_offsets, get_grid  # noqa: E402
from bdhires.models import RectifiedFlow, UNet, select_weights  # noqa: E402
from bdhires.transforms import (  # noqa: E402
    CondTransform,
    PrecipTransform,
    ResidualSpec,
    load_climatology,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/da.yaml")
    parser.add_argument("--ckpt", default="runs/prior_h100_cpc/best.pt")
    parser.add_argument("--stations", default="data/processed/bmd_daily_may2018.csv")
    parser.add_argument("--start", default="2018-05-01")
    parser.add_argument("--end", default="2018-05-31")
    parser.add_argument("--members", type=int, default=16)
    parser.add_argument("--withhold", type=float, default=0.2)
    parser.add_argument("--min-coverage", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=201805)
    parser.add_argument("--out", default="data/processed/bmd_may2018_example.npz")
    parser.add_argument("--report", default="data/processed/bmd_may2018_example.json")
    return parser.parse_args()


def sample_at_stations(field_mm: np.ndarray, grid, lat, lon) -> np.ndarray:
    """Bilinearly sample physical rainfall arrays ending in ``(H, W)``."""
    shape = field_mm.shape[:-2]
    flat = np.asarray(field_mm, dtype=np.float32).reshape(-1, grid.nlat, grid.nlon)
    operator = BilinearObsOperator(grid, np.asarray(lat), np.asarray(lon))
    tensor = torch.from_numpy(np.nan_to_num(flat, nan=0.0))[:, None]
    sampled = operator(tensor)[:, 0].numpy()
    return sampled.reshape(*shape, len(lat))


def ensemble_score(ensemble: np.ndarray, observed: np.ndarray) -> dict:
    """Score ``(M, ...)`` ensemble values against matching observations."""
    observed = np.asarray(observed)
    valid = np.isfinite(observed) & np.all(np.isfinite(ensemble), axis=0)
    if not valid.any():
        return {"n": 0}
    members = ensemble[:, valid]
    truth = observed[valid]
    mean = members.mean(axis=0)
    difference = mean - truth
    low, high = np.quantile(members, [0.05, 0.95], axis=0)
    correlation = (
        float(np.corrcoef(mean, truth)[0, 1])
        if mean.std() > 0 and truth.std() > 0
        else float("nan")
    )
    skill = float(np.sqrt(np.mean(difference**2)))
    spread = float(np.sqrt(np.mean(members.var(axis=0, ddof=1))))
    return {
        "n": int(valid.sum()),
        "rmse_mm": skill,
        "mae_mm": float(np.mean(np.abs(difference))),
        "bias_mm": float(np.mean(difference)),
        "crps_mm": float(crps_ensemble(members, truth)),
        "correlation": correlation,
        "spread_mm": spread,
        "spread_skill": float(spread / skill) if skill else float("nan"),
        "coverage_90": float(np.mean((truth >= low) & (truth <= high))),
    }


def deterministic_score(predicted: np.ndarray, observed: np.ndarray) -> dict:
    valid = np.isfinite(predicted) & np.isfinite(observed)
    if not valid.any():
        return {"n": 0}
    predicted = predicted[valid]
    observed = observed[valid]
    difference = predicted - observed
    correlation = (
        float(np.corrcoef(predicted, observed)[0, 1])
        if predicted.std() > 0 and observed.std() > 0
        else float("nan")
    )
    return {
        "n": int(valid.sum()),
        "rmse_mm": float(np.sqrt(np.mean(difference**2))),
        "mae_mm": float(np.mean(np.abs(difference))),
        "bias_mm": float(np.mean(difference)),
        "correlation": correlation,
    }


def percent_reduction(background: dict, analysis: dict, metric: str) -> float:
    before = background.get(metric, float("nan"))
    after = analysis.get(metric, float("nan"))
    if not np.isfinite(before) or before == 0:
        return float("nan")
    return float(100.0 * (before - after) / before)


def main() -> None:
    args = parse_args()
    if not 0 < args.withhold < 1:
        raise ValueError("--withhold must lie strictly between zero and one")
    config = yaml.safe_load(Path(args.config).read_text())
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    training_config = checkpoint["cfg"]
    training_data = training_config["data"]

    # Bind inference inputs to the checkpoint just as the OSSE does.  The CPC
    # checkpoint cannot safely use the older ERA5 Zarr/statistics by accident.
    data_zarr = str(training_data.get("zarr", config["data"]["zarr"]))
    data_stats = str(training_data.get("stats", config["data"]["stats"]))
    stats = json.loads(Path(data_stats).read_text())
    transform = PrecipTransform.from_dict(stats["precip_transform"])
    residual = ResidualSpec.from_stats(stats)
    grid = get_grid(config["data"]["grid"])
    selected_channels = training_data.get("cond_channels")
    dataset = PrecipDataset(
        DatasetConfig(
            root=data_zarr,
            crop=grid.nlon,
            random_crop=False,
            crop_origin=crop_offsets(WIDE, grid),
            seasonal_encoding=bool(training_data.get("seasonal_encoding", True)),
            cond_channels=tuple(selected_channels) if selected_channels is not None else None,
            min_valid_fraction=float(training_data.get("min_valid_fraction", 0.3)),
        ),
        transform,
        cond_mean=np.asarray(stats["cond_mean"], np.float32),
        cond_std=np.asarray(stats["cond_std"], np.float32),
        cond_transform=CondTransform.from_stats(stats),
        residual=residual,
        climatology=load_climatology(data_stats, stats),
    )
    times = dataset.time
    selected = np.where(
        (times >= np.datetime64(args.start)) & (times <= np.datetime64(args.end))
    )[0]
    if not len(selected):
        raise ValueError(f"no checkpoint-bound data between {args.start} and {args.end}")
    selected_times = times[selected]

    # Coverage is intentionally evaluated over the requested month, not over
    # the model's full 1981-2025 time axis.
    stations, gauge_mm = load_stations(
        args.stations,
        selected_times,
        grid=grid,
        min_coverage=args.min_coverage,
    )
    if len(stations) < 5:
        raise ValueError(f"only {len(stations)} BMD stations remain after coverage filtering")
    n_withheld = max(1, min(len(stations) - 1, int(round(args.withhold * len(stations)))))
    eval_idx = spread_holdout(stations.lat, stations.lon, n_withheld)
    assim_idx = np.setdiff1d(np.arange(len(stations)), eval_idx)

    import pandas as pd

    station_table = pd.read_csv(args.stations)
    station_names = (
        station_table.groupby("station_id")["name"]
        .first()
        .reindex(stations.ids)
        .astype(str)
        .to_numpy()
    )
    print(
        f"[bmd] {selected_times[0].astype('datetime64[D]')} to "
        f"{selected_times[-1].astype('datetime64[D]')}: {len(stations)} stations, "
        f"{len(assim_idx)} assimilated, {len(eval_idx)} withheld",
        flush=True,
    )
    print("[bmd] withheld:", ", ".join(station_names[eval_idx]), flush=True)
    print(
        f"[bmd] checkpoint-bound data: {data_zarr} | {data_stats} | "
        f"channels={selected_channels or 'all'}",
        flush=True,
    )

    valid = dataset.fixed_valid > 0
    slices = dataset.fixed_spatial_slices()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(
        in_channels=1,
        cond_channels=dataset.total_cond_channels,
        out_channels=1,
        image_size=grid.nlon,
        **training_config["model"],
    )
    model.load_state_dict(select_weights(checkpoint), strict=True)
    model = model.to(device).eval()
    flow = RectifiedFlow()
    mask = torch.from_numpy(valid.astype(np.float32)[None, None]).to(device)

    sampler = replace(SamplerConfig(**config["sampler"]), mask_fill=dataset.mask_fill)
    guidance = GuidanceConfig(**config["guidance"])
    gauge_config = config["observations"]["gauges"]
    R = build_R(
        len(assim_idx),
        float(gauge_config["sigma_obs"]),
        device=device,
        representativeness=float(gauge_config["representativeness"]),
    )
    operator = PhysicalBilinearObsOperator(
        grid,
        stations.lat[assim_idx],
        stations.lon[assim_idx],
        transform,
        valid=valid,
    ).to(device)

    shape = (len(selected), args.members, grid.nlat, grid.nlon)
    background = np.empty(shape, dtype=np.float32)
    analysis = np.empty(shape, dtype=np.float32)
    chirps = np.empty((len(selected), grid.nlat, grid.nlon), dtype=np.float32)
    condition = np.empty_like(chirps)
    cpc_full_index = (
        dataset.all_cond_channels.index("cpc_precip")
        if "cpc_precip" in dataset.all_cond_channels
        else None
    )

    for day_position, data_index in enumerate(selected):
        dataset_position = int(np.where(dataset.index == data_index)[0][0])
        item = dataset[dataset_position]
        cond = item["cond"][None].to(device)
        base = item["base"][None].to(device)
        day_seed = args.seed + int(data_index)
        day_sampler = replace(sampler, seed=day_seed)

        # Same seed and prior temperature give the background and analysis the
        # same initial ensemble; their difference isolates gauge guidance.
        with torch.inference_mode():
            generated_background = run_assim(
                model,
                cond,
                (args.members, 1, grid.nlat, grid.nlon),
                device,
                cfg=day_sampler,
                flow=flow,
                mask=mask,
                to_precip=lambda x, b=base: residual.decode(x, b),
            )
            background_mm = transform.inverse(
                residual.decode(generated_background, base)[:, 0].float().cpu().numpy()
            )
        background[day_position] = np.where(valid[None], background_mm, np.nan)

        observation = transform.forward(gauge_mm[day_position, assim_idx]).astype(np.float32)
        perturbed = perturb_observations(
            observation,
            R,
            args.members,
            seed=day_seed + 1_000_000,
        ).astype(np.float32)
        perturbed[:, ~np.isfinite(observation)] = np.nan
        y = torch.from_numpy(perturbed[:, None]).to(device)
        generated_analysis = run_assim(
            model,
            cond,
            (args.members, 1, grid.nlat, grid.nlon),
            device,
            H=operator,
            y=y,
            R=R,
            cfg=day_sampler,
            gcfg=guidance,
            flow=flow,
            mask=mask,
            to_precip=lambda x, b=base: residual.decode(x, b),
        ).detach()
        analysis_mm = transform.inverse(
            residual.decode(generated_analysis, base)[:, 0].float().cpu().numpy()
        )
        analysis[day_position] = np.where(valid[None], analysis_mm, np.nan)

        chirps[day_position] = np.asarray(dataset.z["target"][int(data_index)][slices])
        if cpc_full_index is not None:
            condition[day_position] = np.asarray(
                dataset.z["cond"][int(data_index)][cpc_full_index][slices]
            )
        else:
            condition[day_position].fill(np.nan)
        print(
            f"[bmd] {day_position + 1:02d}/{len(selected)} "
            f"{selected_times[day_position].astype('datetime64[D]')}",
            flush=True,
        )

    background_at_stations = sample_at_stations(
        background, grid, stations.lat, stations.lon
    )
    analysis_at_stations = sample_at_stations(analysis, grid, stations.lat, stations.lon)
    chirps_at_stations = sample_at_stations(chirps, grid, stations.lat, stations.lon)
    condition_at_stations = sample_at_stations(condition, grid, stations.lat, stations.lon)
    background_eval = np.moveaxis(background_at_stations[:, :, eval_idx], 1, 0)
    analysis_eval = np.moveaxis(analysis_at_stations[:, :, eval_idx], 1, 0)
    observed_eval = gauge_mm[:, eval_idx]
    background_score = ensemble_score(background_eval, observed_eval)
    analysis_score = ensemble_score(analysis_eval, observed_eval)
    chirps_score = deterministic_score(chirps_at_stations[:, eval_idx], observed_eval)
    condition_score = deterministic_score(condition_at_stations[:, eval_idx], observed_eval)
    background_mean_grid = np.where(
        valid[None], np.nan_to_num(background, nan=0.0).mean(axis=1), np.nan
    )
    analysis_mean_grid = np.where(
        valid[None], np.nan_to_num(analysis, nan=0.0).mean(axis=1), np.nan
    )
    grid_background = deterministic_score(background_mean_grid, chirps)
    grid_analysis = deterministic_score(analysis_mean_grid, chirps)

    daily = []
    for day in range(len(selected)):
        b = ensemble_score(background_eval[:, day], observed_eval[day])
        a = ensemble_score(analysis_eval[:, day], observed_eval[day])
        daily.append(
            {
                "date": str(selected_times[day].astype("datetime64[D]")),
                "background": b,
                "analysis": a,
                "crps_reduction_percent": percent_reduction(b, a, "crps_mm"),
            }
        )

    train_years = training_data.get("years", {}).get("train")
    in_training_period = bool(
        train_years
        and int(str(np.datetime64(args.start, "Y"))[:4]) >= int(train_years[0])
        and int(str(np.datetime64(args.end, "Y"))[:4]) <= int(train_years[1])
    )
    report = {
        "experiment": "May 2018 real-BMD gauge-only DA process example",
        "scope": {
            "start": args.start,
            "end": args.end,
            "checkpoint": args.ckpt,
            "checkpoint_data": data_zarr,
            "checkpoint_stats": data_stats,
            "conditioning_channels": selected_channels,
            "in_checkpoint_training_period": in_training_period,
            "scientific_status": (
                "workflow demonstration; not an independent temporal skill estimate"
                if in_training_period
                else "temporally independent if checkpoint metadata are correct"
            ),
            "imerg_assimilated": False,
            "imerg_note": (
                "Gauge-only gate; real IMERG ingestion and bias correction remain separate."
            ),
            "timing_note": (
                "Confirm the BMD 24-hour accumulation boundary against CPC/CHIRPS dates "
                "before interpreting event-level differences scientifically."
            ),
        },
        "network": {
            "stations": int(len(stations)),
            "assimilated": int(len(assim_idx)),
            "withheld": int(len(eval_idx)),
            "withhold_fraction": float(args.withhold),
            "assimilated_ids": stations.ids[assim_idx].tolist(),
            "withheld_ids": stations.ids[eval_idx].tolist(),
            "withheld_names": station_names[eval_idx].tolist(),
            "selection": "deterministic farthest-point spatial holdout",
        },
        "observation_error": {
            "sigma_transformed": float(gauge_config["sigma_obs"]),
            "representativeness_transformed": float(gauge_config["representativeness"]),
            "note": "Provisional OSSE values; tune with rotated real-gauge withholding later.",
        },
        "withheld_gauges": {
            "background": background_score,
            "analysis": analysis_score,
            "crps_reduction_percent": percent_reduction(
                background_score, analysis_score, "crps_mm"
            ),
            "rmse_reduction_percent": percent_reduction(
                background_score, analysis_score, "rmse_mm"
            ),
            "chirps": chirps_score,
            "cpc_condition": condition_score,
        },
        "chirps_grid_consistency": {
            "background": grid_background,
            "analysis": grid_analysis,
            "interpretation": (
                "CHIRPS is the checkpoint target and May 2018 is in training; these are "
                "consistency diagnostics, not independent truth scores."
            ),
        },
        "daily_withheld_scores": daily,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        background=background,
        analysis=analysis,
        chirps=chirps,
        condition=condition,
        gauge_mm=gauge_mm,
        background_at_stations=background_at_stations,
        analysis_at_stations=analysis_at_stations,
        chirps_at_stations=chirps_at_stations,
        condition_at_stations=condition_at_stations,
        station_id=stations.ids,
        station_name=station_names,
        station_lat=stations.lat,
        station_lon=stations.lon,
        assim_idx=assim_idx,
        eval_idx=eval_idx,
        time=selected_times.astype("datetime64[ns]").astype("i8"),
        grid_lat=grid.lat,
        grid_lon=grid.lon,
        valid=valid,
    )
    Path(args.report).write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")
    print(f"wrote {args.out}")
    print(f"wrote {args.report}")
    print(
        f"withheld CRPS {background_score['crps_mm']:.2f} -> "
        f"{analysis_score['crps_mm']:.2f} "
        f"({report['withheld_gauges']['crps_reduction_percent']:+.1f}%)"
    )


if __name__ == "__main__":
    main()
