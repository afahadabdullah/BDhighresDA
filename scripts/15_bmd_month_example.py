#!/usr/bin/env python
"""One-month real BMD + IMERG assimilation and withheld-station evaluation.

This is a process-validation experiment, not a final independent skill claim.
The default May 2018 period is inside the CPC checkpoint's training years, but
the withheld BMD gauges never enter this likelihood and therefore reveal
whether real gauge ingestion, guidance, and station-space verification work.

When ``--imerg`` is supplied, four controlled arms are run with matched seeds:
background, gauges-only, IMERG-only, and simultaneous gauges+IMERG. Satellite
footprints are thinned and their variance is inflated by an approximate
correlation area so thousands of correlated pixels are not treated as
independent evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import xarray as xr
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bdhires.bmd import spread_folds, spread_holdout  # noqa: E402
from bdhires.da import (  # noqa: E402
    BilinearObsOperator,
    CompositeObsOperator,
    GuidanceConfig,
    PhysicalBilinearObsOperator,
    PhysicalBlockAverageObsOperator,
    SamplerConfig,
    build_R,
    perturb_observations,
)
from bdhires.da.sampler import assimilate as run_assim  # noqa: E402
from bdhires.data import DatasetConfig, PrecipDataset, load_stations  # noqa: E402
from bdhires.eval import crps_ensemble  # noqa: E402
from bdhires.grids import Grid, WIDE, crop_offsets, get_grid  # noqa: E402
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
    parser.add_argument(
        "--imerg",
        default=None,
        help="regional file produced by scripts/08_prepare_imerg_observations.py",
    )
    parser.add_argument("--start", default="2018-05-01")
    parser.add_argument("--end", default="2018-05-31")
    parser.add_argument(
        "--background-day-offset",
        type=int,
        default=0,
        help=(
            "whole checkpoint-item offset relative to each BMD/IMERG date; "
            "-1 uses previous-day CPC, ERA5 states, residual base and season; "
            "observation-date CHIRPS remains contextual only"
        ),
    )
    parser.add_argument("--members", type=int, default=16)
    parser.add_argument("--withhold", type=float, default=0.2)
    parser.add_argument(
        "--holdout-folds",
        type=int,
        default=1,
        help="number of geographically spread folds; 1 retains the legacy holdout",
    )
    parser.add_argument(
        "--holdout-fold",
        type=int,
        default=0,
        help="zero-based fold to withhold when --holdout-folds is greater than 1",
    )
    parser.add_argument("--min-coverage", type=float, default=0.8)
    parser.add_argument(
        "--imerg-stride",
        type=int,
        default=3,
        help="retain every Nth IMERG footprint in each direction (default: 3)",
    )
    parser.add_argument(
        "--imerg-r-multiplier",
        type=float,
        default=1.0,
        help="extra multiplier after spatial-correlation variance inflation",
    )
    parser.add_argument("--seed", type=int, default=201805)
    parser.add_argument("--out", default="data/processed/bmd_may2018_example.npz")
    parser.add_argument("--report", default="data/processed/bmd_may2018_example.json")
    return parser.parse_args()


def load_prepared_imerg(path: str | Path, times: np.ndarray, grid, factor: int) -> dict:
    """Load BMD-window IMERG and enforce exact temporal/date/grid alignment."""
    with xr.open_dataset(path) as dataset:
        required = {"precipitation", "randomError", "precipitation_cnt"}
        if not required.issubset(dataset):
            raise ValueError(f"{path} lacks required IMERG variables {sorted(required)}")
        for variable in ("precipitation", "randomError"):
            units = str(dataset[variable].attrs.get("units", ""))
            if units.lower().replace(" ", "") not in {
                "mm/day", "mmday-1", "mmd-1", "mmday^-1", "mmd^-1"
            }:
                raise ValueError(
                    f"{path} {variable} units are {units!r}; expected mm/day"
                )
        imerg_time = np.asarray(dataset.time.values).astype("datetime64[D]")
        expected_time = np.asarray(times).astype("datetime64[D]")
        if not np.array_equal(imerg_time, expected_time):
            raise ValueError(
                f"IMERG dates do not exactly match checkpoint dates: "
                f"{imerg_time[[0, -1]]} versus {expected_time[[0, -1]]}"
            )
        end_hour = dataset.attrs.get("bmd_accumulation_end_hour_utc")
        try:
            end_hour = int(end_hour)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{path} does not declare a BMD accumulation end hour; regenerate it "
                "from half-hourly IMERG with scripts/08_prepare_imerg_observations.py"
            ) from exc
        if end_hour != 3:
            raise ValueError(
                f"{path} ends its accumulation at {end_hour:02d}:00 UTC, but BMD daily "
                "rainfall ends at 03:00 UTC; calendar-day IMERG cannot be assimilated"
            )
        if str(dataset.attrs.get("source_frequency", "")) != "half-hourly":
            raise ValueError(
                f"{path} was not prepared from half-hourly IMERG and cannot represent "
                "the exact BMD 03:00-03:00 UTC window"
            )
        expected_lat = grid.lat.reshape(grid.nlat // factor, factor).mean(axis=1)
        expected_lon = grid.lon.reshape(grid.nlon // factor, factor).mean(axis=1)
        if not np.allclose(dataset.lat.values, expected_lat, atol=2e-3, rtol=0):
            raise ValueError("prepared IMERG latitude does not nest on the model grid")
        if not np.allclose(dataset.lon.values, expected_lon, atol=2e-3, rtol=0):
            raise ValueError("prepared IMERG longitude does not nest on the model grid")
        return {
            "precipitation": np.asarray(dataset.precipitation.values, np.float32),
            "random_error": np.asarray(dataset.randomError.values, np.float32),
            "count": np.asarray(dataset.precipitation_cnt.values, np.int16),
            "lat": np.asarray(dataset.lat.values, np.float32),
            "lon": np.asarray(dataset.lon.values, np.float32),
            "attrs": dict(dataset.attrs),
        }


def transformed_imerg_variance(
    precipitation_mm: np.ndarray,
    random_error_mm: np.ndarray,
    transform: PrecipTransform,
    sigma_floor: float,
    representativeness: float,
) -> np.ndarray:
    """Delta-like conversion of native mm/day RMS error to model space.

    A symmetric physical interval is transformed explicitly, which is more
    stable near zero than dividing by precipitation under a log transform.
    ``sigma_floor`` prevents unrealistically confident retrievals.
    """
    lower = np.clip(precipitation_mm - random_error_mm, 0.0, None)
    upper = np.clip(precipitation_mm + random_error_mm, 0.0, None)
    native_sigma = 0.5 * np.abs(transform.forward(upper) - transform.forward(lower))
    sigma = np.maximum(native_sigma, sigma_floor)
    return sigma.astype(np.float32) ** 2 + np.float32(representativeness**2)


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


def physical_summary(ensemble: np.ndarray) -> dict:
    values = np.asarray(ensemble)
    finite = values[np.isfinite(values)]
    if not finite.size:
        return {"n": 0}
    return {
        "n": int(finite.size),
        "mean_mm": float(finite.mean()),
        "p99_mm": float(np.percentile(finite, 99.0)),
        "p99_9_mm": float(np.percentile(finite, 99.9)),
        "max_mm": float(finite.max()),
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
    if args.holdout_folds < 1:
        raise ValueError("--holdout-folds must be at least one")
    if not 0 <= args.holdout_fold < args.holdout_folds:
        raise ValueError("--holdout-fold must be in [0, --holdout-folds)")
    if args.imerg_stride < 1:
        raise ValueError("--imerg-stride must be at least one")
    if args.imerg_r_multiplier <= 0:
        raise ValueError("--imerg-r-multiplier must be positive")
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
    observation_selected = np.where(
        (times >= np.datetime64(args.start)) & (times <= np.datetime64(args.end))
    )[0]
    if not len(observation_selected):
        raise ValueError(f"no checkpoint-bound data between {args.start} and {args.end}")
    selected_times = times[observation_selected]
    background_times = selected_times + np.timedelta64(args.background_day_offset, "D")
    time_to_store_index = {
        np.datetime64(value, "D"): index for index, value in enumerate(times)
    }
    missing_background = [
        str(np.datetime64(value, "D"))
        for value in background_times
        if np.datetime64(value, "D") not in time_to_store_index
    ]
    if missing_background:
        raise ValueError(
            "checkpoint lacks background dates required by "
            f"--background-day-offset={args.background_day_offset}: "
            + ", ".join(missing_background)
        )
    selected = np.array(
        [time_to_store_index[np.datetime64(value, "D")] for value in background_times],
        dtype=np.int64,
    )

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
    if args.holdout_folds > 1:
        folds = spread_folds(stations.lat, stations.lon, args.holdout_folds)
        eval_idx = folds[args.holdout_fold]
        selection_description = (
            f"geographically spread fold {args.holdout_fold + 1}/{args.holdout_folds}; "
            "folds are disjoint and exhaustive"
        )
    else:
        eval_idx = spread_holdout(stations.lat, stations.lon, n_withheld)
        selection_description = "deterministic farthest-point spatial holdout"
    assim_idx = np.setdiff1d(np.arange(len(stations)), eval_idx)

    import pandas as pd

    station_table = pd.read_csv(args.stations)
    station_names = (
        station_table.groupby("station_id")["name"]
        .first()
        .reindex(stations.ids)
        .astype(str)
        .to_numpy(dtype=str)
    )
    print(
        f"[bmd] {selected_times[0].astype('datetime64[D]')} to "
        f"{selected_times[-1].astype('datetime64[D]')}: {len(stations)} stations, "
        f"{len(assim_idx)} assimilated, {len(eval_idx)} withheld",
        flush=True,
    )
    print(
        f"[bmd] checkpoint background offset {args.background_day_offset:+d} day(s): "
        f"{background_times[0].astype('datetime64[D]')} to "
        f"{background_times[-1].astype('datetime64[D]')}",
        flush=True,
    )
    print(
        f"[bmd] holdout: {selection_description}; "
        f"{len(eval_idx)} withheld and {len(assim_idx)} assimilated",
        flush=True,
    )
    print("[bmd] withheld:", ", ".join(station_names[eval_idx]), flush=True)
    print(
        f"[bmd] checkpoint-bound data: {data_zarr} | {data_stats} | "
        f"channels={selected_channels or 'all'}",
        flush=True,
    )

    imerg = None
    imerg_config = config["observations"]["imerg"]
    imerg_factor = int(imerg_config.get("factor", 2))
    if args.imerg:
        imerg = load_prepared_imerg(args.imerg, selected_times, grid, imerg_factor)
        print(
            f"[imerg] {args.imerg}: {np.isfinite(imerg['precipitation']).mean():.1%} "
            "regional footprints pass file QC; exact BMD 03:00-03:00 UTC windows; "
            "aggregated randomError will set daily R",
            flush=True,
        )
        print(
            "[imerg] WARNING: no fitted bias correction in this bounded process run",
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
    background_sampler = replace(
        SamplerConfig(**config.get("background_sampler", config["sampler"])),
        mask_fill=dataset.mask_fill,
    )
    guidance = GuidanceConfig(**config["guidance"])
    gauge_config = config["observations"]["gauges"]
    gauge_R = build_R(
        len(assim_idx),
        float(gauge_config["sigma_obs"]),
        device=device,
        representativeness=float(gauge_config["representativeness"]),
    )
    gauge_operator = PhysicalBilinearObsOperator(
        grid,
        stations.lat[assim_idx],
        stations.lon[assim_idx],
        transform,
        valid=valid,
    ).to(device)
    satellite_operator = None
    satellite_keep = None
    combined_operator = None
    satellite_correlation_inflation = 1.0
    if imerg is not None:
        satellite_operator = PhysicalBlockAverageObsOperator(
            imerg_factor, transform, valid=valid
        ).to(device)
        satellite_keep = satellite_operator.valid_mask().detach().cpu().numpy().astype(bool)
        coarse_shape = imerg["precipitation"].shape[1:]
        thinning = np.zeros(coarse_shape, dtype=bool)
        offset = args.imerg_stride // 2
        thinning[offset::args.imerg_stride, offset::args.imerg_stride] = True
        satellite_keep &= thinning.reshape(-1)
        error_corr_cells = float(imerg_config.get("error_corr_cells", 0.0))
        # Approximate the integral correlation area on the retained-footprint
        # lattice. This turns a diagonal-R likelihood into a conservative
        # effective-sample approximation without pretending 4k pixels are
        # independent. The extra multiplier supports a controlled sensitivity.
        satellite_correlation_inflation = max(
            1.0,
            2.0 * np.pi * (error_corr_cells / args.imerg_stride) ** 2,
        ) * args.imerg_r_multiplier
        combined_operator = CompositeObsOperator(
            [gauge_operator, satellite_operator]
        ).to(device)
        print(
            f"[imerg] controlled likelihood: stride={args.imerg_stride}, "
            f"selected land footprints={satellite_keep.sum()}/{satellite_keep.size}, "
            f"R inflation={satellite_correlation_inflation:.2f}x",
            flush=True,
        )

    shape = (len(selected), args.members, grid.nlat, grid.nlon)
    background = np.empty(shape, dtype=np.float32)
    analysis_gauge = np.empty(shape, dtype=np.float32)
    analysis_imerg = np.empty(shape, dtype=np.float32) if imerg is not None else None
    analysis_combined = np.empty(shape, dtype=np.float32) if imerg is not None else None
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
        # Key every stochastic arm to the observation date, not the shifted
        # checkpoint date. Offset 0 and -1 therefore use matched prior noise,
        # gauge perturbations and IMERG perturbations.
        day_seed = args.seed + int(observation_selected[day_position])
        day_sampler = replace(sampler, seed=day_seed)
        day_background_sampler = replace(background_sampler, seed=day_seed)

        # All generative arms share the same seed; all guided arms additionally
        # share the sampler and prior temperature. The background uses the
        # production background sampler, which has no analysis inflation.
        with torch.inference_mode():
            generated_background = run_assim(
                model,
                cond,
                (args.members, 1, grid.nlat, grid.nlon),
                device,
                cfg=day_background_sampler,
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
            gauge_R,
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
            H=gauge_operator,
            y=y,
            R=gauge_R,
            cfg=day_sampler,
            gcfg=guidance,
            flow=flow,
            mask=mask,
            to_precip=lambda x, b=base: residual.decode(x, b),
        ).detach()
        analysis_mm = transform.inverse(
            residual.decode(generated_analysis, base)[:, 0].float().cpu().numpy()
        )
        analysis_gauge[day_position] = np.where(valid[None], analysis_mm, np.nan)

        if imerg is not None:
            satellite_mm = imerg["precipitation"][day_position].reshape(-1)
            satellite_error_mm = imerg["random_error"][day_position].reshape(-1)
            satellite_observation = transform.forward(satellite_mm).astype(np.float32)
            satellite_variance = transformed_imerg_variance(
                satellite_mm,
                satellite_error_mm,
                transform,
                sigma_floor=float(imerg_config["sigma_obs"]),
                representativeness=float(imerg_config["representativeness"]),
            )
            satellite_variance *= np.float32(satellite_correlation_inflation)
            satellite_valid = (
                satellite_keep
                & np.isfinite(satellite_observation)
                & np.isfinite(satellite_variance)
            )
            satellite_observation[~satellite_valid] = np.nan
            # Missing observations are masked by NaN in the likelihood.  Give
            # their unused R entries a finite value so tensor arithmetic stays
            # clean at every guidance time.
            satellite_variance[~satellite_valid] = 1.0
            satellite_R = torch.from_numpy(satellite_variance).to(device)
            satellite_perturbed = perturb_observations(
                satellite_observation,
                satellite_R,
                args.members,
                seed=day_seed + 2_000_000,
                corr_blocks=[
                    (
                        0,
                        imerg["precipitation"].shape[1],
                        imerg["precipitation"].shape[2],
                        float(imerg_config.get("error_corr_cells", 0.0)),
                    )
                ],
            ).astype(np.float32)
            satellite_perturbed[:, ~np.isfinite(satellite_observation)] = np.nan
            satellite_y = torch.from_numpy(satellite_perturbed[:, None]).to(device)

            # Arm 3: satellite-only generative posterior. This must remain
            # physically bounded before it is considered a viable product.
            generated_imerg = run_assim(
                model,
                cond,
                (args.members, 1, grid.nlat, grid.nlon),
                device,
                H=satellite_operator,
                y=satellite_y,
                R=satellite_R,
                cfg=day_sampler,
                gcfg=guidance,
                flow=flow,
                mask=mask,
                to_precip=lambda x, b=base: residual.decode(x, b),
            ).detach()
            imerg_analysis_mm = transform.inverse(
                residual.decode(generated_imerg, base)[:, 0].float().cpu().numpy()
            )
            analysis_imerg[day_position] = np.where(
                valid[None], imerg_analysis_mm, np.nan
            )

            # Arm 4: stabilized simultaneous likelihood. Reuse exactly the
            # gauge and satellite perturbations from their single-stream arms.
            combined_R = torch.cat([gauge_R, satellite_R])
            combined_perturbed = np.concatenate(
                [perturbed, satellite_perturbed], axis=1
            ).astype(np.float32)
            combined_y = torch.from_numpy(combined_perturbed[:, None]).to(device)
            generated_combined = run_assim(
                model,
                cond,
                (args.members, 1, grid.nlat, grid.nlon),
                device,
                H=combined_operator,
                y=combined_y,
                R=combined_R,
                cfg=day_sampler,
                gcfg=guidance,
                flow=flow,
                mask=mask,
                to_precip=lambda x, b=base: residual.decode(x, b),
            ).detach()
            combined_mm = transform.inverse(
                residual.decode(generated_combined, base)[:, 0].float().cpu().numpy()
            )
            analysis_combined[day_position] = np.where(
                valid[None], combined_mm, np.nan
            )

        # CHIRPS is not a checkpoint input. Keep this non-independent context
        # on the observation date so timing arms differ only in their prior.
        observation_store_index = int(observation_selected[day_position])
        chirps[day_position] = np.asarray(
            dataset.z["target"][observation_store_index][slices]
        )
        if cpc_full_index is not None:
            condition[day_position] = np.asarray(
                dataset.z["cond"][int(data_index)][cpc_full_index][slices]
            )
        else:
            condition[day_position].fill(np.nan)
        print(
            f"[bmd] {day_position + 1:02d}/{len(selected)} "
            f"obs={selected_times[day_position].astype('datetime64[D]')} "
            f"background={background_times[day_position].astype('datetime64[D]')}",
            flush=True,
        )

    analysis = analysis_combined if analysis_combined is not None else analysis_gauge
    background_at_stations = sample_at_stations(background, grid, stations.lat, stations.lon)
    gauge_at_stations = sample_at_stations(
        analysis_gauge, grid, stations.lat, stations.lon
    )
    imerg_analysis_at_stations = (
        sample_at_stations(analysis_imerg, grid, stations.lat, stations.lon)
        if analysis_imerg is not None
        else gauge_at_stations
    )
    combined_at_stations = (
        sample_at_stations(analysis_combined, grid, stations.lat, stations.lon)
        if analysis_combined is not None
        else gauge_at_stations
    )
    analysis_at_stations = combined_at_stations
    chirps_at_stations = sample_at_stations(chirps, grid, stations.lat, stations.lon)
    condition_at_stations = sample_at_stations(condition, grid, stations.lat, stations.lon)
    imerg_at_stations = None
    if imerg is not None:
        imerg_grid = Grid(
            "imerg_bd",
            grid.lon_min,
            grid.lat_min,
            grid.nlon // imerg_factor,
            grid.nlat // imerg_factor,
            grid.res * imerg_factor,
        )
        imerg_at_stations = sample_at_stations(
            imerg["precipitation"], imerg_grid, stations.lat, stations.lon
        )
    background_eval = np.moveaxis(background_at_stations[:, :, eval_idx], 1, 0)
    gauge_eval = np.moveaxis(gauge_at_stations[:, :, eval_idx], 1, 0)
    imerg_analysis_eval = np.moveaxis(
        imerg_analysis_at_stations[:, :, eval_idx], 1, 0
    )
    combined_eval = np.moveaxis(combined_at_stations[:, :, eval_idx], 1, 0)
    observed_eval = gauge_mm[:, eval_idx]
    background_score = ensemble_score(background_eval, observed_eval)
    gauge_score = ensemble_score(gauge_eval, observed_eval)
    imerg_analysis_score = ensemble_score(imerg_analysis_eval, observed_eval)
    combined_score = ensemble_score(combined_eval, observed_eval)
    analysis_score = combined_score
    chirps_score = deterministic_score(chirps_at_stations[:, eval_idx], observed_eval)
    condition_score = deterministic_score(condition_at_stations[:, eval_idx], observed_eval)
    imerg_score = (
        deterministic_score(imerg_at_stations[:, eval_idx], observed_eval)
        if imerg_at_stations is not None
        else None
    )
    background_mean_grid = np.where(
        valid[None], np.nan_to_num(background, nan=0.0).mean(axis=1), np.nan
    )
    gauge_mean_grid = np.where(
        valid[None], np.nan_to_num(analysis_gauge, nan=0.0).mean(axis=1), np.nan
    )
    imerg_analysis_mean_grid = np.where(
        valid[None],
        np.nan_to_num(
            analysis_imerg if analysis_imerg is not None else analysis_gauge, nan=0.0
        ).mean(axis=1),
        np.nan,
    )
    combined_mean_grid = np.where(
        valid[None],
        np.nan_to_num(
            analysis_combined if analysis_combined is not None else analysis_gauge,
            nan=0.0,
        ).mean(axis=1),
        np.nan,
    )
    grid_background = deterministic_score(background_mean_grid, chirps)
    grid_gauge = deterministic_score(gauge_mean_grid, chirps)
    grid_imerg = deterministic_score(imerg_analysis_mean_grid, chirps)
    grid_combined = deterministic_score(combined_mean_grid, chirps)
    chirps_ensemble_scores = {
        "background": ensemble_score(np.moveaxis(background, 1, 0), chirps),
        "gauges_only": ensemble_score(np.moveaxis(analysis_gauge, 1, 0), chirps),
        "imerg_only": ensemble_score(
            np.moveaxis(
                analysis_imerg if analysis_imerg is not None else analysis_gauge,
                1,
                0,
            ),
            chirps,
        ),
        "simultaneous": ensemble_score(
            np.moveaxis(
                analysis_combined if analysis_combined is not None else analysis_gauge,
                1,
                0,
            ),
            chirps,
        ),
    }

    daily = []
    for day in range(len(selected)):
        b = ensemble_score(background_eval[:, day], observed_eval[day])
        g = ensemble_score(gauge_eval[:, day], observed_eval[day])
        satellite = ensemble_score(imerg_analysis_eval[:, day], observed_eval[day])
        c = ensemble_score(combined_eval[:, day], observed_eval[day])
        daily.append(
            {
                "date": str(selected_times[day].astype("datetime64[D]")),
                "background": b,
                "gauges_only": g,
                "imerg_only": satellite,
                "simultaneous": c,
                "background_to_gauges_crps_reduction_percent": percent_reduction(
                    b, g, "crps_mm"
                ),
                "background_to_simultaneous_crps_reduction_percent": percent_reduction(
                    b, c, "crps_mm"
                ),
            }
        )

    train_years = training_data.get("years", {}).get("train")
    in_training_period = bool(
        train_years
        and int(str(np.datetime64(args.start, "Y"))[:4]) >= int(train_years[0])
        and int(str(np.datetime64(args.end, "Y"))[:4]) <= int(train_years[1])
    )
    period_label = f"{args.start} to {args.end}"
    temporal_status = (
        "inside the checkpoint training period; these are process and consistency "
        "diagnostics, not independent temporal validation"
        if in_training_period
        else "outside the checkpoint training period, subject to confirming that all "
        "method tuning and configuration choices were fixed independently"
    )
    report = {
        "experiment": (
            f"{period_label} controlled four-arm real-BMD plus real-IMERG DA experiment"
            if imerg is not None
            else f"{period_label} real-BMD gauge-only DA process experiment"
        ),
        "scope": {
            "start": args.start,
            "end": args.end,
            "observation_dates": [
                str(value.astype("datetime64[D]")) for value in selected_times
            ],
            "background_day_offset": int(args.background_day_offset),
            "background_dates": [
                str(value.astype("datetime64[D]")) for value in background_times
            ],
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
            "imerg_assimilated": imerg is not None,
            "imerg_file": args.imerg,
            "imerg_note": (
                "GPM IMERG Final V07B half-hourly precipitation is accumulated over "
                "the exact BMD reporting day ending at 03:00 UTC. Precipitation and "
                "aggregated randomError enter in mm/day. No fitted bias correction is "
                "applied, so this is a bounded process test and not the final real-data "
                "configuration."
                if imerg is not None
                else "Gauge-only run; no IMERG observation was supplied."
            ),
            "independence_note": (
                "IMERG Final is gauge-adjusted and CHIRPS is gauge-based; neither should "
                "be treated as independent truth against the same BMD network. Headline "
                "scores use BMD stations withheld from this assimilation only."
            ),
            "timing_note": (
                "IMERG is aligned exactly to each BMD 24-hour window from previous-day "
                "03:00 UTC through selected-day 03:00 UTC. The complete checkpoint "
                f"item uses an offset of {args.background_day_offset:+d} day(s), so CPC, "
                "ERA5 state means, the residual base and seasonal encoding remain "
                "internally paired. CHIRPS is not a model input and remains on the "
                "observation date as non-independent context only."
            ),
        },
        "network": {
            "stations": int(len(stations)),
            "assimilated": int(len(assim_idx)),
            "withheld": int(len(eval_idx)),
            "withhold_fraction": float(args.withhold),
            "holdout_fold": int(args.holdout_fold),
            "holdout_folds": int(args.holdout_folds),
            "assimilated_ids": stations.ids[assim_idx].tolist(),
            "withheld_ids": stations.ids[eval_idx].tolist(),
            "withheld_names": station_names[eval_idx].tolist(),
            "selection": selection_description,
        },
        "observation_error": {
            "gauges": {
                "sigma_transformed": float(gauge_config["sigma_obs"]),
                "representativeness_transformed": float(
                    gauge_config["representativeness"]
                ),
            },
            "imerg": (
                {
                    "native": imerg["attrs"].get(
                        "random_error_aggregation", "aggregated V07B randomError in mm/day"
                    ),
                    "source_frequency": imerg["attrs"].get("source_frequency"),
                    "accumulation_window": imerg["attrs"].get("accumulation_window"),
                    "conversion": "symmetric physical interval transformed to model space",
                    "sigma_floor_transformed": float(imerg_config["sigma_obs"]),
                    "representativeness_transformed": float(
                        imerg_config["representativeness"]
                    ),
                    "spatial_correlation_cells": float(
                        imerg_config.get("error_corr_cells", 0.0)
                    ),
                    "footprint_stride": int(args.imerg_stride),
                    "selected_land_footprints_per_day": int(satellite_keep.sum()),
                    "correlation_variance_inflation": float(
                        satellite_correlation_inflation
                    ),
                    "extra_r_multiplier": float(args.imerg_r_multiplier),
                    "valid_raw_footprints": int(
                        np.isfinite(imerg["precipitation"]).sum()
                    ),
                }
                if imerg is not None
                else None
            ),
            "note": "Provisional values; select methods with all rotated real-gauge folds.",
        },
        "withheld_gauges": {
            "background": background_score,
            "gauges_only": gauge_score,
            "imerg_only": imerg_analysis_score,
            "simultaneous": combined_score,
            "analysis": analysis_score,
            "background_to_gauges_crps_reduction_percent": percent_reduction(
                background_score, gauge_score, "crps_mm"
            ),
            "background_to_simultaneous_crps_reduction_percent": percent_reduction(
                background_score, combined_score, "crps_mm"
            ),
            "chirps": chirps_score,
            "cpc_condition": condition_score,
            "imerg": imerg_score,
        },
        "chirps_grid_consistency": {
            "background": grid_background,
            "gauges_only": grid_gauge,
            "imerg_only": grid_imerg,
            "simultaneous": grid_combined,
            "interpretation": (
                f"CHIRPS is the checkpoint target and {period_label} is {temporal_status}. "
                "CHIRPS scores are contextual consistency diagnostics, not an independent "
                "truth evaluation."
            ),
        },
        "chirps_spatial_evaluation": {
            **chirps_ensemble_scores,
            "target": "checkpoint-bound CHIRPS daily 0.05-degree field",
            "domain": "all valid land grid cells and requested days, unweighted",
            "interpretation": (
                "CHIRPS is treated as the gridded target for comparing products, but "
                f"{period_label} is {temporal_status}. CHIRPS is also gauge-based, so "
                "these are consistency scores rather than independent validation."
            ),
        },
        "physical_ranges": {
            "background": physical_summary(background),
            "gauges_only": physical_summary(analysis_gauge),
            "imerg_only": physical_summary(
                analysis_imerg if analysis_imerg is not None else analysis_gauge
            ),
            "simultaneous": physical_summary(
                analysis_combined if analysis_combined is not None else analysis_gauge
            ),
            "interpretation": (
                "Compare p99, p99.9 and max across arms. Very large satellite-arm "
                "values indicate guidance instability even if a station metric improves."
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
        analysis_gauge=analysis_gauge,
        analysis_imerg=(analysis_gauge if analysis_imerg is None else analysis_imerg),
        analysis_combined=(
            analysis_gauge if analysis_combined is None else analysis_combined
        ),
        chirps=chirps,
        condition=condition,
        imerg=(
            np.full((len(selected), grid.nlat // imerg_factor, grid.nlon // imerg_factor), np.nan)
            if imerg is None
            else imerg["precipitation"]
        ),
        imerg_random_error=(
            np.full((len(selected), grid.nlat // imerg_factor, grid.nlon // imerg_factor), np.nan)
            if imerg is None
            else imerg["random_error"]
        ),
        gauge_mm=gauge_mm,
        background_at_stations=background_at_stations,
        analysis_at_stations=analysis_at_stations,
        gauge_analysis_at_stations=gauge_at_stations,
        imerg_analysis_at_stations=imerg_analysis_at_stations,
        combined_analysis_at_stations=combined_at_stations,
        chirps_at_stations=chirps_at_stations,
        condition_at_stations=condition_at_stations,
        imerg_at_stations=(
            np.full_like(gauge_mm, np.nan) if imerg_at_stations is None else imerg_at_stations
        ),
        # Fixed-width Unicode keeps the NPZ non-pickled and safely loadable
        # with allow_pickle=False.
        station_id=np.asarray(stations.ids).astype(str),
        station_name=np.asarray(station_names).astype(str),
        station_lat=stations.lat,
        station_lon=stations.lon,
        assim_idx=assim_idx,
        eval_idx=eval_idx,
        time=selected_times.astype("datetime64[ns]").astype("i8"),
        background_time=background_times.astype("datetime64[ns]").astype("i8"),
        grid_lat=grid.lat,
        grid_lon=grid.lon,
        valid=valid,
    )
    Path(args.report).write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")
    print(f"wrote {args.out}")
    print(f"wrote {args.report}")
    print(
        f"withheld CRPS background {background_score['crps_mm']:.2f}; "
        f"gauges {gauge_score['crps_mm']:.2f}; "
        f"IMERG {imerg_analysis_score['crps_mm']:.2f}; "
        f"simultaneous {combined_score['crps_mm']:.2f}"
    )


if __name__ == "__main__":
    main()
