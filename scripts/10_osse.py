#!/usr/bin/env python
"""Observing-system simulation experiment: how much does assimilation buy?

CHIRPS is treated as the nature run.  Pseudo-gauges sample it at station
locations and, optionally, exact nested 2x2 means provide a 0.1-degree
pseudo-satellite. Observation error is added, a subset of gauges plus the dense
satellite are assimilated, and the 0.05-degree analysis is scored against the
truth. Because the true full field is known, this answers questions that real
gauges never can: how much skill the DA adds, whether it reconstructs subgrid
structure rather than merely copying coarse footprints, and whether the guidance
hyperparameters are set sensibly.

THREE SCOPES, WHICH ANSWER DIFFERENT QUESTIONS
    assimilated  Does the analysis actually move to the observations it was
                 given?  This is a FIT diagnostic, not a skill one -- it is
                 circular as a measure of accuracy, but it is the first thing to
                 look at when a DA system underperforms.  Reported alongside the
                 assumed observation error: if the analysis sits much further
                 from the observations than sigma, the guidance is too weak
                 (gamma too large, or R too loose); much closer, and it is
                 over-fitting noise it should be smoothing.

    withheld     Stations excluded from the gauge likelihood.  This is honest
                 spatial validation in a gauge-only run.  With a dense
                 pseudo-satellite it is not fully independent: it tests how the
                 model allocates rain inside an observed coarse footprint.

    full field   Every land cell.  Separate coarse-footprint skill from the
                 unobserved 0.05-degree subgrid residual when satellite data are
                 present; otherwise direct coarse constraints can dominate.

WHAT IS MEASURED
    background   checkpoint-conditioned generation, no observations (control)
    analysis     the same, guided by the assimilated pseudo-gauges
    improvement  the reduction in error from one to the other

    Reporting both, on identical days with identical seeds, is the point.  An
    analysis RMSE in isolation says nothing; the drop from the background is the
    quantity that measures the assimilation.

OBSERVATION ERROR
    Added in TRANSFORMED space with standard deviation sqrt(sigma_obs^2 +
    representativeness^2), which is exactly what ``build_R`` tells the likelihood
    to expect.  Perturbing in mm and then transforming would make the assumed R
    wrong and would quietly mis-tune everything downstream.

    ``--obs-error perfect`` sets it small but NOT zero.  The likelihood variance
    is V(t) = R + gamma*(1-t)^2/t^2, which collapses to R as t -> 1; with R ~ 0
    the gradient explodes and ``clip_norm`` reduces it to a pure direction with
    no magnitude.  The first run of this experiment used R = 1e-3 and the
    analysis came out 475% WORSE than the background at the very stations it had
    assimilated.  That is a property of the guidance formulation, not of the
    model, and it is why R is floored at 0.05.

    python scripts/10_osse.py --ckpt runs/prior_h100_cpc/best.pt \
        --networks 40 --station-layout spread --withhold 0.2 \
        --pseudo-satellite --days 30 --members 16
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bdhires.da import (  # noqa: E402
    BilinearObsOperator,
    CompositeObsOperator,
    GuidanceConfig,
    PhysicalBilinearObsOperator,
    PhysicalBlockAverageObsOperator,
    SamplerConfig,
    StationSet,
    build_R_multi,
    perturb_observations,
)
from bdhires.da.sampler import assimilate as run_assim  # noqa: E402
from bdhires.data import DatasetConfig, PrecipDataset, load_stations  # noqa: E402
from bdhires.eval import crps_ensemble  # noqa: E402
from bdhires.grids import WIDE, crop_offsets, get_grid  # noqa: E402
from bdhires.models import RectifiedFlow, UNet, select_weights  # noqa: E402
from bdhires.transforms import (  # noqa: E402
    load_climatology,
    CondTransform,
    PrecipTransform,
    ResidualSpec,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/da.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--members", type=int, default=16)
    parser.add_argument("--days", type=int, default=20)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--month", type=int, default=7)
    parser.add_argument(
        "--networks",
        default="10,25,50,100,200,bmd",
        help="comma-separated station counts, plus 'bmd' for the real geometry",
    )
    parser.add_argument(
        "--obs-error",
        default="realistic,perfect",
        help="comma-separated: 'realistic' uses da.yaml, 'perfect' uses ~0",
    )
    parser.add_argument(
        "--pseudo-satellite",
        action="store_true",
        help="also assimilate CHIRPS averaged to exact nested 0.1-degree "
             "footprints; this is the pseudo-IMERG OSSE requested for the "
             "combined satellite + station experiment",
    )
    parser.add_argument(
        "--observation-mode",
        choices=["gauges", "satellite", "combined"],
        default=None,
        help="matched OSSE arm. Default preserves legacy behaviour: gauges, or "
             "combined when --pseudo-satellite is present",
    )
    parser.add_argument(
        "--satellite-stride",
        type=int,
        default=1,
        help="retain every Nth pseudo-satellite footprint in each direction",
    )
    parser.add_argument(
        "--satellite-correlation-control",
        action="store_true",
        help="inflate pseudo-satellite R by 2*pi*(correlation_length/stride)^2 "
             "to avoid treating correlated footprints as independent",
    )
    parser.add_argument(
        "--withhold",
        type=float,
        default=0.4,
        help="fraction of stations held out of the assimilation for scoring",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="sweep gamma / sigma_obs / prior_temperature instead of network size",
    )
    parser.add_argument(
        "--station-layout",
        default="random",
        choices=["random", "spread"],
        help="'spread' uses farthest-point sampling for even coverage, which is "
             "what a real gauge network looks like. Matters most for sparse "
             "networks, where a uniform draw leaves big gaps and clustered pairs.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-figure", default="data/processed/osse.png")
    parser.add_argument("--out-report", default="data/processed/osse.json")
    parser.add_argument(
        "--dump",
        default=None,
        help="write the raw ensembles for ONE configuration to this .npz so "
             "scripts/11_da_diagnostics.py can plot the verification suite "
             "without re-running the sampling",
    )
    parser.add_argument(
        "--dump-network",
        default=None,
        help="which network to dump (default: the largest synthetic one)",
    )
    parser.add_argument("--dump-obs-error", default="realistic")
    parser.add_argument(
        "--tune-network",
        default=None,
        help="network to tune on (default: the LARGEST). Tuning on a sparse "
             "network is a trap: with 10 stations and 40%% withheld the score "
             "rests on 4 stations and the setting-to-setting differences are "
             "swamped by sampling noise.",
    )
    parser.add_argument(
        "--tune-days",
        type=int,
        default=None,
        help="days to use in tuning mode (default: --days). The grid is "
             "gamma x sigma x temperature, so cost multiplies fast.",
    )
    return parser.parse_args()


def synthetic_network(
    n: int, grid, valid: np.ndarray, rng: np.random.Generator,
    layout: str = "random",
) -> StationSet:
    """Station locations over land, at least a cell from the boundary.

    ``layout="random"`` draws uniformly, which clusters: with 35 stations a
    uniform draw leaves large gaps and several near-duplicate pairs, and a pair
    two cells apart carries barely more information than one station.

    ``layout="spread"`` uses greedy farthest-point sampling -- repeatedly add the
    land cell furthest from everything chosen so far.  That is much closer to how
    a real gauge network is laid out, and for a sparse network it is the
    difference between covering the domain and sampling a few corners of it.
    """
    land = np.argwhere(valid)
    margin = 2
    inside = land[
        (land[:, 0] >= margin)
        & (land[:, 0] < valid.shape[0] - margin)
        & (land[:, 1] >= margin)
        & (land[:, 1] < valid.shape[1] - margin)
    ]
    if n > len(inside):
        raise ValueError(f"cannot place {n} stations in {len(inside)} land cells")

    if layout == "spread":
        chosen = [int(rng.integers(len(inside)))]
        distance = np.hypot(
            inside[:, 0] - inside[chosen[0], 0],
            inside[:, 1] - inside[chosen[0], 1],
        ).astype(np.float64)
        for _ in range(n - 1):
            nxt = int(np.argmax(distance))
            chosen.append(nxt)
            distance = np.minimum(
                distance,
                np.hypot(inside[:, 0] - inside[nxt, 0], inside[:, 1] - inside[nxt, 1]),
            )
        picked = inside[np.array(chosen)]
    elif layout == "random":
        picked = inside[rng.choice(len(inside), size=n, replace=False)]
    else:
        raise ValueError(f"unknown station layout {layout!r}")
    # jitter within the cell so stations are not all exactly on grid centres
    lat = grid.lat[picked[:, 0]] + (rng.random(n) - 0.5) * grid.res
    lon = grid.lon[picked[:, 1]] + (rng.random(n) - 0.5) * grid.res
    return StationSet(lat=lat, lon=lon, ids=np.arange(n))


def sample_at_stations(field_mm: np.ndarray, grid, stations: StationSet) -> np.ndarray:
    """Bilinear sample of (N, H, W) mm fields at station points -> (N, S)."""
    operator = BilinearObsOperator(grid, stations.lat, stations.lon)
    tensor = torch.from_numpy(np.nan_to_num(field_mm, nan=0.0)).float()
    if tensor.ndim == 2:
        tensor = tensor[None]
    return operator(tensor[:, None])[:, 0].numpy()


def block_mean_mm(
    field_mm: np.ndarray,
    valid: np.ndarray,
    factor: int = 2,
    min_valid_fraction: float = 0.999,
) -> tuple[np.ndarray, np.ndarray]:
    """Physical footprint means and validity on the nested coarse grid.

    The mean is taken in mm/day, before any nonlinear precipitation transform.
    Footprints touching the CHIRPS ocean mask are excluded from assimilation so
    zeros used for the numerical mean cannot leak into coastal observations.
    """
    if field_mm.shape != valid.shape:
        raise ValueError(f"field {field_mm.shape} and valid mask {valid.shape} differ")
    height, width = field_mm.shape
    if height % factor or width % factor:
        raise ValueError(f"grid {field_mm.shape} is not divisible by factor {factor}")
    shape = (height // factor, factor, width // factor, factor)
    support = valid.reshape(shape).mean(axis=(1, 3)) >= min_valid_fraction
    filled = np.where(valid, field_mm, 0.0)
    coarse = filled.reshape(shape).mean(axis=(1, 3)).astype(np.float32)
    return np.where(support, coarse, np.nan), support


def score(members_mm: np.ndarray, truth: np.ndarray) -> dict:
    """Ensemble scores for (M, ...) members against (...) truth."""
    finite = np.isfinite(truth)
    if not finite.any():
        return {}
    ensemble = members_mm[:, finite]
    observed = truth[finite]
    mean = ensemble.mean(axis=0)
    spread = ensemble.std(axis=0, ddof=1)
    difference = mean - observed
    correlation = (
        float(np.corrcoef(mean, observed)[0, 1])
        if mean.std() > 0 and observed.std() > 0
        else float("nan")
    )
    low, high = np.quantile(ensemble, [0.05, 0.95], axis=0)
    return {
        "rmse_mm": float(np.sqrt(np.mean(difference**2))),
        "mae_mm": float(np.mean(np.abs(difference))),
        "bias_mm": float(np.mean(difference)),
        "crps_mm": float(crps_ensemble(ensemble, observed)),
        "correlation": correlation,
        "spread_mm": float(spread.mean()),
        "coverage_90": float(np.mean((observed >= low) & (observed <= high))),
        "n": int(finite.sum()),
    }


def aggregate(per_day: list[dict]) -> dict:
    """Mean over days of each metric, ignoring days that produced nothing."""
    keys = {k for d in per_day for k in d}
    out = {}
    for key in sorted(keys):
        values = [d[key] for d in per_day if key in d and np.isfinite(d[key])]
        out[key] = float(np.mean(values)) if values else float("nan")
    return out


def improvement(background: dict, analysis: dict, key: str) -> float:
    """Percent reduction in an error metric, positive = analysis is better."""
    b, a = background.get(key), analysis.get(key)
    if b is None or a is None or not np.isfinite(b) or b == 0:
        return float("nan")
    return float(100.0 * (b - a) / b)


def main() -> None:
    args = parse_args()
    observation_mode = args.observation_mode
    if observation_mode is None:
        observation_mode = "combined" if args.pseudo_satellite else "gauges"
    args.observation_mode = observation_mode
    use_gauges = observation_mode in {"gauges", "combined"}
    use_satellite = observation_mode in {"satellite", "combined"}
    if args.satellite_stride < 1:
        raise ValueError("--satellite-stride must be >= 1")
    if not 0.0 <= args.withhold < 1.0:
        raise ValueError("--withhold must satisfy 0 <= fraction < 1")
    config = yaml.safe_load(Path(args.config).read_text())
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    training_config = checkpoint["cfg"]

    # The input representation is part of the checkpoint.  The CPC prior uses
    # a different Zarr store, statistics, residual base and conditioning subset
    # than the older ERA5 prior, so never take those values blindly from da.yaml.
    training_data = training_config["data"]
    data_zarr = str(training_data.get("zarr", config["data"]["zarr"]))
    data_stats = str(training_data.get("stats", config["data"]["stats"]))
    stats = json.loads(Path(data_stats).read_text())
    transform = PrecipTransform.from_dict(stats["precip_transform"])
    residual = ResidualSpec.from_stats(stats)
    grid = get_grid(config["data"]["grid"])

    selected_channels = training_data.get("cond_channels")
    print(
        f"[osse] checkpoint-bound data: {data_zarr} | {data_stats} | "
        f"channels={selected_channels or 'all'}",
        flush=True,
    )

    dataset = PrecipDataset(
        DatasetConfig(
            root=data_zarr,
            crop=grid.nlon,
            random_crop=False,
            crop_origin=crop_offsets(WIDE, grid),
            seasonal_encoding=bool(training_data.get("seasonal_encoding", True)),
            cond_channels=(
                tuple(selected_channels) if selected_channels is not None else None
            ),
            min_valid_fraction=float(training_data.get("min_valid_fraction", 0.3)),
        ),
        transform,
        cond_mean=np.asarray(stats["cond_mean"], np.float32),
        cond_std=np.asarray(stats["cond_std"], np.float32),
        cond_transform=CondTransform.from_stats(stats),
        residual=residual,
        climatology=load_climatology(data_stats, stats),
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

    # -- days -----------------------------------------------------------------
    times = dataset.time
    test_years = training_config["data"]["years"]["test"]
    start = np.datetime64(args.start or f"{test_years[0]}-01-01")
    end = np.datetime64(args.end or f"{test_years[1]}-12-31")
    eligible = np.where((times >= start) & (times <= end))[0]
    if args.month:
        months = times[eligible].astype("datetime64[M]").astype(int) % 12 + 1
        eligible = eligible[months == args.month]
    if not len(eligible):
        raise ValueError("no days match the requested window")
    days = eligible[
        np.linspace(0, len(eligible) - 1, min(args.days, len(eligible))).astype(int)
    ]

    # -- station networks -----------------------------------------------------
    rng = np.random.default_rng(args.seed)
    networks: dict[str, StationSet] = {}
    for token in args.networks.split(","):
        token = token.strip()
        if not token:
            continue
        if token.lower() == "bmd":
            csv = config["observations"]["gauges"]["csv"]
            if not Path(csv).is_file():
                print(f"[osse] skipping 'bmd': {csv} not found")
                continue
            bmd, _ = load_stations(csv, times, grid=grid, min_coverage=0.0)
            networks[f"bmd ({len(bmd)})"] = bmd
        else:
            n = int(token)
            networks[f"{n}"] = synthetic_network(
                n, grid, valid, rng, layout=args.station_layout
            )
    if not networks:
        raise ValueError("no station networks were built")
    for name, stations in networks.items():
        if len(stations) < 2:
            continue
        points = np.stack([stations.lat, stations.lon], axis=1)
        gaps = np.hypot(
            points[:, None, 0] - points[None, :, 0],
            (points[:, None, 1] - points[None, :, 1])
            * np.cos(np.radians(points[:, None, 0])),
        )
        np.fill_diagonal(gaps, np.inf)
        nearest = gaps.min(axis=1) / grid.res
        print(
            f"[osse] network '{name}': layout={args.station_layout}, "
            f"nearest-neighbour separation min {nearest.min():.1f} / "
            f"median {np.median(nearest):.1f} cells"
        )

    gauge_cfg = config["observations"]["gauges"]
    satellite_cfg = config["observations"]["imerg"]
    satellite_factor = int(satellite_cfg["factor"])
    if use_satellite and (
        grid.nlat % satellite_factor or grid.nlon % satellite_factor
    ):
        raise ValueError(
            f"grid {grid.nlat}x{grid.nlon} is not divisible by satellite factor "
            f"{satellite_factor}"
        )
    satellite_shape = (
        grid.nlat // satellite_factor,
        grid.nlon // satellite_factor,
    )
    satellite_count = int(np.prod(satellite_shape)) if use_satellite else 0
    satellite_selection = np.zeros(satellite_shape, dtype=bool)
    satellite_selection[::args.satellite_stride, ::args.satellite_stride] = True
    selected_satellite_count = int(satellite_selection.sum()) if use_satellite else 0
    satellite_r_inflation = 1.0
    if use_satellite and args.satellite_correlation_control:
        correlation_cells = float(satellite_cfg.get("error_corr_cells", 0.0))
        satellite_r_inflation = max(
            1.0,
            2.0 * np.pi * (correlation_cells / args.satellite_stride) ** 2,
        )
    error_levels = {}
    for token in args.obs_error.split(","):
        token = token.strip().lower()
        if token == "realistic":
            error_levels["realistic"] = (
                float(gauge_cfg["sigma_obs"]),
                float(gauge_cfg["representativeness"]),
            )
        elif token == "perfect":
            # NOT zero: the likelihood variance is V(t) = R + gamma(1-t)^2/t^2,
            # which collapses to R as t -> 1. With R ~ 0 the gradient explodes
            # and clip_norm turns it into a pure direction with no magnitude,
            # which destroys the analysis rather than sharpening it. 0.05 is
            # small enough to bound the achievable skill without that.
            error_levels["perfect"] = (0.05, 0.0)
    if not error_levels:
        raise ValueError("no observation-error levels selected")

    base_sampler = replace(
        SamplerConfig(**config["sampler"]), mask_fill=dataset.mask_fill
    )
    base_guidance = GuidanceConfig(**config["guidance"])

    if args.tune:
        combinations = [
            {"gamma": g, "sigma_obs": s, "prior_temperature": t}
            for g, s, t in itertools.product(
                config["tuning"]["gamma"],
                config["tuning"]["sigma_obs"],
                config["tuning"]["prior_temperature"],
            )
        ]
        if args.tune_network and args.tune_network in networks:
            chosen_network = args.tune_network
        else:
            synthetic = [n for n in networks if n.replace(" ", "").isdigit()]
            chosen_network = (
                max(synthetic, key=int) if synthetic else list(networks)[0]
            )
        networks = {chosen_network: networks[chosen_network]}
        error_levels = {"realistic": error_levels.get("realistic", (0.1, 0.25))}
        if args.tune_days:
            days = days[
                np.linspace(0, len(days) - 1, min(args.tune_days, len(days)))
                .astype(int)
            ]
        tune_station_count = len(networks[chosen_network])
        withheld = min(
            tune_station_count - 1,
            max(1, int(round(args.withhold * tune_station_count))),
        )
        print(
            f"[osse] tuning mode: {len(combinations)} combinations x "
            f"{len(days)} days on network '{chosen_network}' "
            f"({withheld} withheld stations)"
        )
        if withheld < 15:
            print(
                f"[osse] WARNING: only {withheld} withheld stations. Differences "
                f"between settings will be dominated by sampling noise; prefer a "
                f"denser --tune-network."
            )
    else:
        combinations = [{}]

    print(
        f"[osse] {len(days)} days x {args.members} members x "
        f"{len(networks)} networks x {len(error_levels)} error levels x "
        f"{len(combinations)} settings on {device}",
        flush=True,
    )
    print(
        f"[osse] observation mode={observation_mode}; gauges={use_gauges}; "
        f"pseudo-satellite={use_satellite}"
        + (
            f"; selected footprints={selected_satellite_count}/{satellite_count}; "
            f"R inflation={satellite_r_inflation:.2f}x"
            if use_satellite
            else ""
        ),
        flush=True,
    )

    # -- the background depends only on the day and the prior temperature -----
    # It must use the SAME prior temperature as the analysis, or the comparison
    # measures inflation as well as assimilation.  In tuning mode the temperature
    # varies, so the cache is keyed on it.
    background_cache: dict[tuple[int, float], np.ndarray] = {}
    truth_cache: dict[int, np.ndarray] = {}

    def background_for(index: int, temperature: float) -> np.ndarray:
        key = (int(index), round(float(temperature), 6))
        if key in background_cache:
            return background_cache[key]
        position = int(np.where(dataset.index == index)[0][0])
        item = dataset[position]
        base = item["base"][None].to(device)
        with torch.inference_mode():
            generated = run_assim(
                model,
                item["cond"][None].to(device),
                (args.members, 1, grid.nlat, grid.nlon),
                device,
                cfg=replace(
                    base_sampler,
                    seed=args.seed + int(index),
                    prior_temperature=temperature,
                ),
                flow=flow,
                mask=mask,
                to_precip=lambda x, b=base: residual.decode(x, b),
            )
        field = transform.inverse(
            residual.decode(generated, base)[:, 0].float().cpu().numpy()
        )
        background_cache[key] = np.where(valid[None], field, np.nan)
        return background_cache[key]

    for index in days:
        target = np.asarray(dataset.z["target"][int(index)][slices], dtype=np.float32)
        truth_cache[int(index)] = np.where(valid, target, np.nan)

    dump_target = args.dump_network
    if args.dump and dump_target is None:
        synthetic = [n for n in networks if n.replace(" ", "").isdigit()]
        dump_target = max(synthetic, key=int) if synthetic else list(networks)[0]
    dumped = None

    results = []
    for name, stations in networks.items():
        n_stations = len(stations)
        if n_stations < 2:
            raise ValueError(f"network {name!r} needs at least two stations")
        n_withhold = min(
            n_stations - 1, max(1, int(round(args.withhold * n_stations)))
        )
        split_rng = np.random.default_rng(args.seed + n_stations)
        order = split_rng.permutation(n_stations)
        eval_idx = np.sort(order[:n_withhold])
        assim_idx = np.sort(order[n_withhold:])
        active_assim_idx = assim_idx if use_gauges else np.array([], dtype=int)

        for error_name, (sigma, representativeness) in error_levels.items():
            for setting in combinations:
                sigma_used = setting.get("sigma_obs", sigma)
                guidance = replace(
                    base_guidance, gamma=setting.get("gamma", base_guidance.gamma)
                )
                sampler_cfg = replace(
                    base_sampler,
                    prior_temperature=setting.get(
                        "prior_temperature", base_sampler.prior_temperature
                    ),
                )
                gauge_noise_sd = float(
                    np.sqrt(sigma_used**2 + representativeness**2)
                )
                operators = []
                r_specs = []
                corr_blocks: list[tuple[int, int, int, float]] = []
                if use_gauges:
                    operators.append(PhysicalBilinearObsOperator(
                        grid,
                        stations.lat[active_assim_idx],
                        stations.lon[active_assim_idx],
                        transform,
                        valid=valid,
                    ))
                    r_specs.append(
                        (len(active_assim_idx), sigma_used, representativeness)
                    )
                satellite_noise_sd = float("nan")
                if use_satellite:
                    operators.append(
                        PhysicalBlockAverageObsOperator(
                            satellite_factor, transform, valid=valid
                        )
                    )
                    if error_name == "perfect":
                        satellite_sigma, satellite_repr = 0.05, 0.0
                    else:
                        satellite_sigma = float(satellite_cfg["sigma_obs"])
                        satellite_repr = float(
                            satellite_cfg["representativeness"]
                        )
                    r_specs.append(
                        (satellite_count, satellite_sigma, satellite_repr)
                    )
                    satellite_noise_sd = float(
                        np.sqrt(satellite_sigma**2 + satellite_repr**2)
                    )
                    corr_blocks.append(
                        (
                            len(active_assim_idx),
                            satellite_shape[0],
                            satellite_shape[1],
                            float(satellite_cfg.get("error_corr_cells", 0.0)),
                        )
                    )
                R = build_R_multi(r_specs, device=device)
                if use_satellite and satellite_r_inflation != 1.0:
                    R[len(active_assim_idx):] *= satellite_r_inflation
                operator = (
                    CompositeObsOperator(operators)
                    if len(operators) > 1
                    else operators[0]
                ).to(device)

                background_days, analysis_days = [], []
                field_background, field_analysis = [], []
                assim_background, assim_analysis = [], []
                fit_transformed = []
                collect = (
                    args.dump
                    and name == dump_target
                    and error_name == args.dump_obs_error
                    and not setting
                )
                store: dict[str, list] = {
                    k: [] for k in (
                        "background", "analysis", "truth",
                        "obs_transformed", "truth_at_stations",
                        "pseudo_satellite_mm", "pseudo_satellite_truth_mm",
                        # The 0.5-degree conditioning precipitation mapped onto
                        # the fine grid.  Without it no script downstream can
                        # measure what downscaling actually bought, because the
                        # honest null for that question is the coarse input
                        # itself, not a zero field.
                        "coarse_base_mm",
                    )
                }
                for index in days:
                    index = int(index)
                    position = int(np.where(dataset.index == index)[0][0])
                    item = dataset[position]
                    base = item["base"][None].to(device)
                    truth = truth_cache[index]
                    # The coarse precipitation the prior starts from, in mm.
                    # `residual.fill` -- not zero -- is the network-space value
                    # meaning "no correction": decode() adds the standardisation
                    # mean back, so feeding zeros would offset the whole field by
                    # mu_r and quietly bias every downscaling-gain score.
                    coarse_base_mm = transform.inverse(
                        residual.decode(
                            torch.full_like(base, residual.fill), base
                        )[:, 0].float().cpu().numpy()
                    )[0]

                    # Observations: truth sampled at stations, error added in
                    # TRANSFORMED space so it matches what R claims.
                    truth_at_stations = sample_at_stations(truth, grid, stations)[0]
                    observation_rng = np.random.default_rng(args.seed + index)
                    gauge_obs_transformed = transform.forward(
                        np.clip(truth_at_stations, 0.0, None)
                    ) + observation_rng.normal(
                        0.0, gauge_noise_sd, n_stations
                    )
                    truth_obs = []
                    observed = []
                    if use_gauges:
                        truth_obs.append(
                            transform.forward(
                                np.clip(
                                    truth_at_stations[active_assim_idx], 0.0, None
                                )
                            ).astype(np.float32)
                        )
                        observed.append(
                            gauge_obs_transformed[active_assim_idx].astype(np.float32)
                        )
                    satellite_truth_mm = None
                    satellite_observed_mm = None
                    if use_satellite:
                        satellite_truth_mm, _ = block_mean_mm(
                            truth, valid, satellite_factor
                        )
                        satellite_truth_transformed = transform.forward(
                            satellite_truth_mm
                        ).astype(np.float32).reshape(-1)
                        satellite_truth_transformed[
                            ~satellite_selection.reshape(-1)
                        ] = np.nan
                        # One correlated error realisation creates the actual
                        # pseudo-IMERG product.  Member-wise perturbations below
                        # then preserve posterior ensemble variance.
                        satellite_R = R[len(active_assim_idx):]
                        satellite_observed = perturb_observations(
                            satellite_truth_transformed,
                            satellite_R,
                            1,
                            seed=args.seed + index + 100_000,
                            corr_blocks=[
                                (
                                    0,
                                    satellite_shape[0],
                                    satellite_shape[1],
                                    float(
                                        satellite_cfg.get("error_corr_cells", 0.0)
                                    ),
                                )
                            ],
                        )[0].astype(np.float32)
                        satellite_observed[
                            ~np.isfinite(satellite_truth_transformed)
                        ] = np.nan
                        truth_obs.append(satellite_truth_transformed)
                        observed.append(satellite_observed)
                        satellite_observed_mm = transform.inverse(
                            satellite_observed
                        ).reshape(satellite_shape).astype(np.float32)

                    y_truth = np.concatenate(truth_obs).astype(np.float32)
                    y_assim = np.concatenate(observed).astype(np.float32)
                    y_assim[~np.isfinite(y_truth)] = np.nan

                    perturbed = perturb_observations(
                        y_assim,
                        R,
                        args.members,
                        seed=index,
                        corr_blocks=corr_blocks,
                    )
                    y_tensor = torch.from_numpy(
                        perturbed[:, None].astype(np.float32)
                    ).to(device)

                    # NOT under torch.inference_mode(): guidance differentiates
                    # the likelihood back through the network, and inference mode
                    # permanently marks its tensors as unusable by autograd --
                    # torch.enable_grad() inside it does not help.  The sampler
                    # already applies no_grad() where it is safe to.
                    generated = run_assim(
                        model,
                        item["cond"][None].to(device),
                        (args.members, 1, grid.nlat, grid.nlon),
                        device,
                        H=operator,
                        y=y_tensor,
                        R=R,
                        cfg=replace(sampler_cfg, seed=args.seed + index),
                        gcfg=guidance,
                        flow=flow,
                        mask=mask,
                        to_precip=lambda x, b=base: residual.decode(x, b),
                    ).detach()
                    analysis = transform.inverse(
                        residual.decode(generated, base)[:, 0].float().cpu().numpy()
                    )
                    analysis = np.where(valid[None], analysis, np.nan)
                    background = background_for(
                        index, sampler_cfg.prior_temperature
                    )

                    # HEADLINE: withheld stations only.
                    truth_eval = truth_at_stations[eval_idx]
                    background_days.append(
                        score(
                            sample_at_stations(background, grid, stations)[:, eval_idx],
                            truth_eval,
                        )
                    )
                    analysis_days.append(
                        score(
                            sample_at_stations(analysis, grid, stations)[:, eval_idx],
                            truth_eval,
                        )
                    )
                    # Circular but useful OSSE score at assimilated locations,
                    # against nature truth. The transformed-space fit below is
                    # the separate diagnostic against the noisy observation the
                    # likelihood was actually handed.
                    if use_gauges:
                        analysis_at_assim = sample_at_stations(
                            analysis, grid, stations
                        )[:, active_assim_idx]
                        background_at_assim = sample_at_stations(
                            background, grid, stations
                        )[:, active_assim_idx]
                        assim_analysis.append(
                            score(
                                analysis_at_assim,
                                truth_at_stations[active_assim_idx],
                            )
                        )
                        assim_background.append(
                            score(
                                background_at_assim,
                                truth_at_stations[active_assim_idx],
                            )
                        )
                        # Distance from the ensemble mean to the gauge observation
                        # in the transformed units used by the likelihood.
                        fit_transformed.append(
                            float(
                                np.sqrt(
                                    np.mean(
                                        (
                                            transform.forward(
                                                np.clip(
                                                    analysis_at_assim.mean(axis=0),
                                                    0,
                                                    None,
                                                )
                                            )
                                            - gauge_obs_transformed[active_assim_idx]
                                        )
                                        ** 2
                                    )
                                )
                            )
                        )

                    # Secondary: the whole field.
                    field_background.append(score(background, truth))
                    field_analysis.append(score(analysis, truth))

                    if collect:
                        store["background"].append(background.astype(np.float32))
                        store["analysis"].append(analysis.astype(np.float32))
                        store["truth"].append(truth.astype(np.float32))
                        store["coarse_base_mm"].append(
                            np.where(valid, coarse_base_mm, np.nan).astype(np.float32)
                        )
                        store["obs_transformed"].append(
                            gauge_obs_transformed.astype(np.float32)
                        )
                        store["truth_at_stations"].append(
                            truth_at_stations.astype(np.float32)
                        )
                        if use_satellite:
                            store["pseudo_satellite_mm"].append(
                                satellite_observed_mm
                            )
                            store["pseudo_satellite_truth_mm"].append(
                                satellite_truth_mm.astype(np.float32)
                            )

                entry = {
                    "network": name,
                    "n_stations": n_stations,
                    "n_assimilated": int(len(active_assim_idx)),
                    "n_withheld": int(len(eval_idx)),
                    "obs_error": error_name,
                    "observation_mode": observation_mode,
                    "obs_noise_sd_transformed": gauge_noise_sd,
                    "pseudo_satellite": bool(use_satellite),
                    "satellite_noise_sd_transformed": satellite_noise_sd,
                    "satellite_stride": int(args.satellite_stride),
                    "satellite_selected_footprints": selected_satellite_count,
                    "satellite_r_inflation": satellite_r_inflation,
                    **{f"setting_{k}": v for k, v in setting.items()},
                    "withheld_background": aggregate(background_days),
                    "withheld_analysis": aggregate(analysis_days),
                    "assimilated_background": aggregate(assim_background),
                    "assimilated_analysis": aggregate(assim_analysis),
                    "field_background": aggregate(field_background),
                    "field_analysis": aggregate(field_analysis),
                    # Consistency check: analysis-minus-observation distance in
                    # transformed units against the sd the likelihood assumed.
                    "fit_rms_transformed": float(np.mean(fit_transformed))
                    if fit_transformed
                    else float("nan"),
                    "assumed_obs_sd_transformed": gauge_noise_sd,
                    "fit_ratio": float(np.mean(fit_transformed) / gauge_noise_sd)
                    if fit_transformed and gauge_noise_sd > 0
                    else float("nan"),
                }
                for metric in ("rmse_mm", "crps_mm", "mae_mm"):
                    for scope in ("withheld", "assimilated", "field"):
                        entry[f"{scope}_improvement_{metric}"] = improvement(
                            entry[f"{scope}_background"],
                            entry[f"{scope}_analysis"],
                            metric,
                        )
                if collect and store["truth"]:
                    dumped = dict(
                        background=np.stack(store["background"]),
                        analysis=np.stack(store["analysis"]),
                        truth=np.stack(store["truth"]),
                        array_layout=np.str_("day,member,latitude,longitude"),
                        coarse_base_mm=np.stack(store["coarse_base_mm"]),
                        grid_lat=np.asarray(grid.lat, dtype=np.float32),
                        grid_lon=np.asarray(grid.lon, dtype=np.float32),
                        grid_res=np.float32(grid.res),
                        obs_transformed=np.stack(store["obs_transformed"]),
                        truth_at_stations=np.stack(store["truth_at_stations"]),
                        station_lat=stations.lat,
                        station_lon=stations.lon,
                        assim_idx=active_assim_idx,
                        eval_idx=eval_idx,
                        valid=valid,
                        obs_noise_sd=np.float32(gauge_noise_sd),
                        satellite_obs_noise_sd=np.float32(satellite_noise_sd),
                        pseudo_satellite_enabled=np.bool_(use_satellite),
                        satellite_factor=np.int32(satellite_factor),
                        satellite_stride=np.int32(args.satellite_stride),
                        satellite_selection=satellite_selection,
                        satellite_r_inflation=np.float32(satellite_r_inflation),
                        observation_mode=np.str_(observation_mode),
                        grid_name=np.str_(grid.name),
                        network=np.str_(name),
                        obs_error=np.str_(error_name),
                        days=np.array(
                            [str(times[int(i)].astype("datetime64[D]")) for i in days]
                        ),
                        precip_transform=np.str_(
                            json.dumps(stats["precip_transform"])
                        ),
                        checkpoint=np.str_(args.ckpt),
                        checkpoint_epoch=np.int32(checkpoint.get("epoch", -1)),
                        data_zarr=np.str_(data_zarr),
                        data_stats=np.str_(data_stats),
                    )
                    if use_satellite:
                        dumped["pseudo_satellite_mm"] = np.stack(
                            store["pseudo_satellite_mm"]
                        )
                        dumped["pseudo_satellite_truth_mm"] = np.stack(
                            store["pseudo_satellite_truth_mm"]
                        )

                results.append(entry)
                print(
                    f"  {name:>12s}  {error_name:<9s} "
                    + (f"{setting} " if setting else "")
                    + f"withheld CRPS {entry['withheld_background']['crps_mm']:6.2f} -> "
                    f"{entry['withheld_analysis']['crps_mm']:6.2f} "
                    f"({entry['withheld_improvement_crps_mm']:+5.1f}%)   "
                    f"assim {entry['assimilated_improvement_crps_mm']:+5.1f}%   "
                    f"field {entry['field_improvement_crps_mm']:+5.1f}%   "
                    f"fit {entry['fit_ratio']:.2f}x sigma",
                    flush=True,
                )

    report = {
        "checkpoint": str(args.ckpt),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "days": [str(times[int(i)].astype("datetime64[D]")) for i in days],
        "members": args.members,
        "withhold_fraction": args.withhold,
        "data_zarr": data_zarr,
        "data_stats": data_stats,
        "observation_mode": observation_mode,
        "pseudo_satellite": bool(use_satellite),
        "satellite_factor": satellite_factor if use_satellite else None,
        "satellite_stride": args.satellite_stride if use_satellite else None,
        "satellite_selected_footprints": selected_satellite_count
        if use_satellite
        else None,
        "satellite_r_inflation": satellite_r_inflation if use_satellite else None,
        "station_layout": args.station_layout,
        "mode": "tuning" if args.tune else "network sweep",
        "note": (
            "This is an optimistic upper-bound OSSE because CHIRPS supplies both "
            "the nature truth and pseudo-observations. Withheld gauges are not "
            "fully independent when dense pseudo-satellite footprints are also "
            "assimilated; they test sub-footprint allocation at unseen point "
            "locations. Read the 0.1-degree footprint and 0.05-degree subgrid "
            "scores separately before concluding that fine structure was recovered."
        ),
        "results": results,
    }
    Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_report).write_text(json.dumps(report, indent=2) + "\n")

    if args.dump:
        if dumped is None:
            print(f"[osse] nothing dumped: no configuration matched "
                  f"network={dump_target!r} obs_error={args.dump_obs_error!r}")
        else:
            Path(args.dump).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.dump, **dumped)
            size = Path(args.dump).stat().st_size / 2**20
            print(f"wrote {args.dump} ({size:.0f} MiB) "
                  f"-- network '{dumped['network']}', {dumped['obs_error']} obs")

    plot_results(results, args, Path(args.out_figure))
    print(f"\nwrote {args.out_figure}")
    print(f"wrote {args.out_report}")


def plot_results(results: list[dict], args, output: Path) -> None:
    if not results:
        return
    figure, axes = plt.subplots(2, 3, figsize=(18, 9.5), constrained_layout=True)
    colours = plt.get_cmap("tab10").colors
    levels = sorted({r["obs_error"] for r in results})

    panels = [
        ("crps_mm", "CRPS (mm day$^{-1}$)", "A.  Withheld-station CRPS", "withheld"),
        ("rmse_mm", "RMSE (mm day$^{-1}$)", "B.  Withheld-station RMSE", "withheld"),
        ("coverage_90", "Fraction", "C.  Withheld 90% coverage", "withheld"),
        ("crps_mm", "CRPS (mm day$^{-1}$)", "D.  Full-field CRPS (secondary)", "field"),
        ("spread_mm", "mm day$^{-1}$", "E.  Full-field ensemble spread", "field"),
    ]
    for axis, (metric, ylabel, title, scope) in zip(axes.ravel(), panels):
        for number, level in enumerate(levels):
            rows = [r for r in results if r["obs_error"] == level]
            rows.sort(key=lambda r: r["n_stations"])
            x = [r["n_assimilated"] for r in rows]
            axis.plot(
                x, [r[f"{scope}_background"].get(metric, np.nan) for r in rows],
                marker="s", ms=4, ls="--", color=colours[number],
                label=f"background ({level})", alpha=0.6,
            )
            axis.plot(
                x, [r[f"{scope}_analysis"].get(metric, np.nan) for r in rows],
                marker="o", ms=5, color=colours[number], label=f"analysis ({level})",
            )
        if metric == "coverage_90":
            axis.axhline(0.90, color="black", ls="--", lw=1.0, label="nominal 0.90")
        if any(r["n_assimilated"] == 0 for r in results):
            axis.set_xscale("symlog", linthresh=1)
        else:
            axis.set_xscale("log")
        axis.set_xlabel("Stations assimilated")
        axis.set_ylabel(ylabel)
        axis.set_title(title, fontsize=10.5)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, frameon=False)

    axis = axes.ravel()[5]
    for number, level in enumerate(levels):
        rows = [r for r in results if r["obs_error"] == level]
        rows.sort(key=lambda r: r["n_stations"])
        axis.plot(
            [r["n_assimilated"] for r in rows],
            [r["withheld_improvement_crps_mm"] for r in rows],
            marker="o", ms=5, color=colours[number], label=f"withheld ({level})",
        )
        axis.plot(
            [r["n_assimilated"] for r in rows],
            [r["field_improvement_crps_mm"] for r in rows],
            marker="s", ms=4, ls="--", color=colours[number],
            label=f"full field ({level})", alpha=0.6,
        )
    axis.axhline(0.0, color="black", lw=0.9)
    if any(r["n_assimilated"] == 0 for r in results):
        axis.set_xscale("symlog", linthresh=1)
    else:
        axis.set_xscale("log")
    axis.set_xlabel("Stations assimilated")
    axis.set_ylabel("CRPS reduction (%)")
    axis.set_title("F.  What the assimilation buys", fontsize=10.5)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, frameon=False)

    figure.suptitle(
        "BDhighresDA OSSE - CHIRPS nature truth; "
        + {
            "gauges": "pseudo-gauges",
            "satellite": "0.1-degree pseudo-satellite",
            "combined": "0.1-degree pseudo-satellite + pseudo-gauges",
        }[args.observation_mode]
        + "\n"
        f"{args.ckpt}   |   {args.days} days   |   {args.members} members   |   "
        f"{args.withhold:.0%} of each network withheld from assimilation\n"
        "Dashed = background (no observations), solid = analysis. "
        "The gap between them IS the assimilation.",
        fontsize=13,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=115)
    plt.close(figure)


if __name__ == "__main__":
    main()
