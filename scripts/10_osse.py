#!/usr/bin/env python
"""Observing-system simulation experiment: how much does assimilation buy?

CHIRPS is treated as the nature run.  Pseudo-gauges sample it at station
locations and, optionally, exact nested block means provide configurable
pseudo-satellite footprints. The primary ``exact`` experiment does not perturb either
pseudo-observation stream; the small positive likelihood variance is only a
numerical regulariser. A subset of gauges plus the dense satellite are
assimilated, and the 0.05-degree analysis is scored against the truth. Because
the true full field is known, this answers questions that real
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
    ``--obs-error exact`` is the primary mechanistic OSSE. It passes exact
    same-day CHIRPS values at gauge locations and exact physical CHIRPS block
    means. It adds no shared or member-wise random error. R remains small and
    positive only because the diffusion-posterior likelihood is singular at
    zero variance near the final integration time.

    The legacy ``realistic`` sensitivity adds Gaussian error in transformed
    space. Because inverse transformation exponentiates that error, it is not a
    calibrated IMERG simulator and must not be used as the primary OSSE.

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
from bdhires.bmd import (  # noqa: E402
    max_bearing_gap_deg,
    nearest_neighbour_km,
    neighbored_holdout,
    read_station_catalog,
    spread_holdout,
)
from bdhires.data import DatasetConfig, PrecipDataset  # noqa: E402
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
        "--months",
        default=None,
        help="comma-separated months. When supplied, days are selected evenly "
             "across every available year-month group (for example, "
             "--months 6,7,8 --days 12 over 2021--2024 selects one day from "
             "each month in each year). Overrides --month.",
    )
    parser.add_argument(
        "--networks",
        default="10,25,50,100,200,bmd",
        help="comma-separated station counts, plus 'bmd' for the real geometry",
    )
    parser.add_argument(
        "--bmd-stations",
        default="data/bmd/Stations.csv",
        help="BMD coordinate catalogue used by the 'bmd' network token. The "
             "rainfall column is not used: pseudo-gauge values come from CHIRPS.",
    )
    parser.add_argument(
        "--obs-error",
        default="exact",
        help="comma-separated: 'exact' uses noiseless CHIRPS pseudo-values with "
             "a small numerical likelihood variance; 'realistic' uses the "
             "legacy synthetic-error experiment; 'perfect' is retained as a "
             "backward-compatible small-noise experiment",
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
        "--satellite-factor",
        type=int,
        default=None,
        help="override the configured fine-grid footprint factor. On the 0.05° "
             "grid, factors 2, 4, and 10 represent 0.1°, 0.2°, and 0.5°.",
    )
    parser.add_argument(
        "--satellite-crop",
        type=int,
        nargs=4,
        metavar=("ROW_START", "ROW_STOP", "COL_START", "COL_STOP"),
        default=None,
        help="optional fine-grid window used by the footprint operator. This "
             "allows non-divisor factors to share identical spatial coverage; "
             "all four indices use Python half-open slicing.",
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
    parser.add_argument(
        "--holdout-layout",
        default="spread",
        choices=["random", "spread", "neighbored"],
        help="how withheld gauges are chosen. 'spread' seeds at the station "
             "farthest from the centroid and samples farthest-point, so it "
             "selects the most ISOLATED stations by construction -- a "
             "maximally adversarial verification set, since a gauge with no "
             "neighbour inside the background correlation length can only be "
             "reconstructed from the prior. 'neighbored' spreads the selection "
             "among stations that HAVE a neighbour within "
             "--holdout-neighbor-km, which measures the assimilation rather "
             "than the prior. Report it as such, and alongside 'spread'.",
    )
    parser.add_argument(
        "--holdout-neighbor-km",
        type=float,
        default=75.0,
        help="neighbour radius for --holdout-layout neighbored. 75 km is "
             "inside the ~146 km variogram range measured for this network, "
             "so an eligible station has genuinely informative neighbours. "
             "The check is applied against the ASSIMILATED stations, not all "
             "stations, so a holdout cannot strip its own support.",
    )
    parser.add_argument(
        "--holdout-max-gap-deg",
        type=float,
        default=200.0,
        help="largest permitted angular gap between bearings to a candidate's "
             "nearest neighbours. Excludes stations at the network EDGE, whose "
             "neighbours lie on one side only, so reconstructing them is "
             "extrapolation rather than interpolation. 180 is strictly "
             "interior; 200 keeps a workable pool while still excluding the "
             "clearly peripheral (the old spread holdout reached 322).",
    )
    parser.add_argument(
        "--guidance-spread-cells",
        type=float,
        default=None,
        help="override guidance.spread_cells: Gaussian spreading of the "
             "likelihood gradient in grid cells (1 cell ~ 5.5 km). Without it "
             "a point gauge only touches the 4 cells a bilinear operator "
             "reaches, and the correction does not propagate.",
    )
    parser.add_argument(
        "--satellite-sigma",
        type=float,
        default=None,
        help="observation sigma for the pseudo-satellite in the exact/perfect "
             "case, overriding the hardcoded 0.05. That value was chosen so the "
             "likelihood would not explode with ~34 GAUGES; with a few hundred "
             "block observations the aggregate gradient is far larger, "
             "clip_norm turns it into a pure direction with no magnitude, and "
             "the analysis diverges -- observed at 55x sigma misfit against "
             "observations accurate to 8e-06 mm/day.",
    )
    parser.add_argument(
        "--satellite-source",
        choices=["truth", "cpc"],
        default="truth",
        help="what the pseudo-satellite OBSERVES. 'truth' is block means of the "
             "CHIRPS nature run -- perfect coarse information, the upper bound "
             "on what coarse forcing can do. 'cpc' assimilates the 0.5-degree "
             "CPC conditioning field instead, which is what actually exists in "
             "the real system. Note CPC is ALSO the prior's input channel, so "
             "assimilating it adds no information: it forces the model to "
             "HONOUR conditioning it already had. And CPC is not the nature "
             "run, so it is scored against a field it genuinely differs from; "
             "read the pair, not either alone.",
    )
    parser.add_argument(
        "--guidance-gamma", type=float, default=None,
        help="override guidance.gamma without going through --tune",
    )
    parser.add_argument(
        "--prior-temperature", type=float, default=None,
        help="override sampler.prior_temperature without going through --tune. "
             "Raising it widens the prior ensemble, which is the other candidate "
             "for a background too smooth to carry an increment.",
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
    parser.add_argument("--dump-obs-error", default="exact")
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


def select_nature_days(
    times: np.ndarray,
    start: np.datetime64,
    end: np.datetime64,
    requested_days: int,
    month: int = 0,
    months_csv: str | None = None,
) -> np.ndarray:
    """Select deterministic nature-run days, optionally balanced by year-month.

    The old single-month behaviour is retained when ``months_csv`` is absent.
    A multi-month experiment is explicitly stratified so a short run cannot
    accidentally become an all-season linspace sample dominated by endpoints.
    """
    if requested_days < 1:
        raise ValueError("--days must be >= 1")
    eligible = np.where((times >= start) & (times <= end))[0]
    eligible_years = np.unique(times[eligible].astype("datetime64[Y]"))

    requested_months: list[int] | None = None
    if months_csv is not None:
        requested_months = sorted(
            {int(value.strip()) for value in months_csv.split(",") if value.strip()}
        )
        if not requested_months or any(
            value < 1 or value > 12 for value in requested_months
        ):
            raise ValueError("--months must contain comma-separated values in 1..12")
        calendar_month = times[eligible].astype("datetime64[M]").astype(int) % 12 + 1
        eligible = eligible[np.isin(calendar_month, requested_months)]
    elif month:
        if month < 1 or month > 12:
            raise ValueError("--month must be 0 or a value in 1..12")
        calendar_month = times[eligible].astype("datetime64[M]").astype(int) % 12 + 1
        eligible = eligible[calendar_month == month]

    if not len(eligible):
        raise ValueError("no days match the requested window and months")

    count = min(requested_days, len(eligible))
    if requested_months is None:
        return eligible[np.linspace(0, len(eligible) - 1, count).astype(int)]

    # One group for each available year-month, kept in chronological order.
    group_labels = times[eligible].astype("datetime64[M]")
    unique_groups = np.unique(group_labels)
    expected_group_count = len(eligible_years) * len(requested_months)
    if len(unique_groups) != expected_group_count:
        raise ValueError(
            "incomplete balanced month coverage: expected "
            f"{expected_group_count} year-month groups but found "
            f"{len(unique_groups)}"
        )
    groups = [eligible[group_labels == label] for label in unique_groups]

    if count < len(groups):
        chosen = np.linspace(0, len(groups) - 1, count).round().astype(int)
        groups = [groups[index] for index in np.unique(chosen)]
        count = len(groups)

    # Allocate as evenly as possible, then choose evenly spaced interior dates
    # within each group. For 12 JJA samples over four years every quota is one.
    quotas = np.zeros(len(groups), dtype=int)
    remaining = count
    while remaining:
        progressed = False
        for group_index, group in enumerate(groups):
            if remaining == 0:
                break
            if quotas[group_index] < len(group):
                quotas[group_index] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break

    selected: list[int] = []
    for group, quota in zip(groups, quotas):
        if quota:
            positions = np.floor(
                (np.arange(quota, dtype=float) + 0.5) * len(group) / quota
            ).astype(int)
            selected.extend(int(group[position]) for position in positions)
    return np.asarray(sorted(selected), dtype=int)


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
    crop: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Physical footprint means and validity on the nested coarse grid.

    The mean is taken in mm/day, before any nonlinear precipitation transform.
    Footprints touching the CHIRPS ocean mask are excluded from assimilation so
    zeros used for the numerical mean cannot leak into coastal observations.
    """
    if field_mm.shape != valid.shape:
        raise ValueError(f"field {field_mm.shape} and valid mask {valid.shape} differ")
    if crop is not None:
        row_start, row_stop, col_start, col_stop = crop
        field_mm = field_mm[row_start:row_stop, col_start:col_stop]
        valid = valid[row_start:row_stop, col_start:col_stop]
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
    rmse = float(np.sqrt(np.mean(difference**2)))
    finite_sample_inflation = np.sqrt((ensemble.shape[0] + 1) / ensemble.shape[0])
    return {
        "rmse_mm": rmse,
        "mae_mm": float(np.mean(np.abs(difference))),
        "bias_mm": float(np.mean(difference)),
        "crps_mm": float(crps_ensemble(ensemble, observed)),
        "correlation": correlation,
        "spread_mm": float(spread.mean()),
        "spread_skill": (
            float(spread.mean() * finite_sample_inflation / rmse)
            if rmse > 0
            else float("nan")
        ),
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
    days = select_nature_days(
        times,
        start,
        end,
        args.days,
        month=args.month,
        months_csv=args.months,
    )
    nature_dates = [
        str(times[int(index)].astype("datetime64[D]")) for index in days
    ]
    print(
        "[osse] temporal alignment: same checkpoint day for CPC/ERA5 "
        "conditioning and CHIRPS nature truth; no date offset",
        flush=True,
    )
    print(f"[osse] selected CHIRPS nature days: {', '.join(nature_dates)}", flush=True)

    # -- station networks -----------------------------------------------------
    rng = np.random.default_rng(args.seed)
    networks: dict[str, StationSet] = {}
    bmd_station_source: str | None = None
    for token in args.networks.split(","):
        token = token.strip()
        if not token:
            continue
        if token.lower() == "bmd":
            station_path = Path(args.bmd_stations)
            if not station_path.is_file():
                fallback = Path("data/stations/Stations.csv")
                if fallback.is_file():
                    station_path = fallback
                else:
                    print(
                        f"[osse] skipping 'bmd': station catalogue not found: "
                        f"{args.bmd_stations}"
                    )
                    continue
            catalog = read_station_catalog(station_path)
            lo, la, hi, ha = grid.bbox
            margin = grid.res / 2
            inside = (
                (catalog["lon"] > lo + margin)
                & (catalog["lon"] < hi - margin)
                & (catalog["lat"] > la + margin)
                & (catalog["lat"] < ha - margin)
            )
            if (~inside).any():
                print(
                    f"[osse] dropping {int((~inside).sum())} BMD station(s) "
                    f"outside the {grid.name} interpolation domain"
                )
            catalog = catalog.loc[inside].reset_index(drop=True)
            bmd = StationSet(
                lat=catalog["lat"].to_numpy(float),
                lon=catalog["lon"].to_numpy(float),
                ids=catalog["station_id"].to_numpy(),
            )
            print(
                f"[osse] BMD geometry: {len(bmd)} locations from {station_path}; "
                "rainfall values will be sampled from same-day CHIRPS"
            )
            bmd_station_source = str(station_path)
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
    satellite_factor = int(
        args.satellite_factor
        if args.satellite_factor is not None
        else satellite_cfg["factor"]
    )
    if satellite_factor < 1:
        raise ValueError("--satellite-factor must be >= 1")
    satellite_crop = tuple(
        args.satellite_crop or (0, grid.nlat, 0, grid.nlon)
    )
    row_start, row_stop, col_start, col_stop = satellite_crop
    if not (
        0 <= row_start < row_stop <= grid.nlat
        and 0 <= col_start < col_stop <= grid.nlon
    ):
        raise ValueError(
            f"invalid --satellite-crop {satellite_crop} for "
            f"{grid.nlat}x{grid.nlon} grid"
        )
    satellite_height = row_stop - row_start
    satellite_width = col_stop - col_start
    if use_satellite and (
        satellite_height % satellite_factor
        or satellite_width % satellite_factor
    ):
        raise ValueError(
            f"satellite crop {satellite_height}x{satellite_width} is not "
            f"divisible by factor {satellite_factor}"
        )
    satellite_shape = (
        satellite_height // satellite_factor,
        satellite_width // satellite_factor,
    )
    satellite_count = int(np.prod(satellite_shape)) if use_satellite else 0
    satellite_selection = np.zeros(satellite_shape, dtype=bool)
    satellite_selection[::args.satellite_stride, ::args.satellite_stride] = True
    satellite_resolution_deg = float(grid.res * satellite_factor)
    if use_satellite:
        print(
            f"[osse] exact pseudo-satellite: {satellite_resolution_deg:g}° "
            f"({satellite_factor}x{satellite_factor} fine cells), "
            f"fine-grid crop={satellite_crop}, coarse shape={satellite_shape}",
            flush=True,
        )
    selected_satellite_count = int(satellite_selection.sum()) if use_satellite else 0
    satellite_r_inflation = 1.0
    if use_satellite and args.satellite_correlation_control:
        correlation_cells = float(satellite_cfg.get("error_corr_cells", 0.0))
        satellite_r_inflation = max(
            1.0,
            2.0 * np.pi * (correlation_cells / args.satellite_stride) ** 2,
        )
    error_levels = {}
    unknown_error_levels: list[str] = []
    for token in args.obs_error.split(","):
        token = token.strip().lower()
        if token == "exact":
            # Observed values are the exact CHIRPS pseudo-data. R must remain
            # positive because the diffusion-posterior likelihood becomes
            # singular at t -> 1 when R=0; 0.05 is a numerical regulariser,
            # not noise added to the observations.
            error_levels["exact"] = (0.05, 0.0)
        elif token == "realistic":
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
        elif token:
            unknown_error_levels.append(token)
    if unknown_error_levels:
        raise ValueError(
            "unknown observation-error level(s): "
            + ", ".join(sorted(set(unknown_error_levels)))
        )
    if not error_levels:
        raise ValueError("no observation-error levels selected")
    if "realistic" in error_levels:
        print(
            "[osse] WARNING: 'realistic' is the legacy uncalibrated "
            "transformed-Gaussian error sensitivity. It is not the primary "
            "CHIRPS OSSE; use --obs-error exact for exact pseudo-data.",
            flush=True,
        )

    # Overrides must be applied BEFORE the configs are constructed. They were
    # not: base_sampler was built first, so --prior-temperature mutated a dict
    # nobody read again and the temp15 arm came out bit-identical to base --
    # which reads as "temperature does nothing" rather than "the flag did
    # nothing". The guidance overrides happened to sit on the right side of
    # GuidanceConfig and did work, which made the failure harder to spot.
    if args.guidance_spread_cells is not None:
        config["guidance"]["spread_cells"] = float(args.guidance_spread_cells)
    if args.guidance_gamma is not None:
        config["guidance"]["gamma"] = float(args.guidance_gamma)
    if args.prior_temperature is not None:
        config["sampler"]["prior_temperature"] = float(args.prior_temperature)

    base_sampler = replace(
        SamplerConfig(**config["sampler"]), mask_fill=dataset.mask_fill
    )
    base_guidance = GuidanceConfig(**config["guidance"])
    print(
        f"[osse] sampler prior_temperature={base_sampler.prior_temperature:g}, "
        f"guidance gamma={base_guidance.gamma:g}, "
        f"spread_cells={base_guidance.spread_cells:g}",
        flush=True,
    )
    if base_guidance.spread_cells:
        print(
            f"[osse] guidance gradient spread over {base_guidance.spread_cells:g} "
            f"cells (~{base_guidance.spread_cells * 5.5:.0f} km). This asserts a "
            f"broader background covariance than the network implies and is an "
            f"APPROXIMATION to exact posterior guidance; report it as one.",
            flush=True,
        )

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

    # Tuning can intentionally reduce the selected day set.
    nature_dates = [
        str(times[int(index)].astype("datetime64[D]")) for index in days
    ]

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
    item_cache: dict[int, dict[str, torch.Tensor]] = {}
    truth_cache: dict[int, np.ndarray] = {}
    position_for_index = {
        int(store_index): position
        for position, store_index in enumerate(dataset.index)
    }

    # Materialise conditioning and truth together. This makes a temporal shift
    # structurally impossible: every pseudo-observation below is derived from
    # target_mm in the exact same dataset item handed to the checkpoint.
    for index_value in days:
        index = int(index_value)
        if index not in position_for_index:
            raise RuntimeError(
                f"selected store index {index} is absent from dataset.index"
            )
        item = dataset[position_for_index[index]]
        item_date = np.datetime64(int(item["time"].item()), "s").astype(
            "datetime64[D]"
        )
        expected_date = times[index].astype("datetime64[D]")
        if item_date != expected_date:
            raise RuntimeError(
                "OSSE temporal-alignment failure: dataset item is "
                f"{item_date}, but selected CHIRPS day is {expected_date}"
            )
        day_valid = item["mask"][0].cpu().numpy() > 0
        truth = item["target_mm"][0].cpu().numpy().astype(np.float32)
        item_cache[index] = item
        truth_cache[index] = np.where(day_valid, truth, np.nan)

    def background_for(index: int, temperature: float) -> np.ndarray:
        key = (int(index), round(float(temperature), 6))
        if key in background_cache:
            return background_cache[key]
        item = item_cache[int(index)]
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

    dump_target = args.dump_network
    if dump_target is not None and dump_target.strip().lower() == "bmd":
        bmd_labels = [name for name in networks if name.startswith("bmd (")]
        if len(bmd_labels) != 1:
            raise ValueError(
                "--dump-network bmd requires exactly one BMD network; found "
                f"{bmd_labels}"
            )
        dump_target = bmd_labels[0]
    if args.dump and dump_target is None:
        synthetic = [n for n in networks if n.replace(" ", "").isdigit()]
        dump_target = max(synthetic, key=int) if synthetic else list(networks)[0]
    dumped = None

    results = []
    for name, stations in networks.items():
        is_bmd_network = name.startswith("bmd (")
        n_stations = len(stations)
        if n_stations < 2:
            raise ValueError(f"network {name!r} needs at least two stations")
        n_withhold = min(
            n_stations - 1, max(1, int(round(args.withhold * n_stations)))
        )
        split_rng = np.random.default_rng(args.seed + n_stations)
        if args.holdout_layout == "spread":
            eval_idx = np.sort(
                spread_holdout(stations.lat, stations.lon, n_withhold)
            )
        elif args.holdout_layout == "neighbored":
            eval_idx = np.sort(
                neighbored_holdout(
                    stations.lat, stations.lon, n_withhold,
                    radius_km=args.holdout_neighbor_km,
                    max_gap_deg=args.holdout_max_gap_deg,
                )
            )
        else:
            order = split_rng.permutation(n_stations)
            eval_idx = np.sort(order[:n_withhold])
        station_nn_km = nearest_neighbour_km(stations.lat, stations.lon)
        station_gap_deg = max_bearing_gap_deg(stations.lat, stations.lon)
        print(
            f"[osse] holdout '{args.holdout_layout}': {n_withhold} withheld of "
            f"{n_stations}; nearest-neighbour distance of withheld stations "
            f"min/median/max = {station_nn_km[eval_idx].min():.1f}/"
            f"{np.median(station_nn_km[eval_idx]):.1f}/"
            f"{station_nn_km[eval_idx].max():.1f} km "
            f"(all stations median {np.median(station_nn_km):.1f} km); "
            f"bearing gap median {np.median(station_gap_deg[eval_idx]):.0f} deg, "
            f"max {station_gap_deg[eval_idx].max():.0f} deg",
            flush=True,
        )
        eval_set = set(eval_idx.tolist())
        assim_idx = np.asarray(
            [index for index in range(n_stations) if index not in eval_set],
            dtype=int,
        )
        active_assim_idx = assim_idx if use_gauges else np.array([], dtype=int)

        for error_name, (sigma, representativeness) in error_levels.items():
            for setting in combinations:
                exact_observations = error_name == "exact"
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
                            satellite_factor,
                            transform,
                            valid=valid,
                            crop=satellite_crop,
                        )
                    )
                    if error_name in {"perfect", "exact"}:
                        satellite_sigma = (
                            0.05 if args.satellite_sigma is None
                            else float(args.satellite_sigma)
                        )
                        satellite_repr = 0.0
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
                exact_satellite_max_abs_error_mm = 0.0
                exact_gauge_max_abs_error_transformed = 0.0
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
                    item = item_cache[index]
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
                    # TRANSFORMED space only for the legacy noisy sensitivity.
                    # The primary exact OSSE passes CHIRPS values unchanged.
                    truth_at_stations = sample_at_stations(truth, grid, stations)[0]
                    observation_rng = np.random.default_rng(args.seed + index)
                    gauge_truth_transformed = transform.forward(
                        np.clip(truth_at_stations, 0.0, None)
                    )
                    if exact_observations:
                        gauge_obs_transformed = gauge_truth_transformed.copy()
                        exact_gauge_max_abs_error_transformed = max(
                            exact_gauge_max_abs_error_transformed,
                            float(
                                np.nanmax(
                                    np.abs(
                                        gauge_obs_transformed
                                        - gauge_truth_transformed
                                    )
                                )
                            ),
                        )
                    else:
                        gauge_obs_transformed = (
                            gauge_truth_transformed
                            + observation_rng.normal(
                                0.0, gauge_noise_sd, n_stations
                            )
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
                            truth, valid, satellite_factor, crop=satellite_crop
                        )
                        satellite_truth_transformed = transform.forward(
                            satellite_truth_mm
                        ).astype(np.float32).reshape(-1)
                        satellite_truth_transformed[
                            ~satellite_selection.reshape(-1)
                        ] = np.nan
                        # The OBSERVATION may differ from the truth: with
                        # --satellite-source cpc the analysis is forced toward
                        # the coarse conditioning field, while scoring still
                        # uses the CHIRPS block means. Keeping the two arrays
                        # separate is the whole point -- collapsing them would
                        # silently make CPC the truth.
                        if args.satellite_source == "cpc":
                            satellite_obs_mm, _ = block_mean_mm(
                                coarse_base_mm, valid, satellite_factor,
                                crop=satellite_crop,
                            )
                            satellite_source_transformed = transform.forward(
                                satellite_obs_mm
                            ).astype(np.float32).reshape(-1)
                            satellite_source_transformed[
                                ~satellite_selection.reshape(-1)
                            ] = np.nan
                        else:
                            satellite_source_transformed = satellite_truth_transformed
                        # One correlated error realisation creates the actual
                        # pseudo-IMERG product.  Member-wise perturbations below
                        # then preserve posterior ensemble variance.
                        satellite_R = R[len(active_assim_idx):]
                        if exact_observations:
                            satellite_observed = satellite_source_transformed.copy()
                        else:
                            satellite_observed = perturb_observations(
                                satellite_source_transformed,
                                satellite_R,
                                1,
                                seed=args.seed + index + 100_000,
                                corr_blocks=[
                                    (
                                        0,
                                        satellite_shape[0],
                                        satellite_shape[1],
                                        float(
                                            satellite_cfg.get(
                                                "error_corr_cells", 0.0
                                            )
                                        ),
                                    )
                                ],
                            )[0].astype(np.float32)
                        satellite_observed[
                            ~np.isfinite(satellite_source_transformed)
                        ] = np.nan
                        truth_obs.append(satellite_truth_transformed)
                        observed.append(satellite_observed)
                        satellite_observed_mm = transform.inverse(
                            satellite_observed
                        ).reshape(satellite_shape).astype(np.float32)
                        if exact_observations and args.satellite_source == "truth":
                            exact_footprints = (
                                satellite_selection
                                & np.isfinite(satellite_truth_mm)
                            )
                            if not np.allclose(
                                satellite_observed_mm[exact_footprints],
                                satellite_truth_mm[exact_footprints],
                                rtol=1e-5,
                                atol=1e-5,
                            ):
                                raise RuntimeError(
                                    "exact OSSE invariant failed: "
                                    "pseudo-satellite differs from the same-day "
                                    f"CHIRPS {satellite_factor}x{satellite_factor} mean"
                                )
                            exact_satellite_max_abs_error_mm = max(
                                exact_satellite_max_abs_error_mm,
                                float(
                                    np.nanmax(
                                        np.abs(
                                            satellite_observed_mm[exact_footprints]
                                            - satellite_truth_mm[exact_footprints]
                                        )
                                    )
                                ),
                            )

                    y_truth = np.concatenate(truth_obs).astype(np.float32)
                    y_assim = np.concatenate(observed).astype(np.float32)
                    y_assim[~np.isfinite(y_truth)] = np.nan

                    if exact_observations:
                        # R remains a small positive regulariser in the
                        # likelihood, but no random error is added either to the
                        # shared pseudo-observation or member-specific copies.
                        perturbed = np.repeat(
                            y_assim[None, :], args.members, axis=0
                        )
                    else:
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
                    "pseudo_observation_values": (
                        "exact CHIRPS" if exact_observations else "synthetically noisy"
                    ),
                    "observation_noise_added": not exact_observations,
                    "likelihood_sd_transformed": gauge_noise_sd,
                    "exact_gauge_max_abs_error_transformed": (
                        exact_gauge_max_abs_error_transformed
                        if exact_observations
                        else None
                    ),
                    "exact_satellite_max_abs_error_mm": (
                        exact_satellite_max_abs_error_mm
                        if exact_observations and use_satellite
                        else None
                    ),
                    "station_geometry": "BMD catalogue" if is_bmd_network else name,
                    "observation_mode": observation_mode,
                    "obs_noise_sd_transformed": gauge_noise_sd,
                    "pseudo_satellite": bool(use_satellite),
                    "satellite_noise_sd_transformed": satellite_noise_sd,
                    "satellite_stride": int(args.satellite_stride),
                    "satellite_factor": int(satellite_factor),
                    "satellite_resolution_deg": satellite_resolution_deg,
                    "satellite_crop": list(satellite_crop),
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
                        observation_noise_added=np.bool_(not exact_observations),
                        likelihood_sd_transformed=np.float32(gauge_noise_sd),
                        exact_gauge_max_abs_error_transformed=np.float32(
                            exact_gauge_max_abs_error_transformed
                            if exact_observations
                            else np.nan
                        ),
                        exact_satellite_max_abs_error_mm=np.float32(
                            exact_satellite_max_abs_error_mm
                            if exact_observations and use_satellite
                            else np.nan
                        ),
                        satellite_obs_noise_sd=np.float32(satellite_noise_sd),
                        pseudo_satellite_enabled=np.bool_(use_satellite),
                        satellite_factor=np.int32(satellite_factor),
                        satellite_resolution_deg=np.float32(
                            satellite_resolution_deg
                        ),
                        satellite_crop=np.asarray(
                            satellite_crop, dtype=np.int32
                        ),
                        satellite_stride=np.int32(args.satellite_stride),
                        satellite_selection=satellite_selection,
                        satellite_r_inflation=np.float32(satellite_r_inflation),
                        observation_mode=np.str_(observation_mode),
                        pseudo_observation_values=np.str_(
                            "exact CHIRPS"
                            if exact_observations
                            else "synthetically noisy"
                        ),
                        station_geometry=np.str_(
                            "BMD catalogue" if is_bmd_network else name
                        ),
                        temporal_alignment=np.str_(
                            "same checkpoint day; no offset"
                        ),
                        nature_source=np.str_("CHIRPS 0.05-degree target"),
                        pseudo_gauge_source=np.str_(
                            "same-day CHIRPS sampled at BMD catalogue locations"
                            if is_bmd_network
                            else "same-day CHIRPS sampled at synthetic stations"
                        ),
                        pseudo_satellite_source=np.str_(
                            "same-day CHIRPS exact physical "
                            f"{satellite_factor}x{satellite_factor} block means"
                        ),
                        day_selection=np.str_(
                            "balanced by available year-month"
                            if args.months is not None
                            else "evenly spaced over eligible dates"
                        ),
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
                if exact_observations:
                    satellite_qc = (
                        f"; satellite max|error| "
                        f"{exact_satellite_max_abs_error_mm:.3g} mm/day"
                        if use_satellite
                        else ""
                    )
                    print(
                        "[osse] exact observation QC: gauge max|error| "
                        f"{exact_gauge_max_abs_error_transformed:.3g} transformed"
                        f"{satellite_qc}",
                        flush=True,
                    )
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
        "days": nature_dates,
        "members": args.members,
        "requested_months": (
            [int(value) for value in args.months.split(",")]
            if args.months is not None
            else ([args.month] if args.month else [])
        ),
        "day_selection": (
            "balanced by available year-month"
            if args.months is not None
            else "evenly spaced over eligible dates"
        ),
        "temporal_alignment": "same checkpoint day; no offset",
        "nature_source": "CHIRPS 0.05-degree checkpoint target",
        "pseudo_gauge_source": (
            "same-day CHIRPS sampled at BMD catalogue locations"
            if all(name.startswith("bmd (") for name in networks)
            else "same-day CHIRPS sampled at configured station networks"
        ),
        "pseudo_satellite_source": (
            "same-day CHIRPS exact physical "
            f"{satellite_factor}x{satellite_factor} block means"
            if use_satellite
            else None
        ),
        "withhold_fraction": args.withhold,
        "data_zarr": data_zarr,
        "data_stats": data_stats,
        "observation_mode": observation_mode,
        "pseudo_satellite": bool(use_satellite),
        "satellite_factor": satellite_factor if use_satellite else None,
        "satellite_resolution_deg": satellite_resolution_deg
        if use_satellite
        else None,
        "satellite_crop": list(satellite_crop) if use_satellite else None,
        "satellite_stride": args.satellite_stride if use_satellite else None,
        "satellite_selected_footprints": selected_satellite_count
        if use_satellite
        else None,
        "satellite_r_inflation": satellite_r_inflation if use_satellite else None,
        "station_layout": args.station_layout,
        "holdout_layout": args.holdout_layout,
        "holdout_neighbor_km": float(args.holdout_neighbor_km),
        "holdout_max_gap_deg": float(args.holdout_max_gap_deg),
        "guidance_spread_cells": float(base_guidance.spread_cells),
        "satellite_source": args.satellite_source,
        "bmd_station_catalog": bmd_station_source,
        "mode": "tuning" if args.tune else "network sweep",
        "note": (
            "This is an optimistic upper-bound OSSE because CHIRPS supplies both "
            "the nature truth and pseudo-observations. Withheld gauges are not "
            "fully independent when dense pseudo-satellite footprints are also "
            "assimilated; they test sub-footprint allocation at unseen point "
            "locations. Read the footprint and within-footprint subgrid "
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
