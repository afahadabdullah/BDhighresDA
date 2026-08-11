#!/usr/bin/env python
"""Short-window sweep over ways of assimilating BMD gauges and optional IMERG.

Purpose
-------
The 2021-2024 pooled evaluation showed simultaneous DA beating gauges-only by
2.6% CRPS while making bias 2.4x worse and MAE 7% worse, and showed IMERG-only
removing just 4% of the background's +10.3 mm/day wet bias.  That is not a
satellite that is being ingested well.  This script runs the candidate fixes
against each other on a handful of days so the expensive multi-year run is only
ever spent on a method that has already earned it.

It is deliberately NOT a skill evaluation.  Five days cannot resolve a 2% CRPS
difference -- ``scripts/29_summarize_method_sweep.py`` prints the paired
bootstrap interval to make that concrete.  What five days CAN resolve is a
10 mm/day bias, a factor-of-two change in increment locality, and an arm that
diverges.  Read the bias, the wet-day frequency and the bullseye diagnostic;
treat CRPS differences smaller than the printed interval as unresolved.

Every arm shares the checkpoint, the conditioning, the spatial holdout fold, the
per-day prior noise seed and the observation perturbation seeds, so any
difference between two arms is the method and nothing else.

What each variant tests
-----------------------
``background``           unguided prior, production settings (T = 1.0)
``background_T125``      unguided prior at the analysis temperature.  Isolates
                         how much of the analysis bias is Jensen inflation from
                         tempering in transformed space rather than DA.
``gauges_only``          production gauge-only arm; the bar every fusion arm
                         must clear.
``imerg_only*``          satellite-only, with and without bias correction and
                         with matched temperature.
``simul_joint``          production simultaneous arm: one composite likelihood,
                         both streams guided together.  The current baseline.
``simul_joint_bc``       same, with leave-one-year-out quantile-mapped IMERG.
                         METHODOLOGY.md 4.4 says a Gaussian likelihood assumes an
                         unbiased observation; this is the arm that honours that.
``simul_joint_huber``    robust cost, so a handful of badly wrong footprints
                         cannot drag the field.
``simul_joint_r*``       explicit satellite weight.  The production R already
                         carries a 6.28x correlation inflation on top of stride-3
                         thinning; this asks whether IMERG is simply muted.
``simul_joint_dense``    stride 1.  Keeps every footprint and lets the automatic
                         correlation inflation compensate -- a direct test of
                         whether that approximation is doing the right thing.
``simul_perstream``      separate likelihood weight per stream, so the satellite
                         can set pattern while gauges keep amplitude authority.
``simul_twostep_ensrf``  IMERG guides the generative sampler, then the gauges
                         update THAT posterior through a localized serial EnSRF
                         (src/bdhires/ensrf.py).  Gaspari-Cohn tapering should
                         spread each gauge increment over a meteorological scale
                         instead of the discs that joint guidance leaves behind.

The ``v2_gauges_*`` groups are a separate, gauges-only tournament for the CPC-v2
checkpoint.  They compare likelihood guidance (spread/no-spread and matched/
tempered prior) with a localized EnSRF applied to the *same saved background
ensemble*.  No satellite file is needed for those groups.

Usage
-----
    python scripts/28_simultaneous_method_sweep.py \
        --config configs/da.yaml \
        --ckpt runs/prior_h100_cpc/best.pt \
        --stations data/processed/sweep_may2024_bmd.csv \
        --imerg data/processed/sweep_may2024_imerg.nc \
        --start 2024-05-01 --end 2024-05-05 \
        --imerg-qm data/processed/imerg_qm_loyo.npz \
        --group core \
        --out data/processed/sweep_may2024.npz \
        --report data/processed/sweep_may2024.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time as walltime
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
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
from bdhires.ensrf import localized_serial_ensrf  # noqa: E402
from bdhires.eval import crps_ensemble  # noqa: E402
from bdhires.grids import WIDE, crop_offsets, get_grid  # noqa: E402
from bdhires.models import RectifiedFlow, UNet, select_weights  # noqa: E402
from bdhires.transforms import (  # noqa: E402
    CondTransform,
    PrecipTransform,
    ResidualSpec,
    load_climatology,
)

# Reuse the IMERG loader and the variance conversion from the production script
# rather than reimplementing them, so the two paths cannot silently diverge.
_MODULE_DIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "_bmd_month_example", _MODULE_DIR / "15_bmd_month_example.py"
)
_bmd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bmd)
load_prepared_imerg = _bmd.load_prepared_imerg
transformed_imerg_variance = _bmd.transformed_imerg_variance
sample_at_stations = _bmd.sample_at_stations

_qm_spec = importlib.util.spec_from_file_location(
    "_imerg_qm", _MODULE_DIR / "27_fit_imerg_bias_correction.py"
)
_qm = importlib.util.module_from_spec(_qm_spec)
_qm_spec.loader.exec_module(_qm)
load_and_apply_qm = _qm.load_and_apply


# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Variant:
    """One assimilation method. Every field that is not a sampler knob is a
    statement about how the observations enter the analysis."""

    name: str
    streams: str = "both"          # none | gauges | imerg | both
    algorithm: str = "joint"       # joint | ensrf | twostep_ensrf
    bias_correct: bool = False
    imerg_stride: int | None = None        # None -> command-line default
    imerg_r_multiplier: float = 1.0
    gauge_weight: float = 1.0              # likelihood weight, 1 = production
    imerg_weight: float = 1.0
    huber_delta: float | None = None
    prior_temperature: float | None = None  # None -> config value
    n_corrections: int | None = None
    guidance_spread_cells: float | None = None
    ensrf_localization_km: float = 200.0
    note: str = ""

    @property
    def uses_imerg(self) -> bool:
        return self.streams in {"imerg", "both"}

    @property
    def uses_gauges(self) -> bool:
        return self.streams in {"gauges", "both"}


CORE = [
    Variant("background", streams="none", prior_temperature=1.0,
            note="production unguided prior"),
    Variant("gauges_only", streams="gauges",
            note="the bar every fusion arm must clear"),
    Variant("imerg_only", streams="imerg",
            note="production satellite-only baseline"),
    Variant("simul_joint", streams="both",
            note="production simultaneous baseline"),
    Variant("simul_joint_bc", streams="both", bias_correct=True,
            note="quantile-mapped IMERG, everything else identical"),
    Variant("simul_joint_bc_perstream", streams="both", bias_correct=True,
            imerg_weight=0.25,
            note="satellite sets pattern, gauges keep amplitude authority"),
    Variant("simul_twostep_ensrf", streams="both", algorithm="twostep_ensrf",
            note="IMERG guides the sampler, gauges update it via localized EnSRF"),
    Variant("simul_twostep_ensrf_bc", streams="both", algorithm="twostep_ensrf",
            bias_correct=True,
            note="the two fixes combined; the leading candidate"),
]

TEMPERING = [
    Variant("background_T125", streams="none", prior_temperature=1.25,
            note="isolates Jensen inflation from DA"),
    Variant("gauges_only_T100", streams="gauges", prior_temperature=1.0),
    Variant("imerg_only_T100", streams="imerg", prior_temperature=1.0,
            note="how much of the +9.88 mm bias is tempering, not the satellite"),
    Variant("simul_joint_T100", streams="both", prior_temperature=1.0),
]

BIAS = [
    Variant("imerg_only_bc", streams="imerg", bias_correct=True,
            note="does correcting IMERG alone fix the bias"),
    Variant("simul_joint_bc", streams="both", bias_correct=True),
    Variant("simul_joint_bc_T100", streams="both", bias_correct=True,
            prior_temperature=1.0,
            note="bias correction with the tempering confound removed"),
]

WEIGHTING = [
    Variant("simul_joint_bc_r0p25", streams="both", bias_correct=True,
            imerg_r_multiplier=0.25, note="trust the satellite 4x more"),
    Variant("simul_joint_bc_r4", streams="both", bias_correct=True,
            imerg_r_multiplier=4.0, note="trust the satellite 4x less"),
    Variant("simul_joint_bc_dense", streams="both", bias_correct=True,
            imerg_stride=1,
            note="every footprint; tests the correlation-inflation approximation"),
    Variant("simul_joint_bc_huber", streams="both", bias_correct=True,
            huber_delta=3.0, note="robust cost against outlier footprints"),
    Variant("simul_joint_bc_perstream_tight", streams="both", bias_correct=True,
            imerg_weight=0.1),
]

TWOSTEP = [
    Variant("simul_twostep_ensrf_bc", streams="both", algorithm="twostep_ensrf",
            bias_correct=True),
    Variant("simul_twostep_ensrf_bc_loc100", streams="both",
            algorithm="twostep_ensrf", bias_correct=True,
            ensrf_localization_km=100.0),
    Variant("simul_twostep_ensrf_bc_loc400", streams="both",
            algorithm="twostep_ensrf", bias_correct=True,
            ensrf_localization_km=400.0),
]

# Gauges-only CPC-v2 tournament.  ``core`` is a compact method comparison, not
# a hyperparameter fishing expedition: the four guided arms form a 2x2
# factorial (temperature 1.0/1.25 x spread 0/6), and EnSRF is the genuinely
# different covariance-based update.  The follow-up groups should only be run
# after core identifies which mechanism is worth resolving more finely.
V2_GAUGES_CORE = [
    CORE[0],
    Variant("guided_s0_t125", streams="gauges", prior_temperature=1.25,
            guidance_spread_cells=0.0,
            note="current v2 control: tempered prior and point guidance"),
    Variant("guided_s6_t125", streams="gauges", prior_temperature=1.25,
            guidance_spread_cells=6.0,
            note="measured v2 spread-6 candidate (~33 km Gaussian sigma)"),
    Variant("guided_s0_t100", streams="gauges", prior_temperature=1.0,
            guidance_spread_cells=0.0,
            note="isolates analysis tempering from spatial spreading"),
    Variant("guided_s6_t100", streams="gauges", prior_temperature=1.0,
            guidance_spread_cells=6.0,
            note="spread-6 with the analysis temperature matched to background"),
    Variant("ensrf_loc150", streams="gauges", algorithm="ensrf",
            ensrf_localization_km=150.0,
            note="localized EnSRF on the exact v2 background; support ~variogram range"),
]

V2_GAUGES_SPREAD = [
    CORE[0],
    Variant("guided_s0_t125", streams="gauges", prior_temperature=1.25,
            guidance_spread_cells=0.0,
            note="current v2 operational comparator"),
    Variant("guided_s0_t100", streams="gauges", prior_temperature=1.0,
            guidance_spread_cells=0.0),
    Variant("guided_s3_t100", streams="gauges", prior_temperature=1.0,
            guidance_spread_cells=3.0,
            note="intermediate spread (~17 km Gaussian sigma)"),
    Variant("guided_s6_t100", streams="gauges", prior_temperature=1.0,
            guidance_spread_cells=6.0),
    Variant("guided_s12_t100", streams="gauges", prior_temperature=1.0,
            guidance_spread_cells=12.0,
            note="broad-spread sensitivity, not a production default"),
]

V2_GAUGES_ENSRF = [
    CORE[0],
    Variant("guided_s0_t125", streams="gauges", prior_temperature=1.25,
            guidance_spread_cells=0.0,
            note="current v2 operational comparator"),
    Variant("ensrf_loc75", streams="gauges", algorithm="ensrf",
            ensrf_localization_km=75.0),
    Variant("ensrf_loc150", streams="gauges", algorithm="ensrf",
            ensrf_localization_km=150.0),
    Variant("ensrf_loc300", streams="gauges", algorithm="ensrf",
            ensrf_localization_km=300.0),
]


def _unique_variants(variants: list[Variant]) -> list[Variant]:
    """De-duplicate catalogue unions by name while preserving first occurrence."""
    seen: set[str] = set()
    unique = []
    for variant in variants:
        if variant.name in seen:
            continue
        seen.add(variant.name)
        unique.append(variant)
    return unique

GROUPS = {
    "core": CORE,
    "tempering": TEMPERING,
    "bias": BIAS,
    "weighting": WEIGHTING,
    "twostep": TWOSTEP,
    "v2_gauges_core": V2_GAUGES_CORE,
    "v2_gauges_spread": V2_GAUGES_SPREAD,
    "v2_gauges_ensrf": V2_GAUGES_ENSRF,
    "all": _unique_variants(
        CORE + TEMPERING + BIAS + WEIGHTING + TWOSTEP
        + V2_GAUGES_CORE + V2_GAUGES_SPREAD + V2_GAUGES_ENSRF
    ),
}


def resolve_variants(group: str, names: list[str] | None) -> list[Variant]:
    """Group selection plus explicit names, de-duplicated, order preserved."""
    chosen: list[Variant] = list(GROUPS[group])
    if names:
        catalogue = {variant.name: variant for variant in GROUPS["all"]}
        unknown = [name for name in names if name not in catalogue]
        if unknown:
            raise ValueError(
                f"unknown variant(s) {unknown}; available: {sorted(catalogue)}"
            )
        chosen.extend(catalogue[name] for name in names)
    seen, unique = set(), []
    for variant in chosen:
        if variant.name in seen:
            continue
        seen.add(variant.name)
        unique.append(variant)
    # The increment-locality diagnostic differences every arm against the
    # production background, so that arm is always present regardless of group.
    if "background" not in seen:
        unique.insert(0, CORE[0])
    return unique


# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/da.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--stations", required=True, help="canonical BMD daily CSV")
    parser.add_argument(
        "--imerg", default=None,
        help="prepared IMERG NetCDF from script 08; omitted for v2_gauges_* groups",
    )
    parser.add_argument(
        "--set", action="append", default=[], metavar="KEY.PATH=VALUE",
        help="repeatable config override; unknown keys fail loudly",
    )
    parser.add_argument("--start", default="2024-05-01")
    parser.add_argument("--end", default="2024-05-05")
    parser.add_argument("--background-day-offset", type=int, default=-1)
    parser.add_argument("--members", type=int, default=16)
    parser.add_argument("--min-coverage", type=float, default=0.5)
    parser.add_argument("--withhold", type=float, default=0.2)
    parser.add_argument("--holdout-folds", type=int, default=5)
    parser.add_argument("--holdout-fold", type=int, default=0)
    parser.add_argument("--imerg-stride", type=int, default=3)
    parser.add_argument("--imerg-r-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--imerg-qm",
        default=None,
        help="leave-one-year-out map from script 27; required by bias_correct variants",
    )
    parser.add_argument("--group", default="core", choices=sorted(GROUPS))
    parser.add_argument("--variants", nargs="*", default=None, help="extra variants by name")
    parser.add_argument("--n-steps", type=int, default=None, help="override sampler steps")
    parser.add_argument("--seed", type=int, default=202405)
    parser.add_argument("--out", default="data/processed/sweep_may2024.npz")
    parser.add_argument("--report", default="data/processed/sweep_may2024.json")
    parser.add_argument(
        "--list-variants", action="store_true", help="print the catalogue and exit"
    )
    return parser.parse_args()


def fitted_sigma_floor(
    intensity_mm: np.ndarray,
    error_bins_mm: list[float] | None,
    error_sigma_transformed: list[float] | None,
    fallback: float,
) -> np.ndarray | float:
    """Map corrected IMERG intensity to script 27's fitted transformed-space sigma.

    The config's ``sigma_obs: 0.35`` is a guess. Script 27's ``--fit-error-model``
    measures it instead -- typically ~1.1-1.24 in transformed space here, three
    times larger -- and METHODOLOGY.md 4.3 says to use the measurement, not the
    guess. Bins with too few holdout samples are NaN in the fit; those, and any
    intensity outside the fitted range's extremes, fall back to ``fallback``
    rather than silently trusting an unmeasured value.
    """
    if not error_bins_mm or not error_sigma_transformed:
        return fallback
    edges = np.asarray(error_bins_mm, dtype=float)
    sigma_by_bin = np.asarray(error_sigma_transformed, dtype=float)
    index = np.clip(np.digitize(np.nan_to_num(intensity_mm, nan=0.0), edges[1:-1]), 0, len(sigma_by_bin) - 1)
    sigma = sigma_by_bin[index].astype(np.float32)
    invalid = ~np.isfinite(sigma)
    sigma[invalid] = fallback
    return sigma


def great_circle_km(lat0: np.ndarray, lon0: np.ndarray, lat1: float, lon1: float) -> np.ndarray:
    """Local-tangent distance, matching the approximation used in ensrf.py."""
    return 111.0 * np.sqrt(
        (lat0 - lat1) ** 2 + (np.cos(np.deg2rad(lat1)) * (lon0 - lon1)) ** 2
    )


def distance_to_nearest_station(grid, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """(nlat, nlon) distance in km to the nearest assimilated gauge."""
    grid_lon, grid_lat = np.meshgrid(grid.lon, grid.lat)
    stacked = np.stack(
        [great_circle_km(grid_lat, grid_lon, float(a), float(b)) for a, b in zip(lat, lon)]
    )
    return stacked.min(axis=0).astype(np.float32)


def jensen_estimators(members_mm: np.ndarray, transform: PrecipTransform) -> dict:
    """Three point estimators from the same ensemble.

    ``mean`` is what the production evaluation reports.  ``median`` and the
    transform-space mean are insensitive to the Jensen inflation that prior
    tempering introduces when the ensemble is broadened in log/sqrt space and
    then inverted, so a large gap between them is itself the diagnosis.
    """
    return {
        "mean": np.nanmean(members_mm, axis=0),
        "median": np.nanmedian(members_mm, axis=0),
        "transform_mean": transform.inverse(
            np.nanmean(transform.forward(np.nan_to_num(members_mm, nan=0.0)), axis=0)
        ),
    }


def score_stations(members: np.ndarray, observed: np.ndarray, transform) -> dict:
    """Score a (T, M, S) ensemble against (T, S) observations."""
    observed = np.asarray(observed, dtype=float)
    finite = np.isfinite(observed) & np.all(np.isfinite(members), axis=1)
    if not finite.any():
        return {"n": 0}
    flat_members = np.moveaxis(members, 1, 0)[:, finite]   # (M, N)
    truth = observed[finite]
    estimators = jensen_estimators(flat_members, transform)
    out = {
        "n": int(finite.sum()),
        "crps_mm": float(crps_ensemble(flat_members, truth)),
        "spread_mm": float(np.sqrt(np.mean(flat_members.var(axis=0, ddof=1)))),
    }
    low, high = np.quantile(flat_members, [0.05, 0.95], axis=0)
    out["coverage_90"] = float(np.mean((truth >= low) & (truth <= high)))
    for label, estimate in estimators.items():
        difference = estimate - truth
        rmse = float(np.sqrt(np.mean(difference**2)))
        out[f"{label}_rmse_mm"] = rmse
        out[f"{label}_mae_mm"] = float(np.mean(np.abs(difference)))
        out[f"{label}_bias_mm"] = float(np.mean(difference))
        out[f"{label}_correlation"] = (
            float(np.corrcoef(estimate, truth)[0, 1])
            if estimate.std() > 0 and truth.std() > 0
            else float("nan")
        )
    out["spread_skill"] = (
        out["spread_mm"] / out["mean_rmse_mm"] if out["mean_rmse_mm"] else float("nan")
    )
    out["jensen_gap_mm"] = out["mean_bias_mm"] - out["median_bias_mm"]
    return out


def brier(members: np.ndarray, observed: np.ndarray, threshold: float) -> float:
    observed = np.asarray(observed, dtype=float)
    finite = np.isfinite(observed) & np.all(np.isfinite(members), axis=1)
    if not finite.any():
        return float("nan")
    probability = np.mean(np.moveaxis(members, 1, 0)[:, finite] >= threshold, axis=0)
    return float(np.mean((probability - (observed[finite] >= threshold)) ** 2))


def increment_locality(
    analysis_mean: np.ndarray,
    background_mean: np.ndarray,
    distance_km: np.ndarray,
    valid: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Time-mean |analysis - background| binned by distance to the nearest gauge.

    This is the quantitative form of the station bullseyes visible in the
    intercomparison figure.  A method that spreads gauge information along
    meteorological structure gives a nearly flat curve; a method that grows discs
    around gauges gives one that falls steeply with distance.  The reported
    ``locality_ratio`` is the innermost bin over the outermost.
    """
    increment = np.abs(np.nanmean(analysis_mean - background_mean, axis=0))
    values = np.full(len(edges) - 1, np.nan)
    counts = np.zeros(len(edges) - 1, dtype=int)
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        inside = valid & (distance_km >= lower) & (distance_km < upper)
        counts[index] = int(inside.sum())
        if inside.any():
            values[index] = float(np.nanmean(increment[inside]))
    return values, counts


def main() -> None:
    args = parse_args()
    if args.list_variants:
        for group, variants in GROUPS.items():
            print(f"\n{group}:")
            for variant in variants:
                print(f"  {variant.name:34s} {variant.note}")
        return

    variants = resolve_variants(args.group, args.variants)
    uses_imerg = any(variant.uses_imerg for variant in variants)
    if uses_imerg and not args.imerg:
        raise ValueError(
            f"group {args.group!r} contains satellite arms; provide --imerg or "
            "select one of the v2_gauges_* groups"
        )
    if any(v.bias_correct for v in variants) and not args.imerg_qm:
        raise ValueError(
            "the selected variants include bias-corrected arms but --imerg-qm was "
            "not given. Fit one with scripts/27_fit_imerg_bias_correction.py first."
        )
    config = yaml.safe_load(Path(args.config).read_text())
    config_overrides = _bmd.apply_config_overrides(config, args.set)
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    training_config = checkpoint["cfg"]
    training_data = training_config["data"]

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
    time_to_index = {np.datetime64(v, "D"): i for i, v in enumerate(times)}
    missing = [
        str(np.datetime64(v, "D"))
        for v in background_times
        if np.datetime64(v, "D") not in time_to_index
    ]
    if missing:
        raise ValueError(f"checkpoint lacks background dates {missing}")
    selected = np.array(
        [time_to_index[np.datetime64(v, "D")] for v in background_times], dtype=np.int64
    )
    n_days = len(selected)

    stations, gauge_mm = load_stations(
        args.stations, selected_times, grid=grid, min_coverage=args.min_coverage
    )
    if len(stations) < 5:
        raise ValueError(f"only {len(stations)} stations survive coverage filtering")
    if args.holdout_folds > 1:
        eval_idx = spread_folds(stations.lat, stations.lon, args.holdout_folds)[
            args.holdout_fold
        ]
    else:
        n_withheld = max(1, min(len(stations) - 1, int(round(args.withhold * len(stations)))))
        eval_idx = spread_holdout(stations.lat, stations.lon, n_withheld)
    assim_idx = np.setdiff1d(np.arange(len(stations)), eval_idx)

    print(
        f"[sweep] {selected_times[0].astype('datetime64[D]')} to "
        f"{selected_times[-1].astype('datetime64[D]')} ({n_days} days), "
        f"{len(assim_idx)} assimilated / {len(eval_idx)} withheld stations, "
        f"{args.members} members",
        flush=True,
    )
    print(f"[sweep] variants: {', '.join(v.name for v in variants)}", flush=True)
    print(
        f"[sweep] checkpoint-bound data: {data_zarr} | {data_stats} | "
        f"transform={transform.kind} | channels={selected_channels or 'all'}",
        flush=True,
    )

    imerg_config = config["observations"]["imerg"]
    imerg_factor = int(imerg_config.get("factor", 2))
    imerg = None
    raw_imerg_mm = None
    if uses_imerg:
        imerg = load_prepared_imerg(args.imerg, selected_times, grid, imerg_factor)
        raw_imerg_mm = imerg["precipitation"].copy()

    corrected_imerg_mm, qm_meta = None, None
    if args.imerg_qm and uses_imerg:
        evaluation_year = int(str(selected_times[0].astype("datetime64[Y]")))
        corrected_imerg_mm, qm_meta = load_and_apply_qm(
            args.imerg_qm, evaluation_year, raw_imerg_mm, selected_times
        )
        finite = np.isfinite(raw_imerg_mm) & np.isfinite(corrected_imerg_mm)
        print(
            f"[sweep] IMERG bias correction from {args.imerg_qm} "
            f"(fit excludes {evaluation_year}): domain mean "
            f"{raw_imerg_mm[finite].mean():.2f} -> "
            f"{corrected_imerg_mm[finite].mean():.2f} mm/day, wet fraction "
            f"{(raw_imerg_mm[finite] >= 0.1).mean():.3f} -> "
            f"{(corrected_imerg_mm[finite] >= 0.1).mean():.3f}",
            flush=True,
        )
        if qm_meta.get("error_sigma_transformed"):
            fitted = [v for v in qm_meta["error_sigma_transformed"] if np.isfinite(v)]
            config_floor = float(imerg_config["sigma_obs"])
            print(
                f"[sweep] fitted IMERG error sd (transformed space, by intensity bin): "
                f"{[round(v, 2) for v in qm_meta['error_sigma_transformed']]} "
                f"vs configs/da.yaml sigma_obs={config_floor:g} "
                f"({np.mean(fitted) / config_floor:.1f}x larger on average). "
                "bias_correct=True variants use the fitted value, not the config guess.",
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

    base_sampler = replace(SamplerConfig(**config["sampler"]), mask_fill=dataset.mask_fill)
    base_background_sampler = replace(
        SamplerConfig(**config.get("background_sampler", config["sampler"])),
        mask_fill=dataset.mask_fill,
    )
    if args.n_steps:
        base_sampler = replace(base_sampler, n_steps=args.n_steps)
        base_background_sampler = replace(base_background_sampler, n_steps=args.n_steps)
    base_guidance = GuidanceConfig(**config["guidance"])

    gauge_config = config["observations"]["gauges"]
    gauge_variance = float(gauge_config["sigma_obs"]) ** 2 + float(
        gauge_config["representativeness"]
    ) ** 2
    gauge_R = build_R(
        len(assim_idx),
        float(gauge_config["sigma_obs"]),
        device=device,
        representativeness=float(gauge_config["representativeness"]),
    )
    gauge_operator = PhysicalBilinearObsOperator(
        grid, stations.lat[assim_idx], stations.lon[assim_idx], transform, valid=valid
    ).to(device)
    satellite_operator = None
    combined_operator = None
    land_footprints = None
    coarse_shape = None
    if uses_imerg:
        satellite_operator = PhysicalBlockAverageObsOperator(
            imerg_factor, transform, valid=valid
        ).to(device)
        combined_operator = CompositeObsOperator(
            [gauge_operator, satellite_operator]
        ).to(device)
        land_footprints = (
            satellite_operator.valid_mask().detach().cpu().numpy().astype(bool)
        )
        coarse_shape = raw_imerg_mm.shape[1:]
    error_corr_cells = float(imerg_config.get("error_corr_cells", 0.0))

    def satellite_setup(variant: Variant):
        """Thinning mask and correlation inflation for one variant."""
        if land_footprints is None or coarse_shape is None:
            raise ValueError(f"{variant.name}: satellite setup requested without --imerg")
        stride = variant.imerg_stride or args.imerg_stride
        thinning = np.zeros(coarse_shape, dtype=bool)
        offset = stride // 2
        thinning[offset::stride, offset::stride] = True
        keep = land_footprints & thinning.reshape(-1)
        inflation = max(1.0, 2.0 * np.pi * (error_corr_cells / stride) ** 2) * (
            args.imerg_r_multiplier * variant.imerg_r_multiplier
        )
        return stride, keep, float(inflation)

    results: dict[str, dict] = {}
    fields: dict[str, np.ndarray] = {}
    station_ensembles: dict[str, np.ndarray] = {}
    diagnostics: dict[str, dict] = {}

    shape = (n_days, args.members, grid.nlat, grid.nlon)
    for variant in variants:
        fields[variant.name] = np.full(shape, np.nan, dtype=np.float32)

    chirps = np.full((n_days, grid.nlat, grid.nlon), np.nan, dtype=np.float32)
    condition = np.full_like(chirps, np.nan)
    cpc_full_index = (
        dataset.all_cond_channels.index("cpc_precip")
        if "cpc_precip" in dataset.all_cond_channels
        else None
    )

    started = walltime.time()
    for day_position, data_index in enumerate(selected):
        dataset_position = int(np.where(dataset.index == data_index)[0][0])
        item = dataset[dataset_position]
        cond = item["cond"][None].to(device)
        base = item["base"][None].to(device)
        day_seed = args.seed + int(observation_selected[day_position])

        gauge_observation = transform.forward(
            gauge_mm[day_position, assim_idx]
        ).astype(np.float32)
        gauge_perturbed = perturb_observations(
            gauge_observation, gauge_R, args.members, seed=day_seed + 1_000_000
        ).astype(np.float32)
        gauge_perturbed[:, ~np.isfinite(gauge_observation)] = np.nan
        gauge_y = torch.from_numpy(gauge_perturbed[:, None]).to(device)

        def decode(generated) -> np.ndarray:
            physical = transform.inverse(
                residual.decode(generated, base)[:, 0].float().cpu().numpy()
            )
            return np.where(valid[None], physical, np.nan)

        for variant in variants:
            sampler = base_sampler if variant.streams != "none" else base_background_sampler
            sampler = replace(sampler, seed=day_seed)
            if variant.prior_temperature is not None:
                sampler = replace(sampler, prior_temperature=variant.prior_temperature)
            if variant.n_corrections is not None:
                sampler = replace(sampler, n_corrections=variant.n_corrections)
            guidance = base_guidance
            if variant.huber_delta is not None:
                guidance = replace(guidance, huber_delta=variant.huber_delta)
            if variant.guidance_spread_cells is not None:
                guidance = replace(
                    guidance, spread_cells=variant.guidance_spread_cells
                )

            # --- unguided background ------------------------------------------------
            if variant.streams == "none":
                with torch.inference_mode():
                    generated = run_assim(
                        model,
                        cond,
                        (args.members, 1, grid.nlat, grid.nlon),
                        device,
                        cfg=sampler,
                        flow=flow,
                        mask=mask,
                        to_precip=lambda x, b=base: residual.decode(x, b),
                    )
                fields[variant.name][day_position] = decode(generated)
                continue

            # --- gauges-only localized EnSRF ---------------------------------------
            # The background arm is deliberately first in every resolved group, so
            # this reuses its exact ensemble rather than paying for another draw or
            # introducing a seed/temperature confound.
            if variant.algorithm == "ensrf":
                if variant.streams != "gauges":
                    raise ValueError(
                        f"{variant.name}: algorithm='ensrf' requires streams='gauges'"
                    )
                background_ensemble = fields["background"][day_position]
                if not np.isfinite(background_ensemble[:, valid]).all():
                    raise RuntimeError(
                        f"{variant.name}: background must be generated before EnSRF"
                    )
                updated, ensrf_diagnostic = localized_serial_ensrf(
                    ensemble_mm=background_ensemble,
                    observations_mm=gauge_mm[day_position, assim_idx],
                    station_lat=stations.lat[assim_idx],
                    station_lon=stations.lon[assim_idx],
                    grid=grid,
                    transform=transform,
                    valid=valid,
                    observation_variance=gauge_variance / variant.gauge_weight,
                    localization_km=variant.ensrf_localization_km,
                    seed=day_seed + 3_000_000,
                )
                fields[variant.name][day_position] = updated
                diagnostics.setdefault(variant.name, {}).setdefault("ensrf", []).append(
                    ensrf_diagnostic
                )
                continue

            # --- satellite observation vector for this variant ----------------------
            satellite_y = satellite_R = None
            if variant.uses_imerg:
                source = corrected_imerg_mm if variant.bias_correct else raw_imerg_mm
                satellite_mm = source[day_position].reshape(-1)
                satellite_error_mm = imerg["random_error"][day_position].reshape(-1)
                if variant.bias_correct:
                    # Scale the retrieval error by the local correction ratio so a
                    # map that halves an amount does not leave it twice as certain.
                    raw_flat = raw_imerg_mm[day_position].reshape(-1)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        ratio = np.where(raw_flat > 0.1, satellite_mm / raw_flat, 1.0)
                    satellite_error_mm = satellite_error_mm * np.clip(ratio, 0.25, 4.0)
                _, keep, inflation = satellite_setup(variant)
                satellite_observation = transform.forward(satellite_mm).astype(np.float32)
                if variant.bias_correct and qm_meta:
                    sigma_floor = fitted_sigma_floor(
                        satellite_mm,
                        qm_meta.get("error_bins_mm"),
                        qm_meta.get("error_sigma_transformed"),
                        fallback=float(imerg_config["sigma_obs"]),
                    )
                else:
                    sigma_floor = float(imerg_config["sigma_obs"])
                satellite_variance = transformed_imerg_variance(
                    satellite_mm,
                    satellite_error_mm,
                    transform,
                    sigma_floor=sigma_floor,
                    representativeness=float(imerg_config["representativeness"]),
                )
                satellite_variance = (
                    satellite_variance * np.float32(inflation) / np.float32(variant.imerg_weight)
                )
                keep_now = (
                    keep & np.isfinite(satellite_observation) & np.isfinite(satellite_variance)
                )
                satellite_observation[~keep_now] = np.nan
                satellite_variance[~keep_now] = 1.0
                satellite_R = torch.from_numpy(satellite_variance).to(device)
                satellite_perturbed = perturb_observations(
                    satellite_observation,
                    satellite_R,
                    args.members,
                    seed=day_seed + 2_000_000,
                    corr_blocks=[(0, coarse_shape[0], coarse_shape[1], error_corr_cells)],
                ).astype(np.float32)
                satellite_perturbed[:, ~np.isfinite(satellite_observation)] = np.nan
                satellite_y = torch.from_numpy(satellite_perturbed[:, None]).to(device)

            # --- per-stream likelihood weighting ------------------------------------
            # Weighting a stream by w means dividing BOTH its R and its Gamma
            # inflation by w, so the whole V(t) scales and the cost term scales by
            # exactly w at every t. Scaling R alone would leave the early-time
            # inflation term unweighted.
            gamma = float(base_guidance.gamma)

            if variant.streams == "gauges":
                operator, y, R = gauge_operator, gauge_y, gauge_R / variant.gauge_weight
                gamma_vector = torch.full(
                    (len(assim_idx),), gamma / variant.gauge_weight, device=device
                )
            elif variant.streams == "imerg":
                operator, y, R = satellite_operator, satellite_y, satellite_R
                gamma_vector = torch.full(
                    satellite_R.shape, gamma / variant.imerg_weight, device=device
                )
            else:
                operator = combined_operator
                y = torch.cat([gauge_y, satellite_y], dim=2)
                R = torch.cat([gauge_R / variant.gauge_weight, satellite_R])
                gamma_vector = torch.cat(
                    [
                        torch.full(
                            (len(assim_idx),), gamma / variant.gauge_weight, device=device
                        ),
                        torch.full(
                            satellite_R.shape, gamma / variant.imerg_weight, device=device
                        ),
                    ]
                )
            guidance = replace(guidance, gamma=gamma_vector)

            # --- two-step: satellite in the sampler, gauges by EnSRF afterwards -----
            if variant.algorithm == "twostep_ensrf":
                if not (variant.uses_imerg and variant.uses_gauges):
                    raise ValueError(
                        f"{variant.name}: the two-step scheme needs both streams; "
                        "with one stream it is just the corresponding single-stream arm"
                    )
                generated = run_assim(
                    model,
                    cond,
                    (args.members, 1, grid.nlat, grid.nlon),
                    device,
                    H=satellite_operator,
                    y=satellite_y,
                    R=satellite_R,
                    cfg=sampler,
                    gcfg=replace(
                        guidance,
                        gamma=torch.full(
                            satellite_R.shape, gamma / variant.imerg_weight, device=device
                        ),
                    ),
                    flow=flow,
                    mask=mask,
                    to_precip=lambda x, b=base: residual.decode(x, b),
                ).detach()
                satellite_posterior = decode(generated)
                updated, ensrf_diagnostic = localized_serial_ensrf(
                    ensemble_mm=satellite_posterior,
                    observations_mm=gauge_mm[day_position, assim_idx],
                    station_lat=stations.lat[assim_idx],
                    station_lon=stations.lon[assim_idx],
                    grid=grid,
                    transform=transform,
                    valid=valid,
                    observation_variance=gauge_variance / variant.gauge_weight,
                    localization_km=variant.ensrf_localization_km,
                    seed=day_seed + 3_000_000,
                )
                fields[variant.name][day_position] = updated
                diagnostics.setdefault(variant.name, {}).setdefault("ensrf", []).append(
                    ensrf_diagnostic
                )
                continue

            # --- joint guidance ------------------------------------------------------
            generated = run_assim(
                model,
                cond,
                (args.members, 1, grid.nlat, grid.nlon),
                device,
                H=operator,
                y=y,
                R=R,
                cfg=sampler,
                gcfg=guidance,
                flow=flow,
                mask=mask,
                to_precip=lambda x, b=base: residual.decode(x, b),
            ).detach()
            fields[variant.name][day_position] = decode(generated)

        elapsed = walltime.time() - started
        print(
            f"[sweep] day {day_position + 1}/{n_days} "
            f"{selected_times[day_position].astype('datetime64[D]')} "
            f"({elapsed / (day_position + 1):.0f} s/day)",
            flush=True,
        )
        observation_store_index = int(observation_selected[day_position])
        chirps[day_position] = np.asarray(
            dataset.z["target"][observation_store_index][slices]
        )
        if cpc_full_index is not None:
            condition[day_position] = np.asarray(
                dataset.z["cond"][int(data_index)][cpc_full_index][slices]
            )

    # ---------------------------------------------------------------- scoring
    distance_km = distance_to_nearest_station(
        grid, stations.lat[assim_idx], stations.lon[assim_idx]
    )
    locality_edges = np.array([0, 25, 50, 100, 150, 250, 1e4], dtype=float)
    background_mean = np.nanmean(fields["background"], axis=1)
    withheld_observed = gauge_mm[:, eval_idx]

    for variant in variants:
        members = fields[variant.name]
        at_stations = sample_at_stations(members, grid, stations.lat, stations.lon)
        station_ensembles[variant.name] = at_stations.astype(np.float32)
        entry = score_stations(at_stations[:, :, eval_idx], withheld_observed, transform)
        entry["thresholds"] = {
            str(threshold): brier(at_stations[:, :, eval_idx], withheld_observed, threshold)
            for threshold in (1, 10, 25, 50)
        }
        entry["assimilated_fit"] = score_stations(
            at_stations[:, :, assim_idx], gauge_mm[:, assim_idx], transform
        )
        analysis_mean = np.nanmean(members, axis=1)
        entry["domain_mean_mm"] = float(np.nanmean(analysis_mean))
        entry["wet_day_fraction"] = float(np.nanmean(analysis_mean >= 1.0))
        values, counts = increment_locality(
            analysis_mean, background_mean, distance_km, valid, locality_edges
        )
        entry["increment_locality"] = {
            "edges_km": locality_edges.tolist(),
            "mean_abs_increment_mm": [None if not np.isfinite(v) else float(v) for v in values],
            "n_cells": counts.tolist(),
            "locality_ratio": (
                float(values[0] / values[-2])
                if np.isfinite(values[0]) and np.isfinite(values[-2]) and values[-2] > 0
                else None
            ),
        }
        entry["spec"] = asdict(variant)
        results[variant.name] = entry
        print(
            f"[score] {variant.name:34s} CRPS {entry.get('crps_mm', float('nan')):6.3f}  "
            f"bias {entry.get('mean_bias_mm', float('nan')):+7.3f}  "
            f"MAE {entry.get('mean_mae_mm', float('nan')):6.3f}  "
            f"corr {entry.get('mean_correlation', float('nan')):5.3f}  "
            f"locality {entry['increment_locality']['locality_ratio']}",
            flush=True,
        )

    report = {
        "scope": {
            "start": str(selected_times[0].astype("datetime64[D]")),
            "end": str(selected_times[-1].astype("datetime64[D]")),
            "n_days": n_days,
            "members": args.members,
            "background_day_offset": args.background_day_offset,
            "checkpoint": args.ckpt,
            "checkpoint_data": data_zarr,
            "checkpoint_stats": data_stats,
            "precip_transform": transform.to_dict(),
            "config": args.config,
            "config_overrides": config_overrides,
            "group": args.group,
            "holdout_fold": args.holdout_fold,
            "holdout_folds": args.holdout_folds,
            "n_assimilated_stations": int(len(assim_idx)),
            "n_withheld_stations": int(len(eval_idx)),
            "withheld_station_days": int(np.isfinite(withheld_observed).sum()),
            "imerg_stride_default": args.imerg_stride,
            "imerg_r_multiplier_default": args.imerg_r_multiplier,
            "imerg_bias_correction": qm_meta,
            "seed": args.seed,
            "caveat": (
                "A window this short cannot adjudicate CRPS differences of a few "
                "percent. Use it to screen bias, wet-day frequency, increment "
                "locality and stability; promote at most two arms to the full "
                "multi-year run."
            ),
        },
        "variants": results,
        "ensrf_diagnostics": {
            name: payload.get("ensrf", []) for name, payload in diagnostics.items()
        },
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=float))
    print(f"[sweep] wrote {report_path}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        times=selected_times.astype("datetime64[D]").astype(str),
        station_ids=np.asarray(stations.ids, dtype=str),
        station_lat=stations.lat,
        station_lon=stations.lon,
        eval_idx=eval_idx,
        assim_idx=assim_idx,
        gauge_mm=gauge_mm,
        distance_km=distance_km,
        valid=valid,
        variant_names=np.asarray([v.name for v in variants], dtype=str),
        chirps=chirps,
        condition=condition,
        **{f"station_{name}": values for name, values in station_ensembles.items()},
        **{f"meanfield_{name}": np.nanmean(fields[name], axis=1) for name in fields},
        **({"raw_imerg_mm": raw_imerg_mm} if raw_imerg_mm is not None else {}),
    )
    print(f"[sweep] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
