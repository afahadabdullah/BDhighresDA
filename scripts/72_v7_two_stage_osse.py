#!/usr/bin/env python3
"""V7 end to end, as an OSSE, against mid-training checkpoints.

Stage A samples 0.1-degree rainfall from CPC and ERA5.  Stage B allocates it to
0.05 degrees.  Gauges can be assimilated at either scale, or both, and the whole
point of this script is to find out WHERE they help:

    background   no observations anywhere -- the prior
    da_meso      gauges at 0.1 deg only, then an unguided downscale
    da_fine      unguided 0.1 deg, then gauges at 0.05 deg only
    da_both      gauges at both scales -- the V7 product path

The decomposition is the deliverable.  V5 failed with a single number that said
"DA hurt" and no way to attribute it, and the eventual diagnosis -- that hard
conservation to 0.5 degrees quantised every increment to a 55 km uniform rescale
-- took a dozen runs to isolate.  Stage B conserves to 0.1 degrees, so its
increment is quantised to 11 km instead.  Plausible, not proven.  ``da_fine``
minus ``background`` is the measurement that settles it.

OSSE, so the observations are perfect: pseudo-gauges read CHIRPS at the station
coordinates through the SAME bilinear operator the analysis uses, and the
withheld stations are scored against the same truth.  That removes gauge error
and representativeness from the comparison entirely.  If a perfect observation
degrades a withheld neighbour, the fault is in how the increment propagates, not
in the data -- which is exactly the question.

Both models are expected to be MID-TRAINING.  Checkpoints are snapshotted to a
frozen, timestamped path before anything is sampled: the trainers rewrite
best.pt whenever validation improves, so reading it directly races a writer and
gives results that cannot be attributed to an epoch afterwards.

    python scripts/72_v7_two_stage_osse.py \\
      --meso-checkpoint runs/v7/meso/best.pt \\
      --allocation-checkpoint runs/v7/allocation/best.pt \\
      --meso-archive data/processed/bd_wide_cpc_0p1.zarr \\
      --meso-stats data/processed/stats_v7_meso.json \\
      --subgrid-archive data/processed/v7/wide_v7.zarr \\
      --stations data/stations/data_2020_2025/Stations.csv \\
      --start 2022-05-01 --end 2022-05-03 \\
      --out data/processed/v7_osse/may2022
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bdhires.bmd import neighbored_holdout  # noqa: E402
from bdhires.da import (  # noqa: E402
    BilinearObsOperator,
    CompositeObsOperator,
    GuidanceConfig,
    PhysicalBlockAverageObsOperator,
    SamplerConfig,
    build_R,
    perturb_observations,
)
from bdhires.da.guidance import guided_velocity  # noqa: E402
from bdhires.da.sampler import assimilate as meso_assimilate  # noqa: E402
from bdhires.data.subgrid_dataset import _dequantized_binary_logits  # noqa: E402
from bdhires.data import (  # noqa: E402
    DatasetConfig,
    PrecipDataset,
    SubgridDataset,
    SubgridDatasetConfig,
    area_weighted_block_mean,
    load_stations,
    reconstruct_from_amount,
)
from bdhires.eval.v7_window import bangladesh_window  # noqa: E402
from bdhires.models import (  # noqa: E402
    AllocationFlow,
    RectifiedFlow,
    UNet,
    select_weights,
)
from bdhires.transforms import (  # noqa: E402
    CondTransform,
    PrecipTransform,
    ResidualSpec,
    load_climatology,
)

# (what is assimilated at 0.1 deg, whether gauges act at 0.05 deg).
# The 0.1-degree set is one of: none / gauges / imerg / both.
ARMS: dict[str, tuple[str, bool]] = {
    "background": ("none", False),
    "da_meso": ("gauges", False),
    "da_fine": ("none", True),
    "da_both": ("gauges", True),
    # The three below need --imerg.  BMD-aligned IMERG at observation_factor 2
    # sits on exactly stage A's 0.1-degree cells, so its operator is an identity
    # -- one footprint per state cell, nothing averaged or interpolated.
    "da_imerg": ("imerg", False),
    "da_sim": ("both", False),
    "da_sim_fine": ("both", True),
}

IMERG_ARMS = {name for name, (obs, _) in ARMS.items() if obs in ("imerg", "both")}

DEFAULT_ARMS = ("background", "da_meso", "da_fine", "da_both")

ARM_NOTES = {
    "background": "no observations; the prior both analyses start from",
    "da_meso": "gauges at 0.1 deg only, then an unguided downscale",
    "da_fine": "unguided 0.1 deg, then gauges at 0.05 deg -- the 11 km question",
    "da_both": "gauges at both scales; the V7 product path",
    "da_imerg": "IMERG at 0.1 deg only -- what the satellite alone buys",
    "da_sim": "IMERG + gauges SIMULTANEOUSLY at 0.1 deg",
    "da_sim_fine": "IMERG + gauges at 0.1 deg, then gauges at 0.05 deg",
}


def lag_check(transform, stations, csv_path, day_times, fine_grid,
              min_coverage, applied_offset, truth_field_at_stations) -> dict:
    """Measure the model-vs-gauge day alignment instead of trusting the flag.

    The offset is a convention, and conventions get changed by whoever last
    prepared the data.  A wrong one is invisible: every array has the right
    shape, every station is in the domain, and the only symptom is a large
    unexplained bias -- which is exactly what a badly calibrated prior looks
    like, so the two are easy to confuse.

    This correlates the CHIRPS field at the station points against the gauge
    reports at several lags.  docs/METHOD_SWEEP_PLAN.md measured 0.626 at lag -1
    against 0.271 at lag 0 over many days; one day is far noisier than that, so
    this WARNS rather than fails.
    """
    from bdhires.data import load_stations as _load

    scores: dict[str, float] = {}
    for offset in (-1, 0, 1, 2):
        try:
            probe, values = _load(
                csv_path, day_times + np.timedelta64(offset, "D"),
                grid=fine_grid, min_coverage=0.0,
            )
        except Exception:
            continue
        if len(probe) != len(stations):
            continue
        observed = np.asarray(values, np.float64)
        modelled = np.asarray(truth_field_at_stations, np.float64)[None]
        keep = np.isfinite(observed) & np.isfinite(modelled)
        if keep.sum() < 5:
            continue
        x, y = modelled[keep], observed[keep]
        if x.std() <= 0 or y.std() <= 0:
            continue
        scores[str(offset)] = float(np.corrcoef(x, y)[0, 1])
    if not scores:
        return {}
    best = max(scores, key=scores.get)
    print("gauge alignment check (CHIRPS at stations vs BMD reports):", flush=True)
    for offset, value in sorted(scores.items(), key=lambda kv: int(kv[0])):
        mark = "  <- applied" if int(offset) == applied_offset else ""
        star = " *best" if offset == best else ""
        print(f"    offset {int(offset):+d}: r = {value:+.3f}{star}{mark}", flush=True)
    if int(best) != applied_offset:
        print(
            f"    WARNING: correlation peaks at {int(best):+d}, not the applied "
            f"{applied_offset:+d}.  On a single day this is noisy, but if it "
            f"persists the gauge series and the model are a day apart and every "
            f"score below is meaningless.",
            flush=True,
        )
    return {"correlation_by_offset": scores, "applied": applied_offset, "peak": int(best)}


def transformed_sigma(transform, mm: np.ndarray, sigma_mm: np.ndarray) -> np.ndarray:
    """Carry a physical error into the transform's units by linearisation.

    Stage A's likelihood compares ``tf.forward(rainfall)``, so an error quoted
    in mm/day -- which is how IMERG reports randomError, and how anyone
    naturally thinks about gauges -- is in the wrong units for R.  Ignoring that
    is not a small mistake: with a sqrt transform a 3 mm error is ~0.3
    transformed units in a wet cell and much larger in a dry one, so a flat mm
    figure used directly is wrong by orders of magnitude AND wrong in a
    rain-rate-dependent direction.

    Differentiating numerically rather than by hand keeps this correct for
    whichever transform the statistics were fitted with (sqrt, cbrt, log1p);
    the derivative is evaluated at the observed value, which is the standard
    first-order propagation.
    """
    rainfall = np.clip(np.nan_to_num(mm, nan=0.0), 0.0, None).astype(np.float32)
    error = np.abs(np.asarray(sigma_mm, np.float32))
    # A SECANT across +/- sigma, not a derivative at the point.  sqrt has an
    # infinite slope at zero rainfall, so a pointwise derivative reports a 3 mm
    # error as ~25 transformed units in a dry cell and effectively discards the
    # observation.  The secant asks the question that is actually meant -- how
    # far does this much rainfall error move the transformed value -- and stays
    # finite everywhere.
    upper = np.asarray(transform.forward(rainfall + error), np.float32)
    lower = np.asarray(transform.forward(np.maximum(rainfall - error, 0.0)), np.float32)
    return np.maximum(0.5 * np.abs(upper - lower), 1.0e-4)


def corrupt_satellite(
    truth_mm: np.ndarray,
    *,
    sigma_mm: float,
    sigma_frac: float,
    corr_cells: float,
    bias_frac: float,
    perfect: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn an exact area mean into something a satellite could plausibly report.

    THIS IS THE DIFFERENCE BETWEEN AN OSSE AND A LEAK.  A perfect satellite
    observes every 0.1-degree cell -- including the cells the withheld gauges
    sit in -- so assimilating it hands the analysis the verification truth.
    Every arm that uses it then scores near-perfectly at "withheld" stations,
    gauges appear to add nothing on top, and simultaneous assimilation looks
    slightly WORSE than the satellite alone because the gauges can only perturb
    a field that is already correct.  None of that is a statement about
    observing systems; it is a statement about the experiment.

    The error has three parts, because IMERG's does:

    * an intensity-dependent random term -- retrieval error grows with rain
      rate, so a flat sigma over-trusts heavy scenes and wastes light ones;
    * a SPATIALLY CORRELATED component -- passive-microwave and IR retrieval
      errors travel with the storm, and white noise would be averaged away by
      the analysis almost for free, which flatters the satellite;
    * a multiplicative bias -- the systematic part no amount of data assimilation
      can average out, and the reason gauges carry independent information.

    Returns ``(observed_mm, sigma_mm)`` where sigma is the error standard
    deviation actually used, so R is consistent with how the field was made.
    """
    finite = np.isfinite(truth_mm)
    clean = np.where(finite, truth_mm, 0.0).astype(np.float32)
    sigma = (sigma_mm + sigma_frac * clean).astype(np.float32)
    if perfect:
        # Still report a nonzero sigma: an exactly zero R is a singular
        # likelihood, and the caller is told loudly what this mode means.
        return truth_mm, np.maximum(sigma, 1.0e-3)

    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(clean.shape).astype(np.float32)
    if corr_cells > 0:
        from bdhires.da.observation import _smooth2d

        smoothed = _smooth2d(noise[None], float(corr_cells))[0]
        # Smoothing destroys variance; rescale so sigma still means sigma.
        deviation = float(smoothed.std())
        noise = smoothed / deviation if deviation > 0 else smoothed
    observed = clean * (1.0 + bias_frac) + sigma * noise
    observed = np.maximum(observed, 0.0)          # a satellite cannot see < 0
    return np.where(finite, observed, np.nan).astype(np.float32), sigma


def load_imerg_meso(path: str, model_times: np.ndarray, gauge_offset: int,
                    window, grid, native_corr_cells: float = 3.0) -> dict:
    """Read nested IMERG footprints, choosing the day shift from the file.

    Two attributes decide everything, and both are written by
    ``08_prepare_imerg_observations.py``:

    * ``bmd_accumulation_end_hour_utc`` -- 3 means the file was built on the BMD
      24 h window, so its day D is ~87 percent calendar day D-1 and it takes the
      SAME offset as a gauge report.  0 or 24 means calendar days, which is what
      the model already runs on, so it takes no offset.  Choosing this from the
      file rather than a flag is what stops a calendar-day file being shifted a
      day, and vice versa.
    * ``observation_factor`` -- written only by ``44_coarsen_imerg_observations``
      when a prepared file is coarsened FURTHER.  Its absence therefore means
      native 0.1-degree footprints. Factor 8 is S04 (0.4 degrees) and maps to a
      4x4 physical block-average operator on stage A's 0.1-degree state.

    Neither attribute is authoritative on its own.  The coordinates are: a file
    on a different lattice would have the right shape and the right attributes
    and still put rainfall in the wrong cells, so the lat/lon check below is the
    one that cannot be talked around.
    """
    import xarray as xr

    with xr.open_dataset(path) as dataset:
        for name in ("precipitation", "randomError"):
            if name not in dataset.variables:
                raise SystemExit(f"{path} lacks the IMERG variable {name!r}")
            units = str(dataset[name].attrs.get("units", "")).lower().replace(" ", "")
            if units not in ("mm/day", "mmday-1", "mm day-1", "mm/d"):
                raise SystemExit(f"{path} {name} units are {units!r}; expected mm/day")
        observation_factor = int(dataset.attrs.get("observation_factor", 2))
        if observation_factor < 2 or observation_factor % 2:
            raise SystemExit(
                f"{path} has observation_factor {observation_factor}; V7 stage A "
                "needs footprints nested in an integer number of its 0.1-degree cells"
            )
        state_factor = observation_factor // 2
        native = state_factor == 1
        if native:
            error_corr_cells = float(native_corr_cells)
        else:
            required = dataset.attrs.get("required_error_corr_cells")
            if required is None:
                raise SystemExit(
                    f"{path} is coarsened but lacks required_error_corr_cells; "
                    "rebuild it with scripts/44_coarsen_imerg_observations.py so "
                    "the physical error correlation length is preserved"
                )
            error_corr_cells = float(required)

        end_hour = dataset.attrs.get("bmd_accumulation_end_hour_utc")
        if end_hour is None:
            raise SystemExit(
                f"{path} does not declare bmd_accumulation_end_hour_utc, so its "
                f"24-hour window is unknown and the day alignment cannot be "
                f"determined; rebuild it with 08_prepare_imerg_observations.py"
            )
        end_hour = int(end_hour)
        if end_hour == 3:
            offset = int(gauge_offset)          # BMD window: same shift as a gauge
            window_note = "BMD 03 UTC window"
        elif end_hour in (0, 24):
            offset = 0                          # calendar days: same as the model
            window_note = "calendar 00-24 UTC window"
        else:
            raise SystemExit(
                f"{path} declares an accumulation window ending at {end_hour} UTC, "
                f"which is neither the BMD window (3) nor a calendar day (0/24); "
                f"the correct day shift is undefined"
            )
        days = model_times + np.timedelta64(offset, "D")

        source_days = np.asarray(dataset.time.values).astype("datetime64[D]")
        lookup = {day: index for index, day in enumerate(source_days)}
        missing = [str(day) for day in days if day not in lookup]
        if missing:
            raise SystemExit(f"{path} lacks requested days {missing}")
        index = np.asarray([lookup[day] for day in days], int)
        precipitation = np.asarray(
            dataset["precipitation"].isel(time=index).transpose("time", "lat", "lon"),
            np.float32,
        )
        error = np.asarray(
            dataset["randomError"].isel(time=index).transpose("time", "lat", "lon"),
            np.float32,
        )
        lat = np.asarray(dataset.lat.values, np.float64)
        lon = np.asarray(dataset.lon.values, np.float64)

    if grid.nlat % state_factor or grid.nlon % state_factor:
        raise SystemExit(
            f"stage A grid {(grid.nlat, grid.nlon)} is not divisible by IMERG "
            f"state factor {state_factor}"
        )
    expected_shape = (grid.nlat // state_factor, grid.nlon // state_factor)
    if precipitation.shape[1:] != expected_shape:
        raise SystemExit(
            f"{path} is {precipitation.shape[1:]} but stage A's window is "
            f"{(grid.nlat, grid.nlon)} and factor {state_factor} requires "
            f"{expected_shape}; the IMERG file and analysis window differ"
        )
    expected_lat = grid.lat.reshape(-1, state_factor).mean(axis=1)
    expected_lon = grid.lon.reshape(-1, state_factor).mean(axis=1)
    if not np.allclose(lat, expected_lat, atol=1.0e-5) or not np.allclose(
        lon, expected_lon, atol=1.0e-5
    ):
        raise SystemExit(
            f"{path} footprint centres do not sit on the expected nested stage-A grid "
            f"(file lat {lat[0]:.3f}..{lat[-1]:.3f}, window {grid.lat[0]:.3f}.."
            f"{grid.lat[-1]:.3f}); assimilating it would displace rainfall"
        )
    print(
        f"[imerg] {path}\n"
        f"        {precipitation.shape[0]} day(s), {precipitation.shape[1]}x"
        f"{precipitation.shape[2]} @ {0.1 * state_factor:.1f} deg; "
        f"{state_factor}x{state_factor} physical block operator\n"
        f"        {window_note}; day offset {offset:+d} "
        f"(model {model_times[0]} <- IMERG {days[0]})\n"
        f"        footprints: {'native 0.1 deg' if native else f'coarsened factor {observation_factor}'}; "
        f"error correlation {error_corr_cells:g} footprint cells",
        flush=True,
    )
    return {
        "precipitation": precipitation,
        "error": error,
        "day_offset": offset,
        "state_factor": state_factor,
        "observation_factor": observation_factor,
        "error_corr_cells": error_corr_cells,
        "path": path,
    }


# --------------------------------------------------------------------------
# checkpoints
# --------------------------------------------------------------------------


def snapshot(path: str, destination: Path, label: str) -> tuple[Path, dict]:
    """Freeze a checkpoint that a live trainer is still rewriting.

    ``best.pt`` is replaced whenever validation improves.  Reading it directly
    races that writer -- torch.load on a partially replaced file fails outright,
    and even when it succeeds the epoch it came from is unknowable an hour
    later.  Both trainers write atomically (tmp then rename), so a plain copy
    catches a whole file; this records which one.
    """
    source = Path(path)
    if not source.is_file():
        raise SystemExit(
            f"{label} checkpoint is absent: {source}\n"
            f"Training may not have reached its first validation yet."
        )
    destination.mkdir(parents=True, exist_ok=True)
    frozen = destination / f"{label}_frozen.pt"
    shutil.copy2(source, frozen)
    checkpoint = torch.load(frozen, map_location="cpu", weights_only=False)
    info = {
        "source": str(source),
        "frozen": str(frozen),
        "epoch": int(checkpoint.get("epoch", -1)) + 1,
        # The two trainers name this differently -- scripts/train.py writes
        # best_val_loss, script 57 writes best_val -- and reading only one of
        # them reports a healthy run as nan.
        "best_val": float(
            checkpoint.get("best_val_loss", checkpoint.get("best_val", float("nan")))
        ),
        "weights": str(checkpoint.get("weights", "unknown")),
    }
    print(
        f"[{label}] epoch {info['epoch']}, best_val {info['best_val']:.5f}, "
        f"weights={info['weights']}  -> {frozen}",
        flush=True,
    )
    return frozen, info


def meso_config(frozen: Path) -> dict:
    """The resolved training config stage A was built with.

    The conditioning stack is a CONTRACT, not something to re-derive: the
    channel subset in ``data.cond_channels`` and the seasonal encoding flag both
    change how many channels the network's first convolution expects.  Rebuild
    the dataset from anything other than this and the weights will not load --
    or worse, would load and mean something else.
    """
    checkpoint = torch.load(frozen, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("cfg") or checkpoint.get("config")
    if cfg is None:
        raise SystemExit("stage A checkpoint carries no config; cannot rebuild the model")
    return cfg


def meso_expected_cond_channels(frozen: Path, in_channels: int = 1) -> int:
    """How many conditioning channels the checkpoint's own weights expect.

    Read from ``in_conv.weight`` rather than counted from the config, so a
    mismatch is reported here, in its own terms, instead of surfacing as a
    torch size-mismatch traceback that names two numbers and no cause.
    """
    state = select_weights(torch.load(frozen, map_location="cpu", weights_only=False))
    return int(state["in_conv.weight"].shape[1]) - int(in_channels)


def load_meso(frozen: Path, cond_channels: int, size: int, cfg: dict, device):
    """Stage A: the CPCv2 UNet, rebuilt from the checkpoint's own config."""
    checkpoint = torch.load(frozen, map_location="cpu", weights_only=False)
    model = UNet(
        in_channels=1, cond_channels=cond_channels, out_channels=1,
        image_size=size, **cfg["model"],
    )
    model.load_state_dict(select_weights(checkpoint), strict=True)
    return model.to(device).eval(), cfg


def load_allocation(frozen: Path, fine_cond_channels: int, size: int, device):
    """Stage B: the allocation branch, rebuilt from the checkpoint's own config."""
    checkpoint = torch.load(frozen, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    if cfg["stage"] != "allocation":
        raise SystemExit(f"expected an allocation checkpoint, found {cfg['stage']!r}")
    model = AllocationFlow(fine_cond_channels, image_size=size, **cfg["model"])
    model.load_state_dict(select_weights(checkpoint), strict=True)
    return model.to(device).eval(), cfg


# --------------------------------------------------------------------------
# stage B sampling, guided or not
# --------------------------------------------------------------------------


def allocation_sample(
    model,
    coarse_mm: torch.Tensor,
    fine_cond: torch.Tensor,
    fine_valid: torch.Tensor,
    area: torch.Tensor,
    encoding,
    n_steps: int,
    generator,
    observation=None,
    gcfg: GuidanceConfig | None = None,
    flow: RectifiedFlow | None = None,
) -> torch.Tensor:
    """Heun-integrate the allocation flow, optionally guided by point gauges.

    The coarse amounts are FIXED here -- they came from stage A's analysis, and
    stage B's job is only to place them.  So the guidance gradient is taken with
    respect to the allocation latent alone, and the likelihood is evaluated on
    the reconstructed 0.05-degree field, which is what a gauge actually measures.

    That fixing is also what bounds the increment: reconstruction conserves each
    0.1-degree block exactly, so a gauge here can only move rain BETWEEN the four
    cells of its own block.  11 km, against the 55 km that broke V5.
    """
    flow = flow or RectifiedFlow()
    members = coarse_mm.shape[0]
    shape = (members, 2, *fine_cond.shape[-2:])
    state = torch.randn(shape, device=coarse_mm.device, generator=generator)
    times = torch.linspace(0.0, 1.0, n_steps + 1, device=coarse_mm.device)

    def reconstruct(z: torch.Tensor) -> torch.Tensor:
        return reconstruct_from_amount(
            coarse_mm, z, fine_valid, area, encoding, hard=True
        )

    def velocity(z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if observation is None:
            with torch.no_grad():
                return model(z, t, fine_cond, coarse_context, 0.0)
        # Guided: the network evaluation has to be inside the graph so the
        # likelihood gradient can reach the latent.
        with torch.enable_grad():
            z = z.detach().requires_grad_(True)
            u = model(z, t, fine_cond, coarse_context, 0.0)
            z1 = flow.x1_hat(z, t, u)
            predicted = observation["H"](reconstruct(z1))
            residual = predicted - observation["y"]
            finite = torch.isfinite(residual)
            residual = torch.where(finite, residual, torch.zeros_like(residual))
            variance = observation["R"].view(1, 1, -1)
            delta = gcfg.huber_delta
            if delta is None:
                cost = 0.5 * (residual**2) / variance
            else:
                # Huber: a single wildly wrong gauge must not dominate the
                # analysis, and with perfect observations the tails are the
                # only place a disagreement can come from.
                absolute = residual.abs()
                quadratic = torch.minimum(absolute, torch.full_like(absolute, delta))
                cost = (0.5 * quadratic**2 + delta * (absolute - quadratic)) / variance
            log_likelihood = -(cost * finite).sum()
            grad = torch.autograd.grad(log_likelihood, z)[0]
        if gcfg.clip_norm:
            norm = grad.flatten(1).norm(dim=1).clamp_min(1e-12)
            factor = (gcfg.clip_norm / norm).clamp(max=1.0)
            grad = grad * factor.view(-1, 1, 1, 1)
        return guided_velocity(u.detach(), grad.detach(), t, flow, gcfg, z.detach())

    # Built once, with its own stream, so every Heun evaluation sees the same
    # conditioning -- a context that changed between steps would make the ODE
    # non-autonomous and the trajectory meaningless.
    context_generator = torch.Generator(device="cpu").manual_seed(
        int(torch.randint(0, 2**31 - 1, (1,), generator=generator,
                          device=coarse_mm.device).item())
    )
    coarse_context = coarse_context_of(coarse_mm, encoding, context_generator)
    for i in range(n_steps):
        t0, t1 = times[i], times[i + 1]
        dt = t1 - t0
        v0 = velocity(state, t0.expand(members))
        if i == n_steps - 1:
            state = state + dt * v0
        else:
            v1 = velocity(state + dt * v0, t1.expand(members))
            state = state + 0.5 * dt * (v0 + v1)
        state = state.detach()
    return reconstruct(state).detach()


def coarse_context_of(coarse_mm: torch.Tensor, encoding, generator) -> torch.Tensor:
    """The two-channel coarse state the allocation branch was trained to read.

    Stage B was trained on the archive's stored ``coarse_state``, which is the
    ENCODED coarse field, not raw millimetres.  Re-encoding stage A's analysis
    the same way is what keeps the conditioning distribution the one the branch
    actually saw; handing it mm/day would be a silent domain shift that no shape
    check catches.

    Channel 0 is the standardised square-root amount.  Channel 1 is NOT a +/-1
    flag -- it is a dequantised binary LOGIT, and it is built here by the very
    function ``encode_subgrid_targets`` uses, so the occurrence channel carries
    the same distribution in inference as in training.  Reimplementing it would
    be a second definition of the encoding, which is how the two drift apart.
    """
    amount_root = torch.sqrt(coarse_mm.clamp_min(0.0))
    standardised = (
        amount_root - float(encoding.amount_sqrt_mean)
    ) / float(encoding.amount_sqrt_std)
    wet = (coarse_mm >= float(encoding.wet_threshold_mm)).to(torch.float32)
    logits = _dequantized_binary_logits(wet.cpu(), encoding, generator).to(
        coarse_mm.device, coarse_mm.dtype
    )
    return torch.cat([standardised, logits], dim=1)


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------


def crps(ensemble: np.ndarray, truth: np.ndarray) -> float:
    """Fair CRPS in mm/day: (m-1)-corrected, so ensemble size does not bias it."""
    m = ensemble.shape[0]
    skill = np.abs(ensemble - truth[None]).mean()
    if m < 2:
        return float(skill)
    spread = np.abs(ensemble[:, None] - ensemble[None, :]).mean() * m / (2.0 * (m - 1))
    return float(skill - spread)


def field_pattern_r(analysis: np.ndarray, reference: np.ndarray,
                    valid: np.ndarray) -> float:
    """Spatial pattern correlation of two fields over the valid mask.

    Eleven withheld stations say almost nothing about a 14,400-cell field.  This
    asks the complementary question -- does the analysis LOOK like the
    independent gridded products -- and it is the number that catches an
    analysis fitting its gauges by putting rain in the wrong places.  It is
    scale-free, so it separates "the pattern is right but the amounts are not"
    from "the pattern is wrong", which bias alone cannot do.
    """
    keep = valid & np.isfinite(analysis) & np.isfinite(reference)
    if keep.sum() < 10:
        return float("nan")
    x = analysis[keep].astype(np.float64)
    y = reference[keep].astype(np.float64)
    if x.std() <= 0.0 or y.std() <= 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def score_stations(ensemble_mm: np.ndarray, truth_mm: np.ndarray) -> dict:
    """Score an (M, S) ensemble of station values against (S,) truth."""
    finite = np.isfinite(truth_mm) & np.isfinite(ensemble_mm).all(axis=0)
    if not finite.any():
        return {"stations": 0}
    members = ensemble_mm[:, finite]
    observed = truth_mm[finite]
    mean = members.mean(axis=0)
    rmse = float(np.sqrt(((mean - observed) ** 2).mean()))
    spread = float(members.std(axis=0).mean())
    return {
        "stations": int(finite.sum()),
        "crps_mm": crps(members, observed),
        "mae_mm": float(np.abs(mean - observed).mean()),
        "bias_mm": float((mean - observed).mean()),
        "rmse_mm": rmse,
        "spread_mm": spread,
        # Spread-skill ratio. ~1 is calibrated, >1 over-dispersed, <1
        # over-confident.  CRPS rewards sharpness, so an over-dispersed ensemble
        # improves its CRPS simply by tightening -- which is what shrinking the
        # observation error does.  Without this column that shows up as
        # "smaller R is better" and gets mistaken for a statement about gauges.
        "spread_skill": float(spread / rmse) if rmse > 0 else float("nan"),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--meso-checkpoint", default="runs/v7/meso/best.pt")
    p.add_argument("--allocation-checkpoint", default="runs/v7/allocation/best.pt")
    p.add_argument("--meso-archive", default="data/processed/bd_wide_cpc_0p1.zarr")
    p.add_argument("--meso-stats", default="data/processed/stats_v7_meso.json")
    p.add_argument("--subgrid-archive", default="data/processed/v7/wide_v7.zarr")
    p.add_argument("--stations", default="data/processed/bmd_daily.csv",
                   help="canonical BMD daily CSV from scripts/05_convert_bmd_dir.py "
                        "-- NOT the Stations.csv catalog")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--station-dump",
        default=None,
        help=(
            "optional compressed NPZ of station-space ensembles, observations, "
            "and the exact holdout split. Use it to compare this run against a "
            "different model without reselecting stations."
        ),
    )
    p.add_argument(
        "--map-dump",
        default=None,
        help=(
            "optional compressed NPZ of daily ensemble-mean 0.05-degree fields. "
            "This retains the V7 subgrid output for a matched spatial comparison "
            "with another DA model without writing the full ensemble."
        ),
    )
    p.add_argument("--members", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=50)
    p.add_argument("--withhold", type=float, default=0.30,
                   help="fraction of stations withheld from assimilation")
    p.add_argument("--holdout-neighbor-km", type=float, default=75.0)
    p.add_argument("--holdout-max-gap-deg", type=float, default=200.0)
    p.add_argument("--osse-sigma-mm", type=float, default=0.5,
                   help="assumed gauge error in OSSE mode; small but nonzero keeps "
                        "R invertible")
    # TWO sigmas, because the two stages evaluate their likelihoods in different
    # spaces and there is no single number that is correct for both:
    #   stage A compares tf.forward(gauge_mm) -> TRANSFORMED units
    #   stage B compares reconstruct_from_amount(...) -> PHYSICAL mm/day
    # Sharing one value silently made the 0.1-degree gauge variance 124x too
    # large, so the analysis ignored the observations it was assimilating.
    p.add_argument("--meso-gauge-sigma", type=float, default=None,
                   help="stage A gauge error in TRANSFORMED units (build_R's own "
                        "space). Defaults to configs/da.yaml's tuned 0.10 for real "
                        "observations and 0.05 for OSSE")
    p.add_argument("--meso-sigma-sweep", default=None,
                   help="comma-separated TOTAL stage-A gauge errors (transformed "
                        "units): each value replaces sqrt(sigma^2 + "
                        "representativeness^2) outright, so the sweep can go below "
                        "the representativeness floor. CPCv2's tuned pair 0.10+0.25 "
                        "is a total of 0.269, and it was tuned against 0.05-degree "
                        "cells; stage A assimilates onto 0.1-degree cells, four "
                        "times the area, so it is not obviously transferable")
    p.add_argument("--meso-gauge-representativeness", type=float, default=None,
                   help="stage A point-vs-cell mismatch, TRANSFORMED units. Defaults "
                        "to 0.25 for real (usually the DOMINANT term) and 0.0 for "
                        "OSSE, where the pseudo-gauge is read with the same operator "
                        "the analysis uses and there is no mismatch")
    p.add_argument("--fine-gauge-sigma-mm", type=float, default=None,
                   help="stage B gauge error in PHYSICAL mm/day, because that "
                        "likelihood is evaluated on reconstructed rainfall. "
                        "Defaults to 3.0 for real and 0.5 for OSSE")
    p.add_argument("--representativeness-mm", type=float, default=0.0)
    p.add_argument("--min-coverage", type=float, default=0.8)
    p.add_argument("--gauge-day-offset", type=int, default=None,
                   help="days added to the MODEL date to find the BMD date. "
                        "BMD day D is the 24h ending 03:00 UTC on D, so it is ~87 "
                        "percent calendar day D-1; CHIRPS and CPC are 00-00 UTC "
                        "calendar days.  Model calendar day C therefore corresponds "
                        "to BMD day C+1, and the default is +1 for real observations "
                        "(0 for OSSE, where the pseudo-gauge IS the model-day field "
                        "and no window mismatch exists).  docs/METHOD_SWEEP_PLAN.md "
                        "measures the lag: CHIRPS-vs-BMD correlation is 0.626 at "
                        "lag -1 against 0.271 at lag 0")
    p.add_argument("--guidance-gamma", type=float, default=1.0e-3)
    p.add_argument("--guidance-scale", type=float, default=1.0)
    p.add_argument("--guidance-clip", type=float, default=50.0)
    p.add_argument("--huber-delta", type=float, default=3.0)
    p.add_argument("--spread-cells", type=float, default=0.0,
                   help="Gaussian spreading of the 0.1-degree guidance gradient, in cells")
    p.add_argument("--seed", type=int, default=20220503)
    p.add_argument("--observations", choices=("osse", "real"), default="osse",
                   help="osse: pseudo-gauges read CHIRPS at the station coordinates, "
                        "so gauge error and representativeness drop out. "
                        "real: assimilate and verify against the ACTUAL BMD reports, "
                        "which is the test CPCv2 was judged on")
    p.add_argument("--osse-satellite", action="store_true",
                   help="synthesise the 0.1-degree satellite stream from CHIRPS "
                        "(perfect AREA-AVERAGE observations) instead of reading a "
                        "real IMERG file; consistent with the perfect gauges and "
                        "needs no external data")
    # A pseudo-satellite MUST carry error.  A perfect one observes every cell,
    # including the cells the withheld gauges sit in, so it hands the analysis
    # the verification truth and every arm that uses it scores near-perfectly
    # for no scientific reason.  Defaults are IMERG-like: a floor plus an
    # intensity-dependent term, a spatially correlated component because
    # retrieval error is not white, and a multiplicative bias.
    p.add_argument("--osse-satellite-sigma-mm", type=float, default=2.0,
                   help="random error floor on the pseudo-satellite (mm/day)")
    p.add_argument("--osse-satellite-sigma-frac", type=float, default=0.35,
                   help="additional random error as a fraction of the rain rate")
    p.add_argument("--osse-satellite-corr-cells", type=float, default=2.0,
                   help="spatial correlation length of the satellite error, in "
                        "0.1-degree cells; 0 makes it white, which is unrealistic")
    p.add_argument("--osse-satellite-bias-frac", type=float, default=0.10,
                   help="systematic multiplicative bias, e.g. 0.10 = 10 percent wet")
    p.add_argument("--osse-satellite-perfect", action="store_true",
                   help="DIAGNOSTIC ONLY: no satellite error at all. This leaks the "
                        "verification truth into the analysis and its scores are "
                        "meaningless as skill -- use it to bound what is achievable, "
                        "never to compare observing systems")
    p.add_argument("--imerg", default=None,
                   help="BMD-aligned IMERG netCDF at observation_factor 2 (0.1 deg); "
                        "enables the da_imerg / da_sim / da_sim_fine arms")
    p.add_argument("--imerg-s04", default=None,
                   help="the same BMD-windowed IMERG coarsened to factor 8 (0.4 "
                        "degrees). Adds controlled S04 simultaneous arms with a "
                        "4x4 physical footprint operator")
    p.add_argument("--imerg-sigma-floor-mm", type=float, default=1.0,
                   help="floor on IMERG randomError, so a zero error is not infinite weight")
    p.add_argument("--imerg-error-corr-cells", type=float, default=3.0,
                   help="spatial correlation length of the satellite error, in "
                        "0.1-degree cells, matching configs/da.yaml. Satellite "
                        "retrieval error is correlated over tens of kilometres; "
                        "WHITE perturbations average out over any neighbourhood and "
                        "add almost no ensemble spread, which is how a 4096-footprint "
                        "field collapses the analysis")
    p.add_argument("--imerg-r-multiplier", type=float, default=None,
                   help="inflate the satellite variance to reflect how many of its "
                        "footprints are actually INDEPENDENT. With a correlation "
                        "length of L cells a patch of about L^2 cells carries one "
                        "datum, so the default is L^2: without it a diagonal R "
                        "asserts 4096 independent constraints where there are a few "
                        "hundred, and the ensemble is crushed")
    p.add_argument("--imerg-r-sweep", default=None,
                   help="comma-separated satellite R multipliers. Adds one da_sim "
                        "arm per value, so the RELATIVE weight of satellite against "
                        "gauges can be scanned. 1 trusts every footprint as an "
                        "independent datum; L^2 = 9 assumes one datum per "
                        "correlation patch; larger still treats the satellite as "
                        "broad-scale guidance only")
    p.add_argument("--imerg-refine-r", type=float, default=None,
                   help="add CPCv2-derived simultaneous refinements at this R "
                        "multiplier: ig010 (gamma 0.01) with its original L2 loss. "
                        "The ordinary da_sim_rN arm is the gamma-0.001/Huber-3 control")
    p.add_argument("--imerg-representativeness", type=float, default=0.10,
                   help="footprint-vs-block-mean mismatch in TRANSFORMED units, "
                        "matching configs/da.yaml; the retrieval error itself is "
                        "read from the file in mm and converted")
    p.add_argument("--arms", default=",".join(DEFAULT_ARMS),
                   help="comma-separated subset of " + ",".join(ARMS)
                        + " (IMERG arms require --imerg; --imerg-s04 adds its own arms)")
    p.add_argument("--no-plots", action="store_true",
                   help="write the JSON only; figures are on by default")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    arms = [name.strip() for name in args.arms.split(",") if name.strip()]
    unknown = sorted(set(arms) - set(ARMS))
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; choose from {sorted(ARMS)}")
    if not arms:
        raise SystemExit("no arms selected")
    needs_satellite = sorted(set(arms) & IMERG_ARMS)
    if needs_satellite and not (args.imerg or args.osse_satellite):
        # Checked here, before any sampling: discovering it after three arms
        # have run wastes the whole allocation.
        raise SystemExit(
            f"arms {needs_satellite} need a 0.1-degree satellite stream.\n"
            f"  --osse-satellite   synthesise it from CHIRPS (perfect, no file)\n"
            f"  --imerg PATH       assimilate a real BMD-aligned IMERG file"
        )
    if args.imerg and args.osse_satellite:
        raise SystemExit("--imerg and --osse-satellite are alternatives, not both")
    if args.imerg_s04 and not args.imerg:
        raise SystemExit("--imerg-s04 is a scale comparison and also needs native --imerg")
    if args.observations == "real":
        if args.osse_satellite:
            raise SystemExit(
                "--osse-satellite synthesises the satellite from CHIRPS, which is "
                "not a real observation; pass --imerg with real observations"
            )
        if args.min_coverage <= 0.0:
            # With real gauges a station that filed no report contributes
            # nothing but a NaN, and a network silently full of them would make
            # the withheld score rest on two or three stations.
            raise SystemExit(
                "--observations real needs --min-coverage > 0 so that stations "
                "which did not report are dropped rather than carried as NaN"
            )

    # Each sweep value becomes its own da_meso arm with its own sigma; the rest
    # of the configuration is identical, so a difference between them is the
    # weighting and nothing else.
    arm_sigma: dict[str, float] = {}
    arm_imerg_r: dict[str, float] = {}
    arm_imerg_stream: dict[str, str] = {}
    arm_guidance_gamma: dict[str, float] = {}
    arm_huber_delta: dict[str, float | None] = {}
    if args.imerg_r_sweep:
        if not (args.imerg or args.osse_satellite):
            raise SystemExit("--imerg-r-sweep needs --imerg or --osse-satellite")
        for token in args.imerg_r_sweep.split(","):
            token = token.strip()
            if not token:
                continue
            value = float(token)
            if value <= 0.0:
                raise SystemExit(f"sweep R multiplier {value} must be positive")
            name = f"da_sim_r{token}"
            ARMS[name] = ("both", False)
            ARM_NOTES[name] = (
                f"IMERG + gauges simultaneously, satellite R x{value:g}"
            )
            arm_imerg_r[name] = value
            if name not in arms:
                arms.append(name)
        for anchor in ("da_meso", "background"):
            if anchor not in arms:
                arms.insert(0, anchor)
    if args.imerg_refine_r is not None:
        if not (args.imerg or args.osse_satellite):
            raise SystemExit("--imerg-refine-r needs --imerg or --osse-satellite")
        if args.imerg_refine_r <= 0.0:
            raise SystemExit("--imerg-refine-r must be positive")
        r_value = float(args.imerg_refine_r)
        r_token = f"{r_value:g}".replace(".", "p")
        control = f"da_sim_r{r_token}"
        if control not in ARMS:
            ARMS[control] = ("both", False)
            ARM_NOTES[control] = (
                f"IMERG + gauges simultaneously, satellite R x{r_value:g}; "
                "gamma 0.001 with Huber-3 refinement control"
            )
        arm_imerg_r[control] = r_value
        if control not in arms:
            arms.append(control)
        refinements = (("g010_l2", 1.0e-2, None,
                        "CPCv2 primary ig010 mechanism: gamma 0.01 with L2"),)
        for suffix, gamma, huber, note in refinements:
            name = f"da_sim_r{r_token}_{suffix}"
            ARMS[name] = ("both", False)
            ARM_NOTES[name] = (
                f"IMERG + gauges simultaneously, satellite R x{r_value:g}; {note}"
            )
            arm_imerg_r[name] = r_value
            arm_guidance_gamma[name] = gamma
            arm_huber_delta[name] = huber
            if name not in arms:
                arms.append(name)
        for anchor in ("da_meso", "background"):
            if anchor not in arms:
                arms.insert(0, anchor)
    if args.imerg_s04:
        if args.osse_satellite:
            raise SystemExit("--imerg-s04 cannot be combined with --osse-satellite")
        s04_arms = (
            ("da_sim_s04_r1_g001_h3", 1.0, 1.0e-3, 3.0,
             "S04 strong-likelihood check"),
            ("da_sim_s04_corr_g001_h3", None, 1.0e-3, 3.0,
             "S04 correlation-adjusted V7 robust configuration"),
            ("da_sim_s04_corr_g010_l2", None, 1.0e-2, None,
             "S04 correlation-adjusted CPCv2 ig010 configuration"),
        )
        for name, multiplier, gamma, huber, note in s04_arms:
            ARMS[name] = ("both", False)
            ARM_NOTES[name] = f"IMERG 0.4-degree S04 + gauges; {note}"
            if multiplier is not None:
                arm_imerg_r[name] = multiplier
            arm_imerg_stream[name] = "s04"
            arm_guidance_gamma[name] = gamma
            arm_huber_delta[name] = huber
            if name not in arms:
                arms.append(name)
    if args.meso_sigma_sweep:
        for token in args.meso_sigma_sweep.split(","):
            token = token.strip()
            if not token:
                continue
            value = float(token)
            if value <= 0.0:
                raise SystemExit(f"sweep sigma {value} must be positive")
            name = f"da_meso_tot{token}"
            ARMS[name] = ("gauges", False)
            ARM_NOTES[name] = (
                f"gauges at 0.1 deg, TOTAL observation error {value:g} (transformed)"
            )
            arm_sigma[name] = value
            if name not in arms:
                arms.append(name)
        if "background" not in arms:
            arms.insert(0, "background")

    device = torch.device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    window = bangladesh_window()
    print(window.describe(), flush=True)
    # Resolved once and printed: an observation error that silently differs
    # between runs makes a configuration difference look like a scientific one.
    real = args.observations == "real"
    if args.meso_gauge_sigma is None:
        args.meso_gauge_sigma = 0.10 if real else 0.05
    if args.meso_gauge_representativeness is None:
        args.meso_gauge_representativeness = 0.25 if real else 0.0
    if args.fine_gauge_sigma_mm is None:
        args.fine_gauge_sigma_mm = 3.0 if real else args.osse_sigma_mm
    if args.gauge_day_offset is None:
        args.gauge_day_offset = 1 if real else 0
    if args.imerg_r_multiplier is None:
        # L^2 cells per independent datum, floored at 1 (no inflation when the
        # error really is white).
        args.imerg_r_multiplier = max(1.0, float(args.imerg_error_corr_cells) ** 2)
    # Put the unsuffixed simultaneous arm into the same reporting axis as any
    # additional R-sweep arms.  This is metadata only: its likelihood already
    # uses args.imerg_r_multiplier below.  Without it a 9/27/81 experiment
    # prints only 27 and 81 in the sweep table and hides its control.
    if args.imerg_r_sweep and "da_sim" in arms:
        arm_imerg_r["da_sim"] = float(args.imerg_r_multiplier)
    for arm in arms:
        arm_imerg_stream.setdefault(arm, "native")
        arm_guidance_gamma.setdefault(arm, float(args.guidance_gamma))
        arm_huber_delta.setdefault(arm, args.huber_delta)
    print(
        f"observations: {args.observations.upper()}"
        + ("  (pseudo-gauges and pseudo-satellite read CHIRPS)"
           if args.observations == "osse"
           else "  (actual BMD reports, assimilated and verified)")
        + f"\n  gauge error: stage A sigma {args.meso_gauge_sigma:g} + "
        f"representativeness {args.meso_gauge_representativeness:g} (transformed); "
        f"stage B sigma {args.fine_gauge_sigma_mm:g} mm/day (physical)",
        flush=True,
    )

    # ---- frozen checkpoints, before anything is sampled -------------------
    meso_frozen, meso_info = snapshot(
        args.meso_checkpoint, out_dir / "checkpoints", "meso"
    )
    alloc_frozen, alloc_info = snapshot(
        args.allocation_checkpoint, out_dir / "checkpoints", "allocation"
    )

    # ---- stage A dataset, built from the checkpoint's own data contract ----
    meso_cfg = meso_config(meso_frozen)
    trained_stats = meso_cfg["data"].get("stats")
    if trained_stats and Path(trained_stats) != Path(args.meso_stats):
        # Normalisation moments are part of the weights' meaning.  Different
        # statistics would load cleanly and silently mis-scale every input.
        raise SystemExit(
            f"stage A was trained with stats {trained_stats!r} but this run was "
            f"given {args.meso_stats!r}; they must be the same file"
        )
    stats = json.loads(Path(args.meso_stats).read_text())
    tf = PrecipTransform.from_dict(stats["precip_transform"])
    selected = meso_cfg["data"].get("cond_channels")
    meso_ds = PrecipDataset(
        DatasetConfig(
            root=args.meso_archive,
            crop=window.meso_size,
            random_crop=False,
            crop_origin=window.meso_origin,
            # Straight from the training config: the channel subset and the
            # seasonal encoding each change the conditioning width, and the
            # first convolution was sized for exactly one combination.
            seasonal_encoding=meso_cfg["data"].get("seasonal_encoding", True),
            cond_channels=tuple(selected) if selected else None,
            era5_member=meso_cfg["data"].get("era5_member"),
        ),
        tf,
        cond_mean=np.asarray(stats["cond_mean"], np.float32),
        cond_std=np.asarray(stats["cond_std"], np.float32),
        cond_transform=CondTransform.from_stats(stats),
        residual=ResidualSpec.from_stats(stats),
        climatology=load_climatology(args.meso_stats, stats),
    )
    expected = meso_expected_cond_channels(meso_frozen)
    if meso_ds.total_cond_channels != expected:
        raise SystemExit(
            f"stage A expects {expected} conditioning channels but this dataset "
            f"builds {meso_ds.total_cond_channels}.\n"
            f"  cond_channels    : {selected}\n"
            f"  seasonal_encoding: {meso_cfg['data'].get('seasonal_encoding', True)}\n"
            f"  archive          : {args.meso_archive}\n"
            f"The checkpoint and the archive disagree about the conditioning "
            f"stack; they must come from the same run."
        )
    print(
        f"stage A conditioning: {meso_ds.total_cond_channels} channels "
        f"({len(selected) if selected else 'all'} selected + seasonal)",
        flush=True,
    )
    meso_model, _ = load_meso(
        meso_frozen, meso_ds.total_cond_channels, window.meso_size, meso_cfg, device
    )

    # ---- stage B dataset --------------------------------------------------
    subgrid_ds = SubgridDataset(
        SubgridDatasetConfig(
            root=args.subgrid_archive,
            crop=window.fine_size,
            random_crop=False,
            crop_origin=window.fine_origin,
            factor=2,
        )
    )
    encoding = subgrid_ds.encoding
    if int(encoding.factor) != 2:
        raise SystemExit(f"stage B archive is factor {encoding.factor}, not 2")
    fine_cond_channels = int(subgrid_ds.z["fine_cond"].shape[1])
    alloc_model, _ = load_allocation(
        alloc_frozen, fine_cond_channels, window.fine_size, device
    )

    # ---- days -------------------------------------------------------------
    meso_times = meso_ds.time.astype("datetime64[D]")
    sub_times = subgrid_ds.time[subgrid_ds.index].astype("datetime64[D]")
    wanted = np.arange(
        np.datetime64(args.start, "D"), np.datetime64(args.end, "D") + 1
    )
    days = []
    for day in wanted:
        a = np.flatnonzero(meso_times == day)
        b = np.flatnonzero(sub_times == day)
        if a.size and b.size:
            days.append((str(day), int(a[0]), int(b[0])))
        else:
            print(f"  skipping {day}: absent from "
                  f"{'stage A' if not a.size else 'stage B'}", flush=True)
    if not days:
        raise SystemExit(f"no day in {args.start}..{args.end} exists in both archives")
    print(f"days: {', '.join(d for d, _, _ in days)}", flush=True)

    # ---- stations, and the OSSE truth they read ---------------------------
    fine_grid = window.fine_grid()
    day_times = np.asarray([np.datetime64(d) for d, _, _ in days])
    # The BMD accumulation window does not align with the model's calendar day,
    # so the gauge series is read on shifted dates and then indexed by MODEL day.
    gauge_times = day_times + np.timedelta64(int(args.gauge_day_offset), "D")
    if args.gauge_day_offset:
        print(
            f"gauge day offset: {args.gauge_day_offset:+d} "
            f"(model {day_times[0]} <- BMD {gauge_times[0]}); BMD day D is the "
            f"24h ending 03 UTC on D, so it is ~87% calendar day D-1",
            flush=True,
        )
    try:
        stations, gauge_mm = load_stations(
            args.stations, gauge_times, grid=fine_grid,
            min_coverage=args.min_coverage,
        )
    except ValueError as error:
        # Stations.csv is the station CATALOG (id, name, lat, lon).  What
        # load_stations wants is the canonical DAILY file, which
        # scripts/05_convert_bmd_dir.py builds from the per-station directory
        # plus that catalog.  Handing over the catalog is the easy mistake, so
        # name the fix instead of re-raising a column list.
        if "missing columns" not in str(error):
            raise
        raise SystemExit(
            f"{args.stations} is not the canonical BMD daily CSV.\n"
            f"  {error}\n"
            f"That file looks like the station CATALOG.  Build the daily file "
            f"first:\n"
            f"    python scripts/05_convert_bmd_dir.py \\\n"
            f"      --data-dir data/stations/data_2020_2025 \\\n"
            f"      --stations {args.stations} \\\n"
            f"      --start {args.start} --end {args.end} \\\n"
            f"      --out <out>/bmd_daily.csv\n"
            f"then pass --stations <out>/bmd_daily.csv"
        ) from error
    if len(stations) < 5:
        raise SystemExit(f"only {len(stations)} stations fall inside {fine_grid.name}")
    n_withheld = max(1, min(len(stations) - 1,
                            int(round(args.withhold * len(stations)))))
    withheld = neighbored_holdout(
        stations.lat, stations.lon, n_withheld,
        radius_km=args.holdout_neighbor_km,
        max_gap_deg=args.holdout_max_gap_deg,
    )
    assimilated = np.setdiff1d(np.arange(len(stations)), withheld)
    print(
        f"stations: {len(stations)} total, {len(assimilated)} assimilated, "
        f"{len(withheld)} withheld (neighboured, {args.holdout_neighbor_km:g} km)",
        flush=True,
    )

    fine_operator = BilinearObsOperator(fine_grid, stations.lat, stations.lon).to(device)
    assim_operator = BilinearObsOperator(
        fine_grid, stations.lat[assimilated], stations.lon[assimilated]
    ).to(device)

    meso_grid = _meso_grid(window)
    imerg = None
    imerg_streams: dict[str, dict] = {}
    if args.imerg:
        # SAME shift as the gauges, and for the same reason.  load_imerg_meso
        # refuses any file that is not on the BMD 03:00 UTC window, so the file
        # IS BMD-windowed by construction: IMERG day D covers the 24 h ending
        # 03 UTC on D and is therefore ~87 percent calendar day D-1, exactly
        # like a BMD report.  The model is on calendar days, so model day C
        # reads IMERG day C+1.
        #
        # This is easy to get backwards, because docs/METHOD_SWEEP_PLAN.md's lag
        # table shows IMERG peaking at lag 0 while CHIRPS and CPC peak at -1.
        # That table is measured against BMD GAUGES, not against the model: it
        # says IMERG and the gauges already share a window, which is precisely
        # why they take the same offset relative to a calendar-day model.
        imerg = load_imerg_meso(
            args.imerg, day_times, int(args.gauge_day_offset), window, meso_grid,
            native_corr_cells=float(args.imerg_error_corr_cells),
        )
        imerg_streams["native"] = imerg
    if args.imerg_s04:
        s04 = load_imerg_meso(
            args.imerg_s04, day_times, int(args.gauge_day_offset), window, meso_grid,
            native_corr_cells=float(args.imerg_error_corr_cells),
        )
        if s04["observation_factor"] != 8 or s04["state_factor"] != 4:
            raise SystemExit(
                f"--imerg-s04 must be factor 8 / 0.4 degrees, got factor "
                f"{s04['observation_factor']}"
            )
        imerg_streams["s04"] = s04
        correlation_inflation = max(
            1.0, 2.0 * np.pi * float(s04["error_corr_cells"]) ** 2
        )
        for arm in (
            "da_sim_s04_corr_g001_h3", "da_sim_s04_corr_g010_l2"
        ):
            arm_imerg_r[arm] = correlation_inflation
            ARM_NOTES[arm] += f" (R x{correlation_inflation:.3f})"

    gcfg = GuidanceConfig(
        gamma=args.guidance_gamma,
        scale=args.guidance_scale,
        clip_norm=args.guidance_clip,
        huber_delta=args.huber_delta,
        spread_cells=args.spread_cells,
    )
    scfg = SamplerConfig(
        n_steps=args.n_steps, heun=True, seed=args.seed,
        mask_fill=meso_ds.mask_fill,
    )
    flow = RectifiedFlow()
    mask = torch.from_numpy(meso_ds.fixed_valid[None, None]).to(device)

    results = {
        "window": window.describe(),
        "arm_sigma": arm_sigma,
        "arm_imerg_r": arm_imerg_r,
        "arm_imerg_stream": arm_imerg_stream,
        "arm_guidance_gamma": arm_guidance_gamma,
        "arm_huber_delta": arm_huber_delta,
        "checkpoints": {"meso": meso_info, "allocation": alloc_info},
        "members": args.members,
        "n_steps": args.n_steps,
        "observations": args.observations,
        "osse": args.observations == "osse",
        "imerg_day_offset": (imerg or {}).get("day_offset"),
        "imerg_error_corr_cells": {
            name: stream["error_corr_cells"] for name, stream in imerg_streams.items()
        },
        "imerg_r_multiplier": args.imerg_r_multiplier,
        "satellite": (
            "pseudo (CHIRPS + error model)" if args.osse_satellite
            else ({name: stream["path"] for name, stream in imerg_streams.items()}
                  if imerg_streams else "none")
        ),
        "meso_gauge_sigma_transformed": args.meso_gauge_sigma,
        "meso_gauge_representativeness": args.meso_gauge_representativeness,
        "fine_gauge_sigma_mm": args.fine_gauge_sigma_mm,
        "gauge_day_offset": int(args.gauge_day_offset),
        "model_dates": [str(value) for value in day_times.astype("datetime64[D]")],
        "gauge_dates": [str(value) for value in gauge_times.astype("datetime64[D]")],
        "stations": {
            "total": int(len(stations)),
            "assimilated": [str(s) for s in stations.ids[assimilated]],
            "withheld": [str(s) for s in stations.ids[withheld]],
            "holdout_neighbor_km": args.holdout_neighbor_km,
        },
        "arms": {name: {"note": ARM_NOTES[name], "days": []} for name in arms},
    }

    # Keep the raw station ensembles only when explicitly requested.  A scalar
    # summary cannot prove that a CPCv2 comparison used the same BMD values and
    # withheld IDs; this compact dump can, while avoiding large gridded output.
    station_ensembles = (
        {
            arm: np.full((len(days), args.members, len(stations)), np.nan,
                         dtype=np.float32)
            for arm in arms
        }
        if args.station_dump else None
    )
    observed_at_stations = (
        np.full((len(days), len(stations)), np.nan, dtype=np.float32)
        if args.station_dump else None
    )
    # A mean-field dump is deliberately separate from the station dump: maps
    # are useful for checking whether a model retains 0.05-degree structure,
    # while the raw station ensembles are the audit trail for fair CRPS.
    map_means = (
        {
            arm: np.full((len(days), fine_grid.nlat, fine_grid.nlon), np.nan,
                         dtype=np.float32)
            for arm in arms
        }
        if args.map_dump else None
    )
    map_valid = (
        np.zeros((len(days), fine_grid.nlat, fine_grid.nlon), dtype=bool)
        if args.map_dump else None
    )

    panels: dict = {}
    truth_fields: dict = {}
    for day_position, (date, meso_index, sub_index) in enumerate(days):
        item = meso_ds[meso_index]
        cond = item["cond"][None].to(device)
        base = item["base"][None].to(device)
        sub = subgrid_ds[sub_index]
        fine_truth = sub["fine_mm"][None].to(device)
        fine_valid = sub["fine_valid"][None].to(device)
        area = sub["cell_area"][None].to(device)
        fine_cond = sub["fine_cond"][None].to(device)

        if args.observations == "osse":
            # Perfect observations: CHIRPS read at the station coordinates
            # through the SAME operator the analysis uses, so a pseudo-gauge and
            # the analysis see the identical quantity and no interpolation
            # mismatch can confound the result.
            with torch.no_grad():
                truth_at_stations = fine_operator(fine_truth)[0, 0].cpu().numpy()
        else:
            # Real BMD reports, assimilated AND verified.  Now gauge error,
            # representativeness and the CHIRPS-vs-gauge difference are all in
            # play -- which is the point: it is the test CPCv2 was judged on,
            # and the only one whose numbers describe the actual product.
            truth_at_stations = np.asarray(gauge_mm[day_position], np.float32)
            reporting = int(np.isfinite(truth_at_stations).sum())
            if reporting < len(stations):
                print(f"      {len(stations) - reporting} station(s) did not report "
                      f"on {date}; they are scored as missing, not as zero",
                      flush=True)
        truth_assim = truth_at_stations[assimilated]
        truth_withheld = truth_at_stations[withheld]
        if observed_at_stations is not None:
            observed_at_stations[day_position] = truth_at_stations
        if not np.isfinite(truth_withheld).any():
            raise SystemExit(
                f"no withheld station reported on {date}; the verification set "
                f"is empty and every score would be vacuous"
            )
        # A perfect satellite sees the exact 0.1-degree area mean of the truth.
        # That is precisely the archive's coarse_mm, so nothing is recomputed.
        # It covers stage B's window, which is a SUB-window of stage A's state,
        # so the rest of the field is NaN -- unobserved, not zero.
        satellite_mm = satellite_sigma = None
        if args.osse_satellite:
            satellite_mm = np.full(
                (window.meso_size, window.meso_size), np.nan, np.float32
            )
            row, column = window.meso_local
            satellite_mm[
                row:row + window.coarse_size, column:column + window.coarse_size
            ] = sub["coarse_mm"][0].numpy()
            satellite_mm, satellite_sigma = corrupt_satellite(
                satellite_mm,
                sigma_mm=args.osse_satellite_sigma_mm,
                sigma_frac=args.osse_satellite_sigma_frac,
                corr_cells=args.osse_satellite_corr_cells,
                bias_frac=args.osse_satellite_bias_frac,
                perfect=args.osse_satellite_perfect,
                seed=args.seed + meso_index,
            )

        if args.observations == "real" and day_position == 0:
            with torch.no_grad():
                chirps_at_stations = fine_operator(fine_truth)[0, 0].cpu().numpy()
            results["alignment"] = lag_check(
                tf, stations, args.stations, day_times[:1], fine_grid,
                args.min_coverage, int(args.gauge_day_offset), chirps_at_stations,
            )

        truth_fields[date] = {
            "field": fine_truth[0, 0].cpu().numpy(),
            "valid": sub["fine_valid"][0].numpy().astype(bool),
            "at_stations": truth_at_stations,
        }

        for arm in arms:
            meso_obs, guide_fine = ARMS[arm]
            guide_meso = meso_obs != "none"
            arm_gcfg = replace(
                gcfg,
                gamma=arm_guidance_gamma[arm],
                huber_delta=arm_huber_delta[arm],
            )
            generator = torch.Generator(device=device).manual_seed(
                args.seed + 1000 * meso_index
            )
            print(f"  {date}  {arm:<12s} "
                  f"(0.1 deg: {meso_obs:<6s}  0.05 deg: "
                  f"{'gauges' if guide_fine else 'none  '}; "
                  f"gamma {arm_gcfg.gamma:g}; "
                  f"loss {'L2' if arm_gcfg.huber_delta is None else f'Huber-{arm_gcfg.huber_delta:g}'})...",
                  flush=True)

            # -- stage A ------------------------------------------------------
            meso_H = meso_y = meso_R = None
            if guide_meso:
                # Both streams go through the SAME likelihood; neither is a
                # conditioning channel.  Stacking them into one operator and one
                # R is what makes "simultaneous" mean simultaneous rather than
                # two sequential updates that double-count the prior.
                # TWO variance vectors, because they answer different questions.
                #   variances       -> the LIKELIHOOD weight. Inflated for
                #                      correlated satellite footprints, because
                #                      4096 of them carry far less information
                #                      than 4096 independent data.
                #   perturb_variance -> the actual OBSERVATION ERROR, used to
                #                      draw one plausible realisation per member.
                # Inflating both double-counts, and worse: the draws live in
                # transformed space where sqrt is convex, so tripling their
                # amplitude rectifies into a large WET bias in mm (Jensen; see
                # scripts/31_jensen_bias_audit.py). That is what turned bias
                # +0.4 into +15.0 and spread 7.6 into 49.8.
                operators, values, variances, perturb_variances = [], [], [], []

                if meso_obs in ("gauges", "both"):
                    # A gauge is a POINT measurement, read here as the value of
                    # the 0.1-degree cell containing it.  That is the honest
                    # operator for a 0.1-degree state; sub-cell placement is
                    # stage B's job.
                    operators.append(
                        BilinearObsOperator(
                            meso_grid, stations.lat[assimilated],
                            stations.lon[assimilated],
                        ).to(device)
                    )
                    # Transformed units: this likelihood sees tf.forward(mm).
                    # A swept value is the TOTAL error, so its representativeness
                    # folds to zero -- otherwise 0.25 floors the sweep and the
                    # interesting region below CPCv2's 0.269 is unreachable.
                    swept = arm in arm_sigma
                    gauge_R = build_R(
                        len(assimilated),
                        arm_sigma[arm] if swept else args.meso_gauge_sigma,
                        device=device,
                        representativeness=(
                            0.0 if swept else args.meso_gauge_representativeness
                        ),
                    )
                    variances.append(gauge_R)
                    perturb_variances.append(gauge_R)   # points: no inflation
                    # A station that did not report enters as NaN and the
                    # likelihood skips it, rather than assimilating a zero.
                    transformed = tf.forward(
                        np.nan_to_num(truth_assim, nan=0.0).astype(np.float32)
                    )
                    transformed[~np.isfinite(truth_assim)] = np.nan
                    values.append(transformed)

                if meso_obs in ("imerg", "both"):
                    if satellite_mm is not None:
                        field, sigma = satellite_mm, satellite_sigma
                        state_factor = 1
                        satellite_corr_cells = float(args.imerg_error_corr_cells)
                    else:
                        stream_name = arm_imerg_stream[arm]
                        stream = imerg_streams[stream_name]
                        field = stream["precipitation"][day_position]
                        sigma = np.maximum(
                            stream["error"][day_position], args.imerg_sigma_floor_mm
                        )
                        state_factor = int(stream["state_factor"])
                        satellite_corr_cells = float(stream["error_corr_cells"])
                    # Average in physical mm/day and transform afterwards. This
                    # is an identity for native 0.1-degree footprints and an
                    # exact 4x4 physical footprint operator for S04.
                    satellite_operator = PhysicalBlockAverageObsOperator(
                        state_factor, tf, valid=meso_ds.fixed_valid
                    ).to(device)
                    operators.append(satellite_operator)
                    satellite_keep = (
                        satellite_operator.valid_mask().detach().cpu().numpy().astype(bool)
                    )
                    # sigma is mm/day; this likelihood is in transformed units.
                    sigma_t = transformed_sigma(tf, field, sigma)
                    satellite_offset = sum(v.numel() for v in variances)
                    satellite_shape = field.shape
                    true_variance = (
                        sigma_t.reshape(-1) ** 2 + args.imerg_representativeness ** 2
                    ).astype(np.float32)
                    multiplier = arm_imerg_r.get(arm, args.imerg_r_multiplier)
                    variances.append(
                        torch.from_numpy(
                            (multiplier * true_variance).astype(np.float32)
                        ).to(device)
                    )
                    perturb_variances.append(
                        torch.from_numpy(true_variance).to(device)
                    )
                    effective = field.size / max(
                        1.0, satellite_corr_cells ** 2
                    )
                    print(
                        f"      satellite: {int((np.isfinite(field.reshape(-1)) & satellite_keep).sum())} of "
                        f"{field.size} footprints "
                        f"({'pseudo, from CHIRPS' if satellite_mm is not None else 'IMERG'})"
                        f"; {0.1 * state_factor:.1f}-degree support; error correlated over "
                        f"{satellite_corr_cells:g} footprint cells -> ~{effective:.0f} "
                        f"independent, R inflated x"
                        f"{arm_imerg_r.get(arm, args.imerg_r_multiplier):g}",
                        flush=True,
                    )
                    flat = field.reshape(-1)
                    transformed = tf.forward(flat.astype(np.float32))
                    transformed[~np.isfinite(flat) | ~satellite_keep] = np.nan
                    values.append(transformed)

                meso_H = (
                    operators[0] if len(operators) == 1
                    else CompositeObsOperator(operators).to(device)
                )
                meso_R = torch.cat(variances)
                stacked = np.concatenate(values)
                # Correlated perturbations over the satellite footprints only.
                # White noise on a 0.1-degree field averages out over any
                # neighbourhood, so every member ends up seeing effectively the
                # same satellite and the analysis loses its spread.
                corr_blocks = []
                if meso_obs in ("imerg", "both") and satellite_corr_cells > 0:
                    corr_blocks.append((
                        satellite_offset, satellite_shape[0], satellite_shape[1],
                        satellite_corr_cells,
                    ))
                # Draw from the TRUE error, weight by the inflated one.
                draws = perturb_observations(
                    stacked, torch.cat(perturb_variances), args.members,
                    seed=meso_index, corr_blocks=corr_blocks or None,
                )
                if meso_obs in ("imerg", "both") and arm.endswith(("imerg", "sim")):
                    # Jensen check, in mm: symmetric noise in transformed space
                    # is asymmetric in rainfall, so the perturbed observations
                    # can be systematically wetter than the field they came from.
                    block = slice(satellite_offset,
                                  satellite_offset + int(np.prod(satellite_shape)))
                    finite_obs = np.isfinite(stacked[block])
                    if finite_obs.any():
                        raw = float(np.nanmean(
                            tf.inverse(stacked[block][finite_obs].astype(np.float32))
                        ))
                        drawn = float(np.nanmean(
                            tf.inverse(draws[:, block][:, finite_obs].astype(np.float32))
                        ))
                        print(f"      perturbed obs mean {drawn:6.2f} mm vs "
                              f"{raw:6.2f} mm as read  (Jensen {drawn - raw:+.2f} mm)"
                              + ("   <- large; the draws are too wide"
                                 if abs(drawn - raw) > 0.1 * max(raw, 1.0) else ""),
                              flush=True)
                draws[:, ~np.isfinite(stacked)] = np.nan
                meso_y = torch.from_numpy(
                    draws[:, None].astype(np.float32)
                ).to(device)

            raw = meso_assimilate(
                meso_model, cond,
                (args.members, 1, window.meso_size, window.meso_size),
                device, H=meso_H, y=meso_y, R=meso_R, cfg=scfg, gcfg=arm_gcfg,
                flow=flow, mask=mask,
                to_precip=lambda x, b=base: meso_ds.residual.decode(x, b),
            )
            meso_mm = tf.inverse(
                meso_ds.residual.decode(raw, base.expand_as(raw))
            ).clamp_min(0.0)

            # -- crop stage A's output to stage B's coarse window --------------
            row, column = window.meso_local
            coarse_mm = meso_mm[
                :, :, row:row + window.coarse_size, column:column + window.coarse_size
            ].contiguous()

            # -- stage B ------------------------------------------------------
            observation = None
            if guide_fine:
                # Physical mm: this likelihood sees reconstructed rainfall.
                fine_R = build_R(
                    len(assimilated), args.fine_gauge_sigma_mm, device=device,
                    representativeness=args.representativeness_mm,
                )
                draws = perturb_observations(
                    np.nan_to_num(truth_assim, nan=0.0).astype(np.float32),
                    fine_R, args.members, seed=meso_index + 7,
                )
                draws[:, ~np.isfinite(truth_assim)] = np.nan
                observation = {
                    "H": assim_operator,
                    # Physical millimetres: reconstruct_from_amount returns mm,
                    # so the likelihood is evaluated in the gauge's own units.
                    "y": torch.from_numpy(
                        draws[:, None].astype(np.float32)
                    ).to(device),
                    "R": fine_R,
                }
            fine_mm = allocation_sample(
                alloc_model,
                coarse_mm,
                fine_cond.expand(args.members, -1, -1, -1),
                fine_valid.expand(args.members, -1, -1, -1),
                area.expand(args.members, -1, -1, -1),
                encoding,
                args.n_steps,
                generator,
                observation=observation,
                gcfg=arm_gcfg,
                flow=flow,
            )

            with torch.no_grad():
                at_stations = fine_operator(fine_mm)[:, 0].cpu().numpy()
                block_mean, _, _ = area_weighted_block_mean(
                    fine_mm.cpu(), area.cpu(), fine_valid.cpu(), 2, 0.0
                )
            conservation = float(
                np.abs(block_mean[:, 0].numpy() - coarse_mm[:, 0].cpu().numpy()).max()
            )
            if station_ensembles is not None:
                station_ensembles[arm][day_position] = at_stations

            # Field-level agreement with each INDEPENDENT gridded product, at
            # that product's own resolution.  The 0.05-degree comparison is
            # against CHIRPS (the training target); the 0.1-degree ones are
            # against the CPC field the model was conditioned on and, when
            # supplied, the satellite.  None is a conditioning input to the DA,
            # so all are external checks on the analysis.
            meso_mean = meso_mm[:, 0].mean(dim=0).cpu().numpy()
            fine_mean = fine_mm[:, 0].mean(dim=0).cpu().numpy()
            meso_valid = np.asarray(meso_ds.fixed_valid).astype(bool)
            fine_keep = sub["fine_valid"][0].numpy().astype(bool)
            if map_means is not None:
                map_means[arm][day_position] = np.where(
                    fine_keep, fine_mean, np.nan
                ).astype(np.float32)
                map_valid[day_position] = fine_keep
            pattern = {
                "chirps_0p05": field_pattern_r(
                    fine_mean, fine_truth[0, 0].cpu().numpy(), fine_keep
                ),
                "cpc_0p1": field_pattern_r(
                    meso_mean, item["base_mm"][0].numpy(), meso_valid
                ),
            }
            if imerg is not None:
                satellite_field = imerg["precipitation"][day_position]
                pattern["imerg_0p1"] = field_pattern_r(
                    meso_mean, satellite_field, meso_valid
                )
                if arm == "background":
                    # IMERG against the CPC field the model was conditioned on,
                    # both at 0.1 degrees.  Two independent products of the same
                    # day should agree well; a low value here means one of them
                    # is describing a different day, and every satellite arm
                    # below would then be assimilating the wrong rainfall.
                    agreement = field_pattern_r(
                        satellite_field, item["base_mm"][0].numpy(), meso_valid
                    )
                    print(f"      imerg vs cpc (same day, both 0.1 deg): "
                          f"r = {agreement:+.3f}"
                          + ("   <- suspiciously low; check the day alignment"
                             if np.isfinite(agreement) and agreement < 0.3 else ""),
                          flush=True)
            if satellite_mm is not None:
                pattern["pseudo_sat_0p1"] = field_pattern_r(
                    meso_mean, satellite_mm, meso_valid
                )

            entry = {
                "date": date,
                "pattern_r": pattern,
                "withheld": score_stations(at_stations[:, withheld], truth_withheld),
                "assimilated": score_stations(
                    at_stations[:, assimilated], truth_assim
                ),
                "conservation_max_mm": conservation,
                "domain_mean_mm": float(
                    fine_mm[:, 0].mean(dim=0).cpu().numpy()[
                        sub["fine_valid"][0].numpy().astype(bool)
                    ].mean()
                ),
            }
            results["arms"][arm]["days"].append(entry)
            if not args.no_plots:
                panels.setdefault(date, {})[arm] = {
                    "mean": fine_mm[:, 0].mean(dim=0).cpu().numpy(),
                    "member": fine_mm[0, 0].cpu().numpy(),
                    "at_stations": at_stations,
                }
            w = entry["withheld"]
            print(
                f"      withheld CRPS {w.get('crps_mm', float('nan')):.3f} mm  "
                f"MAE {w.get('mae_mm', float('nan')):.3f}  "
                f"bias {w.get('bias_mm', float('nan')):+.3f}  "
                f"conservation {conservation:.2e} mm",
                flush=True,
            )

    for arm in arms:
        entries = results["arms"][arm]["days"]
        results["arms"][arm]["mean"] = {
            key: float(np.nanmean([e["withheld"].get(key, np.nan) for e in entries]))
            for key in ("crps_mm", "mae_mm", "bias_mm", "rmse_mm", "spread_mm",
                    "spread_skill")
        }
        keys = sorted({k for e in entries for k in e.get("pattern_r", {})})
        results["arms"][arm]["pattern_r"] = {
            key: float(np.nanmean([e["pattern_r"].get(key, np.nan) for e in entries]))
            for key in keys
        }

    # The locked split is always written.  It is both a short audit trail and
    # the hand-off contract for a matched CPCv2 run.
    holdout_path = out_dir / "v7_withheld_station_ids.txt"
    holdout_path.write_text("\n".join(str(value) for value in stations.ids[withheld]) + "\n")
    results["stations"]["withheld_ids_file"] = str(holdout_path)

    if args.station_dump:
        dump_path = Path(args.station_dump)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            dump_path,
            times=gauge_times.astype("datetime64[D]").astype(str),
            model_times=day_times.astype("datetime64[D]").astype(str),
            station_ids=np.asarray(stations.ids, dtype=str),
            station_lat=stations.lat,
            station_lon=stations.lon,
            eval_idx=withheld,
            assim_idx=assimilated,
            observed_mm=observed_at_stations,
            arm_names=np.asarray(arms, dtype=str),
            **{f"station_{arm}": values for arm, values in station_ensembles.items()},
        )
        results["station_dump"] = str(dump_path)

    if args.map_dump:
        map_path = Path(args.map_dump)
        map_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            map_path,
            times=gauge_times.astype("datetime64[D]").astype(str),
            model_times=day_times.astype("datetime64[D]").astype(str),
            grid_lat=fine_grid.lat,
            grid_lon=fine_grid.lon,
            valid=map_valid,
            arm_names=np.asarray(arms, dtype=str),
            **{f"meanfield_{arm}": values for arm, values in map_means.items()},
        )
        results["map_dump"] = str(map_path)

    # Named for the experiment it actually is.  A file called ..._osse.json
    # holding real-data results is a trap for whoever opens it in a month.
    path = out_dir / f"v7_two_stage_{args.observations}.json"
    path.write_text(json.dumps(results, indent=2))
    if not args.no_plots:
        # Non-fatal by design: a broken figure must not cost the numbers, which
        # are already on disk by this point.
        try:
            make_figures(panels, truth_fields, results, stations, assimilated,
                         withheld, window, arms, out_dir,
                         observations=args.observations)
        except Exception as error:  # pragma: no cover - defensive
            import traceback

            print(f"[plots] SKIPPED: {error!r}", flush=True)
            traceback.print_exc()
    report(results, arms)
    print(f"\nwrote {path}", flush=True)


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------


def make_figures(panels, truth_fields, results, stations, assimilated, withheld,
                 window, arms, out_dir: Path, observations: str = "osse") -> None:
    """Maps, increments and station scatters -- one figure set per day.

    The increment column is the point of this.  A CRPS table says an arm helped
    or did not; only the increment map says WHERE the analysis moved, and for a
    conserving stage that is the difference between "the correction is small"
    and "the correction cannot leave its own block".
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # In OSSE the gridded CHIRPS field IS the verification truth.  With real
    # gauges it is only a reference field -- the scores are against BMD reports,
    # which exist at 38 points and nowhere else.  Calling it "truth" in that
    # case would misdescribe every number on the figure.
    osse = observations == "osse"
    field_label = "CHIRPS truth (OSSE)" if osse else "CHIRPS (reference field)"
    station_label = (
        "CHIRPS at station (mm/day)" if osse else "BMD gauge report (mm/day)"
    )
    verified_against = (
        "CHIRPS at withheld stations" if osse else "withheld BMD gauge reports"
    )

    grid = window.fine_grid()
    extent = (
        grid.lon_min, grid.lon_min + grid.nlon * grid.res,
        grid.lat_min, grid.lat_min + grid.nlat * grid.res,
    )
    a_lat, a_lon = stations.lat[assimilated], stations.lon[assimilated]
    w_lat, w_lon = stations.lat[withheld], stations.lon[withheld]

    for date, by_arm in panels.items():
        truth = truth_fields[date]
        keep = truth["valid"]
        vmax = float(np.nanpercentile(truth["field"][keep], 99.0)) or 1.0
        reference = by_arm.get("background", {}).get("mean")

        rows = len(arms)
        figure, axes = plt.subplots(
            rows, 3, figsize=(14.0, 4.2 * rows), squeeze=False
        )

        def stamp(axis, legend=False):
            axis.scatter(a_lon, a_lat, s=26, marker="o", facecolors="none",
                         edgecolors="black", linewidths=0.9,
                         label="assimilated" if legend else None)
            axis.scatter(w_lon, w_lat, s=44, marker="^", facecolors="none",
                         edgecolors="magenta", linewidths=1.3,
                         label="withheld" if legend else None)
            axis.set_xlim(extent[0], extent[1])
            axis.set_ylim(extent[2], extent[3])
            axis.set_xticks([])
            axis.set_yticks([])

        for row, arm in enumerate(arms):
            entry = by_arm[arm]
            shown = np.where(keep, entry["mean"], np.nan)
            image = axes[row][0].imshow(
                shown, origin="lower", extent=extent, cmap="turbo",
                vmin=0.0, vmax=vmax, aspect="auto",
            )
            axes[row][0].set_title(f"{arm}: ensemble mean", fontsize=10)
            stamp(axes[row][0], legend=(row == 0))
            if row == 0:
                axes[row][0].legend(loc="upper right", fontsize=7, framealpha=0.85)
            figure.colorbar(image, ax=axes[row][0], fraction=0.046, label="mm/day")

            # Increment against the background.  Symmetric scale, diverging map:
            # the sign is the whole story.
            if reference is not None and arm != "background":
                delta = np.where(keep, entry["mean"] - reference, np.nan)
                span = float(np.nanmax(np.abs(delta))) or 1.0
                image = axes[row][1].imshow(
                    delta, origin="lower", extent=extent, cmap="RdBu_r",
                    vmin=-span, vmax=span, aspect="auto",
                )
                axes[row][1].set_title(
                    f"increment vs background  (max |d| {span:.1f} mm)", fontsize=10
                )
                figure.colorbar(image, ax=axes[row][1], fraction=0.046, label="mm/day")
            else:
                image = axes[row][1].imshow(
                    np.where(keep, truth["field"], np.nan), origin="lower",
                    extent=extent, cmap="turbo", vmin=0.0, vmax=vmax, aspect="auto",
                )
                axes[row][1].set_title(field_label, fontsize=10)
                figure.colorbar(image, ax=axes[row][1], fraction=0.046, label="mm/day")
            stamp(axes[row][1])

            # Withheld stations: predicted against truth.  y=x is the target.
            axis = axes[row][2]
            observed = truth["at_stations"][withheld]
            predicted = entry["at_stations"][:, withheld]
            axis.plot([0, vmax], [0, vmax], color="0.6", lw=1.0, zorder=1)
            for member in predicted:
                axis.scatter(observed, member, s=8, alpha=0.25,
                             color="tab:blue", zorder=2)
            axis.scatter(observed, predicted.mean(axis=0), s=48, marker="^",
                         color="magenta", edgecolors="black", linewidths=0.6,
                         zorder=3, label="ensemble mean")
            score = next(
                d for d in results["arms"][arm]["days"] if d["date"] == date
            )["withheld"]
            axis.set_title(
                f"WITHHELD stations: CRPS "
                f"{score.get('crps_mm', float('nan')):.2f}  "
                f"bias {score.get('bias_mm', float('nan')):+.2f} mm",
                fontsize=10,
            )
            axis.set_xlabel(station_label)
            axis.set_ylabel("analysis (mm/day)")
            axis.grid(alpha=0.3)
            if row == 0:
                axis.legend(fontsize=7)

        figure.suptitle(
            f"V7 two stage, {observations.upper()} observations   {date}   "
            f"stage A epoch {results['checkpoints']['meso']['epoch']}, "
            f"stage B epoch {results['checkpoints']['allocation']['epoch']}\n"
            f"CRPS is scored against {verified_against} "
            f"({len(withheld)} of {len(stations)} stations, never assimilated)",
            fontsize=11,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.98))
        figure.savefig(out_dir / f"maps_{date}.png", dpi=110, bbox_inches="tight")
        plt.close(figure)
        print(f"[plots] wrote {out_dir / f'maps_{date}.png'}", flush=True)

    # The matrix, as a heatmap: one row per arm, CRPS beside the pattern
    # correlations.  Each column is normalised on its own scale because a CRPS
    # in mm and a correlation share no units; the colour says rank within the
    # column, and the printed number says the value.
    references = sorted({
        key for arm in arms for key in results["arms"][arm].get("pattern_r", {})
    })
    if references:
        columns = ["withheld CRPS\n(mm, lower better)"] + [
            f"pattern r\nvs {reference}" for reference in references
        ]
        table = np.full((len(arms), len(columns)), np.nan)
        for row, arm in enumerate(arms):
            table[row, 0] = results["arms"][arm]["mean"]["crps_mm"]
            for column, reference in enumerate(references, start=1):
                table[row, column] = results["arms"][arm].get(
                    "pattern_r", {}
                ).get(reference, np.nan)
        figure, axis = plt.subplots(
            figsize=(2.6 + 2.0 * len(columns), 1.0 + 0.55 * len(arms))
        )
        shown = np.empty_like(table)
        for column in range(table.shape[1]):
            values = table[:, column]
            finite = np.isfinite(values)
            if finite.sum() < 2 or np.nanmax(values) == np.nanmin(values):
                shown[:, column] = 0.5
                continue
            scaled = (values - np.nanmin(values)) / (
                np.nanmax(values) - np.nanmin(values)
            )
            # Lower CRPS is better; higher correlation is better.
            shown[:, column] = 1.0 - scaled if column == 0 else scaled
        axis.imshow(shown, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
        axis.set_xticks(range(len(columns)))
        axis.set_xticklabels(columns, fontsize=8)
        axis.set_yticks(range(len(arms)))
        axis.set_yticklabels(arms, fontsize=9)
        for row in range(table.shape[0]):
            for column in range(table.shape[1]):
                value = table[row, column]
                if np.isfinite(value):
                    axis.text(column, row, f"{value:.3f}", ha="center",
                              va="center", fontsize=9)
        axis.set_title(
            f"V7 {observations.upper()} observations   "
            f"CRPS at {len(withheld)} withheld gauges vs field-wide pattern "
            f"correlation\ncolour ranks within each column; green is better",
            fontsize=10,
        )
        figure.tight_layout()
        figure.savefig(out_dir / "matrix.png", dpi=110, bbox_inches="tight")
        plt.close(figure)
        print(f"[plots] wrote {out_dir / 'matrix.png'}", flush=True)

    # Summary bars across arms.
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 3.6))
    for axis, key, label in (
        (axes[0], "crps_mm", f"CRPS vs {verified_against} (mm/day)"),
        (axes[1], "mae_mm", "withheld MAE (mm/day)"),
        (axes[2], "bias_mm", "withheld bias (mm/day)"),
    ):
        values = [results["arms"][arm]["mean"][key] for arm in arms]
        colours = ["0.6" if arm == "background" else "tab:blue" for arm in arms]
        axis.bar(range(len(arms)), values, color=colours)
        axis.set_xticks(range(len(arms)))
        axis.set_xticklabels(arms, rotation=20, ha="right", fontsize=8)
        axis.set_title(label, fontsize=10)
        axis.grid(alpha=0.3, axis="y")
        if key == "bias_mm":
            axis.axhline(0.0, color="black", lw=0.8)
    figure.tight_layout()
    figure.savefig(out_dir / "summary.png", dpi=110)
    plt.close(figure)
    print(f"[plots] wrote {out_dir / 'summary.png'}", flush=True)


def _meso_grid(window):
    """Stage A's output window as a Grid, for the 0.1-degree observation operator."""
    from bdhires.grids import WIDE, Grid, at_resolution

    outer = at_resolution(WIDE, 0.1)
    return Grid(
        name="v7_meso_window",
        lon_min=outer.lon_min + window.meso_origin[1] * outer.res,
        lat_min=outer.lat_min + window.meso_origin[0] * outer.res,
        nlon=window.meso_size,
        nlat=window.meso_size,
        res=outer.res,
    )


def report(results: dict, arms: list[str]) -> None:
    print("\n" + "=" * 78)
    print(
        "WITHHELD-GAUGE SCORES  "
        + ("(OSSE: pseudo-gauges read CHIRPS)"
           if results.get("osse", True)
           else "(REAL: actual BMD reports, assimilated and verified)")
    )
    print("=" * 78)
    print(f"{'arm':<16} {'CRPS':>8} {'MAE':>8} {'bias':>8} {'RMSE':>8} "
          f"{'spread':>8} {'sprd/rmse':>10}")
    for arm in arms:
        m = results["arms"][arm]["mean"]
        ratio = m.get("spread_skill", float("nan"))
        flag = "" if 0.8 <= ratio <= 1.25 else ("  over" if ratio > 1.25 else "  under")
        print(f"{arm:<16} {m['crps_mm']:8.3f} {m['mae_mm']:8.3f} "
              f"{m['bias_mm']:+8.3f} {m['rmse_mm']:8.3f} {m['spread_mm']:8.3f} "
              f"{ratio:10.2f}{flag}")
    print("  spread/rmse ~1 is calibrated; >1 over-dispersed, <1 over-confident.")

    # The matrix: point verification beside field verification.  Eleven withheld
    # stations cannot tell you whether the analysis put rain in the right places;
    # the pattern correlations against the independent gridded products can, and
    # an arm that wins on CRPS while losing pattern correlation is fitting its
    # gauges by distorting the field.
    references = sorted({
        key for arm in arms for key in results["arms"][arm].get("pattern_r", {})
    })
    if references:
        print("\n" + "=" * 78)
        print("MATRIX: withheld-gauge CRPS beside spatial pattern correlation")
        print("=" * 78)
        header = f"{'arm':<16} {'CRPS':>8}"
        for reference in references:
            header += f" {reference:>14}"
        print(header)
        for arm in arms:
            row = f"{arm:<16} {results['arms'][arm]['mean']['crps_mm']:8.3f}"
            for reference in references:
                row += (
                    f" {results['arms'][arm].get('pattern_r', {}).get(reference, float('nan')):14.3f}"
                )
            print(row)
        print("  CRPS is verified at 11 withheld gauge points; the correlations")
        print("  cover the whole field against products the DA never saw.")

    if "background" not in arms:
        return
    reference = results["arms"]["background"]["mean"]["crps_mm"]
    print("\nCRPS change against the background (negative = DA helped):")
    for arm in arms:
        if arm == "background":
            continue
        delta = results["arms"][arm]["mean"]["crps_mm"] - reference
        percent = 100.0 * delta / reference if reference else float("nan")
        print(f"  {arm:<11} {delta:+.3f} mm ({percent:+.1f}%)   {ARM_NOTES[arm]}")

    def sweep_table(axis_key: str, title: str, unit: str, reference=None) -> None:
        axis = results.get(axis_key) or {}
        members = [a for a in arms if a in axis]
        if not members:
            return
        print(f"\n{title} (withheld scores):")
        print(f"  {'arm':<26} {'value':>8} {'CRPS':>8} {'MAE':>8} {'RMSE':>8} {'bias':>8} "
              f"{'sp/rm':>7}")
        floor = min(results["arms"][a]["mean"]["crps_mm"] for a in members)
        for arm in sorted(members, key=lambda a: axis[a]):
            m = results["arms"][arm]["mean"]
            mark = "   <- best CRPS" if m["crps_mm"] == floor else ""
            if reference is not None and abs(axis[arm] - reference) < 1e-9:
                mark += "   (current default)"
            print(f"  {arm:<26} {axis[arm]:8.3g} {m['crps_mm']:8.3f} {m['mae_mm']:8.3f} "
                  f"{m['rmse_mm']:8.3f} {m['bias_mm']:+8.3f} "
                  f"{m['spread_skill']:7.2f}{mark}")
        for label, key in (("CRPS", "crps_mm"), ("RMSE", "rmse_mm"),
                           ("|bias|", "bias_mm"), ("calibration", "spread_skill")):
            if key == "bias_mm":
                pick = min(members, key=lambda a: abs(results["arms"][a]["mean"][key]))
            elif key == "spread_skill":
                pick = min(members,
                           key=lambda a: abs(results["arms"][a]["mean"][key] - 1.0))
            else:
                pick = min(members, key=lambda a: results["arms"][a]["mean"][key])
            print(f"    best by {label:<12} {pick} ({axis[pick]:g} {unit})")
        values = [axis[a] for a in members]
        best = min(members, key=lambda a: results["arms"][a]["mean"]["crps_mm"])
        if axis[best] in (min(values), max(values)):
            print("    The CRPS optimum sits at the EDGE of the swept range; the "
                  "true minimum is outside it.")
        print("    Treat small CRPS differences over this limited withheld sample "
              "as unresolved without paired uncertainty analysis.")

    sweep_table(
        "arm_imerg_r",
        "Satellite configurations -- R multiplier within each footprint scale",
        "x", reference=results.get("imerg_r_multiplier"),
    )

    sweep = {a: results["arms"][a]["mean"]["crps_mm"]
             for a in arms if a in results.get("arm_sigma", {})}
    if sweep:
        import math

        cpcv2 = math.sqrt(0.10**2 + 0.25**2)
        print("\nStage-A TOTAL gauge error sweep (withheld CRPS; lower is better):")
        floor = min(sweep.values())
        for arm in sorted(sweep, key=lambda a: results["arm_sigma"][a]):
            total = results["arm_sigma"][arm]
            marks = "   <- best" if sweep[arm] == floor else ""
            if abs(total - cpcv2) < 0.01:
                marks += "   (CPCv2's tuned 0.10+0.25)"
            mean = results["arms"][arm]["mean"]
            print(f"  total {total:5.3f}  CRPS {sweep[arm]:7.3f} mm"
                  f"  ({100 * (sweep[arm] - floor) / floor:+5.1f}%)"
                  f"  RMSE {mean['rmse_mm']:6.3f}  bias {mean['bias_mm']:+6.3f}"
                  f"  sprd/rmse {mean.get('spread_skill', float('nan')):4.2f}"
                  f"{marks}")
        best = min(sweep, key=sweep.get)
        values = list(results["arm_sigma"].values())
        if results["arm_sigma"][best] in (min(values), max(values)):
            print("  The optimum sits at the EDGE of the swept range, so the true "
                  "minimum is outside it -- widen the sweep before concluding.")
        print("  Treat small CRPS differences over this limited withheld sample as "
              "unresolved without paired uncertainty analysis.")
        # The caveat that matters more than the number.
        # Which metric wins is the real question, and they often disagree.
        for label, key, pick in (
            ("CRPS", "crps_mm", min),
            ("RMSE", "rmse_mm", min),
            ("|bias|", "bias_mm", min),
        ):
            chosen = pick(
                sweep,
                key=lambda a: abs(results["arms"][a]["mean"][key])
                if key == "bias_mm" else results["arms"][a]["mean"][key],
            )
            print(f"  best by {label:<7} total {results['arm_sigma'][chosen]:5.3f}")
        print("  When these disagree, prefer bias and RMSE. CRPS rewards SHARPNESS,")
        print("  so an over-dispersed analysis improves its CRPS merely by")
        print("  tightening -- which is exactly what shrinking R does -- and that")
        print("  is a spread-calibration problem being paid for with an")
        print("  observation error that is physically wrong.")
        print("  CAUTION: assimilated and withheld gauges are the SAME instrument")
        print("  network. Any bias BMD shares against the model is corrected at")
        print("  both, so driving this error down improves withheld-gauge CRPS")
        print("  partly by fitting a network-wide offset rather than by making the")
        print("  field more accurate. A total error near zero asserts a gauge is a")
        print("  perfect measurement of an 11 km cell average, which it is not.")

    if "da_fine" not in arms:
        return
    delta = results["arms"]["da_fine"]["mean"]["crps_mm"] - reference
    fraction = delta / reference if reference else float("nan")
    print("\nThe question this run exists to answer:")
    # THREE outcomes, not two.  A sub-percent change on a handful of withheld
    # stations is not a degradation and must not be reported as one: "neutral"
    # and "harmful" have completely different consequences for the design.
    if fraction < -0.02:
        print("  Stage-B gauge assimilation IMPROVED withheld gauges. The 11 km")
        print("  increment propagates where the 55 km one did not -- keep it.")
    elif fraction > 0.02:
        print("  Stage-B gauge assimilation DEGRADED withheld gauges, as V5's did.")
        print("  Quantisation to 11 km was not enough. The design doc's fallback")
        print("  applies: drop stage-B DA and downscale the stage-A analysis.")
    else:
        print(f"  Stage-B gauge assimilation is NEUTRAL at withheld gauges "
              f"({fraction * 100:+.1f}%). It did not hurt -- it did nothing.")
        print("  That is the expected signature of exact conservation, not a")
        print("  failure of the increment: reconstruction fixes each 0.1-degree")
        print("  block total, so a gauge can only move rain BETWEEN the four")
        print("  0.05-degree cells of its own 11 km block. With ~38 stations")
        print("  over Bangladesh the mean spacing is ~60 km, about six blocks,")
        print("  so no withheld station shares a block with an assimilated one")
        print("  and the increment provably cannot reach it.")
        print("  Check the ASSIMILATED score in the JSON: if that improved")
        print("  sharply while withheld did not, the mechanism is confirmed and")
        print("  stage-B DA is a cosmetic fit to its own gauges. It is then")
        print("  safe to keep (it costs nothing) but must not be claimed as")
        print("  skill, and the honest product is the stage-A analysis.")
    print("  This reading assumes the checkpoints are trained enough to be")
    print(f"  meaningful -- these are epoch "
          f"{results['checkpoints']['meso']['epoch']} (A) and "
          f"{results['checkpoints']['allocation']['epoch']} (B).")


if __name__ == "__main__":
    main()
