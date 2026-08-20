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
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bdhires.bmd import neighbored_holdout  # noqa: E402
from bdhires.da import (  # noqa: E402
    BilinearObsOperator,
    BlockAverageObsOperator,
    CompositeObsOperator,
    GuidanceConfig,
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


def load_imerg_meso(path: str, days: np.ndarray, window, grid) -> dict:
    """Read BMD-aligned IMERG onto stage A's own cells, or refuse.

    ``observation_factor`` 2 means footprint centres on BD-at-0.1, which is
    exactly the grid stage A runs on when the window is anchored there.  This
    checks that rather than trusting it: a file on a different lattice would
    still have the right shape and would assimilate rainfall into the wrong
    places.
    """
    import xarray as xr

    with xr.open_dataset(path) as dataset:
        for name in ("precipitation", "randomError"):
            if name not in dataset.variables:
                raise SystemExit(f"{path} lacks the IMERG variable {name!r}")
            units = str(dataset[name].attrs.get("units", "")).lower().replace(" ", "")
            if units not in ("mm/day", "mmday-1", "mm day-1", "mm/d"):
                raise SystemExit(f"{path} {name} units are {units!r}; expected mm/day")
        factor = dataset.attrs.get("observation_factor")
        if factor is None or int(factor) != 2:
            raise SystemExit(
                f"{path} declares observation_factor {factor!r}; stage A needs 2, "
                f"which is the 0.1-degree BMD-aligned product"
            )
        end_hour = dataset.attrs.get("bmd_accumulation_end_hour_utc")
        if end_hour is None or int(end_hour) != 3:
            raise SystemExit(
                f"{path} is not on the BMD 03:00 UTC accumulation window "
                f"(declares {end_hour!r}); it would be a different day"
            )
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

    if precipitation.shape[1:] != (grid.nlat, grid.nlon):
        raise SystemExit(
            f"{path} is {precipitation.shape[1:]} but stage A's window is "
            f"{(grid.nlat, grid.nlon)}; the IMERG file and the analysis window "
            f"do not describe the same area"
        )
    if not np.allclose(lat, grid.lat, atol=1.0e-5) or not np.allclose(
        lon, grid.lon, atol=1.0e-5
    ):
        raise SystemExit(
            f"{path} footprint centres do not sit on stage A's 0.1-degree cells "
            f"(file lat {lat[0]:.3f}..{lat[-1]:.3f}, window {grid.lat[0]:.3f}.."
            f"{grid.lat[-1]:.3f}); assimilating it would displace rainfall"
        )
    print(
        f"[imerg] {path}: {precipitation.shape[0]} days on stage A's own cells "
        f"({grid.nlat}x{grid.nlon} @0.1 deg), identity operator",
        flush=True,
    )
    return {"precipitation": precipitation, "error": error}


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


def score_stations(ensemble_mm: np.ndarray, truth_mm: np.ndarray) -> dict:
    """Score an (M, S) ensemble of station values against (S,) truth."""
    finite = np.isfinite(truth_mm) & np.isfinite(ensemble_mm).all(axis=0)
    if not finite.any():
        return {"stations": 0}
    members = ensemble_mm[:, finite]
    observed = truth_mm[finite]
    mean = members.mean(axis=0)
    return {
        "stations": int(finite.sum()),
        "crps_mm": crps(members, observed),
        "mae_mm": float(np.abs(mean - observed).mean()),
        "bias_mm": float((mean - observed).mean()),
        "rmse_mm": float(np.sqrt(((mean - observed) ** 2).mean())),
        "spread_mm": float(members.std(axis=0).mean()),
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
    p.add_argument("--members", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=50)
    p.add_argument("--withhold", type=float, default=0.30,
                   help="fraction of stations withheld from assimilation")
    p.add_argument("--holdout-neighbor-km", type=float, default=75.0)
    p.add_argument("--holdout-max-gap-deg", type=float, default=200.0)
    p.add_argument("--osse-sigma-mm", type=float, default=0.5,
                   help="assumed gauge error in OSSE mode; small but nonzero keeps "
                        "R invertible")
    p.add_argument("--gauge-sigma-mm", type=float, default=None,
                   help="assumed gauge error in mm/day. Defaults to --osse-sigma-mm "
                        "under --observations osse and to 3.0 under real, matching "
                        "the v4 diagnostic. A REAL gauge assimilated at the OSSE "
                        "sigma is treated as near-perfect and the analysis chases "
                        "its noise")
    p.add_argument("--representativeness-mm", type=float, default=0.0)
    p.add_argument("--min-coverage", type=float, default=0.8)
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
    p.add_argument("--imerg-sigma-floor-mm", type=float, default=1.0,
                   help="floor on IMERG randomError, so a zero error is not infinite weight")
    p.add_argument("--imerg-representativeness-mm", type=float, default=0.0)
    p.add_argument("--arms", default=",".join(DEFAULT_ARMS),
                   help="comma-separated subset of " + ",".join(ARMS)
                        + " (IMERG arms require --imerg)")
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

    device = torch.device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    window = bangladesh_window()
    print(window.describe(), flush=True)
    # One sigma, resolved once and printed: an observation error that silently
    # differs between modes is the kind of thing that makes two runs look like a
    # scientific difference when it is a configuration difference.
    if args.gauge_sigma_mm is None:
        args.gauge_sigma_mm = (
            args.osse_sigma_mm if args.observations == "osse" else 3.0
        )
    print(
        f"observations: {args.observations.upper()}"
        + ("  (pseudo-gauges and pseudo-satellite read CHIRPS)"
           if args.observations == "osse"
           else "  (actual BMD reports, assimilated and verified)")
        + f"   gauge sigma {args.gauge_sigma_mm:g} mm/day",
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
    try:
        stations, gauge_mm = load_stations(
            args.stations, day_times, grid=fine_grid, min_coverage=args.min_coverage
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
    if args.imerg:
        imerg = load_imerg_meso(
            args.imerg, np.asarray([np.datetime64(d) for d, _, _ in days]),
            window, meso_grid,
        )

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
        "checkpoints": {"meso": meso_info, "allocation": alloc_info},
        "members": args.members,
        "n_steps": args.n_steps,
        "observations": args.observations,
        "osse": args.observations == "osse",
        "satellite": (
            "pseudo (CHIRPS + error model)" if args.osse_satellite
            else (args.imerg if args.imerg else "none")
        ),
        "gauge_sigma_mm": args.gauge_sigma_mm,
        "stations": {
            "total": int(len(stations)),
            "assimilated": [str(s) for s in stations.ids[assimilated]],
            "withheld": [str(s) for s in stations.ids[withheld]],
            "holdout_neighbor_km": args.holdout_neighbor_km,
        },
        "arms": {name: {"note": ARM_NOTES[name], "days": []} for name in arms},
    }

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

        truth_fields[date] = {
            "field": fine_truth[0, 0].cpu().numpy(),
            "valid": sub["fine_valid"][0].numpy().astype(bool),
            "at_stations": truth_at_stations,
        }

        for arm in arms:
            meso_obs, guide_fine = ARMS[arm]
            guide_meso = meso_obs != "none"
            generator = torch.Generator(device=device).manual_seed(
                args.seed + 1000 * meso_index
            )
            print(f"  {date}  {arm:<12s} "
                  f"(0.1 deg: {meso_obs:<6s}  0.05 deg: "
                  f"{'gauges' if guide_fine else 'none  '})...", flush=True)

            # -- stage A ------------------------------------------------------
            meso_H = meso_y = meso_R = None
            if guide_meso:
                # Both streams go through the SAME likelihood; neither is a
                # conditioning channel.  Stacking them into one operator and one
                # R is what makes "simultaneous" mean simultaneous rather than
                # two sequential updates that double-count the prior.
                operators, values, variances = [], [], []

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
                    gauge_R = build_R(
                        len(assimilated), args.gauge_sigma_mm, device=device,
                        representativeness=args.representativeness_mm,
                    )
                    variances.append(gauge_R)
                    # A station that did not report enters as NaN and the
                    # likelihood skips it, rather than assimilating a zero.
                    transformed = tf.forward(
                        np.nan_to_num(truth_assim, nan=0.0).astype(np.float32)
                    )
                    transformed[~np.isfinite(truth_assim)] = np.nan
                    values.append(transformed)

                if meso_obs in ("imerg", "both"):
                    # observation_factor 2 puts one footprint on each stage A
                    # cell, so the forward operator is the identity -- a
                    # factor-1 block average, reusing the audited operator
                    # rather than adding a second one that means the same thing.
                    operators.append(
                        BlockAverageObsOperator(
                            1, valid=meso_ds.fixed_valid
                        ).to(device)
                    )
                    if satellite_mm is not None:
                        field, sigma = satellite_mm, satellite_sigma
                    else:
                        field = imerg["precipitation"][day_position]
                        sigma = np.maximum(
                            imerg["error"][day_position], args.imerg_sigma_floor_mm
                        )
                    variances.append(
                        torch.from_numpy(
                            (sigma.reshape(-1) ** 2
                             + args.imerg_representativeness_mm ** 2).astype(np.float32)
                        ).to(device)
                    )
                    print(
                        f"      satellite: {int(np.isfinite(field).sum())} of "
                        f"{field.size} 0.1-degree footprints "
                        f"({'pseudo, from CHIRPS' if satellite_mm is not None else 'IMERG'})",
                        flush=True,
                    )
                    flat = field.reshape(-1)
                    transformed = tf.forward(flat.astype(np.float32))
                    transformed[~np.isfinite(flat)] = np.nan
                    values.append(transformed)

                meso_H = (
                    operators[0] if len(operators) == 1
                    else CompositeObsOperator(operators).to(device)
                )
                meso_R = torch.cat(variances)
                stacked = np.concatenate(values)
                draws = perturb_observations(
                    stacked, meso_R, args.members, seed=meso_index
                )
                draws[:, ~np.isfinite(stacked)] = np.nan
                meso_y = torch.from_numpy(
                    draws[:, None].astype(np.float32)
                ).to(device)

            raw = meso_assimilate(
                meso_model, cond,
                (args.members, 1, window.meso_size, window.meso_size),
                device, H=meso_H, y=meso_y, R=meso_R, cfg=scfg, gcfg=gcfg,
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
                fine_R = build_R(
                    len(assimilated), args.gauge_sigma_mm, device=device,
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
                gcfg=gcfg,
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

            entry = {
                "date": date,
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
            for key in ("crps_mm", "mae_mm", "bias_mm", "rmse_mm", "spread_mm")
        }

    path = out_dir / "v7_two_stage_osse.json"
    path.write_text(json.dumps(results, indent=2))
    if not args.no_plots:
        # Non-fatal by design: a broken figure must not cost the numbers, which
        # are already on disk by this point.
        try:
            make_figures(panels, truth_fields, results, stations, assimilated,
                         withheld, window, arms, out_dir)
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
                 window, arms, out_dir: Path) -> None:
    """Maps, increments and station scatters -- one figure set per day.

    The increment column is the point of this.  A CRPS table says an arm helped
    or did not; only the increment map says WHERE the analysis moved, and for a
    conserving stage that is the difference between "the correction is small"
    and "the correction cannot leave its own block".
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
                axes[row][1].set_title("CHIRPS truth (OSSE)", fontsize=10)
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
                f"withheld: CRPS {score.get('crps_mm', float('nan')):.2f}  "
                f"bias {score.get('bias_mm', float('nan')):+.2f} mm",
                fontsize=10,
            )
            axis.set_xlabel("CHIRPS at station (mm/day)")
            axis.set_ylabel("analysis (mm/day)")
            axis.grid(alpha=0.3)
            if row == 0:
                axis.legend(fontsize=7)

        figure.suptitle(
            f"V7 two-stage OSSE  {date}   stage A epoch "
            f"{results['checkpoints']['meso']['epoch']}, stage B epoch "
            f"{results['checkpoints']['allocation']['epoch']}",
            fontsize=12,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.98))
        figure.savefig(out_dir / f"maps_{date}.png", dpi=110, bbox_inches="tight")
        plt.close(figure)
        print(f"[plots] wrote {out_dir / f'maps_{date}.png'}", flush=True)

    # Summary bars across arms.
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 3.6))
    for axis, key, label in (
        (axes[0], "crps_mm", "withheld CRPS (mm/day)"),
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
    print("WITHHELD-GAUGE SCORES (OSSE, perfect observations)")
    print("=" * 78)
    print(f"{'arm':<12} {'CRPS':>8} {'MAE':>8} {'bias':>8} {'RMSE':>8} {'spread':>8}")
    for arm in arms:
        m = results["arms"][arm]["mean"]
        print(f"{arm:<12} {m['crps_mm']:8.3f} {m['mae_mm']:8.3f} "
              f"{m['bias_mm']:+8.3f} {m['rmse_mm']:8.3f} {m['spread_mm']:8.3f}")

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
