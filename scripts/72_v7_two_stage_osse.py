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

# (guide at 0.1 deg, guide at 0.05 deg).  Order matters only for the report.
ARMS: dict[str, tuple[bool, bool]] = {
    "background": (False, False),
    "da_meso": (True, False),
    "da_fine": (False, True),
    "da_both": (True, True),
}

ARM_NOTES = {
    "background": "no observations; the prior both analyses start from",
    "da_meso": "gauges at 0.1 deg only, then an unguided downscale",
    "da_fine": "unguided 0.1 deg, then gauges at 0.05 deg -- the 11 km question",
    "da_both": "gauges at both scales; the V7 product path",
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
        "best_val": float(checkpoint.get("best_val", float("nan"))),
        "weights": str(checkpoint.get("weights", "unknown")),
    }
    print(
        f"[{label}] epoch {info['epoch']}, best_val {info['best_val']:.5f}, "
        f"weights={info['weights']}  -> {frozen}",
        flush=True,
    )
    return frozen, info


def load_meso(frozen: Path, cond_channels: int, size: int, device):
    """Stage A: the CPCv2 UNet, rebuilt from the checkpoint's own config."""
    checkpoint = torch.load(frozen, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("cfg") or checkpoint.get("config")
    if cfg is None:
        raise SystemExit("stage A checkpoint carries no config; cannot rebuild the model")
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
    p.add_argument("--stations", default="data/stations/data_2020_2025/Stations.csv")
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
                   help="assumed observation error; small but nonzero keeps R invertible")
    p.add_argument("--representativeness-mm", type=float, default=0.0)
    p.add_argument("--min-coverage", type=float, default=0.8)
    p.add_argument("--guidance-gamma", type=float, default=1.0e-3)
    p.add_argument("--guidance-scale", type=float, default=1.0)
    p.add_argument("--guidance-clip", type=float, default=50.0)
    p.add_argument("--huber-delta", type=float, default=3.0)
    p.add_argument("--spread-cells", type=float, default=0.0,
                   help="Gaussian spreading of the 0.1-degree guidance gradient, in cells")
    p.add_argument("--seed", type=int, default=20220503)
    p.add_argument("--arms", default=",".join(ARMS),
                   help="comma-separated subset of " + ",".join(ARMS))
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

    device = torch.device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    window = bangladesh_window()
    print(window.describe(), flush=True)

    # ---- frozen checkpoints, before anything is sampled -------------------
    meso_frozen, meso_info = snapshot(
        args.meso_checkpoint, out_dir / "checkpoints", "meso"
    )
    alloc_frozen, alloc_info = snapshot(
        args.allocation_checkpoint, out_dir / "checkpoints", "allocation"
    )

    # ---- stage A dataset --------------------------------------------------
    stats = json.loads(Path(args.meso_stats).read_text())
    tf = PrecipTransform.from_dict(stats["precip_transform"])
    meso_ds = PrecipDataset(
        DatasetConfig(
            root=args.meso_archive,
            crop=window.meso_size,
            random_crop=False,
            crop_origin=window.meso_origin,
        ),
        tf,
        cond_mean=np.asarray(stats["cond_mean"], np.float32),
        cond_std=np.asarray(stats["cond_std"], np.float32),
        cond_transform=CondTransform.from_stats(stats),
        residual=ResidualSpec.from_stats(stats),
        climatology=load_climatology(args.meso_stats, stats),
    )
    meso_model, _ = load_meso(
        meso_frozen, meso_ds.total_cond_channels, window.meso_size, device
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
    stations, _ = load_stations(
        args.stations, day_times, grid=fine_grid, min_coverage=args.min_coverage
    )
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
        "osse": True,
        "osse_sigma_mm": args.osse_sigma_mm,
        "stations": {
            "total": int(len(stations)),
            "assimilated": [str(s) for s in stations.ids[assimilated]],
            "withheld": [str(s) for s in stations.ids[withheld]],
            "holdout_neighbor_km": args.holdout_neighbor_km,
        },
        "arms": {name: {"note": ARM_NOTES[name], "days": []} for name in arms},
    }

    for date, meso_index, sub_index in days:
        item = meso_ds[meso_index]
        cond = item["cond"][None].to(device)
        base = item["base"][None].to(device)
        sub = subgrid_ds[sub_index]
        fine_truth = sub["fine_mm"][None].to(device)
        fine_valid = sub["fine_valid"][None].to(device)
        area = sub["cell_area"][None].to(device)
        fine_cond = sub["fine_cond"][None].to(device)

        # Perfect observations: CHIRPS read at the station coordinates through
        # the same operator the analysis uses, so a pseudo-gauge and the
        # analysis see the identical quantity and no interpolation mismatch can
        # confound the result.
        with torch.no_grad():
            truth_at_stations = fine_operator(fine_truth)[0, 0].cpu().numpy()
        truth_assim = truth_at_stations[assimilated]
        truth_withheld = truth_at_stations[withheld]

        for arm in arms:
            guide_meso, guide_fine = ARMS[arm]
            generator = torch.Generator(device=device).manual_seed(
                args.seed + 1000 * meso_index
            )
            print(f"  {date}  {arm:<11s} "
                  f"(meso DA {'on ' if guide_meso else 'off'}, "
                  f"fine DA {'on ' if guide_fine else 'off'})...", flush=True)

            # -- stage A ------------------------------------------------------
            meso_H = meso_y = meso_R = None
            if guide_meso:
                # The gauge is a POINT measurement of 0.05-degree rainfall, but
                # at this stage the state is 0.1 degree, so it is read as the
                # value of the cell containing it.  That is the honest operator
                # for a 0.1-degree state; the sub-cell placement is stage B's job.
                meso_grid = _meso_grid(window)
                meso_H = BilinearObsOperator(
                    meso_grid, stations.lat[assimilated], stations.lon[assimilated]
                ).to(device)
                meso_R = build_R(
                    len(assimilated), args.osse_sigma_mm, device=device,
                    representativeness=args.representativeness_mm,
                )
                y_transformed = tf.forward(truth_assim.astype(np.float32))
                draws = perturb_observations(
                    y_transformed, meso_R, args.members, seed=meso_index
                )
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
                    len(assimilated), args.osse_sigma_mm, device=device,
                    representativeness=args.representativeness_mm,
                )
                draws = perturb_observations(
                    truth_assim.astype(np.float32), fine_R, args.members,
                    seed=meso_index + 7,
                )
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
    report(results, arms)
    print(f"\nwrote {path}", flush=True)


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

    if "da_fine" in arms:
        delta = results["arms"]["da_fine"]["mean"]["crps_mm"] - reference
        print("\nThe question this run exists to answer:")
        if delta < 0:
            print("  Stage-B gauge assimilation IMPROVED withheld gauges.  The 11 km")
            print("  increment propagates where the 55 km one did not -- V7's stage-B")
            print("  DA is worth keeping.")
        else:
            print("  Stage-B gauge assimilation DEGRADED withheld gauges, as V5's did.")
            print("  Quantisation to 11 km was not enough.  The fallback in the design")
            print("  doc applies: drop stage-B DA and take the product from the")
            print("  stage-A analysis downscaled by the emulator alone.")
        print("  Both readings assume the checkpoints are trained enough to be")
        print(f"  meaningful -- these are epoch "
              f"{results['checkpoints']['meso']['epoch']} (A) and "
              f"{results['checkpoints']['allocation']['epoch']} (B).")


if __name__ == "__main__":
    main()
