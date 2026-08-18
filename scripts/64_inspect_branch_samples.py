#!/usr/bin/env python3
"""Look at what the coarse and allocation branches actually produce.

The joint model does not exist until both branches finish, but each branch is
independently checkable long before that -- and the two failure modes worth
catching early are branch-local:

  coarse       does p(m | c) put the right amount of rain in the right
               0.5-degree cells, and is the wet/dry gate calibrated?  The v4
               pilot's largest residual error was a dry bias that grew as
               observations were added, which points at the occurrence gate
               rather than the intensity head.

  allocation   given the TRUE coarse amounts, does p(z | m, c) split them into
               0.05-degree structure that beats simply spreading them smoothly?
               This is the ``oracle_flow`` arm of the claim ladder, and it is
               the only place the v5 smooth base can be seen working: the seam
               index reported here is the direct successor to the blocky
               artifact in the v4 pilot figures.

Both are scored against nulls rather than in isolation, because a plausible
looking rainfall map is not evidence of anything.  Allocation in particular is
compared against the smooth base with no allocation at all; if the model does
not beat that, it has learned nothing about subgrid structure and the extra
machinery is not earning its place.

Reads the model configuration from the checkpoints themselves, so it cannot be
pointed at a config that disagrees with the weights.

    python scripts/64_inspect_branch_samples.py \\
      --target-store data/processed/cpc_v3_subgrid/wide_cpc_v5.zarr \\
      --coarse-checkpoint runs/prior_h100_cpc_v3_subgrid_v5/coarse/best.pt \\
      --allocation-checkpoint runs/prior_h100_cpc_v3_subgrid_v5/allocation/best.pt \\
      --out-dir data/processed/branch_inspection

Either checkpoint may be omitted; the script reports whichever it is given.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bdhires.data import (  # noqa: E402
    SubgridDataset,
    SubgridDatasetConfig,
    area_weighted_block_mean,
    conservative_smooth_upsample,
    decode_coarse_amount,
    encoding_metadata,
    reconstruct_from_amount,
)
from bdhires.models import (  # noqa: E402
    AllocationFlow,
    CoarseHurdleFlow,
    select_weights,
)


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------


def heun_sample(velocity_fn, shape, n_steps: int, device, generator) -> torch.Tensor:
    """Integrate the rectified-flow ODE from noise at t=0 to the sample at t=1.

    Deliberately the same Heun scheme ``bdhires.da.hierarchical_sampler`` uses,
    with a linear time schedule, so a branch inspected here behaves the way it
    will inside the joint sampler.  The final step is Euler because the second
    evaluation would land at t=1 exactly, where the velocity is not defined.
    """
    state = torch.randn(shape, device=device, generator=generator)
    times = torch.linspace(0.0, 1.0, n_steps + 1, device=device)
    for index in range(n_steps):
        t0, t1 = float(times[index]), float(times[index + 1])
        step = t1 - t0
        with torch.no_grad():
            v0 = velocity_fn(state, torch.full((shape[0],), t0, device=device))
            euler = state + step * v0
            if index < n_steps - 1:
                v1 = velocity_fn(euler, torch.full((shape[0],), t1, device=device))
                state = state + 0.5 * step * (v0 + v1)
            else:
                state = euler
    return state


def load_branch(path: str, stage: str, dataset: SubgridDataset, device):
    """Rebuild a branch from its own checkpoint, refusing a mismatched archive."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("stage") != stage:
        raise ValueError(f"{path} is a {checkpoint.get('stage')!r} checkpoint, not {stage!r}")
    if encoding_metadata(dataset.encoding) != dict(checkpoint["subgrid_encoding"]):
        raise ValueError(
            f"{path} was trained against a different subgrid encoding than "
            "this archive; its samples would be decoded under the wrong contract"
        )
    config = checkpoint["config"]
    crop = int(config["data"]["crop"])
    factor = int(config["data"].get("factor", 10))
    coarse_channels = int(dataset.z["coarse_cond"].shape[1]) if "coarse_cond" in dataset.z else 0
    fine_channels = int(dataset.z["fine_cond"].shape[1]) if "fine_cond" in dataset.z else 0
    if stage == "coarse":
        model = CoarseHurdleFlow(
            coarse_channels, image_size=crop // factor, **config["model"]
        )
    else:
        model = AllocationFlow(fine_channels, image_size=crop, **config["model"])
    model.load_state_dict(select_weights(checkpoint))
    model.eval().to(device)
    return model, checkpoint, crop


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------


def pattern_correlation(a: np.ndarray, b: np.ndarray, keep: np.ndarray) -> float:
    x, y = a.reshape(-1)[keep.reshape(-1)], b.reshape(-1)[keep.reshape(-1)]
    if x.size < 10 or x.std() <= 0.0 or y.std() <= 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def crps(ensemble: np.ndarray, truth: np.ndarray, keep: np.ndarray) -> float:
    """Fair CRPS over an (M, H, W) ensemble against one truth field."""
    members = ensemble.shape[0]
    flat = ensemble.reshape(members, -1)[:, keep.reshape(-1)]
    observed = truth.reshape(-1)[keep.reshape(-1)]
    first = np.abs(flat - observed[None]).mean(axis=0)
    if members == 1:
        return float(first.mean())
    ordered = np.sort(flat, axis=0)
    rank = np.arange(1, members + 1)[:, None]
    spread = (ordered * (2.0 * rank - members - 1.0)).sum(axis=0) / (members * members)
    return float((first - spread).mean())


def seam_index(field: np.ndarray, valid: np.ndarray, factor: int) -> float:
    """Mean gradient across block edges over the mean gradient inside blocks.

    A block-constant base is flat inside a block, so every gradient it has is a
    seam and the index diverges.  A field with genuine subgrid structure has an
    index near one.
    """
    vertical = np.abs(np.diff(field, axis=1))
    horizontal = np.abs(np.diff(field, axis=0))
    vvalid = valid[:, 1:] & valid[:, :-1]
    hvalid = valid[1:] & valid[:-1]
    vseam = np.zeros(vertical.shape, bool)
    hseam = np.zeros(horizontal.shape, bool)
    vseam[:, factor - 1::factor] = True
    hseam[factor - 1::factor, :] = True
    edge = np.concatenate([vertical[vvalid & vseam], horizontal[hvalid & hseam]])
    interior = np.concatenate([vertical[vvalid & ~vseam], horizontal[hvalid & ~hseam]])
    if interior.size == 0 or float(interior.mean()) <= 0.0:
        return float("inf")
    return float(edge.mean() / interior.mean())


def within_block_anomaly(
    field: torch.Tensor, area: torch.Tensor, valid: torch.Tensor, factor: int
) -> np.ndarray:
    """Remove each block's own mean, leaving only what allocation controls.

    Full-field correlation against CHIRPS is dominated by the coarse amounts,
    which in the oracle arm are handed to the model for free.  Subtracting the
    block mean is what makes the number a statement about the allocation.
    """
    mean, _, _ = area_weighted_block_mean(field, area, valid, factor, 0.0)
    expanded = mean.repeat_interleave(factor, -2).repeat_interleave(factor, -1)
    return (field - expanded)[:, 0].numpy()


def field_summary(field: np.ndarray, keep: np.ndarray, wet_threshold: float) -> dict:
    values = field.reshape(-1)[keep.reshape(-1)]
    return {
        "mean_mm": float(values.mean()),
        "wet_fraction": float((values >= wet_threshold).mean()),
        "p99_mm": float(np.percentile(values, 99.0)),
        "max_mm": float(values.max()),
    }


# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-store", required=True)
    parser.add_argument("--coarse-checkpoint", default=None)
    parser.add_argument("--allocation-checkpoint", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--members", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dates", default=None,
        help="comma-separated YYYY-MM-DD; overrides --days",
    )
    parser.add_argument(
        "--crop-origin", default=None,
        help="r,c origin of the evaluated crop; defaults to the centred crop",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not args.coarse_checkpoint and not args.allocation_checkpoint:
        raise SystemExit("give at least one of --coarse-checkpoint / --allocation-checkpoint")

    device = torch.device(args.device)
    reference = args.coarse_checkpoint or args.allocation_checkpoint
    probe = torch.load(reference, map_location="cpu", weights_only=False)
    config = probe["config"]
    crop = int(config["data"]["crop"])
    factor = int(config["data"].get("factor", 10))
    years = config["data"]["years"]["val"]

    origin = None
    if args.crop_origin:
        row, column = (int(value) for value in args.crop_origin.split(","))
        origin = (row, column)

    dataset = SubgridDataset(
        SubgridDatasetConfig(
            root=args.target_store, crop=crop, random_crop=False,
            crop_origin=origin, years=(int(years[0]), int(years[1])),
            factor=factor, downsamplings=int(config["data"].get("downsamplings", 3)),
        )
    )
    encoding = dataset.encoding
    wet_threshold = float(encoding.wet_threshold_mm)

    # Validation days, so nothing here was seen in training.
    available = dataset.time[dataset.index].astype("datetime64[D]")
    if args.dates:
        wanted = [np.datetime64(value, "D") for value in args.dates.split(",")]
        positions = []
        for value in wanted:
            match = np.flatnonzero(available == value)
            if match.size == 0:
                raise SystemExit(f"{value} is not in the validation split {years}")
            positions.append(int(match[0]))
    else:
        # Spread across the split rather than taking the first N consecutive
        # days, which would all belong to one synoptic event.
        positions = np.linspace(0, len(available) - 1, args.days).round().astype(int).tolist()

    print(f"archive     : {args.target_store}")
    print(f"validation  : {years[0]}-{years[1]}  ({len(available)} days)")
    print(f"crop        : {crop} at {origin or 'centre'}   factor {factor}")
    print(f"device      : {device}   members {args.members}   steps {args.n_steps}")
    if device.type == "cpu":
        cost = args.members * args.n_steps * 2
        print(f"              WARNING: no GPU.  Each day costs ~{cost} forward passes")
        print("              per branch; the allocation U-Net is slow on CPU.  Submit")
        print("              this to a GPU node, or lower --members / --n-steps.")
    print(f"days        : {', '.join(str(available[p]) for p in positions)}")
    print()

    coarse_model = allocation_model = None
    if args.coarse_checkpoint:
        coarse_model, checkpoint, _ = load_branch(
            args.coarse_checkpoint, "coarse", dataset, device
        )
        print(f"coarse      : epoch {checkpoint['epoch'] + 1}, "
              f"best_val {checkpoint['best_val']:.5f}, weights={checkpoint['weights']}")
    if args.allocation_checkpoint:
        allocation_model, checkpoint, _ = load_branch(
            args.allocation_checkpoint, "allocation", dataset, device
        )
        print(f"allocation  : epoch {checkpoint['epoch'] + 1}, "
              f"best_val {checkpoint['best_val']:.5f}, weights={checkpoint['weights']}")
    print()

    generator = torch.Generator(device=device).manual_seed(int(args.seed))
    results = {
        "store": args.target_store,
        "validation_years": [int(years[0]), int(years[1])],
        "members": int(args.members),
        "n_steps": int(args.n_steps),
        "days": [],
    }
    panels = []

    for position in positions:
        item = dataset[position]
        date = str(dataset.time[dataset.index[position]].astype("datetime64[D]"))
        fine_valid = item["fine_valid"][None].to(device)
        coarse_valid = item["coarse_valid"][None].to(device)
        area = item["cell_area"][None].to(device)
        coarse_cond = item["coarse_cond"][None].to(device)
        fine_cond = item["fine_cond"][None].to(device)
        coarse_truth = item["coarse_mm"][None].to(device)
        fine_truth = item["fine_mm"][None].to(device)
        keep = item["fine_valid"][0].numpy().astype(bool)
        coarse_keep = item["coarse_valid"][0].numpy().astype(bool)

        members = int(args.members)
        entry = {"date": date}
        panel = {"date": date}

        # ---- coarse branch ------------------------------------------------
        if coarse_model is not None:
            print(f"  {date}  sampling coarse ({members} members, "
                  f"{args.n_steps} steps)...", flush=True)
            condition = coarse_cond.expand(members, -1, -1, -1)

            def coarse_velocity(state, t):
                return coarse_model(state, t, condition)

            state = heun_sample(
                coarse_velocity,
                (members, 2, *coarse_truth.shape[-2:]),
                args.n_steps, device, generator,
            )
            amount = decode_coarse_amount(
                state, coarse_valid.expand(members, -1, -1, -1), encoding, hard=True
            )
            sampled = amount[:, 0].cpu().numpy()
            truth = coarse_truth[0, 0].cpu().numpy()
            entry["coarse"] = {
                "truth": field_summary(truth, coarse_keep, wet_threshold),
                "model": field_summary(sampled.mean(axis=0), coarse_keep, wet_threshold),
                "member_pattern_r": [
                    pattern_correlation(member, truth, coarse_keep) for member in sampled
                ],
                "ensemble_mean_pattern_r": pattern_correlation(
                    sampled.mean(axis=0), truth, coarse_keep
                ),
                "crps_mm": crps(sampled, truth, coarse_keep),
                "bias_mm": float(
                    sampled.mean(axis=0)[coarse_keep].mean() - truth[coarse_keep].mean()
                ),
            }
            panel["coarse_keep"] = coarse_keep
            panel["coarse_truth"] = truth
            panel["coarse_members"] = sampled

        # ---- allocation branch, given the true coarse amounts -------------
        if allocation_model is not None:
            print(f"  {date}  sampling allocation ({members} members, "
                  f"{args.n_steps} steps)...", flush=True)
            coarse_context = item["coarse_state"][None].to(device).expand(members, -1, -1, -1)
            condition = fine_cond.expand(members, -1, -1, -1)

            def allocation_velocity(state, t):
                return allocation_model(state, t, condition, coarse_context, 0.0)

            state = heun_sample(
                allocation_velocity,
                (members, 2, *fine_truth.shape[-2:]),
                args.n_steps, device, generator,
            )
            reconstruction = reconstruct_from_amount(
                coarse_truth.expand(members, -1, -1, -1),
                state,
                fine_valid.expand(members, -1, -1, -1),
                area.expand(members, -1, -1, -1),
                encoding, hard=True,
            )
            model_field = reconstruction.cpu()
            truth = fine_truth.cpu()

            # Nulls: no subgrid information at all, and the smooth base alone.
            block_null = coarse_truth.cpu().repeat_interleave(
                factor, -2
            ).repeat_interleave(factor, -1)
            smooth_null = conservative_smooth_upsample(
                coarse_truth.cpu(), area.cpu(), fine_valid.cpu(),
                factor, int(encoding.smooth_base_iterations),
            )

            anomaly_truth = within_block_anomaly(
                truth, area.cpu(), fine_valid.cpu(), factor
            )[0]
            anomaly_model = within_block_anomaly(
                model_field, area.cpu(), fine_valid.cpu(), factor
            )
            anomaly_smooth = within_block_anomaly(
                smooth_null, area.cpu(), fine_valid.cpu(), factor
            )[0]

            # Conservation, reported two ways.  A pure ratio is meaningless in a
            # dry block -- any rounding dust divided by zero looks catastrophic --
            # so the relative figure is taken only over blocks that actually
            # carry rain, and the absolute figure covers everything else.
            block_mean, _, _ = area_weighted_block_mean(
                model_field, area.cpu(), fine_valid.cpu(), factor, 0.0
            )
            error = (block_mean - coarse_truth.cpu())[:, 0].numpy()[:, coarse_keep]
            reference = coarse_truth.cpu()[0, 0].numpy()[coarse_keep]
            wet = reference >= wet_threshold
            conservation_abs = float(np.abs(error).max())
            conservation_rel = (
                float((np.abs(error[:, wet]) / reference[wet]).max()) if wet.any()
                else 0.0
            )

            numpy_model = model_field[:, 0].numpy()
            entry["allocation"] = {
                "truth": field_summary(truth[0, 0].numpy(), keep, wet_threshold),
                "model": field_summary(numpy_model.mean(axis=0), keep, wet_threshold),
                "conservation_max_abs_mm": conservation_abs,
                "conservation_max_relative_wet": conservation_rel,
                "full_field_r": [
                    pattern_correlation(member, truth[0, 0].numpy(), keep)
                    for member in numpy_model
                ],
                "within_block_r_model": [
                    pattern_correlation(member, anomaly_truth, keep)
                    for member in anomaly_model
                ],
                "within_block_r_smooth_null": pattern_correlation(
                    anomaly_smooth, anomaly_truth, keep
                ),
                "crps_mm_model": crps(numpy_model, truth[0, 0].numpy(), keep),
                "crps_mm_smooth_null": crps(
                    smooth_null[:, 0].numpy(), truth[0, 0].numpy(), keep
                ),
                "crps_mm_block_null": crps(
                    block_null[:, 0].numpy(), truth[0, 0].numpy(), keep
                ),
                "seam_index_model": seam_index(numpy_model[0], keep, factor),
                "seam_index_smooth_null": seam_index(smooth_null[0, 0].numpy(), keep, factor),
                "seam_index_block_null": seam_index(block_null[0, 0].numpy(), keep, factor),
                "seam_index_truth": seam_index(truth[0, 0].numpy(), keep, factor),
            }
            panel["fine_keep"] = keep
            panel["fine_truth"] = truth[0, 0].numpy()
            panel["fine_block_null"] = block_null[0, 0].numpy()
            panel["fine_smooth_null"] = smooth_null[0, 0].numpy()
            panel["fine_members"] = numpy_model

        results["days"].append(entry)
        panels.append(panel)
        report(entry)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "branch_inspection.json").write_text(
        json.dumps(results, indent=2, sort_keys=True)
    )
    print(f"\nwrote {out_dir / 'branch_inspection.json'}")
    make_figures(panels, out_dir)
    verdict(results)


def report(entry: dict) -> None:
    print(f"--- {entry['date']} " + "-" * 46, flush=True)
    if "coarse" in entry:
        block = entry["coarse"]
        print("  coarse 0.5deg amount")
        print(f"    pattern r      ensemble mean {block['ensemble_mean_pattern_r']:6.3f}   "
              f"members {' '.join(f'{r:.3f}' for r in block['member_pattern_r'])}")
        print(f"    CRPS           {block['crps_mm']:6.3f} mm/day        "
              f"bias {block['bias_mm']:+.3f}")
        print(f"    wet fraction   model {block['model']['wet_fraction']:.3f}   "
              f"truth {block['truth']['wet_fraction']:.3f}")
        print(f"    mean mm/day    model {block['model']['mean_mm']:6.3f}   "
              f"truth {block['truth']['mean_mm']:6.3f}")
    if "allocation" in entry:
        block = entry["allocation"]
        print("  allocation 0.05deg, given the true coarse amounts")
        print(f"    conservation   {block['conservation_max_abs_mm']:.2e} mm/day max, "
              f"{block['conservation_max_relative_wet']:.2e} relative where wet")
        print(f"    within-block r model  "
              f"{np.mean(block['within_block_r_model']):6.3f}   "
              f"smooth null {block['within_block_r_smooth_null']:6.3f}")
        print(f"    CRPS mm/day    model {block['crps_mm_model']:6.3f}   "
              f"smooth {block['crps_mm_smooth_null']:6.3f}   "
              f"block {block['crps_mm_block_null']:6.3f}")
        print(f"    seam index     model {block['seam_index_model']:6.3f}   "
              f"smooth {block['seam_index_smooth_null']:6.3f}   "
              f"truth {block['seam_index_truth']:6.3f}   "
              f"block {block['seam_index_block_null']:.3g}")
        print(f"    wet fraction   model {block['model']['wet_fraction']:.3f}   "
              f"truth {block['truth']['wet_fraction']:.3f}")
    print()


def verdict(results: dict) -> None:
    """State plainly whether each branch is doing its job."""
    print("=" * 62)
    days = results["days"]

    if "coarse" in days[0]:
        scores = [np.mean(day["coarse"]["member_pattern_r"]) for day in days]
        bias = [day["coarse"]["bias_mm"] for day in days]
        print(f"COARSE      mean member pattern r {np.mean(scores):.3f}, "
              f"mean bias {np.mean(bias):+.3f} mm/day")
        if np.mean(scores) < 0.2:
            print("            LOW.  The branch is not reproducing the 0.5-degree")
            print("            field its own conditioning should determine.  Check the")
            print("            conditioning channels before blaming the flow.")
        if np.mean(bias) < -0.5:
            print("            DRY BIAS, the v4 pilot's largest residual error.  It")
            print("            originates in the coarse occurrence gate, not in DA.")

    if "allocation" in days[0]:
        model = [np.mean(day["allocation"]["within_block_r_model"]) for day in days]
        null = [day["allocation"]["within_block_r_smooth_null"] for day in days]
        seam = [day["allocation"]["seam_index_model"] for day in days]
        truth_seam = [day["allocation"]["seam_index_truth"] for day in days]
        conservation = max(
            day["allocation"]["conservation_max_relative_wet"] for day in days
        )
        print(f"ALLOCATION  within-block r {np.mean(model):.3f} against a smooth-base "
              f"null of {np.mean(null):.3f}")
        print(f"            seam index {np.mean(seam):.2f}, CHIRPS itself "
              f"{np.mean(truth_seam):.2f}")
        print(f"            conservation holds to {conservation:.1e} relative")
        if np.mean(model) <= np.mean(null) + 0.01:
            print("            NO GAIN over the smooth base.  The flow is not adding")
            print("            subgrid information; whatever skill the reconstruction")
            print("            shows comes from the coarse amounts it was handed.")
        if np.mean(seam) > 2.0 * np.mean(truth_seam):
            print("            SEAM SURVIVES.  Block edges are still preferred over")
            print("            interior gradients; the blockiness fix is not landing.")
        if conservation > 1.0e-4:
            print("            CONSERVATION BROKEN.  This is the decoder's one hard")
            print("            guarantee; treat every other number here as void.")
    print("=" * 62)


def make_figures(panels: list[dict], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def draw(axis, field, title, vmax, mask):
        # Blank the cells with no data rather than drawing them as zero rain,
        # which would read as a real dry area.
        shown = np.where(mask, field, np.nan)
        image = axis.imshow(shown, origin="lower", cmap="turbo", vmin=0.0, vmax=vmax)
        axis.set_title(title, fontsize=8)
        axis.set_xticks([])
        axis.set_yticks([])
        return image

    if "coarse_truth" in panels[0]:
        columns = 2 + panels[0]["coarse_members"].shape[0]
        figure, axes = plt.subplots(
            len(panels), columns, figsize=(2.1 * columns, 2.3 * len(panels)), squeeze=False
        )
        for row, panel in enumerate(panels):
            mask = panel["coarse_keep"]
            vmax = max(float(np.nanpercentile(panel["coarse_truth"][mask], 99.5)), 1.0)
            draw(axes[row][0], panel["coarse_truth"], "CHIRPS @0.5deg", vmax, mask)
            axes[row][0].set_ylabel(panel["date"], fontsize=8)
            draw(axes[row][1], panel["coarse_members"].mean(axis=0),
                 "ensemble mean", vmax, mask)
            for column, member in enumerate(panel["coarse_members"]):
                draw(axes[row][2 + column], member, f"member {column + 1}", vmax, mask)
        figure.suptitle("Coarse branch: p(m | conditioning), decoded to mm/day", fontsize=10)
        figure.tight_layout()
        figure.savefig(out_dir / "branch_coarse.png", dpi=150)
        print(f"wrote {out_dir / 'branch_coarse.png'}")

    if "fine_truth" in panels[0]:
        columns = 3 + panels[0]["fine_members"].shape[0]
        figure, axes = plt.subplots(
            len(panels), columns, figsize=(2.1 * columns, 2.3 * len(panels)), squeeze=False
        )
        for row, panel in enumerate(panels):
            mask = panel["fine_keep"]
            vmax = max(float(np.nanpercentile(panel["fine_truth"][mask], 99.5)), 1.0)
            draw(axes[row][0], panel["fine_truth"], "CHIRPS @0.05deg", vmax, mask)
            axes[row][0].set_ylabel(panel["date"], fontsize=8)
            draw(axes[row][1], panel["fine_block_null"], "block null", vmax, mask)
            draw(axes[row][2], panel["fine_smooth_null"], "smooth base null", vmax, mask)
            for column, member in enumerate(panel["fine_members"]):
                draw(axes[row][3 + column], member, f"member {column + 1}", vmax, mask)
        figure.suptitle(
            "Allocation branch: p(z | true m, conditioning), reconstructed to mm/day",
            fontsize=10,
        )
        figure.tight_layout()
        figure.savefig(out_dir / "branch_allocation.png", dpi=150)
        print(f"wrote {out_dir / 'branch_allocation.png'}")


if __name__ == "__main__":
    main()
