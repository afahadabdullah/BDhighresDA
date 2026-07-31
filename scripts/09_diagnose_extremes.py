#!/usr/bin/env python
"""Locate and attribute unphysical extremes in individual ensemble members.

At epoch 150 the v5 single-member panel showed blocky saturated patches in the
south-east corner -- 236 mm/day on a day whose observed maximum was 95.6, and
498 mm/day against an observed 284.4 -- with almost all the ensemble spread
concentrated in the same place.  That is invisible in the ensemble mean and in
every aggregate score, so it needs its own diagnostic.

Three hypotheses, which this script separates:

1. DOMAIN EDGE.  The network was trained on random 128x128 crops and is applied
   to a fixed crop whose eastern boundary is the domain edge.  Convolutions and
   attention behave differently where context runs out.  Signature: exceedance
   concentrated in the outermost few columns/rows, falling off sharply inland.

2. MASK BOUNDARY.  CHIRPS is land-only.  Masked cells are held at a constant
   fill for the whole trajectory and re-imposed at every sampler step, so the
   coastline is a hard discontinuity in the field the network sees.  Signature:
   exceedance concentrated within a few cells of the land-sea boundary,
   regardless of position in the domain.

3. RESIDUAL DIPOLE.  ERA5 misplaces rainfall over the Chittagong coast and the
   Meghalaya barrier.  Where the base field is wrong the residual must remove
   rain in one place and add it in another, which is harder to learn than the
   field itself.  Signature: exceedance tracking |ERA5 - CHIRPS| rather than any
   geometric feature.

These are not mutually exclusive -- the south-east corner is simultaneously the
domain edge, a coastline, and where ERA5 is worst, which is exactly why the eye
cannot separate them and a quantitative profile is needed.

Extremeness is measured against a PER-PIXEL physical ceiling: the largest daily
CHIRPS value ever observed at that pixel across a sample of training days.  A
member exceeding it is claiming something that has never happened there, which
is a stronger and more local statement than any domain-wide threshold.

    python scripts/09_diagnose_extremes.py \
        --ckpt runs/prior_h100_v5/best.pt --config configs/da.yaml \
        --days 40 --members 8
"""
from __future__ import annotations

import argparse
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

from bdhires.da import SamplerConfig  # noqa: E402
from bdhires.da.sampler import sample  # noqa: E402
from bdhires.data import DatasetConfig, PrecipDataset  # noqa: E402
from bdhires.grids import WIDE, crop_offsets, get_grid  # noqa: E402
from bdhires.models import RectifiedFlow, UNet, select_weights  # noqa: E402
from bdhires.transforms import (  # noqa: E402
    CondTransform,
    PrecipTransform,
    ResidualSpec,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/da.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--members", type=int, default=8)
    parser.add_argument("--days", type=int, default=40)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument(
        "--month",
        type=int,
        default=7,
        help="restrict to this month (0 = all); extremes are a monsoon problem",
    )
    parser.add_argument(
        "--climatology-days",
        type=int,
        default=1200,
        help="training days sampled to build the per-pixel CHIRPS ceiling",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out-figure", default="data/processed/extreme_diagnosis.png")
    parser.add_argument("--out-report", default="data/processed/extreme_diagnosis.json")
    return parser.parse_args()


def distance_to_false(mask: np.ndarray) -> np.ndarray:
    """Euclidean distance from each True cell to the nearest False cell."""
    try:
        from scipy.ndimage import distance_transform_edt

        return distance_transform_edt(mask).astype(np.float32)
    except ImportError:  # pragma: no cover - scipy is present on the cluster
        # Chebyshev fallback by iterative dilation; adequate for binned profiles.
        distance = np.zeros(mask.shape, np.float32)
        frontier = ~mask
        current = frontier.copy()
        step = 0
        while not current.all():
            step += 1
            grown = current.copy()
            grown[1:, :] |= current[:-1, :]
            grown[:-1, :] |= current[1:, :]
            grown[:, 1:] |= current[:, :-1]
            grown[:, :-1] |= current[:, 1:]
            distance[grown & ~current] = step
            current = grown
        return distance


def edge_distance(shape: tuple[int, int]) -> np.ndarray:
    """Cells to the nearest domain boundary.

    ``np.minimum.reduce`` cannot reduce a list of (H,1) and (1,W) arrays -- it
    tries to stack them first.  Chain pairwise minima so broadcasting applies.
    """
    rows = np.arange(shape[0])[:, None]
    cols = np.arange(shape[1])[None, :]
    nearest = np.minimum(
        np.minimum(rows, cols),
        np.minimum(shape[0] - 1 - rows, shape[1] - 1 - cols),
    )
    return np.broadcast_to(nearest, shape).astype(np.float32).copy()


def binned_profile(
    values: np.ndarray, covariate: np.ndarray, keep: np.ndarray, edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean of ``values`` within bins of ``covariate``."""
    centres, means, counts = [], [], []
    for low, high in zip(edges[:-1], edges[1:]):
        inside = keep & (covariate >= low) & (covariate < high)
        centres.append(0.5 * (low + high))
        counts.append(int(inside.sum()))
        means.append(float(values[inside].mean()) if inside.any() else np.nan)
    return np.array(centres), np.array(means), np.array(counts)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation without a scipy dependency."""
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denominator = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denominator) if denominator > 0 else float("nan")


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    stats = json.loads(Path(config["data"]["stats"]).read_text())
    transform = PrecipTransform.from_dict(stats["precip_transform"])
    residual = ResidualSpec.from_stats(stats)
    grid = get_grid(config["data"]["grid"])

    dataset = PrecipDataset(
        DatasetConfig(
            root=config["data"]["zarr"],
            crop=grid.nlon,
            random_crop=False,
            crop_origin=crop_offsets(WIDE, grid),
        ),
        transform,
        cond_mean=np.asarray(stats["cond_mean"], np.float32),
        cond_std=np.asarray(stats["cond_std"], np.float32),
        cond_transform=CondTransform.from_stats(stats),
        residual=residual,
    )
    valid = dataset.fixed_valid > 0
    slices = dataset.fixed_spatial_slices()

    checkpoint = torch.load(args.ckpt, map_location="cpu")
    training_config = checkpoint["cfg"]
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

    # -- per-pixel physical ceiling from the training period ------------------
    years = training_config["data"]["years"]["train"]
    times = dataset.time
    calendar_years = times.astype("datetime64[Y]").astype(int) + 1970
    train_index = np.where(
        (calendar_years >= years[0]) & (calendar_years <= years[1])
    )[0]
    rng = np.random.default_rng(0)
    sampled = np.sort(
        rng.choice(
            train_index,
            size=min(args.climatology_days, len(train_index)),
            replace=False,
        )
    )
    print(f"building the per-pixel CHIRPS ceiling from {len(sampled)} training days",
          flush=True)
    ceiling = np.zeros(grid.shape, np.float32)
    for index in sampled:
        field = np.asarray(dataset.z["target"][int(index)][slices], dtype=np.float32)
        np.fmax(ceiling, np.nan_to_num(field, nan=0.0), out=ceiling)
    ceiling = np.maximum(ceiling, 1.0)      # guard against all-dry pixels

    # -- evaluation days ------------------------------------------------------
    test_years = training_config["data"]["years"]["test"]
    start = np.datetime64(args.start or f"{test_years[0]}-01-01")
    end = np.datetime64(args.end or f"{test_years[1]}-12-31")
    eligible = np.where((times >= start) & (times <= end))[0]
    if args.month:
        months = times[eligible].astype("datetime64[M]").astype(int) % 12 + 1
        eligible = eligible[months == args.month]
    if not len(eligible):
        raise ValueError("no evaluation days match the requested window")
    chosen = eligible[
        np.linspace(0, len(eligible) - 1, min(args.days, len(eligible))).astype(int)
    ]
    print(f"sampling {len(chosen)} days x {args.members} members on {device}",
          flush=True)

    sampler_cfg = SamplerConfig(
        **config.get("background_sampler", config["sampler"])
    )
    sampler_cfg = replace(sampler_cfg, mask_fill=dataset.mask_fill)
    flow = RectifiedFlow()
    mask = torch.from_numpy(valid.astype(np.float32)[None, None]).to(device)

    exceed_count = np.zeros(grid.shape, np.float64)
    worst_ratio = np.zeros(grid.shape, np.float32)
    spread_sum = np.zeros(grid.shape, np.float64)
    era5_error_sum = np.zeros(grid.shape, np.float64)
    n_member_days = 0

    for position, index in enumerate(chosen):
        item = dataset[int(np.where(dataset.index == index)[0][0])]
        base = item["base"][None].to(device)
        with torch.inference_mode():
            generated = sample(
                model,
                item["cond"][None].to(device),
                (args.members, 1, grid.nlat, grid.nlon),
                device,
                cfg=replace(sampler_cfg, seed=args.seed + position),
                flow=flow,
                mask=mask,
                to_precip=lambda x, b=base: residual.decode(x, b),
            )
        members = transform.inverse(
            residual.decode(generated, base)[:, 0].float().cpu().numpy()
        )
        target = np.asarray(dataset.z["target"][int(index)][slices], dtype=np.float32)
        era5 = transform.inverse(item["base"][0].numpy())

        ratio = members / ceiling[None]
        exceed_count += (ratio > 1.0).sum(axis=0)
        np.fmax(worst_ratio, ratio.max(axis=0), out=worst_ratio)
        spread_sum += members.std(axis=0, ddof=1)
        era5_error_sum += np.abs(era5 - np.nan_to_num(target, nan=0.0))
        n_member_days += args.members
        if position % 10 == 0:
            print(f"  {position}/{len(chosen)}", flush=True)

    exceed_fraction = exceed_count / max(1, n_member_days)
    mean_spread = spread_sum / len(chosen)
    era5_error = era5_error_sum / len(chosen)

    # -- covariates -----------------------------------------------------------
    to_edge = edge_distance(grid.shape)
    to_ocean = distance_to_false(valid)
    elevation = dataset.static[0][slices]

    keep = valid & np.isfinite(exceed_fraction)
    flat = {
        "exceedance": exceed_fraction[keep],
        "edge_distance": to_edge[keep],
        "ocean_distance": to_ocean[keep],
        "elevation": elevation[keep],
        "era5_error": era5_error[keep],
        "spread": mean_spread[keep],
    }

    correlations = {
        name: spearman(flat["exceedance"], flat[name])
        for name in ("edge_distance", "ocean_distance", "elevation", "era5_error")
    }

    # Crisp attribution numbers: where do the exceedances actually live?
    total = float(exceed_count[keep].sum())
    def share(selector: np.ndarray) -> float:
        return float(exceed_count[keep & selector].sum() / total) if total else 0.0

    def area(selector: np.ndarray) -> float:
        return float((keep & selector).sum() / keep.sum())

    regions = {
        "within 4 cells of the domain edge": to_edge <= 4,
        "within 4 cells of the coastline": to_ocean <= 4,
        "top decile of |ERA5 - CHIRPS|": era5_error
        >= np.quantile(era5_error[keep], 0.9),
        "elevation above the 90th percentile": elevation
        >= np.quantile(elevation[keep], 0.9),
    }
    attribution = {
        name: {
            "share_of_exceedances": share(selector),
            "share_of_area": area(selector),
            "enrichment": (
                share(selector) / area(selector) if area(selector) > 0 else float("nan")
            ),
        }
        for name, selector in regions.items()
    }

    report = {
        "checkpoint": str(args.ckpt),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "days": int(len(chosen)),
        "members": int(args.members),
        "month": args.month or "all",
        "n_member_days": int(n_member_days),
        "ceiling": "per-pixel CHIRPS maximum over sampled training days",
        "climatology_days": int(len(sampled)),
        "exceedance_rate_overall": float(exceed_fraction[keep].mean()),
        "worst_ratio_to_ceiling": float(worst_ratio[keep].max()),
        "spearman_with_exceedance": correlations,
        "attribution": attribution,
    }
    Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_report).write_text(json.dumps(report, indent=2) + "\n")

    print("\n=== exceedance attribution ===")
    print(f"overall exceedance rate: {report['exceedance_rate_overall']:.4%}")
    print(f"worst member / ceiling : {report['worst_ratio_to_ceiling']:.2f}x")
    print("\nSpearman correlation with exceedance rate:")
    for name, value in correlations.items():
        print(f"  {name:<16s} {value:+.3f}")
    print("\nregion                                    exceedances   area   enrichment")
    for name, values in attribution.items():
        print(
            f"  {name:<38s} {values['share_of_exceedances']:6.1%} "
            f"{values['share_of_area']:6.1%} "
            f"{values['enrichment']:8.2f}x"
        )
    print(
        "\nEnrichment >> 1 means exceedances concentrate there beyond what its "
        "area alone would give.\nThe largest enrichment identifies the dominant "
        "mechanism."
    )

    # -- figure ---------------------------------------------------------------
    extent = [grid.lon_min, grid.lon_max, grid.lat_min, grid.lat_max]
    figure, axes = plt.subplots(2, 4, figsize=(22, 10), constrained_layout=True)

    maps = [
        (exceed_fraction * 100, "magma", "A.  Exceedance rate\n% of member-days above the local CHIRPS record", "%"),
        (worst_ratio, "magma", "B.  Worst member / local record\nratio, 1.0 = never exceeded", "x"),
        (mean_spread, "magma", "C.  Mean ensemble spread\nmm day$^{-1}$", "mm day$^{-1}$"),
        (era5_error, "viridis", "D.  Mean |ERA5 - CHIRPS|\nmm day$^{-1}$", "mm day$^{-1}$"),
    ]
    for axis, (values, cmap, title, unit) in zip(axes[0], maps):
        shown = np.where(valid, values, np.nan)
        colours = plt.get_cmap(cmap).copy()
        colours.set_bad("white")
        image = axis.imshow(shown, origin="lower", extent=extent, cmap=colours,
                            interpolation="nearest")
        axis.set_title(title, fontsize=10.5)
        axis.set_xlabel("Longitude (deg E)", fontsize=8)
        axis.set_ylabel("Latitude (deg N)", fontsize=8)
        axis.tick_params(labelsize=7)
        bar = figure.colorbar(image, ax=axis, shrink=0.85)
        bar.set_label(unit, fontsize=8)
        bar.ax.tick_params(labelsize=7)

    profiles = [
        ("edge_distance", "Distance to domain edge (cells)",
         "E.  Edge effect?\nfalling profile = yes", np.arange(0, 33, 2)),
        ("ocean_distance", "Distance to coastline (cells)",
         "F.  Mask boundary?\nfalling profile = yes", np.arange(0, 33, 2)),
        ("era5_error", "Mean |ERA5 - CHIRPS| (mm day$^{-1}$)",
         "G.  Residual dipole?\nrising profile = yes", None),
        ("elevation", "Elevation (scaled)",
         "H.  Orography?\nrising profile = yes", None),
    ]
    for axis, (name, xlabel, title, edges) in zip(axes[1], profiles):
        covariate = flat[name]
        if edges is None:
            edges = np.quantile(covariate, np.linspace(0, 1, 13))
            edges = np.unique(edges)
        centres, means, counts = binned_profile(
            exceed_fraction, {
                "edge_distance": to_edge, "ocean_distance": to_ocean,
                "era5_error": era5_error, "elevation": elevation,
            }[name], keep, edges,
        )
        axis.plot(centres, means * 100, marker="o", ms=4, color="#c1442e")
        axis.set_xlabel(xlabel, fontsize=9)
        axis.set_ylabel("Exceedance rate (%)", fontsize=9)
        axis.set_title(
            f"{title}   (Spearman {correlations[name]:+.2f})", fontsize=10.5
        )
        axis.grid(alpha=0.25)

    figure.suptitle(
        "BDhighresDA - where do unphysical member extremes come from?\n"
        f"{args.ckpt}"
        + (f" (epoch {checkpoint['epoch'] + 1})" if checkpoint.get("epoch") is not None else "")
        + f"   |   {len(chosen)} days x {args.members} members"
        + (f"   |   month {args.month}" if args.month else "")
        + "   |   ceiling = per-pixel CHIRPS record over "
        f"{len(sampled)} training days\n"
        "Top row: maps.  Bottom row: exceedance rate against each candidate "
        "explanation -- edge geometry, mask boundary, ERA5 error, orography",
        fontsize=13.5,
    )
    Path(args.out_figure).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out_figure, dpi=115)
    plt.close(figure)
    print(f"\nwrote {args.out_figure}")
    print(f"wrote {args.out_report}")


if __name__ == "__main__":
    main()
