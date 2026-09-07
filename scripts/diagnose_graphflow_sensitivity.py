#!/usr/bin/env python3
"""Compare point-innovation sensitivity maps on validation cases or synthetic smoke data."""
# ruff: noqa: E402 -- repository imports follow the standalone-script path setup

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
from scipy.ndimage import binary_erosion

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from graphflow_experiment import experiment_directory
from train import build_monitor

from bdhires.eval.graphflow import response_metrics, single_observation_response
from bdhires.grids import Grid, get_grid
from bdhires.models import RectifiedFlow, build_model, model_from_checkpoint
from bdhires.transforms import PrecipTransform, ResidualSpec


def load_case(args):
    torch.manual_seed(args.seed)
    if args.synthetic:
        size = 32
        kwargs = dict(in_channels=1, cond_channels=16, out_channels=1, image_size=size,
                      base_channels=8, channel_mult=(1, 2, 3, 4), num_res_blocks=1,
                      attn_resolutions=(4, 8), dropout=0., num_heads=1, multiscale_conditioning=True)
        unet = build_model(**kwargs)
        with torch.no_grad():
            unet.out_conv.weight.normal_(0, .03)
        graph = build_model(**kwargs, architecture="graphflow_unet",
                            graph=dict(hidden_dim=16, num_blocks=2))
        graph.load_state_dict(unet.state_dict(), strict=False)
        # Keep the graph residual exactly zero: these two maps should agree.
        grid = Grid("synthetic", 88, 21, size, size, .05)
        yy, xx = np.indices(grid.shape)
        terrain = np.exp(-((xx - 23)**2 + (yy - 24)**2) / 36)
        valid = yy > 4 + 2 * np.sin(xx / 5)
        mask = torch.tensor(valid.astype(np.float32))[None, None]
        x = torch.randn(1, 1, size, size) * mask
        cond = torch.randn(1, 16, size, size)
        cond[:, 7] = torch.tensor(terrain)
        cond[:, 9] = mask[0, 0]
        base = torch.full_like(x, 2.)
        tf, residual = PrecipTransform(kind="sqrt"), ResidualSpec(enabled=True, base="cpc_precip")
        models = {"cpcv2_control": unet, "graphflow_g0_multimesh": graph}
    else:
        if not args.unet_ckpt or not args.graphflow_ckpt or not args.date:
            raise ValueError("real diagnostic requires both checkpoints and a 2019/2020 --date")
        checkpoints = {
            "cpcv2_control": torch.load(args.unet_ckpt, map_location="cpu", weights_only=False),
            "graphflow_g0_multimesh": torch.load(args.graphflow_ckpt, map_location="cpu", weights_only=False),
        }
        cfg = checkpoints["cpcv2_control"]["cfg"]
        if cfg["data"] != checkpoints["graphflow_g0_multimesh"]["cfg"]["data"]:
            raise ValueError("checkpoint data contracts differ; sensitivity inputs must match")
        if any(ck.get("synthetic", False) for ck in checkpoints.values()):
            raise ValueError("synthetic smoke checkpoints are not trained priors")
        monitor = build_monitor(cfg, torch.device(args.device), Path(args.out))
        if monitor is None:
            raise ValueError("checkpoint has no compatible validation monitor")
        ds = monitor.ds
        matches = np.flatnonzero(ds.time[ds.index].astype("datetime64[D]") == np.datetime64(args.date))
        if len(matches) != 1:
            raise ValueError("--date must select exactly one day in the frozen validation split")
        item = ds[int(matches[0])]
        tf, residual = ds.transform, ds.residual
        grid = get_grid(cfg["data"].get("monitor_grid", "bd"))
        if grid.shape != tuple(item["x1"].shape[-2:]):
            raise ValueError("diagnostic requires the CPCv2 0.05-degree validation grid")
        cond, base, mask = (item[key][None] for key in ("cond", "base", "mask"))
        # Same RF interpolated held-out target/noise state for both Jacobians.
        x, _, _ = RectifiedFlow().interpolate(item["x1"][None], torch.tensor([args.time]))
        x = torch.where(mask.bool(), x, torch.full_like(x, residual.fill))
        slices = ds.fixed_spatial_slices()
        terrain = np.asarray(ds.static[0][slices])  # documented sqrt-scaled elevation channel
        models = {label: model_from_checkpoint(ck) for label, ck in checkpoints.items()}
        expected = ["UNet", "GraphFlowUNet"]
        if [type(m).__name__ for m in models.values()] != expected:
            raise ValueError("expected a CPCv2 UNet and an experimental GraphFlow checkpoint")
    return ({label: model.to(args.device).eval() for label, model in models.items()},
            dict(x=x.to(args.device), cond=cond.to(args.device), base=base.to(args.device),
                 mask=mask.to(args.device), grid=grid, transform=tf, residual=residual), terrain)


def choose_locations(mask, terrain):
    valid = mask.astype(bool)
    if not valid.any():
        raise ValueError("sensitivity case has no valid land")
    yy, xx = np.indices(valid.shape)
    centre = (yy - valid.shape[0] / 2)**2 + (xx - valid.shape[1] / 2)**2
    locations = {"interior": np.unravel_index(np.where(valid, centre, np.inf).argmin(), valid.shape),
                 "terrain": np.unravel_index(np.where(valid, terrain, -np.inf).argmax(), valid.shape)}
    coast = valid & ~binary_erosion(valid, border_value=1)
    if coast.any():
        locations["mask_coast"] = np.unravel_index(np.where(coast, centre, np.inf).argmin(), valid.shape)
    return locations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unet-ckpt")
    parser.add_argument("--graphflow-ckpt")
    parser.add_argument("--date", help="must belong to checkpoint validation years")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--time", type=float, default=.5)
    parser.add_argument("--innovation", type=float, default=.25, help="standardized transformed-precipitation units")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="runs/graphflow_g0_multimesh/sensitivity")
    args = parser.parse_args()
    torch.set_num_threads(2)
    out = experiment_directory(args.out)
    models, case, terrain = load_case(args)
    land = case["mask"][0, 0].cpu().numpy()
    report = dict(arguments=vars(args), synthetic=args.synthetic, locations={},
                  interpretation="One-step network-space guidance increment, not final analysis. "
                  "Distances are fine-grid cells. Mask boundary is a coast proxy. "
                  "Broader influence is not automatically better.")
    for site, (row, col) in choose_locations(land, terrain).items():
        records = {}
        arrays = dict(terrain=terrain, land_mask=land)
        fig, axes = plt.subplots(2, 4, figsize=(15, 7), layout="constrained")
        responses = {label: single_observation_response(model, **case, row=float(row), col=float(col),
                                                       time=args.time, innovation=args.innovation)
                     for label, model in models.items()}
        vmax = max(np.abs(response["spread_increment"]).max() for response in responses.values()) or 1.
        for index, (label, response) in enumerate(responses.items()):
            records[label] = {}
            for mode in ("raw", "spread"):
                gradient, increment = response[mode + "_gradient"], response[mode + "_increment"]
                metrics = response_metrics(increment, row, col, min(land.shape) / 4)
                metrics["total_gradient_norm"] = float(np.linalg.norm(gradient))
                records[label][mode] = metrics
                arrays[label + "_" + mode + "_increment"] = increment
                arrays[label + "_" + mode + "_gradient"] = gradient
            metrics = records[label]["spread"]
            ax = axes[index, 0]
            im = ax.imshow(response["spread_increment"], origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.scatter([col], [row], marker="x", c="black")
            if land.min() < land.max():
                ax.contour(land, levels=[.5], colors="black", linewidths=.6)
            ax.set_title(label + " increment")
            fig.colorbar(im, ax=ax, shrink=.75)
            ax = axes[index, 1]
            ax.imshow(terrain, origin="lower", cmap="terrain")
            ax.scatter([col], [row], c="red", marker="x")
            ax.set_title("Scaled terrain / station")
            ax = axes[index, 2]
            for mode in ("raw", "spread"):
                m = records[label][mode]
                ax.semilogy(m["radial_distance_cells"], np.maximum(m["radial_mean_abs"], 1e-15), label=mode)
            ax.set(xlabel="Distance (fine-grid cells)", ylabel="Mean |increment|")
            ax.legend()
            ax = axes[index, 3]
            ax.loglog(metrics["spectrum_frequency_cycles_per_cell"][1:],
                      np.maximum(metrics["spectrum_power"][1:], 1e-30))
            ax.set(xlabel="Cycles / fine-grid cell", ylabel="Increment power",
                   title=f"Anisotropy {metrics['anisotropy']:.3f}")
            ax.xaxis.set_major_locator(FixedLocator([.05, .1, .2, .4]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            ax.xaxis.set_minor_formatter(NullFormatter())
        fig.suptitle(f"{'SYNTHETIC zero-init equivalence' if args.synthetic else args.date}: {site}; "
                     f"t={args.time}, innovation={args.innovation}; not a skill test")
        fig.savefig(out / f"{site}.png", dpi=150)
        plt.close(fig)
        np.savez_compressed(out / f"{site}.npz", **arrays)
        report["locations"][site] = dict(row=int(row), col=int(col), responses=records)
    (out / "sensitivity.json").write_text(json.dumps(report, indent=2) + "\n")
    print(out, flush=True)


if __name__ == "__main__":
    main()
