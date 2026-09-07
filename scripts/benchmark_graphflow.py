#!/usr/bin/env python3
"""Compare synthetic CPCv2 / G0 compute; no training-data or DA settings change."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train import build_coarse_clean_loss

from bdhires.da.guidance import GuidanceConfig, guidance_grad
from bdhires.da.observation import (
    CompositeObsOperator,
    PhysicalBilinearObsOperator,
    PhysicalBlockAverageObsOperator,
)
from bdhires.da.sampler import SamplerConfig, sample
from bdhires.grids import Grid
from bdhires.models import RectifiedFlow, build_model, flow_matching_loss, model_metadata
from bdhires.transforms import PrecipTransform, ResidualSpec
from bdhires.utils.dist import amp_dtype


def measure(fn, device, repeats, warmup, batch) -> dict:
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed.append((time.perf_counter() - start) * 1000)
    ms = float(np.median(elapsed))
    return dict(milliseconds=ms, samples_per_second=batch * 1000 / ms,
                measurements_ms=elapsed,
                peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None,
                peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 1024**3 if device.type == "cuda" else None)


def run_case(config, batch, args):
    device = torch.device(args.device)
    torch.manual_seed(123)
    model = build_model(in_channels=1, cond_channels=args.cond_channels, out_channels=1,
                        image_size=128, **config["model"]).to(device)
    # Nonzero velocity head exercises a nonzero network Jacobian even before training.
    with torch.no_grad():
        model.out_conv.weight.normal_(0, 0.01)
        if hasattr(model, "graph_processor"):
            model.graph_processor.output_projection.weight.normal_(0, 0.01)
    x = torch.randn(batch, 1, args.height, args.width, device=device)
    cond = torch.randn(batch, args.cond_channels, args.height, args.width, device=device)
    mask, base = torch.ones_like(x), torch.full_like(x, 2.0)
    tf, residual = PrecipTransform(kind="sqrt"), ResidualSpec(enabled=True, base="cpc_precip")
    def decode(field):
        return residual.decode(field, base)
    dataset = SimpleNamespace(transform=tf, residual=residual)
    train_batch = dict(base=base, target_mm=tf.inverse(decode(x)), base_mm=tf.inverse(base),
                       mask=mask, base_valid=mask)
    coarse_loss = build_coarse_clean_loss(config, dataset, train_batch, device)
    grid = Grid("synthetic", 88, 21, args.width, args.height, .05)
    rows = np.linspace(0, args.height - 1, min(6, args.height)).astype(int)
    cols = np.linspace(0, args.width - 1, min(6, args.width)).astype(int)
    yy, xx = np.meshgrid(rows, cols, indexing="ij")
    gauge = PhysicalBilinearObsOperator(grid, grid.lat[yy.ravel()], grid.lon[xx.ravel()], tf)
    imerg = PhysicalBlockAverageObsOperator(8, tf)
    H = CompositeObsOperator([gauge, imerg], component_spread_cells=[6., 0.]).to(device)
    y = H(decode(torch.zeros_like(x))).detach() + .25
    R = torch.cat([torch.full((yy.size,), .10**2 + .25**2),
                   torch.full((y.shape[-1] - yy.size,), .35**2 + .10**2)]).to(device)
    t = torch.full((batch,), .5, device=device)
    flow = RectifiedFlow()
    dtype = amp_dtype(device) if args.amp else torch.float32

    def autocast():
        return torch.autocast("cuda", dtype=dtype) if device.type == "cuda" and args.amp else nullcontext()

    def train_step():
        model.train()
        model.zero_grad(set_to_none=True)
        with autocast():
            loss = flow_matching_loss(model, x, cond, flow, cond_dropout=0,
                                      mask=mask, clean_loss_fn=coarse_loss)
        loss.backward()

    def forward():
        model.eval()
        with torch.no_grad(), autocast():
            model(x, t, cond)

    def trajectory():
        model.eval()
        with autocast():
            sample(model, cond, x.shape, device,
                   cfg=SamplerConfig(n_steps=args.sample_steps, seed=123))

    def guidance():
        model.eval()
        with autocast():
            guidance_grad(x, t, model, flow, cond, H, y, R,
                          GuidanceConfig(gamma=.01), mask=mask, to_precip=decode)

    result = dict(batch=batch, model_config=model_metadata(model), parameters=model.num_parameters,
                  precision=str(dtype), status="ok")
    for name, fn in (("train", train_step), ("forward", forward),
                     ("sample", trajectory), ("guidance", guidance)):
        model.zero_grad(set_to_none=True)
        result[name] = measure(fn, device, args.repeats, args.warmup, batch)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_h100_cpc_graphflow_g0.yaml")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batches", nargs="+", type=int, default=[1])
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--cond-channels", type=int, default=16, help="7 dynamic + 7 static + 2 seasonal")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--sample-steps", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tiny", action="store_true", help="explicit reduced CPU engineering case, not G0 performance")
    parser.add_argument("--alternative", action="store_true", help="add 192x6; primary 256x8 still runs")
    parser.add_argument("--out", default="runs/graphflow_g0_multimesh/benchmark.json")
    args = parser.parse_args()
    if min(args.batches + [args.height, args.width, args.repeats, args.sample_steps, args.threads]) < 1 or args.warmup < 0:
        parser.error("sizes/counts must be positive and warmup nonnegative")
    torch.set_num_threads(args.threads)
    config = yaml.safe_load(Path(args.config).read_text())
    if config["model"].get("architecture") != "graphflow_unet":
        parser.error("--config must describe graphflow_unet")
    if args.tiny:
        config["model"].update(base_channels=8, num_res_blocks=1, num_heads=1)
        config["model"]["graph"].update(hidden_dim=16, num_blocks=2)
    baseline = deepcopy(config)
    baseline["model"].pop("architecture")
    baseline["model"].pop("graph")
    candidates = [("unet", baseline), ("graphflow_g0_multimesh", config)]
    if args.alternative:
        alternative = deepcopy(config)
        alternative["model"]["graph"].update(hidden_dim=192, num_blocks=6)
        candidates.append(("graphflow_192x6", alternative))
    report = dict(platform=platform.platform(), torch=torch.__version__, device=args.device,
                  gpu=torch.cuda.get_device_name() if args.device == "cuda" else None,
                  synthetic=True, tiny=args.tiny, arguments=vars(args), cases=[])
    for batch in args.batches:
        for label, cfg in candidates:
            print(f"Benchmark {label}, batch={batch}, grid={args.height}x{args.width}", flush=True)
            try:
                result = run_case(cfg, batch, args)
            except torch.cuda.OutOfMemoryError as error:
                result = dict(batch=batch, status="cuda_oom", error=str(error))
            report["cases"].append(dict(architecture=label, **result))
            gc.collect()
            if args.device == "cuda":
                torch.cuda.empty_cache()
        base = next(r for r in report["cases"] if r["batch"] == batch and r["architecture"] == "unet")
        for r in report["cases"]:
            if r["batch"] == batch and r["status"] == base["status"] == "ok":
                ratio = r["guidance"]["milliseconds"] / base["guidance"]["milliseconds"]
                r["guided_ratio_to_unet"] = ratio
                r["within_1_5x_target"] = ratio <= 1.5
                print(f"  {r['architecture']}: {r['parameters']:,} parameters; guided ratio {ratio:.3f}", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(out, flush=True)


if __name__ == "__main__":
    main()
