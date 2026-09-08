#!/usr/bin/env python3
"""Isolated G0 entry points reusing the CPCv2 trainer, monitor and frozen DA."""
# ruff: noqa: E402 -- repository imports follow the standalone-script path setup

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from train import build_monitor, save_checkpoint

from bdhires.da.sampler import SamplerConfig, sample
from bdhires.models import (
    EMA,
    RectifiedFlow,
    build_model,
    flow_matching_loss,
    model_from_checkpoint,
    model_metadata,
)

G0_CONFIG = ROOT / "configs/train_h100_cpc_graphflow_g0.yaml"
EXPERIMENT = "graphflow_g0_multimesh"


def graphflow_da_screen_variants(sweep, frozen):
    """Broad one-fold DA screen around the frozen GraphFlow baseline.

    Each arm changes a named mechanism.  This catalogue is private to the
    GraphFlow process and never changes the historical CPCv2 method catalogue.
    """
    variants = [replace(
        frozen, name="gfs_joint_base",
        note="current frozen GraphFlow DA baseline",
    )]

    # Joint likelihood authority: both streams together, then each stream alone.
    for label, weight in (("025", 0.25), ("050", 0.50), ("075", 0.75), ("125", 1.25)):
        variants.append(replace(
            frozen, name=f"gfs_joint_w{label}",
            gauge_weight=weight, imerg_weight=weight,
            note=f"joint gauge and IMERG likelihood weight {weight:g}",
        ))
    for label, weight in (("050", 0.50), ("075", 0.75), ("125", 1.25)):
        variants.append(replace(
            frozen, name=f"gfs_joint_gw{label}", gauge_weight=weight,
            note=f"gauge likelihood weight {weight:g}; IMERG fixed",
        ))
    for label, weight in (("025", 0.25), ("050", 0.50), ("075", 0.75), ("125", 1.25)):
        variants.append(replace(
            frozen, name=f"gfs_joint_iw{label}", imerg_weight=weight,
            note=f"IMERG likelihood weight {weight:g}; gauges fixed",
        ))

    # Prior inflation and a small temperature/authority interaction grid.
    for label, temperature in (("105", 1.05), ("110", 1.10), ("115", 1.15),
                               ("125", 1.25), ("150", 1.50)):
        variants.append(replace(
            frozen, name=f"gfs_joint_t{label}", prior_temperature=temperature,
            note=f"prior temperature {temperature:g}; likelihood fixed",
        ))
    for tlabel, temperature in (("110", 1.10), ("125", 1.25), ("150", 1.50)):
        for wlabel, weight in (("050", 0.50), ("075", 0.75)):
            variants.append(replace(
                frozen, name=f"gfs_joint_t{tlabel}_w{wlabel}",
                prior_temperature=temperature,
                gauge_weight=weight, imerg_weight=weight,
                note=f"temperature {temperature:g}, both likelihood weights {weight:g}",
            ))

    # Larger gamma softens early-time guidance and can preserve posterior spread.
    for label, gamma in (("020", 2.0e-2), ("050", 5.0e-2), ("100", 1.0e-1)):
        variants.append(replace(
            frozen, name=f"gfs_joint_gamma{label}",
            gauge_guidance_gamma=gamma, imerg_guidance_gamma=gamma,
            note=f"both guidance gamma values {gamma:g}",
        ))
    variants.extend([
        replace(frozen, name="gfs_joint_gg020", gauge_guidance_gamma=2.0e-2,
                note="gauge gamma 0.02; IMERG gamma fixed"),
        replace(frozen, name="gfs_joint_ig020", imerg_guidance_gamma=2.0e-2,
                note="IMERG gamma 0.02; gauge gamma fixed"),
    ])

    # Gauge-footprint scale and robust likelihood.
    for label, spread in (("s0", 0.0), ("s3", 3.0), ("s9", 9.0), ("s12", 12.0)):
        variants.append(replace(
            frozen, name=f"gfs_joint_{label}",
            gauge_component_spread_cells=spread,
            note=f"gauge component gradient spread {spread:g} grid cells",
        ))
    variants.extend([
        replace(frozen, name="gfs_joint_huber3", huber_delta=3.0,
                note="robust joint likelihood with Huber delta 3"),
        replace(frozen, name="gfs_joint_huber5", huber_delta=5.0,
                note="robust joint likelihood with Huber delta 5"),
    ])

    # A genuinely different process: IMERG-guided flow followed by gauge EnSRF.
    for radius in (50.0, 100.0, 150.0, 200.0, 300.0):
        variants.append(replace(
            frozen, name=f"gfs_twostep_l{int(radius)}",
            algorithm="twostep_ensrf", gauge_component_spread_cells=None,
            ensrf_localization_km=radius,
            note=f"IMERG-guided flow then gauge EnSRF at {radius:g} km",
        ))
    for radius in (100.0, 150.0, 200.0):
        variants.append(replace(
            frozen, name=f"gfs_twostep_t110_l{int(radius)}",
            algorithm="twostep_ensrf", gauge_component_spread_cells=None,
            prior_temperature=1.10, ensrf_localization_km=radius,
            note=f"temperature 1.1, IMERG flow then gauge EnSRF at {radius:g} km",
        ))

    # Stream and covariance controls clarify which observation source helps.
    variants.extend([
        replace(frozen, name="gfs_gauges_guided", streams="gauges",
                gauge_component_spread_cells=None, guidance_spread_cells=6.0,
                note="gauges-only guided-flow control"),
        replace(frozen, name="gfs_imerg_guided", streams="imerg",
                gauge_component_spread_cells=None,
                note="IMERG-only guided-flow control"),
    ])
    for radius in (50.0, 100.0, 150.0, 200.0, 300.0):
        variants.append(replace(
            frozen, name=f"gfs_gauge_ensrf_l{int(radius)}",
            streams="gauges", algorithm="ensrf",
            gauge_component_spread_cells=None,
            ensrf_localization_km=radius,
            note=f"gauges-only EnSRF at {radius:g} km",
        ))
    return variants


def experiment_directory(path: str | Path) -> Path:
    """Keep the experiment outside every historical production directory."""
    path = Path(path)
    relative = path.resolve().relative_to((ROOT / "runs").resolve())
    if not relative.parts or "graphflow" not in relative.parts[0]:
        raise ValueError("experiment output must be runs/<name-containing-graphflow>/...")
    path.mkdir(parents=True, exist_ok=True)
    return path


def training_config(stage: str, baseline: bool = False) -> dict:
    cfg = yaml.safe_load(G0_CONFIG.read_text())
    if baseline:
        cfg["model"].pop("architecture")
        cfg["model"].pop("graph")
    # Separate matched 30-epoch controls, with the SAME optimization budget.
    # The shorter horizon changes the cosine schedule for BOTH screen arms.
    if stage == "screen":
        cfg["train"]["epochs"] = 30
    label = "cpcv2_graphflow_control" if baseline else "prior_h100_cpc_graphflow_g0"
    cfg["train"]["out_dir"] = f"runs/{label}" + ("_screen" if stage == "screen" else "")
    return cfg


def train(args):
    cfg = training_config(args.stage, args.baseline)
    out = experiment_directory(cfg["train"]["out_dir"])
    if list(out.glob("*.pt")) and args.resume is None:
        raise ValueError(f"checkpoints already exist in {out}; use --resume explicitly")
    config_path = out / "training_config.yaml"
    if config_path.exists() and yaml.safe_load(config_path.read_text()) != cfg:
        raise ValueError(f"existing experiment configuration differs: {config_path}")
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    command = [sys.executable, str(ROOT / "scripts/train.py"), "--config", str(config_path)]
    if args.resume:
        command += ["--resume", args.resume]
    print(" ".join(command), flush=True)
    if not args.prepare_only:
        subprocess.run(command, cwd=ROOT, check=True)


def validate(args):
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    out = experiment_directory(args.out)
    model = model_from_checkpoint(ck, device=args.device)
    monitor = build_monitor(ck["cfg"], torch.device(args.device), out)
    if monitor is None:
        raise ValueError("checkpoint validation monitor is disabled or grid incompatible")
    summary = monitor.run(model, None, ck.get("epoch", 0), ck.get("step", 0))
    if summary is None:
        raise RuntimeError("validation monitor failed; see preceding diagnostic")
    summary["checkpoint"] = str(Path(args.ckpt).resolve())
    summary["weights"] = ck.get("weights", "ema" if ck.get("ema") is not None else "model")
    (out / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def smoke(args):
    """Repeat a fixed synthetic RF mini-batch to check optimization and EMA I/O."""
    torch.set_num_threads(2)
    torch.manual_seed(20)
    cfg = training_config("screen")
    cfg["model"].update(base_channels=8, num_res_blocks=1, num_heads=1, dropout=0.0)
    cfg["model"]["graph"].update(hidden_dim=16, num_blocks=2)
    cfg["data"]["crop"] = 32
    cfg["train"]["out_dir"] = args.out
    out = experiment_directory(args.out)
    if (out / "synthetic.pt").exists():
        raise ValueError("synthetic.pt already exists; choose a fresh experimental output directory")
    model = build_model(in_channels=1, cond_channels=16, out_channels=1, image_size=32,
                        **cfg["model"]).to(args.device)
    x, cond = torch.randn(2, 1, 32, 32, device=args.device), torch.randn(2, 16, 32, 32, device=args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    ema, flow = EMA(model, .9), RectifiedFlow()
    losses = []
    for _ in range(args.steps):
        torch.manual_seed(314)  # fixed times/noise: this is an overfit engineering check
        optimizer.zero_grad(set_to_none=True)
        loss = flow_matching_loss(model, x, cond, flow, cond_dropout=0)
        loss.backward()
        if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError("non-finite synthetic training loss/gradient")
        optimizer.step()
        ema.update(model)
        losses.append(float(loss.detach()))
    if losses[-1] >= losses[0]:
        raise RuntimeError("fixed-batch smoke loss did not decrease")
    ck = dict(model=model.state_dict(), ema=ema.state_dict(), weights="ema",
              model_config=model_metadata(model), cfg=cfg, synthetic=True,
              experiment=EXPERIMENT, step=args.steps)
    save_checkpoint(ck, out / "synthetic.pt")
    restored = model_from_checkpoint(torch.load(out / "synthetic.pt", weights_only=False), device=args.device)
    generated = sample(restored, cond[:1], x.shape, args.device, cfg=SamplerConfig(n_steps=3, seed=41))
    if not torch.isfinite(generated).all():
        raise FloatingPointError("non-finite EMA validation sample")
    report = dict(synthetic=True, reduced_architecture=True, losses=losses,
                  checkpoint_reload=True, ema=True, validation_sampler_finite=True,
                  model_config=model_metadata(model))
    (out / "smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Synthetic loss {losses[0]:.6f} -> {losses[-1]:.6f}; EMA reload and RF sampler passed")


def frozen_da(args):
    """Reuse the frozen simultaneous method under NEW experimental method names.

    The imported catalogue is private to this process. Production source files,
    method definitions, schemas and artifacts are not edited or overwritten.
    """
    out = experiment_directory(args.out)
    for path, architecture in ((args.unet_ckpt, "unet"), (args.graphflow_ckpt, "graphflow_unet")):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        actual = ck.get("model_config", ck["cfg"]["model"]).get("architecture", "unet")
        if actual != architecture or ck.get("synthetic", False):
            raise ValueError(f"{path} must be a trained {architecture} checkpoint")
    a = torch.load(args.unet_ckpt, map_location="cpu", weights_only=False)["cfg"]
    b = torch.load(args.graphflow_ckpt, map_location="cpu", weights_only=False)["cfg"]
    for key in ("data", "validation"):
        if a[key] != b[key]:
            raise ValueError(f"paired checkpoints have different {key} contracts")
    spec = importlib.util.spec_from_file_location("_graphflow_frozen_sweep", ROOT / "scripts/28_simultaneous_method_sweep.py")
    sweep = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = sweep
    spec.loader.exec_module(sweep)
    frozen = next(v for v in sweep.V2_CONFIRMATORY if v.name == "v2_simul_s04_ig010")
    if args.variant_set == "screen" and args.model != "graphflow":
        raise ValueError("the broad DA screen is GraphFlow-only; use --model graphflow")
    runs = {
        "cpcv2": ("cpcv2_control", args.unet_ckpt),
        "graphflow": (EXPERIMENT, args.graphflow_ckpt),
    }
    selected_runs = runs.values() if args.model == "both" else (runs[args.model],)
    folds = range(5) if args.fold is None else (args.fold,)
    for label, ckpt in selected_runs:
        group = f"{label}_{args.variant_set}_da"
        sweep.GROUPS[group] = (
            graphflow_da_screen_variants(sweep, frozen)
            if args.variant_set == "screen"
            else [replace(frozen, name=group)]
        )
        for fold in folds:
            prefix = out / label / f"fold{fold}"
            prefix.parent.mkdir(exist_ok=True)
            if prefix.with_suffix(".npz").exists() or prefix.with_suffix(".json").exists():
                raise ValueError(f"refusing to overwrite experimental fold {prefix}")
            previous_argv = sys.argv
            sys.argv = [str(ROOT / "scripts/28_simultaneous_method_sweep.py"),
                        "--config", "configs/da.yaml", "--ckpt", ckpt,
                        "--stations", args.stations, "--imerg", args.imerg,
                        "--start", args.start, "--end", args.end,
                        "--members", "30", "--seed", "201805",
                        "--background-day-offset", "-1", "--imerg-stride", "1",
                        "--set", "observations.imerg.factor=8",
                        "--set", "observations.imerg.error_corr_cells=0.75",
                        "--holdout-folds", "5", "--holdout-fold", str(fold),
                        "--group", group, "--out", str(prefix.with_suffix(".npz")),
                        "--report", str(prefix.with_suffix(".json"))]
            try:
                sweep.main()
            finally:
                sys.argv = previous_argv
    metadata_name = (
        "comparison.json"
        if args.model == "both" and args.fold is None
        else f"run_{args.model}_{args.variant_set}_"
             f"{'all' if args.fold is None else f'fold{args.fold}'}.json"
    )
    (out / metadata_name).write_text(json.dumps(dict(
        experiment=EXPERIMENT, frozen_reference="v2_simul_s04_ig010",
        unet_checkpoint=args.unet_ckpt, graphflow_checkpoint=args.graphflow_ckpt,
        start=args.start, end=args.end, members=30, folds=5,
        selected_model=args.model, selected_fold=args.fold,
        variant_set=args.variant_set,
        interpretation=(
            "Broad one-fold development screen; shortlist at most three arms for "
            "five-fold verification."
            if args.variant_set == "screen" else
            "Four cases: each prior, and each prior with identical frozen DA. "
            "Default dates are development-only; not independent confirmation."
        ),
    ), indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    train_parser = commands.add_parser("train")
    train_parser.add_argument("--stage", choices=["screen", "full"], default="screen")
    train_parser.add_argument("--baseline", action="store_true")
    train_parser.add_argument("--resume")
    train_parser.add_argument("--prepare-only", action="store_true")
    val = commands.add_parser("validate")
    val.add_argument("--ckpt", required=True)
    val.add_argument("--out", default=f"runs/{EXPERIMENT}/prior_validation")
    smoke_parser = commands.add_parser("smoke")
    smoke_parser.add_argument("--steps", type=int, default=20)
    smoke_parser.add_argument("--out", default=f"runs/{EXPERIMENT}/smoke")
    for sub in (val, smoke_parser):
        sub.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    da = commands.add_parser("da")
    da.add_argument("--unet-ckpt", default="runs/prior_h100_cpc_v2/best.pt")
    da.add_argument("--graphflow-ckpt", default="runs/prior_h100_cpc_graphflow_g0/best.pt")
    da.add_argument("--stations", required=True)
    da.add_argument("--imerg", required=True, help="prepared S04 IMERG, not native product")
    da.add_argument("--start", default="2022-05-01")
    da.add_argument("--end", default="2022-05-10")
    da.add_argument("--out", default=f"runs/{EXPERIMENT}/frozen_da")
    da.add_argument(
        "--model", choices=("both", "cpcv2", "graphflow"), default="both",
        help="run both priors, or only one side of the paired comparison",
    )
    da.add_argument(
        "--fold", type=int, choices=range(5),
        help="run one zero-based spatial fold; default runs all five",
    )
    da.add_argument(
        "--variant-set", choices=("frozen", "screen"), default="frozen",
        help="frozen comparison arm or broad GraphFlow-only DA screening catalogue",
    )
    args = parser.parse_args()
    if args.action == "smoke" and args.steps < 2:
        parser.error("smoke requires at least two optimization steps")
    {"train": train, "validate": validate, "smoke": smoke, "da": frozen_da}[args.action](args)


if __name__ == "__main__":
    main()
