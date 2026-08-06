#!/usr/bin/env python
"""Train the conditional flow-matching downscaler.

Single GPU:
    python scripts/train.py --config configs/train_h100.yaml

2 x V100 (one node):
    torchrun --nproc_per_node=2 scripts/train.py --config configs/train_v100.yaml

Single stage: the prior is conditioned on the dynamic channels selected in the
configuration plus static fields, so it trains on the full 1981-2018 record.
IMERG never reaches the network -- it is assimilated as an observation at
inference time (docs/METHODOLOGY.md S4).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, WeightedRandomSampler

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bdhires.data import DatasetConfig, PrecipDataset  # noqa: E402
from bdhires.eval import MonitorConfig, ValidationMonitor  # noqa: E402
from bdhires.grids import WIDE, crop_offsets, get_grid  # noqa: E402
from bdhires.models import EMA, RectifiedFlow, UNet, flow_matching_loss  # noqa: E402
from bdhires.transforms import (  # noqa: E402
    load_climatology,
    CondTransform,
    PrecipTransform,
    ResidualSpec,
)
from bdhires.utils.dist import cleanup_distributed, is_main, setup_distributed  # noqa: E402
from bdhires.utils.dist import amp_dtype, broadcast_flag  # noqa: E402
from bdhires.utils.progress import ProgressReporter, format_duration  # noqa: E402
from bdhires.utils.summary import training_summary  # noqa: E402


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_checkpoint(state: dict, path: Path) -> None:
    """Atomically replace a checkpoint so an interrupted write cannot corrupt it."""
    partial = path.with_suffix(path.suffix + ".part")
    torch.save(state, partial)
    partial.replace(path)


def resolve_residual(cfg: dict, stats: dict) -> ResidualSpec:
    """ResidualSpec from the stats file, with a config-level override.

    ``data.residual_override: none`` forces the ABSOLUTE parameterisation --
    the network predicts T(CHIRPS) rather than T(CHIRPS) - T(base).

    Why that matters here. In residual mode, emitting dry rainfall requires the
    network to output exactly -(T(base) + T(0)), a value that moves per pixel
    with the base. CPC is wet over 84% of the domain once 0.5 degree cells are
    smeared to 0.05 degrees, so the dry target is almost never near zero and the
    network -- which shrinks toward its conditional mean -- consistently
    undershoots. Measured on v1: predicted residual -0.71 where the truth was
    -2.59. In absolute mode "dry" is the single fixed value T(0), shared by
    ~54% of training pixels, which is a mode a flow model can actually learn.
    No information is lost: cpc_precip stays in cond_channels either way.
    """
    spec = ResidualSpec.from_stats(stats)
    override = str(cfg.get("data", {}).get("residual_override", "")).strip().lower()
    if override in {"none", "off", "absolute", "false"}:
        return ResidualSpec(enabled=False)
    return spec


def build_dataset(cfg: dict, split: str) -> PrecipDataset:
    stats = json.loads(Path(cfg["data"]["stats"]).read_text())
    tf = PrecipTransform.from_dict(stats["precip_transform"])
    selected = cfg["data"].get("cond_channels")
    wet = cfg["train"].get("wet_sampling", {}) if split == "train" else {}
    if not wet.get("enabled", False):
        wet = {}
    dcfg = DatasetConfig(
        root=cfg["data"]["zarr"],
        crop=cfg["data"]["crop"],
        random_crop=(split == "train"),
        years=tuple(cfg["data"]["years"][split]),
        seasonal_encoding=cfg["data"].get("seasonal_encoding", True),
        min_valid_fraction=cfg["data"].get("min_valid_fraction", 0.3),
        cond_channels=tuple(selected) if selected else None,
        wet_crop_probability=float(wet.get("crop_probability", 0.0)),
        wet_crop_quantile=float(wet.get("crop_quantile", 0.95)),
        wet_crop_tries=int(wet.get("crop_tries", 8)),
    )
    return PrecipDataset(
        dcfg,
        tf,
        cond_mean=np.asarray(stats["cond_mean"], np.float32),
        cond_std=np.asarray(stats["cond_std"], np.float32),
        cond_transform=CondTransform.from_stats(stats),
        residual=resolve_residual(cfg, stats),
        climatology=load_climatology(cfg["data"]["stats"], stats),
    )


def build_training_sampler(cfg: dict, dataset: PrecipDataset, world_size: int):
    """Return distributed or controlled wet-day sampling for the training split."""
    wet = cfg["train"].get("wet_sampling") or {}
    if not wet.get("enabled", False):
        return DistributedSampler(dataset) if world_size > 1 else None
    if world_size > 1:
        raise ValueError(
            "wet-day sampling currently supports a single training process; "
            "the CPC v2 GH200 configuration uses world_size=1"
        )

    stats = json.loads(Path(cfg["data"]["stats"]).read_text())
    daily = stats.get("daily_wetness")
    if not daily:
        raise ValueError(
            "wet_sampling is enabled but the statistics file has no "
            "daily_wetness; recompute it with 06_compute_stats.py --daily-wetness"
        )
    stored_indices = np.asarray(daily["time_indices"], dtype=np.int64)
    means = np.asarray(daily["land_mean_mm_day"], dtype=np.float64)
    if not np.array_equal(stored_indices, dataset.index):
        by_index = dict(zip(stored_indices.tolist(), means.tolist()))
        try:
            means = np.asarray([by_index[int(i)] for i in dataset.index], np.float64)
        except KeyError as exc:
            raise ValueError(
                f"daily wetness statistics do not cover training index {exc.args[0]}"
            ) from exc
    quantile = float(wet.get("day_quantile", 0.9))
    component = float(wet.get("wet_component_fraction", 0.35))
    if not 0.0 <= quantile < 1.0:
        raise ValueError("wet_sampling.day_quantile must lie in [0, 1)")
    if not 0.0 <= component < 1.0:
        raise ValueError("wet_component_fraction must lie in [0, 1)")
    wet_mask = means >= np.quantile(means, quantile)
    if not wet_mask.any():
        raise ValueError("wet-day sampler selected no days")

    # A probability mixture retains a uniform component while deliberately
    # spending more optimizer steps on the wettest days.
    weights = np.full(len(dataset), (1.0 - component) / len(dataset), np.float64)
    weights[wet_mask] += component / int(wet_mask.sum())
    generator = torch.Generator().manual_seed(int(cfg["train"]["seed"]))
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )


def _masked_coarse_mean(
    field: torch.Tensor,
    weight: torch.Tensor,
    factor: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable weighted means on approximately 0.5-degree blocks."""
    height, width = field.shape[-2:]
    pad_h = (-height) % factor
    pad_w = (-width) % factor
    padding = (0, pad_w, 0, pad_h)
    numerator = F.avg_pool2d(
        F.pad(field * weight, padding), factor, stride=factor
    )
    denominator = F.avg_pool2d(F.pad(weight, padding), factor, stride=factor)
    return numerator / denominator.clamp_min(1.0e-6), denominator > 1.0e-6


def _stable_coarse_transform(transform: PrecipTransform, precipitation: torch.Tensor):
    """Variance stabilisation with a finite derivative at zero rainfall.

    ``sqrt(p)`` and ``cbrt(p)`` are perfectly valid data transforms, but their
    derivatives diverge at ``p=0``.  Targets do not require gradients, so this is
    invisible in ordinary flow matching.  The clean-field auxiliary loss does
    backpropagate through physical precipitation and therefore needs a small
    offset at the dry boundary.  Applying the same mapping to prediction and
    target preserves equality and the intended tail compression.
    """
    offset = max(float(transform.eps), 1.0e-6)
    if transform.kind == "sqrt":
        raw = torch.sqrt(precipitation.clamp_min(0.0) + offset)
        return (raw - transform.mu) / transform.sd
    if transform.kind == "cbrt":
        raw = torch.pow(precipitation.clamp_min(0.0) + offset, 1.0 / 3.0)
        return (raw - transform.mu) / transform.sd
    return transform.forward(precipitation)


def build_coarse_clean_loss(cfg: dict, dataset: PrecipDataset, batch: dict, device):
    """Build the optional clean-field magnitude loss for one training batch."""
    options = cfg["train"].get("coarse_consistency") or {}
    target_weight = float(options.get("target_weight", 0.0))
    cpc_weight = float(options.get("cpc_weight", 0.0))
    if target_weight <= 0.0 and cpc_weight <= 0.0:
        return None
    factor = int(options.get("factor", 10))
    if factor < 1:
        raise ValueError("coarse_consistency.factor must be positive")

    base = batch["base"].to(device, non_blocking=True)
    target_mm = batch["target_mm"].to(device, non_blocking=True).float()
    base_mm = batch["base_mm"].to(device, non_blocking=True).float()
    mask = batch["mask"].to(device, non_blocking=True).float()
    base_valid = batch["base_valid"].to(device, non_blocking=True).float()
    transform = dataset.transform
    residual = dataset.residual

    target_coarse, target_support = _masked_coarse_mean(target_mm, mask, factor)
    base_weight = mask * base_valid
    base_coarse, base_support = _masked_coarse_mean(base_mm, base_weight, factor)

    def clean_loss(clean: torch.Tensor) -> torch.Tensor:
        predicted_t = residual.decode(clean.float(), base.float())
        predicted_mm = transform.inverse(predicted_t)
        predicted_target, _ = _masked_coarse_mean(predicted_mm, mask, factor)
        loss = predicted_mm.new_zeros(())
        if target_weight > 0.0:
            difference = F.smooth_l1_loss(
                _stable_coarse_transform(transform, predicted_target),
                _stable_coarse_transform(transform, target_coarse),
                reduction="none",
            )
            loss = loss + target_weight * difference[target_support].mean()
        if cpc_weight > 0.0:
            predicted_base, _ = _masked_coarse_mean(
                predicted_mm, base_weight, factor
            )
            difference = F.smooth_l1_loss(
                _stable_coarse_transform(transform, predicted_base),
                _stable_coarse_transform(transform, base_coarse),
                reduction="none",
            )
            valid_difference = difference[base_support]
            if valid_difference.numel():
                loss = loss + cpc_weight * valid_difference.mean()
        return loss

    return clean_loss


def build_monitor(cfg: dict, device, out_dir: Path) -> ValidationMonitor | None:
    """Build the sampled-validation monitor, or None if it is switched off.

    Uses a FIXED crop on the production grid (not the random training crops), so
    the same geography is re-sampled at every epoch and the panels are directly
    comparable across the run.
    """
    mcfg = MonitorConfig.from_dict(cfg.get("validation"))
    if not mcfg.enabled:
        return None
    stats = json.loads(Path(cfg["data"]["stats"]).read_text())
    tf = PrecipTransform.from_dict(stats["precip_transform"])
    grid = get_grid(cfg["data"].get("monitor_grid", "bd"))
    if grid.nlon != cfg["data"]["crop"]:
        print(
            f"[validation monitor] disabled: grid {grid.name} is {grid.nlon} wide "
            f"but the model was built for {cfg['data']['crop']}",
            flush=True,
        )
        return None
    selected = cfg["data"].get("cond_channels")
    dataset = PrecipDataset(
        DatasetConfig(
            root=cfg["data"]["zarr"],
            crop=grid.nlon,
            random_crop=False,
            crop_origin=crop_offsets(WIDE, grid),
            years=tuple(cfg["data"]["years"]["val"]),
            seasonal_encoding=cfg["data"].get("seasonal_encoding", True),
            cond_channels=tuple(selected) if selected else None,
        ),
        tf,
        cond_mean=np.asarray(stats["cond_mean"], np.float32),
        cond_std=np.asarray(stats["cond_std"], np.float32),
        cond_transform=CondTransform.from_stats(stats),
        residual=resolve_residual(cfg, stats),
        climatology=load_climatology(cfg["data"]["stats"], stats),
    )
    return ValidationMonitor(
        dataset,
        tf,
        device,
        out_dir / "validation",
        cfg=mcfg,
        era5_tp_index=int(
            cfg["data"].get(
                "precip_cond_index",
                cfg["data"].get("era5_tp_cond_index", 0),
            )
        ),
        baseline_label=cfg["data"].get("precip_baseline_label", "ERA5 input"),
        baseline_channel=cfg["data"].get(
            "precip_baseline_channel",
            "era5_tp",
        ),
        baseline_valid_channel=cfg["data"].get(
            "precip_baseline_coverage_channel"
        ),
        baseline_valid_index=cfg["data"].get("precip_coverage_cond_index"),
        cond_transform=dataset.cond_transform,
        cond_mean=dataset.cond_mean,
        cond_std=dataset.cond_std,
        extent=(grid.lon_min, grid.lon_max, grid.lat_min, grid.lat_max),
        hurdle=bool((cfg["train"].get("hurdle") or {}).get("enabled", False)),
        dry_mask_mode=str((cfg["train"].get("hurdle") or {}).get("mask_mode", "threshold")),
        dry_threshold=float((cfg["train"].get("hurdle") or {}).get("mask_threshold", 0.5)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--init-from", default=None, help="warm start from a stage-A checkpoint")
    args = ap.parse_args()
    cfg = load_cfg(args.config)

    rank, world, local, device = setup_distributed()
    torch.manual_seed(cfg["train"]["seed"] + rank)

    train_ds = build_dataset(cfg, "train")
    val_ds = build_dataset(cfg, "val")

    sampler = build_training_sampler(cfg, train_ds, world)
    dl = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg["train"]["num_workers"] > 0,
    )
    val_dl = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], num_workers=2)

    hurdle_cfg = cfg["train"].get("hurdle") or {}
    hurdle_enabled = bool(hurdle_cfg.get("enabled", False))
    dry_threshold_mm = float(hurdle_cfg.get("wet_threshold_mm", 0.1))
    # Channel 0 is the flow velocity; channel 1 is the dry-probability logit.
    model = UNet(
        in_channels=1,
        cond_channels=train_ds.total_cond_channels,
        out_channels=2 if hurdle_enabled else 1,
        image_size=cfg["data"]["crop"],
        **cfg["model"],
    ).to(device)

    if args.init_from:
        sd = torch.load(args.init_from, map_location="cpu")
        missing, unexpected = model.load_state_dict(sd["ema"], strict=False)
        if is_main():
            print(f"warm start: {len(missing)} missing, {len(unexpected)} unexpected keys")

    # EMA is optional.  It costs a full state-dict traversal every step plus a
    # second copy of the weights, and it smooths over exactly the checkpoint-to-
    # checkpoint variation you may want to see.  With `use_ema: false` the online
    # weights are validated and saved directly.
    use_ema = bool(cfg["train"].get("use_ema", True))
    ema = EMA(model, decay=cfg["train"]["ema_decay"]) if use_ema else None
    flow = RectifiedFlow()
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"],
        betas=(0.9, 0.999),
    )

    dtype = amp_dtype(device)
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))
    net = DDP(model, device_ids=[local]) if world > 1 else model

    out_dir = Path(cfg["train"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sampled validation runs on rank 0 only: it is a diagnostic, not a reduction.
    monitor = build_monitor(cfg, device, out_dir) if is_main() else None
    if monitor is not None:
        monitor.validate_cadence(cfg["train"]["ckpt_every"])
        print(f"[validation monitor] {monitor.describe()}", flush=True)

    step, start_epoch = 0, 0
    best_val_loss = float("inf")
    best_crps = float("inf")
    # Evaluations since the sampled CRPS last improved.  The v3 run peaked around
    # epoch 80-125 and then degraded for 50 epochs; nothing stopped it.
    stale_evaluations = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ck["model"])
        if ema is not None:
            if ck.get("ema") is None:
                raise ValueError(
                    f"{args.resume} carries no EMA weights but this config sets "
                    f"use_ema: true; resume with use_ema: false or start fresh"
                )
            ema.shadow = {
                key: value.to(device=ema.shadow[key].device)
                for key, value in ck["ema"].items()
            }
        opt.load_state_dict(ck["opt"])
        step, start_epoch = ck["step"], ck["epoch"] + 1
        # New checkpoints carry the global best.  Falling back to val_loss
        # keeps checkpoints written by older versions resumable.
        best_val_loss = float(
            ck.get("best_val_loss", ck.get("val_loss", float("inf")))
        )
        best_crps = float(ck.get("best_crps", float("inf")))

    total_steps = cfg["train"]["epochs"] * max(1, len(dl))
    warmup = cfg["train"]["warmup_steps"]

    if is_main():
        print(
            training_summary(
                cfg=cfg,
                config_path=args.config,
                model=model,
                train_ds=train_ds,
                val_ds=val_ds,
                device=device,
                world_size=world,
                amp_dtype=dtype,
                steps_per_epoch=max(1, len(dl)),
                total_steps=total_steps,
                stats=json.loads(Path(cfg["data"]["stats"]).read_text()),
                monitor=monitor,
                resumed_from=args.resume,
                start_epoch=start_epoch,
            ),
            flush=True,
        )

    reporter = (
        ProgressReporter(
            total_epochs=cfg["train"]["epochs"],
            steps_per_epoch=max(1, len(dl)),
            log_every=cfg["train"]["log_every"],
            start_epoch=start_epoch,
        )
        if is_main()
        else None
    )
    run_started = time.time()

    def lr_at(s):
        if s < warmup:
            return cfg["train"]["lr"] * s / max(1, warmup)
        p = (s - warmup) / max(1, total_steps - warmup)
        return cfg["train"]["lr"] * (0.5 * (1 + math.cos(math.pi * min(p, 1.0))))

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        net.train()
        run_fm, run_coarse, run_hurdle, batches = 0.0, 0.0, 0.0, 0
        if reporter is not None:
            reporter.begin_epoch(epoch)
        for i, batch in enumerate(dl):
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            x1 = batch["x1"].to(device, non_blocking=True)
            cond = batch["cond"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            clean_loss_fn = build_coarse_clean_loss(
                cfg, train_ds, batch, device
            )
            # Dry mask from RAW mm, not from transformed space: the threshold is
            # a physical statement about the rain gauge, not about the network.
            dry_target = None
            if hurdle_enabled:
                dry_target = (
                    batch["target_mm"].to(device, non_blocking=True) < dry_threshold_mm
                ).float()

            with torch.autocast("cuda", dtype=dtype, enabled=device.type == "cuda"):
                loss, fm_loss, coarse_loss, hurdle_loss = flow_matching_loss(
                    net, x1, cond, flow, mask=mask,
                    cond_dropout=cfg["train"]["cond_dropout"],
                    logit_normal_t=cfg["train"].get("logit_normal_t", True),
                    clean_loss_fn=clean_loss_fn,
                    return_components=True,
                    dry_target=dry_target,
                    hurdle_weight=float(hurdle_cfg.get("weight", 1.0)),
                    dry_weight=float(cfg["train"].get("dry_weight", 1.0)),
                )
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            scaler.step(opt)
            scaler.update()
            if ema is not None:
                ema.update(model)
            run_fm += fm_loss.item()
            run_coarse += coarse_loss.item()
            run_hurdle += hurdle_loss.item()
            batches += 1
            step += 1
            if reporter is not None:
                reporter.update(loss.item(), opt.param_groups[0]["lr"])

        if reporter is not None:
            peak = ""
            if device.type == "cuda":
                peak = (
                    f"peak {torch.cuda.max_memory_allocated(device) / 2**30:.1f} GiB"
                )
            components = (
                f"FM {run_fm / max(1, batches):.4f}  "
                f"coarse {run_coarse / max(1, batches):.4f}"
                + (f"  hurdle {run_hurdle / max(1, batches):.4f}" if hurdle_enabled else "")
            )
            if peak:
                components += f"  {peak}"
            print(reporter.end_epoch(components), flush=True)

        stop_now = False
        if is_main() and (epoch + 1) % cfg["train"]["ckpt_every"] == 0:
            val = evaluate_ema_loss(
                model,
                ema,
                val_dl,
                flow,
                device,
                dtype,
                seed=cfg["train"]["seed"] + 10_000,
            )
            # Sampled validation: the selection signal that actually tracks
            # forecast quality.  The flow-matching loss above is kept for
            # continuity but is too noisy to choose a checkpoint with
            # (docs/DIAGNOSIS_epoch119.md item 4).
            crps = None
            if monitor is not None and monitor.should_run(epoch):
                summary = monitor.run(model, ema, epoch, step)
                if summary is not None:
                    crps = summary["mean_crps_mm"]
                    print(
                        f"    sampled validation ({summary['seconds']}s):  "
                        f"mean CRPS {crps:.3f} mm",
                        flush=True,
                    )
                    for case in summary["cases"]:
                        print(
                            f"      {case['date']} "
                            f"q{int(round(case['quantile'] * 100)):02d}   "
                            f"CRPS {case['crps_mm']:6.2f}   "
                            f"bias {case['bias_mm']:+6.2f}   "
                            f"r {case['spatial_correlation']:5.2f}   "
                            f"spread {case['mean_spread_mm']:5.2f}   "
                            f"cov90 {case['interval_90_coverage'] * 100:5.1f}%",
                            flush=True,
                        )
            label = "val_ema" if ema is not None else "val"
            print(f"    {label} (flow-matching loss) {val:.4f}", flush=True)

            # Prefer CRPS whenever we have it; fall back to the FM loss on the
            # epochs in between so early checkpoints are still ranked somehow.
            if crps is not None:
                improved = crps < best_crps
                if improved:
                    best_crps = crps
                    stale_evaluations = 0
                else:
                    stale_evaluations += 1
                    patience = int(cfg["train"].get("early_stop_patience", 0))
                    if patience and stale_evaluations >= patience:
                        print(
                            f"    early stop: sampled CRPS has not improved in "
                            f"{stale_evaluations} evaluations "
                            f"(best {best_crps:.4f} mm)",
                            flush=True,
                        )
                        stop_now = True
            else:
                improved = False
            if val < best_val_loss:
                best_val_loss = val
                if best_crps == float("inf"):
                    improved = True     # no sampled score yet
            state = dict(
                model=model.state_dict(),
                ema=ema.state_dict() if ema is not None else None,
                weights="ema" if ema is not None else "model",
                opt=opt.state_dict(),
                step=step,
                epoch=epoch,
                cfg=cfg,
                val_loss=val,
                best_val_loss=best_val_loss,
                crps=crps,
                best_crps=best_crps,
                stale_evaluations=stale_evaluations,
                selected_by="sampled_crps" if best_crps < float("inf") else "fm_loss",
            )
            # Write the new best first.  If the allocation ends between the
            # two atomic writes, production still has the best model and
            # resume falls back safely to the preceding latest checkpoint.
            if improved:
                save_checkpoint(state, out_dir / "best.pt")
                criterion = (
                    f"sampled CRPS={best_crps:.4f} mm"
                    if best_crps < float("inf")
                    else f"val={best_val_loss:.6f}"
                )
                print(
                    f"saved new best checkpoint: {out_dir / 'best.pt'} ({criterion})",
                    flush=True,
                )
            save_checkpoint(state, out_dir / "last.pt")
            print(f"saved latest checkpoint: {out_dir / 'last.pt'}", flush=True)

            # Retained snapshots.  best.pt and last.pt are both overwritten every
            # ckpt_every epochs, so when the v3 run peaked near epoch 80 there was
            # no way to get back to it.  These keep the optimiser state out to
            # stay small -- they are for evaluation, not for resuming.
            keep_every = int(cfg["train"].get("keep_every", 0))
            if keep_every and (epoch + 1) % keep_every == 0:
                snapshot = {
                    key: value for key, value in state.items() if key != "opt"
                }
                path = out_dir / f"epoch_{epoch + 1:04d}.pt"
                save_checkpoint(snapshot, path)
                print(f"retained snapshot: {path}", flush=True)

        # Every rank must agree, or the others hang at the next collective.
        if broadcast_flag(stop_now, device):
            if is_main():
                print(
                    f"stopping at epoch {epoch + 1} of {cfg['train']['epochs']}",
                    flush=True,
                )
            break

    if is_main():
        save_checkpoint(
            dict(
                model=model.state_dict(),
                ema=ema.state_dict() if ema is not None else None,
                weights="ema" if ema is not None else "model",
                cfg=cfg,
                epoch=cfg["train"]["epochs"] - 1,
                best_val_loss=best_val_loss,
                best_crps=best_crps,
            ),
            out_dir / "final.pt",
        )
        print("=" * 78)
        print(" TRAINING COMPLETE")
        print(f"   wall time        {format_duration(time.time() - run_started)}")
        print(f"   steps            {step:,}")
        if best_crps < float("inf"):
            print(f"   best CRPS        {best_crps:.4f} mm  -> {out_dir / 'best.pt'}")
        print(f"   best val loss    {best_val_loss:.6f}")
        print(f"   final weights    {out_dir / 'final.pt'}")
        if monitor is not None:
            print(f"   validation       {out_dir / 'validation' / 'progress.png'}")
        print("=" * 78, flush=True)
    cleanup_distributed()


@torch.no_grad()
def evaluate_loss(model, dl, flow, device, dtype, seed: int | None = None) -> float:
    model.eval()
    fork_devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    try:
        with torch.random.fork_rng(devices=fork_devices):
            if seed is not None:
                torch.manual_seed(seed)
            tot, n = 0.0, 0
            for batch in dl:
                x1 = batch["x1"].to(device)
                cond = batch["cond"].to(device)
                mask = batch["mask"].to(device)
                with torch.autocast(
                    "cuda", dtype=dtype, enabled=device.type == "cuda"
                ):
                    tot += flow_matching_loss(
                        model,
                        x1,
                        cond,
                        flow,
                        mask=mask,
                        cond_dropout=0.0,
                    ).item()
                n += 1
                if n >= 50:
                    break
            return tot / max(1, n)
    finally:
        model.train()


@torch.no_grad()
def evaluate_ema_loss(model, ema, dl, flow, device, dtype, seed: int) -> float:
    """Validate the weights that testing and production will use.

    ``ema=None`` validates the online weights directly, which is the whole point
    of ``use_ema: false`` -- otherwise the reported score would not describe the
    weights actually written to the checkpoint.
    """
    if ema is None:
        return evaluate_loss(model, dl, flow, device, dtype, seed=seed)
    online_state = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }
    try:
        ema.copy_to(model)
        return evaluate_loss(model, dl, flow, device, dtype, seed=seed)
    finally:
        model.load_state_dict(online_state)
        model.train()


if __name__ == "__main__":
    main()
