#!/usr/bin/env python
"""Train the conditional flow-matching downscaler.

Single GPU:
    python scripts/train.py --config configs/train_h100.yaml

2 x V100 (one node):
    torchrun --nproc_per_node=2 scripts/train.py --config configs/train_v100.yaml

Single stage: the prior is conditioned on ERA5 and the static fields only, so
it trains on the full 1981-2018 record.  IMERG never reaches the network -- it
is assimilated as an observation at inference time (docs/METHODOLOGY.md S4).
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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bdhires.data import DatasetConfig, PrecipDataset  # noqa: E402
from bdhires.eval import MonitorConfig, ValidationMonitor  # noqa: E402
from bdhires.grids import WIDE, crop_offsets, get_grid  # noqa: E402
from bdhires.models import EMA, RectifiedFlow, UNet, flow_matching_loss  # noqa: E402
from bdhires.transforms import CondTransform, PrecipTransform  # noqa: E402
from bdhires.utils.dist import cleanup_distributed, is_main, setup_distributed  # noqa: E402
from bdhires.utils.dist import amp_dtype  # noqa: E402
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


def build_dataset(cfg: dict, split: str) -> PrecipDataset:
    stats = json.loads(Path(cfg["data"]["stats"]).read_text())
    tf = PrecipTransform.from_dict(stats["precip_transform"])
    dcfg = DatasetConfig(
        root=cfg["data"]["zarr"],
        crop=cfg["data"]["crop"],
        random_crop=(split == "train"),
        years=tuple(cfg["data"]["years"][split]),
        seasonal_encoding=cfg["data"].get("seasonal_encoding", True),
        min_valid_fraction=cfg["data"].get("min_valid_fraction", 0.3),
    )
    return PrecipDataset(
        dcfg,
        tf,
        cond_mean=np.asarray(stats["cond_mean"], np.float32),
        cond_std=np.asarray(stats["cond_std"], np.float32),
        cond_transform=CondTransform.from_stats(stats),
    )


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
    dataset = PrecipDataset(
        DatasetConfig(
            root=cfg["data"]["zarr"],
            crop=grid.nlon,
            random_crop=False,
            crop_origin=crop_offsets(WIDE, grid),
            years=tuple(cfg["data"]["years"]["val"]),
            seasonal_encoding=cfg["data"].get("seasonal_encoding", True),
        ),
        tf,
        cond_mean=np.asarray(stats["cond_mean"], np.float32),
        cond_std=np.asarray(stats["cond_std"], np.float32),
        cond_transform=CondTransform.from_stats(stats),
    )
    return ValidationMonitor(
        dataset,
        tf,
        device,
        out_dir / "validation",
        cfg=mcfg,
        era5_tp_index=int(cfg["data"].get("era5_tp_cond_index", 0)),
        cond_transform=CondTransform.from_stats(stats),
        cond_mean=np.asarray(stats["cond_mean"], np.float32),
        cond_std=np.asarray(stats["cond_std"], np.float32),
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

    sampler = DistributedSampler(train_ds) if world > 1 else None
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

    model = UNet(
        in_channels=1,
        cond_channels=train_ds.total_cond_channels,
        out_channels=1,
        image_size=cfg["data"]["crop"],
        **cfg["model"],
    ).to(device)

    if args.init_from:
        sd = torch.load(args.init_from, map_location="cpu")
        missing, unexpected = model.load_state_dict(sd["ema"], strict=False)
        if is_main():
            print(f"warm start: {len(missing)} missing, {len(unexpected)} unexpected keys")

    ema = EMA(model, decay=cfg["train"]["ema_decay"])
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
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ck["model"])
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
        if sampler is not None:
            sampler.set_epoch(epoch)
        net.train()
        t0, run = time.time(), 0.0
        if reporter is not None:
            reporter.begin_epoch(epoch)
        for i, batch in enumerate(dl):
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            x1 = batch["x1"].to(device, non_blocking=True)
            cond = batch["cond"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            with torch.autocast("cuda", dtype=dtype, enabled=device.type == "cuda"):
                loss = flow_matching_loss(
                    net, x1, cond, flow, mask=mask,
                    cond_dropout=cfg["train"]["cond_dropout"],
                    logit_normal_t=cfg["train"].get("logit_normal_t", True),
                )
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            scaler.step(opt)
            scaler.update()
            ema.update(model)
            run += loss.item()
            step += 1
            if reporter is not None:
                reporter.update(loss.item(), opt.param_groups[0]["lr"])

        if reporter is not None:
            peak = ""
            if device.type == "cuda":
                peak = (
                    f"peak {torch.cuda.max_memory_allocated(device) / 2**30:.1f} GiB"
                )
            print(reporter.end_epoch(peak), flush=True)

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
            print(f"    val_ema (flow-matching loss) {val:.4f}", flush=True)

            # Prefer CRPS whenever we have it; fall back to the FM loss on the
            # epochs in between so early checkpoints are still ranked somehow.
            if crps is not None:
                improved = crps < best_crps
                if improved:
                    best_crps = crps
            else:
                improved = False
            if val < best_val_loss:
                best_val_loss = val
                if best_crps == float("inf"):
                    improved = True     # no sampled score yet
            state = dict(
                model=model.state_dict(),
                ema=ema.state_dict(),
                opt=opt.state_dict(),
                step=step,
                epoch=epoch,
                cfg=cfg,
                val_loss=val,
                best_val_loss=best_val_loss,
                crps=crps,
                best_crps=best_crps,
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

    if is_main():
        save_checkpoint(
            dict(
                model=model.state_dict(),
                ema=ema.state_dict(),
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
    """Validate the EMA weights used by testing and production."""
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
