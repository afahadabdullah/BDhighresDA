#!/usr/bin/env python
"""Train the conditional flow-matching downscaler.

Single GPU:
    python scripts/train.py --config configs/train_h100.yaml

2 x V100 (one node):
    torchrun --nproc_per_node=2 scripts/train.py --config configs/train_v100.yaml

Two-stage recipe (recommended, see docs/METHODOLOGY.md):
    stage A  1981-2000, ERA5-only conditioning (IMERG channel masked)
    stage B  2001-2018, ERA5 + IMERG, initialised from stage A
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
from bdhires.models import EMA, RectifiedFlow, UNet, flow_matching_loss  # noqa: E402
from bdhires.transforms import PrecipTransform  # noqa: E402
from bdhires.utils.dist import cleanup_distributed, is_main, setup_distributed  # noqa: E402
from bdhires.utils.dist import amp_dtype  # noqa: E402


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_dataset(cfg: dict, split: str) -> PrecipDataset:
    stats = json.loads(Path(cfg["data"]["stats"]).read_text())
    tf = PrecipTransform.from_dict(stats["precip_transform"])
    dcfg = DatasetConfig(
        root=cfg["data"]["zarr"],
        crop=cfg["data"]["crop"],
        random_crop=(split == "train"),
        years=tuple(cfg["data"]["years"][split]),
        seasonal_encoding=cfg["data"].get("seasonal_encoding", True),
        zero_cond_channels=tuple(cfg["data"].get("zero_cond_channels", ())),
    )
    return PrecipDataset(
        dcfg,
        tf,
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
    if is_main():
        print(f"train days={len(train_ds)}  val days={len(val_ds)}  "
              f"cond channels={train_ds.total_cond_channels}")

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
    if is_main():
        print(f"model parameters: {model.num_parameters/1e6:.1f} M")

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
    step, start_epoch = 0, 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ck["model"])
        ema.shadow = {k: v for k, v in ck["ema"].items()}
        opt.load_state_dict(ck["opt"])
        step, start_epoch = ck["step"], ck["epoch"] + 1

    total_steps = cfg["train"]["epochs"] * max(1, len(dl))
    warmup = cfg["train"]["warmup_steps"]

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
            if is_main() and step % cfg["train"]["log_every"] == 0:
                print(f"ep {epoch} step {step} loss {run/(i+1):.4f} "
                      f"lr {opt.param_groups[0]['lr']:.2e} "
                      f"{(time.time()-t0)/(i+1):.2f}s/it", flush=True)

        if is_main() and (epoch + 1) % cfg["train"]["ckpt_every"] == 0:
            val = evaluate_loss(model, val_dl, flow, device, dtype)
            print(f"[epoch {epoch}] train {run/max(1,len(dl)):.4f}  val {val:.4f}", flush=True)
            torch.save(
                dict(model=model.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(),
                     step=step, epoch=epoch, cfg=cfg, val_loss=val),
                out_dir / "last.pt",
            )

    if is_main():
        torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), cfg=cfg),
                   out_dir / "final.pt")
    cleanup_distributed()


@torch.no_grad()
def evaluate_loss(model, dl, flow, device, dtype) -> float:
    model.eval()
    tot, n = 0.0, 0
    for batch in dl:
        x1 = batch["x1"].to(device)
        cond = batch["cond"].to(device)
        mask = batch["mask"].to(device)
        with torch.autocast("cuda", dtype=dtype, enabled=device.type == "cuda"):
            tot += flow_matching_loss(model, x1, cond, flow, mask=mask, cond_dropout=0.0).item()
        n += 1
        if n >= 50:
            break
    model.train()
    return tot / max(1, n)


if __name__ == "__main__":
    main()
