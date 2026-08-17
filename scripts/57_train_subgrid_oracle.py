#!/usr/bin/env python3
"""Train one V3-SG phase (coarse, allocation, or coupled joint flow).

Examples
--------
python scripts/57_train_subgrid_oracle.py \
  --config configs/train_h100_cpc_v3_subgrid_coarse.yaml

torchrun --nproc_per_node=1 scripts/57_train_subgrid_oracle.py \
  --config configs/train_h100_cpc_v3_subgrid_joint.yaml
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bdhires.data import (  # noqa: E402
    SUBGRID_SCHEMA,
    SubgridDataset,
    SubgridDatasetConfig,
    SubgridEncoding,
    encoding_metadata,
)
from bdhires.models import (  # noqa: E402
    AllocationFlow,
    CoarseHurdleFlow,
    CoupledSubgridFlow,
    EMA,
    HierarchicalState,
    allocation_flow_matching_loss,
    coarse_flow_matching_loss,
    hierarchical_flow_matching_loss,
    select_weights,
)
from bdhires.utils.dist import cleanup_distributed, is_main, setup_distributed  # noqa: E402


def atomic_save(payload: dict, path: Path) -> None:
    partial = path.with_suffix(path.suffix + ".part")
    torch.save(payload, partial)
    partial.replace(path)


def dataset_from_config(config: dict, split: str) -> SubgridDataset:
    data = config["data"]
    return SubgridDataset(
        SubgridDatasetConfig(
            root=data["zarr"],
            crop=int(data["crop"]),
            random_crop=split == "train",
            years=tuple(data["years"][split]),
            factor=int(data.get("factor", 10)),
            downsamplings=int(data.get("downsamplings", 3)),
            seed=int(config["train"]["seed"]) + (0 if split == "train" else 100_000),
            tile_domain=split == "val",
        )
    )


def _condition_channels(dataset: SubgridDataset) -> tuple[int, int]:
    coarse = int(dataset.z["coarse_cond"].shape[1]) if "coarse_cond" in dataset.z else 0
    fine = int(dataset.z["fine_cond"].shape[1]) if "fine_cond" in dataset.z else 0
    return coarse, fine


def validate_checkpoint(
    checkpoint: dict,
    *,
    expected_stage: str,
    dataset: SubgridDataset,
    label: str,
    expected_config: dict | None = None,
) -> None:
    """Reject checkpoints trained under a different target/loss contract."""
    if checkpoint.get("schema") != SUBGRID_SCHEMA:
        raise ValueError(
            f"{label} schema {checkpoint.get('schema')!r} is incompatible with "
            f"{SUBGRID_SCHEMA}"
        )
    if checkpoint.get("stage") != expected_stage:
        raise ValueError(
            f"{label} stage {checkpoint.get('stage')!r} != {expected_stage!r}"
        )
    if "subgrid_encoding" not in checkpoint:
        raise ValueError(f"{label} lacks frozen subgrid encoding metadata")
    checkpoint_encoding = SubgridEncoding.from_mapping(checkpoint["subgrid_encoding"])
    if encoding_metadata(checkpoint_encoding) != encoding_metadata(dataset.encoding):
        raise ValueError(f"{label} subgrid encoding differs from the target archive")
    if expected_config is not None and checkpoint.get("config") != expected_config:
        raise ValueError(f"{label} resolved config differs from the requested config")


def build_model(config: dict, dataset: SubgridDataset):
    stage = config["stage"]
    coarse_channels, fine_channels = _condition_channels(dataset)
    fine_size = int(config["data"]["crop"])
    coarse_size = fine_size // int(config["data"].get("factor", 10))
    if stage == "coarse":
        return CoarseHurdleFlow(
            coarse_channels, image_size=coarse_size, **config["model"]
        )
    if stage == "allocation":
        return AllocationFlow(
            fine_channels, image_size=fine_size, **config["model"]
        )
    if stage != "joint":
        raise ValueError("stage must be coarse, allocation, or joint")
    coarse = CoarseHurdleFlow(
        coarse_channels, image_size=coarse_size, **config["model"]["coarse"]
    )
    allocation = AllocationFlow(
        fine_channels, image_size=fine_size, **config["model"]["allocation"]
    )
    model = CoupledSubgridFlow(
        coarse,
        allocation,
        clean_context_probability=float(
            config["train"].get("clean_context_probability", 0.0)
        ),
    )
    coarse_path = config["train"].get("init_coarse")
    allocation_path = config["train"].get("init_allocation")
    if not coarse_path or not allocation_path:
        raise ValueError("joint training requires init_coarse and init_allocation checkpoints")
    coarse_checkpoint = torch.load(coarse_path, map_location="cpu")
    allocation_checkpoint = torch.load(allocation_path, map_location="cpu")
    validate_checkpoint(
        coarse_checkpoint,
        expected_stage="coarse",
        dataset=dataset,
        label=f"coarse initializer {coarse_path}",
    )
    validate_checkpoint(
        allocation_checkpoint,
        expected_stage="allocation",
        dataset=dataset,
        label=f"allocation initializer {allocation_path}",
    )
    model.load_pretrained_branches(
        select_weights(coarse_checkpoint), select_weights(allocation_checkpoint)
    )
    return model


def batch_loss(model, batch: dict, config: dict):
    stage = config["stage"]
    if stage == "coarse":
        return coarse_flow_matching_loss(
            model, batch["coarse_state"], batch["coarse_cond"], batch["coarse_valid"],
            occurrence_weight=float(config["train"].get("occurrence_weight", 0.1)),
            inactive_positive_weight=float(
                config["train"].get("inactive_positive_weight", 0.05)
            ),
        )
    if stage == "allocation":
        augmentation = config["train"].get("conditioning_augmentation", {})
        return allocation_flow_matching_loss(
            model,
            batch["allocation_state"],
            batch["fine_cond"],
            batch["coarse_state"],
            batch["fine_valid"],
            max_coarse_noise=float(augmentation.get("max_coarse_noise", 1.0)),
            clean_probability=float(augmentation.get("clean_probability", 0.15)),
            occurrence_weight=float(config["train"].get("occurrence_weight", 0.1)),
            inactive_positive_weight=float(
                config["train"].get("inactive_positive_weight", 0.05)
            ),
        )
    return hierarchical_flow_matching_loss(
        model,
        HierarchicalState(batch["coarse_state"], batch["allocation_state"]),
        batch["coarse_cond"],
        batch["fine_cond"],
        batch["coarse_valid"],
        batch["fine_valid"],
        cond_dropout=float(config["train"].get("cond_dropout", 0.0)),
        coarse_weight=float(config["train"].get("coarse_loss_weight", 1.0)),
        allocation_weight=float(config["train"].get("allocation_loss_weight", 1.0)),
        occurrence_weight=float(config["train"].get("occurrence_weight", 0.1)),
        inactive_positive_weight=float(
            config["train"].get("inactive_positive_weight", 0.05)
        ),
        clean_context_probability=float(
            config["train"].get("clean_context_probability", 0.15)
        ),
    )


def move_batch(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


@torch.no_grad()
def validate(model, loader, config, device, max_batches: int | None = None) -> float:
    model.eval()
    values = []
    devices = [device.index or 0] if device.type == "cuda" else []
    # Validation uses fixed flow noise without perturbing the training RNG
    # stream (especially important because only rank zero validates under DDP).
    with torch.random.fork_rng(devices=devices):
        for index, batch in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            batch = move_batch(batch, device)
            torch.manual_seed(int(config["train"]["seed"]) + 900_000 + index)
            loss = batch_loss(model, batch, config)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite validation loss at batch {index}"
                )
            values.append(float(loss.detach().cpu()))
    model.train()
    if not values:
        raise ValueError("validation loader produced no batches")
    return float(sum(values) / len(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    rank, world, local_rank, device = setup_distributed()
    torch.manual_seed(int(config["train"]["seed"]) + rank)

    train_dataset = dataset_from_config(config, "train")
    val_dataset = dataset_from_config(config, "val")
    sampler = DistributedSampler(train_dataset, shuffle=True) if world > 1 else None
    loader = DataLoader(
        train_dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=int(config["train"].get("num_workers", 4)),
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=int(config["train"].get("num_workers", 4)) > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["train"]["batch_size"]),
        num_workers=min(2, int(config["train"].get("num_workers", 4))),
    )
    model = build_model(config, train_dataset).to(device)
    train_model = DDP(model, device_ids=[local_rank]) if world > 1 else model
    optimizer = torch.optim.AdamW(
        train_model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"].get("weight_decay", 0.0)),
    )
    epochs = int(config["train"]["epochs"])
    total_steps = max(1, epochs * len(loader))
    # Linear warmup then cosine decay to a small floor.
    #
    # Warmup matters most for the joint stage: its branches arrive pretrained,
    # and a full-rate step on the very first batch can move them off a good
    # initialisation before the zero-started cross-scale path has learned
    # anything.  The floor keeps the tail of a long run doing useful work
    # instead of taking zero-length steps for the last few epochs.
    warmup_fraction = float(config["train"].get("warmup_fraction", 0.02))
    if not 0.0 <= warmup_fraction < 0.5:
        raise ValueError("train.warmup_fraction must lie in [0, 0.5)")
    lr_min_fraction = float(config["train"].get("lr_min_fraction", 0.01))
    if not 0.0 <= lr_min_fraction < 1.0:
        raise ValueError("train.lr_min_fraction must lie in [0, 1)")
    warmup_steps = int(round(warmup_fraction * total_steps))

    def lr_scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return lr_min_fraction + (1.0 - lr_min_fraction) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    if is_main():
        print(
            f"lr schedule: warmup {warmup_steps} steps "
            f"({warmup_fraction:.1%}), cosine to {lr_min_fraction:.1%} of base "
            f"over {total_steps} total steps",
            flush=True,
        )
    use_ema = bool(config["train"].get("use_ema", True))
    precision = str(config["train"].get("precision", "fp32")).lower()
    if precision not in {"fp32", "bf16", "fp16"}:
        raise ValueError("train.precision must be fp32, bf16, or fp16")
    amp_enabled = device.type == "cuda" and precision != "fp32"
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and precision == "fp16")
    ema = EMA(model, decay=float(config["train"].get("ema_decay", 0.9995))) if use_ema else None
    start_epoch = 0
    best = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        validate_checkpoint(
            checkpoint,
            expected_stage=config["stage"],
            dataset=train_dataset,
            label=f"resume checkpoint {args.resume}",
            expected_config=config,
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        stored_total = int(checkpoint.get("total_steps", total_steps))
        if stored_total != total_steps:
            raise ValueError(
                f"resume checkpoint was built for {stored_total} total steps but "
                f"this config implies {total_steps}.  The cosine schedule would "
                "jump: either restore the original train.epochs or start a fresh "
                "run directory."
            )
        scheduler.load_state_dict(checkpoint["scheduler"])
        if ema is not None and checkpoint.get("ema") is not None:
            # Checkpoints are loaded on CPU, but EMA updates run beside the
            # model. Move the restored shadow before the first update.
            ema.shadow = {
                key: value.to(device=device) for key, value in checkpoint["ema"].items()
            }
        start_epoch = int(checkpoint["epoch"]) + 1
        best = float(checkpoint.get("best_val", best))

    output = Path(config["train"]["out_dir"])
    if is_main():
        output.mkdir(parents=True, exist_ok=True)
        (output / "config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        print(
            f"V3-SG stage={config['stage']} train={len(train_dataset)} val={len(val_dataset)} "
            f"parameters={sum(p.numel() for p in model.parameters()):,} device={device}",
            flush=True,
        )

    for epoch in range(start_epoch, epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        train_model.train()
        running = 0.0
        for batch in loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                loss = batch_loss(train_model, batch, config)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite training loss at epoch {epoch + 1}"
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                train_model.parameters(),
                float(config["train"].get("grad_clip", 1.0)),
                error_if_nonfinite=True,
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if ema is not None:
                ema.update(model)
            running += float(loss.detach().cpu())

        if is_main():
            validation_model = model
            online = None
            if ema is not None:
                online = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                ema.copy_to(model)
            maximum = config["train"].get("validation_max_batches")
            val = validate(
                validation_model,
                val_loader,
                config,
                device,
                None if maximum is None else int(maximum),
            )
            payload = {
                "schema": SUBGRID_SCHEMA,
                "stage": config["stage"],
                "epoch": epoch,
                "model": online if online is not None else model.state_dict(),
                "ema": None if ema is None else ema.state_dict(),
                "weights": "ema" if ema is not None else "model",
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "total_steps": total_steps,
                "best_val": min(best, val),
                "config": config,
                "subgrid_encoding": encoding_metadata(train_dataset.encoding),
            }
            if online is not None:
                model.load_state_dict(online)
            atomic_save(payload, output / "last.pt")
            if val < best:
                best = val
                atomic_save(payload, output / "best.pt")
            if (epoch + 1) % int(config["train"].get("ckpt_every", 5)) == 0:
                atomic_save(payload, output / f"epoch_{epoch + 1:04d}.pt")
            print(
                f"epoch {epoch + 1:4d}/{epochs} "
                f"train={running / max(len(loader), 1):.5f} "
                f"val={val:.5f} best={best:.5f}",
                flush=True,
            )
        if world > 1:
            # Other ranks wait while rank zero validates with EMA weights;
            # none can enter the next DDP forward prematurely.
            torch.distributed.barrier()

    cleanup_distributed()


if __name__ == "__main__":
    main()
