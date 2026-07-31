#!/usr/bin/env python3
"""Run a short production-equivalent GH200 training preflight on real data.

The preflight loads the configured Zarr and statistics, exercises the actual
multi-worker data loader, constructs the production U-Net, and runs a small
number of forward/backward/optimizer/EMA updates plus one validation batch.
It writes a machine-readable pass report but no model checkpoint.
"""
from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train import build_dataset, load_cfg  # noqa: E402

from bdhires.models import EMA, RectifiedFlow, UNet, flow_matching_loss  # noqa: E402
from bdhires.utils.dist import amp_dtype  # noqa: E402

EXPECTED_STATIC_CHANNELS = 7
EXPECTED_SEASONAL_CHANNELS = 2


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(repository: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def expected_days(years: tuple[int, int]) -> int:
    return sum(
        366 if calendar.isleap(year) else 365
        for year in range(years[0], years[1] + 1)
    )


def require_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        count = int((~torch.isfinite(tensor)).sum().item())
        raise ValueError(f"{name} contains {count} non-finite values")


def validate_batch(
    batch: dict,
    *,
    batch_size: int,
    crop: int,
    condition_channels: int,
) -> None:
    expected = {
        "x1": (batch_size, 1, crop, crop),
        "cond": (batch_size, condition_channels, crop, crop),
        "mask": (batch_size, 1, crop, crop),
        "target_mm": (batch_size, 1, crop, crop),
    }
    for name, shape in expected.items():
        if tuple(batch[name].shape) != shape:
            raise ValueError(
                f"real-data batch {name} shape {tuple(batch[name].shape)}, "
                f"expected {shape}"
            )
        require_finite(name, batch[name])
    if not torch.all((batch["mask"] == 0) | (batch["mask"] == 1)):
        raise ValueError("land mask contains values other than zero and one")
    if not torch.any(batch["mask"] > 0):
        raise ValueError("real-data batch contains no valid land pixels")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_h100.yaml")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument(
        "--out",
        default="data/processed/training_preflight.json",
    )
    parser.add_argument(
        "--normalization-report",
        default="data/processed/normalization_diagnostics.json",
    )
    parser.add_argument(
        "--strict-commit",
        action="store_true",
        help="fail if the repository has moved since the normalization "
             "diagnostics ran. Off by default: a code change does not "
             "invalidate a data diagnostic, and pinning the commit mostly "
             "blocked submissions without catching anything.",
    )
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("training preflight requires an allocated CUDA GPU")

    config_path = Path(args.config).resolve()
    repository = config_path.parents[1]
    config = load_cfg(str(config_path))
    stats_path = (repository / config["data"]["stats"]).resolve()
    zarr_path = (repository / config["data"]["zarr"]).resolve()
    stats = json.loads(stats_path.read_text())
    normalization_report_path = (
        repository / args.normalization_report
    ).resolve()
    normalization_report = json.loads(normalization_report_path.read_text())
    current_commit = git_commit(repository)
    if normalization_report.get("passed") is not True:
        raise ValueError("normalization diagnostic report did not pass")
    if normalization_report.get("git_commit") != current_commit:
        message = (
            f"repository moved since the normalization diagnostics "
            f"({normalization_report.get('git_commit', '?')[:8]} -> "
            f"{(current_commit or '?')[:8]})"
        )
        if args.strict_commit:
            raise ValueError(message + "; rerun them or drop --strict-commit")
        print(f"WARNING: {message}; continuing (--strict-commit not set)")
    # The statistics checksum is a DATA check, not a code one: if the statistics
    # were recomputed, the diagnostic figure describes different numbers than the
    # ones training will load.  That is worth failing on.
    if normalization_report.get("stats_sha256") != file_sha256(stats_path):
        raise ValueError(
            f"{stats_path.name} changed after the normalization diagnostics ran; "
            f"rerun them so the diagnostic describes the statistics in use"
        )
    normalization_figure_path = (
        repository / normalization_report["figure"]
    ).resolve()
    if not normalization_figure_path.is_file():
        raise FileNotFoundError(
            f"normalization diagnostic figure not found: "
            f"{normalization_figure_path}"
        )
    if normalization_report.get("figure_sha256") != file_sha256(
        normalization_figure_path
    ):
        raise ValueError("normalization diagnostic figure checksum changed")

    packed_channels = list(stats["cond_channels"])
    selected_channels = list(
        config["data"].get("cond_channels") or packed_channels
    )
    missing_channels = sorted(set(selected_channels) - set(packed_channels))
    if missing_channels:
        raise ValueError(
            f"configured condition channels are absent from statistics: "
            f"{missing_channels}"
        )
    dynamic_channels = len(selected_channels)
    static_channels = len(stats["static_channels"])
    seasonal_channels = (
        EXPECTED_SEASONAL_CHANNELS
        if config["data"].get("seasonal_encoding", True)
        else 0
    )
    if static_channels != EXPECTED_STATIC_CHANNELS:
        raise ValueError(
            f"expected {EXPECTED_STATIC_CHANNELS} static channels, found "
            f"{static_channels}: {stats['static_channels']}"
        )
    expected_conditions = dynamic_channels + static_channels + seasonal_channels

    torch.manual_seed(config["train"]["seed"])
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    train_dataset = build_dataset(config, "train")
    validation_dataset = build_dataset(config, "val")
    train_years = tuple(config["data"]["years"]["train"])
    validation_years = tuple(config["data"]["years"]["val"])
    if len(train_dataset) != expected_days(train_years):
        raise ValueError(
            f"training split has {len(train_dataset)} days, expected "
            f"{expected_days(train_years)}"
        )
    if len(validation_dataset) != expected_days(validation_years):
        raise ValueError(
            f"validation split has {len(validation_dataset)} days, expected "
            f"{expected_days(validation_years)}"
        )
    if train_dataset.total_cond_channels != expected_conditions:
        raise ValueError(
            f"dataset exposes {train_dataset.total_cond_channels} condition "
            f"channels, expected {expected_conditions}"
        )

    batch_size = int(config["train"]["batch_size"])
    workers = int(config["train"]["num_workers"])
    crop = int(config["data"]["crop"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=workers > 0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    model = UNet(
        in_channels=1,
        cond_channels=train_dataset.total_cond_channels,
        out_channels=1,
        image_size=crop,
        **config["model"],
    ).to(device)
    ema = EMA(model, decay=config["train"]["ema_decay"])
    flow = RectifiedFlow()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
        betas=(0.9, 0.999),
    )
    dtype = amp_dtype(device)
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))

    print(
        f"real data: train={len(train_dataset)} days, "
        f"validation={len(validation_dataset)} days",
        flush=True,
    )
    print(
        f"channels: dynamic={dynamic_channels} {selected_channels}, "
        f"static={static_channels}, "
        f"seasonal={seasonal_channels}, total={expected_conditions}",
        flush=True,
    )
    print(
        f"batch: {batch_size}, crop: {crop}x{crop}, workers: {workers}",
        flush=True,
    )
    print(f"model parameters: {model.num_parameters / 1e6:.1f} M", flush=True)
    print(f"autocast dtype: {dtype}", flush=True)

    losses = []
    gradient_norms = []
    model.train()
    iterator = iter(train_loader)
    for step in range(1, args.steps + 1):
        batch = next(iterator)
        validate_batch(
            batch,
            batch_size=batch_size,
            crop=crop,
            condition_channels=expected_conditions,
        )
        x1 = batch["x1"].to(device, non_blocking=True)
        condition = batch["cond"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=dtype):
            loss = flow_matching_loss(
                model,
                x1,
                condition,
                flow,
                mask=mask,
                cond_dropout=config["train"]["cond_dropout"],
                logit_normal_t=config["train"].get("logit_normal_t", True),
            )
        require_finite("training loss", loss)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config["train"]["grad_clip"],
        )
        require_finite("gradient norm", gradient_norm)
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)
        torch.cuda.synchronize(device)
        losses.append(float(loss.item()))
        gradient_norms.append(float(gradient_norm.item()))
        print(
            f"preflight step {step}/{args.steps}: loss={losses[-1]:.6f}, "
            f"gradient_norm={gradient_norms[-1]:.6f}",
            flush=True,
        )

    model.eval()
    validation_batch = next(iter(validation_loader))
    validate_batch(
        validation_batch,
        batch_size=batch_size,
        crop=crop,
        condition_channels=expected_conditions,
    )
    with torch.no_grad():
        x1 = validation_batch["x1"].to(device, non_blocking=True)
        condition = validation_batch["cond"].to(device, non_blocking=True)
        mask = validation_batch["mask"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=dtype):
            validation_loss = flow_matching_loss(
                model,
                x1,
                condition,
                flow,
                mask=mask,
                cond_dropout=0.0,
                logit_normal_t=config["train"].get("logit_normal_t", True),
            )
    require_finite("validation loss", validation_loss)
    torch.cuda.synchronize(device)

    gib = 1024**3
    peak_allocated = torch.cuda.max_memory_allocated(device) / gib
    peak_reserved = torch.cuda.max_memory_reserved(device) / gib
    report = {
        "passed": True,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": current_commit,
        "config": str(config_path.relative_to(repository)),
        "config_sha256": file_sha256(config_path),
        "stats": str(stats_path.relative_to(repository)),
        "stats_sha256": file_sha256(stats_path),
        "normalization_report": str(
            normalization_report_path.relative_to(repository)
        ),
        "normalization_report_sha256": file_sha256(
            normalization_report_path
        ),
        "normalization_figure": str(
            normalization_figure_path.relative_to(repository)
        ),
        "normalization_figure_sha256": file_sha256(
            normalization_figure_path
        ),
        "zarr": str(zarr_path.relative_to(repository)),
        "gpu": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "autocast_dtype": str(dtype),
        "train_days": len(train_dataset),
        "validation_days": len(validation_dataset),
        "batch_size": batch_size,
        "crop": crop,
        "condition_channels": expected_conditions,
        "model_parameters": model.num_parameters,
        "steps": args.steps,
        "training_losses": losses,
        "gradient_norms": gradient_norms,
        "validation_loss": float(validation_loss.item()),
        "peak_memory_allocated_gib": round(peak_allocated, 3),
        "peak_memory_reserved_gib": round(peak_reserved, 3),
    }
    output = (repository / args.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    partial.write_text(json.dumps(report, indent=2) + "\n")
    partial.replace(output)

    print(
        f"validation loss={report['validation_loss']:.6f}; "
        f"peak allocated={peak_allocated:.2f} GiB, "
        f"peak reserved={peak_reserved:.2f} GiB",
        flush=True,
    )
    print(f"PREFLIGHT PASSED; wrote {output}", flush=True)


if __name__ == "__main__":
    main()
