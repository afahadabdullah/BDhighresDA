"""Minimal torchrun/DDP helpers (2x V100 or 1x H100)."""
from __future__ import annotations

import os

import torch
import torch.distributed as dist


def setup_distributed():
    """Return (rank, world_size, local_rank, device)."""
    if "RANK" not in os.environ:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return 0, 1, 0, dev
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    dist.init_process_group(backend="nccl", init_method="env://")
    return rank, world, local, torch.device("cuda", local)


def is_main() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def amp_dtype(device) -> torch.dtype:
    """V100 (sm_70) has no bf16 tensor cores -- fall back to fp16 + GradScaler."""
    if device.type != "cuda":
        return torch.float32
    major, _ = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16
