"""Single architecture boundary for training and checkpoint-based inference."""

from __future__ import annotations

from copy import deepcopy

from .flow import select_weights
from .graphflow_unet import GraphFlowUNet
from .unet import UNet


def build_model(*, architecture: str = "unet", graph: dict | None = None, **kwargs) -> UNet:
    """Missing architecture means the legacy U-Net; inconsistent configs fail."""
    if architecture == "unet":
        if graph is not None:
            raise ValueError("graph settings require architecture: graphflow_unet")
        return UNet(**kwargs)
    if architecture == "graphflow_unet":
        return GraphFlowUNet(graph=graph, **kwargs)
    raise ValueError(f"unknown velocity architecture {architecture!r}")


def model_metadata(model: UNet) -> dict:
    """Complete constructor metadata, independent of datasets and device caches."""
    result = {key: getattr(model, key) for key in (
        "in_channels", "cond_channels", "out_channels", "base_channels",
        "channel_mult", "num_res_blocks", "attn_resolutions", "dropout",
        "image_size", "num_heads", "multiscale_conditioning",
    )}
    result["architecture"] = "graphflow_unet" if isinstance(model, GraphFlowUNet) else "unet"
    if isinstance(model, GraphFlowUNet):
        result["graph"] = deepcopy(model.graph_config)
    return result


def model_from_checkpoint(
    checkpoint: dict, *, cond_channels: int | None = None,
    image_size: int | None = None, device="cpu",
) -> UNet:
    """Strictly load selected online/EMA weights, defaulting old checkpoints to UNet.

    New checkpoints freeze construction-time image_size (attention placement).
    Forward still accepts other pyramid-compatible canvases. Legacy callers may
    supply their historical construction size to reproduce archived behavior.
    """
    if "model_config" in checkpoint:
        settings = deepcopy(checkpoint["model_config"])
        if cond_channels is not None and cond_channels != settings["cond_channels"]:
            raise ValueError("checkpoint conditioning channel count does not match dataset")
    else:
        cfg = checkpoint["cfg"]
        settings = deepcopy(cfg["model"])
        weights = select_weights(checkpoint)
        in_channels = settings.setdefault("in_channels", 1)
        settings.setdefault("cond_channels", weights["in_conv.weight"].shape[1] - in_channels)
        if cond_channels is not None and cond_channels != settings["cond_channels"]:
            raise ValueError("checkpoint conditioning channel count does not match dataset")
        settings.setdefault("out_channels", weights["out_conv.weight"].shape[0])
        settings.setdefault("image_size", image_size if image_size is not None else cfg["data"]["crop"])
    model = build_model(**settings)
    model.load_state_dict(select_weights(checkpoint), strict=True)
    return model.to(device).eval()
