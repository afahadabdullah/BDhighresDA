"""SURMA-GraphFlow G0: an experimental residual augmentation of the CPCv2 U-Net."""

from __future__ import annotations

from .graph_processor import MultimeshProcessor
from .unet import UNet


class GraphFlowUNet(UNet):
    """Insert graph communication after encoder level 2, including its last skip.

    Time and meteorological conditioning arrive through the existing encoder.
    Graph features add only static geometry. The final velocity/hurdle interface
    and the existing attention layers are inherited without modification.
    """

    def __init__(self, *, graph: dict | None = None, **kwargs):
        super().__init__(**kwargs)
        settings = dict(
            enabled=True, level=2, hidden_dim=256, num_blocks=8,
            mesh_strides=[1, 2, 4, 8], diagonal_edges=True, aggregation="mean",
            zero_init_output=True, checkpoint_blocks=False,
        )
        unknown = set(graph or {}) - settings.keys()
        if unknown:
            raise ValueError(f"unknown graph configuration keys: {sorted(unknown)}")
        settings.update(graph or {})
        level = settings["level"]
        if type(level) is not int or not 0 <= level < len(self.channel_mult):
            raise ValueError("graph level must select an existing U-Net encoder level")
        if self.num_res_blocks < 1:
            raise ValueError("GraphFlow requires at least one ResBlock per encoder level")
        for key in ("enabled", "diagonal_edges", "zero_init_output", "checkpoint_blocks"):
            if type(settings[key]) is not bool:
                raise ValueError(f"graph.{key} must be a boolean")
        self.graph_config = settings
        self.graph_level = level
        processor_settings = {k: v for k, v in settings.items() if k not in {"enabled", "level"}}
        self.graph_processor = (
            MultimeshProcessor(self.base_channels * self.channel_mult[level], **processor_settings)
            if settings["enabled"] else None
        )

    def _process_encoder_level(self, h, level, emb, condition_features):
        if level == self.graph_level and self.graph_processor is not None:
            return self.graph_processor(h)
        return h
