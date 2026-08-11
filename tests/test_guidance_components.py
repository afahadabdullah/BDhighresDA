"""Focused tests for stream-specific simultaneous guidance."""

import torch

from bdhires.da import CompositeObsOperator, GuidanceConfig, guidance_grad
from bdhires.models import RectifiedFlow


class _ZeroVelocity(torch.nn.Module):
    def forward(self, x, t, cond):
        return x * 0.0


class _Pixel(torch.nn.Module):
    def __init__(self, row: int, column: int):
        super().__init__()
        self.row = row
        self.column = column

    def forward(self, x):
        return x[:, :, self.row, self.column].unsqueeze(-1)


def test_component_spread_blurs_gauge_but_not_satellite_gradient():
    gauge = _Pixel(5, 5)
    satellite = _Pixel(26, 26)
    operator = CompositeObsOperator(
        [gauge, satellite], component_spread_cells=[2.0, 0.0]
    )
    state = torch.zeros(1, 1, 32, 32)
    time = torch.full((1,), 0.5)
    observations = torch.ones(1, 1, 2)
    variance = torch.full((2,), 0.1)
    config = GuidanceConfig(gamma=1.0e-3, clip_norm=None)

    _, gradient = guidance_grad(
        state, time, _ZeroVelocity(), RectifiedFlow(), None,
        operator, observations, variance, config,
    )

    assert gradient[0, 0, 5, 6].abs() > 0.0  # gauge reaches a neighbour
    assert gradient[0, 0, 26, 26].abs() > 0.0  # satellite still acts
    assert gradient[0, 0, 26, 25] == 0.0  # satellite was not blurred
    assert torch.isfinite(gradient).all()


def test_composite_rejects_mismatched_component_spread_count():
    try:
        CompositeObsOperator([_Pixel(1, 1), _Pixel(2, 2)], [1.0])
    except ValueError as error:
        assert "one value per operator" in str(error)
    else:
        raise AssertionError("mismatched component spreads were accepted")


def test_missing_satellite_observation_produces_zero_finite_gradient():
    operator = CompositeObsOperator(
        [_Pixel(5, 5), _Pixel(26, 26)], component_spread_cells=[2.0, 0.0]
    )
    observations = torch.tensor([[[1.0, float("nan")]]])
    _, gradient = guidance_grad(
        torch.zeros(1, 1, 32, 32), torch.full((1,), 0.5),
        _ZeroVelocity(), RectifiedFlow(), None, operator, observations,
        torch.full((2,), 0.1), GuidanceConfig(gamma=1.0e-3, clip_norm=None),
    )
    assert torch.isfinite(gradient).all()
    assert gradient[0, 0, 26, 26] == 0.0
