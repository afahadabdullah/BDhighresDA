#!/usr/bin/env python
"""Fail fast unless the CPC-v2 all-dry observation gradient is finite."""

from __future__ import annotations

import numpy as np
import torch

from bdhires.da import (
    GuidanceConfig,
    PhysicalBilinearObsOperator,
    PhysicalBlockAverageObsOperator,
    obs_log_likelihood,
)
from bdhires.grids import Grid
from bdhires.transforms import PrecipTransform


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = PrecipTransform(kind="sqrt", mu=0.4, sd=1.3)
    grid = Grid("sqrt_gradient_smoke", 90.0, 20.0, 4, 4, 0.05)
    valid = np.ones((4, 4), dtype=np.float32)
    lat = np.array([(grid.lat[1] + grid.lat[2]) / 2])
    lon = np.array([(grid.lon[1] + grid.lon[2]) / 2])

    state = torch.full(
        (1, 1, 4, 4), -10.0, device=device, requires_grad=True
    )
    block = PhysicalBlockAverageObsOperator(
        2, transform, valid=valid
    ).to(device)
    gauge = PhysicalBilinearObsOperator(
        grid, lat, lon, transform, valid=valid
    ).to(device)
    predicted = torch.cat([block(state), gauge(state)], dim=-1)
    dry_value = transform.forward(torch.tensor(0.0, device=device))
    if not torch.all(predicted == dry_value).item():
        raise AssertionError("sqrt DA smoke check changed the exact dry forward value")

    wet_observation = transform.forward(torch.full_like(predicted, 4.0))
    variance = torch.full_like(predicted, 0.25)
    time = torch.tensor([0.5], device=device)
    likelihood = obs_log_likelihood(
        wet_observation,
        predicted,
        variance,
        time,
        GuidanceConfig(gamma=1.0e-3),
    )
    likelihood.sum().backward()
    if state.grad is None or not torch.isfinite(state.grad).all().item():
        raise FloatingPointError("sqrt DA smoke check produced a non-finite gradient")
    if torch.count_nonzero(state.grad).item() != 0:
        raise AssertionError("sqrt DA smoke check did not select the zero dry subgradient")

    print(f"[preflight] sqrt dry-observation gradient: finite on {device}", flush=True)


if __name__ == "__main__":
    main()
