"""Physical-space observation operators used by the CHIRPS OSSE."""

from __future__ import annotations

import numpy as np
import torch

from bdhires.da import PhysicalBilinearObsOperator, PhysicalBlockAverageObsOperator
from bdhires.grids import Grid
from bdhires.transforms import PrecipTransform


def test_block_average_is_taken_in_mm_before_transforming():
    transform = PrecipTransform(kind="log1p", eps=0.1, mu=1.2, sd=0.8)
    physical = torch.tensor([[[[0.0, 10.0], [20.0, 30.0]]]])
    transformed = transform.forward(physical)
    operator = PhysicalBlockAverageObsOperator(2, transform)

    actual = operator(transformed)[0, 0, 0]
    expected = transform.forward(physical.mean())

    assert torch.allclose(actual, expected)
    assert not torch.allclose(actual, transformed.mean())


def test_bilinear_station_interpolates_mm_before_transforming():
    grid = Grid("tiny", lon_min=90.0, lat_min=20.0, nlon=2, nlat=2, res=0.05)
    transform = PrecipTransform(kind="sqrt", mu=0.4, sd=1.3)
    physical = torch.tensor([[[[0.0, 4.0], [16.0, 36.0]]]])
    transformed = transform.forward(physical)
    lat = np.array([(grid.lat[0] + grid.lat[1]) / 2])
    lon = np.array([(grid.lon[0] + grid.lon[1]) / 2])
    operator = PhysicalBilinearObsOperator(grid, lat, lon, transform)

    actual = operator(transformed)[0, 0, 0]
    expected = transform.forward(physical.mean())

    assert torch.allclose(actual, expected, atol=1e-6)


def test_bilinear_station_zeros_masked_ocean_before_coastal_interpolation():
    grid = Grid("tiny", lon_min=90.0, lat_min=20.0, nlon=2, nlat=2, res=0.05)
    transform = PrecipTransform(kind="log1p", eps=0.1)
    # A residual checkpoint can decode a non-zero precipitation base over its
    # masked ocean cells. The physical observation operator must not let that
    # value contaminate a coastal gauge interpolation.
    physical = torch.full((1, 1, 2, 2), 8.0)
    valid = np.array([[1, 0], [1, 0]], dtype=np.float32)
    lat = np.array([(grid.lat[0] + grid.lat[1]) / 2])
    lon = np.array([(grid.lon[0] + grid.lon[1]) / 2])
    operator = PhysicalBilinearObsOperator(
        grid, lat, lon, transform, valid=valid
    )

    actual = operator(transform.forward(physical))[0, 0, 0]
    expected = transform.forward(torch.tensor(4.0))

    assert torch.allclose(actual, expected, atol=1e-6)
