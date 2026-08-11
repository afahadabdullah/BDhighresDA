"""Physical-space observation operators used by the CHIRPS OSSE."""

from __future__ import annotations

import numpy as np
import torch

import pytest

from bdhires.da import (
    BilinearObsOperator,
    PhysicalBilinearObsOperator,
    PhysicalBlockAverageObsOperator,
)
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


def test_block_average_supports_common_crop_for_half_degree_footprints():
    transform = PrecipTransform(kind="log1p", eps=0.1)
    physical = torch.arange(128 * 128, dtype=torch.float32).reshape(1, 1, 128, 128)
    operator = PhysicalBlockAverageObsOperator(
        10, transform, crop=(4, 124, 8, 128)
    )

    actual = transform.inverse(operator(transform.forward(physical))).reshape(12, 12)
    expected = physical[..., 4:124, 8:128].reshape(1, 1, 12, 10, 12, 10).mean((3, 5))[0, 0]

    assert actual.shape == (12, 12)
    assert torch.allclose(actual, expected, atol=2e-3)


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


def test_physical_operators_mask_nonfinite_ocean_before_inverse_transform():
    """A masked NaN must not poison a finite likelihood gradient."""
    grid = Grid("tiny", lon_min=90.0, lat_min=20.0, nlon=4, nlat=4, res=0.05)
    transform = PrecipTransform(kind="log1p", eps=0.1, mu=1.2, sd=0.8)
    valid = np.zeros((4, 4), dtype=np.float32)
    valid[:, :2] = 1.0
    physical = torch.full((1, 1, 4, 4), 8.0)
    transformed = transform.forward(physical)
    transformed[..., :, 2:] = float("nan")
    transformed.requires_grad_(True)

    block = PhysicalBlockAverageObsOperator(2, transform, valid=valid)
    block_values = block(transformed)
    assert torch.isfinite(block_values).all()
    assert torch.allclose(
        block_values[0, 0, 0], transform.forward(torch.tensor(8.0)), atol=1e-6
    )

    lat = np.array([(grid.lat[1] + grid.lat[2]) / 2])
    lon = np.array([(grid.lon[0] + grid.lon[1]) / 2])
    gauge = PhysicalBilinearObsOperator(
        grid, lat, lon, transform, valid=valid
    )
    objective = block_values[0, 0, 0] + gauge(transformed).sum()
    objective.backward()
    assert torch.isfinite(transformed.grad).all()
    assert torch.count_nonzero(transformed.grad[..., :, 2:]) == 0


def test_coarse_footprint_grid_admits_a_station_inside_the_domain():
    """A station can sit inside the domain but outside the box of cell centres.

    The 0.4 deg CPC footprint grid spans exactly the same box as the 0.05 deg
    fine grid it was averaged from (top edge 26.7N), but its outermost centres
    are inset by half a footprint, putting the last one at 26.5N.  Tetulia at
    26.583N is therefore 0.21 footprints past the last centre while still being
    0.12 deg inside the domain.  Rejecting it would discard a real gauge for a
    coordinate-convention reason; the correct read of a box average there is the
    nearest footprint, which is what clamping gives.
    """
    coarse = Grid("cpc_bd", lon_min=87.6, lat_min=20.3, nlon=16, nlat=16, res=0.4)
    lat = np.array([26.58333])
    lon = np.array([88.55])

    with pytest.raises(ValueError, match="fall outside grid"):
        BilinearObsOperator(coarse, lat, lon)

    operator = BilinearObsOperator(coarse, lat, lon, tolerance_cells=0.5)
    field = torch.arange(16 * 16, dtype=torch.float32).reshape(1, 1, 16, 16)
    sampled = operator(field)

    assert torch.isfinite(sampled).all()
    # Clamped to the northern edge row, so it reads the last row of footprints.
    assert sampled[0, 0, 0] >= field[0, 0, -1].min()


def test_station_genuinely_outside_the_domain_still_raises():
    """Tolerance must not turn a mislocated station into a silent edge read."""
    coarse = Grid("cpc_bd", lon_min=87.6, lat_min=20.3, nlon=16, nlat=16, res=0.4)

    with pytest.raises(ValueError, match="fall outside grid"):
        BilinearObsOperator(
            coarse, np.array([31.0]), np.array([88.55]), tolerance_cells=0.5
        )


def test_fine_grid_keeps_a_strict_check_by_default():
    """0.5 cells on the 0.05 deg grid is 2.7 km -- harmless, but not the default.

    Gauge assimilation runs on the fine grid, where a station outside the domain
    is a data error worth failing loudly on rather than clamping to a coast.
    """
    fine = Grid("bd", lon_min=87.6, lat_min=20.3, nlon=128, nlat=128, res=0.05)

    with pytest.raises(ValueError, match="fall outside grid"):
        BilinearObsOperator(fine, np.array([26.69]), np.array([88.55]))

    operator = BilinearObsOperator(
        fine, np.array([26.69]), np.array([88.55]), tolerance_cells=0.5
    )
    assert operator.n_stations == 1
