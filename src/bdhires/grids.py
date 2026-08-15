"""Domain / grid definitions for BDhighresDA.

Two nested domains are used:

* ``wide``  -- a 256 x 256 pretraining domain at 0.05 deg covering NE India,
  Bangladesh, Myanmar and the northern Bay of Bengal.  Training on random
  128 x 128 crops of this domain massively increases the effective number of
  training samples (the single biggest risk in this project is that ~16k daily
  fields is a small dataset for a generative model).
* ``bd``    -- the 128 x 128 evaluation/production domain at 0.05 deg centred
  on Bangladesh, including the Meghalaya/Shillong orographic hotspot and the
  Chittagong Hill Tracts.

All grids are cell-*centre* based and latitude is stored ASCENDING (row 0 is
the southernmost row).  Keep this convention everywhere -- the differentiable
observation operator in ``bdhires.da.observation`` depends on it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid:
    name: str
    lon_min: float
    lat_min: float
    nlon: int
    nlat: int
    res: float

    @property
    def lon(self) -> np.ndarray:
        return self.lon_min + self.res * (np.arange(self.nlon) + 0.5)

    @property
    def lat(self) -> np.ndarray:
        return self.lat_min + self.res * (np.arange(self.nlat) + 0.5)

    @property
    def lon_max(self) -> float:
        return self.lon_min + self.res * self.nlon

    @property
    def lat_max(self) -> float:
        return self.lat_min + self.res * self.nlat

    @property
    def shape(self) -> tuple[int, int]:
        return (self.nlat, self.nlon)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """(lon_min, lat_min, lon_max, lat_max) of the outer cell edges."""
        return (self.lon_min, self.lat_min, self.lon_max, self.lat_max)

    def slice_kwargs(self, pad: float = 0.0) -> dict:
        """Convenience for ``xarray.sel`` on a lat/lon dataset."""
        return dict(
            longitude=slice(self.lon_min - pad, self.lon_max + pad),
            latitude=slice(self.lat_min - pad, self.lat_max + pad),
        )


# 128 x 128 @ 0.05 deg = 6.4 deg box.  Bangladesh is 88.0-92.7E, 20.6-26.7N;
# the box below adds ~1 deg of margin on the north/east so the Meghalaya and
# Arakan orographic barriers that drive BD rainfall are inside the domain.
# Bangladesh spans 88.01-92.67E and 20.57-26.63N; this box clears all of it.
BD = Grid(name="bd", lon_min=87.6, lat_min=20.3, nlon=128, nlat=128, res=0.05)

# 256 x 256 @ 0.05 deg = 12.8 deg box, used for legacy pretraining with random crops.
WIDE = Grid(name="wide", lon_min=84.0, lat_min=16.0, nlon=256, nlat=256, res=0.05)

# V3-SG domains.  Their *edges* close on the native 0.5-degree CPC grid, so
# every coarse cell is exactly 10 x 10 fine cells.  They intentionally coexist
# with (rather than replace) the legacy grids: changing BD/WIDE would invalidate
# all V1/V2 checkpoints and comparisons.
BD_CPC = Grid(
    name="bd_cpc", lon_min=87.5, lat_min=20.0,
    nlon=130, nlat=140, res=0.05,
)
WIDE_CPC = Grid(
    name="wide_cpc", lon_min=84.0, lat_min=16.0,
    # Use the largest inward CPC-aligned square that is an exact subset of
    # the legacy 256 x 256 WIDE grid.  This preserves the complete 160-cell
    # production canvas while allowing the existing CHIRPS and DEM files to
    # be reused without interpolation or another download.
    nlon=240, nlat=240, res=0.05,
)

GRIDS = {g.name: g for g in (BD, WIDE, BD_CPC, WIDE_CPC)}


def get_grid(name: str) -> Grid:
    try:
        return GRIDS[name]
    except KeyError as exc:  # pragma: no cover
        raise KeyError(f"unknown grid {name!r}; choose from {sorted(GRIDS)}") from exc


def crop_offsets(outer: Grid, inner: Grid) -> tuple[int, int]:
    """Row/col offset of ``inner`` inside ``outer`` (both must share ``res``)."""
    if abs(outer.res - inner.res) > 1e-9:
        raise ValueError("grids must share the same resolution")
    row = int(round((inner.lat_min - outer.lat_min) / outer.res))
    col = int(round((inner.lon_min - outer.lon_min) / outer.res))
    if row < 0 or col < 0:
        raise ValueError("inner grid is not contained in outer grid")
    if row + inner.nlat > outer.nlat or col + inner.nlon > outer.nlon:
        raise ValueError("inner grid is not contained in outer grid")
    return row, col
