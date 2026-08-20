"""Where stage A and stage B meet, solved rather than hardcoded.

V7's two stages were trained at different crop sizes -- stage A on 64 cells of
0.1 degree (6.4 degrees of ground), stage B on 120 cells of 0.05 degree (6.0
degrees).  Those are not the same window, so composing the stages at inference
means choosing one explicitly.

Two constraints make the choice, and both are hard:

* **Attention resolutions.** A flow U-Net is convolutional, so it will happily
  accept a differently sized input -- and silently stop applying attention,
  because ``attn_resolutions`` are matched against the actual level sizes.
  Stage A's ``[16, 32]`` land on 64 -> 32 -> 16 -> 8; stage B's ``[30, 60]``
  land on 120 -> 60 -> 30 -> 15.  Run either at the other's size and it becomes
  a different model with the same weights.  So each stage runs at ITS OWN
  training size and the windows nest.
* **The lattice.** ``WIDE`` (stage A) and ``WIDE_CPC`` (stage B) share the
  origin 84E/16N and the 0.1-degree lattice, so a coarse index means the same
  place in both, and a stage B fine index is exactly twice its coarse index.
  Nothing is interpolated between the stages; the interface is a crop.

The remaining freedom is *where*, and the answer is: centred on Bangladesh.
6.0 degrees of latitude against Bangladesh's 5.89 leaves ~0.05 degrees at each
end, so this is tight by construction and :func:`bangladesh_window` refuses
rather than silently clipping the country.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..grids import WIDE, WIDE_CPC, Grid, at_resolution

# Bangladesh's land extent, generous enough to include the offshore islands and
# the northern salient.  Used only to place and check the window.
BANGLADESH_LON = (88.01, 92.68)
BANGLADESH_LAT = (20.74, 26.63)


@dataclass(frozen=True)
class V7Window:
    """One geographic window, expressed on both stages' grids.

    ``meso_origin`` indexes stage A's 0.1-degree grid (``WIDE`` coarsened).
    ``coarse_origin`` indexes stage B's 0.1-degree coarse grid, and
    ``fine_origin`` its 0.05-degree fine grid.  All three are (row, column) into
    the FULL archive arrays, so they can be used as slice offsets directly.
    """

    meso_origin: tuple[int, int]
    meso_size: int
    coarse_origin: tuple[int, int]
    coarse_size: int
    fine_origin: tuple[int, int]
    fine_size: int

    @property
    def meso_local(self) -> tuple[int, int]:
        """Stage B's coarse window, as an offset INSIDE stage A's output."""
        return (
            self.coarse_origin[0] - self.meso_origin[0],
            self.coarse_origin[1] - self.meso_origin[1],
        )

    def fine_grid(self) -> Grid:
        return Grid(
            name="v7_product",
            lon_min=WIDE_CPC.lon_min + self.fine_origin[1] * WIDE_CPC.res,
            lat_min=WIDE_CPC.lat_min + self.fine_origin[0] * WIDE_CPC.res,
            nlon=self.fine_size,
            nlat=self.fine_size,
            res=WIDE_CPC.res,
        )

    def describe(self) -> str:
        g = self.fine_grid()
        return (
            f"product {g.nlat}x{g.nlon} @{g.res} deg  "
            f"lon {g.lon_min:.2f}-{g.lon_min + g.nlon * g.res:.2f}  "
            f"lat {g.lat_min:.2f}-{g.lat_min + g.nlat * g.res:.2f}   "
            f"stage A {self.meso_size}^2 @0.1 at {self.meso_origin}, "
            f"stage B coarse at {self.coarse_origin} "
            f"(local {self.meso_local})"
        )


def bangladesh_window(meso_size: int = 64, fine_size: int = 120) -> V7Window:
    """Place both stages' windows over Bangladesh, or refuse.

    ``fine_size`` is stage B's training crop and ``meso_size`` stage A's, both in
    their own cells.  The fine window is centred on the country; the meso window
    is then placed to contain it with the margin split as evenly as the lattice
    allows.
    """
    meso_grid = at_resolution(WIDE, 0.1)
    coarse_grid = at_resolution(WIDE_CPC, 0.1)
    if (meso_grid.lat_min, meso_grid.lon_min, meso_grid.res) != (
        coarse_grid.lat_min, coarse_grid.lon_min, coarse_grid.res
    ):
        raise ValueError(
            "stage A and stage B 0.1-degree grids do not share a lattice; the "
            "interface would be a regrid rather than a crop"
        )
    if fine_size % 2:
        raise ValueError("fine_size must be even so it maps to whole coarse cells")
    coarse_size = fine_size // 2
    if coarse_size > meso_size:
        raise ValueError(
            f"stage B needs {coarse_size} coarse cells but stage A only produces "
            f"{meso_size}; the stages cannot be composed at these crops"
        )

    res = WIDE_CPC.res
    span = fine_size * res

    def place(low: float, high: float, axis_min: float, extent_cells: int) -> int:
        """First EVEN fine index whose window is centred on [low, high]."""
        if span < (high - low):
            raise ValueError(
                f"a {span:g}-degree window cannot cover {high - low:.2f} degrees "
                f"of Bangladesh; raise fine_size"
            )
            # (unreachable, but states the failure in its own terms)
        centre = 0.5 * (low + high)
        start = (centre - 0.5 * span - axis_min) / res
        index = int(round(start / 2.0)) * 2          # even -> whole coarse cells
        index = max(0, min(index, extent_cells - fine_size))
        edge = axis_min + index * res
        if edge > low or edge + span < high:
            raise ValueError(
                f"window {edge:.2f}-{edge + span:.2f} does not cover "
                f"{low:.2f}-{high:.2f}; the archive does not extend far enough"
            )
        return index

    fine_row = place(BANGLADESH_LAT[0], BANGLADESH_LAT[1], WIDE_CPC.lat_min, WIDE_CPC.nlat)
    fine_col = place(BANGLADESH_LON[0], BANGLADESH_LON[1], WIDE_CPC.lon_min, WIDE_CPC.nlon)
    coarse_row, coarse_col = fine_row // 2, fine_col // 2

    def meso_start(coarse_start: int, extent: int) -> int:
        """Centre the meso window on the coarse one, then pull it into range."""
        start = coarse_start - (meso_size - coarse_size) // 2
        return max(0, min(start, extent - meso_size))

    meso_row = meso_start(coarse_row, meso_grid.nlat)
    meso_col = meso_start(coarse_col, meso_grid.nlon)

    window = V7Window(
        meso_origin=(meso_row, meso_col),
        meso_size=int(meso_size),
        coarse_origin=(coarse_row, coarse_col),
        coarse_size=int(coarse_size),
        fine_origin=(fine_row, fine_col),
        fine_size=int(fine_size),
    )

    # Containment, stated as an assertion rather than trusted: a negative local
    # offset would slice from the wrong end of stage A's output in numpy and
    # produce a plausible-looking field for the wrong place.
    local_row, local_col = window.meso_local
    for name, local in (("row", local_row), ("column", local_col)):
        if local < 0 or local + coarse_size > meso_size:
            raise ValueError(
                f"stage B's coarse window falls outside stage A's output on the "
                f"{name} axis (local offset {local}, size {coarse_size}, "
                f"stage A {meso_size})"
            )
    for name, origin, size, extent in (
        ("stage B coarse row", coarse_row, coarse_size, coarse_grid.nlat),
        ("stage B coarse column", coarse_col, coarse_size, coarse_grid.nlon),
        ("stage B fine row", fine_row, fine_size, WIDE_CPC.nlat),
        ("stage B fine column", fine_col, fine_size, WIDE_CPC.nlon),
        ("stage A row", meso_row, meso_size, meso_grid.nlat),
        ("stage A column", meso_col, meso_size, meso_grid.nlon),
    ):
        if origin < 0 or origin + size > extent:
            raise ValueError(f"{name} window {origin}..{origin + size} leaves the archive")
    return window
