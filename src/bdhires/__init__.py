"""BDhighresDA: generative downscaling + data assimilation for daily rainfall over Bangladesh."""

__version__ = "0.1.0"

from .grids import BD, WIDE, Grid, get_grid  # noqa: F401
from .transforms import PrecipTransform  # noqa: F401
