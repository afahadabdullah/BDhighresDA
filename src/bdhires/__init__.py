"""BDhighresDA: generative downscaling + data assimilation for daily rainfall over Bangladesh."""

__version__ = "0.1.0"

from .grids import BD, WIDE, Grid, get_grid  # noqa: F401
# Station/grid preprocessing does not need the optional tensor backend.
# Retain the package-level API while loading it only when explicitly requested.
__all__ = ["BD", "WIDE", "Grid", "get_grid", "PrecipTransform"]


def __getattr__(name):
    if name == "PrecipTransform":
        from .transforms import PrecipTransform

        globals()[name] = PrecipTransform
        return PrecipTransform
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
