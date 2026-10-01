"""Strict country masks and geographic plotting, independent of model code.

Grid scores use cell-centre inclusion intersected with model validity.
Station inclusion uses the same polygon with numerical boundary tolerance
only (1e-9 degrees), never a kilometre-scale country buffer. Holes and
disjoint polygons are preserved. Raster maps are additionally vector-clipped
so boundary cells cannot paint colour outside the country.
"""
from pathlib import Path
import hashlib
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BOUNDARY = ROOT / "configs/geography/geoBoundaries-BGD-ADM0.geojson"


def geometry(payload):
    kind = payload.get("type")
    if kind == "FeatureCollection":
        features = payload.get("features", [])
        if len(features) != 1:
            raise ValueError("country ADM0 must contain exactly one feature")
        return geometry(features[0])
    if kind == "Feature":
        return geometry(payload["geometry"])
    if kind not in ("Polygon", "MultiPolygon"):
        raise ValueError(f"expected Polygon/MultiPolygon, got {kind}")
    for polygon in parts(payload):
        if not polygon:
            raise ValueError("empty country polygon")
        for ring in polygon:
            if ring.ndim != 2 or ring.shape[1] != 2 or len(ring) < 4 or not np.isfinite(ring).all() or not np.array_equal(ring[0], ring[-1]):
                raise ValueError("country rings must be finite closed longitude/latitude coordinates")
    return payload


def parts(country):
    coords = country["coordinates"]
    return [[np.asarray(ring, float) for ring in polygon]
            for polygon in ([coords] if country["type"] == "Polygon" else coords)]


def read_boundary(path=DEFAULT_BOUNDARY):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"country boundary missing: {path}; use scripts/80_fetch_bangladesh_boundary.py")
    country = geometry(json.loads(path.read_text()))
    return country, {"name": "Bangladesh ADM0", "path": str(path.resolve()),
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "grid_rule": "cell centre inside country intersected with model validity",
                     "station_rule": "inside country; numerical boundary tolerance 1e-9 degrees",
                     "outside_map_colour": "white"}


def points_inside(lat, lon, country):
    from matplotlib.path import Path as MplPath
    lat, lon = np.broadcast_arrays(np.asarray(lat, float), np.asarray(lon, float))
    points = np.column_stack([lon.ravel(), lat.ravel()])
    finite = np.isfinite(points).all(axis=1)
    inside = np.zeros(len(points), bool)
    for polygon in parts(country):
        selected = MplPath(polygon[0]).contains_points(points, radius=1e-9)
        for hole in polygon[1:]:
            selected &= ~MplPath(hole).contains_points(points, radius=-1e-9)
        inside |= selected
    return (inside & finite).reshape(lat.shape)


def grid_mask(lat, lon, country):
    xx, yy = np.meshgrid(lon, lat)
    return points_inside(yy, xx, country)


def bounds(country, padding=.15):
    points = np.concatenate([ring for polygon in parts(country) for ring in polygon])
    return (points[:, 0].min() - padding, points[:, 0].max() + padding,
            points[:, 1].min() - padding, points[:, 1].max() + padding)


def compound_path(country):
    from matplotlib.path import Path as MplPath
    vertices, codes = [], []
    for polygon in parts(country):
        for i, ring in enumerate(polygon):
            # Nonzero winding: outer ring counterclockwise, holes clockwise.
            signed = np.sum(ring[:-1, 0] * ring[1:, 1] - ring[1:, 0] * ring[:-1, 1])
            ring = ring[::-1] if (signed > 0) != (i == 0) else ring
            vertices.extend(ring.tolist())
            codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(ring) - 2) + [MplPath.CLOSEPOLY])
    return MplPath(vertices, codes)


def map_axes(ax, country):
    for polygon in parts(country):
        for ring in polygon:
            ax.plot(ring[:, 0], ring[:, 1], color="#202020", lw=.65, zorder=6)
    lo, hi, la, ha = bounds(country)
    ax.set(xlim=(lo, hi), ylim=(la, ha), facecolor="white")
    ax.set_aspect(1 / np.cos(np.radians((la + ha) / 2)))
    ax.grid(False)


def map_layer(ax, field, lat, lon, country, **kwargs):
    from matplotlib import pyplot as plt
    from matplotlib.patches import PathPatch
    cmap = kwargs.pop("cmap", "YlGnBu")
    cmap = plt.get_cmap(cmap).copy() if isinstance(cmap, str) else cmap.copy()
    cmap.set_bad("white")
    lat, lon = np.asarray(lat), np.asarray(lon)
    if len(lat) < 2 or len(lon) < 2:
        raise ValueError("country raster needs at least two coordinates per axis")
    dlat, dlon = np.median(np.diff(lat)), np.median(np.diff(lon))
    extent = [lon[0] - dlon / 2, lon[-1] + dlon / 2, lat[0] - dlat / 2, lat[-1] + dlat / 2]
    layer = np.where(grid_mask(lat, lon, country), field, np.nan)
    artist = ax.imshow(layer, extent=extent, origin="lower", interpolation="nearest", cmap=cmap, **kwargs)
    artist.set_clip_path(PathPatch(compound_path(country), transform=ax.transData))
    map_axes(ax, country)
    return artist
