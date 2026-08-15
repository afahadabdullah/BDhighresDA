#!/usr/bin/env python3
"""Download and aggregate Copernicus GLO-90 elevation over the WIDE domain.

The source is the credential-free Copernicus DEM 2021 release in the AWS Open
Data Registry. Individual one-degree Cloud Optimized GeoTIFF tiles are
downloaded, validated and averaged directly onto the selected project
0.05-degree wide grid. Missing tiles are ocean and are assigned zero elevation.

The compact regional NetCDF is the only retained output by default:

    python scripts/03_download_dem.py \
        --out data/raw/dem/copernicus_glo90_wide.nc

Use ``--keep-tiles`` only when the original ~90 m tiles are needed for another
application. The output is resumable and atomically published after validation.
"""
from __future__ import annotations

import argparse
import math
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bdhires.grids import Grid, WIDE, get_grid  # noqa: E402

BASE = "https://copernicus-dem-90m.s3.amazonaws.com"
PRODUCT = "Copernicus_DSM_COG_30"
SOURCE_RESOLUTION = "3 arc-seconds (approximately 90 m)"


def tile_stem(latitude: int, longitude: int) -> str:
    """Return the Copernicus tile stem for a lower-left integer coordinate."""
    ns = "N" if latitude >= 0 else "S"
    ew = "E" if longitude >= 0 else "W"
    return (
        f"{PRODUCT}_{ns}{abs(latitude):02d}_00_"
        f"{ew}{abs(longitude):03d}_00_DEM"
    )


def tile_url(stem: str) -> str:
    return f"{BASE}/{stem}/{stem}.tif"


def candidate_tiles(grid: Grid = WIDE) -> list[tuple[str, str]]:
    """Return (stem, URL) pairs intersecting the selected grid."""
    west, south, east, north = grid.bbox
    tiles = []
    for latitude in range(math.floor(south), math.ceil(north)):
        for longitude in range(math.floor(west), math.ceil(east)):
            stem = tile_stem(latitude, longitude)
            tiles.append((stem, tile_url(stem)))
    return tiles


def validate_tile(path: Path) -> None:
    import rasterio

    with rasterio.open(path) as source:
        if source.count != 1:
            raise ValueError(f"{path} has {source.count} bands, expected one")
        if source.width < 100 or source.height < 100:
            raise ValueError(f"{path} has unexpected shape {source.shape}")
        if source.crs is None or source.crs.to_epsg() != 4326:
            raise ValueError(f"{path} has unexpected CRS {source.crs}")
        sample = source.read(
            1,
            out_shape=(1, min(32, source.height), min(32, source.width)),
            masked=True,
        )
        if sample.count() == 0:
            raise ValueError(f"{path} contains no valid elevation values")


def fetch_tile(stem: str, url: str, tile_dir: Path) -> Path | None:
    """Download one tile, returning None for an absent all-ocean tile."""
    target = tile_dir / f"{stem}.tif"
    partial = target.with_suffix(target.suffix + ".part")

    if target.exists():
        try:
            validate_tile(target)
            print("already complete", target.name, flush=True)
            return target
        except (OSError, ValueError):
            target.unlink()
    partial.unlink(missing_ok=True)

    request = Request(url, headers={"User-Agent": "BDhighresDA/0.1"})
    for attempt in range(1, 4):
        try:
            with urlopen(request, timeout=120) as response, partial.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
            validate_tile(partial)
            partial.replace(target)
            print("downloaded", target.name, flush=True)
            return target
        except HTTPError as exc:
            partial.unlink(missing_ok=True)
            if exc.code == 404:
                print("ocean tile (not stored)", stem, flush=True)
                return None
            if attempt == 3:
                raise
            print(f"retrying {stem} after HTTP {exc.code}", flush=True)
        except (OSError, URLError, ValueError) as exc:
            partial.unlink(missing_ok=True)
            if attempt == 3:
                raise RuntimeError(f"failed to download {url}") from exc
            print(f"retrying {stem}: {exc}", flush=True)
        time.sleep(5 * attempt)
    raise AssertionError("unreachable")


def validate_dem(path: Path, grid: Grid = WIDE) -> None:
    with xr.open_dataarray(path) as elevation:
        if elevation.dims != ("lat", "lon"):
            raise ValueError(f"{path} has unexpected dimensions {elevation.dims}")
        if elevation.shape != grid.shape:
            raise ValueError(
                f"{path} has shape {elevation.shape}, expected {grid.shape}"
            )
        np.testing.assert_allclose(elevation.lat.values, grid.lat, atol=1e-6)
        np.testing.assert_allclose(elevation.lon.values, grid.lon, atol=1e-6)
        values = elevation.values
        if not np.isfinite(values).all():
            raise ValueError(f"{path} contains non-finite elevation values")
        minimum = float(values.min())
        maximum = float(values.max())
        if minimum < -500 or maximum < 1000 or maximum > 10000:
            raise ValueError(
                f"{path} has implausible elevation range {minimum}..{maximum} m"
            )


def aggregate_tiles(paths: list[Path], output: Path, grid: Grid = WIDE) -> None:
    """Average source tiles onto the grid and atomically write a NetCDF."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.merge import merge

    west, south, east, north = grid.bbox
    sources = [rasterio.open(path) for path in paths]
    try:
        mosaic, _ = merge(
            sources,
            bounds=(west, south, east, north),
            res=(grid.res, grid.res),
            nodata=0.0,
            dtype="float32",
            resampling=Resampling.average,
            target_aligned_pixels=True,
        )
    finally:
        for source in sources:
            source.close()

    values = np.nan_to_num(mosaic[0], nan=0.0, posinf=0.0, neginf=0.0)
    values = np.flipud(values).astype("float32")  # project latitude is ascending
    if values.shape != grid.shape:
        raise ValueError(
            f"resampled DEM has shape {values.shape}, expected {grid.shape}"
        )

    elevation = xr.DataArray(
        values,
        dims=("lat", "lon"),
        coords={"lat": grid.lat, "lon": grid.lon},
        name="elevation",
        attrs={
            "long_name": "mean surface elevation within each 0.05 degree cell",
            "units": "m",
            "source": "Copernicus DEM GLO-90, 2021 release",
            "source_resolution": SOURCE_RESOLUTION,
            "aggregation": f"area-average resampling to the BDhighresDA {grid.name} grid",
            "license_notice": (
                "Contains modified Copernicus DEM data (2021), accessed from "
                "the AWS Open Data Registry."
            ),
        },
    )
    elevation.lat.attrs.update(units="degrees_north")
    elevation.lon.attrs.update(units="degrees_east")

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    partial.unlink(missing_ok=True)
    elevation.to_netcdf(
        partial,
        engine="netcdf4",
        encoding={
            "elevation": {
                "dtype": "float32",
                "zlib": True,
                "complevel": 4,
                "shuffle": True,
            }
        },
    )
    validate_dem(partial, grid)
    partial.replace(output)
    print(
        f"wrote {output} {values.shape}; "
        f"range={values.min():.1f}..{values.max():.1f} m",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="data/raw/dem/copernicus_glo90_wide.nc",
    )
    parser.add_argument("--tile-dir", default="data/raw/dem/copernicus_glo90_tiles")
    parser.add_argument(
        "--grid", default="wide", choices=("wide", "wide_cpc"),
        help="use wide_cpc for the V3-SG CPC-edge-aligned domain",
    )
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--keep-tiles", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.jobs < 1:
        parser.error("--jobs must be positive")
    grid = get_grid(args.grid)

    output = Path(args.out)
    if output.exists():
        try:
            validate_dem(output, grid)
            print("already complete", output)
            return
        except (OSError, ValueError, AssertionError) as exc:
            print(f"removing invalid DEM {output}: {exc}", flush=True)
            output.unlink()

    tiles = candidate_tiles(grid)
    print(f"{grid.name} bounds: {grid.bbox}; {len(tiles)} candidate one-degree tiles")
    if args.dry_run:
        for _, url in tiles:
            print(url)
        return

    tile_dir = Path(args.tile_dir)
    tile_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(fetch_tile, stem, url, tile_dir): stem
            for stem, url in tiles
        }
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                paths.append(result)

    # The WIDE box contains extensive land across Bangladesh, India and
    # Myanmar. A very small count therefore indicates a naming/service failure,
    # not a legitimately ocean-dominated domain.
    if len(paths) < 60:
        raise RuntimeError(
            f"only {len(paths)} Copernicus DEM land tiles were available; "
            "expected at least 60 for the WIDE domain"
        )
    paths.sort()
    print(f"aggregating {len(paths)} land tiles", flush=True)
    aggregate_tiles(paths, output, grid)

    if not args.keep_tiles:
        for path in paths:
            path.unlink()
        try:
            tile_dir.rmdir()
        except OSError:
            pass
        print("removed source tiles; use --keep-tiles to retain them", flush=True)


if __name__ == "__main__":
    main()
