#!/usr/bin/env python
"""Build the time-invariant conditioning channels on the 0.05 deg wide grid.

Channels
--------
0  elevation       (sqrt-scaled, then rescaled to [0, 1])
1  slope magnitude (|grad z|, standardised)  -- proxy for orographic forcing
2  land-sea mask   (from CHIRPS validity)
3  sin(pi * (lon - lon0) / dlon)   absolute-position encoding
4  cos(...)
5  sin(pi * (lat - lat0) / dlat)
6  cos(...)

Positional encodings matter here: we train on random crops of the wide domain,
so the network needs to know *where* a crop sits to reproduce location-specific
climatology (the Meghalaya maximum, the dry northwest).

    python scripts/03_build_static.py \
        --dem data/raw/dem/copernicus_glo90_wide.nc \
        --chirps data/raw/chirps/chirps_wide_2010.nc --out data/static/static_wide.nc

Use ``03_download_dem.py`` to create the recommended regional Copernicus DEM.
If no DEM is supplied, the script writes a zero-orography placeholder.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bdhires.grids import get_grid  # noqa: E402

CHANNELS = ["elev", "slope", "lsm", "sin_lon", "cos_lon", "sin_lat", "cos_lat"]


def regrid_dem(dem_path: str, grid) -> np.ndarray:
    path = Path(dem_path)
    if path.suffix.lower() in {".tif", ".tiff"}:
        import rioxarray

        da = rioxarray.open_rasterio(path).squeeze(drop=True)
    else:
        da = xr.open_dataarray(path).squeeze(drop=True)

    rename = {}
    for old, new in (
        ("latitude", "y"),
        ("lat", "y"),
        ("longitude", "x"),
        ("lon", "x"),
    ):
        if old in da.dims or old in da.coords:
            rename[old] = new
    da = da.rename(rename)
    if "x" not in da.coords or "y" not in da.coords:
        raise ValueError(f"{dem_path} must have geographic x/y or lon/lat coordinates")
    if bool((da.y[0] > da.y[-1]).item()):
        da = da.sortby("y")

    out = da.interp(y=grid.lat, x=grid.lon, method="linear")
    values = np.nan_to_num(out.values, nan=0.0).astype(np.float32)
    da.close()
    return values


def validate_static(path: Path, grid) -> None:
    with xr.open_dataset(path) as dataset:
        if dataset["static"].dims != ("channel", "lat", "lon"):
            raise ValueError(
                f"{path} has unexpected static dimensions "
                f"{dataset['static'].dims}"
            )
        if dataset["static"].shape != (len(CHANNELS), *grid.shape):
            raise ValueError(
                f"{path} has static shape {dataset['static'].shape}, expected "
                f"{(len(CHANNELS), *grid.shape)}"
            )
        if dataset["valid"].shape != grid.shape:
            raise ValueError(
                f"{path} has valid-mask shape {dataset['valid'].shape}, "
                f"expected {grid.shape}"
            )
        if list(map(str, dataset.channel.values)) != CHANNELS:
            raise ValueError(f"{path} has unexpected static channel names")
        np.testing.assert_allclose(dataset.lat.values, grid.lat, atol=1e-6)
        np.testing.assert_allclose(dataset.lon.values, grid.lon, atol=1e-6)
        if not np.isfinite(dataset["static"].values).all():
            raise ValueError(f"{path} contains non-finite static values")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dem", default=None)
    ap.add_argument("--chirps", required=True, help="any CHIRPS wide-domain year file")
    ap.add_argument("--out", default="data/static/static_wide.nc")
    ap.add_argument(
        "--grid", default="wide", choices=("wide", "wide_cpc"),
        help="use wide_cpc for the V3-SG CPC-edge-aligned domain",
    )
    args = ap.parse_args()

    grid = get_grid(args.grid)
    with xr.open_dataset(args.chirps) as ds:
        p = ds["precip"]
        lat_name = "latitude" if "latitude" in p.dims else "lat"
        lon_name = "longitude" if "longitude" in p.dims else "lon"
        p = p.interp({lat_name: grid.lat, lon_name: grid.lon}, method="nearest")
        valid = np.isfinite(p).any(dim="time").values.astype(np.float32)
    if not np.any(valid > 0.5):
        raise ValueError(f"{args.chirps} produced an empty CHIRPS land mask")

    if args.dem:
        elev = regrid_dem(args.dem, grid)
    else:
        print("no --dem supplied; using a zero orography placeholder. "
              "Run scripts/03_download_dem.py and rerun -- orography is the "
              "single most informative static channel for Bangladesh rainfall.")
        elev = np.zeros(grid.shape, np.float32)

    elev_s = np.sqrt(np.clip(elev, 0, None))
    elev_s = (elev_s - elev_s.min()) / (np.ptp(elev_s) + 1e-6)
    dy_m = grid.res * 111_320.0
    dx_m = grid.res * 111_320.0 * np.cos(np.deg2rad(grid.lat))[:, None]
    gy = np.gradient(elev, axis=0) / dy_m
    gx = np.gradient(elev, axis=1) / dx_m
    slope_raw = np.hypot(gy, gx)
    land = valid > 0.5
    slope_mean = slope_raw[land].mean()
    slope_std = slope_raw[land].std() + 1e-6
    slope = np.where(land, (slope_raw - slope_mean) / slope_std, 0.0)

    lon2, lat2 = np.meshgrid(grid.lon, grid.lat)
    u = (lon2 - grid.lon[0]) / (grid.lon[-1] - grid.lon[0])
    v = (lat2 - grid.lat[0]) / (grid.lat[-1] - grid.lat[0])
    pos = [np.sin(np.pi * u), np.cos(np.pi * u), np.sin(np.pi * v), np.cos(np.pi * v)]

    static = np.stack([elev_s, slope, valid, *pos]).astype(np.float32)

    out = xr.Dataset(
        dict(
            static=(("channel", "lat", "lon"), static),
            valid=(("lat", "lon"), valid),
        ),
        coords=dict(channel=CHANNELS, lat=grid.lat, lon=grid.lon),
        attrs=dict(
            elevation_source=(
                str(args.dem) if args.dem else "zero-orography placeholder"
            ),
            slope_definition="terrain gradient magnitude in m/m, standardized over land",
        ),
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    partial.unlink(missing_ok=True)
    out.to_netcdf(
        partial,
        engine="netcdf4",
        encoding={
            "static": {
                "dtype": "float32",
                "zlib": True,
                "complevel": 4,
                "shuffle": True,
            },
            "valid": {
                "dtype": "float32",
                "zlib": True,
                "complevel": 4,
                "shuffle": True,
            },
        },
    )
    validate_static(partial, grid)
    partial.replace(output)
    print(f"wrote {args.out} {static.shape}; land fraction = {valid.mean():.2%}")


if __name__ == "__main__":
    main()
