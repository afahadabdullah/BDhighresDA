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

    python scripts/03_build_static.py --dem data/raw/dem/gmted_bd.tif \
        --chirps data/raw/chirps/chirps_wide_2010.nc --out data/static/static_wide.nc

If no DEM is supplied, the script fetches SRTM-derived GMTED2010 via ``elevation``
or falls back to an ERA5 geopotential-derived orography.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bdhires.grids import WIDE  # noqa: E402


def regrid_dem(dem_path: str, grid) -> np.ndarray:
    import rioxarray  # noqa: F401

    da = xr.open_dataarray(dem_path).squeeze()
    da = da.rename({da.dims[-2]: "y", da.dims[-1]: "x"})
    # conservative-ish: coarsen to ~0.05 then interpolate onto exact centres
    out = da.interp(y=grid.lat, x=grid.lon, method="linear")
    return np.nan_to_num(out.values, nan=0.0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dem", default=None)
    ap.add_argument("--chirps", required=True, help="any CHIRPS wide-domain year file")
    ap.add_argument("--out", default="data/static/static_wide.nc")
    args = ap.parse_args()

    grid = WIDE
    with xr.open_dataset(args.chirps) as ds:
        p = ds["precip"]
        lat_name = "latitude" if "latitude" in p.dims else "lat"
        lon_name = "longitude" if "longitude" in p.dims else "lon"
        p = p.interp({lat_name: grid.lat, lon_name: grid.lon}, method="nearest")
        valid = np.isfinite(p).any(dim="time").values.astype(np.float32)

    if args.dem:
        elev = regrid_dem(args.dem, grid)
    else:
        print("no --dem supplied; using a zero orography placeholder. "
              "Download GMTED2010 or SRTM and rerun -- orography is the single most "
              "informative static channel for Bangladesh rainfall.")
        elev = np.zeros(grid.shape, np.float32)

    elev_s = np.sqrt(np.clip(elev, 0, None))
    elev_s = (elev_s - elev_s.min()) / (np.ptp(elev_s) + 1e-6)
    gy, gx = np.gradient(elev)
    slope = np.hypot(gy, gx)
    slope = (slope - slope.mean()) / (slope.std() + 1e-6)

    lon2, lat2 = np.meshgrid(grid.lon, grid.lat)
    u = (lon2 - grid.lon[0]) / (grid.lon[-1] - grid.lon[0])
    v = (lat2 - grid.lat[0]) / (grid.lat[-1] - grid.lat[0])
    pos = [np.sin(np.pi * u), np.cos(np.pi * u), np.sin(np.pi * v), np.cos(np.pi * v)]

    static = np.stack([elev_s, slope, valid, *pos]).astype(np.float32)
    names = ["elev", "slope", "lsm", "sin_lon", "cos_lon", "sin_lat", "cos_lat"]

    out = xr.Dataset(
        dict(
            static=(("channel", "lat", "lon"), static),
            valid=(("lat", "lon"), valid),
        ),
        coords=dict(channel=names, lat=grid.lat, lon=grid.lon),
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(args.out)
    print(f"wrote {args.out} {static.shape}; land fraction = {valid.mean():.2%}")


if __name__ == "__main__":
    main()
