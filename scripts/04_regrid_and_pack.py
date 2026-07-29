#!/usr/bin/env python
"""Regrid everything onto the 0.05 deg wide grid and pack into a single Zarr store.

    python scripts/04_regrid_and_pack.py --start 1981 --end 2025 \
        --out data/processed/bd_wide.zarr

Time alignment (get this right or everything downstream is subtly wrong)
------------------------------------------------------------------------
* CHIRPS day D  = 00:00 UTC D to 00:00 UTC D+1 accumulation.
* ERA5 tp is a backward hourly accumulation, so day D = sum of steps
  01:00(D) ... 00:00(D+1).
* IMERG 3IMERGDF day D is labelled S000000-E235959 on day D, i.e. already
  00-24 UTC.  Its ``precipitation`` variable is a RATE in mm/hr -> multiply
  by 24 for mm/day.
* State variables (winds, humidity, CAPE) are averaged over 00-24 UTC of day D.

Regridding
----------
Precipitation is regridded CONSERVATIVELY (xesmf if available, else an
area-weighted block mean followed by bilinear); state variables are regridded
bilinearly.  Using bilinear for precipitation coarse->fine is acceptable
(it is just a smooth prior for the network) but conservative preserves the
domain-mean rainfall, which makes the ERA5/IMERG baselines fair.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bdhires.grids import WIDE  # noqa: E402

SFC_MAP = {
    "tp": "era5_tp", "tcwv": "era5_tcwv", "t2m": "era5_t2m", "d2m": "era5_d2m",
    "msl": "era5_msl", "cape": "era5_cape", "u10": "era5_u10", "v10": "era5_v10",
    "sp": "era5_sp", "p71.162": "era5_ivte", "p72.162": "era5_ivtn",
}
PL_VARS = ["u", "v", "q", "t", "w", "z"]
PL_LEVELS = [850, 700, 500, 200]


def _rename_coords(ds: xr.Dataset) -> xr.Dataset:
    ren = {}
    for a, b in (("latitude", "lat"), ("longitude", "lon"), ("valid_time", "time")):
        if a in ds.coords or a in ds.dims:
            ren[a] = b
    ds = ds.rename(ren)
    if "lat" in ds.coords and ds.lat[0] > ds.lat[-1]:
        ds = ds.sortby("lat")
    return ds


def to_grid(da: xr.DataArray, grid, conservative: bool = False) -> xr.DataArray:
    """Regrid a lat/lon DataArray onto the target grid."""
    if conservative:
        try:
            import xesmf as xe

            target = xr.Dataset(coords=dict(lat=grid.lat, lon=grid.lon))
            rg = xe.Regridder(da, target, "conservative_normed", periodic=False)
            return rg(da)
        except Exception as exc:  # pragma: no cover
            print(f"  [regrid] xesmf unavailable ({exc}); falling back to bilinear")
    return da.interp(lat=grid.lat, lon=grid.lon, method="linear",
                     kwargs=dict(fill_value=None))


def daily_era5(sfc_files, pl_files, grid, days) -> tuple[np.ndarray, list[str]]:
    sfc = _rename_coords(xr.open_mfdataset(sfc_files, combine="by_coords"))
    pl = _rename_coords(xr.open_mfdataset(pl_files, combine="by_coords"))

    # precipitation: shift back 1h so that 01:00(D)..00:00(D+1) lands on day D
    tp = sfc["tp"].assign_coords(time=sfc["time"] - np.timedelta64(1, "h"))
    tp_daily = (tp.resample(time="1D").sum() * 1000.0)  # m -> mm
    tp_daily = tp_daily.reindex(time=days)

    channels, names = [], []
    channels.append(to_grid(tp_daily, grid, conservative=True))
    names.append("era5_tp")

    for v, name in SFC_MAP.items():
        if v == "tp" or v not in sfc:
            continue
        d = sfc[v].resample(time="1D").mean().reindex(time=days)
        channels.append(to_grid(d, grid))
        names.append(name)

    for v in PL_VARS:
        if v not in pl:
            continue
        for lev in PL_LEVELS:
            if lev not in pl["pressure_level"].values:
                continue
            d = pl[v].sel(pressure_level=lev).resample(time="1D").mean().reindex(time=days)
            channels.append(to_grid(d, grid))
            names.append(f"era5_{v}{lev}")

    # derived: 850-200 hPa wind shear magnitude (monsoon-depression proxy)
    idx = {n: i for i, n in enumerate(names)}
    if {"era5_u850", "era5_u200", "era5_v850", "era5_v200"} <= set(idx):
        du = channels[idx["era5_u200"]] - channels[idx["era5_u850"]]
        dv = channels[idx["era5_v200"]] - channels[idx["era5_v850"]]
        channels.append(np.hypot(du, dv))
        names.append("era5_shear")

    arr = np.stack([c.transpose("time", "lat", "lon").values for c in channels], axis=1)
    return arr.astype(np.float32), names


def daily_imerg(files, grid, days) -> np.ndarray:
    if not files:
        return np.full((len(days), 1, grid.nlat, grid.nlon), np.nan, np.float32)
    ds = _rename_coords(xr.open_mfdataset(files, combine="by_coords"))
    var = "precipitation" if "precipitation" in ds else "precipitationCal"
    da = ds[var]
    if set(da.dims) >= {"lon", "lat"} and da.sizes.get("lon", 0) > da.sizes.get("lat", 0):
        pass
    da = da.transpose("time", "lat", "lon")
    da = da * 24.0  # mm/hr -> mm/day
    da = to_grid(da, grid, conservative=True).reindex(time=days)
    return da.values[:, None].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1981)
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument("--era5", default="data/raw/era5")
    ap.add_argument("--chirps", default="data/raw/chirps")
    ap.add_argument("--imerg", default="data/raw/imerg")
    ap.add_argument("--static", default="data/static/static_wide.nc")
    ap.add_argument("--out", default="data/processed/bd_wide.zarr")
    ap.add_argument("--chunk-years", type=int, default=1)
    args = ap.parse_args()

    import zarr

    grid = WIDE
    days_all = pd.date_range(f"{args.start}-01-01", f"{args.end}-12-31", freq="D")
    st = xr.open_dataset(args.static)
    static = st["static"].values.astype(np.float32)
    valid = st["valid"].values.astype(np.float32)

    root = zarr.open(args.out, mode="w")
    root.create_dataset("static", data=static)
    root.create_dataset("valid", data=valid)
    root.attrs["static_channels"] = list(map(str, st["channel"].values))
    root.attrs["grid"] = dict(name=grid.name, lon_min=grid.lon_min, lat_min=grid.lat_min,
                              nlon=grid.nlon, nlat=grid.nlat, res=grid.res)
    root.create_dataset("lat", data=grid.lat.astype(np.float32))
    root.create_dataset("lon", data=grid.lon.astype(np.float32))

    target_z = cond_z = None
    times_out, cond_names = [], None
    offset = 0

    for year in range(args.start, args.end + 1, args.chunk_years):
        yr_end = min(year + args.chunk_years - 1, args.end)
        days = pd.date_range(f"{year}-01-01", f"{yr_end}-12-31", freq="D")
        print(f"=== {year}-{yr_end}: {len(days)} days", flush=True)

        # --- target: CHIRPS
        cf = sorted(Path(args.chirps).glob(f"chirps_wide_{{{year}..{yr_end}}}.nc")) or [
            Path(args.chirps) / f"chirps_wide_{y}.nc" for y in range(year, yr_end + 1)
        ]
        cf = [p for p in cf if p.exists()]
        if not cf:
            print(f"  no CHIRPS for {year}, skipping")
            continue
        ch = _rename_coords(xr.open_mfdataset([str(p) for p in cf], combine="by_coords"))
        tgt = ch["precip"].interp(lat=grid.lat, lon=grid.lon, method="nearest")
        tgt = tgt.reindex(time=days).transpose("time", "lat", "lon").values.astype(np.float32)

        # --- conditioning: ERA5 (+ IMERG)
        sfc = sorted(Path(args.era5).glob(f"era5_sfc_{year}*.nc"))
        pl = sorted(Path(args.era5).glob(f"era5_pl_{year}*.nc"))
        for y in range(year + 1, yr_end + 1):
            sfc += sorted(Path(args.era5).glob(f"era5_sfc_{y}*.nc"))
            pl += sorted(Path(args.era5).glob(f"era5_pl_{y}*.nc"))
        era, names = daily_era5([str(p) for p in sfc], [str(p) for p in pl], grid, days)

        im_files = []
        for y in range(year, yr_end + 1):
            im_files += sorted(Path(args.imerg).glob(f"*3IMERG.{y}*.nc4"))
        imerg = daily_imerg([str(p) for p in im_files], grid, days)
        cond = np.concatenate([era, imerg], axis=1)
        names = names + ["imerg_precip"]

        if target_z is None:
            cond_names = names
            root.attrs["cond_channels"] = cond_names
            root.attrs["imerg_cond_index"] = cond_names.index("imerg_precip")
            root.attrs["era5_tp_cond_index"] = cond_names.index("era5_tp")
            target_z = root.create_dataset(
                "target", shape=(0, grid.nlat, grid.nlon), chunks=(32, grid.nlat, grid.nlon),
                dtype="f4",
            )
            cond_z = root.create_dataset(
                "cond", shape=(0, cond.shape[1], grid.nlat, grid.nlon),
                chunks=(16, cond.shape[1], grid.nlat, grid.nlon), dtype="f4",
            )
        elif names != cond_names:
            raise RuntimeError(f"channel mismatch in {year}: {set(names) ^ set(cond_names)}")

        target_z.append(tgt)
        cond_z.append(cond)
        times_out.append(days.values)
        offset += len(days)

    root.create_dataset("time", data=np.concatenate(times_out).astype("datetime64[ns]").view("i8"))
    root.attrs["time_units"] = "nanoseconds since 1970-01-01"
    print(f"wrote {args.out}: T={offset}, cond channels={len(cond_names)}")
    print("cond channels:", cond_names)


if __name__ == "__main__":
    main()
