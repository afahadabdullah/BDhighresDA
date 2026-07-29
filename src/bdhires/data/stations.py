"""BMD station ingestion and pseudo-station generation.

Expected raw format (``data/stations/bmd_daily.csv``), long form::

    station_id,name,lat,lon,date,precip_mm
    11111,Dhaka,23.7776,90.3795,2020-01-01,0.0

Missing values may be blank, ``NA``, ``-999`` or ``-9999``.  Trace amounts
reported as ``T`` are mapped to 0.05 mm.

Two derived products:

* ``load_stations`` -> ``StationSet`` + a (T, S) matrix aligned to a date index.
* ``pseudo_stations`` -> samples CHIRPS at the BMD coordinates for the whole
  record.  These let you tune Gamma / sigma_obs and validate the whole DA
  pipeline over 1981-2025 in a setting where the true full field is known --
  exactly the experiment in Manshausen et al. Section 4.1.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..da.observation import StationSet
from ..grids import Grid

MISSING = {-999.0, -9999.0, -99.9, 999.0}


def load_stations(
    csv_path: str | Path,
    dates: np.ndarray,
    grid: Grid | None = None,
    min_coverage: float = 0.5,
) -> tuple[StationSet, np.ndarray]:
    """Return ``(StationSet, values)`` with ``values`` of shape (len(dates), S)."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    req = {"station_id", "lat", "lon", "date", "precip_mm"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")

    df["precip_mm"] = (
        df["precip_mm"].astype(str).str.strip().replace({"T": "0.05", "t": "0.05", "": None, "NA": None})
    )
    df["precip_mm"] = pd.to_numeric(df["precip_mm"], errors="coerce")
    df.loc[df["precip_mm"].isin(MISSING), "precip_mm"] = np.nan
    df.loc[df["precip_mm"] < 0, "precip_mm"] = np.nan
    df["date"] = pd.to_datetime(df["date"]).values.astype("datetime64[D]")

    meta = df.groupby("station_id")[["lat", "lon"]].first().reset_index()

    if grid is not None:
        lo, la, hi, ha = grid.bbox
        # keep a half-cell margin so bilinear interpolation stays in-domain
        m = grid.res / 2
        keep = (
            (meta["lon"] > lo + m) & (meta["lon"] < hi - m)
            & (meta["lat"] > la + m) & (meta["lat"] < ha - m)
        )
        dropped = meta.loc[~keep, "station_id"].tolist()
        if dropped:
            print(f"[stations] dropping {len(dropped)} station(s) outside {grid.name}: {dropped}")
        meta = meta[keep]

    d_index = pd.DatetimeIndex(pd.to_datetime(dates))
    wide = (
        df[df["station_id"].isin(meta["station_id"])]
        .pivot_table(index="date", columns="station_id", values="precip_mm", aggfunc="mean")
        .reindex(d_index)
        .reindex(columns=meta["station_id"].values)
    )

    cov = wide.notna().mean(axis=0).values
    keep = cov >= min_coverage
    if (~keep).any():
        print(
            f"[stations] dropping {int((~keep).sum())} station(s) with <{min_coverage:.0%} "
            f"coverage over the requested period"
        )
    meta = meta[keep]
    wide = wide.loc[:, meta["station_id"].values]

    ss = StationSet(
        lat=meta["lat"].to_numpy(float),
        lon=meta["lon"].to_numpy(float),
        ids=meta["station_id"].to_numpy(),
    )
    return ss, wide.to_numpy(np.float32)


def pseudo_stations(
    field: np.ndarray,      # (T, H, W) mm/day
    grid: Grid,
    stations: StationSet,
    noise_sd_mm: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Sample a gridded field at station locations (bilinear), optionally noisy."""
    from scipy.interpolate import RegularGridInterpolator

    rng = np.random.default_rng(seed)
    pts = np.stack([stations.lat, stations.lon], axis=-1)
    out = np.empty((field.shape[0], len(stations)), np.float32)
    for t in range(field.shape[0]):
        f = np.nan_to_num(field[t], nan=0.0)
        itp = RegularGridInterpolator((grid.lat, grid.lon), f, bounds_error=False, fill_value=None)
        out[t] = itp(pts)
    if noise_sd_mm > 0:
        out = np.clip(out + rng.normal(0, noise_sd_mm, out.shape), 0, None)
    return out


def station_summary(ss: StationSet, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station_id": ss.ids,
            "lat": ss.lat,
            "lon": ss.lon,
            "n_obs": np.isfinite(values).sum(axis=0),
            "coverage": np.isfinite(values).mean(axis=0),
            "mean_mm": np.nanmean(values, axis=0),
            "p99_mm": np.nanpercentile(values, 99, axis=0),
            "max_mm": np.nanmax(values, axis=0),
        }
    )
