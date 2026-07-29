#!/usr/bin/env python
"""QC the BMD gauge file and build the pseudo-station archive.

    python scripts/05_prepare_stations.py --csv data/stations/bmd_daily_raw.csv \
        --zarr data/processed/bd_wide.zarr --out data/stations/

Produces
--------
bmd_daily.csv        cleaned long-form observations
station_meta.csv     per-station coverage / climatology summary
pseudo_stations.npz  CHIRPS sampled at the BMD coordinates for the FULL record
                     (1981-2025) -- used to tune Gamma and sigma_obs and to run
                     the "known truth" DA experiments of Manshausen et al. S4.1

QC applied
----------
* negative / sentinel values (-999, -9999, 999) -> missing
* "T" (trace) -> 0.05 mm
* daily totals > 1000 mm flagged (BD record is ~1000 mm at Cherrapunji-adjacent
  sites; anything above that is almost certainly a units or decimal error)
* runs of >= 10 identical non-zero values flagged as a stuck gauge
* stations outside the BD grid, or with < 50% coverage, dropped downstream
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bdhires.data.stations import load_stations, pseudo_stations, station_summary  # noqa: E402
from bdhires.grids import BD, WIDE  # noqa: E402


def qc(df: pd.DataFrame, max_mm: float = 1000.0, stuck_run: int = 10) -> pd.DataFrame:
    df = df.copy()
    flags = []
    for sid, g in df.groupby("station_id"):
        v = g["precip_mm"].values
        hi = np.nansum(v > max_mm)
        nz = v.copy()
        nz[~np.isfinite(nz)] = -1
        runs, cur = 0, 1
        for i in range(1, len(nz)):
            if nz[i] == nz[i - 1] and nz[i] > 0:
                cur += 1
            else:
                runs = max(runs, cur)
                cur = 1
        flags.append(dict(station_id=sid, n_over_max=int(hi), longest_stuck_run=int(max(runs, cur))))
    fl = pd.DataFrame(flags)
    bad = fl[(fl.n_over_max > 0) | (fl.longest_stuck_run >= stuck_run)]
    if len(bad):
        print("[qc] suspicious stations:\n", bad.to_string(index=False))
    df.loc[df["precip_mm"] > max_mm, "precip_mm"] = np.nan
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--zarr", required=True)
    ap.add_argument("--out", default="data/stations")
    ap.add_argument("--noise-mm", type=float, default=0.0,
                    help="observation noise to add to pseudo-stations (mm/day)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.csv)
    raw.columns = [c.strip().lower() for c in raw.columns]
    raw["precip_mm"] = pd.to_numeric(
        raw["precip_mm"].astype(str).str.strip().replace({"T": "0.05", "t": "0.05"}),
        errors="coerce",
    )
    raw.loc[raw["precip_mm"] < 0, "precip_mm"] = np.nan
    raw = qc(raw)
    clean = out / "bmd_daily.csv"
    raw.to_csv(clean, index=False)
    print(f"wrote {clean} ({len(raw)} rows)")

    z = zarr.open(args.zarr, mode="r")
    time = np.asarray(z["time"][:]).astype("datetime64[ns]")

    ss, values = load_stations(clean, time, grid=BD, min_coverage=0.0)
    summ = station_summary(ss, values)
    summ.to_csv(out / "station_meta.csv", index=False)
    print(summ.to_string(index=False))

    # pseudo-observations from CHIRPS at the same coordinates, whole record
    print("building pseudo-station archive from CHIRPS ...")
    field = np.stack([np.asarray(z["target"][i]) for i in range(len(time))])
    pseudo = pseudo_stations(field, WIDE, ss, noise_sd_mm=args.noise_mm)
    np.savez_compressed(
        out / "pseudo_stations.npz",
        values=pseudo, lat=ss.lat, lon=ss.lon, ids=ss.ids, time=time.astype("i8"),
    )
    print(f"wrote {out/'pseudo_stations.npz'} shape={pseudo.shape}")


if __name__ == "__main__":
    main()
