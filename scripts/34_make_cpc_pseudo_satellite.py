#!/usr/bin/env python
"""Write CPC as a coarse pseudo-satellite in the prepared-IMERG schema.

Read this before using the output
---------------------------------
CPC is **not an independent observation of this model**, in two separate ways,
and both change what an assimilation result means:

1. **CPC is the model's conditioning input.** ``configs/train_h100_cpc.yaml``
   feeds ``cpc_precip`` to the network, so the background already contains it.
   Assimilating CPC afterwards uses the same information twice: the analysis
   will look confident and the increments will be small, and neither fact tells
   you anything about assimilating a genuinely new satellite. A clean version of
   this experiment needs an UNCONDITIONAL prior, which requires
   ``cond_dropout > 0`` -- v1 was trained at 0.0, so it cannot produce one. The
   v3 configs set 0.1 precisely so this becomes possible later.

2. **CPC Global Unified is itself a gauge analysis.** It is built from station
   reports on the GTS. If BMD stations feed the GTS -- and the major ones do --
   then assimilating CPC alongside BMD gauges double-counts those gauges, and
   the filter treats one measurement as two independent ones.

So do not read the withheld-gauge scores from this as "how well a 0.5 degree
satellite would do". What the experiment DOES legitimately test:

* **Footprint scale.** Stride-1 IMERG (0.1 deg) made the sampler diverge while
  stride-3 (0.3 deg effective) was stable. CPC at 0.5 deg is coarser still. If
  it assimilates stably, that supports "coarse and few beats fine and many" as a
  property of the guidance rather than of IMERG specifically.
* **The observation operator at large factor.** Exercises the block-average H at
  factor 10 rather than 2, which nothing else in the repo does.

The day-shift, which is easy to get wrong
-----------------------------------------
Measured on 43,781 station-days, CPC correlates with BMD best at **lag -1**
(0.756, against 0.387 at lag 0) because CPC is a 00-00 UTC calendar day while
BMD day D is the 24 h ending 03:00 UTC on D -- about 87% of calendar day D-1.
Prepared IMERG is already built on the BMD window and peaks at lag 0.

So CPC must be shifted by one day to enter the DA on the same convention as the
gauges and the analysis. ``--day-shift -1`` does that and is the default. Get
this wrong and the filter is handed yesterday's rain as today's observation.

Usage
-----
    python scripts/34_make_cpc_pseudo_satellite.py \
        --zarr data/processed/bd_wide_cpc.zarr \
        --start 2024-05-01 --end 2024-05-05 \
        --factor 10 --day-shift -1 \
        --out data/processed/cpc_satellite/cpc_pseudo_20240501_20240505.nc
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bdhires.grids import WIDE, crop_offsets, get_grid  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--grid", default="bd")
    parser.add_argument(
        "--factor",
        type=int,
        default=10,
        help="0.05 deg cells per pseudo-footprint. 10 gives 0.5 deg, CPC's native "
        "resolution, but 128 is not divisible by 10 so the field is cropped to "
        "the largest exact tiling (120 cells, 12 footprints). Use 8 (0.4 deg) to "
        "tile the full domain exactly.",
    )
    parser.add_argument(
        "--day-shift",
        type=int,
        default=-1,
        help="shift CPC by this many days so it lands on the BMD 03-03 UTC "
        "convention. Measured optimum is -1; 0 hands the filter the wrong day.",
    )
    parser.add_argument(
        "--sigma-mm",
        type=float,
        default=None,
        help="constant randomError in mm/day. Default derives a per-cell value "
        "from CPC's measured MAE against gauges (7.66 mm/day) scaled by "
        "intensity, which is closer to honest than a flat number.",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", default=None)
    return parser.parse_args()


def load_zarr_time(store) -> np.ndarray:
    raw = np.asarray(store["time"][:])
    if np.issubdtype(raw.dtype, np.datetime64):
        return raw.astype("datetime64[D]")
    return raw.astype("datetime64[ns]").astype("datetime64[D]")


def block_mean(field: np.ndarray, factor: int) -> np.ndarray:
    """Exact block average, cropping to the largest tiling if factor doesn't divide."""
    n_time, nlat, nlon = field.shape
    lat_blocks, lon_blocks = nlat // factor, nlon // factor
    trimmed = field[:, : lat_blocks * factor, : lon_blocks * factor]
    return trimmed.reshape(
        n_time, lat_blocks, factor, lon_blocks, factor
    ).mean(axis=(2, 4))


def main() -> None:
    args = parse_args()
    store = zarr.open(args.zarr, mode="r")
    time = load_zarr_time(store)
    grid = get_grid(args.grid)
    row0, col0 = crop_offsets(WIDE, grid)

    cpc_index = store.attrs.get("cpc_precip_cond_index")
    if cpc_index is None:
        raise SystemExit(f"{args.zarr} has no cpc_precip_cond_index attribute")

    wanted = np.arange(
        np.datetime64(args.start, "D"), np.datetime64(args.end, "D") + 1,
        dtype="datetime64[D]",
    )
    # Shift BACKWARDS in the source: to serve BMD day D we need CPC day D+shift.
    source_days = wanted + np.timedelta64(args.day_shift, "D")
    lookup = {value: index for index, value in enumerate(time)}
    missing = [str(d) for d in source_days if d not in lookup]
    if missing:
        raise SystemExit(
            f"{args.zarr} lacks CPC for {len(missing)} source day(s) "
            f"(shift {args.day_shift:+d}), first few: {missing[:5]}"
        )

    rows = slice(row0, row0 + grid.nlat)
    cols = slice(col0, col0 + grid.nlon)
    fine = np.stack(
        [
            np.asarray(store["cond"][lookup[d]][int(cpc_index)][rows, cols], np.float32)
            for d in source_days
        ]
    )
    coarse = block_mean(fine, args.factor).astype(np.float32)
    n_lat, n_lon = coarse.shape[1:]
    used_lat, used_lon = n_lat * args.factor, n_lon * args.factor
    print(
        f"[cpc-sat] {len(wanted)} days, {grid.nlat}x{grid.nlon} fine -> "
        f"{n_lat}x{n_lon} footprints at {grid.res * args.factor:.2f} deg"
    )
    if (used_lat, used_lon) != (grid.nlat, grid.nlon):
        print(
            f"[cpc-sat] factor {args.factor} does not divide {grid.nlat}; cropped to "
            f"{used_lat}x{used_lon} cells. The DA config must use the SAME factor "
            "and the operator's crop, or H and the observations will disagree.",
            flush=True,
        )

    # Footprint centres of the cropped, blocked field.
    lat = (
        grid.lat[:used_lat].reshape(n_lat, args.factor).mean(axis=1).astype(np.float64)
    )
    lon = (
        grid.lon[:used_lon].reshape(n_lon, args.factor).mean(axis=1).astype(np.float64)
    )

    if args.sigma_mm is not None:
        error = np.full_like(coarse, float(args.sigma_mm))
    else:
        # CPC's MAE against BMD gauges is 7.66 mm/day on a 6.19 mm/day mean, so
        # roughly proportional error with a floor. Crude, but a flat sigma would
        # be worse: it would make the filter equally confident on a dry day and
        # a 100 mm day.
        error = (0.6 * coarse + 1.5).astype(np.float32)

    dataset = xr.Dataset(
        {
            "precipitation": (
                ("time", "lat", "lon"), coarse, {"units": "mm/day", "source": "CPC"},
            ),
            "randomError": (("time", "lat", "lon"), error, {"units": "mm/day"}),
            "precipitation_cnt": (
                ("time", "lat", "lon"),
                np.full(coarse.shape, 48, np.int16),
                {"long_name": "placeholder count, CPC is already daily"},
            ),
        },
        coords={"time": wanted, "lat": lat, "lon": lon},
        attrs={
            "bmd_accumulation_end_hour_utc": 3,
            "source_frequency": "daily",
            # Declares to scripts/15 that this daily product has already been put
            # on the BMD convention by a whole-day shift. The loader refuses any
            # non-half-hourly file without this, which is the right default: a
            # 00-00 UTC product assimilated as if it were 03-03 UTC is a silent,
            # serious error. A day shift approximates the 3-hour offset and
            # cannot reproduce it, so the loader also warns when it sees this.
            "bmd_window_alignment": "day-shift",
            "pseudo_satellite_source": "cpc_precip from the packed Zarr",
            "block_factor": int(args.factor),
            "day_shift_applied": int(args.day_shift),
            "independence_warning": (
                "CPC conditions the prior AND is a gauge analysis; this is not an "
                "independent observation. See the script docstring."
            ),
        },
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(out)
    print(f"[cpc-sat] wrote {out}")

    report = {
        "zarr": args.zarr,
        "start": args.start,
        "end": args.end,
        "factor": args.factor,
        "footprint_deg": grid.res * args.factor,
        "day_shift": args.day_shift,
        "n_days": int(len(wanted)),
        "n_footprints": int(n_lat * n_lon),
        "cropped_from": [int(grid.nlat), int(grid.nlon)],
        "cropped_to": [int(used_lat), int(used_lon)],
        "mean_mm": float(np.nanmean(coarse)),
        "wet_fraction_1mm": float(np.nanmean(coarse >= 1.0)),
    }
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2))
        print(f"[cpc-sat] wrote {args.report}")
    print(
        f"[cpc-sat] mean {report['mean_mm']:.2f} mm/day, wet fraction "
        f"{report['wet_fraction_1mm']:.3f}, {report['n_footprints']} footprints/day"
    )


if __name__ == "__main__":
    main()
