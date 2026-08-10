#!/usr/bin/env python
"""Block-average prepared IMERG onto a coarser observation footprint.

The scale ladder -- assimilating the same satellite field at 0.1, 0.4 and 0.8
degrees -- cannot be built by changing ``observations.imerg.factor`` alone.
That was tried and both arms died immediately:

    ValueError: operands could not be broadcast together with shapes (64,) (16,)

``factor`` is not an operator-side knob. ``scripts/15_bmd_month_example.py``
builds ``grid.lat[:n*factor].reshape(n, factor).mean(axis=1)`` and requires the
observation FILE's own latitudes to equal it, so ``factor`` is a declaration
about which grid the file is on. A 0.4 degree arm needs a 0.4 degree file.

Three things have to move together with it, and getting any of them wrong makes
the arm measure something other than observation scale:

**1. randomError cannot be averaged as though the footprints were independent.**
A block mean of ``n`` footprints has variance ``(1/n^2) sum_ij rho_ij s_i s_j``.
Under an exponential correlation ``rho(d) = exp(-d/L)`` with the L implied by
``error_corr_cells`` (~30 km), a 4x4 block at 0.1 degrees gives a geometric
factor of 0.52, so the block error is 0.72 of the footprint error -- not the
0.25 that ``s/sqrt(n)`` would claim. Assuming independence would overstate the
satellite's precision by 2.9x at 0.4 degrees and 4.3x at 0.8 degrees, which
would let a deliberately down-weighted observation dominate the analysis.

**2. ``error_corr_cells`` is denominated in CELLS OF THE OBSERVATION GRID.**
The configured 3.0 means 0.30 degrees at 0.1 degree footprints, which is the
physical ~30 km. Left alone at 0.4 degrees it would assert 1.2 degrees, and
``scripts/15`` would inflate R by the square of it. It must be rescaled to keep
the physical length fixed.

**3. Thinning must come off.** Stride exists to stop correlated footprints being
counted as independent evidence; coarsening already does that. Keeping stride 3
on top leaves 28 usable footprints at 0.4 degrees and 7 at 0.8 -- not an
experiment. The script prints the stride each scale wants.

The block mean is exact and mass-conserving, and the coarsened centres nest on
the model grid by construction: averaging 4 consecutive 0.1 degree centres is
averaging 8 consecutive 0.05 degree centres, which is what ``scripts/15``
computes.

Example
-------
    python scripts/44_coarsen_imerg_observations.py \\
        --input data/processed/imerg_prepared_ing2022/imerg_aligned_20220501_20220510.nc \\
        --factor 8 \\
        --out data/processed/imerg_prepared_ing2022/imerg_0p4deg_20220501_20220510.nc
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

MM_PER_DAY = {"mm/day", "mmday-1", "mmd-1", "mmday^-1", "mmd^-1"}
KM_PER_DEGREE = 111.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True,
                        help="Prepared IMERG on the native footprint grid")
    parser.add_argument(
        "--factor", type=int, required=True,
        help="TARGET observations.imerg.factor, in MODEL cells: 8 gives 0.4 deg "
             "on a 0.05 deg grid, 16 gives 0.8 deg. This is the value the DA "
             "config must also be set to.",
    )
    parser.add_argument(
        "--source-factor", type=int, default=2,
        help="The input file's factor, i.e. how many model cells one input "
             "footprint spans. 2 is the native 0.1 deg IMERG.",
    )
    parser.add_argument("--model-resolution", type=float, default=0.05,
                        help="Model grid spacing in degrees")
    parser.add_argument(
        "--source-corr-cells", type=float, default=3.0,
        help="observations.imerg.error_corr_cells as configured FOR THE INPUT "
             "file. The physical correlation length is derived from it, so the "
             "coarsened file inherits exactly the length the DA config already "
             "assumes rather than a second, slightly different constant.",
    )
    parser.add_argument(
        "--correlation-km", type=float, default=None,
        help="Override the derived correlation length. Rarely wanted: "
             "deriving it keeps this file and configs/da.yaml consistent.",
    )
    parser.add_argument(
        "--independent-errors", action="store_true",
        help="Aggregate randomError as s/sqrt(n) instead. This asserts the "
             "footprints are uncorrelated, which contradicts error_corr_cells "
             "and will overstate the satellite's precision severalfold. "
             "Provided only so the assumption can be tested deliberately.",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", default=None)
    return parser.parse_args()


def geometric_factor(block: int, res_deg: float, latitude: float,
                     correlation_km: float) -> float:
    """``(1/n^2) sum_ij rho(d_ij)`` for one block: the variance retained by the mean.

    Bounded by ``1/n`` when every footprint is independent and by ``1`` when they
    move together. The lattice is regular, so this is one scalar for the whole
    field rather than a per-block quantity; longitude spacing is taken at the
    domain's mean latitude, which over a 6-degree domain moves the factor by far
    less than the uncertainty in the correlation length itself.
    """
    i, j = np.meshgrid(np.arange(block), np.arange(block), indexing="ij")
    y = i.ravel() * res_deg * KM_PER_DEGREE
    x = j.ravel() * res_deg * KM_PER_DEGREE * np.cos(np.radians(latitude))
    distance = np.hypot(y[:, None] - y[None, :], x[:, None] - x[None, :])
    return float(np.exp(-distance / correlation_km).mean())


def block_reduce(array: np.ndarray, block: int, how: str) -> np.ndarray:
    """Reduce the trailing two axes in ``block`` x ``block`` tiles, ignoring NaN."""
    time, nlat, nlon = array.shape
    tiles = array.reshape(time, nlat // block, block, nlon // block, block)
    if how == "nanmean":
        with np.errstate(invalid="ignore"):
            return np.nanmean(tiles, axis=(2, 4))
    if how == "nanmin":
        with np.errstate(invalid="ignore"):
            return np.nanmin(tiles, axis=(2, 4))
    if how == "nanrms":
        with np.errstate(invalid="ignore"):
            return np.sqrt(np.nanmean(tiles ** 2, axis=(2, 4)))
    raise ValueError(how)


def main() -> None:
    args = parse_args()
    if args.factor % args.source_factor:
        raise SystemExit(
            f"target factor {args.factor} is not a multiple of the source "
            f"factor {args.source_factor}; the block mean would not be exact"
        )
    block = args.factor // args.source_factor
    if block < 2:
        raise SystemExit(
            f"target factor {args.factor} is not coarser than the source "
            f"{args.source_factor}; there is nothing to coarsen"
        )

    dataset = xr.open_dataset(args.input).load()
    for variable in ("precipitation", "randomError"):
        units = str(dataset[variable].attrs.get("units", ""))
        if units.lower().replace(" ", "") not in MM_PER_DAY:
            raise SystemExit(f"{args.input} {variable} units are {units!r}; expected mm/day")

    nlat, nlon = dataset.sizes["lat"], dataset.sizes["lon"]
    if nlat % block or nlon % block:
        raise SystemExit(
            f"a {block}x{block} block mean does not tile the {nlat}x{nlon} "
            f"source grid exactly. A ragged edge would shift the footprint "
            f"centres off the model grid and scripts/15 would reject the file."
        )

    source_res = args.source_factor * args.model_resolution
    target_res = args.factor * args.model_resolution
    latitude = float(np.mean(dataset.lat.values))

    # Derive the physical length from the configured cell count so that this
    # file and configs/da.yaml cannot drift apart. 3.0 cells at 0.1 deg is
    # 0.30 deg is 33.3 km; the coarsened file then needs 0.75 cells at 0.4 deg,
    # which is the SAME distance expressed on the new grid.
    correlation_km = args.correlation_km if args.correlation_km is not None else \
        args.source_corr_cells * source_res * KM_PER_DEGREE

    if args.independent_errors:
        sigma_scale = 1.0 / np.sqrt(block * block)
        error_model = f"independent footprints, s/sqrt({block*block})"
    else:
        g = geometric_factor(block, source_res, latitude, correlation_km)
        sigma_scale = float(np.sqrt(g))
        error_model = (f"exponential correlation, L={correlation_km:.1f} km, "
                       f"geometric factor {g:.4f}")

    precipitation = block_reduce(dataset["precipitation"].values.astype(np.float64),
                                 block, "nanmean")
    # RMS, because the block error is being approximated as a common sigma times
    # a geometric factor, and the variance-like summary is the right one.
    error = block_reduce(dataset["randomError"].values.astype(np.float64),
                         block, "nanrms") * sigma_scale
    # The block is only as trustworthy as its worst footprint: a block holding
    # one poorly sampled footprint should not inherit the others' count.
    count = block_reduce(dataset["precipitation_cnt"].values.astype(np.float64),
                         block, "nanmin")

    lat = dataset.lat.values.reshape(-1, block).mean(axis=1)
    lon = dataset.lon.values.reshape(-1, block).mean(axis=1)

    corr_cells_source = correlation_km / (source_res * KM_PER_DEGREE)
    corr_cells_target = correlation_km / (target_res * KM_PER_DEGREE)

    attrs = dict(dataset.attrs)
    attrs.update({
        "coarsened_from": str(args.input),
        "coarsened_block": f"{block}x{block}",
        "observation_factor": int(args.factor),
        "observation_resolution_degrees": float(target_res),
        "random_error_aggregation": error_model,
        "random_error_scale_applied": float(sigma_scale),
        "spatial_support": f"exact {target_res:.2f}-degree footprints nested over "
                           f"the {args.model_resolution:.2f}-degree model grid",
        "required_error_corr_cells": float(round(corr_cells_target, 4)),
        "coarsen_tool": "scripts/44_coarsen_imerg_observations.py",
    })

    out = xr.Dataset(
        {
            "precipitation": (("time", "lat", "lon"), precipitation.astype(np.float32),
                              dict(dataset["precipitation"].attrs)),
            "randomError": (("time", "lat", "lon"), error.astype(np.float32),
                            {**dataset["randomError"].attrs,
                             "aggregation": error_model}),
            "precipitation_cnt": (("time", "lat", "lon"), count.astype(np.int16),
                                  dict(dataset["precipitation_cnt"].attrs)),
        },
        coords={"time": dataset.time.values, "lat": lat, "lon": lon},
        attrs=attrs,
    )

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    out.to_netcdf(temporary)
    temporary.replace(output)

    valid = np.isfinite(precipitation)
    report = {
        "derivation": "block mean of prepared IMERG; no re-accumulation",
        "source": str(args.input),
        "block": block,
        "source_resolution_degrees": source_res,
        "target_resolution_degrees": target_res,
        "observation_factor": int(args.factor),
        "correlation_length_km": float(correlation_km),
        "random_error": {
            "model": error_model,
            "scale_applied": float(sigma_scale),
            "scale_if_independent": float(1.0 / np.sqrt(block * block)),
            "overstatement_avoided": float(sigma_scale * np.sqrt(block * block)),
        },
        "required_config": {
            "observations.imerg.factor": int(args.factor),
            "observations.imerg.error_corr_cells": float(round(corr_cells_target, 4)),
            "imerg_stride": 1,
        },
        "grid": {"shape": list(precipitation.shape[1:]),
                 "lat_range_centres": [float(lat[0]), float(lat[-1])],
                 "lon_range_centres": [float(lon[0]), float(lon[-1])]},
        "quality_control": {"valid_fraction": float(valid.mean()),
                            "possible_footprints": int(valid.size),
                            "valid_footprints": int(valid.sum())},
        "precipitation_mm_day": {"mean": float(np.nanmean(precipitation)),
                                 "max": float(np.nanmax(precipitation))},
        "random_error_mm_day": {"median": float(np.nanmedian(error)),
                                "max": float(np.nanmax(error))},
    }
    report_path = Path(args.report) if args.report else \
        output.with_name(output.stem + "_qc.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    source_mean = float(np.nanmean(dataset["precipitation"].values))
    print(f"[coarsen] {nlat}x{nlon} at {source_res:.2f} deg -> "
          f"{precipitation.shape[1]}x{precipitation.shape[2]} at {target_res:.2f} deg "
          f"({block}x{block} block mean)")
    print(f"[coarsen] domain mean {source_mean:.4f} -> "
          f"{np.nanmean(precipitation):.4f} mm/day")
    print(f"[coarsen] randomError: {error_model}")
    print(f"[coarsen]   sigma x{sigma_scale:.3f}, versus x"
          f"{1/np.sqrt(block*block):.3f} if independence were assumed "
          f"({sigma_scale*np.sqrt(block*block):.1f}x less confident)")
    print()
    print("[coarsen] this file REQUIRES all three of these, or the arm measures "
          "something else:")
    print(f"    observations.imerg.factor={args.factor}")
    print(f"    observations.imerg.error_corr_cells={corr_cells_target:.4f}"
          f"   (was {corr_cells_source:.2f} at {source_res:.2f} deg; the "
          f"PHYSICAL length is unchanged)")
    print(f"    stride=1"
          f"   (stride 3 would leave {(precipitation.shape[1]//3)**2} footprints)")
    print(f"wrote {output}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
