#!/usr/bin/env python3
"""Coarsen a v2 packed archive onto a lower-resolution grid, layout preserved.

V7's stage A is CPCv2 at 0.1 degrees rather than 0.05.  The honest way to get
that is not to re-derive the packing -- ``04_regrid_and_pack.py`` already places
CHIRPS, CPC-with-coverage and ERA5 on one common grid, and every choice in it has
been exercised -- but to take its output and reduce it.

Coarsening here is an exact area-weighted block mean, because 0.05 -> 0.1 is a
whole factor on the same lattice.  That is the physically correct reduction for
every field in the archive: precipitation and ERA5 state variables are intensive,
so their area mean is the value of the coarse cell, and ``cpc_valid`` is a
coverage fraction whose area mean is the coarse coverage.  Nothing is
interpolated, so nothing is invented.

The output carries the same array names, the same channel order and the same
attributes as the input, with the grid block updated.  ``06_compute_stats.py``
and ``scripts/train.py`` therefore need no changes at all -- V7 stage A runs the
CPCv2 code path verbatim, one resolution up.

    python scripts/71_coarsen_pack_archive.py \\
      --in data/processed/bd_wide_cpc.zarr \\
      --out data/processed/bd_wide_cpc_0p1.zarr \\
      --factor 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bdhires.data import area_weighted_block_mean, cell_area_weights  # noqa: E402


def block_reduce(
    values: np.ndarray,
    area: torch.Tensor,
    valid: torch.Tensor,
    factor: int,
) -> np.ndarray:
    """Area-weighted block mean of a (T,C,H,W) or (C,H,W) or (H,W) field."""
    array = torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))
    squeeze_channel = array.ndim == 3
    if array.ndim == 2:
        array = array[None, None]
    elif squeeze_channel:
        array = array[:, None]
    # Non-finite cells are masked rather than propagated: CHIRPS is undefined
    # over water, and one NaN must not empty its whole coarse cell.
    finite = torch.isfinite(array)
    array = torch.where(finite, array, torch.zeros_like(array))
    usable = valid.expand_as(finite) & finite
    reduced, _, _ = area_weighted_block_mean(
        array.reshape(-1, 1, *array.shape[-2:]),
        area,
        usable.reshape(-1, 1, *usable.shape[-2:]),
        factor=factor,
        valid_area_threshold=0.0,
    )
    reduced = reduced.reshape(*array.shape[:-2], *reduced.shape[-2:])
    if squeeze_channel:
        reduced = reduced[:, 0]
    elif values.ndim == 2:
        reduced = reduced[0, 0]
    return reduced.numpy().astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="source", required=True)
    parser.add_argument("--out", dest="target", required=True)
    parser.add_argument("--factor", type=int, default=2)
    parser.add_argument(
        "--min-valid-frac", type=float, default=0.5,
        help="coarse cell is valid when at least this fraction of its fine cells are",
    )
    parser.add_argument("--chunk-days", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    factor = int(args.factor)
    if factor < 2:
        raise SystemExit("--factor must be at least 2; nothing to coarsen otherwise")

    source = zarr.open_group(args.source, mode="r")
    if not source.attrs.get("complete", False):
        raise SystemExit(f"{args.source} is not a completed archive")
    grid = dict(source.attrs["grid"])
    if grid["nlat"] % factor or grid["nlon"] % factor:
        raise SystemExit(
            f"grid {grid['nlat']}x{grid['nlon']} is not divisible by {factor}"
        )

    destination = Path(args.target)
    if destination.exists() and not args.overwrite:
        raise SystemExit(f"{destination} exists; pass --overwrite to replace it")

    lat = np.asarray(source["lat"][:], np.float32)
    lon = np.asarray(source["lon"][:], np.float32)
    fine_valid = np.asarray(source["valid"][:]).astype(bool)
    area_np = cell_area_weights(lat, lon).astype(np.float32)
    area = torch.from_numpy(area_np)
    valid = torch.from_numpy(fine_valid)[None, None]

    # A coarse cell is valid when enough of its fine cells are.  Taking the mean
    # of the mask and thresholding is the same statement, and reuses the same
    # area weighting as every other field.
    coverage = block_reduce(
        fine_valid.astype(np.float32),
        area,
        torch.ones_like(valid),
        factor,
    )
    coarse_valid = coverage >= float(args.min_valid_frac)
    coarse_area_valid = torch.from_numpy(coarse_valid)[None, None]

    coarse_lat = lat.reshape(-1, factor).mean(axis=1).astype(np.float32)
    coarse_lon = lon.reshape(-1, factor).mean(axis=1).astype(np.float32)
    shape = (coarse_lat.size, coarse_lon.size)
    times = np.asarray(source["time"][:])
    ntime = times.size
    channels = list(source.attrs["cond_channels"])

    print(f"source : {args.source}  {grid['nlat']}x{grid['nlon']} @{grid['res']} deg")
    print(f"target : {args.target}  {shape[0]}x{shape[1]} @{grid['res'] * factor} deg")
    print(f"days   : {ntime}   conditioning channels: {len(channels)}")
    print(f"valid  : {int(fine_valid.sum())} fine cells -> {int(coarse_valid.sum())} coarse")

    root = zarr.open_group(str(destination), mode="w")
    root.create_array("time", data=np.ascontiguousarray(times))
    root.create_array("lat", data=coarse_lat)
    root.create_array("lon", data=coarse_lon)
    root.create_array("valid", data=np.ascontiguousarray(coarse_valid))
    root.create_array(
        "static",
        data=block_reduce(np.asarray(source["static"][:]), area, valid, factor),
    )
    target_array = root.create_array(
        "target", shape=(ntime, *shape), chunks=(1, *shape), dtype="f4",
    )
    cond_array = root.create_array(
        "cond", shape=(ntime, len(channels), *shape),
        chunks=(1, len(channels), *shape), dtype="f4",
    )

    for start in range(0, ntime, int(args.chunk_days)):
        stop = min(start + int(args.chunk_days), ntime)
        target_array[start:stop] = block_reduce(
            np.asarray(source["target"][start:stop]), area, valid, factor
        )
        cond_array[start:stop] = block_reduce(
            np.asarray(source["cond"][start:stop]), area, valid, factor
        )
        print(f"  {stop}/{ntime}", flush=True)

    attributes = {key: value for key, value in source.attrs.items()}
    attributes["grid"] = {
        "name": f"{grid['name']}_x{factor}",
        "lon_min": grid["lon_min"],
        "lat_min": grid["lat_min"],
        "nlon": shape[1],
        "nlat": shape[0],
        "res": grid["res"] * factor,
    }
    # Provenance, so a coarsened archive can never be mistaken for a packed one.
    attributes["coarsened_from"] = str(args.source)
    attributes["coarsen_factor"] = factor
    attributes["coarsen_min_valid_frac"] = float(args.min_valid_frac)
    root.attrs.update(attributes)

    print(f"\nwrote {destination}")
    print("run scripts/06_compute_stats.py against it before training")


if __name__ == "__main__":
    main()
