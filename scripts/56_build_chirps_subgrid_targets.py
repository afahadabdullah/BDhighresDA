#!/usr/bin/env python3
"""Build the native-CPC V3-SG target/conditioning Zarr archive.

This is intentionally a new preparation path.  It reads native CPC and CHIRPS
coordinates and refuses to relabel the legacy bilinearly interpolated CPC
channel as a 0.5-degree field.

Example
-------
python scripts/56_build_chirps_subgrid_targets.py \
  --chirps-glob 'data/raw/chirps/chirps_wide_*.nc' \
  --cpc-glob 'data/raw/cpc/precip.*.nc' \
  --era5-glob 'data/raw/era5/era5_daily_*.nc' \
  --static data/processed/static_wide_cpc.nc \
  --out data/processed/cpc_v3_subgrid/wide_cpc.zarr
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bdhires.data import (  # noqa: E402
    SubgridEncoding,
    area_weighted_block_mean,
    cell_area_weights,
    encode_subgrid_targets,
    encoding_metadata,
    validate_cpc_alignment,
)
from bdhires.grids import Grid, WIDE_CPC  # noqa: E402


ERA5_DEFAULT = ("tcwv", "cape", "u10", "v10", "msl")


def _paths(pattern: str, label: str) -> list[str]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"{label} pattern matched no files: {pattern}")
    return paths


def _open_many(paths: list[str]) -> xr.Dataset:
    if len(paths) == 1:
        return xr.open_dataset(paths[0])
    return xr.open_mfdataset(paths, combine="by_coords", parallel=False)


def _standardize_names(dataset: xr.Dataset) -> xr.Dataset:
    rename = {}
    for old, new in (
        ("latitude", "lat"), ("longitude", "lon"), ("valid_time", "time")
    ):
        if old in dataset.coords or old in dataset.dims:
            rename[old] = new
    dataset = dataset.rename(rename)
    if "lat" in dataset.coords and bool((dataset.lat[0] > dataset.lat[-1]).item()):
        dataset = dataset.sortby("lat")
    if "time" in dataset.coords:
        day = dataset.time.values.astype("datetime64[D]").astype("datetime64[ns]")
        if len(np.unique(day)) != len(day):
            raise ValueError("source has duplicate calendar days after time normalisation")
        dataset = dataset.assign_coords(time=day).sortby("time")
    return dataset


def _variable(dataset: xr.Dataset, choices: tuple[str, ...], label: str) -> xr.DataArray:
    for name in choices:
        if name in dataset:
            return dataset[name]
    raise ValueError(f"{label} has none of the variables {choices}")


def _exact_grid(data: xr.DataArray, grid: Grid, label: str) -> xr.DataArray:
    if "lat" not in data.coords or "lon" not in data.coords:
        raise ValueError(f"{label} must carry lat/lon coordinates")
    selected = data.sel(lat=grid.lat, lon=grid.lon, method="nearest")
    np.testing.assert_allclose(selected.lat.values, grid.lat, atol=1.0e-5)
    np.testing.assert_allclose(selected.lon.values, grid.lon, atol=1.0e-5)
    return selected


def _interpolate_grid(data: xr.DataArray, grid: Grid, label: str) -> xr.DataArray:
    lat = np.asarray(data.lat.values)
    lon = np.asarray(data.lon.values)
    if lat.min() > grid.lat[0] or lat.max() < grid.lat[-1]:
        raise ValueError(f"{label} latitude does not cover {grid.name}")
    if lon.min() > grid.lon[0] or lon.max() < grid.lon[-1]:
        raise ValueError(f"{label} longitude does not cover {grid.name}")
    return data.interp(lat=grid.lat, lon=grid.lon, method="linear")


def _block_mean_numpy(values: np.ndarray, area: np.ndarray, factor: int) -> np.ndarray:
    time, channels, height, width = values.shape
    ab = area.reshape(height // factor, factor, width // factor, factor)
    vb = values.reshape(
        time, channels, height // factor, factor, width // factor, factor
    )
    numerator = (vb * ab[None, None]).sum(axis=(3, 5))
    denominator = ab.sum(axis=(1, 3))
    return numerator / denominator[None, None]


class RunningChannels:
    def __init__(self, channels: int):
        self.count = np.zeros(channels, np.float64)
        self.total = np.zeros(channels, np.float64)
        self.square = np.zeros(channels, np.float64)

    def update(self, values: np.ndarray) -> None:
        finite = np.isfinite(values)
        self.count += finite.sum(axis=(0, 2, 3))
        safe = np.where(finite, values, 0.0).astype(np.float64)
        self.total += safe.sum(axis=(0, 2, 3))
        self.square += (safe * safe).sum(axis=(0, 2, 3))

    def finish(self) -> tuple[np.ndarray, np.ndarray]:
        mean = self.total / np.maximum(self.count, 1.0)
        variance = self.square / np.maximum(self.count, 1.0) - mean * mean
        return mean.astype(np.float32), np.sqrt(np.maximum(variance, 1.0e-8)).astype(np.float32)


def _season(times: np.ndarray, height: int, width: int) -> np.ndarray:
    days = times.astype("datetime64[D]")
    year = days.astype("datetime64[Y]")
    doy = (days - year).astype(int)
    angle = 2.0 * np.pi * doy / 365.25
    base = np.stack([np.sin(angle), np.cos(angle)], axis=1).astype(np.float32)
    return np.broadcast_to(base[:, :, None, None], (len(times), 2, height, width)).copy()


def _load_static(path: str | None, grid: Grid) -> tuple[np.ndarray, list[str]]:
    if path is None:
        return np.empty((0, *grid.shape), np.float32), []
    with xr.open_dataset(path) as raw:
        dataset = _standardize_names(raw)
        if "static" in dataset:
            data = dataset["static"]
            channel_dim = next(dim for dim in data.dims if dim not in {"lat", "lon"})
            names = [str(value) for value in data[channel_dim].values]
            selected = _interpolate_grid(data, grid, "static").transpose(channel_dim, "lat", "lon")
            values = np.asarray(selected.values, dtype=np.float32)
        else:
            names = [name for name, value in dataset.data_vars.items() if {"lat", "lon"} <= set(value.dims)]
            values = np.stack(
                [
                    np.asarray(
                        _interpolate_grid(dataset[name], grid, f"static {name}").values,
                        dtype=np.float32,
                    )
                    for name in names
                ]
            )
    if not np.isfinite(values).all():
        raise ValueError("static predictors contain non-finite values")
    return values, names


def _condition_chunk(
    times: np.ndarray,
    cpc: xr.DataArray,
    era5: xr.Dataset | None,
    era5_names: tuple[str, ...],
    static: np.ndarray,
    static_names: list[str],
    fine_grid: Grid,
    coarse_grid: Grid,
    area: np.ndarray,
    factor: int,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    cpc_values = np.asarray(
        cpc.reindex(time=times).transpose("time", "lat", "lon").values,
        dtype=np.float32,
    )
    cpc_valid = np.isfinite(cpc_values).astype(np.float32)
    cpc_values = np.sqrt(np.clip(np.nan_to_num(cpc_values, nan=0.0), 0.0, None))
    coarse_parts = [cpc_values[:, None], cpc_valid[:, None]]
    coarse_names = ["sqrt_cpc_precip", "cpc_valid"]
    fine_parts = []
    fine_names = []

    if era5 is not None:
        for name in era5_names:
            if name not in era5:
                raise ValueError(f"ERA5 source is missing {name}")
            field = _interpolate_grid(
                era5[name].reindex(time=times), fine_grid, f"ERA5 {name}"
            ).transpose("time", "lat", "lon")
            values = np.asarray(field.values, dtype=np.float32)
            if not np.isfinite(values).all():
                raise ValueError(f"ERA5 {name} has non-finite values on requested dates")
            fine_parts.append(values[:, None])
            fine_names.append(f"era5_{name}")
            coarse_parts.append(_block_mean_numpy(values[:, None], area, factor))
            coarse_names.append(f"era5_{name}")

    if len(static):
        repeated = np.broadcast_to(static[None], (len(times), *static.shape)).copy()
        fine_parts.append(repeated)
        fine_names.extend(static_names)
        coarse_parts.append(_block_mean_numpy(repeated, area, factor))
        coarse_names.extend(static_names)

    fine_parts.append(_season(times, *fine_grid.shape))
    fine_names.extend(["season_sin", "season_cos"])
    coarse_parts.append(_season(times, *coarse_grid.shape))
    coarse_names.extend(["season_sin", "season_cos"])
    return (
        np.concatenate(coarse_parts, axis=1).astype(np.float32),
        np.concatenate(fine_parts, axis=1).astype(np.float32),
        coarse_names,
        fine_names,
    )


def _zarr_group(path: Path, overwrite: bool):
    import zarr

    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    kwargs = {"mode": "w"}
    try:
        return zarr.open_group(str(path), zarr_format=2, **kwargs)
    except TypeError:
        return zarr.open_group(str(path), zarr_version=2, **kwargs)


def _create(group, name, shape, chunks, dtype="f4", fill_value=0):
    from numcodecs import Blosc

    return group.create_dataset(
        name,
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        fill_value=fill_value,
        compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chirps-glob", required=True)
    parser.add_argument("--cpc-glob", required=True)
    parser.add_argument("--era5-glob")
    parser.add_argument("--era5-vars", default=",".join(ERA5_DEFAULT))
    parser.add_argument("--static")
    parser.add_argument("--out", required=True)
    parser.add_argument("--start", default="1981-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--chunk-days", type=int, default=32)
    parser.add_argument("--train-end", default="2018-12-31")
    parser.add_argument("--wet-threshold", type=float, default=0.1)
    parser.add_argument("--valid-area-threshold", type=float, default=0.50)
    parser.add_argument("--dequant-epsilon", type=float, default=0.02)
    parser.add_argument("--dequant-noise", type=float, default=0.05)
    parser.add_argument("--dequant-seed", type=int, default=314159)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    fine_grid = WIDE_CPC
    factor = 10
    validate_cpc_alignment(fine_grid, factor)
    coarse_grid = Grid(
        name="wide_cpc_native",
        lon_min=fine_grid.lon_min,
        lat_min=fine_grid.lat_min,
        nlon=fine_grid.nlon // factor,
        nlat=fine_grid.nlat // factor,
        res=0.5,
    )
    chirps_ds = _standardize_names(_open_many(_paths(args.chirps_glob, "CHIRPS")))
    cpc_ds = _standardize_names(_open_many(_paths(args.cpc_glob, "CPC")))
    era5_ds = (
        _standardize_names(_open_many(_paths(args.era5_glob, "ERA5")))
        if args.era5_glob else None
    )
    chirps = _exact_grid(
        _variable(chirps_ds, ("precip", "precipitation", "chirps"), "CHIRPS"),
        fine_grid,
        "CHIRPS",
    ).sel(time=slice(args.start, args.end)).transpose("time", "lat", "lon")
    if chirps.sizes.get("time", 0) == 0:
        raise ValueError(f"CHIRPS has no dates in {args.start}..{args.end}")
    cpc = _exact_grid(
        _variable(cpc_ds, ("precip", "precipitation"), "CPC"),
        coarse_grid,
        "native CPC",
    ).transpose("time", "lat", "lon")
    times = np.asarray(chirps.time.values, dtype="datetime64[ns]")
    if len(np.unique(times)) != len(times) or np.any(np.diff(times) <= np.timedelta64(0, "D")):
        raise ValueError("CHIRPS time axis must be unique and increasing")
    expected_start = np.datetime64(args.start, "D")
    expected_end = np.datetime64(args.end, "D")
    days = times.astype("datetime64[D]")
    if days[0] != expected_start or days[-1] != expected_end:
        raise ValueError(
            f"CHIRPS period is {days[0]}..{days[-1]}, expected "
            f"{expected_start}..{expected_end}"
        )
    if np.any(np.diff(days) != np.timedelta64(1, "D")):
        raise ValueError("CHIRPS must contain every calendar day in the requested period")
    static, static_names = _load_static(args.static, fine_grid)
    area = cell_area_weights(fine_grid.lat, fine_grid.lon)
    # A pixel belongs to the physical target support only if every archived
    # day is valid there. Converting an occasional missing value to zero would
    # otherwise teach a false dry event and corrupt the coastal mask.
    fine_valid = np.asarray(chirps.notnull().all("time").compute().values, dtype=bool)
    _, coarse_valid_t, valid_fraction_t = area_weighted_block_mean(
        torch.zeros(1, 1, *fine_grid.shape),
        torch.from_numpy(area),
        torch.from_numpy(fine_valid),
        factor,
        args.valid_area_threshold,
    )
    coarse_valid = coarse_valid_t[0, 0].numpy()
    valid_fraction = valid_fraction_t[0, 0].numpy()
    training = times <= np.datetime64(args.train_end)
    if not training.any():
        raise ValueError("no dates fall in the requested training period")

    # Pass 1: frozen amount and conditioning statistics from training dates.
    roots = []
    coarse_stats = fine_stats = None
    coarse_names = fine_names = None
    era5_names = tuple(value.strip() for value in args.era5_vars.split(",") if value.strip())
    for start in range(0, len(times), args.chunk_days):
        stop = min(start + args.chunk_days, len(times))
        raw = np.asarray(chirps.isel(time=slice(start, stop)).values, np.float32)
        raw = np.clip(np.nan_to_num(raw, nan=0.0), 0.0, None)
        coarse_mm, _, _ = area_weighted_block_mean(
            torch.from_numpy(raw)[:, None], torch.from_numpy(area),
            torch.from_numpy(fine_valid), factor, args.valid_area_threshold,
        )
        local_training = training[start:stop]
        if local_training.any():
            mask = coarse_valid[None] & np.ones((int(local_training.sum()), 1, 1), bool)
            roots.append(np.sqrt(coarse_mm.numpy()[local_training, 0][mask]))
        coarse_cond, fine_cond, coarse_names, fine_names = _condition_chunk(
            times[start:stop], cpc, era5_ds, era5_names, static, static_names,
            fine_grid, coarse_grid, area, factor,
        )
        if coarse_stats is None:
            coarse_stats = RunningChannels(coarse_cond.shape[1])
            fine_stats = RunningChannels(fine_cond.shape[1])
        if local_training.any():
            coarse_stats.update(coarse_cond[local_training])
            fine_stats.update(fine_cond[local_training])
        print(f"statistics {stop}/{len(times)}", flush=True)
    root_values = np.concatenate(roots)
    amount_mean = float(root_values.mean())
    amount_std = float(root_values.std())
    encoding = SubgridEncoding(
        factor=factor,
        wet_threshold_mm=args.wet_threshold,
        dequant_epsilon=args.dequant_epsilon,
        dequant_noise=args.dequant_noise,
        dequant_seed=args.dequant_seed,
        valid_area_threshold=args.valid_area_threshold,
        amount_sqrt_mean=amount_mean,
        amount_sqrt_std=max(amount_std, 1.0e-6),
    )
    coarse_mean, coarse_std = coarse_stats.finish()
    fine_mean, fine_std = fine_stats.finish()
    for index, name in enumerate(coarse_names):
        if name.endswith("valid"):
            coarse_mean[index], coarse_std[index] = 0.0, 1.0

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    root = _zarr_group(output, args.overwrite)
    root.attrs.update(
        schema="cpc_v3_subgrid_v1",
        complete=False,
        subgrid_encoding=encoding_metadata(encoding),
        fine_grid={
            "name": fine_grid.name, "lon_min": fine_grid.lon_min,
            "lat_min": fine_grid.lat_min, "nlon": fine_grid.nlon,
            "nlat": fine_grid.nlat, "res": fine_grid.res,
        },
        coarse_grid={
            "name": coarse_grid.name, "lon_min": coarse_grid.lon_min,
            "lat_min": coarse_grid.lat_min, "nlon": coarse_grid.nlon,
            "nlat": coarse_grid.nlat, "res": coarse_grid.res,
        },
        coarse_cond_channels=coarse_names,
        fine_cond_channels=fine_names,
        coarse_cond_mean=coarse_mean.tolist(),
        coarse_cond_std=coarse_std.tolist(),
        fine_cond_mean=fine_mean.tolist(),
        fine_cond_std=fine_std.tolist(),
        native_cpc=True,
        source_start_date=str(times[0].astype("datetime64[D]")),
        source_end_date=str(times[-1].astype("datetime64[D]")),
        target="CHIRPS 0.05 degree and its exact area-weighted CPC-block mean",
    )
    nt = len(times)
    hc, wc = coarse_grid.shape
    hf, wf = fine_grid.shape
    arrays = {
        "time": _create(root, "time", (nt,), (min(nt, 1024),), "i8"),
        "lat": _create(root, "lat", (hf,), (hf,), "f4"),
        "lon": _create(root, "lon", (wf,), (wf,), "f4"),
        "coarse_lat": _create(root, "coarse_lat", (hc,), (hc,), "f4"),
        "coarse_lon": _create(root, "coarse_lon", (wc,), (wc,), "f4"),
        "fine_valid": _create(root, "fine_valid", (hf, wf), (hf, wf), "b1"),
        "coarse_valid": _create(root, "coarse_valid", (hc, wc), (hc, wc), "b1"),
        "valid_area_fraction": _create(root, "valid_area_fraction", (hc, wc), (hc, wc)),
        "cell_area": _create(root, "cell_area", (hf, wf), (hf, wf)),
        "fine_mm": _create(root, "fine_mm", (nt, hf, wf), (1, hf, wf)),
        "coarse_mm": _create(root, "coarse_mm", (nt, hc, wc), (8, hc, wc)),
        "coarse_state": _create(root, "coarse_state", (nt, 2, hc, wc), (8, 2, hc, wc)),
        "allocation_state": _create(root, "allocation_state", (nt, 2, hf, wf), (1, 2, hf, wf)),
        "coarse_cond": _create(
            root, "coarse_cond", (nt, len(coarse_names), hc, wc),
            (8, len(coarse_names), hc, wc),
        ),
        "fine_cond": _create(
            root, "fine_cond", (nt, len(fine_names), hf, wf),
            (1, len(fine_names), hf, wf),
        ),
    }
    arrays["time"][:] = times.astype("datetime64[ns]").astype(np.int64)
    arrays["lat"][:] = fine_grid.lat
    arrays["lon"][:] = fine_grid.lon
    arrays["coarse_lat"][:] = coarse_grid.lat
    arrays["coarse_lon"][:] = coarse_grid.lon
    arrays["fine_valid"][:] = fine_valid
    arrays["coarse_valid"][:] = coarse_valid
    arrays["valid_area_fraction"][:] = valid_fraction
    arrays["cell_area"][:] = area

    # Pass 2: write chunk-invariant targets and normalized conditions.
    for start in range(0, nt, args.chunk_days):
        stop = min(start + args.chunk_days, nt)
        raw = np.asarray(chirps.isel(time=slice(start, stop)).values, np.float32)
        raw = np.clip(np.nan_to_num(raw, nan=0.0), 0.0, None)
        targets = encode_subgrid_targets(
            torch.from_numpy(raw)[:, None], torch.from_numpy(fine_valid),
            torch.from_numpy(area), encoding, sample_offset=start,
        )
        coarse_cond, fine_cond, _, _ = _condition_chunk(
            times[start:stop], cpc, era5_ds, era5_names, static, static_names,
            fine_grid, coarse_grid, area, factor,
        )
        coarse_cond = (coarse_cond - coarse_mean[None, :, None, None]) / coarse_std[
            None, :, None, None
        ]
        fine_cond = (fine_cond - fine_mean[None, :, None, None]) / fine_std[
            None, :, None, None
        ]
        arrays["fine_mm"][start:stop] = targets.fine_mm[:, 0].numpy()
        arrays["coarse_mm"][start:stop] = targets.coarse_mm[:, 0].numpy()
        arrays["coarse_state"][start:stop] = targets.coarse_state.numpy()
        arrays["allocation_state"][start:stop] = targets.allocation_state.numpy()
        arrays["coarse_cond"][start:stop] = coarse_cond
        arrays["fine_cond"][start:stop] = fine_cond
        print(f"wrote {stop}/{nt}", flush=True)
    root.attrs["complete"] = True
    summary = {
        "output": str(output),
        "dates": [str(times[0].astype("datetime64[D]")), str(times[-1].astype("datetime64[D]"))],
        "days": nt,
        "fine_shape": list(fine_grid.shape),
        "coarse_shape": list(coarse_grid.shape),
        "subgrid_encoding": encoding_metadata(encoding),
        "coarse_cond_channels": coarse_names,
        "fine_cond_channels": fine_names,
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
