#!/usr/bin/env python3
"""Pack ERA5, CHIRPS and static fields into the model-training Zarr store.

This stage intentionally excludes IMERG and gauges. They are observations used
later during assimilation, not predictors or training targets.

The output has fixed time dimensions and is resumable by year. A year is added
to the ``completed_years`` attribute only after its arrays have been validated
and written, so rerunning the same command safely continues an interrupted job.

    python scripts/04_regrid_and_pack.py --start 1981 --end 2025 \
        --out data/processed/bd_wide.zarr

Time conventions
----------------
CHIRPS and the Earthmover ERA5 files both represent 00:00-24:00 UTC day D.
``00_download_era5.py`` has already summed hourly ``tp`` and averaged the five
state variables before this script reads them.

Spatial conventions
-------------------
ERA5 precipitation is conservatively remapped from 0.25 degrees to the exact
0.05-degree WIDE grid with separable spherical cell-overlap weights. State
variables are bilinearly interpolated. CHIRPS is already on a 0.05-degree grid
and is nearest-neighbour selected onto the exact project coordinates.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bdhires.grids import WIDE  # noqa: E402

SCHEMA_VERSION = 1
CORE_MAP = {
    "tp": "era5_tp",
    "tcwv": "era5_tcwv",
    "cape": "era5_cape",
    "u10": "era5_u10",
    "v10": "era5_v10",
    "msl": "era5_msl",
}
CONDITION_CHANNELS = list(CORE_MAP.values())


def _rename_coords(dataset: xr.Dataset) -> xr.Dataset:
    rename = {}
    for old, new in (
        ("latitude", "lat"),
        ("longitude", "lon"),
        ("valid_time", "time"),
    ):
        if old in dataset.coords or old in dataset.dims:
            rename[old] = new
    dataset = dataset.rename(rename)
    if "lat" in dataset.coords and bool((dataset.lat[0] > dataset.lat[-1]).item()):
        dataset = dataset.sortby("lat")
    return dataset


def expected_time(year: int) -> pd.DatetimeIndex:
    return pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")


def coordinate_edges(centres: np.ndarray, *, latitude: bool = False) -> np.ndarray:
    """Infer monotonically increasing cell edges from one-dimensional centres."""
    centres = np.asarray(centres, dtype=np.float64)
    if centres.ndim != 1 or len(centres) < 2:
        raise ValueError("coordinate centres must be a one-dimensional array")
    if not np.all(np.diff(centres) > 0):
        raise ValueError("coordinate centres must be strictly increasing")
    edges = np.empty(len(centres) + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (centres[:-1] + centres[1:])
    edges[0] = centres[0] - 0.5 * (centres[1] - centres[0])
    edges[-1] = centres[-1] + 0.5 * (centres[-1] - centres[-2])
    if latitude:
        if edges[0] < -90.0 - 1e-6 or edges[-1] > 90.0 + 1e-6:
            raise ValueError("latitude edges fall outside -90..90 degrees")
        edges = np.sin(np.deg2rad(np.clip(edges, -90.0, 90.0)))
    return edges


def overlap_weights(
    source_centres: np.ndarray,
    target_centres: np.ndarray,
    *,
    latitude: bool = False,
) -> np.ndarray:
    """Return target-by-source conservative cell-overlap weights."""
    source_edges = coordinate_edges(source_centres, latitude=latitude)
    target_edges = coordinate_edges(target_centres, latitude=latitude)
    left = np.maximum(target_edges[:-1, None], source_edges[None, :-1])
    right = np.minimum(target_edges[1:, None], source_edges[None, 1:])
    overlap = np.maximum(right - left, 0.0)
    target_width = target_edges[1:] - target_edges[:-1]
    coverage = overlap.sum(axis=1)
    if np.any(coverage < target_width * (1.0 - 1e-6)):
        missing = np.where(coverage < target_width * (1.0 - 1e-6))[0]
        raise ValueError(
            "source grid does not cover target cells "
            f"{missing[:5].tolist()}{'...' if len(missing) > 5 else ''}"
        )
    return overlap / coverage[:, None]


def conservative_precipitation(
    data: xr.DataArray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
) -> np.ndarray:
    """Area-average a regular lat/lon precipitation field onto a target grid."""
    data = data.transpose("time", "lat", "lon")
    source = np.asarray(data.values, dtype=np.float32)
    if not np.isfinite(source).all():
        raise ValueError("ERA5 precipitation contains non-finite source values")
    source = np.clip(source, 0.0, None)

    lat_weights = overlap_weights(data.lat.values, target_lat, latitude=True)
    lon_weights = overlap_weights(data.lon.values, target_lon)
    intermediate = np.einsum(
        "ys,tsl->tyl", lat_weights, source, optimize=True
    )
    result = np.einsum(
        "xl,tyl->tyx", lon_weights, intermediate, optimize=True
    )
    return result.astype(np.float32)


def validate_time_axis(dataset: xr.Dataset, year: int, path: Path) -> None:
    if "time" not in dataset.coords:
        raise ValueError(f"{path} does not contain a time coordinate")
    actual = pd.DatetimeIndex(dataset.time.values)
    expected = expected_time(year)
    if not actual.equals(expected):
        raise ValueError(
            f"{path} has an incomplete time axis: expected {len(expected)} "
            f"days {expected[0].date()}..{expected[-1].date()}, found "
            f"{len(actual)} days"
        )


def validate_source_inventory(
    years: list[int],
    era5_dir: Path,
    chirps_dir: Path,
) -> tuple[dict[int, Path], dict[int, Path]]:
    """Validate metadata and edge records for every requested raw-data year."""
    era5_paths = {year: era5_dir / f"era5_daily_{year}.nc" for year in years}
    chirps_paths = {
        year: chirps_dir / f"chirps_wide_{year}.nc" for year in years
    }
    missing_era5 = [year for year, path in era5_paths.items() if not path.is_file()]
    missing_chirps = [
        year for year, path in chirps_paths.items() if not path.is_file()
    ]
    if missing_era5 or missing_chirps:
        raise FileNotFoundError(
            "raw-data inventory is incomplete; "
            f"missing ERA5 years={missing_era5}, "
            f"missing CHIRPS years={missing_chirps}"
        )

    reference_lat = reference_lon = None
    for year in years:
        with xr.open_dataset(era5_paths[year]) as raw:
            dataset = _rename_coords(raw)
            validate_time_axis(dataset, year, era5_paths[year])
            missing = set(CORE_MAP) - set(dataset.data_vars)
            if missing:
                raise ValueError(
                    f"{era5_paths[year]} is missing ERA5 variables "
                    f"{sorted(missing)}"
                )
            if dataset.attrs.get("temporal_resolution") != "daily":
                raise ValueError(
                    f"{era5_paths[year]} is not an Earthmover daily ERA5 file"
                )
            if reference_lat is None:
                reference_lat = dataset.lat.values.copy()
                reference_lon = dataset.lon.values.copy()
                overlap_weights(reference_lat, WIDE.lat, latitude=True)
                overlap_weights(reference_lon, WIDE.lon)
            else:
                np.testing.assert_allclose(dataset.lat.values, reference_lat)
                np.testing.assert_allclose(dataset.lon.values, reference_lon)
            edges = dataset[list(CORE_MAP)].isel(time=[0, -1]).load()
            for variable in CORE_MAP:
                if not np.isfinite(edges[variable].values).all():
                    raise ValueError(
                        f"{era5_paths[year]} has non-finite {variable} edge data"
                    )

        with xr.open_dataset(chirps_paths[year]) as raw:
            dataset = _rename_coords(raw)
            validate_time_axis(dataset, year, chirps_paths[year])
            if "precip" not in dataset:
                raise ValueError(
                    f"{chirps_paths[year]} does not contain precipitation"
                )
            edge = dataset["precip"].isel(time=[0, -1]).load()
            if not np.isfinite(edge.values).any():
                raise ValueError(
                    f"{chirps_paths[year]} has no finite edge precipitation"
                )
        print(f"inventory OK: {year}", flush=True)
    return era5_paths, chirps_paths


def load_static(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    with xr.open_dataset(path) as dataset:
        required = {"static", "valid"}
        missing = required - set(dataset.data_vars)
        if missing:
            raise ValueError(f"{path} is missing variables {sorted(missing)}")
        static = dataset["static"].values.astype(np.float32)
        valid = dataset["valid"].values.astype(np.float32)
        channels = list(map(str, dataset.channel.values))
        np.testing.assert_allclose(dataset.lat.values, WIDE.lat, atol=1e-6)
        np.testing.assert_allclose(dataset.lon.values, WIDE.lon, atol=1e-6)
    if static.shape[1:] != WIDE.shape or valid.shape != WIDE.shape:
        raise ValueError(
            f"{path} has static/valid shapes {static.shape}/{valid.shape}, "
            f"expected (C, {WIDE.nlat}, {WIDE.nlon})/{WIDE.shape}"
        )
    if not np.isfinite(static).all() or not np.isfinite(valid).all():
        raise ValueError(f"{path} contains non-finite static data")
    if not np.any(valid > 0.5):
        raise ValueError(f"{path} has an empty land-validity mask")
    return static, valid, channels


def open_zarr_group(path: Path, mode: str):
    import zarr

    try:
        return zarr.open_group(str(path), mode=mode, zarr_format=2)
    except TypeError:
        return zarr.open_group(str(path), mode=mode, zarr_version=2)


def ensure_array(
    group,
    name: str,
    *,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    dtype: str,
    fill_value=0,
):
    from numcodecs import Blosc

    if name in group:
        array = group[name]
        if tuple(array.shape) != tuple(shape):
            raise ValueError(
                f"existing {name} shape {array.shape} does not match {shape}"
            )
        if np.dtype(array.dtype) != np.dtype(dtype):
            raise ValueError(
                f"existing {name} dtype {array.dtype} does not match {dtype}"
            )
        return array
    return group.create_dataset(
        name,
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        compressor=Blosc(
            cname="zstd",
            clevel=3,
            shuffle=Blosc.BITSHUFFLE,
        ),
        fill_value=fill_value,
    )


def initialize_store(
    output: Path,
    days: pd.DatetimeIndex,
    years: list[int],
    static: np.ndarray,
    valid: np.ndarray,
    static_channels: list[str],
):
    output.parent.mkdir(parents=True, exist_ok=True)
    existed = output.exists()
    root = open_zarr_group(output, "a")
    if existed and "schema_version" not in root.attrs:
        raise ValueError(
            f"{output} is an older or unrecognized store. Move it aside or "
            "choose a different --out path before running the resumable packer."
        )
    if "schema_version" in root.attrs:
        expected = {
            "schema_version": SCHEMA_VERSION,
            "start_year": years[0],
            "end_year": years[-1],
            "cond_channels": CONDITION_CHANNELS,
            "static_channels": static_channels,
        }
        for key, value in expected.items():
            if root.attrs.get(key) != value:
                raise ValueError(
                    f"{output} attribute {key}={root.attrs.get(key)!r}, "
                    f"expected {value!r}"
                )
    else:
        root.attrs.update(
            schema_version=SCHEMA_VERSION,
            start_year=years[0],
            end_year=years[-1],
            completed_years=[],
            complete=False,
            cond_channels=CONDITION_CHANNELS,
            static_channels=static_channels,
            era5_tp_cond_index=0,
            observations=(
                "not included; IMERG and gauges are added during assimilation"
            ),
            grid={
                "name": WIDE.name,
                "lon_min": WIDE.lon_min,
                "lat_min": WIDE.lat_min,
                "nlon": WIDE.nlon,
                "nlat": WIDE.nlat,
                "res": WIDE.res,
            },
        )

    ntime = len(days)
    arrays = {
        "time": ensure_array(
            root,
            "time",
            shape=(ntime,),
            chunks=(min(1024, ntime),),
            dtype="i8",
        ),
        "lat": ensure_array(
            root,
            "lat",
            shape=(WIDE.nlat,),
            chunks=(WIDE.nlat,),
            dtype="f4",
        ),
        "lon": ensure_array(
            root,
            "lon",
            shape=(WIDE.nlon,),
            chunks=(WIDE.nlon,),
            dtype="f4",
        ),
        "static": ensure_array(
            root,
            "static",
            shape=static.shape,
            chunks=(1, WIDE.nlat, WIDE.nlon),
            dtype="f4",
        ),
        "valid": ensure_array(
            root,
            "valid",
            shape=valid.shape,
            chunks=WIDE.shape,
            dtype="f4",
        ),
        "target": ensure_array(
            root,
            "target",
            shape=(ntime, *WIDE.shape),
            chunks=(1, *WIDE.shape),
            dtype="f4",
            fill_value=np.nan,
        ),
        "cond": ensure_array(
            root,
            "cond",
            shape=(ntime, len(CONDITION_CHANNELS), *WIDE.shape),
            chunks=(1, len(CONDITION_CHANNELS), *WIDE.shape),
            dtype="f4",
            fill_value=np.nan,
        ),
    }
    arrays["time"][:] = days.values.astype("datetime64[ns]").view("i8")
    arrays["lat"][:] = WIDE.lat.astype(np.float32)
    arrays["lon"][:] = WIDE.lon.astype(np.float32)
    arrays["static"][:] = static
    arrays["valid"][:] = valid
    root.attrs["time_units"] = "nanoseconds since 1970-01-01"
    return root, arrays


def load_year(
    era5_path: Path,
    chirps_path: Path,
    year: int,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    days = expected_time(year)
    with xr.open_dataset(chirps_path) as raw:
        dataset = _rename_coords(raw)
        target = dataset["precip"].interp(
            lat=WIDE.lat,
            lon=WIDE.lon,
            method="nearest",
        )
        target = target.reindex(time=days).transpose("time", "lat", "lon")
        target_values = target.values.astype(np.float32)

    with xr.open_dataset(era5_path) as raw:
        dataset = _rename_coords(raw)
        dataset = dataset.reindex(time=days)
        precipitation = conservative_precipitation(
            dataset["tp"],
            WIDE.lat,
            WIDE.lon,
        )
        channels = [precipitation]
        for variable in list(CORE_MAP)[1:]:
            field = dataset[variable].interp(
                lat=WIDE.lat,
                lon=WIDE.lon,
                method="linear",
            )
            channels.append(
                field.transpose("time", "lat", "lon").values.astype(np.float32)
            )
        conditions = np.stack(channels, axis=1).astype(np.float32)

    expected_target_shape = (len(days), *WIDE.shape)
    expected_cond_shape = (
        len(days),
        len(CONDITION_CHANNELS),
        *WIDE.shape,
    )
    if target_values.shape != expected_target_shape:
        raise ValueError(
            f"{chirps_path} produced {target_values.shape}, "
            f"expected {expected_target_shape}"
        )
    if conditions.shape != expected_cond_shape:
        raise ValueError(
            f"{era5_path} produced {conditions.shape}, "
            f"expected {expected_cond_shape}"
        )
    if not np.isfinite(conditions).all():
        raise ValueError(f"{era5_path} produced non-finite condition values")
    land_values = target_values[:, valid > 0.5]
    if not np.isfinite(land_values).all():
        missing = int((~np.isfinite(land_values)).sum())
        raise ValueError(
            f"{chirps_path} has {missing} missing target values over valid land"
        )
    if np.nanmin(target_values) < 0:
        raise ValueError(f"{chirps_path} contains negative precipitation")
    return target_values, conditions


def validate_completed_edge(
    arrays: dict,
    start: int,
    stop: int,
    valid: np.ndarray,
) -> None:
    for index in (start, stop - 1):
        conditions = np.asarray(arrays["cond"][index])
        target = np.asarray(arrays["target"][index])
        if not np.isfinite(conditions).all():
            raise ValueError(f"stored conditions are invalid at time index {index}")
        if not np.isfinite(target[valid > 0.5]).all():
            raise ValueError(f"stored target is invalid at time index {index}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1981)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--era5", default="data/raw/era5")
    parser.add_argument("--chirps", default="data/raw/chirps")
    parser.add_argument("--static", default="data/static/static_wide.nc")
    parser.add_argument("--out", default="data/processed/bd_wide.zarr")
    args = parser.parse_args()
    if args.start > args.end:
        parser.error("--start must be less than or equal to --end")

    years = list(range(args.start, args.end + 1))
    days = pd.date_range(f"{args.start}-01-01", f"{args.end}-12-31", freq="D")
    era5_paths, chirps_paths = validate_source_inventory(
        years,
        Path(args.era5),
        Path(args.chirps),
    )
    static, valid, static_channels = load_static(Path(args.static))
    root, arrays = initialize_store(
        Path(args.out),
        days,
        years,
        static,
        valid,
        static_channels,
    )

    completed = {int(year) for year in root.attrs.get("completed_years", [])}
    summaries = json.loads(root.attrs.get("year_summaries_json", "{}"))
    all_years = days.year.to_numpy()
    for year in years:
        indices = np.flatnonzero(all_years == year)
        start = int(indices[0])
        stop = int(indices[-1]) + 1
        if year in completed:
            validate_completed_edge(arrays, start, stop, valid)
            print(f"already packed: {year}", flush=True)
            continue

        print(f"packing {year}: time indices {start}:{stop}", flush=True)
        target, conditions = load_year(
            era5_paths[year],
            chirps_paths[year],
            year,
            valid,
        )
        arrays["target"][start:stop] = target
        arrays["cond"][start:stop] = conditions
        validate_completed_edge(arrays, start, stop, valid)

        summaries[str(year)] = {
            "days": int(stop - start),
            "target_land_mean_mm_day": float(
                np.nanmean(target[:, valid > 0.5])
            ),
            "era5_tp_land_mean_mm_day": float(
                conditions[:, 0, valid > 0.5].mean()
            ),
        }
        completed.add(year)
        root.attrs["year_summaries_json"] = json.dumps(
            summaries,
            sort_keys=True,
        )
        root.attrs["completed_years"] = sorted(completed)
        print(f"completed {year}", flush=True)

    missing = sorted(set(years) - completed)
    if missing:
        raise RuntimeError(f"packed store is still missing years {missing}")
    root.attrs["complete"] = True

    try:
        import zarr

        zarr.consolidate_metadata(str(args.out))
    except Exception as exc:
        print(f"warning: metadata consolidation was unavailable: {exc}", flush=True)

    print(
        f"wrote {args.out}: T={len(days)}, "
        f"ERA5 channels={len(CONDITION_CHANNELS)}, years={years[0]}-{years[-1]}",
        flush=True,
    )
    print("condition channels:", CONDITION_CHANNELS, flush=True)
    print("IMERG and gauges are not stored in the training dataset.", flush=True)


if __name__ == "__main__":
    main()
