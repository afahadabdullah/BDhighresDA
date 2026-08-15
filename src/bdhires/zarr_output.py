"""Restart-safe, xarray-compatible Zarr output for physical DA ensembles."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def _group(path: Path, mode: str):
    import zarr

    try:
        return zarr.open_group(str(path), mode=mode, zarr_format=2)
    except TypeError:
        return zarr.open_group(str(path), mode=mode, zarr_version=2)


def _array(
    group,
    name: str,
    values: np.ndarray,
    dimensions: tuple[str, ...],
    chunks: tuple[int, ...] | None = None,
    attrs: dict | None = None,
):
    from numcodecs import Blosc

    values = np.asarray(values)
    # Coordinate zero, physical zero-rain and False validity are real values,
    # not missing-data sentinels. Without an explicit None, Zarr chooses zero
    # as fill_value and xarray masks every legitimate zero on read.
    # Zarr 3's compatibility API requires ``shape`` even when ``data`` is
    # supplied, while Zarr 2 accepts both. The store itself remains v2.
    kwargs = {"data": values, "shape": values.shape, "fill_value": None}
    if values.ndim:
        kwargs["chunks"] = chunks or values.shape
        if values.dtype.kind not in {"U", "S", "O"}:
            kwargs["compressor"] = Blosc(
                cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE
            )
    array = group.create_dataset(name, **kwargs)
    array.attrs["_ARRAY_DIMENSIONS"] = list(dimensions)
    if attrs:
        array.attrs.update(attrs)
    return array


def write_physical_ensemble_zarr(
    path: str | Path,
    *,
    fields: dict[str, np.ndarray],
    method_specs: dict[str, dict],
    selected_times: np.ndarray,
    grid,
    valid: np.ndarray,
    condition: np.ndarray,
    chirps: np.ndarray,
    raw_imerg_mm: np.ndarray | None,
    imerg_factor: int,
    station_ids: np.ndarray,
    station_lat: np.ndarray,
    station_lon: np.ndarray,
    gauge_mm: np.ndarray,
    assim_idx: np.ndarray,
    scope: dict,
) -> None:
    """Atomically write physical ensembles and matched inputs as Zarr v2.

    ``precipitation`` has dimensions ``method,time,member,lat,lon`` and retains
    every ensemble member. ``ensemble_mean`` and ``ensemble_std`` are convenient
    derived views. Permanently masked ocean cells remain NaN by design.

    The store is built under ``.incomplete`` and renamed only after consolidated
    metadata and the ``complete`` attribute have been written. Existing final or
    incomplete stores are never overwritten implicitly.
    """
    import zarr
    from numcodecs import Blosc

    output = Path(path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite completed field store {output}")
    temporary = output.with_name(output.name + ".incomplete")
    if temporary.exists():
        raise FileExistsError(
            f"incomplete field store exists: {temporary}; inspect or move it "
            "before retrying"
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    methods = list(method_specs)
    if not methods:
        raise ValueError("at least one method is required for a field archive")
    if set(methods) != set(fields):
        raise ValueError(
            f"field methods {sorted(fields)} disagree with specs {sorted(methods)}"
        )
    first = np.asarray(fields[methods[0]])
    if first.ndim != 4:
        raise ValueError(f"expected (time,member,lat,lon), got {first.shape}")
    n_time, n_member, nlat, nlon = first.shape
    if n_member < 2:
        raise ValueError("an ensemble archive requires at least two members")
    if (nlat, nlon) != tuple(valid.shape):
        raise ValueError(f"field grid {(nlat, nlon)} != valid mask {valid.shape}")
    valid = np.asarray(valid, bool)
    times = np.asarray(selected_times).astype("datetime64[D]")
    if len(times) != n_time or len(np.unique(times)) != n_time:
        raise ValueError("selected_times must be unique and match the field time axis")
    if n_time > 1 and not np.all(np.diff(times) == np.timedelta64(1, "D")):
        raise ValueError("a seasonal shard must contain a contiguous daily time axis")
    if str(times[0]) != str(scope.get("start")) or str(times[-1]) != str(scope.get("end")):
        raise ValueError("field dates disagree with scope start/end")
    for label, values in (("condition", condition), ("chirps", chirps)):
        if np.asarray(values).shape != (n_time, nlat, nlon):
            raise ValueError(
                f"{label} shape {np.asarray(values).shape} != "
                f"{(n_time, nlat, nlon)}"
            )
    station_ids = np.asarray(station_ids).astype(str)
    n_station = len(station_ids)
    if not (
        np.asarray(station_lat).shape == (n_station,)
        and np.asarray(station_lon).shape == (n_station,)
        and np.asarray(gauge_mm).shape == (n_time, n_station)
    ):
        raise ValueError("station coordinates and gauge array have inconsistent shapes")
    assimilated_indices = np.asarray(assim_idx, int)
    if (
        len(np.unique(assimilated_indices)) != len(assimilated_indices)
        or np.any(assimilated_indices < 0)
        or np.any(assimilated_indices >= n_station)
    ):
        raise ValueError("assim_idx contains duplicate or out-of-range stations")
    if raw_imerg_mm is not None and np.asarray(raw_imerg_mm).shape[0] != n_time:
        raise ValueError("IMERG time axis does not match the physical fields")
    for name, values in fields.items():
        values = np.asarray(values)
        if values.shape != first.shape:
            raise ValueError(f"{name} field shape {values.shape} != {first.shape}")
        if not np.isfinite(values[:, :, valid]).all():
            raise FloatingPointError(f"{name} contains non-finite land values")

    root = _group(temporary, "w")
    root.attrs.update(
        schema="bdhires.physical_ensemble.v1",
        complete=False,
        units="mm/day",
        description=(
            "CPC-v2 physical precipitation ensembles. Cross-validated skill "
            "must use the separate withheld-station files; assimilated_station "
            "declares which gauges entered this gridded product."
        ),
        scope=scope,
        method_specs=method_specs,
    )

    max_method = max(len(name) for name in methods)
    max_station = max(1, max(len(value) for value in station_ids))
    days_since_epoch = (
        np.asarray(selected_times).astype("datetime64[D]")
        - np.datetime64("1970-01-01", "D")
    ).astype(np.int32)
    _array(root, "method", np.asarray(methods, dtype=f"U{max_method}"), ("method",))
    _array(
        root,
        "time",
        days_since_epoch,
        ("time",),
        attrs={
            "units": "days since 1970-01-01",
            "calendar": "proleptic_gregorian",
        },
    )
    _array(root, "member", np.arange(n_member, dtype=np.int16), ("member",))
    _array(
        root,
        "lat",
        np.asarray(grid.lat, np.float32),
        ("lat",),
        attrs={"units": "degrees_north"},
    )
    _array(
        root,
        "lon",
        np.asarray(grid.lon, np.float32),
        ("lon",),
        attrs={"units": "degrees_east"},
    )
    _array(root, "valid", np.asarray(valid, bool), ("lat", "lon"), chunks=(nlat, nlon))

    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    ensemble = root.create_dataset(
        "precipitation",
        shape=(len(methods), n_time, n_member, nlat, nlon),
        chunks=(1, 1, min(5, n_member), nlat, nlon),
        dtype="f4",
        fill_value=np.nan,
        compressor=compressor,
    )
    ensemble.attrs.update(
        _ARRAY_DIMENSIONS=["method", "time", "member", "lat", "lon"],
        units="mm/day",
        long_name="daily physical precipitation ensemble",
    )
    ensemble_mean = root.create_dataset(
        "ensemble_mean",
        shape=(len(methods), n_time, nlat, nlon),
        chunks=(1, 1, nlat, nlon),
        dtype="f4",
        fill_value=np.nan,
        compressor=compressor,
    )
    ensemble_mean.attrs.update(
        _ARRAY_DIMENSIONS=["method", "time", "lat", "lon"], units="mm/day"
    )
    ensemble_std = root.create_dataset(
        "ensemble_std",
        shape=(len(methods), n_time, nlat, nlon),
        chunks=(1, 1, nlat, nlon),
        dtype="f4",
        fill_value=np.nan,
        compressor=compressor,
    )
    ensemble_std.attrs.update(
        _ARRAY_DIMENSIONS=["method", "time", "lat", "lon"],
        units="mm/day",
        long_name="within-day ensemble standard deviation",
    )
    for index, name in enumerate(methods):
        values = np.asarray(fields[name], np.float32)
        if values.shape != first.shape:
            raise ValueError(f"{name} field shape {values.shape} != {first.shape}")
        ensemble[index] = values
        land_values = values[:, :, valid]
        mean = np.full((n_time, nlat, nlon), np.nan, np.float32)
        std = np.full_like(mean, np.nan)
        mean[:, valid] = np.mean(land_values, axis=1)
        std[:, valid] = np.std(land_values, axis=1, ddof=1)
        ensemble_mean[index] = mean
        ensemble_std[index] = std

    _array(
        root,
        "cpc",
        np.asarray(condition, np.float32),
        ("time", "lat", "lon"),
        chunks=(1, nlat, nlon),
        attrs={"units": "mm/day"},
    )
    _array(
        root,
        "chirps",
        np.asarray(chirps, np.float32),
        ("time", "lat", "lon"),
        chunks=(1, nlat, nlon),
        attrs={"units": "mm/day"},
    )
    _array(root, "station", np.arange(len(station_ids), dtype=np.int16), ("station",))
    _array(
        root,
        "station_id",
        np.asarray(station_ids, dtype=f"U{max_station}"),
        ("station",),
    )
    _array(
        root,
        "station_lat",
        np.asarray(station_lat, np.float32),
        ("station",),
        attrs={"units": "degrees_north"},
    )
    _array(
        root,
        "station_lon",
        np.asarray(station_lon, np.float32),
        ("station",),
        attrs={"units": "degrees_east"},
    )
    assimilated = np.zeros(len(station_ids), dtype=bool)
    assimilated[assimilated_indices] = True
    _array(root, "assimilated_station", assimilated, ("station",))
    _array(
        root,
        "gauge",
        np.asarray(gauge_mm, np.float32),
        ("time", "station"),
        chunks=(min(31, n_time), len(station_ids)),
        attrs={"units": "mm/day"},
    )

    if raw_imerg_mm is not None:
        n_imerg_lat, n_imerg_lon = np.asarray(raw_imerg_mm).shape[1:]
        imerg_lat = grid.lat_min + grid.res * imerg_factor * (
            np.arange(n_imerg_lat) + 0.5
        )
        imerg_lon = grid.lon_min + grid.res * imerg_factor * (
            np.arange(n_imerg_lon) + 0.5
        )
        _array(
            root,
            "imerg_lat",
            np.asarray(imerg_lat, np.float32),
            ("imerg_lat",),
            attrs={"units": "degrees_north"},
        )
        _array(
            root,
            "imerg_lon",
            np.asarray(imerg_lon, np.float32),
            ("imerg_lon",),
            attrs={"units": "degrees_east"},
        )
        _array(
            root,
            "imerg",
            np.asarray(raw_imerg_mm, np.float32),
            ("time", "imerg_lat", "imerg_lon"),
            chunks=(1, n_imerg_lat, n_imerg_lon),
            attrs={"units": "mm/day", "spatial_support": "0.4-degree S04"},
        )

    root.attrs["complete"] = True
    zarr.consolidate_metadata(str(temporary))
    os.replace(temporary, output)


def write_hierarchical_sample_zarr(
    path: str | Path,
    *,
    fields: dict[str, np.ndarray],
    coarse_states: dict[str, np.ndarray],
    allocation_states: dict[str, np.ndarray],
    selected_times: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    valid: np.ndarray,
    diagnostics: dict[str, dict],
    coarse_mm: dict[str, np.ndarray] | None = None,
    method_specs: dict[str, dict] | None = None,
    target_crop: tuple[int, int, int, int] | None = None,
    serialization_tolerance_mm_day: float = 1.0e-5,
) -> None:
    """Atomically write V3 physical samples and both latent states.

    This is the sole supported writer for the archive consumed by
    ``scripts/58_evaluate_subgrid_prior.py``.  Every method must carry the
    sampler's terminal hard-decoder diagnostic.  The temporary store is then
    reopened and compared with the in-memory physical fields before it can be
    marked complete, preventing a stale/soft/redecoded field from being
    presented as the analysis used by the likelihood.
    """
    import zarr

    output = Path(path)
    temporary = output.with_name(output.name + ".incomplete")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite completed sample store {output}")
    if temporary.exists():
        raise FileExistsError(f"incomplete sample store already exists: {temporary}")
    output.parent.mkdir(parents=True, exist_ok=True)

    methods = list(fields)
    if not methods:
        raise ValueError("at least one hierarchical sample method is required")
    expected = set(methods)
    for label, values in (
        ("coarse_states", coarse_states),
        ("allocation_states", allocation_states),
        ("diagnostics", diagnostics),
    ):
        if set(values) != expected:
            raise ValueError(f"{label} methods differ from physical fields")
    if coarse_mm is not None and set(coarse_mm) != expected:
        raise ValueError("coarse_mm methods differ from physical fields")
    if method_specs is not None and set(method_specs) != expected:
        raise ValueError("method_specs methods differ from physical fields")
    if serialization_tolerance_mm_day < 0.0:
        raise ValueError("serialization tolerance must be non-negative")

    first = np.asarray(fields[methods[0]])
    if first.ndim != 4:
        raise ValueError("physical fields must have shape (time,member,lat,lon)")
    n_time, n_member, nlat, nlon = first.shape
    valid = np.asarray(valid, bool)
    lat = np.asarray(lat, np.float32)
    lon = np.asarray(lon, np.float32)
    times = np.asarray(selected_times).astype("datetime64[ns]")
    if valid.shape != (nlat, nlon) or lat.shape != (nlat,) or lon.shape != (nlon,):
        raise ValueError("valid/latitude/longitude geometry differs from physical fields")
    if times.shape != (n_time,) or len(np.unique(times)) != n_time:
        raise ValueError("sample times must be unique and match the field time axis")
    if n_time > 1 and not np.all(np.diff(times.astype(np.int64)) > 0):
        raise ValueError("sample times must be strictly increasing")

    for method in methods:
        physical = np.asarray(fields[method])
        coarse = np.asarray(coarse_states[method])
        allocation = np.asarray(allocation_states[method])
        if physical.shape != first.shape:
            raise ValueError(f"{method} physical shape {physical.shape} != {first.shape}")
        if coarse.ndim != 5 or coarse.shape[:3] != (n_time, n_member, 2):
            raise ValueError(f"{method} coarse state must be (T,M,2,Hc,Wc)")
        if allocation.shape != (n_time, n_member, 2, nlat, nlon):
            raise ValueError(f"{method} allocation state has the wrong shape")
        if (
            nlat % coarse.shape[-2]
            or nlon % coarse.shape[-1]
            or nlat // coarse.shape[-2] != nlon // coarse.shape[-1]
        ):
            raise ValueError(f"{method} coarse/fine state geometry is not block aligned")
        if not np.isfinite(physical[:, :, valid]).all():
            raise FloatingPointError(f"{method} has non-finite values on valid cells")
        if np.isinf(physical).any():
            raise FloatingPointError(f"{method} contains infinite values")
        terminal_error = diagnostics[method].get("terminal_hard_decode_max_abs_mm_day")
        if (
            diagnostics[method].get("terminal_decoder_consistent") is not True
            or terminal_error is None
            or not np.isfinite(float(terminal_error))
            or float(terminal_error) > serialization_tolerance_mm_day
        ):
            raise ValueError(
                f"{method} lacks a passing sampler terminal hard-decoder diagnostic"
            )
        if coarse_mm is not None:
            expected_coarse = (n_time, n_member, *coarse.shape[-2:])
            if np.asarray(coarse_mm[method]).shape != expected_coarse:
                raise ValueError(f"{method} coarse_mm shape must be {expected_coarse}")

    root = _group(temporary, "w")
    root.attrs.update(
        schema="cpc_v3_hierarchical_samples_v1",
        complete=False,
        units="mm/day",
        archive_uses_likelihood_hard_decoder=False,
        method_specs=method_specs or {method: {} for method in methods},
        sampler_diagnostics=diagnostics,
    )
    if target_crop is not None:
        if len(target_crop) != 4:
            raise ValueError("target_crop must be (row_start,row_stop,col_start,col_stop)")
        root.attrs["target_crop"] = [int(value) for value in target_crop]
    _array(root, "time", times.astype(np.int64), ("time",))
    _array(root, "member", np.arange(n_member, dtype=np.int16), ("member",))
    _array(root, "lat", lat, ("lat",), attrs={"units": "degrees_north"})
    _array(root, "lon", lon, ("lon",), attrs={"units": "degrees_east"})
    _array(root, "valid", valid, ("lat", "lon"), chunks=(nlat, nlon))
    for method in methods:
        coarse = np.asarray(coarse_states[method], np.float32)
        _array(
            root, method, np.asarray(fields[method], np.float32),
            ("time", "member", "lat", "lon"),
            chunks=(1, min(5, n_member), nlat, nlon),
            attrs={"units": "mm/day"},
        )
        _array(
            root, f"{method}_coarse_state", coarse,
            ("time", "member", "coarse_channel", "coarse_lat", "coarse_lon"),
            chunks=(1, min(5, n_member), 2, *coarse.shape[-2:]),
        )
        _array(
            root, f"{method}_allocation_state",
            np.asarray(allocation_states[method], np.float32),
            ("time", "member", "allocation_channel", "lat", "lon"),
            chunks=(1, min(5, n_member), 2, nlat, nlon),
        )
        if coarse_mm is not None:
            _array(
                root, f"{method}_coarse_mm", np.asarray(coarse_mm[method], np.float32),
                ("time", "member", "coarse_lat", "coarse_lon"),
                chunks=(1, min(5, n_member), *coarse.shape[-2:]),
                attrs={"units": "mm/day"},
            )

    # Audit the exact values that the evaluator will reopen.  Do this before
    # setting complete=True and before the atomic rename.
    reopened = _group(temporary, "r")
    roundtrip_max = 0.0
    for method in methods:
        source = np.asarray(fields[method], np.float32)
        stored = reopened[method]
        for index in range(n_time):
            stored_chunk = np.asarray(stored[index])
            source_chunk = source[index]
            if not np.array_equal(np.isnan(stored_chunk), np.isnan(source_chunk)):
                raise RuntimeError(f"{method} NaN mask changed during serialization")
            finite = np.isfinite(source_chunk)
            error = (
                np.max(np.abs(stored_chunk[finite] - source_chunk[finite]))
                if finite.any() else 0.0
            )
            roundtrip_max = max(roundtrip_max, float(error))
    if not np.isfinite(roundtrip_max) or roundtrip_max > serialization_tolerance_mm_day:
        raise RuntimeError(
            f"hierarchical archive round-trip error {roundtrip_max:.3g} mm/day "
            f"exceeds {serialization_tolerance_mm_day:.3g}"
        )
    root.attrs["serialization_max_abs_mm_day"] = roundtrip_max
    root.attrs["archive_uses_likelihood_hard_decoder"] = True
    root.attrs["complete"] = True
    zarr.consolidate_metadata(str(temporary))
    os.replace(temporary, output)
