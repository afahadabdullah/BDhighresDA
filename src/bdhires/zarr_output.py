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
    coarse_valid: np.ndarray,
    cell_area: np.ndarray,
    encoding,
    diagnostics: dict[str, dict],
    coarse_mm: dict[str, np.ndarray] | None = None,
    method_specs: dict[str, dict] | None = None,
    target_crop: tuple[int, int, int, int] | None = None,
    serialization_tolerance_mm_day: float = 5.0e-4,
    hard_decode_relative_tolerance: float = 5.0e-6,
    allow_legacy_v2_encoding: bool = False,
) -> None:
    """Atomically write V3 physical samples and both latent states.

    This is the sole supported writer for the archive consumed by
    ``scripts/58_evaluate_subgrid_prior.py``.  The temporary store is reopened
    and its serialized latent states are hard decoded independently. Only a
    physical field matching that saved-state decode can be marked complete,
    preventing a stale or soft field from being presented as the analysis used
    by the likelihood. The explicit legacy-v2 opt-in is reserved for labelled
    diagnostic archives, which the current evaluator still rejects.
    """
    import zarr
    import torch

    from .data.subgrid_dataset import (
        LegacyV2SubgridEncoding,
        SubgridEncoding,
        decode_and_reconstruct,
        decode_coarse_amount,
        encoding_metadata,
    )

    if isinstance(encoding, LegacyV2SubgridEncoding):
        if not allow_legacy_v2_encoding:
            raise ValueError(
                "legacy-v2 encoding is permitted only for an explicitly labelled "
                "diagnostic archive"
            )
        frozen_encoding = encoding
    elif isinstance(encoding, SubgridEncoding):
        frozen_encoding = encoding
    else:
        frozen_encoding = SubgridEncoding.from_mapping(encoding)
    frozen_encoding.validate()

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
    if hard_decode_relative_tolerance < 0.0:
        raise ValueError("hard-decode relative tolerance must be non-negative")

    first = np.asarray(fields[methods[0]])
    if first.ndim != 4:
        raise ValueError("physical fields must have shape (time,member,lat,lon)")
    n_time, n_member, nlat, nlon = first.shape
    valid = np.asarray(valid, bool)
    coarse_valid = np.asarray(coarse_valid, bool)
    cell_area = np.asarray(cell_area, np.float32)
    lat = np.asarray(lat, np.float32)
    lon = np.asarray(lon, np.float32)
    times = np.asarray(selected_times).astype("datetime64[ns]")
    if valid.shape != (nlat, nlon) or lat.shape != (nlat,) or lon.shape != (nlon,):
        raise ValueError("valid/latitude/longitude geometry differs from physical fields")
    if cell_area.shape != (nlat, nlon) or not np.isfinite(cell_area).all():
        raise ValueError("cell_area must be finite and match the physical field geometry")
    if np.any(cell_area <= 0.0):
        raise ValueError("cell_area must be strictly positive")
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
        if coarse_valid.shape != coarse.shape[-2:]:
            raise ValueError(f"{method} coarse_valid shape differs from its coarse state")
        if nlat // coarse.shape[-2] != frozen_encoding.factor:
            raise ValueError(
                f"{method} state factor differs from encoding factor "
                f"{frozen_encoding.factor}"
            )
        if not np.isfinite(physical[:, :, valid]).all():
            raise FloatingPointError(f"{method} has non-finite values on valid cells")
        if np.isinf(physical).any():
            raise FloatingPointError(f"{method} contains infinite values")
        if coarse_mm is not None:
            expected_coarse = (n_time, n_member, *coarse.shape[-2:])
            if np.asarray(coarse_mm[method]).shape != expected_coarse:
                raise ValueError(f"{method} coarse_mm shape must be {expected_coarse}")

    root = _group(temporary, "w")
    root.attrs.update(
        schema="cpc_v3_hierarchical_samples_v3",
        complete=False,
        units="mm/day",
        archive_uses_likelihood_hard_decoder=False,
        method_specs=method_specs or {method: {} for method in methods},
        sampler_diagnostics=diagnostics,
        subgrid_encoding=encoding_metadata(frozen_encoding),
    )
    if isinstance(frozen_encoding, LegacyV2SubgridEncoding):
        root.attrs["legacy_v2_decoder"] = True
    if target_crop is not None:
        if len(target_crop) != 4:
            raise ValueError("target_crop must be (row_start,row_stop,col_start,col_stop)")
        root.attrs["target_crop"] = [int(value) for value in target_crop]
    _array(root, "time", times.astype(np.int64), ("time",))
    _array(root, "member", np.arange(n_member, dtype=np.int16), ("member",))
    _array(root, "lat", lat, ("lat",), attrs={"units": "degrees_north"})
    _array(root, "lon", lon, ("lon",), attrs={"units": "degrees_east"})
    _array(root, "valid", valid, ("lat", "lon"), chunks=(nlat, nlon))
    _array(
        root, "coarse_valid", coarse_valid,
        ("coarse_lat", "coarse_lon"), chunks=coarse_valid.shape,
    )
    _array(root, "cell_area", cell_area, ("lat", "lon"), chunks=(nlat, nlon))
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

    # Audit the exact serialized states and values that the evaluator will
    # reopen. Do this before setting complete=True and before the atomic rename.
    reopened = _group(temporary, "a")
    roundtrip_max = 0.0
    hard_decode_by_method = {}
    source_decode_by_method = {}
    coarse_decode_by_method = {}
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
        source_decode_max = 0.0
        hard_decode_max = 0.0
        source_coarse_max = 0.0
        saved_coarse_max = 0.0
        stored_coarse = reopened[f"{method}_coarse_state"]
        stored_allocation = reopened[f"{method}_allocation_state"]
        stored_physical = reopened[method]
        stored_coarse_mm = (
            reopened[f"{method}_coarse_mm"] if coarse_mm is not None else None
        )
        for index in range(n_time):
            coarse_chunk = torch.from_numpy(np.asarray(stored_coarse[index], np.float32))
            allocation_chunk = torch.from_numpy(
                np.asarray(stored_allocation[index], np.float32)
            )
            decoded_coarse = decode_coarse_amount(
                coarse_chunk,
                torch.from_numpy(coarse_valid),
                frozen_encoding,
                hard=True,
            )[:, 0].numpy()
            decoded = decode_and_reconstruct(
                coarse_chunk,
                allocation_chunk,
                torch.from_numpy(coarse_valid),
                torch.from_numpy(valid),
                torch.from_numpy(cell_area),
                frozen_encoding,
                hard=True,
            )[:, 0].numpy()
            archived = np.asarray(stored_physical[index], np.float32)
            difference = np.abs(decoded[:, valid] - archived[:, valid])
            allowed = serialization_tolerance_mm_day + (
                hard_decode_relative_tolerance
                * np.maximum(np.abs(decoded[:, valid]), np.abs(archived[:, valid]))
            )
            error = float(np.max(difference)) if difference.size else 0.0
            source_decode_max = max(source_decode_max, error)
            if (
                not np.isfinite(error)
                or not np.isfinite(allowed).all()
                or np.any(difference > allowed)
            ):
                maximum_allowed = float(np.max(allowed)) if allowed.size else 0.0
                raise RuntimeError(
                    f"{method} archived field differs from its serialized hard-decoded "
                    f"states by {error:.3g} mm/day (absolute tolerance "
                    f"{serialization_tolerance_mm_day:.3g}, relative tolerance "
                    f"{hard_decode_relative_tolerance:.3g}; maximum applied "
                    f"tolerance {maximum_allowed:.3g})"
                )

            # CUDA and CPU reductions over the same float32 state can differ by
            # a handful of ulps.  Once that bounded source comparison passes,
            # make the serialized states authoritative and store their CPU hard
            # decode.  The evaluator then reads a physical field that is exactly
            # reproducible from the archived latent state, rather than a nearly
            # identical device-specific rendering.
            canonical = archived.copy()
            canonical[:, valid] = decoded[:, valid]
            stored_physical[index] = canonical
            written = np.asarray(stored_physical[index], np.float32)
            saved_error = float(
                np.max(np.abs(decoded[:, valid] - written[:, valid]))
            )
            hard_decode_max = max(hard_decode_max, saved_error)
            if stored_coarse_mm is not None:
                archived_coarse = np.asarray(stored_coarse_mm[index], np.float32)
                coarse_difference = np.abs(decoded_coarse - archived_coarse)
                coarse_allowed = serialization_tolerance_mm_day + (
                    hard_decode_relative_tolerance
                    * np.maximum(np.abs(decoded_coarse), np.abs(archived_coarse))
                )
                coarse_error = float(np.max(coarse_difference))
                source_coarse_max = max(source_coarse_max, coarse_error)
                if (
                    not np.isfinite(coarse_error)
                    or not np.isfinite(coarse_allowed).all()
                    or np.any(coarse_difference > coarse_allowed)
                ):
                    raise RuntimeError(
                        f"{method} archived coarse amount differs from its serialized "
                        f"coarse state by {coarse_error:.3g} mm/day"
                    )
                stored_coarse_mm[index] = decoded_coarse
                written_coarse = np.asarray(stored_coarse_mm[index], np.float32)
                saved_coarse_max = max(
                    saved_coarse_max,
                    float(np.max(np.abs(decoded_coarse - written_coarse))),
                )
        if not np.isfinite(hard_decode_max) or hard_decode_max > serialization_tolerance_mm_day:
            raise RuntimeError(
                f"{method} canonical hard decode changed during serialization by "
                f"{hard_decode_max:.3g} mm/day (tolerance "
                f"{serialization_tolerance_mm_day:.3g})"
            )
        if (
            not np.isfinite(saved_coarse_max)
            or saved_coarse_max > serialization_tolerance_mm_day
        ):
            raise RuntimeError(
                f"{method} canonical coarse amount changed during serialization by "
                f"{saved_coarse_max:.3g} mm/day"
            )
        source_decode_by_method[method] = source_decode_max
        hard_decode_by_method[method] = hard_decode_max
        coarse_decode_by_method[method] = {
            "source_max_abs_mm_day": source_coarse_max,
            "saved_max_abs_mm_day": saved_coarse_max,
        }
    if not np.isfinite(roundtrip_max) or roundtrip_max > serialization_tolerance_mm_day:
        raise RuntimeError(
            f"hierarchical archive round-trip error {roundtrip_max:.3g} mm/day "
            f"exceeds {serialization_tolerance_mm_day:.3g}"
        )
    root.attrs["serialization_max_abs_mm_day"] = roundtrip_max
    root.attrs["hard_decode_absolute_tolerance_mm_day"] = float(
        serialization_tolerance_mm_day
    )
    root.attrs["hard_decode_relative_tolerance"] = float(
        hard_decode_relative_tolerance
    )
    root.attrs[
        "source_to_canonical_hard_decode_max_abs_mm_day"
    ] = source_decode_by_method
    root.attrs["saved_state_hard_decode_max_abs_mm_day"] = hard_decode_by_method
    if coarse_mm is not None:
        root.attrs["saved_coarse_state_decode_max_abs_mm_day"] = coarse_decode_by_method
    root.attrs["archive_uses_likelihood_hard_decoder"] = True
    root.attrs["complete"] = True
    zarr.consolidate_metadata(str(temporary))
    os.replace(temporary, output)


def recover_incomplete_hierarchical_sample_zarr(
    path: str | Path,
    *,
    encoding,
    expected_methods: tuple[str, ...] | list[str] | None = None,
    serialization_tolerance_mm_day: float = 5.0e-4,
    hard_decode_relative_tolerance: float = 5.0e-6,
) -> dict[str, float]:
    """Finish an archive stopped only by the GPU/CPU hard-decode audit.

    The normal writer creates every physical and latent array before reopening
    the temporary store for its independent hard-decode audit.  Consequently a
    device-rounding false positive can leave a complete set of samples under
    ``<output>.incomplete`` even though no final archive was published.  This
    explicit recovery path revalidates that store, rejects material physical
    differences, replaces each device rendering with the canonical CPU decode
    of its serialized states, and then performs the same atomic publication.

    It deliberately does not add experiment-specific context arrays or reports;
    the calling diagnostic must do that after recovery.
    """
    import zarr
    import torch

    from .data.subgrid_dataset import (
        SubgridEncoding,
        decode_and_reconstruct,
        decode_coarse_amount,
        encoding_metadata,
    )

    if serialization_tolerance_mm_day < 0.0:
        raise ValueError("serialization tolerance must be non-negative")
    if hard_decode_relative_tolerance < 0.0:
        raise ValueError("hard-decode relative tolerance must be non-negative")
    frozen_encoding = (
        encoding
        if isinstance(encoding, SubgridEncoding)
        else SubgridEncoding.from_mapping(encoding)
    )
    frozen_encoding.validate()

    output = Path(path)
    temporary = output.with_name(output.name + ".incomplete")
    if output.exists():
        raise FileExistsError(f"completed sample store already exists: {output}")
    if not temporary.is_dir():
        raise FileNotFoundError(f"incomplete sample store is absent: {temporary}")
    root = _group(temporary, "a")
    if root.attrs.get("schema") != "cpc_v3_hierarchical_samples_v3":
        raise ValueError("incomplete store is not a hierarchical sample-v3 archive")
    if root.attrs.get("complete", False):
        raise ValueError("incomplete path is unexpectedly marked complete")
    if root.attrs.get("archive_uses_likelihood_hard_decoder", False):
        raise ValueError("incomplete path is unexpectedly marked as an audited archive")
    if root.attrs.get("subgrid_encoding") != encoding_metadata(frozen_encoding):
        raise ValueError("incomplete store uses a different frozen subgrid encoding")

    specifications = dict(root.attrs.get("method_specs", {}))
    methods = list(specifications)
    if not methods:
        raise ValueError("incomplete store records no methods")
    if expected_methods is not None and set(methods) != set(expected_methods):
        raise ValueError(
            f"incomplete store methods {methods} differ from expected "
            f"{list(expected_methods)}"
        )
    required_shared = {"time", "valid", "coarse_valid", "cell_area"}
    missing_shared = sorted(required_shared - set(root.array_keys()))
    if missing_shared:
        raise ValueError(f"incomplete store lacks shared arrays {missing_shared}")
    valid = np.asarray(root["valid"][:], bool)
    coarse_valid = np.asarray(root["coarse_valid"][:], bool)
    cell_area = np.asarray(root["cell_area"][:], np.float32)
    n_time = int(root["time"].shape[0])
    if valid.shape != cell_area.shape or not np.isfinite(cell_area).all():
        raise ValueError("incomplete store has invalid fine geometry")

    source_decode_by_method: dict[str, float] = {}
    saved_decode_by_method: dict[str, float] = {}
    coarse_decode_by_method: dict[str, dict[str, float]] = {}
    for method in methods:
        required = {
            method,
            f"{method}_coarse_state",
            f"{method}_allocation_state",
        }
        missing = sorted(required - set(root.array_keys()))
        if missing:
            raise ValueError(f"incomplete store lacks {method} arrays {missing}")
        physical = root[method]
        coarse = root[f"{method}_coarse_state"]
        allocation = root[f"{method}_allocation_state"]
        coarse_mm_name = f"{method}_coarse_mm"
        stored_coarse_mm = root[coarse_mm_name] if coarse_mm_name in root else None
        if (
            physical.shape[0] != n_time
            or coarse.shape[0] != n_time
            or allocation.shape[0] != n_time
        ):
            raise ValueError(f"{method} time dimension differs across incomplete arrays")

        source_max = 0.0
        saved_max = 0.0
        source_coarse_max = 0.0
        saved_coarse_max = 0.0
        for index in range(n_time):
            coarse_state = torch.from_numpy(np.asarray(coarse[index], np.float32))
            decoded_coarse = decode_coarse_amount(
                coarse_state,
                torch.from_numpy(coarse_valid),
                frozen_encoding,
                hard=True,
            )[:, 0].numpy()
            decoded = decode_and_reconstruct(
                coarse_state,
                torch.from_numpy(np.asarray(allocation[index], np.float32)),
                torch.from_numpy(coarse_valid),
                torch.from_numpy(valid),
                torch.from_numpy(cell_area),
                frozen_encoding,
                hard=True,
            )[:, 0].numpy()
            archived = np.asarray(physical[index], np.float32)
            if archived.shape != decoded.shape:
                raise ValueError(f"{method} physical/state decode shapes differ")
            difference = np.abs(decoded[:, valid] - archived[:, valid])
            allowed = serialization_tolerance_mm_day + (
                hard_decode_relative_tolerance
                * np.maximum(np.abs(decoded[:, valid]), np.abs(archived[:, valid]))
            )
            error = float(np.max(difference)) if difference.size else 0.0
            source_max = max(source_max, error)
            if (
                not np.isfinite(error)
                or not np.isfinite(allowed).all()
                or np.any(difference > allowed)
            ):
                maximum_allowed = float(np.max(allowed)) if allowed.size else 0.0
                raise RuntimeError(
                    f"cannot recover {method}: stored device field differs from its "
                    f"serialized hard states by {error:.3g} mm/day; maximum applied "
                    f"rounding tolerance is {maximum_allowed:.3g}"
                )
            canonical = archived.copy()
            canonical[:, valid] = decoded[:, valid]
            physical[index] = canonical
            written = np.asarray(physical[index], np.float32)
            saved_error = float(
                np.max(np.abs(decoded[:, valid] - written[:, valid]))
            )
            saved_max = max(saved_max, saved_error)
            if stored_coarse_mm is not None:
                archived_coarse = np.asarray(stored_coarse_mm[index], np.float32)
                difference_coarse = np.abs(decoded_coarse - archived_coarse)
                allowed_coarse = serialization_tolerance_mm_day + (
                    hard_decode_relative_tolerance
                    * np.maximum(np.abs(decoded_coarse), np.abs(archived_coarse))
                )
                error_coarse = float(np.max(difference_coarse))
                source_coarse_max = max(source_coarse_max, error_coarse)
                if (
                    not np.isfinite(error_coarse)
                    or not np.isfinite(allowed_coarse).all()
                    or np.any(difference_coarse > allowed_coarse)
                ):
                    raise RuntimeError(
                        f"cannot recover {method}: stored coarse amount differs "
                        f"from its serialized state by {error_coarse:.3g} mm/day"
                    )
                stored_coarse_mm[index] = decoded_coarse
                written_coarse = np.asarray(stored_coarse_mm[index], np.float32)
                saved_coarse_max = max(
                    saved_coarse_max,
                    float(np.max(np.abs(decoded_coarse - written_coarse))),
                )
        if saved_max > serialization_tolerance_mm_day or not np.isfinite(saved_max):
            raise RuntimeError(
                f"cannot recover {method}: canonical decode changed by "
                f"{saved_max:.3g} mm/day during serialization"
            )
        if (
            saved_coarse_max > serialization_tolerance_mm_day
            or not np.isfinite(saved_coarse_max)
        ):
            raise RuntimeError(
                f"cannot recover {method}: canonical coarse amount changed by "
                f"{saved_coarse_max:.3g} mm/day during serialization"
            )
        source_decode_by_method[method] = source_max
        saved_decode_by_method[method] = saved_max
        coarse_decode_by_method[method] = {
            "source_max_abs_mm_day": source_coarse_max,
            "saved_max_abs_mm_day": saved_coarse_max,
        }

    root.attrs["serialization_max_abs_mm_day"] = 0.0
    root.attrs["hard_decode_absolute_tolerance_mm_day"] = float(
        serialization_tolerance_mm_day
    )
    root.attrs["hard_decode_relative_tolerance"] = float(
        hard_decode_relative_tolerance
    )
    root.attrs[
        "source_to_canonical_hard_decode_max_abs_mm_day"
    ] = source_decode_by_method
    root.attrs["saved_state_hard_decode_max_abs_mm_day"] = saved_decode_by_method
    if coarse_decode_by_method:
        root.attrs["saved_coarse_state_decode_max_abs_mm_day"] = coarse_decode_by_method
    root.attrs["archive_uses_likelihood_hard_decoder"] = True
    root.attrs["recovered_from_device_roundoff_audit"] = True
    root.attrs["complete"] = True
    zarr.consolidate_metadata(str(temporary))
    os.replace(temporary, output)
    return source_decode_by_method
