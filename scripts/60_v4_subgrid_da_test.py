#!/usr/bin/env python3
"""Matched-noise CPC-V3-SG/v4 background and physical-space DA pilot.

This is a bounded post-training diagnostic, not a configuration-selection or
confirmatory experiment.  It samples the corrected schema-v4 joint checkpoint
over a short independent period and writes six audited arms:

``background``
    Unguided joint prior.
``gauges_withheld``
    BMD guidance with a supported, independently withheld station subset.
``imerg_only``
    IMERG S04 0.4-degree area-mean guidance.
``simultaneous_withheld``
    One likelihood over the same assimilated gauges plus the same IMERG values.
``gauges_all`` and ``simultaneous_all``
    All-gauge products used only for spatial-map diagnostics.  They are never
    reported as independent gauge verification.

Every arm starts from identical per-day joint latent noise.  Simultaneous arms
reuse the exact observation perturbations from their corresponding single-
stream arms.  The output is written through the hard-decoder round-trip writer
and can be evaluated by scripts 58 and 61 without rerunning the GPU sampler.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import xarray as xr
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bdhires.bmd import (  # noqa: E402
    max_bearing_gap_deg,
    nearest_neighbour_km,
    neighbored_holdout,
)
from bdhires.da import (  # noqa: E402
    AreaWeightedBlockObsOperator,
    BilinearObsOperator,
    CompositeObsOperator,
    GuidanceConfig,
    HierarchicalObservations,
    HierarchicalSamplerConfig,
    perturb_observations,
    sample_hierarchical,
)
from bdhires.data import (
    resolve_archive_encoding,  # noqa: E402
    SubgridEncoding,
    aligned_production_canvas,
    decode_coarse_amount,
    encoding_metadata,
    load_stations,
)
from bdhires.grids import BD, BD_CPC, WIDE_CPC, Grid, crop_offsets  # noqa: E402
from bdhires.models import (  # noqa: E402
    AllocationFlow,
    CoarseHurdleFlow,
    CoupledSubgridFlow,
    HierarchicalState,
    select_weights,
)
from bdhires.zarr_output import (  # noqa: E402
    recover_incomplete_hierarchical_sample_zarr,
    write_hierarchical_sample_zarr,
)


METHODS = (
    "background",
    "gauges_withheld",
    "imerg_only",
    "simultaneous_withheld",
    "gauges_all",
    "simultaneous_all",
    # Scale-routed arms: IMERG constrains the block amount, gauges constrain only
    # the within-block structure.  The unrouted simultaneous arms above are kept
    # as the declared ablation, because the contrast between them is the evidence
    # that the routing is needed rather than an assertion that it is.
    "routed_withheld",
    "routed_all",
    # Everything routed to the amount.  The gauge fold fit collapses from r=0.94
    # to 0.43 when gauges are barred from the amount, so their information is
    # about how much fell in a block, not about where inside it -- which is what
    # one point in a 55 km box can actually resolve.  These arms test the other
    # side of that: give both streams the amount and let the prior own structure.
    "amount_only_withheld",
    "amount_only_all",
)
MM_PER_DAY = {"mm/day", "mmday-1", "mmd-1", "mmday^-1", "mmd^-1"}


class _BlockSubsetOperator(torch.nn.Module):
    """Area means of the blocks that contain observations, in station order.

    ``AreaWeightedBlockObsOperator`` returns every block on the domain.  A sparse
    network constrains only a few, so the observation vector must select those
    blocks -- and must repeat a block when two stations share it, because that is
    genuinely two measurements of the same quantity.
    """

    def __init__(self, block_operator: torch.nn.Module, index: np.ndarray):
        super().__init__()
        self.block = block_operator
        self.register_buffer("index", torch.as_tensor(index, dtype=torch.long))

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        return self.block(field)[..., self.index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-store",
        default="data/processed/cpc_v3_subgrid/wide_cpc_v4.zarr",
    )
    parser.add_argument(
        "--checkpoint",
        default="runs/prior_h100_cpc_v3_subgrid_v4/joint/best.pt",
    )
    parser.add_argument("--stations", required=True, help="canonical BMD daily CSV")
    parser.add_argument(
        "--imerg",
        default=(
            "data/processed/imerg_prepared_ing2022/"
            "imerg_0p4deg_20220501_20220510.nc"
        ),
    )
    parser.add_argument("--start", default="2022-05-01")
    parser.add_argument("--end", default="2022-05-05")
    parser.add_argument("--background-day-offset", type=int, default=-1)
    parser.add_argument("--members", type=int, default=4)
    parser.add_argument(
        "--occurrence-temperature", type=float, default=None,
        help="override the frozen wetness relaxation temperature; the design "
             "predeclares a sweep chosen on the soft/hard O-A bounds",
    )
    parser.add_argument(
        "--allow-conditioning-lag",
        action="store_true",
        help="permit a conditioning offset that differs from the target store's",
    )
    parser.add_argument("--n-steps", type=int, default=25)
    parser.add_argument("--canvas", type=int, default=160)
    parser.add_argument("--withhold", type=float, default=0.20)
    parser.add_argument("--min-coverage", type=float, default=0.80)
    parser.add_argument("--holdout-neighbor-km", type=float, default=75.0)
    parser.add_argument("--holdout-max-gap-deg", type=float, default=200.0)
    parser.add_argument("--gauge-sigma-mm", type=float, default=3.0)
    parser.add_argument("--imerg-sigma-floor-mm", type=float, default=1.0)
    parser.add_argument("--imerg-representativeness-mm", type=float, default=0.0)
    parser.add_argument("--imerg-factor", type=int, default=8)
    parser.add_argument("--imerg-error-corr-cells", type=float, default=0.75)
    parser.add_argument("--imerg-r-multiplier", type=float, default=1.0)
    parser.add_argument("--guidance-gamma", type=float, default=1.0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--guidance-clip-norm", type=float, default=100.0)
    parser.add_argument("--huber-delta", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=4202205)
    parser.add_argument(
        "--out-store",
        default="data/processed/v4_da_test/may2022_5day/v4_da_samples.zarr",
    )
    parser.add_argument(
        "--report",
        default="data/processed/v4_da_test/may2022_5day/v4_da_sampling.json",
    )
    parser.add_argument(
        "--recover-incomplete",
        action="store_true",
        help=(
            "recover a fully sampled .incomplete archive stopped only by the "
            "GPU/CPU float32 hard-decode audit; never resamples"
        ),
    )
    parser.add_argument(
        "--osse", action="store_true",
        help=(
            "replace gauge observations with the CHIRPS truth sampled at the same "
            "station locations, with near-zero error.  Isolates whether the "
            "machinery can carry information from an assimilated point to a "
            "withheld one, independently of observation error, representativeness, "
            "and of whether the truth is reachable by the prior at all."
        ),
    )
    parser.add_argument(
        "--osse-sigma-mm", type=float, default=0.5,
        help="observation error for OSSE pseudo-gauges; small but not zero",
    )
    parser.add_argument(
        "--osse-gauge-support", choices=("point", "block"), default="point",
        help=(
            "what an OSSE pseudo-gauge measures.  'point' is the truth in the "
            "station's own 0.05-degree cell; 'block' is the area mean of the "
            "truth over the station's whole conservation block.  Verification "
            "stays at points against point truth either way, so the two differ "
            "only in what is assimilated -- which is exactly the point-versus-"
            "block question."
        ),
    )
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def require_v4_contract(root, checkpoint: dict) -> SubgridEncoding:
    if not root.attrs.get("complete", False):
        raise ValueError("v4 target store is not marked complete")
    target_schema = root.attrs.get("schema")
    checkpoint_schema = checkpoint.get("schema")
    if target_schema != checkpoint_schema:
        raise ValueError(
            f"target and checkpoint schemas differ: target={target_schema!r}, "
            f"checkpoint={checkpoint_schema!r}"
        )
    if checkpoint.get("stage") != "joint":
        raise ValueError("v4 test requires a joint-stage best checkpoint")
    if "subgrid_encoding" not in root.attrs or "subgrid_encoding" not in checkpoint:
        raise ValueError("target/checkpoint lacks frozen subgrid encoding metadata")
    target_encoding, _ = resolve_archive_encoding(root.attrs)
    checkpoint_encoding, _ = resolve_archive_encoding(checkpoint)
    if encoding_metadata(target_encoding) != encoding_metadata(checkpoint_encoding):
        raise ValueError("v4 target and checkpoint use different subgrid encodings")
    config = checkpoint.get("config")
    if not isinstance(config, dict) or config.get("stage") != "joint":
        raise ValueError("checkpoint lacks its resolved joint training config")
    return target_encoding


def build_joint_model(checkpoint: dict, root, device: torch.device):
    config = checkpoint["config"]
    model_config = config["model"]
    crop = int(config["data"]["crop"])
    factor = int(config["data"].get("factor", 10))
    coarse = CoarseHurdleFlow(
        int(root["coarse_cond"].shape[1]),
        image_size=crop // factor,
        **model_config["coarse"],
    )
    allocation = AllocationFlow(
        int(root["fine_cond"].shape[1]),
        image_size=crop,
        **model_config["allocation"],
    )
    model = CoupledSubgridFlow(
        coarse,
        allocation,
        clean_context_probability=float(
            config["train"].get("clean_context_probability", 0.0)
        ),
    )
    model.load_state_dict(select_weights(checkpoint), strict=True)
    return model.to(device).eval(), config


def date_indices(root, start: str, end: str, offset: int):
    observation_days = np.arange(
        np.datetime64(start, "D"),
        np.datetime64(end, "D") + np.timedelta64(1, "D"),
    )
    if observation_days.size == 0:
        raise ValueError("requested v4 diagnostic period is empty")
    source_days = np.asarray(root["time"][:], np.int64).astype(
        "datetime64[ns]"
    ).astype("datetime64[D]")
    if len(np.unique(source_days)) != len(source_days):
        raise ValueError("v4 target contains duplicate dates")
    lookup = {day: index for index, day in enumerate(source_days)}
    condition_days = observation_days + np.timedelta64(offset, "D")
    missing = [
        day
        for day in np.concatenate([observation_days, condition_days])
        if day not in lookup
    ]
    if missing:
        raise ValueError(f"v4 target lacks requested observation/condition dates: {missing}")
    observation_index = np.asarray([lookup[day] for day in observation_days], np.int64)
    condition_index = np.asarray([lookup[day] for day in condition_days], np.int64)
    return observation_days, condition_days, observation_index, condition_index


def canvas_grid(canvas_slice: tuple[slice, slice]) -> Grid:
    rows, columns = canvas_slice
    return Grid(
        name="v4_bd_canvas",
        lon_min=WIDE_CPC.lon_min + int(columns.start) * WIDE_CPC.res,
        lat_min=WIDE_CPC.lat_min + int(rows.start) * WIDE_CPC.res,
        nlon=int(columns.stop) - int(columns.start),
        nlat=int(rows.stop) - int(rows.start),
        res=WIDE_CPC.res,
    )


def legacy_bd_crop(canvas_slice: tuple[slice, slice]) -> tuple[int, int, int, int]:
    """Return the exact legacy 128-cell BD/IMERG window inside the v4 canvas."""
    wide_row, wide_column = crop_offsets(WIDE_CPC, BD)
    rows, columns = canvas_slice
    row0 = wide_row - int(rows.start)
    column0 = wide_column - int(columns.start)
    crop = (row0, row0 + BD.nlat, column0, column0 + BD.nlon)
    if (
        row0 < 0
        or column0 < 0
        or crop[1] > rows.stop - rows.start
        or crop[3] > columns.stop - columns.start
    ):
        raise ValueError("the aligned v4 canvas does not contain the legacy BD IMERG grid")
    return crop


def load_imerg_subset(
    path: str,
    days: np.ndarray,
    factor: int,
) -> dict:
    """Load an exact requested subset from a validated BMD-aligned S04 file."""
    with xr.open_dataset(path) as dataset:
        required = {"precipitation", "randomError", "precipitation_cnt"}
        missing_variables = sorted(required - set(dataset.variables))
        if missing_variables:
            raise ValueError(f"{path} lacks IMERG variables {missing_variables}")
        for name in ("precipitation", "randomError"):
            units = str(dataset[name].attrs.get("units", ""))
            if units.lower().replace(" ", "") not in MM_PER_DAY:
                raise ValueError(f"{path} {name} units are {units!r}; expected mm/day")
        end_hour = dataset.attrs.get("bmd_accumulation_end_hour_utc")
        try:
            end_hour = int(end_hour)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{path} does not declare a valid BMD accumulation end hour"
            ) from exc
        if end_hour != 3:
            raise ValueError(f"{path} is not aligned to the BMD 03:00 UTC window")
        source_frequency = str(dataset.attrs.get("source_frequency", ""))
        if source_frequency != "half-hourly":
            raise ValueError(
                f"{path} was not prepared from half-hourly IMERG; the real-data "
                "v4 test rejects daily day-shift approximations"
            )
        stored_factor = dataset.attrs.get("observation_factor")
        if stored_factor is None:
            raise ValueError(f"{path} does not declare its observation_factor")
        if int(stored_factor) != factor:
            raise ValueError(
                f"{path} declares factor {stored_factor}, requested factor {factor}"
            )
        source_days = np.asarray(dataset.time.values).astype("datetime64[D]")
        if len(np.unique(source_days)) != len(source_days):
            raise ValueError("IMERG file contains duplicate dates")
        lookup = {day: index for index, day in enumerate(source_days)}
        missing_days = [day for day in days if day not in lookup]
        if missing_days:
            raise ValueError(f"IMERG file lacks requested days {missing_days}")
        index = np.asarray([lookup[day] for day in days], int)
        precipitation = np.asarray(
            dataset["precipitation"]
            .isel(time=index)
            .transpose("time", "lat", "lon"),
            np.float32,
        )
        random_error = np.asarray(
            dataset["randomError"].isel(time=index).transpose("time", "lat", "lon"),
            np.float32,
        )
        count = np.asarray(
            dataset["precipitation_cnt"]
            .isel(time=index)
            .transpose("time", "lat", "lon")
        )
        lat = np.asarray(dataset.lat.values, np.float64)
        lon = np.asarray(dataset.lon.values, np.float64)
        required_corr = dataset.attrs.get("required_error_corr_cells")
    if required_corr is None:
        raise ValueError(f"{path} does not declare required_error_corr_cells")
    if np.any(np.isfinite(precipitation) & (precipitation < 0.0)):
        raise ValueError("IMERG precipitation contains finite negative values")
    if np.any(np.isfinite(random_error) & (random_error < 0.0)):
        raise ValueError("IMERG randomError contains finite negative values")
    if BD.nlat % factor or BD.nlon % factor:
        raise ValueError(f"IMERG factor {factor} does not tile legacy BD")
    expected_lat = BD.lat.reshape(-1, factor).mean(axis=1)
    expected_lon = BD.lon.reshape(-1, factor).mean(axis=1)
    if precipitation.shape[1:] != (len(expected_lat), len(expected_lon)):
        raise ValueError(
            f"IMERG shape {precipitation.shape[1:]} does not match the exact "
            f"legacy-BD factor-{factor} grid {(len(expected_lat), len(expected_lon))}"
        )
    if not np.allclose(lat, expected_lat, atol=1.0e-5) or not np.allclose(
        lon, expected_lon, atol=1.0e-5
    ):
        raise ValueError("IMERG footprint centres do not nest on the legacy BD grid")
    return {
        "precipitation": precipitation,
        "random_error": random_error,
        "count": count,
        "lat": lat.astype(np.float32),
        "lon": lon.astype(np.float32),
        "required_error_corr_cells": float(required_corr),
    }


def initial_noise(
    members: int,
    fine_shape: tuple[int, int],
    factor: int,
    seed: int,
    device: torch.device,
) -> HierarchicalState:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    height, width = fine_shape
    return HierarchicalState(
        torch.randn(
            members,
            2,
            height // factor,
            width // factor,
            generator=generator,
            device=device,
        ),
        torch.randn(
            members,
            2,
            height,
            width,
            generator=generator,
            device=device,
        ),
    )


def clone_state(state: HierarchicalState) -> HierarchicalState:
    return HierarchicalState(state.coarse.clone(), state.allocation.clone())


def append_array(group, name: str, values, dimensions: tuple[str, ...], chunks=None):
    values = np.asarray(values)
    array = group.create_dataset(
        name,
        data=values,
        shape=values.shape,
        chunks=chunks or values.shape,
        fill_value=None,
        overwrite=False,
    )
    array.attrs["_ARRAY_DIMENSIONS"] = list(dimensions)
    return array


def validate_recoverable_partial(
    output: Path,
    *,
    days: np.ndarray,
    members: int,
    n_steps: int,
    lat: np.ndarray,
    lon: np.ndarray,
    valid: np.ndarray,
    coarse_valid: np.ndarray,
    area: np.ndarray,
    target_crop: tuple[int, int, int, int],
) -> None:
    """Prove that an explicit recovery request matches the stopped pilot."""
    partial = output.with_name(output.name + ".incomplete")
    if output.exists():
        raise FileExistsError(f"completed sample store already exists: {output}")
    root = zarr.open_group(str(partial), mode="r")
    stored_days = np.asarray(root["time"][:], np.int64).astype("datetime64[ns]")
    expected_days = np.asarray(days).astype("datetime64[ns]")
    if not np.array_equal(stored_days, expected_days):
        raise ValueError("incomplete archive dates differ from this recovery request")
    if int(root["member"].shape[0]) != int(members):
        raise ValueError("incomplete archive member count differs from recovery request")
    if set(dict(root.attrs.get("method_specs", {}))) != set(METHODS):
        raise ValueError("incomplete archive does not contain the frozen six pilot arms")
    if list(root.attrs.get("target_crop", [])) != list(target_crop):
        raise ValueError("incomplete archive target crop differs from recovery request")
    checks = (
        ("lat", np.asarray(lat, np.float32)),
        ("lon", np.asarray(lon, np.float32)),
        ("valid", np.asarray(valid, bool)),
        ("coarse_valid", np.asarray(coarse_valid, bool)),
        ("cell_area", np.asarray(area, np.float32)),
    )
    for name, expected in checks:
        if not np.array_equal(np.asarray(root[name][:]), expected):
            raise ValueError(f"incomplete archive {name} differs from recovery request")
    diagnostics = dict(root.attrs.get("sampler_diagnostics", {}))
    for method in METHODS:
        daily = list(diagnostics.get(method, {}).get("daily", []))
        if len(daily) != len(days):
            raise ValueError(f"incomplete archive has the wrong {method} day count")
        if any(int(item.get("n_steps", -1)) != int(n_steps) for item in daily):
            raise ValueError(f"incomplete archive {method} used different ODE steps")
        if any(item.get("heun") is not True for item in daily):
            raise ValueError(f"incomplete archive {method} was not sampled with Heun")


def require_frozen_default_recovery_request(args: argparse.Namespace) -> None:
    """Prevent attaching invented provenance to the pre-fix partial archive.

    The writer version that produced the recoverable store had not yet attached
    the experiment-specific report when its audit stopped. Geometry, dates,
    members and ODE steps are recoverable from the arrays and diagnostics, but
    the checkpoint path and likelihood hyperparameters are not. Recovery is
    therefore restricted to the one frozen default pilot that produced the
    reported failure.
    """
    expected = {
        "target_store": "data/processed/cpc_v3_subgrid/wide_cpc_v4.zarr",
        "checkpoint": "runs/prior_h100_cpc_v3_subgrid_v4/joint/best.pt",
        "imerg": (
            "data/processed/imerg_prepared_ing2022/"
            "imerg_0p4deg_20220501_20220510.nc"
        ),
        "start": "2022-05-01",
        "end": "2022-05-05",
        "background_day_offset": -1,
        "members": 4,
        "n_steps": 25,
        "canvas": 160,
        "withhold": 0.20,
        "min_coverage": 0.80,
        "holdout_neighbor_km": 75.0,
        "holdout_max_gap_deg": 200.0,
        "gauge_sigma_mm": 3.0,
        "imerg_sigma_floor_mm": 1.0,
        "imerg_representativeness_mm": 0.0,
        "imerg_factor": 8,
        "imerg_error_corr_cells": 0.75,
        "imerg_r_multiplier": 1.0,
        "guidance_gamma": 1.0,
        "guidance_scale": 1.0,
        "guidance_clip_norm": 100.0,
        "huber_delta": 3.0,
        "seed": 4202205,
    }
    mismatched = {
        name: (getattr(args, name), value)
        for name, value in expected.items()
        if getattr(args, name) != value
    }
    if mismatched:
        raise ValueError(
            "the stopped archive predates embedded sampling provenance, so recovery "
            f"is allowed only for the frozen default pilot; mismatches={mismatched}"
        )


def complete_diagnostic_archive(
    output: Path,
    *,
    args: argparse.Namespace,
    days: np.ndarray,
    condition_days: np.ndarray,
    grid: Grid,
    canvas_slice: tuple[slice, slice],
    core_slice: tuple[slice, slice],
    imerg_crop: tuple[int, int, int, int],
    imerg: dict,
    chirps: np.ndarray,
    cpc_mm: np.ndarray,
    stations,
    gauge_mm: np.ndarray,
    withheld: np.ndarray,
    assimilated: np.ndarray,
    nearest: np.ndarray,
    bearing_gap: np.ndarray,
    correlation_inflation: float,
    guidance: GuidanceConfig,
) -> None:
    """Attach deterministic context and publish the pilot provenance report."""
    rows, columns = canvas_slice
    fine_shape = tuple(np.asarray(chirps).shape[-2:])
    coarse_shape = tuple(np.asarray(cpc_mm).shape[-2:])
    archive = zarr.open_group(str(output), mode="a")
    if not archive.attrs.get("complete", False) or not archive.attrs.get(
        "archive_uses_likelihood_hard_decoder", False
    ):
        raise ValueError("sample archive was not completed by the hard-decoder audit")
    if archive.attrs.get("diagnostic_complete", False):
        raise FileExistsError(f"diagnostic archive is already complete: {output}")
    expected_time = np.asarray(days).astype("datetime64[ns]").astype(np.int64)
    if not np.array_equal(np.asarray(archive["time"][:], np.int64), expected_time):
        raise ValueError("sample archive dates differ from diagnostic context")
    if int(archive["member"].shape[0]) != int(args.members):
        raise ValueError("sample archive member count differs from diagnostic request")
    for name in (
        "context_chirps_mm",
        "context_cpc_mm",
        "context_imerg_mm",
        "context_imerg_random_error_mm",
        "station_value_mm",
    ):
        if name in archive:
            raise FileExistsError(f"partial diagnostic metadata already contains {name}")

    archive.attrs.update(
        diagnostic_complete=False,
        diagnostic_kind="v4_short_da_pilot",
        source_checkpoint=str(args.checkpoint),
        source_target_store=str(args.target_store),
        condition_day_offset=int(args.background_day_offset),
        context_date_convention=(
            "time = observation label date; context_chirps_mm, context_cpc_mm, "
            "state_date and every model field are on the STATE date, which is "
            "time + condition_day_offset"
        ),
        imerg_source=str(args.imerg),
        imerg_canvas_crop=list(imerg_crop),
        imerg_factor=int(args.imerg_factor),
        imerg_error_corr_cells=float(args.imerg_error_corr_cells),
        imerg_r_inflation=float(correlation_inflation),
        holdout_design="neighbored_holdout; supported interpolation diagnostic",
    )
    append_array(
        archive, "state_date",
        np.asarray(condition_days).astype("datetime64[ns]").astype(np.int64),
        ("time",),
    )
    append_array(
        archive, "context_chirps_mm", chirps, ("time", "lat", "lon"),
        chunks=(1, *fine_shape),
    )
    append_array(
        archive, "context_cpc_mm", cpc_mm.astype(np.float32),
        ("time", "coarse_lat", "coarse_lon"), chunks=(1, *coarse_shape),
    )
    append_array(
        archive, "context_imerg_mm", imerg["precipitation"],
        ("time", "imerg_lat", "imerg_lon"),
        chunks=(1, *imerg["precipitation"].shape[1:]),
    )
    append_array(
        archive, "context_imerg_random_error_mm", imerg["random_error"],
        ("time", "imerg_lat", "imerg_lon"),
        chunks=(1, *imerg["random_error"].shape[1:]),
    )
    append_array(archive, "imerg_lat", imerg["lat"], ("imerg_lat",))
    append_array(archive, "imerg_lon", imerg["lon"], ("imerg_lon",))
    append_array(archive, "station_lat", stations.lat.astype(np.float32), ("station",))
    append_array(archive, "station_lon", stations.lon.astype(np.float32), ("station",))
    append_array(
        archive, "station_value_mm", gauge_mm.astype(np.float32),
        ("time", "station"),
    )
    append_array(
        archive, "station_withheld", np.isin(np.arange(len(stations)), withheld),
        ("station",),
    )
    archive.attrs["station_ids"] = [str(value) for value in stations.ids]
    archive.attrs["diagnostic_complete"] = True
    zarr.consolidate_metadata(str(output))

    report = {
        "schema": "cpc_v3_subgrid_v4_da_pilot_v1",
        "status": "diagnostic_only_not_configuration_selection",
        "target_store": args.target_store,
        "checkpoint": args.checkpoint,
        "sample_store": args.out_store,
        "dates": [str(days[0]), str(days[-1])],
        "condition_dates": [str(condition_days[0]), str(condition_days[-1])],
        "members": args.members,
        "n_steps": args.n_steps,
        "seed": args.seed,
        # Recorded so an OSSE store can never be silently reused for a real-data
        # run: the reuse check compares this report against the request, and an
        # identity-swap of the observations must invalidate it.
        "osse": bool(args.osse),
        "osse_sigma_mm": float(args.osse_sigma_mm) if args.osse else None,
        "osse_gauge_support": args.osse_gauge_support if args.osse else None,
        "methods": list(METHODS),
        "canvas": {
            "grid": grid.name,
            "shape": list(grid.shape),
            "target_crop": [rows.start, rows.stop, columns.start, columns.stop],
            "bd_cpc_core": [
                core_slice[0].start,
                core_slice[0].stop,
                core_slice[1].start,
                core_slice[1].stop,
            ],
            "imerg_crop": list(imerg_crop),
        },
        "stations": {
            "total": len(stations),
            "assimilated_fold": int(len(assimilated)),
            "withheld": int(len(withheld)),
            "withheld_ids": [str(stations.ids[index]) for index in withheld],
            "withheld_nearest_neighbor_km": nearest[withheld].tolist(),
            "withheld_max_bearing_gap_deg": bearing_gap[withheld].tolist(),
            "withhold_fraction": args.withhold,
            "minimum_coverage": args.min_coverage,
            "holdout_neighbor_km": args.holdout_neighbor_km,
            "holdout_max_gap_deg": args.holdout_max_gap_deg,
        },
        "observation_model": {
            "gauge_sigma_mm": args.gauge_sigma_mm,
            "imerg_factor": args.imerg_factor,
            "imerg_sigma_floor_mm": args.imerg_sigma_floor_mm,
            "imerg_representativeness_mm": args.imerg_representativeness_mm,
            "imerg_error_corr_cells": args.imerg_error_corr_cells,
            "imerg_r_multiplier": args.imerg_r_multiplier,
            "imerg_r_inflation": correlation_inflation,
            "guidance": asdict(guidance),
            "warning": "raw IMERG V07B is not bias corrected in this pilot",
        },
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"[done] wrote {output}")
    print(f"[done] wrote {report_path}")


def main() -> None:
    args = parse_args()
    if args.preflight_only and args.recover_incomplete:
        raise ValueError("--preflight-only and --recover-incomplete are mutually exclusive")
    if args.members < 2:
        raise ValueError("--members must be at least 2")
    if args.n_steps < 2:
        raise ValueError("--n-steps must be at least 2")
    if not 0.0 < args.withhold < 1.0:
        raise ValueError("--withhold must lie in (0,1)")
    if not 0.0 < args.min_coverage <= 1.0:
        raise ValueError("--min-coverage must lie in (0,1]")
    if args.holdout_neighbor_km <= 0.0:
        raise ValueError("--holdout-neighbor-km must be positive")
    if not 0.0 < args.holdout_max_gap_deg <= 360.0:
        raise ValueError("--holdout-max-gap-deg must lie in (0,360]")
    if args.gauge_sigma_mm <= 0.0 or args.imerg_sigma_floor_mm <= 0.0:
        raise ValueError("observation-error scales must be positive")
    if args.imerg_representativeness_mm < 0.0:
        raise ValueError("--imerg-representativeness-mm cannot be negative")
    if args.imerg_error_corr_cells < 0.0 or args.imerg_r_multiplier <= 0.0:
        raise ValueError("IMERG correlation length must be non-negative and R multiplier positive")
    if args.guidance_gamma < 0.0 or args.guidance_scale < 0.0:
        raise ValueError("guidance gamma/scale cannot be negative")
    if args.guidance_clip_norm <= 0.0 or args.huber_delta <= 0.0:
        raise ValueError("guidance clip norm and Huber delta must be positive")
    if args.imerg_factor != 8:
        raise ValueError("the frozen pilot is IMERG S04 and requires --imerg-factor 8")

    root = zarr.open_group(args.target_store, mode="r")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    encoding = require_v4_contract(root, checkpoint)
    if encoding.factor != 10:
        raise ValueError(f"the frozen CPC-v4 test requires factor 10, got {encoding.factor}")
    checkpoint_factor = int(checkpoint["config"]["data"].get("factor", 10))
    if checkpoint_factor != encoding.factor:
        raise ValueError("joint checkpoint model factor differs from its frozen encoding")
    # Two different offsets live in this pipeline and must not be conflated.
    #
    #   target build   : CPC, ERA5 and CHIRPS share one date.  The store records
    #                    condition_day_offset=0 and the model is trained on it.
    #   observation set: BMD accumulates to the following morning and the IMERG
    #                    product is aligned to that window, so a file labelled
    #                    D+1 measures the rain of state date D.
    #
    # ``--background-day-offset -1`` therefore means "observations carry the
    # label one day after the state they constrain", which is the correct
    # physical alignment, not a lag to be warned about.
    training_offset = int(root.attrs.get("condition_day_offset", 0))
    if training_offset != 0:
        raise ValueError(
            f"target store was built with condition_day_offset={training_offset}; "
            "V3-SG trains on same-day CPC/ERA5/CHIRPS and the pilot assumes it"
        )
    observation_offset = int(args.background_day_offset)
    if observation_offset > 0:
        raise ValueError(
            "observation labels cannot precede the state date they constrain; "
            f"got --background-day-offset {observation_offset}"
        )
    print(
        f"[dates] state (CPC/ERA5/CHIRPS/background/analysis) = observation "
        f"label {observation_offset:+d} day(s); observations labelled D "
        f"constrain state D{observation_offset:+d}",
        flush=True,
    )
    days, condition_days, target_index, condition_index = date_indices(
        root, args.start, args.end, args.background_day_offset
    )
    canvas_slice, core_slice = aligned_production_canvas(
        WIDE_CPC, BD_CPC, canvas=args.canvas, factor=encoding.factor
    )
    grid = canvas_grid(canvas_slice)
    imerg_crop = legacy_bd_crop(canvas_slice)
    imerg = load_imerg_subset(args.imerg, days, args.imerg_factor)
    stored_corr = imerg["required_error_corr_cells"]
    if stored_corr is not None and not np.isclose(
        stored_corr, args.imerg_error_corr_cells, atol=1.0e-6
    ):
        raise ValueError(
            f"IMERG file requires error_corr_cells={stored_corr}, but the pilot "
            f"was configured with {args.imerg_error_corr_cells}"
        )
    if args.preflight_only:
        print(
            f"[preflight] matched {root.attrs.get('schema')!r} joint "
            f"checkpoint (decoder smooth_base_iterations="
            f"{encoding.smooth_base_iterations}); "
            f"{days[0]}..{days[-1]}; IMERG crop={imerg_crop}",
            flush=True,
        )
        return

    rows, columns = canvas_slice
    coarse_slice = (
        slice(rows.start // encoding.factor, rows.stop // encoding.factor),
        slice(columns.start // encoding.factor, columns.stop // encoding.factor),
    )
    valid = np.asarray(root["fine_valid"][rows, columns], bool)
    coarse_valid = np.asarray(root["coarse_valid"][coarse_slice], bool)
    area = np.asarray(root["cell_area"][rows, columns], np.float32)
    lat = np.asarray(root["lat"][rows], np.float32)
    lon = np.asarray(root["lon"][columns], np.float32)
    # This bounded pilot is small. Explicit per-day reads avoid relying on
    # Zarr-version-specific mixed fancy/slice indexing semantics.
    # CHIRPS is the target the background predicts, so it is read on the STATE
    # date -- the same index as the CPC/ERA5 conditioning, because training is
    # same-day.  Reading it at the observation index instead compared the
    # analysis against the wrong day's rainfall and drove every CHIRPS pattern
    # correlation to zero.
    chirps = np.stack(
        [
            np.asarray(root["fine_mm"][int(index), rows, columns], np.float32)
            for index in condition_index
        ]
    )
    coarse_condition = np.stack(
        [
            np.asarray(
                root["coarse_cond"][
                    int(index), :, coarse_slice[0], coarse_slice[1]
                ],
                np.float32,
            )
            for index in condition_index
        ]
    )
    fine_condition = np.stack(
        [
            np.asarray(
                root["fine_cond"][int(index), :, rows, columns], np.float32
            )
            for index in condition_index
        ]
    )
    channels = list(root.attrs["coarse_cond_channels"])
    if "sqrt_cpc_precip" not in channels:
        raise ValueError("v4 target lacks the frozen sqrt_cpc_precip condition channel")
    cpc_channel = channels.index("sqrt_cpc_precip")
    cpc_mean = float(root.attrs["coarse_cond_mean"][cpc_channel])
    cpc_std = float(root.attrs["coarse_cond_std"][cpc_channel])
    cpc_root = coarse_condition[:, cpc_channel] * cpc_std + cpc_mean
    cpc_mm = np.clip(cpc_root, 0.0, None) ** 2

    stations, gauge_mm = load_stations(
        args.stations, days, grid=grid, min_coverage=args.min_coverage
    )
    if len(stations) < 5:
        raise ValueError(f"only {len(stations)} stations survive coverage filtering")
    n_withheld = max(1, min(len(stations) - 1, int(round(args.withhold * len(stations)))))
    withheld = neighbored_holdout(
        stations.lat,
        stations.lon,
        n_withheld,
        radius_km=args.holdout_neighbor_km,
        max_gap_deg=args.holdout_max_gap_deg,
    )
    assimilated = np.setdiff1d(np.arange(len(stations)), withheld)
    all_stations = np.arange(len(stations))

    if args.osse:
        # Sample the truth with the SAME operator the analysis uses, so a
        # pseudo-gauge reads exactly what the analysis reads at that point and
        # the experiment cannot be confounded by interpolation mismatch.  Both
        # assimilation and verification then use these values, which is what
        # makes the withheld score a clean test of propagation.
        with torch.no_grad():
            probe = BilinearObsOperator(grid, stations.lat, stations.lon)
            gauge_mm = probe(
                torch.from_numpy(chirps)[:, None]
            )[:, 0].numpy().astype(np.float32)
        print(
            f"[osse] gauge values replaced by CHIRPS truth at {len(stations)} "
            f"stations, sigma {args.osse_sigma_mm} mm/day",
            flush=True,
        )
        print(
            "[osse] withheld improvement here means the machinery propagates; "
            "none means it cannot, whatever the observations are",
            flush=True,
        )

    # What is assimilated may differ from what is verified.  Verification is
    # always the point truth at the station, because that is the quantity the
    # product claims to predict; only the observation changes support.
    assimilation_mm = gauge_mm
    osse_block_index = None
    if args.osse and args.osse_gauge_support == "block":
        factor = int(encoding.factor)
        station_row = np.abs(
            lat[:, None] - np.asarray(stations.lat)[None, :]
        ).argmin(axis=0)
        station_column = np.abs(
            lon[:, None] - np.asarray(stations.lon)[None, :]
        ).argmin(axis=0)
        blocks_per_row = chirps.shape[-1] // factor
        osse_block_index = (
            (station_row // factor) * blocks_per_row + (station_column // factor)
        ).astype(np.int64)
        # Reduce the truth with the SAME operator the analysis will be scored
        # through, so the observation is exactly the functional being constrained.
        block_probe = AreaWeightedBlockObsOperator(
            factor, area, valid=valid, min_valid_frac=encoding.valid_area_threshold
        )
        with torch.no_grad():
            truth_blocks = block_probe(
                torch.from_numpy(chirps)[:, None]
            )[:, 0].numpy().astype(np.float32)
        assimilation_mm = truth_blocks[:, osse_block_index]
        distinct = len(np.unique(osse_block_index))
        print(
            f"[osse] gauges assimilated as 0.5-degree block means: "
            f"{len(stations)} stations fall in {distinct} distinct blocks",
            flush=True,
        )
        print(
            "[osse] verification is unchanged -- point truth at the station -- so "
            "any change against the point-support run is the support and nothing else",
            flush=True,
        )
    nearest = nearest_neighbour_km(stations.lat, stations.lon)
    bearing_gap = max_bearing_gap_deg(stations.lat, stations.lon)

    guidance = GuidanceConfig(
        gamma=args.guidance_gamma,
        scale=args.guidance_scale,
        t_start=0.10,
        t_end=0.999,
        clip_norm=args.guidance_clip_norm,
        huber_delta=args.huber_delta,
    )
    correlation_inflation = max(
        1.0, 2.0 * np.pi * args.imerg_error_corr_cells**2
    ) * args.imerg_r_multiplier

    if args.recover_incomplete:
        require_frozen_default_recovery_request(args)
        output = Path(args.out_store)
        target_crop = (rows.start, rows.stop, columns.start, columns.stop)
        validate_recoverable_partial(
            output,
            days=days,
            members=args.members,
            n_steps=args.n_steps,
            lat=lat,
            lon=lon,
            valid=valid,
            coarse_valid=coarse_valid,
            area=area,
            target_crop=target_crop,
        )
        recovered = recover_incomplete_hierarchical_sample_zarr(
            output,
            encoding=encoding,
            expected_methods=METHODS,
        )
        print(
            "[recover] canonicalized completed device samples without resampling; "
            f"maximum source/CPU difference={max(recovered.values()):.6g} mm/day",
            flush=True,
        )
        complete_diagnostic_archive(
            output,
            args=args,
            days=days,
            condition_days=condition_days,
            grid=grid,
            canvas_slice=canvas_slice,
            core_slice=core_slice,
            imerg_crop=imerg_crop,
            imerg=imerg,
            chirps=chirps,
            cpc_mm=cpc_mm,
            stations=stations,
            gauge_mm=gauge_mm,
            withheld=withheld,
            assimilated=assimilated,
            nearest=nearest,
            bearing_gap=bearing_gap,
            correlation_inflation=correlation_inflation,
            guidance=guidance,
        )
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("the v4 joint diagnostic requires an allocated GPU")
    model, training_config = build_joint_model(checkpoint, root, device)
    sampling = training_config.get("sampling", {})
    sampler_config = HierarchicalSamplerConfig(
        n_steps=args.n_steps,
        heun=bool(sampling.get("heun", True)),
        n_corrections=0,
        occurrence_temperature=float(
            args.occurrence_temperature
            if args.occurrence_temperature is not None
            else sampling.get("occurrence_temperature", 1.0)
        ),
        soft_hard_median_sigma=float(sampling.get("soft_hard_median_sigma", 0.10)),
        soft_hard_p95_sigma=float(sampling.get("soft_hard_p95_sigma", 0.50)),
    )
    fine_valid_t = torch.from_numpy(valid).to(device)
    coarse_valid_t = torch.from_numpy(coarse_valid).to(device)
    area_t = torch.from_numpy(area).to(device)

    if osse_block_index is None:
        gauge_operators = {
            "withheld": BilinearObsOperator(
                grid, stations.lat[assimilated], stations.lon[assimilated]
            ).to(device),
            "all": BilinearObsOperator(grid, stations.lat, stations.lon).to(device),
        }
    else:
        # Block-support pseudo-gauges: the operator must reduce the field the
        # same way the observation was formed, or the analysis would be asked to
        # make a point equal an area mean -- a different, and wrong, constraint.
        block_operator = AreaWeightedBlockObsOperator(
            int(encoding.factor), area, valid=valid,
            min_valid_frac=encoding.valid_area_threshold,
        ).to(device)
        gauge_operators = {
            key: _BlockSubsetOperator(block_operator, osse_block_index[index]).to(device)
            for key, index in (("withheld", assimilated), ("all", all_stations))
        }
    satellite_operator = AreaWeightedBlockObsOperator(
        args.imerg_factor,
        area,
        valid=valid,
        min_valid_frac=encoding.valid_area_threshold,
        crop=imerg_crop,
    ).to(device)
    footprint_keep = satellite_operator.valid_mask().detach().cpu().numpy().astype(bool)
    expected_footprints = int(np.prod(imerg["precipitation"].shape[1:]))
    if footprint_keep.shape != (expected_footprints,):
        raise ValueError("IMERG operator and observation file have different footprints")
    combined_operators = {
        key: CompositeObsOperator([operator, satellite_operator]).to(device)
        for key, operator in gauge_operators.items()
    }
    n_days = len(days)
    fine_shape = tuple(valid.shape)
    coarse_shape = tuple(coarse_valid.shape)
    fields = {
        name: np.full((n_days, args.members, *fine_shape), np.nan, np.float32)
        for name in METHODS
    }
    coarse_states = {
        name: np.empty((n_days, args.members, 2, *coarse_shape), np.float32)
        for name in METHODS
    }
    allocation_states = {
        name: np.empty((n_days, args.members, 2, *fine_shape), np.float32)
        for name in METHODS
    }
    coarse_amounts = {
        name: np.empty((n_days, args.members, *coarse_shape), np.float32)
        for name in METHODS
    }
    diagnostics = {name: {"daily": []} for name in METHODS}

    # In OSSE mode the pseudo-gauge is exact, so its error is the small number
    # needed to keep the likelihood well conditioned -- not the real instrument
    # and representativeness budget, which is the thing being held aside.
    gauge_sigma = args.osse_sigma_mm if args.osse else args.gauge_sigma_mm
    gauge_variance = {
        "withheld": np.full(len(assimilated), gauge_sigma**2, np.float32),
        "all": np.full(len(all_stations), gauge_sigma**2, np.float32),
    }
    gauge_indices = {"withheld": assimilated, "all": all_stations}

    for day_position, day in enumerate(days):
        coarse_cond = torch.from_numpy(coarse_condition[day_position : day_position + 1]).to(device)
        fine_cond = torch.from_numpy(fine_condition[day_position : day_position + 1]).to(device)
        day_seed = args.seed + int(target_index[day_position])
        noise = initial_noise(
            args.members, fine_shape, encoding.factor, day_seed, device
        )

        satellite_mm = imerg["precipitation"][day_position].reshape(-1).copy()
        satellite_error = imerg["random_error"][day_position].reshape(-1)
        satellite_variance = (
            np.maximum(satellite_error, args.imerg_sigma_floor_mm) ** 2
            + args.imerg_representativeness_mm**2
        ).astype(np.float32)
        satellite_variance *= np.float32(correlation_inflation)
        satellite_valid = (
            footprint_keep
            & np.isfinite(satellite_mm)
            & np.isfinite(satellite_variance)
            & (imerg["count"][day_position].reshape(-1) > 0)
        )
        satellite_mm[~satellite_valid] = np.nan
        satellite_variance[~satellite_valid] = 1.0
        satellite_perturbed = perturb_observations(
            satellite_mm,
            satellite_variance,
            args.members,
            seed=day_seed + 2_000_000,
            corr_blocks=[
                (
                    0,
                    imerg["precipitation"].shape[1],
                    imerg["precipitation"].shape[2],
                    args.imerg_error_corr_cells,
                )
            ],
        ).astype(np.float32)
        satellite_perturbed[:, ~satellite_valid] = np.nan
        satellite_y = torch.from_numpy(satellite_perturbed[:, None]).to(device)
        satellite_r = torch.from_numpy(satellite_variance).to(device)

        gauge_perturbed = {}
        for key, index in gauge_indices.items():
            observation = assimilation_mm[day_position, index].astype(np.float32)
            perturbed = perturb_observations(
                observation,
                gauge_variance[key],
                args.members,
                seed=day_seed + (1_000_000 if key == "withheld" else 3_000_000),
            ).astype(np.float32)
            perturbed[:, ~np.isfinite(observation)] = np.nan
            gauge_perturbed[key] = perturbed

        observations = {
            "gauges_withheld": HierarchicalObservations(
                gauge_operators["withheld"],
                torch.from_numpy(gauge_perturbed["withheld"][:, None]).to(device),
                torch.from_numpy(gauge_variance["withheld"]).to(device),
                guidance,
            ),
            "imerg_only": HierarchicalObservations(
                satellite_operator, satellite_y, satellite_r, guidance
            ),
            "simultaneous_withheld": HierarchicalObservations(
                combined_operators["withheld"],
                torch.from_numpy(
                    np.concatenate(
                        [gauge_perturbed["withheld"], satellite_perturbed], axis=1
                    )[:, None]
                ).to(device),
                torch.cat(
                    [torch.from_numpy(gauge_variance["withheld"]).to(device), satellite_r]
                ),
                guidance,
            ),
            "gauges_all": HierarchicalObservations(
                gauge_operators["all"],
                torch.from_numpy(gauge_perturbed["all"][:, None]).to(device),
                torch.from_numpy(gauge_variance["all"]).to(device),
                guidance,
            ),
            # Same observations as the simultaneous arms, same concatenation
            # order, differing only in which part of the state each stream is
            # allowed to move.  Anything else held fixed keeps the comparison
            # clean: a difference in scores is the routing and nothing else.
            "routed_withheld": HierarchicalObservations(
                combined_operators["withheld"],
                torch.from_numpy(
                    np.concatenate(
                        [gauge_perturbed["withheld"], satellite_perturbed], axis=1
                    )[:, None]
                ).to(device),
                torch.cat(
                    [torch.from_numpy(gauge_variance["withheld"]).to(device), satellite_r]
                ),
                guidance,
                routing="split",
                amount_mask=torch.cat([
                    torch.zeros(len(gauge_variance["withheld"]), dtype=torch.bool),
                    torch.ones(satellite_r.shape[0], dtype=torch.bool),
                ]).to(device),
            ),
            "amount_only_withheld": HierarchicalObservations(
                combined_operators["withheld"],
                torch.from_numpy(
                    np.concatenate(
                        [gauge_perturbed["withheld"], satellite_perturbed], axis=1
                    )[:, None]
                ).to(device),
                torch.cat(
                    [torch.from_numpy(gauge_variance["withheld"]).to(device), satellite_r]
                ),
                guidance,
                routing="amount",
            ),
            "amount_only_all": HierarchicalObservations(
                combined_operators["all"],
                torch.from_numpy(
                    np.concatenate(
                        [gauge_perturbed["all"], satellite_perturbed], axis=1
                    )[:, None]
                ).to(device),
                torch.cat(
                    [torch.from_numpy(gauge_variance["all"]).to(device), satellite_r]
                ),
                guidance,
                routing="amount",
            ),
            "routed_all": HierarchicalObservations(
                combined_operators["all"],
                torch.from_numpy(
                    np.concatenate(
                        [gauge_perturbed["all"], satellite_perturbed], axis=1
                    )[:, None]
                ).to(device),
                torch.cat(
                    [torch.from_numpy(gauge_variance["all"]).to(device), satellite_r]
                ),
                guidance,
                routing="split",
                amount_mask=torch.cat([
                    torch.zeros(len(gauge_variance["all"]), dtype=torch.bool),
                    torch.ones(satellite_r.shape[0], dtype=torch.bool),
                ]).to(device),
            ),
            "simultaneous_all": HierarchicalObservations(
                combined_operators["all"],
                torch.from_numpy(
                    np.concatenate(
                        [gauge_perturbed["all"], satellite_perturbed], axis=1
                    )[:, None]
                ).to(device),
                torch.cat(
                    [torch.from_numpy(gauge_variance["all"]).to(device), satellite_r]
                ),
                guidance,
            ),
        }

        for name in METHODS:
            sample = sample_hierarchical(
                model,
                coarse_cond,
                fine_cond,
                (args.members, 2, *coarse_shape),
                (args.members, 2, *fine_shape),
                coarse_valid_t,
                fine_valid_t,
                area_t,
                encoding,
                observations=observations.get(name),
                config=sampler_config,
                initial_noise=clone_state(noise),
            )
            fields[name][day_position] = sample.precipitation[:, 0].cpu().numpy()
            coarse_states[name][day_position] = sample.state.coarse.cpu().numpy()
            allocation_states[name][day_position] = sample.state.allocation.cpu().numpy()
            coarse_amounts[name][day_position] = decode_coarse_amount(
                sample.state.coarse,
                coarse_valid_t,
                encoding,
                temperature=sampler_config.occurrence_temperature,
                hard=True,
            )[:, 0].cpu().numpy()
            diagnostics[name]["daily"].append(sample.diagnostics)
        print(
            f"[v4-test] {day}: {len(METHODS)} matched arms complete "
            f"({args.members} members, {args.n_steps} Heun steps)",
            flush=True,
        )

    method_specs = {
        "background": {"observations": "none", "verification_role": "independent"},
        "gauges_withheld": {
            "observations": "BMD except supported holdout",
            "verification_role": "independent withheld gauges",
        },
        "imerg_only": {
            "observations": "IMERG S04",
            "verification_role": "independent BMD gauges",
        },
        "simultaneous_withheld": {
            "observations": "BMD except supported holdout + IMERG S04",
            "verification_role": "independent withheld gauges",
        },
        "gauges_all": {
            "observations": "all BMD",
            "verification_role": "spatial maps only; gauge fit is in-sample",
        },
        "simultaneous_all": {
            "observations": "all BMD + IMERG S04",
            "verification_role": "spatial maps only; gauge fit is in-sample",
        },
        "routed_withheld": {
            "observations": "BMD except supported holdout + IMERG S04, scale-routed",
            "verification_role": "independent withheld gauges",
        },
        "routed_all": {
            "observations": "all BMD + IMERG S04, scale-routed",
            "verification_role": "spatial maps only; gauge fit is in-sample",
        },
        "amount_only_withheld": {
            "observations": "BMD except supported holdout + IMERG S04, amount only",
            "verification_role": "independent withheld gauges",
        },
        "amount_only_all": {
            "observations": "all BMD + IMERG S04, amount only",
            "verification_role": "spatial maps only; gauge fit is in-sample",
        },
    }
    # The archive writer compares these keys against the sampled fields and
    # refuses a mismatch.  That check fires only after every arm has been
    # sampled, so a name added to METHODS but forgotten here costs a whole run.
    if set(method_specs) != set(METHODS):
        missing = set(METHODS) - set(method_specs)
        extra = set(method_specs) - set(METHODS)
        raise ValueError(
            "method_specs does not describe exactly the sampled arms; "
            f"missing {sorted(missing)}, unexpected {sorted(extra)}"
        )
    output = Path(args.out_store)
    write_hierarchical_sample_zarr(
        output,
        fields=fields,
        coarse_states=coarse_states,
        allocation_states=allocation_states,
        selected_times=days,
        lat=lat,
        lon=lon,
        valid=valid,
        coarse_valid=coarse_valid,
        cell_area=area,
        encoding=encoding,
        diagnostics=diagnostics,
        coarse_mm=coarse_amounts,
        method_specs=method_specs,
        target_crop=(rows.start, rows.stop, columns.start, columns.stop),
    )
    complete_diagnostic_archive(
        output,
        args=args,
        days=days,
        condition_days=condition_days,
        grid=grid,
        canvas_slice=canvas_slice,
        core_slice=core_slice,
        imerg_crop=imerg_crop,
        imerg=imerg,
        chirps=chirps,
        cpc_mm=cpc_mm,
        stations=stations,
        gauge_mm=gauge_mm,
        withheld=withheld,
        assimilated=assimilated,
        nearest=nearest,
        bearing_gap=bearing_gap,
        correlation_inflation=correlation_inflation,
        guidance=guidance,
    )


if __name__ == "__main__":
    main()
