"""CPC-aligned targets and datasets for the V3-SG hierarchical model.

The central invariant is physical, not architectural: a decoded 0.05-degree
field must area-average to its decoded 0.5-degree amount in every retained CPC
cell.  This module owns the geometry and target encoding so training, sampling,
DA and evaluation cannot implement slightly different versions of that rule.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, get_worker_info

from ..grids import Grid, crop_offsets


SUBGRID_SCHEMA = "cpc_v3_subgrid_v5"
LEGACY_V4_SUBGRID_SCHEMA = "cpc_v3_subgrid_v4"
LEGACY_V2_SUBGRID_SCHEMA = "cpc_v3_subgrid_v2"


@dataclass(frozen=True)
class SubgridEncoding:
    """Frozen target/decoder choices that define one V3-SG experiment."""

    factor: int = 10
    wet_threshold_mm: float = 0.1
    dequant_epsilon: float = 0.02
    dequant_noise: float = 0.05
    dequant_seed: int = 314159
    intensity_floor: float = 1.0e-5
    denominator_floor: float = 1.0e-8
    valid_area_threshold: float = 0.50
    amount_sqrt_mean: float = 0.0
    amount_sqrt_std: float = 1.0
    intensity_log_mean: float = 0.0
    intensity_log_std: float = 1.0
    # Clip the standardized allocation latent before undoing its training
    # normalization.  A clip in raw log-weight units changes meaning whenever
    # the fitted mean/std change and can still permit pathological exponentials.
    intensity_z_clip: float = 6.0
    # Iterations of the conservative smooth base.  0 reproduces the legacy
    # block-constant reconstruction, whose 0.5-degree steps are the visible
    # blockiness.  2 is enough to push the residual per-block correction to
    # well under a percent.
    smooth_base_iterations: int = 2

    def validate(self) -> None:
        if self.factor <= 0:
            raise ValueError("factor must be positive")
        if not 0.0 < self.dequant_epsilon < 0.5:
            raise ValueError("dequant_epsilon must lie in (0, 0.5)")
        if self.dequant_noise < 0.0:
            raise ValueError("dequant_noise must be non-negative")
        if self.intensity_floor <= 0.0 or self.denominator_floor <= 0.0:
            raise ValueError("intensity and denominator floors must be positive")
        if not 0.0 <= self.valid_area_threshold <= 1.0:
            raise ValueError("valid_area_threshold must lie in [0, 1]")
        if self.amount_sqrt_std <= 0.0:
            raise ValueError("amount_sqrt_std must be positive")
        if self.intensity_log_std <= 0.0:
            raise ValueError("intensity_log_std must be positive")
        if self.intensity_z_clip <= 0.0:
            raise ValueError("intensity_z_clip must be positive")
        if self.smooth_base_iterations < 0:
            raise ValueError("smooth_base_iterations must be non-negative")

    @classmethod
    def from_mapping(cls, values) -> "SubgridEncoding":
        values = dict(values or {})
        known = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = sorted(set(values) - known)
        if unknown:
            # Silently dropping an unknown key would reinterpret an archive
            # written under a different contract (e.g. the removed
            # ``intensity_log_clip``) using current defaults, producing a
            # different decoder with no error anywhere.
            raise ValueError(
                f"unknown subgrid encoding fields {unknown}; this metadata was "
                "written by a different encoding contract and must be rebuilt "
                "rather than reinterpreted with current defaults"
            )
        return cls(**values)


@dataclass(frozen=True)
class LegacyV2SubgridEncoding:
    """Exact decoder metadata for completed schema-v2 checkpoints.

    This type is deliberately separate from :class:`SubgridEncoding`.  V2
    clips the *unnormalized* log weight, while v4 clips the standardized
    allocation latent.  Treating the renamed field as a current encoding would
    silently change both the sampled field and its observation gradient.
    """

    factor: int = 10
    wet_threshold_mm: float = 0.1
    dequant_epsilon: float = 0.02
    dequant_noise: float = 0.05
    dequant_seed: int = 314159
    intensity_floor: float = 1.0e-5
    denominator_floor: float = 1.0e-8
    valid_area_threshold: float = 0.50
    amount_sqrt_mean: float = 0.0
    amount_sqrt_std: float = 1.0
    intensity_log_mean: float = 0.0
    intensity_log_std: float = 1.0
    intensity_log_clip: float = 12.0

    def validate(self) -> None:
        if self.factor <= 0:
            raise ValueError("factor must be positive")
        if not 0.0 < self.dequant_epsilon < 0.5:
            raise ValueError("dequant_epsilon must lie in (0, 0.5)")
        if self.dequant_noise < 0.0:
            raise ValueError("dequant_noise must be non-negative")
        if self.intensity_floor <= 0.0 or self.denominator_floor <= 0.0:
            raise ValueError("intensity and denominator floors must be positive")
        if not 0.0 <= self.valid_area_threshold <= 1.0:
            raise ValueError("valid_area_threshold must lie in [0, 1]")
        if self.amount_sqrt_std <= 0.0:
            raise ValueError("amount_sqrt_std must be positive")
        if self.intensity_log_std <= 0.0:
            raise ValueError("intensity_log_std must be positive")
        if self.intensity_log_clip <= 0.0:
            raise ValueError("intensity_log_clip must be positive")

    @classmethod
    def from_mapping(cls, values) -> "LegacyV2SubgridEncoding":
        values = dict(values or {})
        known = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"unknown legacy-v2 subgrid encoding fields {unknown}")
        missing = sorted(known - set(values))
        if missing:
            raise ValueError(
                "legacy-v2 subgrid encoding metadata is incomplete; missing "
                f"{missing}"
            )
        encoding = cls(**values)
        encoding.validate()
        return encoding

    @property
    def smooth_base_iterations(self) -> int:
        """V2 reconstructed on a repeated block constant.

        Exposed as a property rather than a field so it stays out of the
        legacy completeness contract: a real v2 archive never carries this key,
        and pinning it to zero keeps the replay bit-for-bit.
        """
        return 0


DecoderEncoding = SubgridEncoding | LegacyV2SubgridEncoding


@dataclass
class SubgridTargets:
    fine_mm: torch.Tensor
    coarse_mm: torch.Tensor
    coarse_state: torch.Tensor
    allocation_state: torch.Tensor
    fine_valid: torch.Tensor
    coarse_valid: torch.Tensor
    valid_area_fraction: torch.Tensor


@dataclass
class ReconstructionDiagnostics:
    empty_wet_fallbacks: int
    positive_blocks: int
    fallback_fraction: float
    minimum_denominator: float
    minimum_denominator_fraction: float
    maximum_weight_to_mean_ratio: float
    maximum_cell_mass_fraction: float
    clipped_intensity_fraction: float


def cell_area_weights(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Spherical cell areas (m2) from regular cell-centre coordinates."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    if lat.ndim != 1 or lon.ndim != 1 or len(lat) < 2 or len(lon) < 2:
        raise ValueError("lat and lon must be one-dimensional with at least two cells")
    if not np.all(np.diff(lat) > 0.0) or not np.all(np.diff(lon) > 0.0):
        raise ValueError("lat and lon must be strictly increasing")

    def edges(centres):
        out = np.empty(len(centres) + 1, dtype=np.float64)
        out[1:-1] = 0.5 * (centres[:-1] + centres[1:])
        out[0] = centres[0] - 0.5 * (centres[1] - centres[0])
        out[-1] = centres[-1] + 0.5 * (centres[-1] - centres[-2])
        return out

    lat_edges = np.deg2rad(np.clip(edges(lat), -90.0, 90.0))
    lon_edges = np.deg2rad(edges(lon))
    radius = 6_371_000.0
    meridional = np.sin(lat_edges[1:]) - np.sin(lat_edges[:-1])
    zonal = lon_edges[1:] - lon_edges[:-1]
    return (radius**2 * meridional[:, None] * zonal[None, :]).astype(np.float32)


def validate_cpc_alignment(grid: Grid, factor: int = 10, coarse_res: float = 0.5) -> None:
    """Fail unless a fine grid closes exactly on native CPC cell edges."""
    if not np.isclose(grid.res * factor, coarse_res, atol=1.0e-10):
        raise ValueError(
            f"{factor} x {grid.res} degree cells do not equal {coarse_res} degrees"
        )
    if grid.nlat % factor or grid.nlon % factor:
        raise ValueError(f"grid {grid.name} shape {grid.shape} is not divisible by {factor}")
    for name, edge in (
        ("lon_min", grid.lon_min), ("lon_max", grid.lon_max),
        ("lat_min", grid.lat_min), ("lat_max", grid.lat_max),
    ):
        phase = edge / coarse_res
        if not np.isclose(phase, round(phase), atol=1.0e-9):
            raise ValueError(f"grid {grid.name} {name}={edge} is off the CPC edge phase")


def validate_aligned_crop(
    origin: tuple[int, int], crop: int, factor: int = 10, downsamplings: int = 3
) -> None:
    """Validate the crop lattice required by CPC blocks and a U-Net pyramid."""
    row, column = map(int, origin)
    if row < 0 or column < 0:
        raise ValueError("crop origins must be non-negative")
    if row % factor or column % factor:
        raise ValueError(f"crop origin {origin} must be 0 modulo {factor}")
    divisor = int(np.lcm(factor, 2**downsamplings))
    if crop <= 0 or crop % divisor:
        raise ValueError(
            f"crop={crop} must be divisible by lcm({factor}, 2^{downsamplings})={divisor}"
        )


def aligned_production_canvas(
    outer: Grid,
    core: Grid,
    canvas: int = 160,
    factor: int = 10,
    downsamplings: int = 3,
) -> tuple[tuple[slice, slice], tuple[slice, slice]]:
    """Return an architecture-safe halo canvas and the core slice inside it."""
    row, column = crop_offsets(outer, core)
    if canvas < core.nlat or canvas < core.nlon:
        raise ValueError("production canvas must contain the complete core grid")
    desired_row = row - (canvas - core.nlat) / 2.0
    desired_column = column - (canvas - core.nlon) / 2.0
    row0 = int(round(desired_row / factor) * factor)
    column0 = int(round(desired_column / factor) * factor)
    row0 = min(max(row0, 0), outer.nlat - canvas)
    column0 = min(max(column0, 0), outer.nlon - canvas)
    # Clamping can move an origin off phase only if outer dimensions are bad;
    # validate rather than silently accepting it.
    validate_aligned_crop((row0, column0), canvas, factor, downsamplings)
    if not (
        row0 <= row and row + core.nlat <= row0 + canvas
        and column0 <= column and column + core.nlon <= column0 + canvas
    ):
        raise ValueError("aligned canvas does not contain the requested core")
    outer_slice = (slice(row0, row0 + canvas), slice(column0, column0 + canvas))
    core_slice = (
        slice(row - row0, row - row0 + core.nlat),
        slice(column - column0, column - column0 + core.nlon),
    )
    return outer_slice, core_slice


def _as_bchw(value: torch.Tensor, name: str) -> torch.Tensor:
    value = torch.as_tensor(value)
    if value.ndim == 2:
        value = value[None, None]
    elif value.ndim == 3:
        value = value[:, None]
    if value.ndim != 4 or value.shape[1] != 1:
        raise ValueError(f"{name} must have shape (H,W), (B,H,W), or (B,1,H,W)")
    return value


def _blockify(value: torch.Tensor, factor: int) -> torch.Tensor:
    b, c, height, width = value.shape
    if height % factor or width % factor:
        raise ValueError(f"fine shape {(height, width)} is not divisible by factor {factor}")
    return value.view(
        b, c, height // factor, factor, width // factor, factor
    ).permute(0, 1, 2, 4, 3, 5)


def _unblockify(value: torch.Tensor) -> torch.Tensor:
    b, c, hc, wc, fy, fx = value.shape
    return value.permute(0, 1, 2, 4, 3, 5).reshape(b, c, hc * fy, wc * fx)


def area_weighted_block_mean(
    field: torch.Tensor,
    area: torch.Tensor,
    valid: torch.Tensor,
    factor: int = 10,
    valid_area_threshold: float = 0.50,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return physical block means, retained-block mask and valid-area fraction."""
    field = _as_bchw(field, "field")
    area = _as_bchw(area, "area").to(device=field.device, dtype=field.dtype)
    valid = _as_bchw(valid, "valid").to(device=field.device, dtype=torch.bool)
    if area.shape[0] == 1 and field.shape[0] > 1:
        area = area.expand(field.shape[0], -1, -1, -1)
    if valid.shape[0] == 1 and field.shape[0] > 1:
        valid = valid.expand(field.shape[0], -1, -1, -1)
    if area.shape != field.shape or valid.shape != field.shape:
        raise ValueError("field, area and valid must broadcast to the same Bx1xHxW shape")
    if not torch.isfinite(area).all() or (area <= 0.0).any():
        raise ValueError("cell areas must be finite and positive")

    ab = _blockify(area, factor)
    vb = _blockify(valid.to(field.dtype), factor)
    fb = _blockify(torch.where(valid, field, torch.zeros_like(field)), factor)
    total_area = ab.sum(dim=(-1, -2))
    valid_area = (ab * vb).sum(dim=(-1, -2))
    fraction = valid_area / total_area.clamp_min(torch.finfo(field.dtype).tiny)
    retained = fraction >= float(valid_area_threshold)
    numerator = (ab * vb * fb).sum(dim=(-1, -2))
    mean = numerator / valid_area.clamp_min(torch.finfo(field.dtype).tiny)
    mean = torch.where(retained, mean, torch.zeros_like(mean))
    return mean, retained, fraction


def conservative_smooth_upsample(
    coarse_mm: torch.Tensor,
    area: torch.Tensor,
    valid: torch.Tensor,
    factor: int = 10,
    iterations: int = 2,
) -> torch.Tensor:
    """Continuous fine field whose area-weighted block means equal ``coarse_mm``.

    Repeating the block mean makes the reconstruction base piecewise constant,
    so ``x = base * weight`` jumps by the ratio of neighbouring block amounts at
    every 0.5-degree edge.  That step is the blockiness: it is a property of the
    conservation support, not of the allocation, and no amount of training
    removes it.  Real 0.05-degree rainfall has no discontinuity there.

    Bilinear interpolation of the block means is continuous but not
    conservative.  Alternating a bilinear lift with a bilinearly-smoothed
    multiplicative correction converges to a field that is both, because the
    correction factor tends to one wherever the lift already has the right block
    mean.  A final exact per-block renormalisation restores machine-precision
    conservation with a factor within ~1e-3 of unity, so the residual step is
    three orders of magnitude smaller than the one it replaces.

    Non-negativity is preserved throughout: bilinear interpolation of a
    non-negative field is non-negative and every correction factor is a ratio of
    non-negative block means.
    """
    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    if coarse_mm.ndim != 4 or coarse_mm.shape[1] != 1:
        raise ValueError("coarse_mm must have shape (B,1,Hc,Wc)")
    if iterations == 0:
        return coarse_mm.repeat_interleave(factor, -2).repeat_interleave(factor, -1)

    floor = torch.finfo(coarse_mm.dtype).tiny

    def lift(block_field: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            block_field, scale_factor=factor, mode="bilinear", align_corners=False
        )

    def block_mean(field: torch.Tensor) -> torch.Tensor:
        mean, _, _ = area_weighted_block_mean(field, area, valid, factor, 0.0)
        return mean

    base = lift(coarse_mm).clamp_min(0.0)
    for _ in range(iterations):
        correction = coarse_mm / block_mean(base).clamp_min(floor)
        base = (base * lift(correction)).clamp_min(0.0)
    exact = coarse_mm / block_mean(base).clamp_min(floor)
    base = base * exact.repeat_interleave(factor, -2).repeat_interleave(factor, -1)
    return base.clamp_min(0.0)


def _logit(probability: torch.Tensor) -> torch.Tensor:
    return torch.log(probability) - torch.log1p(-probability)


def _dequantized_binary_logits(
    target: torch.Tensor, encoding: SubgridEncoding, generator: torch.Generator
) -> torch.Tensor:
    eps = float(encoding.dequant_epsilon)
    probability = torch.where(
        target.to(torch.bool), torch.full_like(target, 1.0 - eps), torch.full_like(target, eps)
    )
    logits = _logit(probability)
    if encoding.dequant_noise:
        uniform = torch.rand(
            target.shape, generator=generator, device="cpu", dtype=torch.float32
        ).clamp_(1.0e-6, 1.0 - 1.0e-6)
        logistic = torch.log(uniform) - torch.log1p(-uniform)
        logits = logits + float(encoding.dequant_noise) * logistic.to(logits.device, logits.dtype)
    return logits


def _centered_allocation_log_weights(
    fine: torch.Tensor,
    coarse: torch.Tensor,
    wet: torch.Tensor,
    area: torch.Tensor,
    encoding: SubgridEncoding,
) -> torch.Tensor:
    """Identifiable log allocation target with zero wet-cell block mean.

    Only relative positive weights matter after conservative block
    normalisation.  Removing each block's area-weighted wet-cell log mean
    eliminates that unidentifiable scale.  Dry cells receive raw value zero,
    a neutral positive weight if their occurrence state later becomes wet.
    """
    coarse_fine = conservative_smooth_upsample(
        coarse, area, wet.new_ones(wet.shape), encoding.factor,
        encoding.smooth_base_iterations,
    )
    relative = fine / coarse_fine.clamp_min(float(encoding.intensity_floor))
    raw = torch.log(relative.clamp_min(float(encoding.intensity_floor)))
    ab = _blockify(area, encoding.factor)
    wb = _blockify(wet.to(raw.dtype), encoding.factor)
    rb = _blockify(raw, encoding.factor)
    wet_area = (ab * wb).sum(dim=(-1, -2), keepdim=True)
    centre = (ab * wb * rb).sum(dim=(-1, -2), keepdim=True) / wet_area.clamp_min(
        float(encoding.denominator_floor)
    )
    centre_fine = _unblockify(
        centre.expand(-1, -1, -1, -1, encoding.factor, encoding.factor)
    )
    return torch.where(wet, raw - centre_fine, torch.zeros_like(raw))


def allocation_log_weight_target(
    fine_mm: torch.Tensor,
    valid: torch.Tensor,
    area: torch.Tensor,
    encoding: SubgridEncoding,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unstandardised centred log weights and their wet-cell mask."""
    encoding.validate()
    fine = _as_bchw(fine_mm, "fine_mm").float().clamp_min(0.0)
    valid_b = _as_bchw(valid, "valid").to(device=fine.device, dtype=torch.bool)
    area_b = _as_bchw(area, "area").to(device=fine.device, dtype=fine.dtype)
    if valid_b.shape[0] == 1 and fine.shape[0] > 1:
        valid_b = valid_b.expand(fine.shape[0], -1, -1, -1)
    if area_b.shape[0] == 1 and fine.shape[0] > 1:
        area_b = area_b.expand(fine.shape[0], -1, -1, -1)
    valid_b = valid_b & torch.isfinite(fine)
    fine = torch.where(valid_b, fine, torch.zeros_like(fine))
    coarse, _, _ = area_weighted_block_mean(
        fine, area_b, valid_b, encoding.factor, encoding.valid_area_threshold
    )
    wet = valid_b & (fine >= float(encoding.wet_threshold_mm))
    return _centered_allocation_log_weights(
        fine, coarse, wet, area_b, encoding
    ), wet


def resolve_archive_encoding(attrs, *, allow_legacy_v4: bool = True):
    """Return ``(encoding, schema)`` for a completed target archive.

    Evaluation and diagnostic paths must be able to read a v4 archive whose
    checkpoint predates the conservative smooth base.  Decoding one with the
    current default would silently change the field the model was fitted to, so
    the legacy block-constant base is pinned here.  Every reader shares this
    function: a schema rule enforced separately in each script is a rule that
    drifts, which is exactly how a sampler and its evaluator end up disagreeing
    about what they will accept.

    Training paths pass ``allow_legacy_v4=False`` -- never fit a new model on a
    superseded target.
    """
    schema = attrs.get("schema")
    accepted = (SUBGRID_SCHEMA,)
    if allow_legacy_v4:
        accepted = accepted + (LEGACY_V4_SUBGRID_SCHEMA,)
    if schema not in accepted:
        raise ValueError(
            f"archive schema {schema!r} is not one of {accepted}; rebuild the "
            "target archive or use a reader that accepts this schema"
        )
    values = dict(attrs.get("subgrid_encoding") or {})
    if schema == LEGACY_V4_SUBGRID_SCHEMA:
        values["smooth_base_iterations"] = 0
    encoding = SubgridEncoding.from_mapping(values)
    encoding.validate()
    return encoding, schema


def coarse_wet_from_fine(wet: torch.Tensor, factor: int = 10) -> torch.Tensor:
    """A coarse block is wet iff it contains at least one wet fine cell.

    Thresholding the *area mean* against the per-cell drizzle threshold is not
    equivalent and silently deletes rain: a 0.5-degree block holding one
    1 mm/day convective cell has a mean of 0.01 mm/day, so a 0.1 mm/day test
    marks the whole block dry and the hard occurrence gate erases it.  Isolated
    convective cells are precisely the subgrid signal this experiment exists to
    resolve, so occurrence must be defined consistently with the reconstruction,
    which places mass only on wet fine cells.
    """
    return _blockify(wet.to(torch.float32), factor).sum(dim=(-1, -2)) > 0.0


def encode_subgrid_targets(
    fine_mm: torch.Tensor,
    valid: torch.Tensor,
    area: torch.Tensor,
    encoding: SubgridEncoding,
    *,
    sample_offset: int = 0,
) -> SubgridTargets:
    """Construct the paired coarse hurdle and fine allocation flow targets."""
    encoding.validate()
    fine = _as_bchw(fine_mm, "fine_mm").float().clamp_min(0.0)
    valid_b = _as_bchw(valid, "valid").to(device=fine.device, dtype=torch.bool)
    area_b = _as_bchw(area, "area").to(device=fine.device, dtype=fine.dtype)
    if valid_b.shape[0] == 1 and fine.shape[0] > 1:
        valid_b = valid_b.expand(fine.shape[0], -1, -1, -1)
    if area_b.shape[0] == 1 and fine.shape[0] > 1:
        area_b = area_b.expand(fine.shape[0], -1, -1, -1)
    finite = torch.isfinite(fine)
    valid_b = valid_b & finite
    fine = torch.where(valid_b, fine, torch.zeros_like(fine))
    coarse, coarse_valid, valid_fraction = area_weighted_block_mean(
        fine, area_b, valid_b, encoding.factor, encoding.valid_area_threshold
    )

    wet = valid_b & (fine >= float(encoding.wet_threshold_mm))
    coarse_wet = coarse_valid & coarse_wet_from_fine(wet, encoding.factor)
    amount = (torch.sqrt(coarse) - float(encoding.amount_sqrt_mean)) / float(
        encoding.amount_sqrt_std
    )
    # Positive-amount statistics are fitted on wet blocks only.  The amount
    # channel is inactive behind a hard-dry occurrence gate, so give dry
    # blocks the neutral standardised wet mean instead of a large ignored
    # negative spike at sqrt(0).
    amount = torch.where(coarse_wet, amount, torch.zeros_like(amount))

    # One seed per time sample makes the target independent of preparation
    # chunking and worker count.
    coarse_logits_parts = []
    fine_logits_parts = []
    for member in range(fine.shape[0]):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            int(encoding.dequant_seed) + int(sample_offset) + member
        )
        coarse_logits_parts.append(
            _dequantized_binary_logits(
                coarse_wet[member : member + 1].to(torch.float32).cpu(),
                encoding,
                generator,
            )
        )
        fine_logits_parts.append(
            _dequantized_binary_logits(
                wet[member : member + 1].to(torch.float32).cpu(),
                encoding,
                generator,
            )
        )
    coarse_logits = torch.cat(coarse_logits_parts).to(fine.device)
    fine_logits = torch.cat(fine_logits_parts).to(fine.device)

    centred_log_weight = _centered_allocation_log_weights(
        fine, coarse, wet, area_b, encoding
    )
    intensity = (
        centred_log_weight - float(encoding.intensity_log_mean)
    ) / float(encoding.intensity_log_std)
    neutral = -float(encoding.intensity_log_mean) / float(
        encoding.intensity_log_std
    )
    intensity = torch.where(wet, intensity, torch.full_like(intensity, neutral))
    intensity = torch.where(valid_b, intensity, torch.zeros_like(intensity))
    fine_logits = torch.where(valid_b, fine_logits, torch.zeros_like(fine_logits))
    amount = torch.where(coarse_valid, amount, torch.zeros_like(amount))
    coarse_logits = torch.where(coarse_valid, coarse_logits, torch.zeros_like(coarse_logits))

    return SubgridTargets(
        fine_mm=fine,
        coarse_mm=coarse,
        coarse_state=torch.cat([amount, coarse_logits], dim=1),
        allocation_state=torch.cat([intensity, fine_logits], dim=1),
        fine_valid=valid_b,
        coarse_valid=coarse_valid,
        valid_area_fraction=valid_fraction,
    )


def hard_forward_soft_backward(
    wetness_logit: torch.Tensor,
    valid: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Straight-through occurrence mask with an exact hard forward value."""
    if temperature <= 0.0:
        raise ValueError("wetness temperature must be positive")
    valid = valid.to(device=wetness_logit.device, dtype=torch.bool)
    soft = torch.sigmoid(wetness_logit / float(temperature)) * valid
    hard = (soft >= 0.5).to(soft.dtype) * valid
    straight_through = (hard - soft).detach() + soft
    return straight_through, hard, soft


def decode_coarse_amount(
    coarse_state: torch.Tensor,
    coarse_valid: torch.Tensor,
    encoding: DecoderEncoding,
    *,
    temperature: float = 1.0,
    hard: bool = True,
) -> torch.Tensor:
    """Decode two coarse flow channels to exact-zero/nonnegative mm/day."""
    if coarse_state.ndim != 4 or coarse_state.shape[1] != 2:
        raise ValueError("coarse_state must have shape (B,2,Hc,Wc)")
    amount_root = coarse_state[:, :1] * float(encoding.amount_sqrt_std) + float(
        encoding.amount_sqrt_mean
    )
    positive = amount_root.clamp_min(0.0).square()
    st, _hard_mask, soft = hard_forward_soft_backward(
        coarse_state[:, 1:2], coarse_valid, temperature
    )
    occurrence = st if hard else soft
    # ``occurrence`` is exactly 0/1 in the forward pass but retains the soft
    # derivative.  A subsequent where(hard_mask, ...) would silently sever the
    # occurrence gradient in dry cells.
    decoded = positive * occurrence
    return torch.where(coarse_valid.to(torch.bool), decoded, torch.zeros_like(decoded))


def reconstruct_from_amount(
    coarse_mm: torch.Tensor,
    allocation_state: torch.Tensor,
    valid: torch.Tensor,
    area: torch.Tensor,
    encoding: DecoderEncoding,
    *,
    temperature: float = 1.0,
    hard: bool = True,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ReconstructionDiagnostics]:
    """Mass-conserving reconstruction of physical 0.05-degree rainfall."""
    encoding.validate()
    if coarse_mm.ndim != 4 or coarse_mm.shape[1] != 1:
        raise ValueError("coarse_mm must have shape (B,1,Hc,Wc)")
    if allocation_state.ndim != 4 or allocation_state.shape[1] != 2:
        raise ValueError("allocation_state must have shape (B,2,H,W)")
    valid_b = _as_bchw(valid, "valid").to(allocation_state.device, torch.bool)
    area_b = _as_bchw(area, "area").to(allocation_state.device, allocation_state.dtype)
    if valid_b.shape[0] == 1 and allocation_state.shape[0] > 1:
        valid_b = valid_b.expand(allocation_state.shape[0], -1, -1, -1)
    if area_b.shape[0] == 1 and allocation_state.shape[0] > 1:
        area_b = area_b.expand(allocation_state.shape[0], -1, -1, -1)
    expected = (
        coarse_mm.shape[-2] * encoding.factor,
        coarse_mm.shape[-1] * encoding.factor,
    )
    if allocation_state.shape[-2:] != expected:
        raise ValueError(
            f"allocation shape {allocation_state.shape[-2:]} does not match "
            f"coarse shape {coarse_mm.shape[-2:]} x factor {encoding.factor}"
        )

    wet_st, wet_hard, wet_soft = hard_forward_soft_backward(
        allocation_state[:, 1:2], valid_b, temperature
    )
    wet_blocks = _blockify(wet_hard, encoding.factor)
    valid_blocks = _blockify(valid_b.to(wet_hard.dtype), encoding.factor)
    empty = (wet_blocks.sum(dim=(-1, -2), keepdim=True) == 0.0) & (
        valid_blocks.sum(dim=(-1, -2), keepdim=True) > 0.0
    )
    positive = (coarse_mm > 0.0).unsqueeze(-1).unsqueeze(-1)
    fallback_blocks = empty & positive

    if fallback_blocks.any():
        probability_blocks = _blockify(wet_soft, encoding.factor)
        flat_probability = probability_blocks.flatten(-2)
        flat_valid = valid_blocks.flatten(-2).to(torch.bool)
        flat_probability = torch.where(
            flat_valid, flat_probability, torch.full_like(flat_probability, -1.0)
        )
        winner = flat_probability.argmax(dim=-1, keepdim=True)
        one_hot = torch.zeros_like(flat_probability).scatter_(-1, winner, 1.0)
        fallback_mask = fallback_blocks.flatten(0, 3).view(
            *fallback_blocks.shape[:-2], 1
        )
        hard_flat = wet_blocks.flatten(-2)
        hard_flat = torch.where(fallback_mask, one_hot, hard_flat)
        wet_hard = _unblockify(hard_flat.view_as(wet_blocks))
        wet_st = (wet_hard - wet_soft).detach() + wet_soft

    occurrence = wet_st if hard else wet_soft
    if isinstance(encoding, LegacyV2SubgridEncoding):
        # Reproduce the schema-v2 decoder literally.  Do not apply the v4
        # standardized clip or block-maximum stabilization: either would make
        # this old checkpoint a different model during both sampling and DA.
        raw_log_weight = allocation_state[:, :1] * float(
            encoding.intensity_log_std
        ) + float(encoding.intensity_log_mean)
        clipped_intensity = (
            raw_log_weight.detach().abs() > float(encoding.intensity_log_clip)
        ) & valid_b
        log_weight = raw_log_weight.clamp(
            min=-float(encoding.intensity_log_clip),
            max=float(encoding.intensity_log_clip),
        )
        positive_weight = torch.exp(log_weight)
    else:
        # The clamp is a hard guard: any cell beyond it has an exactly zero
        # likelihood gradient during DA and can no longer be corrected by an
        # observation.  That has to be observable, not silent.
        clipped_intensity = (
            allocation_state[:, :1].detach().abs() > float(encoding.intensity_z_clip)
        ) & valid_b
        standardized_intensity = allocation_state[:, :1].clamp(
            min=-float(encoding.intensity_z_clip),
            max=float(encoding.intensity_z_clip),
        )
        log_weight = standardized_intensity * float(
            encoding.intensity_log_std
        ) + float(encoding.intensity_log_mean)
        # Allocation is invariant to a common log-weight shift within a block.
        # Subtract the candidate-cell block maximum before exp, so even an unusual
        # fitted scale cannot overflow.  For the hard decoder, dry cells must not
        # set that maximum because their weights are gated to zero.
        log_blocks = _blockify(log_weight, encoding.factor)
        candidate_blocks = _blockify(
            (occurrence.detach() > 0.0) & valid_b, encoding.factor
        )
        block_max = torch.where(
            candidate_blocks,
            log_blocks,
            torch.full_like(log_blocks, -torch.inf),
        ).amax(dim=(-1, -2), keepdim=True)
        block_max = torch.where(
            torch.isfinite(block_max), block_max, torch.zeros_like(block_max)
        )
        positive_weight = _unblockify(
            torch.exp((log_blocks - block_max).clamp_max(0.0))
        )
    # The allocation modulates a continuous base rather than a block constant,
    # so the field no longer steps by the ratio of neighbouring block amounts at
    # every 0.5-degree edge.  The base enters as a dimensionless *shape* --
    # smooth(m) / U(m), which is ~1 everywhere -- rather than as the amount
    # itself.  Multiplying the weights by the amount would drive them to exactly
    # zero in a dry block and destroy the occurrence gradient that lets a
    # dry-cell positive innovation turn that block on; ``scale`` below is the
    # only place the amount enters, exactly as before.  Conservation is imposed
    # by the per-block normalisation and holds for any positive weight field.
    shape_amount = coarse_mm.detach().clamp_min(float(encoding.denominator_floor))
    smooth_shape = conservative_smooth_upsample(
        shape_amount, area_b, valid_b, encoding.factor,
        encoding.smooth_base_iterations,
    ) / shape_amount.repeat_interleave(
        encoding.factor, -2
    ).repeat_interleave(encoding.factor, -1)
    weights = occurrence * positive_weight * smooth_shape
    weights = torch.where(valid_b, weights, torch.zeros_like(weights))
    ab = _blockify(area_b, encoding.factor)
    qb = _blockify(weights, encoding.factor)
    vb = _blockify(valid_b.to(weights.dtype), encoding.factor)
    valid_area = (ab * vb).sum(dim=(-1, -2), keepdim=True)
    denominator = (ab * qb).sum(dim=(-1, -2), keepdim=True)
    scale = valid_area * coarse_mm.unsqueeze(-1).unsqueeze(-1)
    xb = scale * qb / denominator.clamp_min(float(encoding.denominator_floor))
    # ``scale`` is exactly zero for a hard-dry coarse block.  Do not replace it
    # with torch.where: coarse_mm carries the straight-through occurrence
    # derivative needed for a dry-cell positive innovation.
    field = _unblockify(xb)
    field = torch.where(valid_b, field, torch.zeros_like(field)).clamp_min(0.0)

    if not torch.isfinite(field).all():
        raise FloatingPointError("hierarchical reconstruction produced non-finite values")
    if not return_diagnostics:
        return field
    fallback_count = int(fallback_blocks.sum().detach().cpu())
    positive_count = int(positive.sum().detach().cpu())
    positive_mask = positive & (denominator > float(encoding.denominator_floor))
    wet_count = _blockify(occurrence.detach(), encoding.factor).sum(
        dim=(-1, -2), keepdim=True
    )
    mean_wet_weight = qb.sum(dim=(-1, -2), keepdim=True) / wet_count.clamp_min(1.0)
    weight_ratio = qb.amax(dim=(-1, -2), keepdim=True) / mean_wet_weight.clamp_min(
        float(encoding.denominator_floor)
    )
    mass_fraction = (ab * qb).amax(dim=(-1, -2), keepdim=True) / denominator.clamp_min(
        float(encoding.denominator_floor)
    )

    def _masked_max(values: torch.Tensor) -> float:
        selected = values.detach()[positive_mask]
        return float(selected.amax().cpu()) if selected.numel() else 0.0

    selected_denominator = denominator.detach()[positive_mask]
    minimum_denominator = (
        float(selected_denominator.amin().cpu())
        if selected_denominator.numel() else 0.0
    )
    # In v4, subtracting the block maximum before exp forces at least one
    # candidate weight to 1.  The same scale-free ratio remains informative
    # for the exactly replayed v2 decoder.
    denominator_fraction = denominator / valid_area.clamp_min(
        float(encoding.denominator_floor)
    )
    selected_fraction = denominator_fraction.detach()[positive_mask]
    minimum_denominator_fraction = (
        float(selected_fraction.amin().cpu())
        if selected_fraction.numel() else 0.0
    )
    valid_cell_count = valid_b.sum().to(torch.float64).clamp_min(1.0)
    diagnostics = ReconstructionDiagnostics(
        empty_wet_fallbacks=fallback_count,
        positive_blocks=positive_count,
        fallback_fraction=fallback_count / max(positive_count, 1),
        minimum_denominator=minimum_denominator,
        minimum_denominator_fraction=minimum_denominator_fraction,
        maximum_weight_to_mean_ratio=_masked_max(weight_ratio),
        maximum_cell_mass_fraction=_masked_max(mass_fraction),
        clipped_intensity_fraction=float(
            (clipped_intensity.sum().to(torch.float64) / valid_cell_count).cpu()
        ),
    )
    return field, diagnostics


def decode_and_reconstruct(
    coarse_state: torch.Tensor,
    allocation_state: torch.Tensor,
    coarse_valid: torch.Tensor,
    fine_valid: torch.Tensor,
    area: torch.Tensor,
    encoding: DecoderEncoding,
    *,
    temperature: float = 1.0,
    hard: bool = True,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ReconstructionDiagnostics]:
    amount = decode_coarse_amount(
        coarse_state, coarse_valid, encoding, temperature=temperature, hard=hard
    )
    return reconstruct_from_amount(
        amount, allocation_state, fine_valid, area, encoding,
        temperature=temperature, hard=hard,
        return_diagnostics=return_diagnostics,
    )


@dataclass
class SubgridDatasetConfig:
    root: str
    crop: int = 120
    random_crop: bool = True
    crop_origin: tuple[int, int] | None = None
    years: tuple[int, int] | None = None
    factor: int = 10
    downsamplings: int = 3
    seed: int = 0
    tile_domain: bool = False


class SubgridDataset(Dataset):
    """Read the auditable V3 target store produced by script 56."""

    def __init__(self, cfg: SubgridDatasetConfig, store=None):
        self.cfg = cfg
        if store is None:
            import zarr

            store = zarr.open_group(str(Path(cfg.root)), mode="r")
        self.z = store
        attrs = getattr(store, "attrs", {})
        schema = attrs.get("schema")
        if schema != SUBGRID_SCHEMA:
            raise ValueError(
                f"target store schema must be {SUBGRID_SCHEMA}, got {schema!r}; "
                "rebuild the V3 target archive because the decoder contract changed"
            )
        if "subgrid_encoding" not in attrs:
            raise ValueError(
                "target store lacks frozen subgrid_encoding metadata; rebuild the "
                "archive rather than applying current decoder defaults"
            )
        self.encoding = SubgridEncoding.from_mapping(attrs["subgrid_encoding"])
        self.encoding.validate()
        if self.encoding.factor != cfg.factor:
            raise ValueError(
                f"store factor {self.encoding.factor} differs from dataset factor {cfg.factor}"
            )
        self.time = np.asarray(store["time"][:], dtype="datetime64[ns]")
        index = np.arange(len(self.time))
        if cfg.years is not None:
            years = self.time.astype("datetime64[Y]").astype(int) + 1970
            index = index[(years >= cfg.years[0]) & (years <= cfg.years[1])]
        self.index = index
        self.valid = np.asarray(store["fine_valid"][:], dtype=np.float32)
        self.area = np.asarray(store["cell_area"][:], dtype=np.float32)
        self.coarse_valid = np.asarray(store["coarse_valid"][:], dtype=np.float32)
        self.height, self.width = self.valid.shape
        validate_aligned_crop((0, 0), cfg.crop, cfg.factor, cfg.downsamplings)
        if cfg.crop > min(self.height, self.width):
            raise ValueError(f"crop {cfg.crop} exceeds store shape {self.valid.shape}")
        if cfg.crop_origin is not None:
            validate_aligned_crop(
                cfg.crop_origin, cfg.crop, cfg.factor, cfg.downsamplings
            )
        if cfg.tile_domain and cfg.random_crop:
            raise ValueError("tile_domain cannot be combined with random_crop")
        if cfg.tile_domain and cfg.crop_origin is not None:
            raise ValueError("tile_domain cannot be combined with crop_origin")
        self.validation_origins: tuple[tuple[int, int], ...] = ()
        if cfg.tile_domain:
            if self.height % cfg.crop or self.width % cfg.crop:
                raise ValueError(
                    "tile_domain requires crop to divide both spatial dimensions; "
                    "overlapping validation tiles would otherwise weight some cells twice"
                )

            def axis_origins(length: int) -> list[int]:
                return list(range(0, length, cfg.crop))

            self.validation_origins = tuple(
                (row, column)
                for row in axis_origins(self.height)
                for column in axis_origins(self.width)
            )
            for origin in self.validation_origins:
                validate_aligned_crop(
                    origin, cfg.crop, cfg.factor, cfg.downsamplings
                )

    def __len__(self):
        multiplier = len(self.validation_origins) or 1
        return len(self.index) * multiplier

    def _origin(self, i: int) -> tuple[int, int]:
        if not self.cfg.random_crop:
            if self.cfg.crop_origin is not None:
                row, column = self.cfg.crop_origin
            else:
                row = ((self.height - self.cfg.crop) // 2 // self.cfg.factor) * self.cfg.factor
                column = ((self.width - self.cfg.crop) // 2 // self.cfg.factor) * self.cfg.factor
            return row, column
        if self.cfg.crop_origin is not None:
            raise ValueError("crop_origin cannot be combined with random_crop")
        # Worker-local state advances across epochs, so a repeatedly seen day
        # does not receive one frozen crop for the entire training run.
        worker = get_worker_info()
        worker_id = -1 if worker is None else int(worker.id)
        worker_seed = int(torch.initial_seed() % (2**32))
        rng_key = (worker_id, worker_seed)
        # A dataset may be sampled once in the parent process before workers
        # are forked. Keying the generator prevents all workers inheriting the
        # same already-created RNG stream.
        if getattr(self, "_crop_rng_key", None) != rng_key:
            self._crop_rng = np.random.default_rng(
                int(self.cfg.seed) + worker_seed
            )
            self._crop_rng_key = rng_key
        generator = self._crop_rng
        rows = np.arange(0, self.height - self.cfg.crop + 1, self.cfg.factor)
        columns = np.arange(0, self.width - self.cfg.crop + 1, self.cfg.factor)
        return int(generator.choice(rows)), int(generator.choice(columns))

    def __getitem__(self, i: int):
        if self.validation_origins:
            count = len(self.validation_origins)
            j = int(self.index[i // count])
            row, column = self.validation_origins[i % count]
        else:
            j = int(self.index[i])
            row, column = self._origin(i)
        size = self.cfg.crop
        fine_slice = (slice(row, row + size), slice(column, column + size))
        cr, cc = row // self.cfg.factor, column // self.cfg.factor
        cs = size // self.cfg.factor
        coarse_slice = (slice(cr, cr + cs), slice(cc, cc + cs))

        def read(name, index, spatial_slice):
            array = self.z[name]
            value = np.asarray(array[index][(slice(None), *spatial_slice)], dtype=np.float32)
            return torch.from_numpy(value)

        item = {
            "time_ns": torch.tensor(
                self.time[j].astype("datetime64[ns]").astype(np.int64), dtype=torch.int64
            ),
            "coarse_state": read("coarse_state", j, coarse_slice),
            "allocation_state": read("allocation_state", j, fine_slice),
            "coarse_mm": torch.from_numpy(
                np.asarray(self.z["coarse_mm"][j][coarse_slice], dtype=np.float32)
            ).unsqueeze(0),
            "fine_mm": torch.from_numpy(
                np.asarray(self.z["fine_mm"][j][fine_slice], dtype=np.float32)
            ).unsqueeze(0),
            "fine_valid": torch.from_numpy(self.valid[fine_slice]).unsqueeze(0),
            "cell_area": torch.from_numpy(self.area[fine_slice]).unsqueeze(0),
            "coarse_valid": torch.from_numpy(
                self.coarse_valid[coarse_slice]
            ).unsqueeze(0),
            "crop_origin": torch.tensor([row, column], dtype=torch.int64),
        }
        for name, spatial_slice in (
            ("coarse_cond", coarse_slice), ("fine_cond", fine_slice)
        ):
            if name in self.z:
                item[name] = read(name, j, spatial_slice)
            else:
                item[name] = torch.empty(0, *item["coarse_state"].shape[-2:]) if (
                    name == "coarse_cond"
                ) else torch.empty(0, size, size)
        return item


def encoding_metadata(encoding: DecoderEncoding) -> dict:
    """JSON/Zarr-safe representation used by preparation and checkpoints."""
    encoding.validate()
    return asdict(encoding)
