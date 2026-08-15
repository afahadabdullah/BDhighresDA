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

    @classmethod
    def from_mapping(cls, values) -> "SubgridEncoding":
        values = dict(values or {})
        known = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in values.items() if key in known})


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


def _logit(probability: torch.Tensor) -> torch.Tensor:
    return torch.log(probability) - torch.log1p(-probability)


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    # y + log(1-exp(-y)) is stable for both tiny and very large y.
    return value + torch.log(-torch.expm1(-value))


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
    coarse_wet = coarse_valid & (coarse >= float(encoding.wet_threshold_mm))
    amount = (torch.sqrt(coarse) - float(encoding.amount_sqrt_mean)) / float(
        encoding.amount_sqrt_std
    )

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

    coarse_fine = coarse.repeat_interleave(encoding.factor, -2).repeat_interleave(
        encoding.factor, -1
    )
    relative = torch.where(
        wet & (coarse_fine > 0.0), fine / coarse_fine.clamp_min(encoding.intensity_floor),
        torch.full_like(fine, encoding.intensity_floor),
    )
    intensity = _inverse_softplus(relative.clamp_min(encoding.intensity_floor))
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
    encoding: SubgridEncoding,
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
    encoding: SubgridEncoding,
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
    positive_weight = F.softplus(allocation_state[:, :1]).clamp_min(
        float(encoding.intensity_floor)
    )
    weights = occurrence * positive_weight
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
    diagnostics = ReconstructionDiagnostics(
        empty_wet_fallbacks=fallback_count,
        positive_blocks=positive_count,
        fallback_fraction=fallback_count / max(positive_count, 1),
        minimum_denominator=float(denominator.detach().amin().cpu()),
    )
    return field, diagnostics


def decode_and_reconstruct(
    coarse_state: torch.Tensor,
    allocation_state: torch.Tensor,
    coarse_valid: torch.Tensor,
    fine_valid: torch.Tensor,
    area: torch.Tensor,
    encoding: SubgridEncoding,
    *,
    temperature: float = 1.0,
    hard: bool = True,
) -> torch.Tensor:
    amount = decode_coarse_amount(
        coarse_state, coarse_valid, encoding, temperature=temperature, hard=hard
    )
    return reconstruct_from_amount(
        amount, allocation_state, fine_valid, area, encoding,
        temperature=temperature, hard=hard,
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


class SubgridDataset(Dataset):
    """Read the auditable V3 target store produced by script 56."""

    def __init__(self, cfg: SubgridDatasetConfig, store=None):
        self.cfg = cfg
        if store is None:
            import zarr

            store = zarr.open_group(str(Path(cfg.root)), mode="r")
        self.z = store
        attrs = getattr(store, "attrs", {})
        self.encoding = SubgridEncoding.from_mapping(attrs.get("subgrid_encoding", {}))
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

    def __len__(self):
        return len(self.index)

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


def encoding_metadata(encoding: SubgridEncoding) -> dict:
    """JSON/Zarr-safe representation used by preparation and checkpoints."""
    encoding.validate()
    return asdict(encoding)
