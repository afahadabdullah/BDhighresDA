"""Differentiable observation operator H mapping a gridded state to stations.

The BMD gauge network is sparse (order 35 synoptic stations over ~148,000
km^2, i.e. one gauge per ~4,200 km^2) and irregular, so H is a bilinear
interpolation from the 0.05 deg grid to arbitrary lat/lon points, implemented
with ``grid_sample`` so that gradients flow back to the state -- which is what
the guidance term in ``bdhires.da.guidance`` needs.

H acts in TRANSFORMED space: the state x is in model units (see
``bdhires.transforms``) and the observations must be transformed identically
before being compared.  This mirrors Manshausen et al. Section 3.2, where
precipitation observations are assimilated as log(P + 1e-4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ..grids import Grid


@dataclass
class StationSet:
    """A set of point observations for a single day (or a batch of days)."""

    lat: np.ndarray  # (S,)
    lon: np.ndarray  # (S,)
    ids: np.ndarray  # (S,) station identifiers
    values: np.ndarray | None = None  # (T, S) mm/day, NaN where missing

    def __len__(self) -> int:
        return len(self.lat)

    def subset(self, idx) -> "StationSet":
        idx = np.asarray(idx)
        return StationSet(
            lat=self.lat[idx],
            lon=self.lon[idx],
            ids=self.ids[idx],
            values=None if self.values is None else self.values[:, idx],
        )


def normalized_coords(grid: Grid, lat: np.ndarray, lon: np.ndarray) -> torch.Tensor:
    """Map lat/lon to ``grid_sample`` coordinates in [-1, 1].

    Uses ``align_corners=True`` semantics: -1 maps to the centre of the first
    cell and +1 to the centre of the last cell.  Latitude is ASCENDING in our
    arrays (row 0 = south), so no flip is required.
    """
    glat, glon = grid.lat, grid.lon
    x = 2.0 * (lon - glon[0]) / (glon[-1] - glon[0]) - 1.0
    y = 2.0 * (lat - glat[0]) / (glat[-1] - glat[0]) - 1.0
    out_of_domain = (np.abs(x) > 1.0) | (np.abs(y) > 1.0)
    if out_of_domain.any():
        raise ValueError(
            f"{out_of_domain.sum()} station(s) fall outside grid {grid.name}: "
            f"{list(zip(np.asarray(lat)[out_of_domain], np.asarray(lon)[out_of_domain]))}"
        )
    return torch.tensor(np.stack([x, y], axis=-1), dtype=torch.float32)


class BilinearObsOperator(torch.nn.Module):
    """H: (B, C, H, W) state -> (B, C, S) station-space values."""

    def __init__(self, grid: Grid, lat: np.ndarray, lon: np.ndarray):
        super().__init__()
        coords = normalized_coords(grid, np.asarray(lat), np.asarray(lon))
        # grid_sample wants (B, H_out, W_out, 2); we use H_out = 1, W_out = S
        self.register_buffer("coords", coords.view(1, 1, -1, 2))
        self.n_stations = coords.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        grid = self.coords.expand(b, -1, -1, -1).to(x.dtype)
        out = F.grid_sample(x, grid, mode="bilinear", align_corners=True, padding_mode="border")
        return out[:, :, 0, :]  # (B, C, S)

    # Adjoint is only needed for diagnostics; autograd handles the real work.
    def nearest_indices(self, grid: Grid, lat, lon):
        row = np.clip(np.round((np.asarray(lat) - grid.lat[0]) / grid.res).astype(int), 0, grid.nlat - 1)
        col = np.clip(np.round((np.asarray(lon) - grid.lon[0]) / grid.res).astype(int), 0, grid.nlon - 1)
        return row, col


class BlockAverageObsOperator(torch.nn.Module):
    """H: 0.05 deg state -> 0.1 deg satellite footprints (exact 2x2 block mean).

    Used to assimilate IMERG as an *observation* rather than feeding it in as a
    conditioning channel.  The block average is the correct forward operator: a
    0.1 deg IMERG value is (to first order) the area-average of the true field
    over that footprint, so H is exactly a 2x2 mean when the grids nest -- which
    ours do, because both ``BD`` and ``WIDE`` have edges on multiples of 0.1 deg.

    Cells whose footprint is not fully valid (partly ocean, where CHIRPS has no
    data) are returned but should be masked out via NaNs in ``y``.
    """

    def __init__(self, factor: int = 2, valid: np.ndarray | None = None,
                 min_valid_frac: float = 0.999):
        super().__init__()
        self.factor = factor
        keep = None
        if valid is not None:
            v = torch.as_tensor(valid, dtype=torch.float32)[None, None]
            frac = F.avg_pool2d(v, factor)
            keep = (frac >= min_valid_frac).flatten()
        self.register_buffer("keep", keep if keep is not None else torch.empty(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        coarse = F.avg_pool2d(x, self.factor)          # (B, C, H/f, W/f)
        out = coarse.reshape(b, c, -1)
        return out

    def valid_mask(self) -> torch.Tensor | None:
        return self.keep if self.keep.numel() else None


class CompositeObsOperator(torch.nn.Module):
    """Concatenate several observation streams into one (B, C, S_total) vector.

    This is how multi-source assimilation is done here: gauges and (optionally)
    IMERG footprints go into a *single* Gaussian likelihood with a
    per-observation error variance.  Because the guidance gradient is a single
    backward pass through the network regardless of how many observations there
    are, adding the ~4k IMERG footprints costs essentially nothing on top of the
    ~35 gauges.
    """

    def __init__(self, operators: list[torch.nn.Module]):
        super().__init__()
        self.ops = torch.nn.ModuleList(operators)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([op(x) for op in self.ops], dim=-1)

    @property
    def sizes(self) -> list[int]:
        return [getattr(op, "n_stations", None) for op in self.ops]


def build_R(
    n_stations: int,
    sigma_obs: float,
    device=None,
    representativeness: float = 0.0,
) -> torch.Tensor:
    """Diagonal observation-error covariance in transformed space.

    ``sigma_obs`` is the gauge measurement/reporting error; the
    ``representativeness`` term inflates it to account for the mismatch
    between a point gauge and a 5 km cell average, which for daily convective
    rainfall in the monsoon is usually the *dominant* term.  Both are tuned on
    held-out stations (see ``scripts/evaluate.py --tune``).
    """
    var = sigma_obs**2 + representativeness**2
    return torch.full((n_stations,), var, dtype=torch.float32, device=device)


def build_R_multi(specs: list[tuple[int, float, float]], device=None) -> torch.Tensor:
    """Concatenated diagonal R for a ``CompositeObsOperator``.

    ``specs`` is a list of ``(n_obs, sigma_obs, representativeness)`` in the same
    order as the operators.  Relative magnitudes are what matter: setting IMERG's
    variance ~10-25x the gauge variance encodes "trust the gauge amplitude, trust
    the satellite pattern", which is the intended division of labour.
    """
    return torch.cat([build_R(n, s, device=device, representativeness=r) for n, s, r in specs])


def imerg_error_variance(
    y_imerg_mm: np.ndarray,
    sigma_floor: float = 0.30,
    sigma_slope: float = 0.15,
    max_sigma: float = 1.5,
) -> np.ndarray:
    """Intensity-dependent IMERG error sd in TRANSFORMED (log1p) space.

    IMERG error is strongly regime-dependent: near-zero for confident dry
    scenes, large for light/warm-rain and orographic events that the passive
    microwave retrieval misses and the IR fill-in guesses at.  A flat R either
    over-trusts the satellite in the pre-monsoon or wastes it in July.  This is
    a deliberately simple heuristic -- replace it with an empirical
    IMERG-vs-CHIRPS error model fitted per season and elevation band
    (``scripts/07_bias_correct_imerg.py --fit-error-model``).
    """
    s = sigma_floor + sigma_slope * np.log1p(np.clip(y_imerg_mm, 0, None))
    return np.clip(s, sigma_floor, max_sigma) ** 2


def split_stations(
    stations: StationSet, n_folds: int = 3, seed: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """K-fold assimilate/evaluate splits over the station network.

    Returns a list of ``(assim_idx, eval_idx)``.  Rotating the withheld set is
    essential with only ~35 gauges: a single split gives evaluation numbers
    with huge sampling error.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(stations))
    folds = np.array_split(idx, n_folds)
    out = []
    for k in range(n_folds):
        eval_idx = np.sort(folds[k])
        assim_idx = np.sort(np.concatenate([folds[j] for j in range(n_folds) if j != k]))
        out.append((assim_idx, eval_idx))
    return out
