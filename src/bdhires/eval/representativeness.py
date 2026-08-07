"""Point-vs-area error estimated from the gauge network itself.

Why this module exists
----------------------
Every score in this project that compares a model cell to a rain gauge is
really measuring three things added in quadrature:

    observed error^2  =  model error^2
                      +  representativeness error^2   (a point is not a cell)
                      +  reference error^2            (CHIRPS is not truth)

The middle term has been a *guess* until now: ``configs/*.yaml`` carries
``representativeness: 0.25`` and ``obs_sd_for_verification: 0.10`` in
transformed units with nothing behind them.  Those two numbers set how hard a
gauge pulls the analysis and how harshly the ensemble is scored, so guessing
them wrong biases both the assimilation and its verification.

This module estimates the term from the BMD network instead, using the standard
geostatistical route: fit a variogram to station pairs, then integrate it over a
grid cell.

The identity being used
-----------------------
For a random point ``x`` inside a block ``V``,

    E[(Z(x) - Z_V)^2] = gammabar(V, V)

where ``Z_V`` is the block average and ``gammabar(V, V)`` is the mean variogram
between two independent random points of ``V``.  So the point-to-block error is
obtained by averaging the fitted variogram over pairs of points within one cell
-- no assumption that any product is truth, and no circularity, because only
gauges are involved.

The honest limitation
---------------------
BMD's closest station pair is tens of kilometres apart, so the nugget is an
EXTRAPOLATION below the shortest observed separation, not a measurement.  It is
the least constrained parameter here and it is also the one that dominates
``gammabar(V, V)`` for a 5 km cell.  ``fit_variogram`` therefore reports
``min_separation_km`` alongside the fit, and any use of the nugget should quote
it.  Treat the result as a well-founded estimate with a stated uncertainty, not
as a measured constant.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EARTH_RADIUS_KM = 6371.0088


def haversine_km(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Great-circle distance in km, broadcasting over the inputs."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(a, float)) for a in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))


def pair_distances_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Full (S, S) distance matrix between stations."""
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    return haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])


def empirical_variogram(
    values: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    bin_edges_km: np.ndarray,
    min_pairs: int = 30,
) -> dict:
    """Classical Matheron variogram pooled over time.

    ``values`` is ``(T, S)`` with NaN for missing.  Each day contributes every
    station pair that reported on that day, so the estimate uses all
    ``T * S * (S - 1) / 2`` available pairs rather than a single snapshot.

    Returns bin centres, ``gamma``, and the pair count per bin.  Bins with fewer
    than ``min_pairs`` contributing pairs are returned as NaN -- a variogram
    point built from a handful of pairs is noise and fitting through it is worse
    than dropping it.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"values must be (T, S), got shape {values.shape}")
    distance = pair_distances_km(lat, lon)
    iu, ju = np.triu_indices(values.shape[1], k=1)
    d_pairs = distance[iu, ju]

    bin_edges_km = np.asarray(bin_edges_km, float)
    which = np.digitize(d_pairs, bin_edges_km) - 1
    n_bins = len(bin_edges_km) - 1

    total = np.zeros(n_bins)
    count = np.zeros(n_bins, dtype=np.int64)
    distance_sum = np.zeros(n_bins)

    for t in range(values.shape[0]):
        row = values[t]
        difference = row[iu] - row[ju]
        ok = np.isfinite(difference) & (which >= 0) & (which < n_bins)
        if not ok.any():
            continue
        np.add.at(total, which[ok], difference[ok] ** 2)
        np.add.at(count, which[ok], 1)
        np.add.at(distance_sum, which[ok], d_pairs[ok])

    with np.errstate(invalid="ignore", divide="ignore"):
        gamma = 0.5 * total / count
        centre = distance_sum / count
    thin = count < min_pairs
    gamma[thin] = np.nan
    centre[thin] = np.nan
    fallback = 0.5 * (bin_edges_km[:-1] + bin_edges_km[1:])
    centre = np.where(np.isfinite(centre), centre, fallback)

    return {
        "distance_km": centre,
        "gamma": gamma,
        "n_pairs": count,
        "min_separation_km": float(d_pairs.min()) if d_pairs.size else float("nan"),
    }


@dataclass
class VariogramFit:
    """Exponential variogram ``gamma(h) = nugget + sill * (1 - exp(-h / range))``.

    ``nugget`` carries the measurement error plus all variance at scales below
    the closest station pair; ``sill`` is the additional variance reached at
    large separation; ``range_km`` is the e-folding decorrelation distance
    (the "practical range" where gamma reaches 95% of its total is ~3x this).
    """

    nugget: float
    sill: float
    range_km: float
    min_separation_km: float = float("nan")
    n_bins_used: int = 0
    rmse: float = float("nan")
    units: str = ""
    metadata: dict = field(default_factory=dict)

    def __call__(self, h: np.ndarray) -> np.ndarray:
        h = np.asarray(h, float)
        out = self.nugget + self.sill * (1.0 - np.exp(-h / max(self.range_km, 1e-9)))
        return np.where(h > 0, out, 0.0)

    @property
    def total_sill(self) -> float:
        return self.nugget + self.sill

    @property
    def nugget_fraction(self) -> float:
        """Share of total variance that is sub-station-separation."""
        return self.nugget / self.total_sill if self.total_sill > 0 else float("nan")

    def to_dict(self) -> dict:
        return {
            "nugget": self.nugget,
            "sill": self.sill,
            "range_km": self.range_km,
            "total_sill": self.total_sill,
            "nugget_fraction": self.nugget_fraction,
            "min_separation_km": self.min_separation_km,
            "n_bins_used": self.n_bins_used,
            "fit_rmse": self.rmse,
            "units": self.units,
            **self.metadata,
        }


def fit_variogram(
    distance_km: np.ndarray,
    gamma: np.ndarray,
    n_pairs: np.ndarray | None = None,
    min_separation_km: float = float("nan"),
    units: str = "",
    min_range_km: float | None = None,
) -> VariogramFit:
    """Least-squares exponential fit, weighted by pair count.

    Solved as a 1-D search over the range with nugget and sill obtained by
    non-negative linear least squares at each candidate.  That is stable without
    SciPy and cannot wander into a negative nugget, which the unconstrained
    solution happily does when the short-separation bins are noisy.

    The range is bounded below by ``min_range_km`` (default: the closest station
    pair).  This matters more than it looks.  A pure-nugget field and an
    exponential with a 3 km range are IDENTICAL at every separation the network
    can see, so an unbounded search will happily return ``nugget = 0, range =
    3 km`` for data that is pure nugget -- inventing sub-network structure that
    nothing in the data supports.  It is not a cosmetic difference: a 5 km cell
    straddles that invented range, so ``block_dispersion_variance`` returns a
    materially smaller number (0.72 against 0.93 in transformed units on the
    synthetic check -- a 28% understatement of the representativeness error, in
    the direction that makes the model look worse than it is).

    Bounding the range at the shortest observed separation forces variance that
    the network cannot resolve into the nugget, where it belongs.  It is the
    conservative choice and it makes the extrapolation explicit rather than
    hiding it inside a fitted parameter.
    """
    distance_km = np.asarray(distance_km, float)
    gamma = np.asarray(gamma, float)
    ok = np.isfinite(distance_km) & np.isfinite(gamma)
    if ok.sum() < 3:
        raise ValueError(
            f"need at least 3 usable variogram bins to fit, got {int(ok.sum())}"
        )
    h = distance_km[ok]
    g = gamma[ok]
    weight = (
        np.asarray(n_pairs, float)[ok] if n_pairs is not None else np.ones_like(g)
    )
    weight = weight / weight.sum()

    if min_range_km is None:
        # The smallest bin the fit can actually SEE, which is not the same as the
        # closest station pair: a lone short pair usually lands in a bin that
        # ``empirical_variogram`` drops for having too few members, so bounding
        # by the raw minimum would license structure at a separation that
        # contributes nothing to the objective.
        min_range_km = float(h.min())
    min_range_km = max(float(min_range_km), 1e-3)
    best = None
    for range_km in np.geomspace(
        min_range_km, max(h.max() * 5.0, min_range_km * 10.0), 400
    ):
        basis = 1.0 - np.exp(-h / range_km)
        design = np.stack([np.ones_like(basis), basis], axis=1)
        wd = design * weight[:, None]
        try:
            coefficients = np.linalg.solve(design.T @ wd, wd.T @ g)
        except np.linalg.LinAlgError:
            continue
        nugget, sill = float(coefficients[0]), float(coefficients[1])
        # Non-negativity: refit the other term alone if either goes negative.
        if nugget < 0 or sill < 0:
            if sill < 0:
                nugget, sill = float(np.sum(weight * g)), 0.0
            else:
                sill = float(np.sum(weight * g * basis) / max(np.sum(weight * basis**2), 1e-12))
                nugget = 0.0
            nugget, sill = max(nugget, 0.0), max(sill, 0.0)
        residual = g - (nugget + sill * basis)
        score = float(np.sum(weight * residual**2))
        if best is None or score < best[0]:
            best = (score, nugget, sill, range_km)

    score, nugget, sill, range_km = best
    return VariogramFit(
        nugget=nugget,
        sill=sill,
        range_km=float(range_km),
        min_separation_km=float(min_separation_km),
        n_bins_used=int(ok.sum()),
        rmse=float(np.sqrt(score)),
        units=units,
    )


def block_dispersion_variance(
    variogram: VariogramFit,
    cell_km: float,
    n_sub: int = 12,
) -> float:
    """``gammabar(V, V)``: mean variogram between two random points of a cell.

    This IS the expected squared difference between a point gauge and the
    average over the cell containing it, which is exactly the representativeness
    error.  Computed by discretising the cell into ``n_sub`` x ``n_sub`` points
    and averaging over all ordered pairs, which converges quickly because the
    variogram is smooth.

    Note the nugget enters at full strength: two distinct points in the cell are
    separated by more than zero, so every pair sees ``nugget``.  For a 5 km cell
    against BMD's ~30 km minimum separation, this term dominates -- which is
    precisely why the extrapolation warning in the module docstring matters.
    """
    if cell_km <= 0:
        return 0.0
    offsets = (np.arange(n_sub) + 0.5) / n_sub * cell_km
    xx, yy = np.meshgrid(offsets, offsets, indexing="ij")
    points = np.stack([xx.ravel(), yy.ravel()], axis=1)
    separation = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    iu, ju = np.triu_indices(len(points), k=1)
    return float(np.mean(variogram(separation[iu, ju])))


def representativeness_sigma(
    variogram: VariogramFit,
    cell_km: float,
    n_sub: int = 12,
) -> float:
    """Point-to-cell error standard deviation, in the variogram's own units."""
    return float(np.sqrt(max(block_dispersion_variance(variogram, cell_km, n_sub), 0.0)))


def aggregation_decomposition(
    windows: np.ndarray, mean_square_error: np.ndarray
) -> dict:
    """Split error into a persistent floor and a part that averages away.

    Fits ``MSE(N) = systematic + random / N`` over aggregation windows ``N``.

    The logic: averaging over N days divides the variance of any zero-mean,
    weakly-correlated error by N, while a persistent offset -- a gauge in a
    valley whose cell average includes a ridge, or a product with a real bias --
    survives untouched.  So the intercept is the part that no amount of
    averaging will remove and the slope is the part that will.

    ``systematic`` therefore bounds how well a product could EVER match a gauge,
    and comparing it across products separates "this product is biased here"
    from "this product is noisy here".

    Two ways the model can fail, both flagged rather than silently returned:

    * ``systematic < 0`` -- the curve falls faster than 1/N.
    * ``random < 0`` -- error GROWS with averaging, which the 1/N law cannot
      produce at all.  The usual cause is error correlated across days (a wet
      spell mistimed by one day hurts a pentad mean more than a daily score) or
      too few independent aggregation windows at the long end.

    Either one means ``systematic`` should not be quoted as a representativeness
    floor.  Check ``model_is_valid`` before using the numbers.
    """
    windows = np.asarray(windows, float)
    mean_square_error = np.asarray(mean_square_error, float)
    ok = np.isfinite(windows) & np.isfinite(mean_square_error) & (windows > 0)
    if ok.sum() < 2:
        raise ValueError("need at least 2 aggregation windows to decompose")
    design = np.stack([np.ones(ok.sum()), 1.0 / windows[ok]], axis=1)
    solution, *_ = np.linalg.lstsq(design, mean_square_error[ok], rcond=None)
    systematic, random = float(solution[0]), float(solution[1])
    systematic_negative = systematic < 0
    # Error rising with the averaging window is outside what MSE = a + b/N can
    # describe, so the fitted floor is meaningless even though it is positive.
    random_negative = random < 0
    systematic = max(systematic, 0.0)
    predicted = design @ np.array([systematic, random])
    residual = mean_square_error[ok] - predicted
    denominator = mean_square_error[ok] - mean_square_error[ok].mean()
    r_squared = (
        1.0 - float(np.sum(residual**2)) / float(np.sum(denominator**2))
        if np.any(denominator)
        else float("nan")
    )
    return {
        "systematic_mse": systematic,
        "random_mse": random,
        "systematic_rmse": float(np.sqrt(systematic)),
        "random_rmse_daily": float(np.sqrt(max(random, 0.0))),
        "r_squared": r_squared,
        "intercept_was_negative": bool(systematic_negative),
        "error_grows_with_averaging": bool(random_negative),
        "model_is_valid": not (systematic_negative or random_negative),
        "n_windows": int(ok.sum()),
    }
