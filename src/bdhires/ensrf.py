"""Localized deterministic ensemble square-root updates for point gauges."""

from __future__ import annotations

import numpy as np

from .transforms import PrecipTransform


def gaspari_cohn(distance_km: np.ndarray, support_km: float) -> np.ndarray:
    """Compact fifth-order localization, equal to zero at ``support_km``."""
    if support_km <= 0:
        raise ValueError("localization support must be positive")
    radius = 2.0 * np.asarray(distance_km, dtype=np.float64) / support_km
    weight = np.zeros_like(radius)
    inner = radius <= 1.0
    value = radius[inner]
    weight[inner] = (
        1.0
        - (5.0 / 3.0) * value**2
        + (5.0 / 8.0) * value**3
        + 0.5 * value**4
        - 0.25 * value**5
    )
    outer = (radius > 1.0) & (radius < 2.0)
    value = radius[outer]
    weight[outer] = (
        4.0
        - 5.0 * value
        + (5.0 / 3.0) * value**2
        + (5.0 / 8.0) * value**3
        - 0.5 * value**4
        + (1.0 / 12.0) * value**5
        - 2.0 / (3.0 * value)
    )
    return np.clip(weight, 0.0, 1.0).astype(np.float32)


def _bilinear_sample_one(
    transformed_ensemble: np.ndarray,
    grid,
    lat: float,
    lon: float,
    transform: PrecipTransform,
) -> np.ndarray:
    """Sample one gauge in physical space, returning transformed values."""
    row = (lat - grid.lat[0]) / grid.res
    col = (lon - grid.lon[0]) / grid.res
    row0 = int(np.clip(np.floor(row), 0, grid.nlat - 1))
    col0 = int(np.clip(np.floor(col), 0, grid.nlon - 1))
    row1 = min(row0 + 1, grid.nlat - 1)
    col1 = min(col0 + 1, grid.nlon - 1)
    wy = float(np.clip(row - row0, 0.0, 1.0))
    wx = float(np.clip(col - col0, 0.0, 1.0))
    physical = transform.inverse(transformed_ensemble)
    sampled = (
        (1.0 - wy) * (1.0 - wx) * physical[:, row0, col0]
        + (1.0 - wy) * wx * physical[:, row0, col1]
        + wy * (1.0 - wx) * physical[:, row1, col0]
        + wy * wx * physical[:, row1, col1]
    )
    return transform.forward(sampled).astype(np.float32)


def localized_serial_ensrf(
    ensemble_mm: np.ndarray,
    observations_mm: np.ndarray,
    station_lat: np.ndarray,
    station_lon: np.ndarray,
    grid,
    transform: PrecipTransform,
    valid: np.ndarray,
    observation_variance: float,
    localization_km: float,
    seed: int,
) -> tuple[np.ndarray, dict]:
    """Apply gauges serially to an IMERG posterior using a localized EnSRF.

    The state is updated in the checkpoint's transformed precipitation space,
    while each nonlinear gauge prediction is computed by interpolation in
    physical mm/day. This makes the IMERG posterior the actual background of
    the gauge update rather than restarting the generative sampler from noise.
    """
    ensemble_mm = np.asarray(ensemble_mm, dtype=np.float32)
    if ensemble_mm.shape[0] < 3:
        raise ValueError("serial EnSRF needs at least three ensemble members")
    state = transform.forward(np.nan_to_num(ensemble_mm, nan=0.0)).astype(np.float32)
    dry = float(transform.forward(np.array(0.0, dtype=np.float32)))
    state[:, ~valid] = dry
    grid_lon, grid_lat = np.meshgrid(grid.lon, grid.lat)
    valid_obs = np.where(np.isfinite(observations_mm))[0]
    order = np.random.default_rng(seed).permutation(valid_obs)
    innovations_before: list[float] = []
    innovations_after: list[float] = []

    for index in order:
        predicted = _bilinear_sample_one(
            state, grid, float(station_lat[index]), float(station_lon[index]), transform
        )
        observed = float(transform.forward(np.array(observations_mm[index])))
        predicted_mean = float(predicted.mean())
        predicted_anomaly = predicted - predicted_mean
        predicted_variance = float(
            np.dot(predicted_anomaly, predicted_anomaly) / (len(predicted) - 1)
        )
        denominator = predicted_variance + observation_variance
        if not np.isfinite(denominator) or denominator <= 1e-8:
            continue
        state_mean = state.mean(axis=0)
        state_anomaly = state - state_mean
        covariance = np.einsum(
            "mij,m->ij", state_anomaly, predicted_anomaly, optimize=True
        ) / (len(predicted) - 1)
        distance = 111.0 * np.sqrt(
            (grid_lat - station_lat[index]) ** 2
            + (
                np.cos(np.deg2rad(station_lat[index]))
                * (grid_lon - station_lon[index])
            )
            ** 2
        )
        gain = gaspari_cohn(distance, localization_km) * covariance / denominator
        innovation = observed - predicted_mean
        alpha = 1.0 / (1.0 + np.sqrt(observation_variance / denominator))
        updated_mean = state_mean + gain * innovation
        updated_anomaly = state_anomaly - alpha * predicted_anomaly[:, None, None] * gain
        state = (updated_mean[None] + updated_anomaly).astype(np.float32)
        state[:, ~valid] = dry
        innovations_before.append(innovation)

    for index in valid_obs:
        predicted = _bilinear_sample_one(
            state, grid, float(station_lat[index]), float(station_lon[index]), transform
        )
        observed = float(transform.forward(np.array(observations_mm[index])))
        innovations_after.append(observed - float(predicted.mean()))

    output = transform.inverse(state).astype(np.float32)
    output[:, ~valid] = np.nan
    return output, {
        "n_gauges": int(len(valid_obs)),
        "innovation_rmse_before_transformed": (
            float(np.sqrt(np.mean(np.square(innovations_before))))
            if innovations_before
            else float("nan")
        ),
        "innovation_rmse_after_transformed": (
            float(np.sqrt(np.mean(np.square(innovations_after))))
            if innovations_after
            else float("nan")
        ),
    }
