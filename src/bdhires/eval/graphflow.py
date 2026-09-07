"""Single-observation response diagnostics for the experimental G0 backbone.

All distances below are fine-grid cells, not kilometres. A one-step guidance
increment is a sensitivity diagnostic, not a completed posterior analysis.
"""

from __future__ import annotations

import numpy as np
import torch

from ..da.guidance import GuidanceConfig, guidance_grad
from ..da.observation import PhysicalBilinearObsOperator
from ..models.flow import RectifiedFlow, VelocityOnly


def response_metrics(field: np.ndarray, row: float, col: float, distant_cells: float) -> dict:
    """Radial amplitude/energy, anisotropy, spectrum and distant response."""
    field = np.asarray(field, dtype=np.float64)
    if field.ndim != 2 or not np.isfinite(field).all():
        raise ValueError("response must be a finite HxW array")
    yy, xx = np.indices(field.shape)
    dx, dy = xx - col, yy - row
    distance = np.hypot(dx, dy)
    bins = np.floor(distance).astype(int)
    absolute = np.abs(field)
    counts = np.bincount(bins.ravel())
    radial = np.bincount(bins.ravel(), weights=absolute.ravel()) / counts.clip(1)
    radial_energy = np.bincount(bins.ravel(), weights=(field**2).ravel())
    energy = float(radial_energy.sum())
    cdf = np.cumsum(radial_energy) / max(energy, 1e-30)
    weights = field**2 / max(energy, 1e-30)
    moments = np.array([[np.sum(weights * dx**2), np.sum(weights * dx * dy)],
                        [np.sum(weights * dx * dy), np.sum(weights * dy**2)]])
    eigenvalues, eigenvectors = np.linalg.eigh(moments)
    anisotropy = (eigenvalues[-1] - eigenvalues[0]) / max(eigenvalues.sum(), 1e-30)
    power = np.abs(np.fft.fft2(field - field.mean()))**2 / field.size**2
    fy, fx = np.meshgrid(np.fft.fftfreq(field.shape[0]), np.fft.fftfreq(field.shape[1]), indexing="ij")
    scale = min(field.shape)
    freq_bins = np.floor(np.hypot(fx, fy) * scale).astype(int)
    spectrum = np.bincount(freq_bins.ravel(), weights=power.ravel())
    far = absolute[distance >= distant_cells]
    return dict(
        l2_norm=float(np.linalg.norm(field)), signed_sum=float(field.sum()),
        absolute_sum=float(absolute.sum()), maximum=float(absolute.max()),
        distant_threshold_cells=float(distant_cells), maximum_distant_response=float(far.max()) if far.size else 0.0,
        radial_distance_cells=np.arange(radial.size).tolist(), radial_mean_abs=radial.tolist(),
        radial_energy=radial_energy.tolist(),
        radius_90_energy_cells=int(np.searchsorted(cdf, 0.9)) if energy else None,
        anisotropy=float(anisotropy),
        principal_axis_degrees=float(np.degrees(np.arctan2(*eigenvectors[::-1, -1]))) if energy else None,
        spectrum_frequency_cycles_per_cell=(np.arange(spectrum.size) / scale).tolist(),
        spectrum_power=spectrum.tolist(),
    )


def single_observation_response(
    model, x, cond, *, grid, transform, residual, base, mask,
    row: float, col: float, time: float = 0.5, innovation: float = 0.25,
    sigma_obs: float = 0.10, representativeness: float = 0.25,
    gamma: float = 0.01, spread_cells: float = 6.0, step_size: float = 0.02,
    clip_norm: float | None = 50.0,
) -> dict:
    """Differentiate a fixed transformed-space point innovation through a model.

    Each backbone receives the same state, condition and innovation relative to
    its own denoised point estimate. The observation is detached before taking
    derivatives. Physical bilinear H and residual.decode are the existing code.
    """
    if x.shape != (1, 1, grid.nlat, grid.nlon):
        raise ValueError("single-observation diagnostic requires one state matching grid")
    if not 0 <= row <= grid.nlat - 1 or not 0 <= col <= grid.nlon - 1:
        raise ValueError("observation position lies outside the grid")
    if not 0 < time < 1 or step_size <= 0:
        raise ValueError("time must lie in (0,1) and step_size must be positive")
    H = PhysicalBilinearObsOperator(
        grid, np.array([grid.lat[0] + row * grid.res]),
        np.array([grid.lon[0] + col * grid.res]), transform,
        valid=mask[0, 0].detach().cpu().numpy(),
    ).to(x.device)
    model.eval()
    velocity = VelocityOnly(model)
    flow = RectifiedFlow()
    t = torch.full((1,), time, device=x.device)
    def decode(field):
        return residual.decode(field, base)
    with torch.no_grad():
        clean = flow.x1_hat(x, t, velocity(x, t, cond))
        clean = clean * mask + residual.fill * (1 - mask)
        y = H(decode(clean)) + innovation
    R = torch.full((1,), sigma_obs**2 + representativeness**2, device=x.device)
    results = {}
    for label, spread in (("raw", 0.0), ("spread", spread_cells)):
        cfg = GuidanceConfig(gamma=gamma, spread_cells=spread, clip_norm=clip_norm)
        _, gradient = guidance_grad(x, t, velocity, flow, cond, H, y, R, cfg,
                                    mask=mask, mask_fill=residual.fill, to_precip=decode)
        increment = step_size * flow.score_to_velocity_factor(t, x) * gradient
        results[label + "_gradient"] = gradient[0, 0].cpu().numpy()
        results[label + "_increment"] = increment[0, 0].cpu().numpy()
    results["observation_transformed"] = float(y.item())
    return results
