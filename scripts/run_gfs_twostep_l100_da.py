#!/usr/bin/env python3
"""Run gfs_twostep_l100 GraphFlow DA and generate 2D spatial diagnostic maps.

This script isolates the ``gfs_twostep_l100`` assimilation arm (IMERG-guided
generative flow followed by localized serial EnSRF with a 100 km Gaspari-Cohn
radius) and produces complete 2D spatial diagnostic maps of background, satellite,
analysis, increments, ensemble spread, and station verification.

It supports two modes:
1. Full DA execution + spatial plotting:
       python scripts/run_gfs_twostep_l100_da.py \
           --graphflow-ckpt runs/prior_h100_cpc_graphflow_g0/best.pt \
           --unet-ckpt runs/prior_h100_cpc_v2/best.pt \
           --stations data/processed/v2_simultaneous_refinement/ing2022_s04/fold0_bmd.csv \
           --imerg data/processed/imerg_prepared_ing2022/imerg_0p4deg_20220501_20220510.nc \
           --out-dir runs/graphflow_g0_multimesh/gfs_twostep_l100_da

2. Plot-only from an existing sweep dump:
       python scripts/run_gfs_twostep_l100_da.py \
           --plot-only \
           --dump runs/graphflow_g0_multimesh/da_screen_may2022/graphflow_g0_multimesh/fold0.npz \
           --out-dir runs/graphflow_g0_multimesh/gfs_twostep_l100_da/plots
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.colors import Normalize  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ARM_NAME = "gfs_twostep_l100"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Execution mode
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip DA run; generate plots from an existing .npz dump",
    )
    parser.add_argument(
        "--run-da",
        action="store_true",
        help="Force running DA even if --dump points to an existing file",
    )

    # Model & DA inputs
    parser.add_argument(
        "--graphflow-ckpt",
        default="runs/prior_h100_cpc_graphflow_g0/best.pt",
        help="Trained GraphFlow multimesh checkpoint",
    )
    parser.add_argument(
        "--unet-ckpt",
        default="runs/prior_h100_cpc_v2/best.pt",
        help="Paired CPCv2 baseline checkpoint for schema validation",
    )
    parser.add_argument(
        "--stations",
        default="data/processed/v2_simultaneous_refinement/ing2022_s04/fold0_bmd.csv",
        help="Canonical BMD station CSV for the selected fold",
    )
    parser.add_argument(
        "--imerg",
        default="data/processed/imerg_prepared_ing2022/imerg_0p4deg_20220501_20220510.nc",
        help="Prepared IMERG NetCDF for the screening period",
    )
    parser.add_argument("--start", default="2022-05-01")
    parser.add_argument("--end", default="2022-05-10")
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        choices=range(5),
        help="Spatial holdout fold (0..4)",
    )
    parser.add_argument(
        "--members", type=int, default=30, help="Number of ensemble members"
    )
    parser.add_argument(
        "--save-members",
        action="store_true",
        help="Save full (days, members, nlat, nlon) grids in output NPZ",
    )

    # I/O destinations
    parser.add_argument(
        "--dump",
        default=None,
        help="Path to .npz file (input for --plot-only, or output destination)",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Path to .json report (input for --plot-only, or output destination)",
    )
    parser.add_argument(
        "--out-dir",
        default="runs/graphflow_g0_multimesh/gfs_twostep_l100_da",
        help="Output directory for spatial plots, summaries, and dumps",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Scoring and Station Metrics Helpers
# ---------------------------------------------------------------------------


def fair_crps_1d(members: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Unbiased ensemble CRPS for (n_samples, n_members) vs (n_samples,)."""
    members = np.asarray(members, dtype=float)
    truth = np.asarray(truth, dtype=float)
    m = members.shape[1]
    if m < 2:
        return np.abs(members[:, 0] - truth)
    first = np.mean(np.abs(members - truth[:, None]), axis=1)
    ordered = np.sort(members, axis=1)
    weights = 2 * np.arange(1, m + 1) - m - 1
    pair = np.sum(ordered * weights[None, :], axis=1) / (m * (m - 1))
    return first - pair


def compute_metrics(predicted: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(predicted) & np.isfinite(truth)
    if not valid.any():
        return {"mae": np.nan, "rmse": np.nan, "bias": np.nan, "corr": np.nan}
    p, t = predicted[valid], truth[valid]
    mae = float(np.mean(np.abs(p - t)))
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    bias = float(np.mean(p - t))
    corr = (
        float(np.corrcoef(p, t)[0, 1])
        if len(p) > 1 and p.std() > 0 and t.std() > 0
        else np.nan
    )
    return {"mae": mae, "rmse": rmse, "bias": bias, "corr": corr}


# ---------------------------------------------------------------------------
# Targeted DA Runner
# ---------------------------------------------------------------------------


def run_targeted_da(args: argparse.Namespace, dump_path: Path, report_path: Path) -> None:
    """Run DA exclusively for background + gfs_twostep_l100."""
    print(f"[gfs_twostep_l100] loading GraphFlow checkpoint: {args.graphflow_ckpt}")
    print(f"[gfs_twostep_l100] period: {args.start} to {args.end}; fold {args.fold}; {args.members} members")

    spec = importlib.util.spec_from_file_location(
        "_graphflow_sweep_module", ROOT / "scripts/28_simultaneous_method_sweep.py"
    )
    sweep = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = sweep
    spec.loader.exec_module(sweep)

    # Base frozen simultaneous candidate
    frozen = next(v for v in sweep.V2_CONFIRMATORY if v.name == "v2_simul_s04_ig010")

    # Define gfs_twostep_l100 variant
    twostep_variant = replace(
        frozen,
        name=ARM_NAME,
        algorithm="twostep_ensrf",
        gauge_component_spread_cells=None,
        ensrf_localization_km=100.0,
        note="IMERG-guided flow then gauge EnSRF at 100 km",
    )

    # Register targeted group
    group_name = "gfs_twostep_l100_targeted"
    sweep.GROUPS[group_name] = [twostep_variant]

    # Hook sample_at_stations to intercept full 4D members (T, M, H, W)
    captured_fields: dict[str, np.ndarray] = {}
    original_sample_at_stations = sweep.sample_at_stations

    expected_order = ["background", ARM_NAME]
    capture_index = 0

    def hooked_sample_at_stations(members, grid, lat, lon):
        nonlocal capture_index
        if capture_index < len(expected_order):
            variant_name = expected_order[capture_index]
            captured_fields[variant_name] = members.copy()
            capture_index += 1
        return original_sample_at_stations(members, grid, lat, lon)

    sweep.sample_at_stations = hooked_sample_at_stations

    dump_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    previous_argv = sys.argv
    sys.argv = [
        str(ROOT / "scripts/28_simultaneous_method_sweep.py"),
        "--config", "configs/da.yaml",
        "--ckpt", args.graphflow_ckpt,
        "--stations", args.stations,
        "--imerg", args.imerg,
        "--start", args.start,
        "--end", args.end,
        "--members", str(args.members),
        "--seed", "201805",
        "--background-day-offset", "-1",
        "--imerg-stride", "1",
        "--set", "observations.imerg.factor=8",
        "--set", "observations.imerg.error_corr_cells=0.75",
        "--holdout-folds", "5",
        "--holdout-fold", str(args.fold),
        "--group", group_name,
        "--out", str(dump_path),
        "--report", str(report_path),
    ]

    try:
        sweep.main()
    finally:
        sys.argv = previous_argv

    # Enrich the written NPZ with spatial standard deviation and full members
    if captured_fields:
        print("[gfs_twostep_l100] enriching dump with spatial standard deviations ...")
        existing_data = dict(np.load(dump_path, allow_pickle=False))

        for name, field in captured_fields.items():
            existing_data[f"stdfield_{name}"] = np.nanstd(field, axis=1).astype(np.float32)
            if args.save_members:
                existing_data[f"fields_{name}"] = field.astype(np.float32)

        np.savez_compressed(dump_path, **existing_data)
        print(f"[gfs_twostep_l100] updated {dump_path} with spatial spread fields")


# ---------------------------------------------------------------------------
# Cartographic Spatial Plotting
# ---------------------------------------------------------------------------


def add_gauge_markers(
    ax: plt.Axes,
    lons: np.ndarray,
    lats: np.ndarray,
    values: np.ndarray,
    assim_idx: np.ndarray,
    eval_idx: np.ndarray,
    vmin: float,
    vmax: float,
    cmap: str = "viridis",
) -> None:
    """Plot assimilated and withheld gauges with distinct edges and values."""
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Assimilated gauges: circles with dark border
    if len(assim_idx) > 0:
        val_assim = values[assim_idx]
        fin_a = np.isfinite(val_assim)
        ax.scatter(
            lons[assim_idx][fin_a],
            lats[assim_idx][fin_a],
            c=val_assim[fin_a],
            cmap=cmap,
            norm=norm,
            s=32,
            marker="o",
            edgecolors="black",
            linewidths=0.7,
            zorder=5,
            label="Assimilated gauge",
        )

    # Withheld evaluation gauges: larger with bright cyan border
    if len(eval_idx) > 0:
        val_eval = values[eval_idx]
        fin_e = np.isfinite(val_eval)
        ax.scatter(
            lons[eval_idx][fin_e],
            lats[eval_idx][fin_e],
            c=val_eval[fin_e],
            cmap=cmap,
            norm=norm,
            s=52,
            marker="o",
            edgecolors="#00E5FF",
            linewidths=1.5,
            zorder=6,
            label="Withheld gauge (eval)",
        )


def plot_single_spatial_diagnostic(
    date_label: str,
    bg_mean: np.ndarray,
    an_mean: np.ndarray,
    imerg: np.ndarray | None,
    bg_std: np.ndarray | None,
    an_std: np.ndarray | None,
    gauge_vals: np.ndarray,
    bg_station_mean: np.ndarray | None,
    an_station_mean: np.ndarray | None,
    extent: list[float],
    valid_mask: np.ndarray,
    station_lon: np.ndarray,
    station_lat: np.ndarray,
    assim_idx: np.ndarray,
    eval_idx: np.ndarray,
    out_path: Path,
) -> None:
    """Render a 2x3 cartographic spatial diagnostic grid for a single day or period mean."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), constrained_layout=True)

    # Color scale limits
    fields_to_check = [f for f in (bg_mean, an_mean, imerg) if f is not None]
    rain_max = max(
        15.0,
        float(np.nanpercentile(np.concatenate([f[valid_mask] for f in fields_to_check]), 99))
        if fields_to_check
        else 25.0,
    )
    inc = an_mean - bg_mean
    inc_max = max(3.0, float(np.nanpercentile(np.abs(inc[valid_mask]), 98.5)))

    masked_bg = np.where(valid_mask, bg_mean, np.nan)
    masked_an = np.where(valid_mask, an_mean, np.nan)
    masked_inc = np.where(valid_mask, inc, np.nan)

    # ------------------------------------------------------------------
    # Panel 0,0: IMERG Satellite Field
    # ------------------------------------------------------------------
    ax00 = axes[0, 0]
    if imerg is not None and np.any(np.isfinite(imerg)):
        masked_imerg = np.where(valid_mask, imerg, np.nan)
        im00 = ax00.imshow(
            masked_imerg,
            origin="lower",
            extent=extent,
            cmap="viridis",
            vmin=0,
            vmax=rain_max,
        )
        cbar00 = plt.colorbar(im00, ax=ax00, orientation="horizontal", shrink=0.75, pad=0.04)
        cbar00.set_label("Precipitation (mm/day)", fontsize=9)
        imerg_mean = np.nanmean(masked_imerg)
        ax00.set_title(f"A. IMERG Satellite (0.05° Ingest)\nDomain mean: {imerg_mean:.1f} mm/day", fontsize=10, weight="bold")
    else:
        ax00.text(0.5, 0.5, "IMERG not available", ha="center", va="center", transform=ax00.transAxes)
        ax00.set_title("A. IMERG Satellite (Not available)", fontsize=10, weight="bold")

    # ------------------------------------------------------------------
    # Panel 0,1: Background (Prior Mean) + Observed Gauges
    # ------------------------------------------------------------------
    ax01 = axes[0, 1]
    im01 = ax01.imshow(
        masked_bg,
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=0,
        vmax=rain_max,
    )
    add_gauge_markers(ax01, station_lon, station_lat, gauge_vals, assim_idx, eval_idx, vmin=0, vmax=rain_max)
    cbar01 = plt.colorbar(im01, ax=ax01, orientation="horizontal", shrink=0.75, pad=0.04)
    cbar01.set_label("Precipitation (mm/day)", fontsize=9)
    bg_domain_mean = np.nanmean(masked_bg)
    ax01.set_title(f"B. GraphFlow Background Prior Mean\nDomain mean: {bg_domain_mean:.1f} mm/day (Gauges overlaid)", fontsize=10, weight="bold")

    # ------------------------------------------------------------------
    # Panel 0,2: gfs_twostep_l100 Analysis Mean
    # ------------------------------------------------------------------
    ax02 = axes[0, 2]
    im02 = ax02.imshow(
        masked_an,
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=0,
        vmax=rain_max,
    )
    add_gauge_markers(ax02, station_lon, station_lat, gauge_vals, assim_idx, eval_idx, vmin=0, vmax=rain_max)
    cbar02 = plt.colorbar(im02, ax=ax02, orientation="horizontal", shrink=0.75, pad=0.04)
    cbar02.set_label("Precipitation (mm/day)", fontsize=9)
    an_domain_mean = np.nanmean(masked_an)
    ax02.set_title(f"C. {ARM_NAME} Analysis Mean\nDomain mean: {an_domain_mean:.1f} mm/day", fontsize=10, weight="bold")
    ax02.legend(loc="lower right", fontsize=8, framealpha=0.85)

    # ------------------------------------------------------------------
    # Panel 1,0: DA Analysis Increment (Analysis - Background)
    # ------------------------------------------------------------------
    ax10 = axes[1, 0]
    im10 = ax10.imshow(
        masked_inc,
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        vmin=-inc_max,
        vmax=inc_max,
    )
    # Highlight station positions on increment
    if len(assim_idx) > 0:
        ax10.scatter(station_lon[assim_idx], station_lat[assim_idx], c="black", s=18, marker="x", alpha=0.7, label="Assimilated station")
    if len(eval_idx) > 0:
        ax10.scatter(station_lon[eval_idx], station_lat[eval_idx], facecolors="none", edgecolors="#00E5FF", s=35, linewidths=1.2, label="Withheld station")
    cbar10 = plt.colorbar(im10, ax=ax10, orientation="horizontal", shrink=0.75, pad=0.04)
    cbar10.set_label("Increment (Analysis − Background, mm/day)", fontsize=9)
    mean_abs_inc = np.nanmean(np.abs(masked_inc))
    ax10.set_title(f"D. DA Analysis Increment (A − B)\nMean |increment|: {mean_abs_inc:.2f} mm/day", fontsize=10, weight="bold")
    ax10.legend(loc="lower right", fontsize=8, framealpha=0.85)

    # ------------------------------------------------------------------
    # Panel 1,1: Ensemble Spread or Locality
    # ------------------------------------------------------------------
    ax11 = axes[1, 1]
    if an_std is not None and np.any(np.isfinite(an_std)):
        masked_std = np.where(valid_mask, an_std, np.nan)
        std_max = max(5.0, float(np.nanpercentile(masked_std[valid_mask], 98)))
        im11 = ax11.imshow(
            masked_std,
            origin="lower",
            extent=extent,
            cmap="magma",
            vmin=0,
            vmax=std_max,
        )
        cbar11 = plt.colorbar(im11, ax=ax11, orientation="horizontal", shrink=0.75, pad=0.04)
        cbar11.set_label("Posterior Spread σ (mm/day)", fontsize=9)
        mean_spread = np.nanmean(masked_std)
        ax11.set_title(f"E. Analysis Ensemble Spread (σ)\nDomain mean spread: {mean_spread:.2f} mm/day", fontsize=10, weight="bold")
    else:
        # Fallback to absolute increment amplitude highlighting Gaspari-Cohn radius
        abs_inc = np.where(valid_mask, np.abs(inc), np.nan)
        im11 = ax11.imshow(
            abs_inc,
            origin="lower",
            extent=extent,
            cmap="inferno",
            vmin=0,
            vmax=inc_max,
        )
        if len(assim_idx) > 0:
            ax11.scatter(station_lon[assim_idx], station_lat[assim_idx], c="cyan", s=20, marker="o", edgecolors="black", linewidths=0.5)
        cbar11 = plt.colorbar(im11, ax=ax11, orientation="horizontal", shrink=0.75, pad=0.04)
        cbar11.set_label("|Increment| (mm/day)", fontsize=9)
        ax11.set_title("E. Increment Locality |A − B|\nHighlighting 100 km EnSRF radius & flow", fontsize=10, weight="bold")

    # ------------------------------------------------------------------
    # Panel 1,2: Station Innovation Scatter / Verification
    # ------------------------------------------------------------------
    ax12 = axes[1, 2]
    if bg_station_mean is not None and an_station_mean is not None:
        g_obs = gauge_vals
        fin_all = np.isfinite(g_obs) & np.isfinite(bg_station_mean) & np.isfinite(an_station_mean)

        max_val = max(
            20.0,
            float(np.nanmax(g_obs[fin_all])) if fin_all.any() else 20.0,
            float(np.nanmax(bg_station_mean[fin_all])) if fin_all.any() else 20.0,
            float(np.nanmax(an_station_mean[fin_all])) if fin_all.any() else 20.0,
        )
        ax12.plot([0, max_val], [0, max_val], "k--", lw=1.0, alpha=0.6, label="1:1 line")

        # Assimilated stations
        a_mask = np.isin(np.arange(len(g_obs)), assim_idx) & fin_all
        if a_mask.any():
            ax12.scatter(
                g_obs[a_mask],
                bg_station_mean[a_mask],
                color="#5FA8D3",
                alpha=0.65,
                s=30,
                label=f"Assimilated Prior (RMSE {compute_metrics(bg_station_mean[a_mask], g_obs[a_mask])['rmse']:.1f})",
            )
            ax12.scatter(
                g_obs[a_mask],
                an_station_mean[a_mask],
                color="#1B4965",
                alpha=0.9,
                s=40,
                marker="^",
                label=f"Assimilated Analysis (RMSE {compute_metrics(an_station_mean[a_mask], g_obs[a_mask])['rmse']:.1f})",
            )

        # Withheld evaluation stations
        e_mask = np.isin(np.arange(len(g_obs)), eval_idx) & fin_all
        if e_mask.any():
            ax12.scatter(
                g_obs[e_mask],
                bg_station_mean[e_mask],
                color="#F4A261",
                alpha=0.7,
                s=45,
                edgecolors="black",
                linewidths=0.5,
                label=f"Withheld Prior (RMSE {compute_metrics(bg_station_mean[e_mask], g_obs[e_mask])['rmse']:.1f})",
            )
            ax12.scatter(
                g_obs[e_mask],
                an_station_mean[e_mask],
                color="#E76F51",
                alpha=0.95,
                s=55,
                marker="s",
                edgecolors="black",
                linewidths=0.6,
                label=f"Withheld Analysis (RMSE {compute_metrics(an_station_mean[e_mask], g_obs[e_mask])['rmse']:.1f})",
            )

        ax12.set_xlabel("Observed Station Precipitation (mm/day)", fontsize=9)
        ax12.set_ylabel("Predicted Precipitation (mm/day)", fontsize=9)
        ax12.set_xlim(0, max_val * 1.05)
        ax12.set_ylim(0, max_val * 1.05)
        ax12.grid(True, linestyle=":", alpha=0.5)
        ax12.legend(loc="upper left", fontsize=7.5, framealpha=0.85)
        ax12.set_title("F. Station Verification (Fit vs Held-out)", fontsize=10, weight="bold")
    else:
        ax12.text(0.5, 0.5, "Station verification data not in dump", ha="center", va="center", transform=ax12.transAxes)
        ax12.set_title("F. Station Verification", fontsize=10, weight="bold")

    # Common map styling for panels (0,0), (0,1), (0,2), (1,0), (1,1)
    for r in range(2):
        for c in range(3):
            if (r, c) == (1, 2):
                continue
            ax = axes[r, c]
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.set_xlabel("Longitude (°E)", fontsize=8)
            ax.set_ylabel("Latitude (°N)", fontsize=8)
            ax.grid(True, linestyle=":", alpha=0.3, color="gray")

    fig.suptitle(
        f"GraphFlow G0 2D Spatial Diagnostics: {ARM_NAME} ({date_label})\n"
        "Two-Step DA: IMERG Flow Guidance + Localized Gauge EnSRF (100 km radius)",
        fontsize=13,
        weight="bold",
        y=1.01,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"[plot] wrote {out_path}", flush=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Diagnostics & Summary Driver
# ---------------------------------------------------------------------------


def generate_spatial_diagnostics(dump_path: Path, out_dir: Path) -> None:
    """Read .npz dump and generate period-mean and daily spatial diagnostic plots."""
    if not dump_path.exists():
        raise FileNotFoundError(f"Dump file does not exist: {dump_path}")

    print(f"[gfs_twostep_l100] reading dump: {dump_path}")
    data = np.load(dump_path, allow_pickle=False)

    times = [str(t) for t in data["times"]]
    n_days = len(times)
    grid_lat = np.asarray(data["grid_lat"], dtype=float)
    grid_lon = np.asarray(data["grid_lon"], dtype=float)
    extent = [float(grid_lon.min()), float(grid_lon.max()), float(grid_lat.min()), float(grid_lat.max())]

    valid = np.asarray(data["valid"], dtype=bool)
    station_lat = np.asarray(data["station_lat"], dtype=float)
    station_lon = np.asarray(data["station_lon"], dtype=float)
    gauge_mm = np.asarray(data["gauge_mm"], dtype=float)
    eval_idx = np.asarray(data["eval_idx"], dtype=int)
    assim_idx = np.asarray(data["assim_idx"], dtype=int)

    # Check for mean fields
    bg_key = "meanfield_background"
    an_key = f"meanfield_{ARM_NAME}"
    if bg_key not in data or an_key not in data:
        available_meanfields = [k for k in data.files if k.startswith("meanfield_")]
        raise KeyError(
            f"Dump lacks {bg_key} or {an_key}. Available mean fields: {available_meanfields}"
        )

    bg_mean = np.asarray(data[bg_key], dtype=float)
    an_mean = np.asarray(data[an_key], dtype=float)
    imerg = np.asarray(data["raw_imerg_mm"], dtype=float) if "raw_imerg_mm" in data else None

    # Check for spread fields
    bg_std = np.asarray(data["stdfield_background"], dtype=float) if "stdfield_background" in data else None
    an_std = np.asarray(data[f"stdfield_{ARM_NAME}"], dtype=float) if f"stdfield_{ARM_NAME}" in data else None

    # Station extractions
    bg_station = np.asarray(data["station_background"], dtype=float) if "station_background" in data else None
    an_station = np.asarray(data[f"station_{ARM_NAME}"], dtype=float) if f"station_{ARM_NAME}" in data else None
    bg_stn_mean = np.nanmean(bg_station, axis=1) if bg_station is not None else None
    an_stn_mean = np.nanmean(an_station, axis=1) if an_station is not None else None

    plots_dir = out_dir / "spatial_maps"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Multi-day Period Mean Plot
    print("[gfs_twostep_l100] plotting 10-day period mean spatial diagnostics ...")
    period_label = f"Period Mean: {times[0]} to {times[-1]} ({n_days} days)"
    plot_single_spatial_diagnostic(
        date_label=period_label,
        bg_mean=np.nanmean(bg_mean, axis=0),
        an_mean=np.nanmean(an_mean, axis=0),
        imerg=np.nanmean(imerg, axis=0) if imerg is not None else None,
        bg_std=np.nanmean(bg_std, axis=0) if bg_std is not None else None,
        an_std=np.nanmean(an_std, axis=0) if an_std is not None else None,
        gauge_vals=np.nanmean(gauge_mm, axis=0),
        bg_station_mean=np.nanmean(bg_stn_mean, axis=0) if bg_stn_mean is not None else None,
        an_station_mean=np.nanmean(an_stn_mean, axis=0) if an_stn_mean is not None else None,
        extent=extent,
        valid_mask=valid,
        station_lon=station_lon,
        station_lat=station_lat,
        assim_idx=assim_idx,
        eval_idx=eval_idx,
        out_path=out_dir / "spatial_summary_period_mean.png",
    )

    # Also save as PDF for publication quality
    plot_single_spatial_diagnostic(
        date_label=period_label,
        bg_mean=np.nanmean(bg_mean, axis=0),
        an_mean=np.nanmean(an_mean, axis=0),
        imerg=np.nanmean(imerg, axis=0) if imerg is not None else None,
        bg_std=np.nanmean(bg_std, axis=0) if bg_std is not None else None,
        an_std=np.nanmean(an_std, axis=0) if an_std is not None else None,
        gauge_vals=np.nanmean(gauge_mm, axis=0),
        bg_station_mean=np.nanmean(bg_stn_mean, axis=0) if bg_stn_mean is not None else None,
        an_station_mean=np.nanmean(an_stn_mean, axis=0) if an_stn_mean is not None else None,
        extent=extent,
        valid_mask=valid,
        station_lon=station_lon,
        station_lat=station_lat,
        assim_idx=assim_idx,
        eval_idx=eval_idx,
        out_path=out_dir / "spatial_summary_period_mean.pdf",
    )

    # 2. Daily Spatial Diagnostic Plots
    print(f"[gfs_twostep_l100] plotting daily spatial diagnostic maps for {n_days} days ...")
    for d, day_str in enumerate(times):
        plot_single_spatial_diagnostic(
            date_label=f"Day {day_str}",
            bg_mean=bg_mean[d],
            an_mean=an_mean[d],
            imerg=imerg[d] if imerg is not None else None,
            bg_std=bg_std[d] if bg_std is not None else None,
            an_std=an_std[d] if an_std is not None else None,
            gauge_vals=gauge_mm[d],
            bg_station_mean=bg_stn_mean[d] if bg_stn_mean is not None else None,
            an_station_mean=an_stn_mean[d] if an_stn_mean is not None else None,
            extent=extent,
            valid_mask=valid,
            station_lon=station_lon,
            station_lat=station_lat,
            assim_idx=assim_idx,
            eval_idx=eval_idx,
            out_path=plots_dir / f"spatial_day_{day_str}.png",
        )

    # 3. Numeric Summary Report
    summary_report = {
        "arm": ARM_NAME,
        "period": f"{times[0]} to {times[-1]}",
        "n_days": n_days,
        "n_stations_assimilated": len(assim_idx),
        "n_stations_withheld": len(eval_idx),
        "domain_mean_rain": {
            "background_mm": float(np.nanmean(bg_mean[:, valid])),
            "analysis_mm": float(np.nanmean(an_mean[:, valid])),
            "increment_abs_mean_mm": float(np.nanmean(np.abs(an_mean - bg_mean)[:, valid])),
            "imerg_mm": float(np.nanmean(imerg[:, valid])) if imerg is not None else None,
        },
    }

    if bg_station is not None and an_station is not None:
        eval_obs = gauge_mm[:, eval_idx].reshape(-1)
        bg_eval_pred = bg_stn_mean[:, eval_idx].reshape(-1)
        an_eval_pred = an_stn_mean[:, eval_idx].reshape(-1)

        eval_bg_crps = float(np.nanmean(fair_crps_1d(bg_station[:, :, eval_idx].reshape(-1, bg_station.shape[1]), eval_obs)))
        eval_an_crps = float(np.nanmean(fair_crps_1d(an_station[:, :, eval_idx].reshape(-1, an_station.shape[1]), eval_obs)))

        summary_report["withheld_station_metrics"] = {
            "background": {**compute_metrics(bg_eval_pred, eval_obs), "crps": eval_bg_crps},
            "analysis": {**compute_metrics(an_eval_pred, eval_obs), "crps": eval_an_crps},
            "crps_gain_mm": eval_bg_crps - eval_an_crps,
            "crps_gain_percent": ((eval_bg_crps - eval_an_crps) / eval_bg_crps * 100.0) if eval_bg_crps > 0 else 0.0,
        }

        assim_obs = gauge_mm[:, assim_idx].reshape(-1)
        bg_assim_pred = bg_stn_mean[:, assim_idx].reshape(-1)
        an_assim_pred = an_stn_mean[:, assim_idx].reshape(-1)
        summary_report["assimilated_station_metrics"] = {
            "background": compute_metrics(bg_assim_pred, assim_obs),
            "analysis": compute_metrics(an_assim_pred, assim_obs),
        }

    # Save JSON summary
    json_path = out_dir / f"{ARM_NAME}_summary.json"
    json_path.write_text(json.dumps(summary_report, indent=2))
    print(f"[gfs_twostep_l100] wrote JSON summary: {json_path}")

    # Write Markdown summary
    md_lines = [
        f"# Spatial Diagnostic Summary: `{ARM_NAME}`",
        "",
        f"- **Period**: {summary_report['period']} ({n_days} days)",
        f"- **Method**: Decoupled Two-Step Assimilation (IMERG Generative Flow + 100 km Gauge EnSRF)",
        f"- **Stations**: {len(assim_idx)} assimilated, {len(eval_idx)} withheld evaluation",
        "",
        "## Domain Spatial Precipitation",
        "",
        f"| Metric | Background Prior | {ARM_NAME} Analysis | Difference / IMERG |",
        "|:--|--:|--:|--:|",
        f"| Domain Mean (mm/day) | {summary_report['domain_mean_rain']['background_mm']:.2f} | {summary_report['domain_mean_rain']['analysis_mm']:.2f} | {summary_report['domain_mean_rain']['analysis_mm'] - summary_report['domain_mean_rain']['background_mm']:+.2f} |",
        f"| Mean Absolute Increment | -- | {summary_report['domain_mean_rain']['increment_abs_mean_mm']:.2f} mm/day | -- |",
    ]

    if summary_report["domain_mean_rain"]["imerg_mm"] is not None:
        md_lines.append(f"| IMERG Satellite Mean | -- | -- | {summary_report['domain_mean_rain']['imerg_mm']:.2f} mm/day |")

    if "withheld_station_metrics" in summary_report:
        eval_m = summary_report["withheld_station_metrics"]
        md_lines += [
            "",
            "## Withheld Gauge Verification (Independent Stations)",
            "",
            "| Metric | Background Prior | Analysis (`gfs_twostep_l100`) | Improvement |",
            "|:--|--:|--:|--:|",
            f"| CRPS (mm/day) | {eval_m['background']['crps']:.3f} | {eval_m['analysis']['crps']:.3f} | {eval_m['crps_gain_mm']:+.3f} ({eval_m['crps_gain_percent']:+.1f}%) |",
            f"| RMSE (mm/day) | {eval_m['background']['rmse']:.3f} | {eval_m['analysis']['rmse']:.3f} | {eval_m['background']['rmse'] - eval_m['analysis']['rmse']:+.3f} |",
            f"| MAE (mm/day) | {eval_m['background']['mae']:.3f} | {eval_m['analysis']['mae']:.3f} | {eval_m['background']['mae'] - eval_m['analysis']['mae']:+.3f} |",
            f"| Correlation | {eval_m['background']['corr']:.3f} | {eval_m['analysis']['corr']:.3f} | {eval_m['analysis']['corr'] - eval_m['background']['corr']:+.3f} |",
        ]

    # 4. Comparative plot against baseline gfs_joint_base if available
    base_key = "meanfield_gfs_joint_base"
    if base_key in data:
        print("[gfs_twostep_l100] found gfs_joint_base in dump; generating comparative diagnostics ...")
        base_mean = np.asarray(data[base_key], dtype=float)
        base_period_mean = np.nanmean(base_mean, axis=0)
        twostep_period_mean = np.nanmean(an_mean, axis=0)
        diff_field = twostep_period_mean - base_period_mean

        fig_comp, ax_comp = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
        max_rain = max(15.0, float(np.nanpercentile(np.concatenate([base_period_mean[valid], twostep_period_mean[valid]]), 99)))
        diff_max = max(2.0, float(np.nanpercentile(np.abs(diff_field[valid]), 98.5)))

        im_a = ax_comp[0].imshow(np.where(valid, base_period_mean, np.nan), origin="lower", extent=extent, cmap="viridis", vmin=0, vmax=max_rain)
        plt.colorbar(im_a, ax=ax_comp[0], orientation="horizontal", shrink=0.75, pad=0.04).set_label("Precipitation (mm/day)", fontsize=9)
        ax_comp[0].set_title("A. Baseline: gfs_joint_base\n(Joint flow guidance)", fontsize=10, weight="bold")

        im_b = ax_comp[1].imshow(np.where(valid, twostep_period_mean, np.nan), origin="lower", extent=extent, cmap="viridis", vmin=0, vmax=max_rain)
        add_gauge_markers(ax_comp[1], station_lon, station_lat, np.nanmean(gauge_mm, axis=0), assim_idx, eval_idx, vmin=0, vmax=max_rain)
        plt.colorbar(im_b, ax=ax_comp[1], orientation="horizontal", shrink=0.75, pad=0.04).set_label("Precipitation (mm/day)", fontsize=9)
        ax_comp[1].set_title(f"B. Candidate: {ARM_NAME}\n(IMERG flow + 100 km gauge EnSRF)", fontsize=10, weight="bold")

        im_c = ax_comp[2].imshow(np.where(valid, diff_field, np.nan), origin="lower", extent=extent, cmap="RdBu_r", vmin=-diff_max, vmax=diff_max)
        plt.colorbar(im_c, ax=ax_comp[2], orientation="horizontal", shrink=0.75, pad=0.04).set_label("Difference (Twostep − Joint, mm/day)", fontsize=9)
        ax_comp[2].set_title("C. Difference: Twostep − Joint Base\nRed = Twostep wetter, Blue = Joint wetter", fontsize=10, weight="bold")

        for ax in ax_comp:
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.set_xlabel("Longitude (°E)", fontsize=8)
            ax.set_ylabel("Latitude (°N)", fontsize=8)
            ax.grid(True, linestyle=":", alpha=0.3, color="gray")

        fig_comp.suptitle("Comparative Analysis: Two-Step EnSRF vs Joint Guidance Baseline (Period Mean)", fontsize=12, weight="bold", y=1.02)
        comp_out = out_dir / "spatial_comparison_vs_joint_base.png"
        fig_comp.savefig(comp_out, dpi=200, bbox_inches="tight")
        fig_comp.savefig(out_dir / "spatial_comparison_vs_joint_base.pdf", bbox_inches="tight")
        plt.close(fig_comp)
        print(f"[plot] wrote {comp_out}")

        # Add to markdown
        md_lines += [
            "",
            "## Comparison against Frozen Baseline (`gfs_joint_base`)",
            f"- Spatial Comparison Figure: `spatial_comparison_vs_joint_base.png` / `.pdf`",
            f"- Mean Absolute Difference (|Twostep − Joint|): {float(np.nanmean(np.abs(diff_field[valid]))):.2f} mm/day",
        ]

    md_lines += [
        "",
        "## Generated Plots",
        "- Period Mean Overview: `spatial_summary_period_mean.png` / `.pdf`",
        f"- Daily Evolution Maps: `spatial_maps/spatial_day_*.png` ({n_days} daily figures)",
    ]

    md_path = out_dir / f"{ARM_NAME}_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"[gfs_twostep_l100] wrote Markdown summary: {md_path}")


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dump_path = Path(args.dump) if args.dump else out_dir / f"{ARM_NAME}_fold{args.fold}.npz"
    report_path = Path(args.report) if args.report else out_dir / f"{ARM_NAME}_fold{args.fold}.json"

    should_run_da = args.run_da or (not args.plot_only and not dump_path.exists())

    if should_run_da:
        print(f"[gfs_twostep_l100] running targeted DA pipeline -> {dump_path}")
        run_targeted_da(args, dump_path=dump_path, report_path=report_path)
    else:
        print(f"[gfs_twostep_l100] skipping DA run; using existing dump: {dump_path}")

    # Generate spatial diagnostic maps
    generate_spatial_diagnostics(dump_path=dump_path, out_dir=out_dir)
    print(f"[gfs_twostep_l100] all spatial diagnostics successfully created under {out_dir}")


if __name__ == "__main__":
    main()
