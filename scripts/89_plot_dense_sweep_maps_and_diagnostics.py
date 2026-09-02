#!/usr/bin/env python3
"""Plot dense gauge DA sweep comparisons and diagnostics against IMERG and CHIRPS.

Generates comprehensive high-resolution figures:
1. Daily spatial maps comparing Gauges, CHIRPS, IMERG, Background, Analysis, and Increments.
2. 5-day event accumulation maps and intercomparison difference fields.
3. Station-level verification diagnostics at withheld and assimilated gauges.
4. Spatial texture, intensity CDF distributions, and inter-product correlations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--npz",
        required=True,
        help="Path to sweep profile NPZ (e.g. data/processed/v2_dense_gauge_sweep/profiles/repr_measured.npz)",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Optional path to profile JSON report (defaults to <npz_prefix>.json)",
    )
    parser.add_argument(
        "--imerg",
        default=None,
        help="Optional path to IMERG NetCDF file (fallback if not in NPZ)",
    )
    parser.add_argument(
        "--chirps",
        default=None,
        help="Optional path to CHIRPS NetCDF file (fallback if not in NPZ)",
    )
    parser.add_argument(
        "--station-summary",
        default=None,
        help="Optional path to station_summary.csv for BMD vs BWDB network tagging",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for plots (defaults to <npz_dir>/../plots)",
    )
    parser.add_argument(
        "--reference-arm",
        default="prod_huber3",
        help="Control arm name (default: prod_huber3)",
    )
    parser.add_argument(
        "--best-arm",
        default=None,
        help="Arm to highlight as leading candidate (default: lowest withheld CRPS)",
    )
    return parser.parse_args()


def block_smooth(field: np.ndarray, factor: int = 4) -> np.ndarray:
    """Coarsen by factor then repeat back to original resolution."""
    height, width = field.shape[-2:]
    if height % factor or width % factor:
        return field
    coarse = field.reshape(
        *field.shape[:-2], height // factor, factor, width // factor, factor
    ).mean(axis=(-3, -1))
    return np.repeat(np.repeat(coarse, factor, axis=-2), factor, axis=-1)


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
    corr = float(np.corrcoef(p, t)[0, 1]) if len(p) > 1 and p.std() > 0 and t.std() > 0 else np.nan
    return {"mae": mae, "rmse": rmse, "bias": bias, "corr": corr}


def extract_eval_ensemble_flat(ens_arr: np.ndarray, eval_idx: np.ndarray) -> np.ndarray:
    """Flatten (n_days, n_members, n_stations) -> (n_days * n_eval, n_members) aligned with gauge_mm[:, eval_idx].reshape(-1)."""
    eval_ens = ens_arr[:, :, eval_idx]
    eval_trans = np.swapaxes(eval_ens, 1, 2)
    return eval_trans.reshape(-1, eval_ens.shape[1])


def add_gauge_markers(
    ax, lons, lats, values, assim_idx, eval_idx, vmin, vmax, cmap="viridis"
):
    """Plot assimilated and withheld gauges with distinct edges."""
    norm = Normalize(vmin=vmin, vmax=vmax)
    # Assimilated gauges: small with dark edge
    if len(assim_idx) > 0:
        val_assim = values[assim_idx]
        fin_a = np.isfinite(val_assim)
        ax.scatter(
            lons[assim_idx][fin_a],
            lats[assim_idx][fin_a],
            c=val_assim[fin_a],
            cmap=cmap,
            norm=norm,
            s=22,
            marker="o",
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
            label="Assimilated gauge" if ax.get_subplotspec().colspan.start == 0 else None,
        )
    # Withheld gauges: larger with bright cyan edge
    if len(eval_idx) > 0:
        val_eval = values[eval_idx]
        fin_e = np.isfinite(val_eval)
        ax.scatter(
            lons[eval_idx][fin_e],
            lats[eval_idx][fin_e],
            c=val_eval[fin_e],
            cmap=cmap,
            norm=norm,
            s=45,
            marker="o",
            edgecolors="#00E5FF",
            linewidths=1.2,
            zorder=6,
            label="Withheld gauge (eval)" if ax.get_subplotspec().colspan.start == 0 else None,
        )


def main():
    args = parse_args()
    npz_path = Path(args.npz)
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    out_dir = Path(args.out_dir) if args.out_dir else npz_path.parent.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = (
        Path(args.report) if args.report else npz_path.with_suffix(".json")
    )
    report = json.loads(report_path.read_text()) if report_path.exists() else {}

    print(f"[plot] loading {npz_path} ...", flush=True)
    with np.load(npz_path, allow_pickle=True) as dump:
        times = dump["times"].astype("datetime64[D]").astype(str)
        n_days = len(times)
        grid_lat = np.asarray(dump["grid_lat"], float)
        grid_lon = np.asarray(dump["grid_lon"], float)
        valid_mask = np.asarray(dump["valid"], bool)

        station_ids = np.asarray(dump["station_ids"], str)
        station_lat = np.asarray(dump["station_lat"], float)
        station_lon = np.asarray(dump["station_lon"], float)
        gauge_mm = np.asarray(dump["gauge_mm"], float)  # (n_days, n_stations)
        eval_idx = np.asarray(dump["eval_idx"], int)
        assim_idx = np.asarray(dump["assim_idx"], int)
        distance_km = np.asarray(dump["distance_km"], float) if "distance_km" in dump else None

        variant_names = dump["variant_names"].astype(str).tolist()
        meanfields = {
            v: np.asarray(dump[f"meanfield_{v}"], float)
            for v in variant_names
            if f"meanfield_{v}" in dump
        }
        station_ensembles = {
            v: np.asarray(dump[f"station_{v}"], float)
            for v in variant_names
            if f"station_{v}" in dump
        }

        # CHIRPS
        chirps = np.asarray(dump["chirps"], float) if "chirps" in dump else None
        # IMERG
        raw_imerg = np.asarray(dump["raw_imerg_mm"], float) if "raw_imerg_mm" in dump else None

    # Domain extent
    d_lat = abs(grid_lat[1] - grid_lat[0]) / 2.0 if len(grid_lat) > 1 else 0.025
    d_lon = abs(grid_lon[1] - grid_lon[0]) / 2.0 if len(grid_lon) > 1 else 0.025
    extent = [
        float(grid_lon[0] - d_lon),
        float(grid_lon[-1] + d_lon),
        float(grid_lat[0] - d_lat),
        float(grid_lat[-1] + d_lat),
    ]

    # Handle IMERG spatial dimensions: regrid/tile if on coarser footprints
    if raw_imerg is not None and raw_imerg.ndim == 3:
        if raw_imerg.shape[1:] != (len(grid_lat), len(grid_lon)):
            scale_lat = max(1, -(-len(grid_lat) // raw_imerg.shape[1]))
            scale_lon = max(1, -(-len(grid_lon) // raw_imerg.shape[2]))
            imerg_regrid = np.repeat(np.repeat(raw_imerg, scale_lat, axis=1), scale_lon, axis=2)
            imerg_field = imerg_regrid[:, : len(grid_lat), : len(grid_lon)]
        else:
            imerg_field = raw_imerg
    else:
        imerg_field = None

    # Identify Reference and Best Arms
    ref_arm = args.reference_arm if args.reference_arm in meanfields else variant_names[0]
    best_arm = args.best_arm
    if not best_arm or best_arm not in meanfields:
        # Determine best arm by withheld CRPS from report if present
        best_candidate = None
        best_crps = float("inf")
        if "variants" in report:
            for name, meta in report["variants"].items():
                if name in meanfields and name != "background":
                    crps_val = meta.get("eval_crps")
                    if crps_val is not None and crps_val < best_crps:
                        best_crps = crps_val
                        best_candidate = name
        best_arm = best_candidate or ("dense_s3_bwdb_r4" if "dense_s3_bwdb_r4" in meanfields else ref_arm)

    print(f"[plot] reference arm: {ref_arm} | highlighted best arm: {best_arm}")
    print(f"[plot] available variants: {list(meanfields.keys())}")

    # =========================================================================
    # FIGURE 1: Daily Spatial Evolution Maps (5 Days x 6 Panels)
    # =========================================================================
    print("[plot] generating Figure 1: Daily spatial evolution maps ...", flush=True)
    fig1, axes1 = plt.subplots(n_days, 6, figsize=(22, 3.4 * n_days), constrained_layout=True)
    if n_days == 1:
        axes1 = np.expand_dims(axes1, axis=0)

    col_titles = [
        "A. Gauges (BMD + BWDB)",
        "B. CHIRPS (0.05°)",
        "C. IMERG (Satellite)",
        "D. Background Prior",
        f"E. Analysis: {best_arm}",
        "F. DA Increment (|A - B|)",
    ]

    for d in range(n_days):
        date_str = times[d]
        bg = meanfields.get("background", np.zeros_like(valid_mask, dtype=float))[d]
        an = meanfields[best_arm][d]
        inc = an - bg
        ch = chirps[d] if chirps is not None and np.any(np.isfinite(chirps[d])) else None
        im = imerg_field[d] if imerg_field is not None else None

        fields_to_check = [f for f in (bg, an, ch, im) if f is not None]
        rain_max = max(
            15.0,
            float(np.nanpercentile(np.concatenate([f[valid_mask] for f in fields_to_check]), 99))
            if fields_to_check
            else 25.0,
        )
        inc_max = max(3.0, float(np.nanpercentile(np.abs(inc[valid_mask]), 98)))

        # Panel 0: Gauges
        ax = axes1[d, 0]
        ax.imshow(
            np.where(valid_mask, 0.0, np.nan),
            origin="lower",
            extent=extent,
            cmap="Greys",
            vmin=0,
            vmax=1,
            alpha=0.25,
        )
        add_gauge_markers(
            ax,
            station_lon,
            station_lat,
            gauge_mm[d],
            assim_idx,
            eval_idx,
            vmin=0,
            vmax=rain_max,
            cmap="viridis",
        )
        ax.set_ylabel(f"{date_str}\nObs mean {np.nanmean(gauge_mm[d]):.1f} mm", fontsize=10, weight="bold")

        # Panel 1: CHIRPS
        ax = axes1[d, 1]
        if ch is not None:
            im_ch = ax.imshow(
                np.where(valid_mask, ch, np.nan),
                origin="lower",
                extent=extent,
                cmap="viridis",
                vmin=0,
                vmax=rain_max,
            )
        else:
            ax.text(0.5, 0.5, "CHIRPS N/A", ha="center", va="center", transform=ax.transAxes)

        # Panel 2: IMERG
        ax = axes1[d, 2]
        if im is not None:
            im_im = ax.imshow(
                np.where(valid_mask, im, np.nan),
                origin="lower",
                extent=extent,
                cmap="viridis",
                vmin=0,
                vmax=rain_max,
            )
        else:
            ax.text(0.5, 0.5, "IMERG N/A", ha="center", va="center", transform=ax.transAxes)

        # Panel 3: Background
        ax = axes1[d, 3]
        im_bg = ax.imshow(
            np.where(valid_mask, bg, np.nan),
            origin="lower",
            extent=extent,
            cmap="viridis",
            vmin=0,
            vmax=rain_max,
        )

        # Panel 4: Best Analysis
        ax = axes1[d, 4]
        im_an = ax.imshow(
            np.where(valid_mask, an, np.nan),
            origin="lower",
            extent=extent,
            cmap="viridis",
            vmin=0,
            vmax=rain_max,
        )
        # Overlay withheld station dots lightly
        ax.scatter(
            station_lon[eval_idx],
            station_lat[eval_idx],
            c=gauge_mm[d, eval_idx],
            cmap="viridis",
            vmin=0,
            vmax=rain_max,
            s=28,
            marker="o",
            edgecolors="#00E5FF",
            linewidths=1.0,
            zorder=6,
        )

        # Panel 5: Increment
        ax = axes1[d, 5]
        im_inc = ax.imshow(
            np.where(valid_mask, inc, np.nan),
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-inc_max,
            vmax=inc_max,
        )

        for col, axis in enumerate(axes1[d]):
            axis.set_xlim(extent[0], extent[1])
            axis.set_ylim(extent[2], extent[3])
            axis.tick_params(labelsize=8)
            if d == 0:
                axis.set_title(col_titles[col], fontsize=11, weight="bold")
            if col == 4:
                cb = fig1.colorbar(im_an, ax=axis, orientation="horizontal", fraction=0.045, pad=0.05)
                cb.set_label("mm day$^{-1}$", fontsize=8)
                cb.ax.tick_params(labelsize=7)
            elif col == 5:
                cb = fig1.colorbar(im_inc, ax=axis, orientation="horizontal", fraction=0.045, pad=0.05)
                cb.set_label("Increment (mm)", fontsize=8)
                cb.ax.tick_params(labelsize=7)

    fig1.suptitle(
        f"Dense BMD+BWDB Gauge Assimilation Daily Progression ({times[0]} to {times[-1]})\n"
        "Withheld validation gauges circled in cyan; analysis assimilates 242 stations and S04 IMERG",
        fontsize=14,
        weight="bold",
    )
    p1 = out_dir / "dense_sweep_daily_spatial_intercomparison.png"
    fig1.savefig(p1, dpi=200)
    plt.close(fig1)
    print(f"[plot] wrote {p1}", flush=True)

    # =========================================================================
    # FIGURE 2: 5-Day Event Accumulation & Differences (2 rows x 5 columns)
    # =========================================================================
    print("[plot] generating Figure 2: 5-day accumulation and difference maps ...", flush=True)
    fig2, axes2 = plt.subplots(2, 5, figsize=(22, 9), constrained_layout=True)

    bg_sum = np.nansum(meanfields.get("background", np.zeros_like(meanfields[best_arm])), axis=0)
    ref_sum = np.nansum(meanfields.get(ref_arm, meanfields[best_arm]), axis=0)
    best_sum = np.nansum(meanfields[best_arm], axis=0)
    ch_sum = np.nansum(chirps, axis=0) if chirps is not None and np.any(np.isfinite(chirps)) else None
    im_sum = np.nansum(imerg_field, axis=0) if imerg_field is not None else None
    gauge_sum = np.nansum(gauge_mm, axis=0)

    accum_fields = [f for f in (bg_sum, ref_sum, best_sum, ch_sum, im_sum) if f is not None]
    accum_max = max(30.0, float(np.nanpercentile(np.concatenate([f[valid_mask] for f in accum_fields]), 98.5)))

    top_specs = [
        ("A. CHIRPS 5-day total", ch_sum),
        ("B. IMERG 5-day total", im_sum),
        ("C. Background 5-day total", bg_sum),
        (f"D. Control ({ref_arm})", ref_sum),
        (f"E. Best Dense ({best_arm})", best_sum),
    ]

    for col, (title, fld) in enumerate(top_specs):
        ax = axes2[0, col]
        if fld is not None:
            sh = ax.imshow(
                np.where(valid_mask, fld, np.nan),
                origin="lower",
                extent=extent,
                cmap="viridis",
                vmin=0,
                vmax=accum_max,
            )
            if col in (3, 4):
                add_gauge_markers(
                    ax, station_lon, station_lat, gauge_sum, assim_idx, eval_idx, 0, accum_max
                )
            fig2.colorbar(sh, ax=ax, orientation="horizontal", fraction=0.046, pad=0.06, label="mm / 5-day")
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=11, weight="bold")
        ax.tick_params(labelsize=8)

    diff_inc = best_sum - bg_sum
    diff_ctrl = best_sum - ref_sum
    diff_ch = (best_sum - ch_sum) if ch_sum is not None else None
    diff_im = (best_sum - im_sum) if im_sum is not None else None
    diff_im_ch = (im_sum - ch_sum) if (im_sum is not None and ch_sum is not None) else None

    diff_list = [d for d in (diff_inc, diff_ctrl, diff_ch, diff_im, diff_im_ch) if d is not None]
    diff_limit = max(10.0, float(np.nanpercentile(np.abs(np.concatenate([d[valid_mask] for d in diff_list])), 98)))

    bot_specs = [
        (f"F. DA Increment ({best_arm} − Background)", diff_inc),
        (f"G. Dense Network Tuning ({best_arm} − {ref_arm})", diff_ctrl),
        (f"H. Analysis − CHIRPS", diff_ch),
        (f"I. Analysis − IMERG", diff_im),
        ("J. IMERG − CHIRPS (Satellite − Blended)", diff_im_ch),
    ]

    for col, (title, dfld) in enumerate(bot_specs):
        ax = axes2[1, col]
        if dfld is not None:
            sh = ax.imshow(
                np.where(valid_mask, dfld, np.nan),
                origin="lower",
                extent=extent,
                cmap="RdBu_r",
                vmin=-diff_limit,
                vmax=diff_limit,
            )
            fig2.colorbar(sh, ax=ax, orientation="horizontal", fraction=0.046, pad=0.06, label="Difference (mm)")
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=10, weight="bold")
        ax.tick_params(labelsize=8)

    fig2.suptitle(
        f"5-Day Event Accumulation & Observational Intercomparison ({times[0]} to {times[-1]})\n"
        "Panels D & E include all ~303 BMD+BWDB station totals (withheld stations circled in cyan)",
        fontsize=14,
        weight="bold",
    )
    p2 = out_dir / "dense_sweep_5day_accumulation_intercomparison.png"
    fig2.savefig(p2, dpi=200)
    plt.close(fig2)
    print(f"[plot] wrote {p2}", flush=True)

    # =========================================================================
    # FIGURE 3: Quantitative Station Verification Diagnostics (2x2 Panels)
    # =========================================================================
    print("[plot] generating Figure 3: Station verification diagnostics ...", flush=True)
    fig3, axes3 = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)

    ax_sc = axes3[0, 0]
    obs_eval_flat = gauge_mm[:, eval_idx].reshape(-1)
    bg_eval_flat = (
        station_ensembles["background"][:, :, eval_idx].mean(axis=1).reshape(-1)
        if "background" in station_ensembles
        else np.zeros_like(obs_eval_flat)
    )
    ref_eval_flat = (
        station_ensembles[ref_arm][:, :, eval_idx].mean(axis=1).reshape(-1)
        if ref_arm in station_ensembles
        else np.zeros_like(obs_eval_flat)
    )
    best_eval_flat = (
        station_ensembles[best_arm][:, :, eval_idx].mean(axis=1).reshape(-1)
        if best_arm in station_ensembles
        else np.zeros_like(obs_eval_flat)
    )

    sc_max = max(20.0, float(np.nanpercentile(obs_eval_flat, 98.5)))
    ax_sc.plot([0, sc_max], [0, sc_max], "k--", alpha=0.6, label="1:1 Perfect Agreement")

    ax_sc.scatter(
        obs_eval_flat,
        bg_eval_flat,
        c="#78909C",
        alpha=0.45,
        s=20,
        label=f"Background (r={compute_metrics(bg_eval_flat, obs_eval_flat)['corr']:.2f})",
    )
    ax_sc.scatter(
        obs_eval_flat,
        ref_eval_flat,
        c="#FFA726",
        alpha=0.55,
        s=24,
        label=f"Control {ref_arm} (r={compute_metrics(ref_eval_flat, obs_eval_flat)['corr']:.2f})",
    )
    ax_sc.scatter(
        obs_eval_flat,
        best_eval_flat,
        c="#1E88E5",
        alpha=0.75,
        s=28,
        marker="^",
        label=f"Best {best_arm} (r={compute_metrics(best_eval_flat, obs_eval_flat)['corr']:.2f})",
    )
    ax_sc.set_xlim(0, sc_max)
    ax_sc.set_ylim(0, sc_max)
    ax_sc.set_xlabel("Withheld Gauge Observed Rainfall (mm/day)", fontsize=10)
    ax_sc.set_ylabel("Predicted Rainfall (mm/day)", fontsize=10)
    ax_sc.set_title("A. Station Scatter at 61 Withheld Gauges", fontsize=11, weight="bold")
    ax_sc.legend(loc="upper left", fontsize=8)
    ax_sc.grid(alpha=0.3)

    ax_bar = axes3[0, 1]
    sorted_variants = sorted(
        variant_names,
        key=lambda v: (
            fair_crps_1d(
                extract_eval_ensemble_flat(station_ensembles[v], eval_idx),
                obs_eval_flat,
            ).mean()
            if v in station_ensembles
            else 999.0
        ),
    )
    var_crps = []
    var_mae = []
    var_names_clean = []
    for v in sorted_variants:
        if v not in station_ensembles:
            continue
        ens_flat = extract_eval_ensemble_flat(station_ensembles[v], eval_idx)
        c = float(np.nanmean(fair_crps_1d(ens_flat, obs_eval_flat)))
        m = float(np.nanmean(np.abs(ens_flat.mean(axis=1) - obs_eval_flat)))
        var_crps.append(c)
        var_mae.append(m)
        var_names_clean.append(v)

    y_pos = np.arange(len(var_names_clean))
    bars = ax_bar.barh(y_pos - 0.18, var_crps, height=0.36, color="#42A5F5", label="CRPS (mm)")
    ax_bar.barh(y_pos + 0.18, var_mae, height=0.36, color="#B0BEC5", alpha=0.6, label="MAE (mm)")

    for idx, name in enumerate(var_names_clean):
        if name == best_arm:
            bars[idx].set_color("#1565C0")
        elif name == "background":
            bars[idx].set_color("#78909C")

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(var_names_clean, fontsize=8)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Error (mm)", fontsize=10)
    ax_bar.set_title("B. Method Ranking on Withheld Gauges (Lower is Better)", fontsize=11, weight="bold")
    ax_bar.legend(fontsize=8)
    ax_bar.grid(axis="x", alpha=0.3)

    ax_cdf = axes3[1, 0]
    thresh_levels = np.linspace(0.1, max(35.0, sc_max), 150)
    obs_valid = obs_eval_flat[np.isfinite(obs_eval_flat)]

    ax_cdf.plot(
        thresh_levels,
        [(obs_valid >= t).mean() for t in thresh_levels],
        "k-",
        lw=2.0,
        label="Withheld Gauges (True)",
    )
    if "background" in station_ensembles:
        ax_cdf.plot(
            thresh_levels,
            [(bg_eval_flat >= t).mean() for t in thresh_levels],
            "--",
            color="#78909C",
            lw=1.5,
            label="Background Prior",
        )
    if ref_arm in station_ensembles:
        ax_cdf.plot(
            thresh_levels,
            [(ref_eval_flat >= t).mean() for t in thresh_levels],
            "-.",
            color="#FFA726",
            lw=1.5,
            label=f"Control {ref_arm}",
        )
    if best_arm in station_ensembles:
        ax_cdf.plot(
            thresh_levels,
            [(best_eval_flat >= t).mean() for t in thresh_levels],
            "-",
            color="#1E88E5",
            lw=2.0,
            label=f"Best {best_arm}",
        )
    ax_cdf.set_xlabel("Rainfall Intensity Threshold (mm/day)", fontsize=10)
    ax_cdf.set_ylabel("Fraction of Gauge-Days Exceeding Threshold", fontsize=10)
    ax_cdf.set_yscale("log")
    ax_cdf.set_ylim(1e-3, 1.05)
    ax_cdf.set_title("C. Rainfall Exceedance Probability Distribution", fontsize=11, weight="bold")
    ax_cdf.legend(fontsize=8)
    ax_cdf.grid(alpha=0.3, which="both")

    ax_dist = axes3[1, 1]
    if len(eval_idx) > 0 and len(assim_idx) > 0:
        eval_lat_pts = station_lat[eval_idx]
        eval_lon_pts = station_lon[eval_idx]
        assim_lat_pts = station_lat[assim_idx]
        assim_lon_pts = station_lon[assim_idx]
        dist_per_eval_station = np.array([
            float(np.min(111.0 * np.sqrt((elat - assim_lat_pts) ** 2 + (np.cos(np.deg2rad(elat)) * (elon - assim_lon_pts)) ** 2)))
            for elat, elon in zip(eval_lat_pts, eval_lon_pts)
        ])
        dist_eval = np.tile(dist_per_eval_station, n_days)
        bins = [(0, 15), (15, 30), (30, 60), (60, 200)]
        bin_labels = ["<15 km\n(dense cluster)", "15–30 km\n(near)", "30–60 km\n(medium)", ">60 km\n(isolated)"]
        bin_crps_bg = []
        bin_crps_ref = []
        bin_crps_best = []
        for b_low, b_high in bins:
            mask_b = (dist_eval >= b_low) & (dist_eval < b_high) & np.isfinite(obs_eval_flat)
            if mask_b.any():
                if "background" in station_ensembles:
                    ens_bg = extract_eval_ensemble_flat(station_ensembles["background"], eval_idx)
                    bin_crps_bg.append(
                        float(np.nanmean(fair_crps_1d(
                            ens_bg[mask_b],
                            obs_eval_flat[mask_b],
                        )))
                    )
                if ref_arm in station_ensembles:
                    ens_ref = extract_eval_ensemble_flat(station_ensembles[ref_arm], eval_idx)
                    bin_crps_ref.append(
                        float(np.nanmean(fair_crps_1d(
                            ens_ref[mask_b],
                            obs_eval_flat[mask_b],
                        )))
                    )
                if best_arm in station_ensembles:
                    ens_best = extract_eval_ensemble_flat(station_ensembles[best_arm], eval_idx)
                    bin_crps_best.append(
                        float(np.nanmean(fair_crps_1d(
                            ens_best[mask_b],
                            obs_eval_flat[mask_b],
                        )))
                    )
            else:
                bin_crps_bg.append(np.nan)
                bin_crps_ref.append(np.nan)
                bin_crps_best.append(np.nan)

        x_idx = np.arange(len(bins))
        width = 0.25
        if bin_crps_bg:
            ax_dist.bar(x_idx - width, bin_crps_bg, width=width, color="#78909C", label="Background")
        if bin_crps_ref:
            ax_dist.bar(x_idx, bin_crps_ref, width=width, color="#FFA726", label=f"Control {ref_arm}")
        if bin_crps_best:
            ax_dist.bar(x_idx + width, bin_crps_best, width=width, color="#1E88E5", label=f"Best {best_arm}")
        ax_dist.set_xticks(x_idx)
        ax_dist.set_xticklabels(bin_labels, fontsize=9)
        ax_dist.set_ylabel("CRPS at Withheld Gauges (mm)", fontsize=10)
        ax_dist.set_title("D. CRPS by Gauge Separation (Locality & Cluster Benefit)", fontsize=11, weight="bold")
        ax_dist.legend(fontsize=8)
        ax_dist.grid(axis="y", alpha=0.3)
    else:
        ax_dist.text(0.5, 0.5, "Distance metrics N/A", ha="center", va="center", transform=ax_dist.transAxes)

    fig3.suptitle(
        f"Withheld Gauge Validation Diagnostics ({times[0]} to {times[-1]})\n"
        "Strict out-of-sample verification at 61 held-out BMD+BWDB stations",
        fontsize=14,
        weight="bold",
    )
    p3 = out_dir / "dense_sweep_station_verification_diagnostics.png"
    fig3.savefig(p3, dpi=200)
    plt.close(fig3)
    print(f"[plot] wrote {p3}", flush=True)

    # =========================================================================
    # FIGURE 4: Subgrid Spatial Texture & Scale Diagnostics (1x3 Panels)
    # =========================================================================
    print("[plot] generating Figure 4: Spatial texture and scale diagnostics ...", flush=True)
    fig4, axes4 = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)

    ax_tex = axes4[0]
    prod_fields = [
        ("Background", bg_sum / n_days),
        (f"Control ({ref_arm})", ref_sum / n_days),
        (f"Best ({best_arm})", best_sum / n_days),
    ]
    if ch_sum is not None:
        prod_fields.insert(0, ("CHIRPS", ch_sum / n_days))
    if im_sum is not None:
        prod_fields.insert(1, ("IMERG", im_sum / n_days))

    tex_names = [p[0] for p in prod_fields]
    textures = [
        float(np.nanmean(np.abs(p[1] - block_smooth(np.nan_to_num(p[1], nan=0.0), factor=4))[valid_mask]))
        for p in prod_fields
    ]
    ax_tex.bar(np.arange(len(tex_names)), textures, color="#26A69A", alpha=0.85)
    ax_tex.set_xticks(np.arange(len(tex_names)))
    ax_tex.set_xticklabels(tex_names, rotation=25, ha="right", fontsize=9)
    ax_tex.set_ylabel("Subgrid Texture Magnitude (mm/day)", fontsize=10)
    ax_tex.set_title("A. Fine-Scale Structure (Departure from 0.25° Mean)", fontsize=11, weight="bold")
    ax_tex.grid(axis="y", alpha=0.3)

    ax_wet = axes4[1]
    days_x = np.arange(n_days)
    ax_wet.plot(
        days_x,
        [np.nanmean(meanfields["background"][d][valid_mask]) for d in range(n_days)],
        "--",
        color="#78909C",
        label="Background mean",
    )
    ax_wet.plot(
        days_x,
        [np.nanmean(meanfields[best_arm][d][valid_mask]) for d in range(n_days)],
        "-",
        color="#1E88E5",
        lw=2.0,
        label=f"{best_arm} mean",
    )
    if chirps is not None:
        ax_wet.plot(
            days_x,
            [np.nanmean(chirps[d][valid_mask]) for d in range(n_days)],
            ":",
            color="#2E7D32",
            label="CHIRPS mean",
        )
    if imerg_field is not None:
        ax_wet.plot(
            days_x,
            [np.nanmean(imerg_field[d][valid_mask]) for d in range(n_days)],
            "-.",
            color="#E65100",
            label="IMERG mean",
        )

    ax_wet.set_xticks(days_x)
    ax_wet.set_xticklabels([t[5:] for t in times], fontsize=9)
    ax_wet.set_ylabel("Domain Land Mean (mm/day)", fontsize=10)
    ax_wet.set_title("B. Daily Domain-Average Rainfall Evolution", fontsize=11, weight="bold")
    ax_wet.legend(fontsize=8)
    ax_wet.grid(alpha=0.3)

    ax_mat = axes4[2]
    grid_matrix_data = []
    labels_mat = []
    if ch_sum is not None:
        grid_matrix_data.append(ch_sum[valid_mask])
        labels_mat.append("CHIRPS")
    if im_sum is not None:
        grid_matrix_data.append(im_sum[valid_mask])
        labels_mat.append("IMERG")
    grid_matrix_data.append(bg_sum[valid_mask])
    labels_mat.append("Background")
    grid_matrix_data.append(ref_sum[valid_mask])
    labels_mat.append(ref_arm)
    grid_matrix_data.append(best_sum[valid_mask])
    labels_mat.append(best_arm)

    c_mat = np.corrcoef(grid_matrix_data)
    im_c = ax_mat.imshow(c_mat, cmap="Blues", vmin=0.4, vmax=1.0)
    ax_mat.set_xticks(np.arange(len(labels_mat)))
    ax_mat.set_yticks(np.arange(len(labels_mat)))
    ax_mat.set_xticklabels(labels_mat, rotation=35, ha="right", fontsize=9)
    ax_mat.set_yticklabels(labels_mat, fontsize=9)
    for i in range(len(labels_mat)):
        for j in range(len(labels_mat)):
            val = c_mat[i, j]
            ax_mat.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if val > 0.75 else "black",
            )
    fig4.colorbar(im_c, ax=ax_mat, fraction=0.046, pad=0.06, label="Pattern Correlation r")
    ax_mat.set_title("C. 5-Day Gridded Field Pattern Correlation", fontsize=11, weight="bold")

    fig4.suptitle(
        f"Spatial Texture and Cross-Product Consistency ({times[0]} to {times[-1]})",
        fontsize=14,
        weight="bold",
    )
    p4 = out_dir / "dense_sweep_spatial_texture_diagnostics.png"
    fig4.savefig(p4, dpi=200)
    plt.close(fig4)
    print(f"[plot] wrote {p4}", flush=True)

    # =========================================================================
    # Write Markdown & JSON Report
    # =========================================================================
    summary_md = [
        f"# Dense BMD+BWDB Gauge Assimilation Diagnostic Summary",
        f"",
        f"- **Period**: {times[0]} to {times[-1]} ({n_days} days)",
        f"- **Input NPZ**: `{npz_path}`",
        f"- **Stations**: {len(station_ids)} total ({len(assim_idx)} assimilated, {len(eval_idx)} withheld)",
        f"- **Reference Arm**: `{ref_arm}`",
        f"- **Best Arm**: `{best_arm}`",
        f"",
        f"## Verification on Withheld Stations (61 Gauges)",
        f"",
        f"| Variant | CRPS (mm) | MAE (mm) | Correlation r | Bias (mm) |",
        f"| :--- | :---: | :---: | :---: | :---: |",
    ]
    for v in sorted_variants:
        if v not in station_ensembles:
            continue
        ens_flat = extract_eval_ensemble_flat(station_ensembles[v], eval_idx)
        crps_v = float(np.nanmean(fair_crps_1d(ens_flat, obs_eval_flat)))
        m_v = compute_metrics(ens_flat.mean(axis=1), obs_eval_flat)
        summary_md.append(
            f"| `{v}` | **{crps_v:.3f}** | {m_v['mae']:.3f} | {m_v['corr']:.3f} | {m_v['bias']:+.3f} |"
        )

    summary_md.extend([
        "",
        "## Generated Figures",
        "",
        f"1. **Daily Spatial Evolution Maps**: [`{p1.name}`]({p1.name})",
        f"2. **5-Day Accumulation & Intercomparison**: [`{p2.name}`]({p2.name})",
        f"3. **Station Verification & Distribution Diagnostics**: [`{p3.name}`]({p3.name})",
        f"4. **Spatial Texture & Cross-Product Correlation**: [`{p4.name}`]({p4.name})",
        "",
    ])

    report_md_path = out_dir / "dense_sweep_maps_report.md"
    report_md_path.write_text("\n".join(summary_md))
    print(f"[plot] wrote {report_md_path}", flush=True)


if __name__ == "__main__":
    main()
