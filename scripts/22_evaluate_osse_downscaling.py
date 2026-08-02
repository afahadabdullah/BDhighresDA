#!/usr/bin/env python
"""Measure what downscaling actually bought, against three explicit null models.

An OSSE dump mixes three questions that a single field score cannot separate:

  A. DOWNSCALING GAIN.  Does the prior add real information below the 0.5-deg
     conditioning scale?  Null: the coarse conditioning field itself
     (``coarse_base_mm``), which has no genuine variance at 0.05 deg.
     Scored on the BACKGROUND, because this is a property of the prior alone
     and must not be credited to assimilated observations.

  B. SUB-FOOTPRINT GAIN.  With 0.1-deg pseudo-IMERG assimilated, is the
     analysis still right BELOW the footprint?  Null: the truth's own 0.1-deg
     block mean upsampled -- perfect coarse information, exactly zero subgrid
     structure.  In the satellite-only arm this structure was unresolved by
     every assimilated observation.  In gauge and simultaneous arms, point
     gauges can constrain it locally, so attribution must be stated more
     narrowly.

  C. TEXTURE REALISM.  Do members carry the right variance at each wavelength,
     or do they merely look sharp?  Judged by spectra, effective resolution,
     variogram and FSS -- never by RMSE, which rewards blurring.

Claims A and B are deliberately scored on different fields.  Scoring the
analysis against the coarse-input null would let assimilated observations be
counted as downscaling skill, which is the most common way this class of paper
overstates its result.

Outputs (all consumed by ``24_osse_paper_suite.py``):
  --out-report        JSON: every metric, keyed by claim
  --out-figure        multi-panel diagnostic PNG
  --out-spatial-data  NetCDF: georeferenced fields for manuscript maps
  --out-curve-data    NPZ: spectra, FSS grids, ladders and variograms for plots
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bdhires.eval import scale as S  # noqa: E402

INTENSITY_EDGES = np.array([0.0, 1.0, 5.0, 10.0, 25.0, 50.0, np.inf])
INTENSITY_LABELS = ["0-1", "1-5", "5-10", "10-25", "25-50", ">50"]
LADDER_FACTORS = (1, 2, 4, 8)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Scale-explicit downscaling evaluation of a CHIRPS OSSE dump."
    )
    p.add_argument("--dump", required=True, help="NPZ written by scripts/10_osse.py")
    p.add_argument("--factor", type=int, default=None,
                   help="Fine cells per satellite footprint dimension "
                        "(default: satellite_factor from the dump, else 2)")
    p.add_argument("--minimum-valid-fraction", type=float, default=1.0,
                   help="Minimum valid fraction per footprint (default 1.0: "
                        "excludes partial coastal blocks)")
    p.add_argument("--fine-degrees", type=float, default=0.05)
    p.add_argument("--case-index", type=int, default=None,
                   help="Day to map; default is the day with the largest "
                        "truth subgrid RMS")
    p.add_argument("--label", default=None,
                   help="Arm label recorded in the report (default: dump network)")
    p.add_argument("--out-figure", default="data/processed/osse_downscaling.png")
    p.add_argument("--out-report", default="data/processed/osse_downscaling.json")
    p.add_argument("--out-spatial-data",
                   default="data/processed/osse_downscaling_spatial.nc")
    p.add_argument("--out-curve-data",
                   default="data/processed/osse_downscaling_curves.npz")
    p.add_argument("--skip-figure", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dump access
# ---------------------------------------------------------------------------

def _scalar(data, name, default=None):
    if name not in data.files:
        return default
    value = data[name]
    return value.item() if value.ndim == 0 else value


def load_dump(path: Path) -> dict:
    data = np.load(path, allow_pickle=False)
    required = {"background", "analysis", "truth", "valid"}
    missing = required - set(data.files)
    if missing:
        raise SystemExit(f"{path} is missing required arrays: {sorted(missing)}")

    truth = np.asarray(data["truth"], dtype=np.float64)
    layout = str(_scalar(data, "array_layout", ""))

    def member_first(name: str) -> np.ndarray:
        """Normalize production (D,M,H,W) dumps to internal (M,D,H,W)."""
        stack = np.asarray(data[name], dtype=np.float64)
        if stack.ndim != 4 or truth.ndim != 3 or stack.shape[-2:] != truth.shape[-2:]:
            raise SystemExit(
                f"{path}: {name} has shape {stack.shape}; expected a 4-D "
                f"ensemble sharing truth grid {truth.shape}"
            )
        if layout == "day,member,latitude,longitude":
            if stack.shape[0] != truth.shape[0]:
                raise SystemExit(
                    f"{path}: array_layout says day-first but {name} shape "
                    f"{stack.shape} disagrees with {truth.shape[0]} truth days"
                )
            return np.moveaxis(stack, 1, 0)
        if layout == "member,day,latitude,longitude":
            if stack.shape[1] != truth.shape[0]:
                raise SystemExit(
                    f"{path}: array_layout says member-first but {name} shape "
                    f"{stack.shape} disagrees with {truth.shape[0]} truth days"
                )
            return stack

        # Legacy dumps have no layout marker. Infer only when unambiguous;
        # failing loudly is safer than silently swapping days and members.
        first_is_day = stack.shape[0] == truth.shape[0]
        second_is_day = stack.shape[1] == truth.shape[0]
        if first_is_day and not second_is_day:
            return np.moveaxis(stack, 1, 0)
        if second_is_day and not first_is_day:
            return stack
        raise SystemExit(
            f"{path}: cannot infer {name} layout from shape {stack.shape}; "
            "rerun scripts/10_osse.py to write array_layout metadata"
        )

    out = {
        "background": member_first("background"),
        "analysis": member_first("analysis"),
        "truth": truth,
        "valid": np.asarray(data["valid"], dtype=bool),
        "days": [str(d) for d in np.atleast_1d(_scalar(data, "days", np.array([])))],
        "network": str(_scalar(data, "network", "unknown")),
        "obs_error": str(_scalar(data, "obs_error", "unknown")),
        "observation_mode": str(_scalar(data, "observation_mode", "unknown")),
        "satellite_factor": int(_scalar(data, "satellite_factor", 2) or 2),
        "pseudo_satellite": bool(_scalar(data, "pseudo_satellite_enabled", False)),
        "checkpoint": str(_scalar(data, "checkpoint", "unknown")),
        "station_lat": np.atleast_1d(_scalar(data, "station_lat", np.array([]))),
        "station_lon": np.atleast_1d(_scalar(data, "station_lon", np.array([]))),
        "assim_idx": np.atleast_1d(
            _scalar(data, "assim_idx", np.array([], dtype=int))).astype(int),
        "eval_idx": np.atleast_1d(
            _scalar(data, "eval_idx", np.array([], dtype=int))).astype(int),
        "grid_lat": _scalar(data, "grid_lat"),
        "grid_lon": _scalar(data, "grid_lon"),
        "array_layout": "member,day,latitude,longitude",
    }
    # The coarse conditioning field is what makes claim A answerable.  Older
    # dumps predate it; degrade loudly rather than silently skipping the claim.
    coarse = _scalar(data, "coarse_base_mm")
    out["coarse_base_mm"] = (
        np.asarray(coarse, dtype=np.float64) if coarse is not None else None
    )
    return out


# ---------------------------------------------------------------------------
# Claim evaluation
# ---------------------------------------------------------------------------

def evaluate_claims(dump: dict, factor: int, minimum_valid_fraction: float):
    truth = dump["truth"]
    valid = dump["valid"]
    if valid.ndim == 2:
        valid = np.broadcast_to(valid, truth.shape)
    mask = S.eligible_mask(truth, factor, minimum_valid_fraction) & valid

    background = dump["background"]
    analysis = dump["analysis"]
    coarse = dump["coarse_base_mm"]
    member_mask = np.broadcast_to(mask, background.shape[1:])

    report: dict = {"factor": factor, "n_days": int(truth.shape[0]),
                    "n_members": int(background.shape[0]),
                    "evaluated_cells": int(mask.sum())}

    # ---- Claim A: downscaling gain over the coarse input --------------------
    # Scored on the FULL field, not the residual: the question is whether the
    # 0.05-deg product beats the 0.5-deg input everywhere, which is what a user
    # of the product would experience.
    masked_truth = np.where(mask, truth, np.nan)
    if coarse is not None:
        null_a = S.coarse_input_null(coarse, mask)
        a_null = S.deterministic_scores(null_a, masked_truth)
        a_bg = S.deterministic_scores(np.where(member_mask[None], background, np.nan),
                                      masked_truth)
        a_an = S.deterministic_scores(np.where(member_mask[None], analysis, np.nan),
                                      masked_truth)
        report["claim_a_downscaling_gain"] = {
            "null_model": "coarse conditioning field on the fine grid",
            "scored_on": "full field",
            "coarse_input": a_null,
            "background": {**a_bg, **S.skill_against(a_bg, a_null)},
            "analysis_for_reference": {**a_an, **S.skill_against(a_an, a_null)},
        }
    else:
        report["claim_a_downscaling_gain"] = {
            "unavailable": "dump predates coarse_base_mm; rerun 10_osse.py to "
                           "measure downscaling gain over the coarse input",
        }

    # ---- Claim B: sub-footprint gain ---------------------------------------
    truth_coarse, truth_sub = S.scale_decompose(truth, factor, mask)

    def _split(ensemble, index):
        return np.stack([S.scale_decompose(m, factor, member_mask)[index]
                         for m in ensemble])

    bg_sub = _split(background, 1)
    an_sub = _split(analysis, 1)
    null_b = S.deterministic_scores(np.zeros((1, *truth_sub.shape)), truth_sub)
    b_bg = S.deterministic_scores(bg_sub, truth_sub)
    b_an = S.deterministic_scores(an_sub, truth_sub)
    report["claim_b_sub_footprint_gain"] = {
        "null_model": "truth's own footprint mean upsampled (zero subgrid)",
        "scored_on": f"residual below {factor}x fine cells",
        "footprint_perfect": null_b,
        "background": {**b_bg, **S.skill_against(b_bg, null_b)},
        "analysis": {**b_an, **S.skill_against(b_an, null_b)},
    }

    # Coarse component: confirms the analysis actually fit the footprints it was
    # given.  A sanity check, not a claim -- it is trivially easy to do well.
    report["footprint_component"] = {
        "background": S.deterministic_scores(_split(background, 0), truth_coarse),
        "analysis": S.deterministic_scores(_split(analysis, 0), truth_coarse),
    }

    report["by_intensity"] = _stratify_intensity(bg_sub, an_sub, truth, truth_sub, mask)
    report["by_year"] = _stratify_year(bg_sub, an_sub, truth_sub, dump["days"])
    return report, mask, truth_sub, bg_sub, an_sub


def _stratify_intensity(bg_sub, an_sub, truth, truth_sub, mask) -> dict:
    """Subgrid skill by CHIRPS intensity: heavy rain is where texture matters."""
    out: dict = {}
    for low, high, label in zip(INTENSITY_EDGES[:-1], INTENSITY_EDGES[1:],
                                INTENSITY_LABELS):
        selected = mask & (truth >= low) & (truth < high)
        if selected.sum() < 64:
            continue
        target = np.where(selected, truth_sub, np.nan)
        null = S.deterministic_scores(np.zeros((1, *target.shape)), target)
        if not null:
            continue
        bg = S.deterministic_scores(np.where(selected[None], bg_sub, np.nan), target)
        an = S.deterministic_scores(np.where(selected[None], an_sub, np.nan), target)
        out[label] = {
            "footprint_perfect": null,
            "background": {**bg, **S.skill_against(bg, null)},
            "analysis": {**an, **S.skill_against(an, null)},
        }
    return out


def _stratify_year(bg_sub, an_sub, truth_sub, days: list[str]) -> dict:
    """Year by year, so one wet season cannot carry the conclusion."""
    if not days or len(days) != truth_sub.shape[0]:
        return {}
    years = np.array([d[:4] for d in days])
    out: dict = {}
    for year in sorted(set(years.tolist())):
        rows = years == year
        target = truth_sub[rows]
        null = S.deterministic_scores(np.zeros((1, *target.shape)), target)
        if not null:
            continue
        bg = S.deterministic_scores(bg_sub[:, rows], target)
        an = S.deterministic_scores(an_sub[:, rows], target)
        out[year] = {
            "n_days": int(rows.sum()),
            "background": {**bg, **S.skill_against(bg, null)},
            "analysis": {**an, **S.skill_against(an, null)},
        }
    return out


def evaluate_structure(dump: dict, mask: np.ndarray, fine_degrees: float) -> dict:
    """Claim C: spectra, effective resolution, FSS ladder, variogram."""
    truth = dump["truth"]
    background = dump["background"]
    analysis = dump["analysis"]
    coarse = dump["coarse_base_mm"]

    # A single member, not the ensemble mean: averaging members destroys the
    # high-wavenumber variance this diagnostic exists to measure.
    fields = {
        "background_member": background[0],
        "analysis_member": analysis[0],
        "background_mean": background.mean(axis=0),
        "analysis_mean": analysis.mean(axis=0),
    }
    if coarse is not None:
        fields["coarse_input"] = coarse
    spectra = S.spectral_summary(fields, truth, mask, fine_degrees)

    ladder_inputs = {"background": background, "analysis": analysis}
    if coarse is not None:
        ladder_inputs["coarse_input"] = coarse[None]
    ladder = S.scale_ladder(ladder_inputs, truth, mask,
                            factors=LADDER_FACTORS, fine_degrees=fine_degrees)

    fss_targets = [
        ("background_mean", background.mean(axis=0)),
        ("analysis_mean", analysis.mean(axis=0)),
        ("analysis_member", analysis[0]),
    ]
    if coarse is not None:
        fss_targets.append(("coarse_input", coarse))
    fss = {name: S.fss_grid(field, truth, mask) for name, field in fss_targets}

    variograms = {
        "truth": S.variogram(truth, mask),
        "background_member": S.variogram(background[0], mask),
        "analysis_member": S.variogram(analysis[0], mask),
    }
    if coarse is not None:
        variograms["coarse_input"] = S.variogram(coarse, mask)

    return {"spectra": spectra, "scale_ladder": ladder.to_dict(),
            "fss": fss, "variogram": variograms}


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

def _nan_moments(stack: np.ndarray):
    """Member mean/spread that stays quiet on all-NaN (ocean) slices."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(stack, axis=0)
        spread = (np.nanstd(stack, axis=0, ddof=1) if stack.shape[0] > 1
                  else np.zeros_like(mean))
    return mean, spread


def save_curve_data(path: Path, structure: dict) -> None:
    """Flat NPZ so a plotting script never has to re-derive anything."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    spectra = structure.get("spectra", {})
    if spectra:
        payload["spectra_wavelength_km"] = np.asarray(spectra["wavelength_km"])
        payload["spectra_truth_power"] = np.asarray(spectra["truth_power"])
        for name, values in spectra["power"].items():
            payload[f"spectra_power__{name}"] = np.asarray(values)
            payload[f"spectra_ratio__{name}"] = np.asarray(spectra["power_ratio"][name])
    ladder = structure.get("scale_ladder", {})
    if ladder:
        payload["ladder_degrees"] = np.asarray(ladder["degrees"])
        for section in ("aggregated", "residual"):
            for name, rows in ladder[section].items():
                for metric in ("rmse_mm", "crps_mm", "correlation"):
                    payload[f"ladder_{section}__{name}__{metric}"] = np.asarray(
                        [r.get(metric, np.nan) for r in rows], dtype=float)
    for name, grid in structure.get("fss", {}).items():
        payload[f"fss_windows__{name}"] = np.asarray(grid["windows"])
        for threshold, curve in grid["fss"].items():
            payload[f"fss__{name}__{threshold}"] = np.asarray(curve, dtype=float)
    for name, vg in structure.get("variogram", {}).items():
        payload[f"variogram_lags__{name}"] = np.asarray(vg["lags"])
        payload[f"variogram__{name}"] = np.asarray(vg["semivariance"], dtype=float)
    np.savez_compressed(path, **payload)


def save_spatial_data(path: Path, dump: dict, mask, factor, truth_sub,
                      bg_sub, an_sub) -> None:
    """Georeferenced NetCDF for the manuscript map panels."""
    try:
        import xarray as xr
    except ImportError:
        print(f"[warn] xarray unavailable; skipping {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    truth = dump["truth"]
    lat, lon = dump["grid_lat"], dump["grid_lon"]
    if lat is None or lon is None:
        lat = np.arange(truth.shape[-2], dtype=np.float32)
        lon = np.arange(truth.shape[-1], dtype=np.float32)
        coord_note = "grid indices (dump predates grid_lat/grid_lon)"
    else:
        lat, lon = np.asarray(lat), np.asarray(lon)
        coord_note = "degrees north / east"

    bg_mean, _ = _nan_moments(bg_sub)
    an_mean, an_spread = _nan_moments(an_sub)

    dims = ("time", "latitude", "longitude")
    variables = {
        "truth_mm": (dims, truth.astype(np.float32)),
        "truth_subgrid_mm": (dims, truth_sub.astype(np.float32)),
        "background_mean_mm": (dims, dump["background"].mean(axis=0).astype(np.float32)),
        "analysis_mean_mm": (dims, dump["analysis"].mean(axis=0).astype(np.float32)),
        "background_subgrid_mean_mm": (dims, bg_mean.astype(np.float32)),
        "analysis_subgrid_mean_mm": (dims, an_mean.astype(np.float32)),
        "analysis_subgrid_spread_mm": (dims, an_spread.astype(np.float32)),
        "background_subgrid_error_mm": (dims, (bg_mean - truth_sub).astype(np.float32)),
        "analysis_subgrid_error_mm": (dims, (an_mean - truth_sub).astype(np.float32)),
        "subgrid_evaluation_mask": (dims, mask.astype(np.int8)),
    }
    if dump["coarse_base_mm"] is not None:
        variables["coarse_input_mm"] = (dims, dump["coarse_base_mm"].astype(np.float32))
    times = (np.array(dump["days"], dtype="datetime64[D]")
             if dump["days"] else np.arange(truth.shape[0]))
    ds = xr.Dataset(variables, coords={"time": times, "latitude": lat, "longitude": lon})
    if dump["station_lat"].size:
        role = np.zeros(dump["station_lat"].size, dtype=np.int8)
        if dump["assim_idx"].size:
            role[dump["assim_idx"]] = 1
        if dump["eval_idx"].size:
            role[dump["eval_idx"]] = 2
        ds = ds.assign(
            station_latitude=(("station",), np.asarray(dump["station_lat"], np.float32)),
            station_longitude=(("station",), np.asarray(dump["station_lon"], np.float32)),
            station_role=(("station",), role),
        )
        ds["station_role"].attrs.update(flag_values="0 1 2",
                                        flag_meanings="unused assimilated withheld")
    ds.attrs.update(
        title="Scale-separated OSSE downscaling diagnostics",
        checkpoint=dump["checkpoint"], network=dump["network"],
        observation_mode=dump["observation_mode"], obs_error=dump["obs_error"],
        satellite_factor=int(factor), coordinates=coord_note,
        subgrid_definition="field minus its own nested footprint-mean",
        note="truth is CHIRPS; subgrid fields are departures from the footprint "
             "mean and are the only fields relevant to claim B",
    )
    ds.to_netcdf(path)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_diagnostics(path: Path, report: dict, structure: dict, dump: dict,
                     truth_sub, bg_sub, an_sub, case: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 13.5), constrained_layout=True)
    gs = fig.add_gridspec(4, 4)
    spectra = structure.get("spectra", {})

    ax = fig.add_subplot(gs[0, 0:2])
    if spectra:
        wl = np.asarray(spectra["wavelength_km"])
        ax.loglog(wl, spectra["truth_power"], "k-", lw=2, label="CHIRPS truth")
        for name, style in (("coarse_input", "--"), ("background_member", "-"),
                            ("analysis_member", "-"), ("analysis_mean", ":")):
            if name in spectra["power"]:
                ax.loglog(wl, spectra["power"][name], style, lw=1.4,
                          label=name.replace("_", " "))
        ax.invert_xaxis()
        ax.set(xlabel="wavelength (km)", ylabel="power",
               title="A.  Radially averaged power spectra")
        ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")

    ax = fig.add_subplot(gs[0, 2:4])
    if spectra:
        wl = np.asarray(spectra["wavelength_km"])
        ax.axhline(1.0, color="k", lw=1)
        ax.axhspan(0.5, 2.0, color="grey", alpha=0.15)
        for name, ratio in spectra["power_ratio"].items():
            ax.semilogx(wl, ratio, lw=1.4, label=name.replace("_", " "))
        for value in spectra["effective_resolution_km"].values():
            if np.isfinite(value):
                ax.axvline(value, ls=":", lw=0.9, alpha=0.6)
        ax.invert_xaxis(); ax.set_yscale("log")
        ax.set(xlabel="wavelength (km)", ylabel="model / truth power",
               title="B.  Spectral ratio; band = within a factor of two")
        ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")

    ax = fig.add_subplot(gs[1, 0:2])
    ladder = structure.get("scale_ladder", {})
    if ladder:
        # factor 1 is skipped: the residual below the finest scale is
        # identically zero by construction and only flattens the axis.
        for name, rows in ladder["residual"].items():
            ax.plot(ladder["degrees"][1:],
                    [r.get("rmse_mm", np.nan) for r in rows[1:]],
                    marker="o", label=name.replace("_", " "))
        ax.set_xscale("log")
        ax.set(xlabel="aggregation scale (deg)",
               ylabel="residual RMSE (mm day$^{-1}$)",
               title="C.  Error in the component BELOW each scale")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 2:4])
    fss = structure.get("fss", {})
    for name, grid in fss.items():
        curve = grid["fss"].get("10")
        if curve:
            ax.plot(grid["windows"], curve, marker="s", label=name.replace("_", " "))
    if fss:
        target = next(iter(fss.values()))["target"].get("10")
        if target is not None and np.isfinite(target):
            ax.axhline(target, color="k", ls="--", lw=1, label="uniform-skill target")
    ax.set_xscale("log")
    ax.set(xlabel="neighbourhood width (fine cells)", ylabel="FSS",
           title="D.  Fractions skill score at 10 mm day$^{-1}$")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[2, 0:2])
    labels, values, colours = [], [], []
    claim_a = report.get("claim_a_downscaling_gain", {})
    if "background" in claim_a:
        labels.append("A: background\nvs coarse input")
        values.append(claim_a["background"].get("mse_skill", np.nan))
        colours.append("#2F5C8F")
    claim_b = report["claim_b_sub_footprint_gain"]
    labels += ["B: background\nvs perfect footprint", "B: analysis\nvs perfect footprint"]
    values += [claim_b["background"].get("mse_skill", np.nan),
               claim_b["analysis"].get("mse_skill", np.nan)]
    colours += ["#0F7A6B", "#B0512E"]
    ax.bar(labels, values, color=colours)
    ax.axhline(0, color="k", lw=1)
    ax.set(ylabel="MSE skill vs null",
           title="E.  Headline claims (>0 beats the null)")
    ax.grid(alpha=0.3, axis="y")

    ax = fig.add_subplot(gs[2, 2:4])
    by_intensity = report.get("by_intensity", {})
    if by_intensity:
        keys = list(by_intensity)
        x = np.arange(len(keys))
        ax.bar(x - 0.2, [by_intensity[k]["background"].get("mse_skill", np.nan)
                         for k in keys], 0.4, label="background", color="#0F7A6B")
        ax.bar(x + 0.2, [by_intensity[k]["analysis"].get("mse_skill", np.nan)
                         for k in keys], 0.4, label="analysis", color="#B0512E")
        ax.set_xticks(x); ax.set_xticklabels(keys)
        ax.axhline(0, color="k", lw=1)
        ax.set(xlabel="CHIRPS intensity (mm day$^{-1}$)", ylabel="subgrid MSE skill",
               title="F.  Sub-footprint skill by rainfall intensity")
        ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="y")

    an_mean, _ = _nan_moments(an_sub)
    bg_mean, _ = _nan_moments(bg_sub)
    panels = [
        ("truth subgrid", truth_sub[case]),
        ("background subgrid mean", bg_mean[case]),
        ("analysis subgrid mean", an_mean[case]),
        ("analysis subgrid error", (an_mean - truth_sub)[case]),
    ]
    finite = np.abs(truth_sub[case][np.isfinite(truth_sub[case])])
    limit = float(np.percentile(finite, 98)) if finite.size else 1.0
    limit = limit or 1.0
    for column, (title, field) in enumerate(panels):
        ax = fig.add_subplot(gs[3, column])
        image = ax.imshow(field, origin="lower", cmap="RdBu_r", vmin=-limit, vmax=limit)
        ax.set_title(title, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(image, ax=ax, fraction=0.046)

    day = dump["days"][case] if dump["days"] else f"index {case}"
    fig.suptitle(
        f"OSSE downscaling diagnostics -- {dump['network']} / "
        f"{dump['observation_mode']} / {dump['obs_error']} errors -- case {day}",
        fontsize=12,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dump = load_dump(Path(args.dump))
    factor = args.factor or dump["satellite_factor"] or 2

    report, mask, truth_sub, bg_sub, an_sub = evaluate_claims(
        dump, factor, args.minimum_valid_fraction)
    structure = evaluate_structure(dump, mask, args.fine_degrees)

    case = args.case_index
    if case is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            energy = np.nansum(truth_sub**2, axis=(1, 2))
        case = int(np.argmax(energy)) if energy.size else 0

    report.update(
        label=args.label
        or f"{dump['network']}_{dump['observation_mode']}_{dump['obs_error']}",
        dump=str(args.dump),
        checkpoint=dump["checkpoint"],
        network=dump["network"],
        observation_mode=dump["observation_mode"],
        obs_error=dump["obs_error"],
        pseudo_satellite=dump["pseudo_satellite"],
        minimum_valid_fraction=args.minimum_valid_fraction,
        case_index=case,
        case_day=dump["days"][case] if dump["days"] else None,
        structure=structure,
        interpretation=(
            "Claim A is scored on the background alone so that assimilated "
            "observations cannot be credited as downscaling skill. Claim B is "
            "scored against a null that already knows the exact footprint means, "
            "so positive skill there is information no observation carried. "
            "Spectral ratio and FSS guard against a sharp-looking field that is "
            "merely noisy."
        ),
    )

    out_report = Path(args.out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2, default=float) + "\n")
    save_curve_data(Path(args.out_curve_data), structure)
    save_spatial_data(Path(args.out_spatial_data), dump, mask, factor,
                      truth_sub, bg_sub, an_sub)
    if not args.skip_figure:
        plot_diagnostics(Path(args.out_figure), report, structure, dump,
                         truth_sub, bg_sub, an_sub, case)

    claim_a = report.get("claim_a_downscaling_gain", {})
    claim_b = report["claim_b_sub_footprint_gain"]
    print(f"\n{report['label']}")
    if "background" in claim_a:
        print(f"  A  downscaling gain (background vs coarse input): "
              f"MSE {claim_a['background']['mse_skill']:+.3f}  "
              f"CRPS {claim_a['background']['crps_skill']:+.3f}")
    else:
        print("  A  unavailable: dump has no coarse_base_mm (rerun 10_osse.py)")
    print(f"  B  sub-footprint  (analysis vs perfect-footprint null): "
          f"MSE {claim_b['analysis']['mse_skill']:+.3f}  "
          f"CRPS {claim_b['analysis']['crps_skill']:+.3f}")
    print(f"     background for contrast:                        "
          f"MSE {claim_b['background']['mse_skill']:+.3f}")
    eff = structure.get("spectra", {}).get("effective_resolution_km", {})
    if eff:
        pretty = "  ".join(f"{k}={v:.0f}km" for k, v in eff.items() if np.isfinite(v))
        print(f"  C  effective resolution: {pretty}")
    for target in (out_report, args.out_curve_data, args.out_spatial_data):
        print(f"wrote {target}")
    if not args.skip_figure:
        print(f"wrote {args.out_figure}")


if __name__ == "__main__":
    main()
