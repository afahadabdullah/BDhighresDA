#!/usr/bin/env python
"""Standard DA verification suite: is the assimilation actually any good?

Consumes the ``.npz`` written by ``scripts/10_osse.py --dump``, so the plots can
be regenerated in seconds without repeating hours of guided sampling.

Nine panels, each answering one question that a headline RMSE cannot:

A  RANK HISTOGRAM         Is the ensemble calibrated?  Truth should be equally
                          likely to fall in any gap between sorted members.  A U
                          means under-dispersed (the classic DA failure), a dome
                          means over-dispersed, a slope means biased.

B  SPREAD-SKILL           Does the ensemble know when it is uncertain?  Binned by
                          predicted spread, the RMSE of the mean should follow
                          the 1:1 line.  Below it = over-confident.

C  CRPS BY INTENSITY      Where does the assimilation help?  Skill on light rain
                          is cheap; skill on the heavy tail is what matters for
                          flood work, and aggregate CRPS hides the difference.

D  RELIABILITY            When the ensemble says 40% chance of exceeding a
                          threshold, does it happen 40% of the time?  On the
                          diagonal = reliable.

E  FSS BY SCALE           At what spatial scale does the forecast become skilful?
                          Point-wise scores punish a correct feature in slightly
                          the wrong place; FSS says how wrong.

F  POWER SPECTRUM         Does the analysis have the right variance at each
                          scale?  The specific failure mode of a generative
                          downscaler is a field that scores well but is too
                          smooth.  The ensemble MEAN is expected to be smooth --
                          it is a mean -- so background and analysis members are
                          plotted too. Their contrast identifies whether bad
                          texture came from the prior or from DA guidance.

G  INCREMENT vs DISTANCE  How far does one observation reach?  |analysis -
                          background| against distance to the nearest assimilated
                          station.  A flat profile means the guidance is not
                          localising; a spike at zero that dies within a cell or
                          two means it is barely spreading information at all.

H  NORMALISED INNOVATION  The classic consistency check.  (y - H(x_b)) divided by
                          sqrt(background variance + R) should be standard
                          normal.  Too wide means the assumed errors are too
                          small; too narrow means they are too large.  This is
                          how you find out whether R is set sensibly, and it is
                          the panel to read first when the DA underperforms.

I  SUMMARY                The numbers behind the panels.

    python scripts/11_da_diagnostics.py --dump data/processed/osse_dump.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", required=True)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[1.0, 10.0, 25.0],
        help="mm/day thresholds for FSS and the reliability diagram",
    )
    parser.add_argument(
        "--reliability-threshold", type=float, default=10.0,
    )
    parser.add_argument(
        "--windows", type=int, nargs="+", default=[1, 3, 5, 9, 17, 33],
    )
    parser.add_argument("--out-figure", default="data/processed/da_diagnostics.png")
    parser.add_argument("--out-report", default="data/processed/da_diagnostics.json")
    return parser.parse_args()


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def rank_histogram(ensemble: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Counts of the truth's rank among ``members`` sorted values.

    Ties are broken randomly, which matters enormously for precipitation: on a
    dry day most members and the observation are all exactly zero, and always
    ranking ties the same way manufactures a spurious U or L shape.
    """
    members = ensemble.shape[0]
    rng = np.random.default_rng(0)
    below = (ensemble < truth[None]).sum(axis=0)
    equal = (ensemble == truth[None]).sum(axis=0)
    ranks = below + (rng.random(below.shape) * (equal + 1)).astype(int)
    return np.bincount(ranks.ravel(), minlength=members + 1)[: members + 1]


def reliability(
    probability: np.ndarray, occurred: np.ndarray, bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    centres, frequencies, counts = [], [], []
    for low, high in zip(edges[:-1], edges[1:]):
        inside = (probability >= low) & (
            probability < high if high < 1.0 else probability <= 1.0
        )
        counts.append(int(inside.sum()))
        centres.append(float(probability[inside].mean()) if inside.any() else np.nan)
        frequencies.append(float(occurred[inside].mean()) if inside.any() else np.nan)
    return np.array(centres), np.array(frequencies), np.array(counts)


def fraction_skill_score(
    forecast: np.ndarray, truth: np.ndarray, threshold: float, window: int, valid
) -> float:
    """FSS over a square neighbourhood, land cells only."""
    from scipy.ndimage import uniform_filter

    exceed_f = (forecast >= threshold).astype(np.float64)
    exceed_o = (truth >= threshold).astype(np.float64)
    if window > 1:
        exceed_f = uniform_filter(exceed_f, size=window, mode="nearest")
        exceed_o = uniform_filter(exceed_o, size=window, mode="nearest")
    f, o = exceed_f[valid], exceed_o[valid]
    denominator = np.mean(f**2) + np.mean(o**2)
    if denominator == 0:
        return np.nan
    return float(1.0 - np.mean((f - o) ** 2) / denominator)


def radial_spectrum(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Radially averaged power spectrum of a 2-D field."""
    values = np.nan_to_num(field, nan=0.0)
    values = values - values.mean()
    power = np.abs(np.fft.fftshift(np.fft.fft2(values))) ** 2
    height, width = values.shape
    y, x = np.indices((height, width))
    radius = np.hypot(y - height // 2, x - width // 2).astype(int)
    n_bins = min(height, width) // 2
    totals = np.bincount(radius.ravel(), power.ravel(), minlength=n_bins)[:n_bins]
    counts = np.bincount(radius.ravel(), minlength=n_bins)[:n_bins]
    with np.errstate(invalid="ignore", divide="ignore"):
        spectrum = totals / counts
    wavenumber = np.arange(n_bins)
    return wavenumber[1:], spectrum[1:]


def haversine_cells(lat1, lon1, lat2, lon2, res_km: float = 5.55) -> np.ndarray:
    """Great-circle distance in units of grid cells (0.05 deg ~ 5.55 km)."""
    radius = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(a)) / res_km


def main() -> None:
    args = parse_args()
    data = np.load(args.dump, allow_pickle=False)
    background = data["background"]          # (D, M, H, W) mm
    analysis = data["analysis"]
    truth = data["truth"]                    # (D, H, W) mm
    valid = data["valid"].astype(bool)
    assim_idx, eval_idx = data["assim_idx"], data["eval_idx"]
    station_lat, station_lon = data["station_lat"], data["station_lon"]
    obs_transformed = data["obs_transformed"]        # (D, S)
    truth_at_stations = data["truth_at_stations"]    # (D, S) mm
    obs_noise_sd = float(data["obs_noise_sd"])
    pseudo_satellite = (
        bool(data["pseudo_satellite_enabled"])
        if "pseudo_satellite_enabled" in data.files
        else False
    )
    n_days, n_members = background.shape[0], background.shape[1]

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from bdhires.grids import get_grid
    from bdhires.transforms import PrecipTransform

    grid = get_grid(str(data["grid_name"]))
    transform = PrecipTransform.from_dict(json.loads(str(data["precip_transform"])))

    # station-space ensembles, reconstructed by nearest cell (adequate here and
    # avoids a torch dependency in a plotting script)
    rows = np.clip(
        np.round((station_lat - grid.lat_min) / grid.res - 0.5).astype(int),
        0, grid.nlat - 1,
    )
    cols = np.clip(
        np.round((station_lon - grid.lon_min) / grid.res - 0.5).astype(int),
        0, grid.nlon - 1,
    )
    background_stations = background[:, :, rows, cols]   # (D, M, S)
    analysis_stations = analysis[:, :, rows, cols]

    report: dict = {
        "dump": str(args.dump),
        "network": str(data["network"]),
        "obs_error": str(data["obs_error"]),
        "days": int(n_days),
        "members": int(n_members),
        "n_assimilated": int(len(assim_idx)),
        "n_withheld": int(len(eval_idx)),
        "assumed_obs_sd_transformed": obs_noise_sd,
    }

    figure, axes = plt.subplots(3, 3, figsize=(19, 15), constrained_layout=True)
    background_colour, analysis_colour = "#3b78b4", "#c1442e"

    # -- A. rank histogram (withheld stations) --------------------------------
    axis = axes[0, 0]
    width = 0.4
    for offset, (name, ensemble, colour) in enumerate(
        [
            ("background", background_stations, background_colour),
            ("analysis", analysis_stations, analysis_colour),
        ]
    ):
        counts = rank_histogram(
            ensemble[:, :, eval_idx].transpose(1, 0, 2).reshape(n_members, -1),
            truth_at_stations[:, eval_idx].ravel(),
        )
        fraction = counts / counts.sum()
        axis.bar(
            np.arange(len(fraction)) + (offset - 0.5) * width,
            fraction, width, color=colour, label=name, alpha=0.85,
        )
        report[f"rank_histogram_deviation_{name}"] = float(
            np.abs(fraction - 1 / len(fraction)).sum()
        )
    axis.axhline(1 / (n_members + 1), color="black", ls="--", lw=1,
                 label="flat = calibrated")
    axis.set_xlabel("Rank of truth among members")
    axis.set_ylabel("Frequency")
    axis.set_title("A.  Rank histogram (withheld)\nU = under-dispersed, "
                   "dome = over-dispersed", fontsize=10.5)
    axis.legend(fontsize=7.5, frameon=False)
    axis.grid(alpha=0.2)

    # -- B. spread-skill ------------------------------------------------------
    axis = axes[0, 1]
    inflation = np.sqrt((n_members + 1) / n_members)
    for name, ensemble, colour in [
        ("background", background, background_colour),
        ("analysis", analysis, analysis_colour),
    ]:
        mean = ensemble.mean(axis=1)
        spread = ensemble.std(axis=1, ddof=1)
        error = np.abs(mean - truth)
        keep = np.broadcast_to(valid, truth.shape) & np.isfinite(truth)
        s, e = spread[keep], error[keep]
        edges = np.quantile(s, np.linspace(0, 1, 13))
        edges = np.unique(edges)
        centres, rmses = [], []
        for low, high in zip(edges[:-1], edges[1:]):
            inside = (s >= low) & (s < high)
            if inside.sum() > 50:
                centres.append(float(s[inside].mean()))
                rmses.append(float(np.sqrt(np.mean(e[inside] ** 2))))
        axis.plot(centres, rmses, marker="o", ms=4, color=colour, label=name)
        report[f"spread_skill_ratio_{name}"] = float(
            s.mean() * inflation / np.sqrt(np.mean(e**2))
        )
    limit = axis.get_xlim()[1]
    axis.plot([0, limit], [0, limit * inflation], color="black", ls="--", lw=1,
              label="1:1 (calibrated)")
    axis.set_xlabel("Ensemble spread (mm day$^{-1}$)")
    axis.set_ylabel("RMSE of the ensemble mean (mm day$^{-1}$)")
    axis.set_title("B.  Spread-skill\nbelow the line = over-confident",
                   fontsize=10.5)
    axis.legend(fontsize=7.5, frameon=False)
    axis.grid(alpha=0.2)

    # -- C. CRPS by intensity -------------------------------------------------
    axis = axes[0, 2]
    from bdhires.eval import crps_ensemble

    bins = [0, 1, 5, 10, 25, 50, 1e9]
    labels = ["0-1", "1-5", "5-10", "10-25", "25-50", ">50"]
    keep = np.broadcast_to(valid, truth.shape) & np.isfinite(truth)
    flat_truth = truth[keep]
    positions = np.arange(len(labels))
    crps_by_bin = {}
    for offset, (name, ensemble, colour) in enumerate(
        [("background", background, background_colour),
         ("analysis", analysis, analysis_colour)]
    ):
        flat = ensemble.transpose(1, 0, 2, 3)[:, keep]
        values = []
        for low, high in zip(bins[:-1], bins[1:]):
            inside = (flat_truth >= low) & (flat_truth < high)
            values.append(
                float(crps_ensemble(flat[:, inside], flat_truth[inside]))
                if inside.sum() > 20 else np.nan
            )
        axis.bar(positions + (offset - 0.5) * 0.4, values, 0.4,
                 color=colour, label=name)
        crps_by_bin[name] = values
    report["crps_by_intensity"] = {"bins_mm": labels, **crps_by_bin}
    axis.set_xticks(positions, labels)
    axis.set_xlabel("Observed intensity (mm day$^{-1}$)")
    axis.set_ylabel("CRPS (mm day$^{-1}$)")
    axis.set_title("C.  CRPS by intensity\nthe tail is what matters",
                   fontsize=10.5)
    axis.legend(fontsize=7.5, frameon=False)
    axis.grid(axis="y", alpha=0.2)

    # -- D. reliability -------------------------------------------------------
    axis = axes[1, 0]
    threshold = args.reliability_threshold
    occurred = (flat_truth >= threshold).astype(float)
    for name, ensemble, colour in [
        ("background", background, background_colour),
        ("analysis", analysis, analysis_colour),
    ]:
        flat = ensemble.transpose(1, 0, 2, 3)[:, keep]
        probability = (flat >= threshold).mean(axis=0)
        centres, frequencies, counts = reliability(probability, occurred)
        axis.plot(centres, frequencies, marker="o", ms=4, color=colour, label=name)
        brier = float(np.mean((probability - occurred) ** 2))
        report[f"brier_score_{name}_at_{threshold:g}mm"] = brier
    axis.plot([0, 1], [0, 1], color="black", ls="--", lw=1, label="perfect")
    axis.set_xlabel(f"Forecast probability of >= {threshold:g} mm day$^{{-1}}$")
    axis.set_ylabel("Observed frequency")
    axis.set_title(f"D.  Reliability at {threshold:g} mm\non the diagonal = "
                   "trustworthy probabilities", fontsize=10.5)
    axis.legend(fontsize=7.5, frameon=False)
    axis.grid(alpha=0.2)

    # -- E. FSS by scale ------------------------------------------------------
    axis = axes[1, 1]
    styles = {"background": "--", "analysis": "-"}
    fss_report: dict = {}
    for name, ensemble in [("background", background), ("analysis", analysis)]:
        mean = ensemble.mean(axis=1)
        for colour_index, level in enumerate(args.thresholds):
            scores = []
            for window in args.windows:
                per_day = [
                    fraction_skill_score(mean[d], truth[d], level, window, valid)
                    for d in range(n_days)
                ]
                # A threshold nothing reaches on any day gives NaN everywhere;
                # np.nanmean would warn and return NaN.  Say so quietly instead.
                usable = [v for v in per_day if np.isfinite(v)]
                scores.append(float(np.mean(usable)) if usable else np.nan)
            if not any(np.isfinite(scores)):
                print(f"[diagnostics] no cells reach {level:g} mm on any day; "
                      f"FSS for that threshold is undefined")
            axis.plot(
                args.windows, scores, ls=styles[name], marker="o", ms=3,
                color=plt.get_cmap("viridis")(colour_index / max(1, len(args.thresholds) - 1)),
                label=f"{name} >{level:g}mm",
            )
            fss_report[f"{name}_{level:g}mm"] = scores
    report["fss"] = {"windows": args.windows, **fss_report}
    axis.set_xscale("log")
    axis.set_xlabel("Neighbourhood width (cells)")
    axis.set_ylabel("Fractions skill score")
    axis.set_title("E.  FSS by scale\ndashed = background, solid = analysis",
                   fontsize=10.5)
    axis.legend(fontsize=6.5, frameon=False, ncol=2)
    axis.grid(alpha=0.2)

    # -- F. power spectrum ----------------------------------------------------
    axis = axes[1, 2]
    spectra = {}
    for name, field in [
        ("CHIRPS truth", truth),
        ("background mean", background.mean(axis=1)),
        ("analysis mean", analysis.mean(axis=1)),
        ("background member", background[:, 0]),
        ("analysis member", analysis[:, 0]),
    ]:
        accumulated = None
        for d in range(n_days):
            k, power = radial_spectrum(np.where(valid, field[d], 0.0))
            accumulated = power if accumulated is None else accumulated + power
        spectra[name] = accumulated / n_days
        axis.loglog(
            k, spectra[name],
            lw=2.0 if name == "CHIRPS truth" else 1.4,
            ls="-" if "member" in name or "truth" in name else "--",
            label=name,
        )
    report["spectral_ratio_member_to_truth"] = float(
        np.nanmean(spectra["analysis member"][-20:] / spectra["CHIRPS truth"][-20:])
    )
    report["spectral_ratio_background_member_to_truth"] = float(
        np.nanmean(
            spectra["background member"][-20:] / spectra["CHIRPS truth"][-20:]
        )
    )
    report["spectral_ratio_mean_to_truth"] = float(
        np.nanmean(spectra["analysis mean"][-20:] / spectra["CHIRPS truth"][-20:])
    )
    axis.set_xlabel("Wavenumber (higher = finer scale)")
    axis.set_ylabel("Power")
    axis.set_title("F.  Power spectrum\na MEMBER should track the truth; "
                   "the mean should not", fontsize=10.5)
    axis.legend(fontsize=7, frameon=False)
    axis.grid(alpha=0.2, which="both")

    # -- G. increment vs distance to the nearest assimilated station ----------
    axis = axes[2, 0]
    lat_grid, lon_grid = np.meshgrid(grid.lat, grid.lon, indexing="ij")
    distance = np.full(grid.shape, np.inf)
    for index in assim_idx:
        distance = np.minimum(
            distance,
            haversine_cells(
                lat_grid, lon_grid, station_lat[index], station_lon[index], grid.res * 111.0
            ),
        )
    increment = np.abs(analysis.mean(axis=1) - background.mean(axis=1)).mean(axis=0)
    edges = np.array([0, 1, 2, 3, 5, 8, 12, 20, 30, 50, 1e9])
    centres, values = [], []
    for low, high in zip(edges[:-1], edges[1:]):
        inside = valid & (distance >= low) & (distance < high)
        if inside.sum() > 20:
            centres.append(float(distance[inside].mean()))
            values.append(float(increment[inside].mean()))
    axis.plot(centres, values, marker="o", ms=5, color=analysis_colour)
    axis.set_xscale("log")
    axis.set_xlabel("Distance to nearest assimilated station (cells)")
    axis.set_ylabel("|analysis - background| (mm day$^{-1}$)")
    axis.set_title(
        "G.  Increment versus gauge distance\n"
        + (
            "dense satellite is present: not a gauge-localisation test"
            if pseudo_satellite
            else "flat = not localising, cliff = barely spreading"
        ),
        fontsize=10.5,
    )
    axis.grid(alpha=0.2, which="both")
    report["increment_profile"] = {
        "distance_cells": centres, "increment_mm": values
    }

    # -- H. normalised innovation --------------------------------------------
    axis = axes[2, 1]
    background_at_obs = transform.forward(
        np.clip(background_stations.mean(axis=1), 0.0, None)
    )
    background_spread = transform.forward(
        np.clip(background_stations, 0.0, None)
    ).std(axis=1, ddof=1)
    innovation = obs_transformed - background_at_obs
    expected = np.sqrt(background_spread**2 + obs_noise_sd**2)
    normalised = (innovation / np.maximum(expected, 1e-6))[:, assim_idx].ravel()
    normalised = normalised[np.isfinite(normalised)]
    axis.hist(normalised, bins=60, density=True, color=background_colour,
              alpha=0.75, label="observed")
    x = np.linspace(-5, 5, 400)
    axis.plot(x, np.exp(-x**2 / 2) / np.sqrt(2 * np.pi), color="black", lw=1.5,
              label="N(0, 1) if R and B are right")
    axis.set_xlim(-5, 5)
    axis.set_xlabel(r"$(y - H(x_b)) / \sqrt{\sigma_b^2 + R}$")
    axis.set_ylabel("Density")
    observed_sd = float(normalised.std())
    axis.set_title(f"H.  Normalised innovation   sd = {observed_sd:.2f}\n"
                   ">1 = assumed errors too small, <1 = too large",
                   fontsize=10.5)
    axis.legend(fontsize=7.5, frameon=False)
    axis.grid(alpha=0.2)
    report["normalised_innovation_sd"] = observed_sd
    report["normalised_innovation_mean"] = float(normalised.mean())

    # -- I. summary -----------------------------------------------------------
    axis = axes[2, 2]
    axis.axis("off")
    verdicts = []

    def verdict(text: str, ok: bool) -> str:
        return f"{'OK  ' if ok else 'FLAG'}  {text}"

    sd = report["normalised_innovation_sd"]
    verdicts.append(verdict(
        f"innovation sd {sd:.2f} (want ~1.0)", 0.7 <= sd <= 1.4))
    ratio = report["spread_skill_ratio_analysis"]
    verdicts.append(verdict(
        f"spread/skill {ratio:.2f} (want ~1.0)", 0.7 <= ratio <= 1.4))
    deviation = report["rank_histogram_deviation_analysis"]
    verdicts.append(verdict(
        f"rank-histogram deviation {deviation:.3f} (flat = 0)", deviation < 0.35))
    background_spectral = report["spectral_ratio_background_member_to_truth"]
    spectral = report["spectral_ratio_member_to_truth"]
    verdicts.append(verdict(
        f"fine power, background/truth {background_spectral:.2f}",
        0.5 <= background_spectral <= 2.0))
    verdicts.append(verdict(
        f"fine power, analysis/truth {spectral:.2f}", 0.5 <= spectral <= 2.0))
    increment_reach = (
        report["increment_profile"]["increment_mm"][0]
        / max(report["increment_profile"]["increment_mm"][-1], 1e-6)
    )
    if pseudo_satellite:
        verdicts.append(
            f"INFO  increment near/far {increment_reach:.1f}; dense satellite present"
        )
    else:
        verdicts.append(verdict(
            f"increment near/far ratio {increment_reach:.1f} (want > 1)",
            increment_reach > 1.2))

    text = (
        f"network      {report['network']}\n"
        f"obs error    {report['obs_error']}  (sd {obs_noise_sd:.3f})\n"
        f"days         {n_days}    members {n_members}\n"
        f"assimilated  {report['n_assimilated']}    withheld "
        f"{report['n_withheld']}\n\n"
        + "\n".join(verdicts)
        + "\n\nFLAG is a prompt to look, not a failure.\n"
        "Read H first: if the assumed errors are wrong,\n"
        "every other panel is being judged against the\n"
        "wrong yardstick."
    )
    axis.text(0.0, 1.0, text, transform=axis.transAxes, va="top", ha="left",
              family="monospace", fontsize=10)
    axis.set_title("I.  Consistency summary", fontsize=10.5)

    figure.suptitle(
        "BDhighresDA - data-assimilation verification suite\n"
        f"{Path(args.dump).name}   |   network '{report['network']}'   |   "
        f"{report['obs_error']} observations   |   {n_days} days x {n_members} "
        f"members   |   {report['n_assimilated']} assimilated, "
        f"{report['n_withheld']} withheld\n"
        + ("0.1° pseudo-satellite is also assimilated\n" if pseudo_satellite else "")
        + "Dashed/blue = background (no observations), solid/red = analysis",
        fontsize=14,
    )
    Path(args.out_figure).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out_figure, dpi=115)
    plt.close(figure)

    Path(args.out_report).write_text(json.dumps(report, indent=2, default=float) + "\n")
    print("\n".join(verdicts))
    print(f"\nwrote {args.out_figure}")
    print(f"wrote {args.out_report}")


if __name__ == "__main__":
    main()
