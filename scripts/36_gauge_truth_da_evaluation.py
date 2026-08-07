#!/usr/bin/env python
"""Evaluate DA runs against GAUGES, with no reference to CHIRPS anywhere.

Why this replaces the CHIRPS-referenced figures
-----------------------------------------------
``scripts/33_plot_real_da_impact.py`` scores the analysis against CHIRPS, which
is not truth: it is a satellite estimate blended with station reports, it is the
weakest of the three products against BMD gauges at daily scale (r = 0.56 versus
CPC 0.76 and IMERG 0.71), and it runs 44 mm/day low where the gauges report
>= 50.  A "brown" cell in that figure can mean the analysis correctly followed a
gauge.  Nothing here uses it.

The gauge is the only measurement of rainfall in this system, so it is the only
thing treated as truth.

Four diagnostics
----------------
1. **Time series and scores at withheld gauges.**  Per-station traces over the
   window with the background spread band, every analysis arm, and the input
   products as context.  Scores are CRPS, MAE, bias and spread-skill.

2. **Innovation statistics.**  O-B and O-A at the ASSIMILATED gauges, plus the
   consistency ratio

       var(O - B) / (sigma_b^2 + R)

   which is 1 when the background spread and R are jointly correct, in the
   standard Desroziers sense.  This needs no truth at all, and it separates two
   explanations that withheld-gauge scores cannot: a ratio well above 1 means R
   (or the background spread) is too SMALL, which is the open question left by
   the R x10 result -- inflating R by 10 fixed the analysis, but the withheld
   scores could not say whether R was wrong or the background was over-confident.

3. **Representativeness-aware CRPS.**  Every CRPS quoted so far compares a cell
   ensemble against a point gauge and charges the ensemble for point variance it
   never claimed to predict.  ``--sigma-rep`` (measured at 0.410 in transformed
   units by script 35, against the 0.10 the configs assume) is added to the
   ensemble before scoring.  Because the penalty depends on ensemble spread and
   spread differs per arm, it can REORDER arms -- the same failure mode as the
   Jensen mean/median gap.

4. **Analysis against every input product, at the same withheld gauges.**  The
   direct test of whether assimilation beat simply taking a product off the
   shelf, plus the assimilated-versus-withheld gap as an overfitting check.

Example
-------
    python scripts/36_gauge_truth_da_evaluation.py \\
        --dumps data/processed/real_obs_confidence/*.npz \\
                data/processed/cpc_satellite/*.npz \\
        --stats data/processed/stats_cpc.json \\
        --sigma-rep 0.410 \\
        --out-dir data/processed/gauge_truth_da
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bdhires.eval.calibration import rank_histogram, spread_skill  # noqa: E402
from bdhires.eval.metrics import crps_ensemble  # noqa: E402
from bdhires.transforms import PrecipTransform  # noqa: E402

ARMS = {
    "background": "background_at_stations",
    "gauges": "gauge_analysis_at_stations",
    "satellite": "imerg_analysis_at_stations",
    "combined": "combined_analysis_at_stations",
}
PRODUCTS = {
    "chirps": "chirps_at_stations",
    "cpc": "condition_at_stations",
    "satellite_obs": "imerg_at_stations",
}
ARM_COLOURS = {
    "background": "#8a8a8a",
    "gauges": "#1f6f8b",
    "satellite": "#4a7c1f",
    "combined": "#c1440e",
}
WET_THRESHOLD_MM = 1.0


# --------------------------------------------------------------------------


def load_dump(path: Path) -> dict:
    """Read one script-15 NPZ into named arrays.

    Ensembles come out as ``(M, T, S)``: the dump stores ``(T, M, S)`` and every
    scoring function here wants members on the leading axis.
    """
    z = np.load(path, allow_pickle=False)
    out = {
        "name": path.stem,
        "gauge_mm": z["gauge_mm"],                       # (T, S) truth
        "assim_idx": z["assim_idx"],
        "eval_idx": z["eval_idx"],
        "station_name": z["station_name"],
        "station_lat": z["station_lat"],
        "station_lon": z["station_lon"],
        "time": z["time"].astype("datetime64[ns]").astype("datetime64[D]"),
    }
    for arm, key in ARMS.items():
        out[arm] = np.moveaxis(z[key], 1, 0) if key in z else None
    for product, key in PRODUCTS.items():
        out[product] = z[key] if key in z else None
    # Provenance, written by script 15 from commit "config overrides" onward.
    # Older dumps have none, which is itself worth reporting: a run whose config
    # edit never took effect looks exactly like one where it did.
    out["config_path"] = str(z["config_path"]) if "config_path" in z else None
    out["config_overrides"] = (
        [str(x) for x in z["config_overrides"]] if "config_overrides" in z else None
    )
    out["config_effective"] = (
        str(z["config_effective"]) if "config_effective" in z else None
    )
    return out


def distinct_arms(dump: dict) -> list[str]:
    """Arms that are genuinely different runs, not aliases.

    Script 15 writes ``analysis_imerg`` and ``analysis_combined`` as copies of
    the gauge-only analysis when no satellite was assimilated, so a gauge-only
    run would otherwise be reported three times as though the agreement between
    identical arrays meant something.
    """
    keep = ["background"]
    for arm in ("gauges", "satellite", "combined"):
        block = dump.get(arm)
        if block is None:
            continue
        if arm != "gauges" and dump.get("gauges") is not None:
            if np.allclose(
                np.nan_to_num(block), np.nan_to_num(dump["gauges"]), equal_nan=True
            ):
                continue
        keep.append(arm)
    return keep


def perturb_for_point_scale(
    ensemble_mm: np.ndarray,
    sigma_rep: float,
    transform: PrecipTransform,
    seed: int = 0,
) -> np.ndarray:
    """Add point-scale variance to a CELL ensemble before scoring against a gauge.

    A member predicts a 5 km cell average.  A gauge measures a point.  Scoring
    one against the other without acknowledging the difference charges the
    ensemble for spread it never claimed to have, and the charge is larger for
    sharper ensembles -- so it can reorder arms rather than shifting them all
    equally.

    The perturbation is applied in TRANSFORMED space, where sigma_rep was
    measured and where the error is closer to additive and symmetric, then
    mapped back to mm.  Doing it in mm would put symmetric noise on a strongly
    skewed variable and generate negative rainfall that then has to be clipped,
    which biases the mean upward.
    """
    if sigma_rep <= 0:
        return ensemble_mm
    rng = np.random.default_rng(seed)
    transformed = transform.forward(np.nan_to_num(ensemble_mm, nan=0.0))
    noisy = transformed + rng.normal(0.0, sigma_rep, size=transformed.shape)
    out = transform.inverse(noisy)
    return np.where(np.isfinite(ensemble_mm), out, np.nan)


def score_arm(
    ensemble_mm: np.ndarray,
    truth_mm: np.ndarray,
    sigma_rep: float,
    transform: PrecipTransform,
    seed: int = 0,
) -> dict:
    """CRPS/MAE/bias/spread-skill for one arm at one set of stations.

    Reports CRPS twice: as computed today, and with point-scale variance added.
    Median bias is carried alongside the mean because the mean carries a
    per-arm Jensen inflation from the log transform and is not comparable across
    arms -- a lesson already learned once on this project.
    """
    ok = np.isfinite(truth_mm) & np.all(np.isfinite(ensemble_mm), axis=0)
    if ok.sum() < 2:
        return {"n": int(ok.sum())}
    members = ensemble_mm[:, ok]
    truth = truth_mm[ok]
    mean = members.mean(axis=0)
    difference = mean - truth

    inflated = perturb_for_point_scale(members, sigma_rep, transform, seed=seed)
    calibration = spread_skill(members, truth)

    return {
        "n": int(ok.sum()),
        "crps": crps_ensemble(members, truth),
        "crps_point_scale": crps_ensemble(inflated, truth),
        "bias_mm": float(np.mean(difference)),
        "median_bias_mm": float(np.median(difference)),
        "mae_mm": float(np.mean(np.abs(difference))),
        "median_ae_mm": float(np.median(np.abs(difference))),
        "rmse_mm": float(np.sqrt(np.mean(difference**2))),
        "spread_mm": float(np.sqrt(np.mean(members.var(axis=0, ddof=1)))),
        "spread_skill_ratio": float(calibration.get("ratio", np.nan)),
        "wet_fraction_pred": float(np.mean(mean >= WET_THRESHOLD_MM)),
        "wet_fraction_obs": float(np.mean(truth >= WET_THRESHOLD_MM)),
        # On a short dry window the MEDIAN is decided by dry-dry pairs that agree
        # exactly, so it rewards anything able to output a hard zero and says
        # little about skill where rain actually fell.
        "wet_median_ae_mm": (
            float(np.median(np.abs(difference[truth >= WET_THRESHOLD_MM])))
            if np.any(truth >= WET_THRESHOLD_MM) else float("nan")
        ),
        "wet_mae_mm": (
            float(np.mean(np.abs(difference[truth >= WET_THRESHOLD_MM])))
            if np.any(truth >= WET_THRESHOLD_MM) else float("nan")
        ),
        "n_wet": int(np.sum(truth >= WET_THRESHOLD_MM)),
    }


def score_product(product_mm: np.ndarray, truth_mm: np.ndarray) -> dict:
    """Deterministic scores for a raw input product at the same stations."""
    ok = np.isfinite(product_mm) & np.isfinite(truth_mm)
    if ok.sum() < 2:
        return {"n": int(ok.sum())}
    difference = product_mm[ok] - truth_mm[ok]
    truth = truth_mm[ok]
    wet = truth >= WET_THRESHOLD_MM
    return {
        "n": int(ok.sum()),
        "bias_mm": float(np.mean(difference)),
        "median_bias_mm": float(np.median(difference)),
        "mae_mm": float(np.mean(np.abs(difference))),
        "median_ae_mm": float(np.median(np.abs(difference))),
        "rmse_mm": float(np.sqrt(np.mean(difference**2))),
        # A deterministic product can output an exact zero and score 0.00 median
        # against a dry gauge; an ensemble mean essentially never can. Comparing
        # the two on median alone is not a fair fight, so the wet subset and the
        # mean are carried alongside.
        "wet_median_ae_mm": (
            float(np.median(np.abs(difference[wet]))) if wet.any() else float("nan")
        ),
        "wet_mae_mm": (
            float(np.mean(np.abs(difference[wet]))) if wet.any() else float("nan")
        ),
        "n_wet": int(wet.sum()),
    }


def innovation_statistics(
    dump: dict,
    transform: PrecipTransform,
    sigma_obs: float,
    sigma_rep: float,
) -> dict:
    """Desroziers-style consistency at the ASSIMILATED gauges.

    ``var(O - B)`` should equal ``sigma_b^2 + R`` if the background spread and
    the observation error are both right.  The ratio is the single most useful
    number for deciding whether R is mis-set, and unlike a withheld-gauge score
    it needs no independent truth.

    Read it as:

    * ratio ~ 1  -- background spread and R are jointly consistent.
    * ratio >> 1 -- the innovations are LARGER than the system claims they can
      be, so R and/or the background spread are too small.  This is the
      signature of over-trusted observations, and it is what the R x10 result
      was compensating for empirically.
    * ratio << 1 -- the system is over-dispersed and the observations are being
      under-used.

    Everything is computed in TRANSFORMED units, which is where R lives.  The
    assimilated gauges are used deliberately: the statistic is about the
    system's own consistency, not about generalisation.
    """
    assim = dump["assim_idx"]
    truth = dump["gauge_mm"][:, assim]
    background = dump["background"][:, :, assim]
    out = {"n_assimilated": int(len(assim)), "sigma_obs": sigma_obs,
           "sigma_rep": sigma_rep}

    truth_t = transform.forward(truth)
    background_t = transform.forward(np.nan_to_num(background, nan=0.0))
    background_mean = background_t.mean(axis=0)
    background_var = background_t.var(axis=0, ddof=1)

    ok = np.isfinite(truth_t) & np.isfinite(background_mean)
    if ok.sum() < 3:
        return {**out, "n": int(ok.sum())}
    innovation = (truth_t - background_mean)[ok]
    r_total = sigma_obs**2 + sigma_rep**2
    expected = float(np.mean(background_var[ok]) + r_total)

    out.update({
        "n": int(ok.sum()),
        "innovation_mean": float(np.mean(innovation)),
        "innovation_var": float(np.var(innovation, ddof=1)),
        "background_spread_var": float(np.mean(background_var[ok])),
        "R_total": r_total,
        "expected_var": expected,
        "consistency_ratio": float(np.var(innovation, ddof=1) / expected)
        if expected > 0 else float("nan"),
    })

    for arm in ("gauges", "combined"):
        block = dump.get(arm)
        if block is None:
            continue
        analysis_t = transform.forward(
            np.nan_to_num(block[:, :, assim], nan=0.0)
        ).mean(axis=0)
        residual = (truth_t - analysis_t)[ok]
        out[f"residual_{arm}_mean"] = float(np.mean(residual))
        out[f"residual_{arm}_var"] = float(np.var(residual, ddof=1))
        # Desroziers: <(O-A)(O-B)> should equal R when the system is consistent.
        out[f"desroziers_R_{arm}"] = float(np.mean(residual * innovation))
    return out


# --------------------------------------------------------------------------
# plots


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
        "axes.grid": True, "grid.alpha": 0.25,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    return plt


def plot_station_timeseries(dump: dict, arms: list[str], out_path: Path) -> None:
    """Withheld gauges through the window, truth in black, arms in colour.

    The background is drawn as a 5-95% band rather than a line, because the
    question at a withheld gauge is whether the truth is INSIDE the ensemble,
    not whether it matches the mean.  Input products appear as thin dashed lines
    so it is visible when an arm simply reproduced one of them.
    """
    plt = _style()
    eval_idx = dump["eval_idx"]
    dates = dump["time"]
    n = len(eval_idx)
    columns = min(4, n)
    rows = int(np.ceil(n / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(3.6 * columns, 2.6 * rows),
                                squeeze=False, sharex=True)

    for position, station in enumerate(eval_idx):
        axis = axes[position // columns][position % columns]
        truth = dump["gauge_mm"][:, station]

        band = dump["background"][:, :, station]
        low, high = np.nanpercentile(band, [5, 95], axis=0)
        axis.fill_between(dates, low, high, color=ARM_COLOURS["background"],
                          alpha=0.22, lw=0, label="background 5-95%")

        for product, style in (("chirps", ":"), ("cpc", "-."), ("satellite_obs", "--")):
            series = dump.get(product)
            if series is None or not np.isfinite(series[:, station]).any():
                continue
            axis.plot(dates, series[:, station], style, color="#999999", lw=0.9,
                      label=product.replace("_obs", ""))

        for arm in arms:
            if arm == "background":
                continue
            axis.plot(dates, np.nanmean(dump[arm][:, :, station], axis=0), "o-",
                      color=ARM_COLOURS.get(arm, "#333333"), ms=3, lw=1.5, label=arm)

        axis.plot(dates, truth, "k-", lw=2.2, label="gauge (truth)", zorder=6)
        axis.plot(dates, truth, "ko", ms=4, zorder=7)
        axis.set_title(str(dump["station_name"][station]), fontsize=8.5)
        axis.tick_params(axis="x", rotation=45, labelsize=6.5)

    for extra in range(n, rows * columns):
        axes[extra // columns][extra % columns].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=min(len(labels), 7),
                  fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    figure.suptitle(f"{dump['name']} — withheld gauges are truth; CHIRPS is context only",
                    y=1.0, fontsize=10)
    figure.tight_layout()
    figure.savefig(out_path, bbox_inches="tight")
    plt.close(figure)


def plot_innovation(results: dict, out_path: Path) -> None:
    """Consistency ratio per run, with the value that means 'R is right' marked."""
    plt = _style()
    names = [n for n, r in results.items() if r.get("innovation", {}).get("n")]
    if not names:
        return
    ratios = [results[n]["innovation"]["consistency_ratio"] for n in names]

    figure, (left, right) = plt.subplots(
        1, 2, figsize=(12.5, 4.4), gridspec_kw={"width_ratios": [1.35, 1]}
    )
    colours = ["#c1440e" if r > 1.5 else "#4a7c1f" if r > 0.67 else "#1f6f8b"
               for r in ratios]
    positions = np.arange(len(names))
    left.barh(positions, ratios, color=colours)
    left.axvline(1.0, color="#111111", lw=1.6, ls="--", label="consistent (=1)")
    left.set_yticks(positions)
    left.set_yticklabels(names, fontsize=7.5)
    left.set_xlabel("var(O-B) / (background spread + R)")
    left.set_title("Consistency ratio at assimilated gauges\n"
                   ">1 means R and/or background spread are TOO SMALL", fontsize=9)
    left.legend(fontsize=8)
    for position, ratio in zip(positions, ratios):
        left.text(ratio, position, f" {ratio:.2f}", va="center", fontsize=7)

    # Observed against EXPECTED, not against R alone: the system's claim is
    # background spread PLUS R, so only that comparison has a meaningful 1:1
    # line. Plotting against R by itself puts every point above the diagonal
    # regardless of whether the system is consistent.
    for name in names:
        block = results[name]["innovation"]
        right.scatter(block["expected_var"], block["innovation_var"], s=55,
                      label=name, alpha=0.85, zorder=3)
        right.annotate(name, (block["expected_var"], block["innovation_var"]),
                       fontsize=6.5, xytext=(4, 4), textcoords="offset points")
    limit = max(
        [results[n]["innovation"]["innovation_var"] for n in names]
        + [results[n]["innovation"]["expected_var"] for n in names]
    ) * 1.15
    right.plot([0, limit], [0, limit], "k--", lw=1.2, label="consistent")
    right.fill_between([0, limit], [0, limit], [0, limit * 2], color="#c1440e",
                       alpha=0.07, lw=0)
    right.text(limit * 0.30, limit * 0.80, "innovations larger than\nthe system allows\n"
               "(R or spread too small)", fontsize=7, color="#c1440e")
    right.set_xlim(0, limit)
    right.set_ylim(0, limit)
    right.set_xlabel("expected: background spread + R (transformed variance)")
    right.set_ylabel("observed var(O-B)")
    right.set_title("Observed innovation variance vs what the system claims",
                    fontsize=9)
    right.legend(fontsize=6.5, loc="lower right")

    figure.tight_layout()
    figure.savefig(out_path, bbox_inches="tight")
    plt.close(figure)


def plot_arm_comparison(results: dict, out_path: Path) -> None:
    """Every arm and every raw product scored on the same withheld gauges."""
    plt = _style()
    names = list(results)
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharey=True)

    for axis, (metric, title) in zip(axes, (
        ("median_ae_mm", "median absolute error (mm/day)"),
        ("crps", "CRPS as scored today"),
        ("crps_point_scale", "CRPS with point-scale variance added"),
    )):
        labels, values, colours = [], [], []
        for name in names:
            for arm, score in results[name]["withheld"].items():
                if metric not in score:
                    continue
                labels.append(f"{name} · {arm}")
                values.append(score[metric])
                colours.append(ARM_COLOURS.get(arm, "#777777"))
            for product, score in results[name].get("products", {}).items():
                if metric != "median_ae_mm" or metric not in score:
                    continue
                labels.append(f"{name} · [{product}]")
                values.append(score[metric])
                colours.append("#cccccc")
        order = np.argsort(values)
        positions = np.arange(len(values))
        axis.barh(positions, np.array(values)[order],
                  color=np.array(colours)[order])
        axis.set_yticks(positions)
        axis.set_yticklabels(np.array(labels)[order], fontsize=6)
        axis.set_xlabel(title)
        axis.invert_yaxis()

    axes[0].set_title("Grey = raw input product, not an analysis", fontsize=9)
    figure.suptitle("Scored against withheld gauges only — lower is better", y=1.01)
    figure.tight_layout()
    figure.savefig(out_path, bbox_inches="tight")
    plt.close(figure)


def plot_rank_histograms(results: dict, dumps: dict, out_path: Path) -> None:
    """Rank histograms at withheld gauges, with and without point-scale variance.

    U-shaped means under-dispersed, dome-shaped over-dispersed.  The pair of
    panels shows how much of any apparent under-dispersion was really the point
    versus cell mismatch rather than an ensemble that is too narrow.
    """
    plt = _style()
    names = list(results)
    if not names:
        return
    figure, axes = plt.subplots(2, len(names), figsize=(3.4 * len(names), 6.2),
                               squeeze=False)
    for column, name in enumerate(names):
        dump = dumps[name]
        # Pick from the arms actually SCORED, not from what the dump contains:
        # distinct_arms drops analysis_imerg/analysis_combined when they are
        # byte-identical copies of the gauge-only analysis, so asking the dump
        # directly can name an arm that has no inflated ensemble behind it.
        scored = [a for a in ("combined", "satellite", "gauges")
                  if a in results[name]["_inflated"]]
        if not scored:
            continue
        arm = scored[0]
        block = dump.get(arm)
        if block is None:
            continue
        eval_idx = dump["eval_idx"]
        truth = dump["gauge_mm"][:, eval_idx].ravel()
        members = block[:, :, eval_idx].reshape(block.shape[0], -1)
        for row, (ens, label) in enumerate((
            (members, "as scored today"),
            (results[name]["_inflated"][arm], "with point-scale variance"),
        )):
            axis = axes[row][column]
            histogram = rank_histogram(ens, truth)
            axis.bar(np.arange(len(histogram)), histogram / max(histogram.sum(), 1),
                     color=ARM_COLOURS.get(arm, "#777777"))
            axis.axhline(1.0 / len(histogram), color="#111111", ls="--", lw=1)
            axis.set_title(f"{name}\n{arm} — {label}", fontsize=7.5)
            axis.set_xticks([])
            if column == 0:
                axis.set_ylabel("frequency")
    figure.suptitle("Rank histograms at withheld gauges — flat is calibrated", y=1.0)
    figure.tight_layout()
    figure.savefig(out_path, bbox_inches="tight")
    plt.close(figure)


# --------------------------------------------------------------------------


def report_duplicate_runs(results: dict) -> list[list[str]]:
    """Flag runs whose withheld scores are identical to the last digit.

    Two configurations cannot agree exactly by chance.  When they do, the run
    did not happen: the usual cause is a wrapper that reuses ``${PREFIX}.npz``
    when the file already exists, so a config change silently replays the old
    dump.  This caught ``measured_R`` replaying ``s1r1`` -- a config edit that
    appeared to run for an hour and changed nothing.
    """
    signature = {}
    for name, block in results.items():
        key = json.dumps(block["withheld"], sort_keys=True, default=float)
        signature.setdefault(key, []).append(name)
    groups = [names for names in signature.values() if len(names) > 1]
    if groups:
        print()
        print("[DUPLICATE] runs with IDENTICAL withheld scores -- these did not")
        print("    actually run as separate configurations. Check for a cached NPZ")
        print("    being reused by the submission wrapper:")
        for names in groups:
            print("      " + "  ==  ".join(names))
    return groups


def print_provenance(dumps: dict) -> None:
    """What config each dump actually used, so a no-op run is visible at once.

    Two arms of this project were compared three times before it emerged that
    the YAML edits behind them had never been made: each job ran, wrote a dump,
    and produced numbers identical to the baseline. Nothing in the output said
    so. This table does.
    """
    print()
    print("[provenance] the config each dump ACTUALLY used:")
    missing = [n for n, d in dumps.items() if d.get("config_effective") is None]
    for name, dump in dumps.items():
        overrides = dump.get("config_overrides")
        if dump.get("config_effective") is None:
            print(f"    {name:28s} (no provenance -- dump predates --set support)")
            continue
        label = "; ".join(overrides) if overrides else "no overrides (config as-is)"
        print(f"    {name:28s} {dump.get('config_path')}  [{label}]")
    if missing:
        print()
        print(f"    {len(missing)} dump(s) carry no provenance. Regenerate them with a")
        print("    current script 15 if their configuration matters to a conclusion.")


def print_tables(results: dict) -> None:
    print()
    print("[withheld] GAUGE IS TRUTH. Scored on withheld stations only.")
    print("    medMAE over a short dry window is decided by dry-dry pairs that agree")
    print("    exactly; wetMAE (gauge >= 1 mm) is the number that reflects skill.")
    print(f"    {'run':28s} {'arm':10s} {'n':>4s} {'nwet':>4s} {'medbias':>8s} "
          f"{'medMAE':>7s} {'meanMAE':>8s} {'wetMAE':>7s} {'CRPS':>7s} "
          f"{'CRPS_pt':>8s} {'sprd/skl':>8s}")
    for name, block in results.items():
        for arm, s in block["withheld"].items():
            if not s.get("n"):
                continue
            print(f"    {name:28s} {arm:10s} {s['n']:>4d} {s.get('n_wet', 0):>4d} "
                  f"{s['median_bias_mm']:>+8.2f} {s['median_ae_mm']:>7.2f} "
                  f"{s['mae_mm']:>8.2f} {s.get('wet_mae_mm', float('nan')):>7.2f} "
                  f"{s['crps']:>7.3f} {s['crps_point_scale']:>8.3f} "
                  f"{s['spread_skill_ratio']:>8.2f}")

    print()
    print("[products] same withheld gauges, raw inputs, no assimilation.")
    print("    A deterministic product can emit an exact zero and score 0.00 median")
    print("    against a dry gauge; an ensemble mean essentially never can, so read")
    print("    meanMAE/wetMAE when comparing these against the arms above.")
    print(f"    {'run':28s} {'product':14s} {'n':>4s} {'medbias':>8s} {'medMAE':>7s} "
          f"{'meanMAE':>8s} {'wetMAE':>7s}")
    for name, block in results.items():
        for product, s in block.get("products", {}).items():
            if not s.get("n"):
                continue
            print(f"    {name:28s} {product:14s} {s['n']:>4d} "
                  f"{s['median_bias_mm']:>+8.2f} {s['median_ae_mm']:>7.2f} "
                  f"{s['mae_mm']:>8.2f} {s.get('wet_mae_mm', float('nan')):>7.2f}")

    print()
    print("[innovation] consistency at ASSIMILATED gauges, transformed units.")
    print("    NOTE: var(O-B) uses only the BACKGROUND, which is the same checkpoint,")
    print("    days and fold for every run here -- so the ratio is a property of the")
    print("    background and is IDENTICAL across configurations by construction. It")
    print("    diagnoses the prior, not the assimilation settings.")
    print(f"    {'run':28s} {'var(O-B)':>9s} {'spread':>8s} {'R':>7s} "
          f"{'expected':>9s} {'ratio':>7s}  verdict")
    for name, block in results.items():
        i = block.get("innovation", {})
        if not i.get("n"):
            continue
        ratio = i["consistency_ratio"]
        verdict = ("R and/or spread TOO SMALL" if ratio > 1.5 else
                   "OVER-DISPERSED: prior spread too large" if ratio < 0.67 else
                   "consistent")
        print(f"    {name:28s} {i['innovation_var']:>9.3f} "
              f"{i['background_spread_var']:>8.3f} {i['R_total']:>7.3f} "
              f"{i['expected_var']:>9.3f} {ratio:>7.2f}  {verdict}")

    print()
    print("[Desroziers] <(O-A)(O-B)> estimates R and DOES depend on the run.")
    print("    This is the run-dependent R diagnostic. If the estimate is far BELOW")
    print("    the assumed R, the analysis is fitting the assimilated gauges harder")
    print("    than the stated observation error permits -- i.e. the effective R the")
    print("    guidance applies is smaller than the R in the config.")
    print(f"    {'run':28s} {'R assumed':>10s} {'R gauges':>9s} {'R combined':>11s} "
          f"{'ratio':>7s}  verdict")
    for name, block in results.items():
        i = block.get("innovation", {})
        if not i.get("n"):
            continue
        estimates = [i.get(f"desroziers_R_{a}") for a in ("gauges", "combined")]
        best = next((e for e in reversed(estimates) if e is not None), None)
        if best is None:
            continue
        ratio = best / i["R_total"] if i["R_total"] else float("nan")
        verdict = ("analysis OVERFITS the gauges" if ratio < 0.5 else
                   "analysis under-uses the gauges" if ratio > 2.0 else
                   "consistent with the stated R")
        print(f"    {name:28s} {i['R_total']:>10.3f} "
              f"{estimates[0] if estimates[0] is not None else float('nan'):>9.3f} "
              f"{estimates[1] if estimates[1] is not None else float('nan'):>11.3f} "
              f"{ratio:>7.2f}  {verdict}")

    print()
    print("[overfit] assimilated minus withheld median MAE (large = fitting the")
    print("    gauges it was given rather than the field):")
    print(f"    {'run':28s} {'arm':10s} {'assim':>7s} {'withheld':>9s} {'gap':>7s}")
    for name, block in results.items():
        for arm in block["withheld"]:
            a = block.get("assimilated", {}).get(arm, {})
            w = block["withheld"][arm]
            if not a.get("n") or not w.get("n"):
                continue
            print(f"    {name:28s} {arm:10s} {a['median_ae_mm']:>7.2f} "
                  f"{w['median_ae_mm']:>9.2f} {w['median_ae_mm'] - a['median_ae_mm']:>+7.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gauge-truth evaluation of DA runs (no CHIRPS anywhere)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dumps", nargs="+", required=True,
                        help="script-15 NPZ dumps; globs are fine")
    parser.add_argument("--stats", required=True, help="stats JSON, for the transform")
    parser.add_argument("--sigma-rep", type=float, default=0.410,
                        help="point-vs-cell sd in TRANSFORMED units, from script 35")
    parser.add_argument("--sigma-obs", type=float, default=0.10,
                        help="gauge measurement error in transformed units, for the "
                             "consistency ratio only")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = json.loads(Path(args.stats).read_text())
    transform = PrecipTransform(**stats["precip_transform"])

    paths = sorted({Path(p) for pattern in args.dumps for p in
                    ([pattern] if Path(pattern).exists()
                     else __import__("glob").glob(pattern))})
    paths = [p for p in paths if p.suffix == ".npz"]
    if not paths:
        raise SystemExit(f"no NPZ dumps matched {args.dumps}")
    print(f"[setup] {len(paths)} dump(s); sigma_rep={args.sigma_rep} "
          f"sigma_obs={args.sigma_obs} (transformed units)")

    results, dumps = {}, {}
    for path in paths:
        try:
            dump = load_dump(path)
        except Exception as error:                      # noqa: BLE001
            print(f"[skip] {path.name}: {error}", flush=True)
            continue
        name = dump["name"]
        dumps[name] = dump
        arms = distinct_arms(dump)
        eval_idx, assim_idx = dump["eval_idx"], dump["assim_idx"]
        print(f"[run] {name}: {len(arms)} distinct arm(s) {arms}, "
              f"{len(assim_idx)} assimilated / {len(eval_idx)} withheld, "
              f"{len(dump['time'])} day(s)", flush=True)

        withheld, assimilated, inflated = {}, {}, {}
        for arm in arms:
            block = dump[arm]
            withheld[arm] = score_arm(
                block[:, :, eval_idx].reshape(block.shape[0], -1),
                dump["gauge_mm"][:, eval_idx].ravel(),
                args.sigma_rep, transform, seed=args.seed,
            )
            assimilated[arm] = score_arm(
                block[:, :, assim_idx].reshape(block.shape[0], -1),
                dump["gauge_mm"][:, assim_idx].ravel(),
                args.sigma_rep, transform, seed=args.seed,
            )
            inflated[arm] = perturb_for_point_scale(
                block[:, :, eval_idx].reshape(block.shape[0], -1),
                args.sigma_rep, transform, seed=args.seed,
            )

        products = {}
        for product in PRODUCTS:
            series = dump.get(product)
            if series is None or not np.isfinite(series).any():
                continue
            products[product] = score_product(
                series[:, eval_idx].ravel(), dump["gauge_mm"][:, eval_idx].ravel()
            )

        results[name] = {
            "withheld": withheld,
            "assimilated": assimilated,
            "products": products,
            "innovation": innovation_statistics(
                dump, transform, args.sigma_obs, args.sigma_rep
            ),
            "_inflated": inflated,
        }
        plot_station_timeseries(dump, arms, out_dir / f"{name}_timeseries.png")

    if not results:
        raise SystemExit("no dumps could be read")

    print_provenance(dumps)
    print_tables(results)
    report_duplicate_runs(results)
    plot_innovation(results, out_dir / "innovation_consistency.png")
    plot_arm_comparison(results, out_dir / "arm_comparison.png")
    plot_rank_histograms(results, dumps, out_dir / "rank_histograms.png")

    payload = {
        name: {k: v for k, v in block.items() if not k.startswith("_")}
        for name, block in results.items()
    }
    out_json = out_dir / "gauge_truth_da.json"
    out_json.write_text(json.dumps(
        {"settings": {"sigma_rep": args.sigma_rep, "sigma_obs": args.sigma_obs},
         "runs": payload,
         "note": "Gauge is truth throughout. CHIRPS is not used as a reference "
                 "anywhere in this script."},
        indent=2, default=float,
    ))
    print()
    print(f"[done] wrote {out_json}")
    for figure in sorted(out_dir.glob("*.png")):
        print(f"[done] wrote {figure}")


if __name__ == "__main__":
    main()
