#!/usr/bin/env python
"""Where does the prior's +6.40 mm/day MEDIAN wet bias come from?

What is already ruled out
-------------------------
* the training target -- CHIRPS is +0.26 mm/day against 43,781 gauge-days
  (``scripts/30_observation_space_audit.py``);
* the satellite -- raw IMERG is +0.29 against the same gauges;
* the averaging convention -- the bias survives the ensemble MEDIAN, so it is
  not the Jensen artefact that inflates the mean by a further 2-6.6 mm/day
  (``scripts/31_jensen_bias_audit.py``).

So the prior does not reproduce its own training target, and the gap is real.
This script asks where it enters, using only dumps that already exist.

The design under test
---------------------
The network predicts a RESIDUAL against a coarse base:

    target    = T(CHIRPS) - T(base),     base = cpc_precip
    rebuild   = T_inv( predicted_residual + T(base) )

CPC is wet: it rains on 49% of station-days against an observed 31%
(script 30). So ``T(base)`` is non-zero over far more of the domain than
``T(CHIRPS)`` is, and the true residual must be strongly NEGATIVE wherever CPC
rains and CHIRPS does not. Cancelling that floor is something the network has to
actively learn. If it under-predicts those negative residuals -- which is
exactly what a loss dominated by wet cases, or wet-day oversampling at 35% with
no importance reweighting, would cause -- the reconstruction keeps part of the
CPC floor and comes out wet everywhere.

Three numbers separate the candidates
-------------------------------------
1. **The zero-residual counterfactual.** If the network predicted nothing, the
   output would be exactly CPC. That bias is an upper bound on how much the
   floor alone can contribute. If it is much smaller than +6.40, the floor is
   not the whole story no matter how badly the residual is predicted.
2. **Residual bias in transformed space**, overall and stratified by the TRUE
   residual. A floor failure shows up as a large positive bias concentrated in
   the most negative true-residual bins -- the network refusing to subtract.
3. **Wet-frequency inheritance.** If the reconstruction's wet fraction tracks
   CPC's rather than CHIRPS's, the floor is leaking through directly.

Usage
-----
    python scripts/32_prior_wet_bias_audit.py \
        --stats data/processed/stats_cpc.json \
        --dump data/processed/bmd_imerg_eval_2021_may_sep/fold*.npz \
               data/processed/bmd_imerg_eval_2024_may_jun/fold*.npz \
        --out-json data/processed/prior_wet_bias_audit.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bdhires.transforms import PrecipTransform  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--dump", nargs="+", required=True, help="per-fold eval .npz files")
    parser.add_argument("--wet-threshold", type=float, default=1.0, help="mm/day")
    parser.add_argument(
        "--background-lag",
        type=int,
        default=None,
        help="compare background day i against CHIRPS day i+LAG. The dumps carry "
        "BOTH 'time' and 'background_time'; with BACKGROUND_DAY_OFFSET=-1 those "
        "differ, so indexing the two arrays positionally compares fields a day "
        "apart and destroys the residual correlation. Default: take the offset "
        "from the stored time axes, or fall back to the lag that maximises "
        "pattern correlation.",
    )
    parser.add_argument(
        "--max-days",
        type=int,
        default=400,
        help="cap on pooled days, to keep memory sane on the full archive",
    )
    parser.add_argument("--out-json", default=None)
    return parser.parse_args()


def wet_fraction(values: np.ndarray, threshold: float) -> float:
    finite = np.isfinite(values)
    return float((values[finite] >= threshold).mean()) if finite.any() else float("nan")


def summarise(predicted: np.ndarray, truth: np.ndarray, threshold: float) -> dict:
    ok = np.isfinite(predicted) & np.isfinite(truth)
    if not ok.any():
        return {}
    a, b = predicted[ok], truth[ok]
    return {
        "n": int(ok.sum()),
        "bias_mm": float(np.mean(a - b)),
        "mae_mm": float(np.mean(np.abs(a - b))),
        "mean_pred_mm": float(a.mean()),
        "mean_truth_mm": float(b.mean()),
        "wet_fraction_pred": wet_fraction(a, threshold),
        "wet_fraction_truth": wet_fraction(b, threshold),
    }


def main() -> None:
    args = parse_args()
    stats = json.loads(Path(args.stats).read_text())
    transform = PrecipTransform.from_dict(stats["precip_transform"])

    backgrounds, chirps_all, base_all = [], [], []
    days = 0
    for path in args.dump:
        archive = np.load(path, allow_pickle=False)
        needed = {"background", "chirps", "condition"}
        if not needed.issubset(archive.files):
            print(f"[prior] {path}: missing {sorted(needed - set(archive.files))}, skipping")
            continue
        members = np.asarray(archive["background"], np.float32)  # (T, M, H, W)
        chirps = np.asarray(archive["chirps"], np.float32)  # (T, H, W)
        base = np.asarray(archive["condition"], np.float32)  # (T, H, W)
        valid = (
            np.asarray(archive["valid"], bool)
            if "valid" in archive.files
            else np.ones(chirps.shape[1:], bool)
        )
        # Median over members: the spread-independent central field.
        median_field = np.median(members, axis=1)
        median_field[:, ~valid] = np.nan
        chirps = chirps.copy()
        chirps[:, ~valid] = np.nan
        base = base.copy()
        base[:, ~valid] = np.nan

        # Align the background to CHIRPS before anything else.
        #
        # The dump stores 'time' (the CHIRPS/analysis day) and 'background_time'
        # separately, because BACKGROUND_DAY_OFFSET shifts the conditioning.
        # Indexing both positionally compares fields from different days, which
        # collapses the residual correlation to near zero and makes a perfectly
        # good network look as though it learned nothing. Trust the stored axes.
        lag = args.background_lag
        if lag is None and {"time", "background_time"}.issubset(archive.files):
            t_main = np.asarray(archive["time"]).astype("datetime64[ns]").astype("datetime64[D]")
            t_bg = (
                np.asarray(archive["background_time"])
                .astype("datetime64[ns]")
                .astype("datetime64[D]")
            )
            offsets = (t_bg - t_main).astype("timedelta64[D]").astype(int)
            if len(set(offsets.tolist())) == 1:
                lag = int(offsets[0])
        if lag is None:
            lag = 0
        if lag:
            # background[i] describes the day CHIRPS calls i+lag
            if lag > 0:
                median_field = median_field[:-lag]
                chirps, base = chirps[lag:], base[lag:]
            else:
                median_field = median_field[-lag:]
                chirps, base = chirps[:lag], base[:lag]
        if path == args.dump[0]:
            print(f"[prior] background-to-CHIRPS lag: {lag:+d} day(s)")

        take = min(len(chirps), max(args.max_days - days, 0))
        if take <= 0:
            break
        backgrounds.append(median_field[:take])
        chirps_all.append(chirps[:take])
        base_all.append(base[:take])
        days += take

    if not backgrounds:
        raise SystemExit("no usable dumps: need background, chirps and condition arrays")

    background = np.concatenate(backgrounds)
    chirps = np.concatenate(chirps_all)
    base = np.concatenate(base_all)
    print(f"[prior] pooled {len(background)} days on the {background.shape[1]}x{background.shape[2]} grid")

    report: dict = {"n_days": int(len(background))}

    # Independent check that the alignment above is right. Pattern correlation
    # against CHIRPS should peak at 0 once aligned; if it peaks elsewhere the
    # residual diagnostics below are measuring a day of weather, not model error.
    print("\n[prior] background-versus-CHIRPS pattern correlation by residual lag:")
    lag_corr = {}
    for probe in (-2, -1, 0, 1, 2):
        if probe > 0:
            a, b = background[:-probe], chirps[probe:]
        elif probe < 0:
            a, b = background[-probe:], chirps[:probe]
        else:
            a, b = background, chirps
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 1000:
            continue
        lag_corr[probe] = float(np.corrcoef(a[ok], b[ok])[0, 1])
    best = max(lag_corr, key=lag_corr.get) if lag_corr else 0
    print(
        "    "
        + "  ".join(
            f"{probe:+d}: {value:.3f}{'*' if probe == best else ''}"
            for probe, value in sorted(lag_corr.items())
        )
    )
    report["residual_lag_correlation"] = lag_corr
    if best != 0:
        print(
            f"[prior] WARNING: correlation still peaks at {best:+d}, so the fields are "
            "NOT aligned. Re-run with --background-lag adjusted; everything below is "
            "unreliable until this reads 0."
        )

    # ---------------------------------------------------------------- (1)
    print("\n[prior] on the model grid, against CHIRPS:")
    print(f"    {'field':<34}{'bias':>9}{'MAE':>9}{'mean':>9}{'wet frac':>10}")
    rows = {
        "background (ensemble median)": background,
        "CPC base (zero-residual case)": base,
    }
    for name, field in rows.items():
        entry = summarise(field, chirps, args.wet_threshold)
        report[name] = entry
        print(
            f"    {name:<34}{entry['bias_mm']:>+9.2f}{entry['mae_mm']:>9.2f}"
            f"{entry['mean_pred_mm']:>9.2f}{entry['wet_fraction_pred']:>10.3f}"
        )
    chirps_wet = wet_fraction(chirps, args.wet_threshold)
    print(f"    {'CHIRPS (target)':<34}{0.0:>+9.2f}{0.0:>9.2f}{np.nanmean(chirps):>9.2f}{chirps_wet:>10.3f}")
    report["chirps_wet_fraction"] = chirps_wet

    floor_bias = report["CPC base (zero-residual case)"]["bias_mm"]
    prior_bias = report["background (ensemble median)"]["bias_mm"]
    print(
        f"\n[prior] zero-residual counterfactual: if the network predicted nothing the "
        f"output would be CPC, biased {floor_bias:+.2f} mm/day against CHIRPS."
    )
    if abs(prior_bias) > abs(floor_bias) + 1.0:
        print(
            f"[prior] the prior is {prior_bias:+.2f}, i.e. WETTER than doing nothing. "
            "The network is not merely failing to remove the CPC floor -- it is adding "
            "rain of its own. A base or floor fix cannot explain this."
        )
    else:
        print(
            f"[prior] the prior is {prior_bias:+.2f}, within reach of the floor. Failing "
            "to cancel the CPC base is a sufficient explanation; fix the base or the "
            "residual target before considering a retrain."
        )

    # ---------------------------------------------------------------- (2)
    # Residual space is where the network actually learns, so look there.
    t_bg = np.asarray(transform.forward(np.nan_to_num(background, nan=0.0)), float)
    t_ch = np.asarray(transform.forward(np.nan_to_num(chirps, nan=0.0)), float)
    t_base = np.asarray(transform.forward(np.nan_to_num(base, nan=0.0)), float)
    good = np.isfinite(background) & np.isfinite(chirps) & np.isfinite(base)
    pred_residual = (t_bg - t_base)[good]
    true_residual = (t_ch - t_base)[good]

    report["residual"] = {
        "pred_mean": float(pred_residual.mean()),
        "true_mean": float(true_residual.mean()),
        "bias": float(np.mean(pred_residual - true_residual)),
        "corr": float(np.corrcoef(pred_residual, true_residual)[0, 1]),
    }
    print(
        f"\n[prior] residual space (transformed): predicted mean "
        f"{pred_residual.mean():+.3f} versus true {true_residual.mean():+.3f}, "
        f"bias {report['residual']['bias']:+.3f}, corr {report['residual']['corr']:.3f}"
    )

    print("\n[prior] residual bias stratified by the TRUE residual:")
    print(f"    {'true residual bin':<22}{'n':>10}{'true':>9}{'pred':>9}{'bias':>9}")
    edges = np.nanquantile(true_residual, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    strata = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        inside = (true_residual >= lower) & (true_residual < upper)
        if inside.sum() < 100:
            continue
        entry = {
            "lower": float(lower),
            "upper": float(upper),
            "n": int(inside.sum()),
            "true_mean": float(true_residual[inside].mean()),
            "pred_mean": float(pred_residual[inside].mean()),
            "bias": float(np.mean(pred_residual[inside] - true_residual[inside])),
        }
        strata.append(entry)
        print(
            f"    [{lower:>+6.2f}, {upper:>+6.2f})  {entry['n']:>10d}"
            f"{entry['true_mean']:>9.2f}{entry['pred_mean']:>9.2f}{entry['bias']:>+9.2f}"
        )
    report["residual_strata"] = strata

    if strata:
        driest = strata[0]
        print()
        if driest["bias"] > 0.5 * abs(driest["true_mean"]) and driest["true_mean"] < -0.2:
            print(
                "[prior] VERDICT: in the most negative true-residual bin -- where CPC "
                f"rains and CHIRPS does not -- the network should predict "
                f"{driest['true_mean']:.2f} and predicts {driest['pred_mean']:.2f}. It is "
                "not subtracting the CPC floor. That is the wet bias, and it is a "
                "TARGET-DESIGN problem: either change the base, drop the residual "
                "formulation, or reweight the loss so dry-over-wet cases are not "
                "swamped. Wet-day oversampling at 35% with no importance reweighting "
                "is the first thing to check."
            )
        else:
            print(
                "[prior] VERDICT: the network tracks the true residual in the dry-CHIRPS "
                "regime, so the CPC floor is not leaking through. The wet bias is in the "
                "learned conditional distribution itself. Check wet-day oversampling and "
                "the loss weighting; this one needs a retrain."
            )

    # ---------------------------------------------------------------- (4)
    # Grid-mean bias and station-located bias disagree, so decompose the chain
    # at the stations themselves. On the grid the prior is +2.29 against CHIRPS
    # and CHIRPS is +0.26 against the gauges, which predicts about +2.55 against
    # the gauges -- but the measured station number is +6.40. Something is
    # location-specific, and only a station-space comparison separates "the model
    # is worse where the gauges are" from "the reference changes".
    print("\n[prior] station-space decomposition (ensemble median, withheld stations):")
    legs = {"background_minus_chirps": [], "chirps_minus_gauge": [], "background_minus_gauge": []}
    for path in args.dump:
        archive = np.load(path, allow_pickle=False)
        need = {"background_at_stations", "chirps_at_stations", "gauge_mm"}
        if not need.issubset(archive.files):
            continue
        keep = (
            np.asarray(archive["eval_idx"], int)
            if "eval_idx" in archive.files
            else slice(None)
        )
        bg = np.median(np.asarray(archive["background_at_stations"], float), axis=1)[:, keep]
        ch = np.asarray(archive["chirps_at_stations"], float)[:, keep]
        ga = np.asarray(archive["gauge_mm"], float)[:, keep]
        ok = np.isfinite(bg) & np.isfinite(ch) & np.isfinite(ga)
        if not ok.any():
            continue
        legs["background_minus_chirps"].append((bg - ch)[ok])
        legs["chirps_minus_gauge"].append((ch - ga)[ok])
        legs["background_minus_gauge"].append((bg - ga)[ok])

    if any(legs.values()):
        station_report = {}
        for name, parts in legs.items():
            pooled = np.concatenate(parts)
            station_report[name] = {"n": int(pooled.size), "bias_mm": float(pooled.mean())}
            print(f"    {name:<28}{pooled.mean():>+9.2f} mm/day   (n={pooled.size})")
        report["station_decomposition"] = station_report

        at_station = station_report["background_minus_chirps"]["bias_mm"]
        on_grid = report["background (ensemble median)"]["bias_mm"]
        print()
        if abs(at_station - on_grid) > 1.5:
            print(
                f"[prior] The prior is {at_station:+.2f} against CHIRPS AT STATIONS but "
                f"{on_grid:+.2f} on the full grid. The wet bias is concentrated where the "
                "gauges are, not spread over the domain. BMD stations sit in the plains "
                "and away from the Meghalaya barrier, so check whether the model is "
                "over-raining the low-lying interior specifically -- a domain-mean "
                "diagnostic will keep hiding this."
            )
        else:
            print(
                f"[prior] The prior is {at_station:+.2f} against CHIRPS at stations versus "
                f"{on_grid:+.2f} on the grid -- consistent. The wet bias is uniform, so "
                "the station-located number is not a sampling artefact."
            )

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(report, indent=2))
        print(f"\n[prior] wrote {args.out_json}")


if __name__ == "__main__":
    main()
