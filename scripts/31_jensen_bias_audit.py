#!/usr/bin/env python
"""Can Jensen's inequality alone account for the +10.3 mm/day background bias?

The question this settles
-------------------------
``scripts/30_observation_space_audit.py`` established, on 43,781 station-days,
that neither the training target nor the satellite is biased against the gauges:

    CHIRPS  +0.26 mm/day      IMERG  +0.29 mm/day      CPC  +1.32 mm/day

yet the background scores +10.30 and the IMERG-only analysis +9.88.  So roughly
ten millimetres a day appear somewhere between an unbiased label and a reported
score.  Before assuming the network learned a wet prior -- which costs a retrain
-- check the cheapest explanation, which costs nothing.

The mechanism
-------------
The model works in ``y = (log1p(p/eps) - mu) / sd`` and is inverted with
``p = eps * expm1(y*sd + mu)``.  That inverse is **convex**, so for an ensemble

    mean(inverse(y))  >  inverse(mean(y))

and the gap grows with ensemble spread.  Writing ``z = y*sd + mu`` for the
un-normalised log-space value, if ``z`` is roughly Gaussian across members with
spread ``sigma_z`` then

    E[p] + eps = (median(p) + eps) * exp(sigma_z**2 / 2)

so reporting the ensemble MEAN in mm inflates the field by ``exp(sigma_z^2/2)``
relative to its median, for free, with no error in the model at all.

This is not a subtle correction.  ``sigma_z = 1.4`` inflates by 2.6x.  On a
6.2 mm/day mean that is +10 mm/day -- the entire discrepancy.

Why this would explain the WHOLE result table, not just one row
---------------------------------------------------------------
If the mechanism is real then reported bias should track ensemble spread, and
every arm falls into line:

* background -- unconstrained, widest spread          -> largest inflation, +10.30
* IMERG only -- weak constraint (6.28x R inflation,
  stride-3 thinning), spread barely reduced           -> +9.88
* simultaneous -- gauges dominate, spread collapses   -> +1.58
* gauges only -- tightest constraint at stations      -> -0.65

On that reading the gauges are not "removing 106% of the bias" by supplying
information about rainfall amount.  They are shrinking the ensemble, which
shrinks a reporting artefact.  That is a very different claim from the one the
current draft makes, and it is testable: the ensemble MEDIAN is invariant under
any monotone transform, so if the story is right, median bias is small for every
arm while mean bias varies by 11 mm/day across them.

What to do with the answer
--------------------------
This does not invalidate CRPS, which uses the whole ensemble and is not a
functional of the mean.  It invalidates the bias column, and any deterministic
product formed by averaging members in mm space.  The fix is a reporting change
-- publish the median field, or average in transformed space and invert once --
not a retrain.

Usage
-----
    # analytic only: what spread would be needed to explain the observed bias
    python scripts/31_jensen_bias_audit.py \
        --stats data/processed/stats_cpc.json \
        --observed-bias 10.30 --mean-observed 6.19

    # with real ensembles: measure the gap directly
    python scripts/31_jensen_bias_audit.py \
        --stats data/processed/stats_cpc.json \
        --observed-bias 10.30 --mean-observed 6.19 \
        --dump data/processed/method_sweep/20240501_20240505_core.npz
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
    parser.add_argument("--stats", required=True, help="stats JSON holding precip_transform")
    parser.add_argument(
        "--observed-bias",
        type=float,
        default=10.30,
        help="the reported mean bias to be explained, mm/day",
    )
    parser.add_argument(
        "--mean-observed",
        type=float,
        default=6.19,
        help="mean gauge rainfall over the same sample, mm/day",
    )
    parser.add_argument(
        "--dump",
        default=None,
        help="optional sweep .npz with per-variant station ensembles, to measure "
        "the mean-versus-median gap directly instead of inferring it",
    )
    parser.add_argument("--out-json", default=None)
    return parser.parse_args()


# --------------------------------------------------------------------------
# Analytic core. Pure numpy, no model, unit tested.


def inflation_factor(sigma_z: float) -> float:
    """Ratio ``(E[p]+eps) / (median(p)+eps)`` for log-normal spread ``sigma_z``."""
    return float(np.exp(sigma_z**2 / 2.0))


def implied_mean(median_mm: float, sigma_z: float, eps: float) -> float:
    """Mean rainfall implied by a median field and an ensemble spread."""
    return (median_mm + eps) * inflation_factor(sigma_z) - eps


def spread_explaining_bias(bias_mm: float, mean_observed: float, eps: float) -> float:
    """Invert the relation: what ``sigma_z`` reproduces this bias?

    Assumes the ensemble median is unbiased, i.e. that the model is right and
    only the averaging is wrong. That is the hypothesis under test.
    """
    ratio = (mean_observed + bias_mm + eps) / (mean_observed + eps)
    if ratio <= 1.0:
        return 0.0
    return float(np.sqrt(2.0 * np.log(ratio)))


def round_trip_error(transform: PrecipTransform, values: np.ndarray) -> dict:
    """Is the transform actually invertible on this data? Rules out a plumbing bug."""
    back = transform.inverse(transform.forward(values))
    residual = np.asarray(back) - np.asarray(values)
    return {
        "max_abs_error_mm": float(np.max(np.abs(residual))),
        "mean_error_mm": float(np.mean(residual)),
        "is_clean": bool(np.max(np.abs(residual)) < 1e-3),
    }


def gap_from_ensemble(ensemble_mm: np.ndarray, observed_mm: np.ndarray) -> dict:
    """Mean-versus-median bias for a real ensemble. ``ensemble_mm`` is (M, N)."""
    finite = np.isfinite(observed_mm)
    if not finite.any():
        return {}
    members = np.asarray(ensemble_mm, float)[:, finite]
    truth = np.asarray(observed_mm, float)[finite]
    mean_field = members.mean(axis=0)
    median_field = np.median(members, axis=0)
    return {
        "n": int(finite.sum()),
        "mean_bias_mm": float(np.mean(mean_field - truth)),
        "median_bias_mm": float(np.mean(median_field - truth)),
        "jensen_gap_mm": float(np.mean(mean_field - median_field)),
        "mean_spread_mm": float(np.mean(members.std(axis=0))),
    }


def main() -> None:
    args = parse_args()
    stats = json.loads(Path(args.stats).read_text())
    transform = PrecipTransform.from_dict(stats["precip_transform"])
    eps = float(transform.eps)
    sd = float(transform.sd)

    print(f"[jensen] transform: kind={transform.kind} eps={eps} mu={transform.mu} sd={sd}")

    probe = np.array([0.0, 0.05, 0.5, 2.0, 10.0, 50.0, 200.0, 400.0], float)
    trip = round_trip_error(transform, probe)
    print(
        f"[jensen] round trip max |error| {trip['max_abs_error_mm']:.2e} mm -> "
        + ("clean, not a plumbing bug" if trip["is_clean"] else "BROKEN, fix this first")
    )
    if transform.kind != "log1p":
        print(
            f"[jensen] NOTE: transform is {transform.kind!r}, not log1p. The inverse "
            "is still convex for sqrt and cbrt, so the direction of the argument "
            "holds, but the closed form below is derived for log1p and the "
            "magnitudes will differ. Use --dump for the exact answer."
        )

    needed = spread_explaining_bias(args.observed_bias, args.mean_observed, eps)
    print()
    print(
        f"[jensen] to explain a {args.observed_bias:+.2f} mm/day mean bias on a "
        f"{args.mean_observed:.2f} mm/day observed mean, assuming the ensemble "
        "MEDIAN is unbiased, the required log-space ensemble spread is"
    )
    print(f"[jensen]     sigma_z = {needed:.3f}   (normalised: sigma_y = {needed / sd:.3f})")
    print(
        f"[jensen] i.e. the mean field would sit {inflation_factor(needed):.2f}x above "
        "the median field."
    )
    print()
    print("[jensen] inflation as a function of ensemble spread:")
    print(f"    {'sigma_z':>8} {'sigma_y':>8} {'x median':>9} {'implied mean':>13} {'implied bias':>13}")
    rows = []
    for sigma_z in (0.25, 0.5, 0.75, 1.0, 1.25, 1.4, 1.5, 1.75, 2.0):
        mean_mm = implied_mean(args.mean_observed, sigma_z, eps)
        bias = mean_mm - args.mean_observed
        flag = "  <-- matches the reported bias" if abs(bias - args.observed_bias) < 0.75 else ""
        print(
            f"    {sigma_z:>8.2f} {sigma_z / sd:>8.2f} {inflation_factor(sigma_z):>9.2f} "
            f"{mean_mm:>13.2f} {bias:>+13.2f}{flag}"
        )
        rows.append(
            {
                "sigma_z": sigma_z,
                "sigma_y": sigma_z / sd,
                "factor": inflation_factor(sigma_z),
                "implied_mean_mm": mean_mm,
                "implied_bias_mm": bias,
            }
        )

    report = {
        "transform": {"kind": transform.kind, "eps": eps, "mu": transform.mu, "sd": sd},
        "round_trip": trip,
        "observed_bias_mm": args.observed_bias,
        "mean_observed_mm": args.mean_observed,
        "sigma_z_explaining_bias": needed,
        "sigma_y_explaining_bias": needed / sd,
        "inflation_curve": rows,
    }

    # ---------------------------------------------------------------- dump
    if args.dump:
        archive = np.load(args.dump, allow_pickle=False)
        keys = [k for k in archive.files if k.endswith("_ensemble")]
        truth_key = next(
            (k for k in ("station_obs", "observations", "truth", "gauges") if k in archive.files),
            None,
        )
        if not keys or truth_key is None:
            print(
                f"\n[jensen] {args.dump} has no *_ensemble / observation arrays "
                f"(found {archive.files[:8]}...). Skipping the measured comparison.",
                flush=True,
            )
        else:
            observed = np.asarray(archive[truth_key], float).ravel()
            print(f"\n[jensen] measured from {args.dump}:")
            print(
                f"    {'variant':<28}{'mean bias':>11}{'median bias':>13}"
                f"{'Jensen gap':>12}{'spread':>9}"
            )
            measured = {}
            for key in sorted(keys):
                name = key[: -len("_ensemble")]
                members = np.asarray(archive[key], float)
                members = members.reshape(members.shape[0], -1)
                if members.shape[1] != observed.size:
                    continue
                entry = gap_from_ensemble(members, observed)
                if not entry:
                    continue
                measured[name] = entry
                print(
                    f"    {name:<28}{entry['mean_bias_mm']:>+11.2f}"
                    f"{entry['median_bias_mm']:>+13.2f}"
                    f"{entry['jensen_gap_mm']:>+12.2f}{entry['mean_spread_mm']:>9.2f}"
                )
            report["measured"] = measured
            if measured:
                spread_of_median = float(
                    np.std([e["median_bias_mm"] for e in measured.values()])
                )
                spread_of_mean = float(np.std([e["mean_bias_mm"] for e in measured.values()]))
                print()
                if spread_of_median < 0.5 * spread_of_mean:
                    print(
                        "[jensen] VERDICT: median bias is far more consistent across arms "
                        f"(sd {spread_of_median:.2f}) than mean bias (sd {spread_of_mean:.2f}). "
                        "The spread between arms is largely a Jensen artefact of averaging "
                        "in mm space, not a difference in physical skill. Report the median "
                        "field, or average in transformed space and invert once."
                    )
                else:
                    print(
                        "[jensen] VERDICT: median bias varies about as much as mean bias "
                        f"(sd {spread_of_median:.2f} versus {spread_of_mean:.2f}). The "
                        "differences between arms are NOT explained by the averaging "
                        "convention, so the wet bias is in the prior itself and a retrain "
                        "is on the table."
                    )

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(report, indent=2))
        print(f"\n[jensen] wrote {args.out_json}")


if __name__ == "__main__":
    main()
