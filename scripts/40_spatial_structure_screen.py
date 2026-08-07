#!/usr/bin/env python
"""Does the analysis look like a rainfall field? Structure, not error.

Point scores at gauges say whether the analysis has the right VALUE where the
37 stations are.  They say nothing about whether the field between the stations
is plausible, and a generative prior can score well at gauges while producing
spatially implausible fields --- overly smooth, overly speckled, or with the
wrong wet-area fraction.  Selecting a DA configuration on point scores alone
optimises 37 locations out of 16,384 cells.

This screens on structure instead, and the reference is deliberately NOT a
single product.  CHIRPS, IMERG and CPC disagree with each other substantially
(daily correlations against BMD gauges of 0.29, 0.78 and 0.79 respectively over
May--June 2024), so no one of them defines correct structure.  What they define
between them is a plausible RANGE.  The question asked here is whether the
analysis falls inside that envelope on each statistic, and the verdict is
reported as inside/outside rather than as a distance.

Statistics compared
-------------------
* **Radially averaged power spectrum**, and the effective resolution derived
  from it: the shortest wavelength at which the analysis still carries
  product-like power. A prior that blurs shows a spectrum falling away early;
  one that speckles shows excess power at the grid scale.
* **Wet-area fraction** at 1, 10 and 25 mm/day: the domain-scale form of the
  defect the prior redesign targeted.
* **Intensity quantiles** of the field (q50, q90, q99): amplitude structure
  independent of location.
* **Day-to-day variability**: the standard deviation of the domain mean across
  days, which distinguishes a field that varies realistically from one that
  reproduces a climatology.
* **Spatial pattern correlation** with each product, day by day. Low against
  all three means the analysis is placing rain somewhere none of them do.

Gauges remain the truth for VALUE; this adds the structural half that gauges,
being 37 points, cannot supply.

Example
-------
    python scripts/40_spatial_structure_screen.py \\
        --dumps 'data/processed/imerg_screen_may2024/*.npz' \\
        --arm combined --out-dir data/processed/structure_screen
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bdhires.eval.scale import effective_resolution, rapsd  # noqa: E402

PRODUCTS = {"CHIRPS": "chirps", "CPC": "condition", "IMERG": "imerg"}
ARM_FIELDS = {
    "background": "background",
    "gauges": "analysis_gauge",
    "satellite": "analysis_imerg",
    "combined": "analysis_combined",
}
THRESHOLDS = (1.0, 10.0, 25.0)


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 140, "savefig.dpi": 140, "font.size": 9,
        "axes.grid": True, "grid.alpha": 0.25,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    return plt


def load_fields(path: Path, arm: str) -> dict | None:
    """Gridded daily fields for one run: the arm's ensemble mean and each product.

    The satellite is stored on its coarse footprint grid and is expanded by
    nearest neighbour, never interpolated: a footprint is a box average, and
    smoothing it would manufacture exactly the small-scale structure this script
    is trying to measure.
    """
    z = np.load(path, allow_pickle=False)
    key = ARM_FIELDS.get(arm)
    if key is None or key not in z:
        return None
    analysis = np.nanmean(z[key], axis=1)               # (T, H, W)
    valid = np.asarray(z["valid"]) > 0 if "valid" in z else None
    nlat, nlon = analysis.shape[-2:]

    fields = {"analysis": analysis}
    for label, k in PRODUCTS.items():
        if k not in z:
            continue
        field = np.asarray(z[k], dtype=np.float64)
        if field.shape[-2:] != (nlat, nlon):
            factor = nlat // field.shape[-2]
            field = np.repeat(np.repeat(field, factor, axis=-2), factor, axis=-1)
            field = field[..., :nlat, :nlon]
        if field.shape[0] != analysis.shape[0] or not np.isfinite(field).any():
            continue
        fields[label] = field
    return {"name": path.stem, "fields": fields, "valid": valid,
            "time": z["time"].astype("datetime64[ns]").astype("datetime64[D]")}


def structure_stats(field: np.ndarray, valid: np.ndarray | None) -> dict:
    """Domain-scale statistics of a daily field stack, ocean excluded."""
    if valid is None:
        valid = np.ones(field.shape[-2:], bool)
    land = field[:, valid]
    finite = np.isfinite(land)
    if not finite.any():
        return {}
    daily_mean = np.array([np.nanmean(d) for d in land])
    out = {
        "domain_mean": float(np.nanmean(land)),
        "day_to_day_sd": float(np.nanstd(daily_mean, ddof=1))
        if len(daily_mean) > 1 else float("nan"),
        "q50": float(np.nanpercentile(land[finite], 50)),
        "q90": float(np.nanpercentile(land[finite], 90)),
        "q99": float(np.nanpercentile(land[finite], 99)),
        "spatial_sd": float(np.nanmean([np.nanstd(d) for d in land])),
    }
    for t in THRESHOLDS:
        out[f"wet_area_{t:g}"] = float(np.nanmean(land >= t))
    return out


def pattern_correlation(a: np.ndarray, b: np.ndarray, valid) -> float:
    """Mean day-by-day spatial correlation between two field stacks."""
    if valid is None:
        valid = np.ones(a.shape[-2:], bool)
    values = []
    for x, y in zip(a, b):
        u, v = x[valid], y[valid]
        ok = np.isfinite(u) & np.isfinite(v)
        if ok.sum() > 10 and u[ok].std() > 0 and v[ok].std() > 0:
            values.append(float(np.corrcoef(u[ok], v[ok])[0, 1]))
    return float(np.mean(values)) if values else float("nan")


def spectra(fields: dict, valid) -> dict:
    """Day-averaged RAPSD per field, plus the analysis-to-product power ratio."""
    if valid is None:
        valid = np.ones(next(iter(fields.values())).shape[-2:], bool)
    mask = valid.astype(np.float32)
    out = {}
    for label, stack in fields.items():
        powers, wavelength = [], None
        for day in stack:
            if not np.isfinite(day).any():
                continue
            try:
                w, p = rapsd(np.nan_to_num(day), mask)
            except Exception:                                   # noqa: BLE001
                continue
            if p is None or not np.isfinite(p).any():
                continue
            wavelength = w
            powers.append(p)
        if powers and wavelength is not None:
            out[label] = {"wavelength_km": np.asarray(wavelength),
                          "power": np.nanmean(np.stack(powers), axis=0)}
    return out


def envelope_verdict(value: float, product_values: list[float]) -> str:
    """Inside or outside the range the three products span.

    The products are not truth and disagree with each other, so the honest
    target is their envelope. 'below'/'above' is reported rather than a distance
    because a distance would imply a reference point that does not exist.
    """
    finite = [v for v in product_values if np.isfinite(v)]
    if not finite or not np.isfinite(value):
        return "n/a"
    low, high = min(finite), max(finite)
    if value < low:
        return f"BELOW  (products {low:.3g}-{high:.3g})"
    if value > high:
        return f"ABOVE  (products {low:.3g}-{high:.3g})"
    return f"inside (products {low:.3g}-{high:.3g})"


# --------------------------------------------------------------------------


def plot_screen(runs: dict, out_path: Path) -> None:
    """Spectra, wet-area fractions and pattern correlations across runs."""
    plt = _style()
    n = len(runs)
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 9.0))

    # --- spectra of every run's analysis, with the product envelope shaded
    axis = axes[0][0]
    reference = next(iter(runs.values()))["spectra"]
    products = [k for k in PRODUCTS if k in reference]
    if products:
        wl = reference[products[0]]["wavelength_km"]
        stack = np.stack([reference[p]["power"] for p in products])
        axis.fill_between(wl, np.nanmin(stack, 0), np.nanmax(stack, 0),
                          color="#999999", alpha=0.30, lw=0,
                          label="product envelope")
        for p in products:
            axis.plot(wl, reference[p]["power"], lw=0.9, ls=":", alpha=0.8, label=p)
    for name, block in runs.items():
        if "analysis" in block["spectra"]:
            s = block["spectra"]["analysis"]
            axis.plot(s["wavelength_km"], s["power"], lw=1.8, label=name)
    axis.set_xscale("log"); axis.set_yscale("log")
    axis.invert_xaxis()
    axis.set_xlabel("wavelength (km)"); axis.set_ylabel("power")
    axis.set_title("Radially averaged spectrum\nbelow the envelope = too smooth, "
                   "above = too speckled", fontsize=9)
    axis.legend(fontsize=6.5, ncol=2)

    # --- wet-area fraction
    axis = axes[0][1]
    width = 0.8 / max(len(runs) + len(products), 1)
    for i, t in enumerate(THRESHOLDS):
        vals = [runs[n]["stats"]["analysis"].get(f"wet_area_{t:g}", np.nan) for n in runs]
        pv = [reference and runs[list(runs)[0]]["stats"].get(p, {}).get(f"wet_area_{t:g}", np.nan)
              for p in products]
        for j, (label, v) in enumerate(list(zip(runs, vals)) + list(zip(products, pv))):
            axis.bar(i + j * width, v, width=width,
                     color="#c1440e" if j < len(runs) else "#999999",
                     label=label if i == 0 else None)
    axis.set_xticks(np.arange(len(THRESHOLDS)) + 0.4 - width / 2)
    axis.set_xticklabels([f">={t:g} mm" for t in THRESHOLDS])
    axis.set_ylabel("fraction of land cells")
    axis.set_title("Wet-area fraction (grey = products)", fontsize=9)
    axis.legend(fontsize=6.5, ncol=2)

    # --- pattern correlation against each product
    axis = axes[1][0]
    labels = list(runs)
    for i, p in enumerate(products):
        vals = [runs[n]["pattern"].get(p, np.nan) for n in labels]
        axis.bar(np.arange(len(labels)) + i * 0.25, vals, width=0.25, label=p)
    axis.set_xticks(np.arange(len(labels)) + 0.25)
    axis.set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
    axis.set_ylabel("mean daily spatial correlation")
    axis.set_title("Pattern agreement with each product\n"
                   "low against ALL THREE means the rain is in the wrong place",
                   fontsize=9)
    axis.legend(fontsize=7.5)

    # --- amplitude structure
    axis = axes[1][1]
    keys = ["q50", "q90", "q99", "spatial_sd", "day_to_day_sd"]
    for i, k in enumerate(keys):
        vals = [runs[n]["stats"]["analysis"].get(k, np.nan) for n in labels]
        axis.plot(vals, np.full(len(vals), i), "o", color="#c1440e", ms=7,
                  label="analysis" if i == 0 else None)
        pv = [runs[labels[0]]["stats"].get(p, {}).get(k, np.nan) for p in products]
        finite = [v for v in pv if np.isfinite(v)]
        if finite:
            axis.plot([min(finite), max(finite)], [i, i], "-", color="#999999",
                      lw=6, alpha=0.5, zorder=0,
                      label="product range" if i == 0 else None)
    axis.set_yticks(range(len(keys)))
    axis.set_yticklabels(keys)
    axis.set_xlabel("mm/day")
    axis.set_title("Amplitude structure against the product range", fontsize=9)
    axis.legend(fontsize=7.5)

    figure.suptitle("Structure screen: does the field look like rainfall?", y=1.0)
    figure.tight_layout()
    figure.savefig(out_path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Screen DA configurations on spatial structure, not error",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dumps", nargs="+", required=True)
    parser.add_argument("--arm", default="combined", choices=sorted(ARM_FIELDS))
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    paths = sorted({Path(p) for pat in args.dumps for p in (glob.glob(pat) or [pat])})
    paths = [p for p in paths if p.suffix == ".npz" and p.exists()]
    if not paths:
        raise SystemExit(f"no NPZ dumps matched {args.dumps}")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    runs = {}
    for path in paths:
        block = load_fields(path, args.arm)
        if block is None:
            print(f"[skip] {path.name}: no gridded field for arm {args.arm!r}")
            continue
        valid = block["valid"]
        stats = {k: structure_stats(v, valid) for k, v in block["fields"].items()}
        pattern = {p: pattern_correlation(block["fields"]["analysis"],
                                          block["fields"][p], valid)
                   for p in PRODUCTS if p in block["fields"]}
        runs[block["name"]] = {"stats": stats, "pattern": pattern,
                               "spectra": spectra(block["fields"], valid),
                               "n_days": len(block["time"])}
        print(f"[run] {block['name']}: {len(block['time'])} day(s), "
              f"products {sorted(p for p in PRODUCTS if p in block['fields'])}")

    if not runs:
        raise SystemExit("nothing to screen")

    products = sorted({p for r in runs.values() for p in r["pattern"]})
    print()
    print("[structure] analysis against the product envelope. Gauges remain the")
    print("    truth for VALUE; this is whether the FIELD is plausible.")
    keys = ["domain_mean", "q90", "q99", "spatial_sd", "day_to_day_sd"] + \
           [f"wet_area_{t:g}" for t in THRESHOLDS]
    for name, block in runs.items():
        print(f"\n  {name}")
        for k in keys:
            v = block["stats"]["analysis"].get(k, np.nan)
            pv = [block["stats"].get(p, {}).get(k, np.nan) for p in products]
            print(f"    {k:16s} {v:9.3g}   {envelope_verdict(v, pv)}")
        print("    pattern correlation: " +
              "  ".join(f"{p} {block['pattern'].get(p, float('nan')):.2f}"
                        for p in products))
        spec = block["spectra"]
        if "analysis" in spec and products and products[0] in spec:
            ratio = spec["analysis"]["power"] / np.clip(
                spec[products[0]]["power"], 1e-30, None)
            res = effective_resolution(spec["analysis"]["wavelength_km"], ratio)
            print(f"    effective resolution vs {products[0]}: {res:.0f} km")

    plot_screen(runs, out_dir / "structure_screen.png")
    payload = {n: {"stats": b["stats"], "pattern": b["pattern"],
                   "n_days": b["n_days"]} for n, b in runs.items()}
    (out_dir / "structure_screen.json").write_text(
        json.dumps(payload, indent=2, default=float))
    print(f"\n[done] wrote {out_dir / 'structure_screen.json'}")
    print(f"[done] wrote {out_dir / 'structure_screen.png'}")


if __name__ == "__main__":
    main()
