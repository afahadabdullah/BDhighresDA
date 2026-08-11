#!/usr/bin/env python
"""Compare single-day OSSE variants side by side: does the reconstruction improve?

Reads every ``<root>/<variant>/ensemble.npz`` written by
``slurm/osse_single_day.sbatch`` and puts one row per variant against a shared
colour scale, so the question "is this actually a better field" is answered by
looking rather than by reading six JSONs.

The metrics are chosen to separate two things that a single RMSE hides:

**Does the analysis get the large scale right?** ``field_rmse`` and
``field_corr`` over the whole 0.05 degree field.

**Does it place structure, or just pin the gauges?** ``subgrid_corr`` is the
correlation of the residual after removing each field's own 0.1 degree block
mean, and ``bullseye`` is the ratio of the analysis increment's amplitude
within one cell of an assimilated gauge to its amplitude more than five cells
away. A method that only pins observations scores high on ``bullseye`` and near
zero on ``subgrid_corr``; that is the failure this sweep exists to fix, and the
number makes it explicit instead of leaving it to the eye.

Withheld-gauge RMSE is reported too, but on ONE day with eight stations it is
eight numbers and should not decide anything on its own.

Example
-------
    python scripts/48_single_day_compare.py \\
        --root data/processed/osse_day_20230319 --day 2023-03-19
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PREFERRED = ["base", "s6", "s12", "s20", "g1e-2", "temp15", "s12temp15"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True)
    parser.add_argument("--day", default="")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--factor", type=int, default=2,
                        help="footprint factor for the subgrid residual")
    return parser.parse_args()


def discover(root: Path) -> list[tuple[str, Path]]:
    found = {d.name: d for d in sorted(root.iterdir())
             if d.is_dir() and (d / "ensemble.npz").exists()}
    ordered = [(n, found.pop(n)) for n in PREFERRED if n in found]
    ordered += sorted(found.items())
    if not ordered:
        raise SystemExit(f"no <variant>/ensemble.npz under {root}")
    return ordered


def block_mean(field: np.ndarray, factor: int) -> np.ndarray:
    """Upsampled block mean, so the residual has the field's own shape."""
    h, w = field.shape[-2:]
    nh, nw = h // factor, w // factor
    core = field[..., : nh * factor, : nw * factor]
    coarse = core.reshape(*core.shape[:-2], nh, factor, nw, factor).mean(axis=(-3, -1))
    up = np.repeat(np.repeat(coarse, factor, axis=-2), factor, axis=-1)
    out = np.array(field, dtype=float)
    out[..., : nh * factor, : nw * factor] = up
    return out


def corr(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10 or a[ok].std() == 0 or b[ok].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def load(path: Path, factor: int) -> dict:
    z = np.load(path, allow_pickle=False)
    # (T, M, H, W) -> the single day, ensemble mean
    background = np.asarray(z["background"], float)
    analysis = np.asarray(z["analysis"], float)
    truth = np.asarray(z["truth"], float)
    if background.ndim == 4:
        background, analysis, truth = background[0], analysis[0], truth[0]
    bg_mean = background.mean(axis=0) if background.ndim == 3 else background
    an_mean = analysis.mean(axis=0) if analysis.ndim == 3 else analysis
    an_member = analysis[0] if analysis.ndim == 3 else analysis
    truth = truth if truth.ndim == 2 else truth.mean(axis=0)

    valid = np.asarray(z["valid"]) > 0 if "valid" in z else np.ones(truth.shape, bool)
    lat = np.asarray(z["station_lat"], float) if "station_lat" in z else np.array([])
    lon = np.asarray(z["station_lon"], float) if "station_lon" in z else np.array([])
    assim = np.asarray(z["assim_idx"], int) if "assim_idx" in z else np.array([], int)
    evl = np.asarray(z["eval_idx"], int) if "eval_idx" in z else np.array([], int)
    glat = np.asarray(z["grid_lat"], float) if "grid_lat" in z else None
    glon = np.asarray(z["grid_lon"], float) if "grid_lon" in z else None

    residual = lambda f: f - block_mean(f, factor)
    metrics = {
        "field_rmse": float(np.sqrt(np.nanmean((an_mean - truth)[valid] ** 2))),
        "background_rmse": float(np.sqrt(np.nanmean((bg_mean - truth)[valid] ** 2))),
        "field_corr": corr(an_mean[valid], truth[valid]),
        "background_corr": corr(bg_mean[valid], truth[valid]),
        "subgrid_corr": corr(residual(an_mean)[valid], residual(truth)[valid]),
        "background_subgrid_corr": corr(residual(bg_mean)[valid],
                                        residual(truth)[valid]),
        "analysis_spread": float(np.nanmean(analysis.std(axis=0)[valid]))
        if analysis.ndim == 3 else float("nan"),
    }
    metrics["rmse_gain_percent"] = 100.0 * (
        metrics["background_rmse"] - metrics["field_rmse"]
    ) / max(metrics["background_rmse"], 1e-9)

    # Bullseye: increment amplitude near an assimilated gauge versus far from
    # any. A method that only pins observations scores high here.
    increment = np.abs(an_mean - bg_mean)
    if glat is not None and glon is not None and assim.size:
        yy = np.abs(glat[:, None] - lat[assim][None, :]).argmin(axis=0)
        xx = np.abs(glon[:, None] - lon[assim][None, :]).argmin(axis=0)
        near = np.zeros(truth.shape, bool)
        for r, c in zip(yy, xx):
            near[max(0, r - 1):r + 2, max(0, c - 1):c + 2] = True
        far = np.zeros(truth.shape, bool)
        for r, c in zip(yy, xx):
            far[max(0, r - 5):r + 6, max(0, c - 5):c + 6] = True
        far = valid & ~far
        near &= valid
        if near.any() and far.any():
            metrics["bullseye"] = float(
                np.nanmean(increment[near]) / max(np.nanmean(increment[far]), 1e-9))
        else:
            metrics["bullseye"] = float("nan")
    else:
        metrics["bullseye"] = float("nan")

    return {"truth": truth, "background": bg_mean, "analysis": an_mean,
            "member": an_member, "valid": valid, "metrics": metrics,
            "lat": lat, "lon": lon, "assim": assim, "eval": evl,
            "glat": glat, "glon": glon}


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir or root)
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = discover(root)
    data = [(name, load(path / "ensemble.npz", args.factor))
            for name, path in variants]

    from bdhires.paper import PALETTE, use_paper_style  # noqa: F401
    plt = use_paper_style()

    truth = data[0][1]["truth"]
    valid = data[0][1]["valid"]
    top = float(np.nanpercentile(truth[valid], 99.5))
    elim = max(1.0, float(np.nanpercentile(
        np.abs(data[0][1]["analysis"] - truth)[valid], 99)))

    rows = len(data)
    figure, axes = plt.subplots(rows, 5, figsize=(15.0, 2.75 * rows),
                                squeeze=False)
    for r, (name, d) in enumerate(data):
        m = d["metrics"]
        show = lambda a: np.where(d["valid"], a, np.nan)
        panels = [
            ("truth", show(d["truth"]), dict(cmap="viridis", vmin=0, vmax=top)),
            ("background mean", show(d["background"]),
             dict(cmap="viridis", vmin=0, vmax=top)),
            ("analysis mean", show(d["analysis"]),
             dict(cmap="viridis", vmin=0, vmax=top)),
            ("analysis - truth", show(d["analysis"] - d["truth"]),
             dict(cmap="RdBu_r", vmin=-elim, vmax=elim)),
            ("increment |an - bg|", show(np.abs(d["analysis"] - d["background"])),
             dict(cmap="magma", vmin=0, vmax=max(1.0, top / 3))),
        ]
        for c, (title, field, kw) in enumerate(panels):
            axis = axes[r][c]
            axis.imshow(field, origin="lower", **kw)
            axis.set_xticks([]); axis.set_yticks([]); axis.grid(False)
            if r == 0:
                axis.set_title(title, fontsize=9)
            if c in (2, 4) and d["glat"] is not None and d["assim"].size:
                yy = np.abs(d["glat"][:, None] - d["lat"][d["assim"]][None, :]).argmin(axis=0)
                xx = np.abs(d["glon"][:, None] - d["lon"][d["assim"]][None, :]).argmin(axis=0)
                axis.plot(xx, yy, "k.", ms=2.5)
                if d["eval"].size:
                    ey = np.abs(d["glat"][:, None] - d["lat"][d["eval"]][None, :]).argmin(axis=0)
                    ex = np.abs(d["glon"][:, None] - d["lon"][d["eval"]][None, :]).argmin(axis=0)
                    axis.plot(ex, ey, "o", mfc="none", mec="cyan", ms=4, mew=0.8)
        axes[r][0].set_ylabel(
            f"{name}\nRMSE {m['field_rmse']:.1f} ({m['rmse_gain_percent']:+.0f}%)\n"
            f"subgrid r {m['subgrid_corr']:.3f}\nbullseye {m['bullseye']:.1f}",
            fontsize=7)

    figure.suptitle(
        f"Single-day OSSE, {args.day or 'nature run day'}: does the "
        f"reconstruction improve?\n"
        f"black = assimilated gauges, cyan = withheld. "
        f"background RMSE {data[0][1]['metrics']['background_rmse']:.1f} mm/day",
        y=1.0)
    figure.tight_layout()
    figure.savefig(out_dir / "single_day_compare.png", bbox_inches="tight")
    figure.savefig(out_dir / "single_day_compare.pdf", bbox_inches="tight")
    plt.close(figure)

    fields = ["variant", "background_rmse", "field_rmse", "rmse_gain_percent",
              "background_corr", "field_corr", "background_subgrid_corr",
              "subgrid_corr", "bullseye", "analysis_spread"]
    with (out_dir / "single_day_compare.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for name, d in data:
            writer.writerow({"variant": name, **d["metrics"]})

    print(f"{'variant':12s} {'RMSE':>7s} {'gain%':>7s} {'corr':>6s} "
          f"{'subgrid r':>10s} {'bullseye':>9s}")
    for name, d in data:
        m = d["metrics"]
        print(f"{name:12s} {m['field_rmse']:>7.2f} {m['rmse_gain_percent']:>+7.1f} "
              f"{m['field_corr']:>6.3f} {m['subgrid_corr']:>10.3f} "
              f"{m['bullseye']:>9.1f}")
    print(f"\n{'background':12s} {data[0][1]['metrics']['background_rmse']:>7.2f} "
          f"{'--':>7s} {data[0][1]['metrics']['background_corr']:>6.3f} "
          f"{data[0][1]['metrics']['background_subgrid_corr']:>10.3f}")
    print(f"\n[done] {out_dir / 'single_day_compare.png'}")
    print(f"[done] {out_dir / 'single_day_compare.csv'}")


if __name__ == "__main__":
    main()
