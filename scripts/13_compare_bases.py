#!/usr/bin/env python
"""Score every candidate residual base against CHIRPS, on the same footing.

The question this settles: what should the generative prior be anchored on?
Argument has not resolved it, so measure it.  Every candidate is compared to the
same target with the same statistic, including the one already known --
``base_correlation = 0.497`` for ERA5 tp over 68M pixel-days, from
``06_compute_stats.py --residual``.

Candidates:

    era5_tp      the ERA5 precipitation forecast, already in the Zarr
    climatology  per-pixel day-of-year CHIRPS mean (the v6 base)
    cpc          CPC Global Unified gauge analysis, 0.5 deg, read from the raw
                 netCDFs and bilinearly interpolated to the target grid

TWO SCALES, AND THE SECOND ONE IS THE POINT
    At 0.5 degree, both fields are aggregated to the CPC grid.  This asks whether
    the products agree on the AMOUNT of rain -- a test of gauge consistency,
    since CHIRPS and CPC share GTS reports.

    At 0.05 degree, CPC is interpolated up.  The variance it CANNOT represent is
    the fraction a 55 km product structurally misses, and that gap is the entire
    value proposition for downscaling it.  A base that explains 90% of the
    aggregate but 40% of the field is telling you the downscaling is doing real
    work; one that explains 90% of both is telling you it is not.

    python scripts/13_compare_bases.py --cpc-dir data/raw/cpc --days 400
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bdhires.grids import WIDE, crop_offsets, get_grid  # noqa: E402
from bdhires.transforms import PrecipTransform  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr", default="data/processed/bd_wide.zarr")
    parser.add_argument("--stats", default=None,
                        help="statistics file supplying the precip transform")
    parser.add_argument("--grid", default="bd")
    parser.add_argument("--cpc-dir", default="data/raw/cpc")
    parser.add_argument("--climatology", default=None,
                        help="(366,H,W) .npy from 06_compute_stats.py")
    parser.add_argument("--era5-tp-index", type=int, default=0)
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--end", default="2018-12-31")
    parser.add_argument("--days", type=int, default=400)
    parser.add_argument("--month", type=int, default=0,
                        help="0 = all months; 7 restricts to July")
    parser.add_argument("--out-figure", default="data/processed/base_comparison.png")
    parser.add_argument("--out-report", default="data/processed/base_comparison.json")
    return parser.parse_args()


def load_cpc(cpc_dir: Path, times, grid) -> np.ndarray | None:
    """Bilinearly interpolate CPC onto the target grid for the requested days."""
    import xarray as xr

    years = sorted({int(str(t)[:4]) for t in times.astype("datetime64[Y]")})
    paths = [cpc_dir / f"precip.{year}.nc" for year in years]
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        print(f"[compare] CPC files missing, skipping CPC: {missing[:4]}"
              f"{' ...' if len(missing) > 4 else ''}")
        return None

    data = xr.open_mfdataset([str(p) for p in paths], combine="by_coords")
    # xarray normally decodes CPC's -9.96921e36 fill value to NaN.  Keep that
    # convention explicit as a guard against files/backends that do not apply
    # ``missing_value`` masking themselves.  Missing CPC cells remain missing;
    # ``score`` below uses a pairwise-finite mask so they do not invalidate an
    # otherwise valid CPC/CHIRPS comparison.
    field = data["precip"].where(
        np.isfinite(data["precip"]) & (data["precip"] >= 0.0) & (data["precip"] <= 1000.0)
    )
    # CPC longitudes run 0..360; the BD domain does not wrap, so a plain sel works
    # once the coordinate is expressed the same way.
    if float(field.lon.min()) >= 0 and grid.lon_min < 0:
        raise ValueError("longitude convention mismatch")
    subset = field.sel(
        lat=slice(grid.lat_min - 1.5, grid.lat_max + 1.5),
        lon=slice(grid.lon_min - 1.5, grid.lon_max + 1.5),
    )
    if subset.sizes.get("lat", 0) == 0:            # descending latitude
        subset = field.sel(
            lat=slice(grid.lat_max + 1.5, grid.lat_min - 1.5),
            lon=slice(grid.lon_min - 1.5, grid.lon_max + 1.5),
        )
    subset = subset.sortby("lat")
    interpolated = subset.interp(
        lat=grid.lat, lon=grid.lon, method="linear",
    ).sel(time=times, method="nearest")
    out = np.asarray(interpolated.values, dtype=np.float32)
    data.close()
    return np.clip(out, 0.0, None)


def block_mean(field: np.ndarray, factor: int) -> np.ndarray:
    """Aggregate (..., H, W) by an integer factor, ignoring NaN."""
    h = field.shape[-2] // factor * factor
    w = field.shape[-1] // factor * factor
    trimmed = field[..., :h, :w]
    shaped = trimmed.reshape(
        *trimmed.shape[:-2], h // factor, factor, w // factor, factor
    )
    with np.errstate(invalid="ignore"):
        return np.nanmean(shaped, axis=(-3, -1))


def score(candidate: np.ndarray, target: np.ndarray, keep: np.ndarray,
          transform: PrecipTransform) -> dict:
    # CPC is a land gauge analysis and has valid missing values.  A single
    # missing pixel must reduce the matched sample count, not turn every metric
    # for that candidate into NaN.  Apply this per candidate because ERA5 and
    # climatology can have different coverage from CPC.
    matched = keep & np.isfinite(candidate) & np.isfinite(target)
    a = candidate[matched].astype(np.float64)
    b = target[matched].astype(np.float64)
    if not len(a):
        return {
            "n": 0,
            "correlation": float("nan"),
            "transformed_correlation": float("nan"),
            "variance_explained": float("nan"),
            "rmse_mm": float("nan"),
            "bias_mm": float("nan"),
            "mae_mm": float("nan"),
            "candidate_std_mm": float("nan"),
            "target_std_mm": float("nan"),
            "std_ratio": float("nan"),
        }
    difference = a - b
    correlation = (
        float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    )
    # the transformed-space correlation is the one directly comparable to the
    # base_correlation reported by 06_compute_stats.py
    ta, tb = transform.forward(a), transform.forward(b)
    transformed = (
        float(np.corrcoef(ta, tb)[0, 1])
        if ta.std() > 0 and tb.std() > 0 else float("nan")
    )
    return {
        "n": int(matched.sum()),
        "correlation": correlation,
        "transformed_correlation": transformed,
        "variance_explained": float(max(correlation, 0.0) ** 2),
        "rmse_mm": float(np.sqrt(np.mean(difference**2))),
        "bias_mm": float(np.mean(difference)),
        "mae_mm": float(np.mean(np.abs(difference))),
        "candidate_std_mm": float(a.std()),
        "target_std_mm": float(b.std()),
        "std_ratio": float(a.std() / b.std()) if b.std() > 0 else float("nan"),
    }


def main() -> None:
    args = parse_args()
    import zarr

    grid = get_grid(args.grid)
    store = zarr.open(args.zarr, mode="r")
    times = np.asarray(store["time"][:], dtype="datetime64[ns]")
    origin = crop_offsets(WIDE, grid)
    slices = (slice(origin[0], origin[0] + grid.nlat),
              slice(origin[1], origin[1] + grid.nlon))
    valid = np.asarray(store["valid"][:])[slices] > 0

    transform = PrecipTransform()
    if args.stats:
        transform = PrecipTransform.from_dict(
            json.loads(Path(args.stats).read_text())["precip_transform"]
        )

    eligible = np.where(
        (times >= np.datetime64(args.start)) & (times <= np.datetime64(args.end))
    )[0]
    if args.month:
        months = times[eligible].astype("datetime64[M]").astype(int) % 12 + 1
        eligible = eligible[months == args.month]
    chosen = eligible[
        np.linspace(0, len(eligible) - 1, min(args.days, len(eligible))).astype(int)
    ]
    print(f"[compare] {len(chosen)} days between {args.start} and {args.end}"
          + (f", month {args.month}" if args.month else ""), flush=True)

    target = np.stack([
        np.asarray(store["target"][int(i)][slices], dtype=np.float32) for i in chosen
    ])
    candidates: dict[str, np.ndarray] = {}

    candidates["era5_tp"] = np.stack([
        np.asarray(store["cond"][int(i)][args.era5_tp_index][slices], dtype=np.float32)
        for i in chosen
    ])

    climatology_path = args.climatology
    if climatology_path is None and args.stats:
        guess = Path(str(args.stats).removesuffix(".json") + "_climatology.npy")
        climatology_path = str(guess) if guess.is_file() else None
    if climatology_path:
        climatology = np.load(climatology_path)
        doys = np.array([
            int((times[int(i)].astype("datetime64[D]")
                 - times[int(i)].astype("datetime64[Y]")).astype(int))
            for i in chosen
        ])
        full = climatology[np.minimum(doys, climatology.shape[0] - 1)]
        candidates["climatology"] = (
            full[:, slices[0], slices[1]] if full.shape[-1] != grid.nlon else full
        )

    cpc = load_cpc(Path(args.cpc_dir), times[chosen], grid)
    if cpc is not None:
        candidates["cpc"] = cpc

    factor = 10          # 0.05 deg -> 0.5 deg, the CPC native scale
    finite = np.isfinite(target) & valid[None]
    target_coarse = block_mean(np.where(finite, target, np.nan), factor)
    valid_coarse = np.isfinite(target_coarse)

    report = {
        "days": int(len(chosen)),
        "period": [args.start, args.end],
        "month": args.month or "all",
        "reference": "CHIRPS",
        "note": (
            "transformed_correlation is directly comparable to the "
            "base_correlation reported by 06_compute_stats.py --residual "
            "(ERA5 tp scored 0.497 over 68M pixel-days)."
        ),
        "fine_0p05deg": {},
        "coarse_0p5deg": {},
    }
    for name, field in candidates.items():
        report["fine_0p05deg"][name] = score(field, target, finite, transform)
        coarse = block_mean(np.where(finite, field, np.nan), factor)
        report["coarse_0p5deg"][name] = score(
            coarse, target_coarse, valid_coarse, transform
        )

    Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_report).write_text(json.dumps(report, indent=2) + "\n")

    print(f"\n{'candidate':<14}{'r @0.05':>10}{'r @0.5':>9}{'var expl':>10}"
          f"{'RMSE':>9}{'bias':>9}{'std ratio':>11}")
    for name in candidates:
        fine, coarse = report["fine_0p05deg"][name], report["coarse_0p5deg"][name]
        print(f"{name:<14}{fine['correlation']:>10.3f}{coarse['correlation']:>9.3f}"
              f"{fine['variance_explained']:>10.1%}{fine['rmse_mm']:>9.2f}"
              f"{fine['bias_mm']:>+9.2f}{fine['std_ratio']:>11.2f}")
    print(f"\n{'candidate':<14}{'transformed r @0.05':>22}   (ERA5 reference: 0.497)")
    for name in candidates:
        print(f"{name:<14}{report['fine_0p05deg'][name]['transformed_correlation']:>22.3f}")

    if "cpc" in candidates:
        fine = report["fine_0p05deg"]["cpc"]["variance_explained"]
        coarse = report["coarse_0p5deg"]["cpc"]["variance_explained"]
        report["cpc_variance_missed_by_coarsening"] = float(coarse - fine)
        print(f"\nCPC explains {coarse:.1%} of CHIRPS at its own 0.5 deg scale "
              f"but only {fine:.1%} at 0.05 deg.")
        print(f"The {coarse - fine:.1%} gap is what downscaling has to supply -- "
              f"and is the case for doing it.")
        Path(args.out_report).write_text(json.dumps(report, indent=2) + "\n")

    # -- figure ---------------------------------------------------------------
    n = len(candidates)
    figure, axes = plt.subplots(2, max(2, n), figsize=(5.4 * max(2, n), 9.5),
                                constrained_layout=True, squeeze=False)
    day = int(np.argmax([np.nanmean(np.where(valid, t, np.nan)) for t in target]))
    vmax = max(5.0, float(np.nanpercentile(target[day][finite[day]], 99)))
    for column, (name, field) in enumerate(candidates.items()):
        axis = axes[0, column]
        image = axis.imshow(np.where(valid, field[day], np.nan), origin="lower",
                            extent=[grid.lon_min, grid.lon_max,
                                    grid.lat_min, grid.lat_max],
                            cmap="viridis", vmin=0, vmax=vmax)
        axis.set_title(f"{name}\nr={report['fine_0p05deg'][name]['correlation']:.2f} "
                       f"at 0.05 deg", fontsize=11)
        figure.colorbar(image, ax=axis, shrink=0.8)
        axes[1, column].scatter(
            target[day][finite[day]], field[day][finite[day]], s=3, alpha=0.25,
        )
        top = float(np.nanpercentile(target[day][finite[day]], 99.9))
        axes[1, column].plot([0, top], [0, top], "k--", lw=1)
        axes[1, column].set_xlabel("CHIRPS (mm day$^{-1}$)")
        axes[1, column].set_ylabel(f"{name} (mm day$^{{-1}}$)")
        axes[1, column].set_xlim(0, top)
        axes[1, column].set_ylim(0, top)
        axes[1, column].grid(alpha=0.25)
    for column in range(n, axes.shape[1]):
        axes[0, column].axis("off")
        axes[1, column].axis("off")
    figure.suptitle(
        "BDhighresDA - candidate residual bases scored against CHIRPS\n"
        f"{len(chosen)} days, {args.start} to {args.end}"
        + (f", month {args.month}" if args.month else "")
        + f"   |   maps show the wettest sampled day ({str(times[chosen[day]])[:10]})\n"
        "Top: the field.  Bottom: against CHIRPS, 1:1 dashed.",
        fontsize=14,
    )
    figure.savefig(args.out_figure, dpi=115)
    plt.close(figure)
    print(f"\nwrote {args.out_figure}")
    print(f"wrote {args.out_report}")


if __name__ == "__main__":
    main()
