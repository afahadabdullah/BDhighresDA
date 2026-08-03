#!/usr/bin/env python
"""Paper matrix for exact 0.1/0.2/0.5-degree CHIRPS footprint ablations."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


RESOLUTIONS = (0.1, 0.2, 0.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def load_summary(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"missing required scale summary: {path}")
    return json.loads(path.read_text())


def scope(report: dict, name: str) -> dict:
    aliases = {
        "field": ("field_0p05",),
        "footprint": ("footprint", "footprint_0p1"),
        "subgrid": ("subgrid", "subgrid_0p05"),
        "withheld": ("withheld_gauges",),
    }
    for key in aliases[name]:
        if key in report:
            return report[key]
    raise KeyError(f"{name} absent from {report.get('dump', 'summary')}")


def crpss(scores: dict) -> float:
    before = scores["background"]["crps_mm"]
    after = scores["analysis"]["crps_mm"]
    return 100.0 * (before - after) / before if before else float("nan")


def rmse_gain(scores: dict) -> float:
    before = scores["background"]["rmse_mm"]
    after = scores["analysis"]["rmse_mm"]
    return 100.0 * (before - after) / before if before else float("nan")


def delta_r(scores: dict) -> float:
    return scores["analysis"]["correlation"] - scores["background"]["correlation"]


def row(label: str, report: dict, resolution: float | None) -> dict:
    withheld = scope(report, "withheld")
    field = scope(report, "field")
    footprint = scope(report, "footprint")
    subgrid = scope(report, "subgrid")
    return {
        "arm": label,
        "resolution_deg": resolution,
        "withheld_crps_mm": withheld["analysis"]["crps_mm"],
        "withheld_crpss_percent": crpss(withheld),
        "field_crpss_percent": crpss(field),
        "footprint_crpss_percent": crpss(footprint),
        "subgrid_crpss_percent": crpss(subgrid),
        "subgrid_rmse_gain_percent": rmse_gain(subgrid),
        "subgrid_correlation_gain": delta_r(subgrid),
        "withheld_coverage90": withheld["analysis"]["coverage_90"],
        "withheld_spread_skill": withheld["analysis"].get(
            "spread_skill_ratio", np.nan
        ),
    }


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    gauge_report = load_summary(root / "gauges_exact_50_common" / "scale_summary.json")
    rows = [row("Gauges only", gauge_report, None)]
    reports: dict[tuple[str, float], dict] = {}
    for mode, prefix in (("Satellite only", "satellite"), ("Simultaneous", "simultaneous")):
        for factor, resolution in ((2, 0.1), (4, 0.2), (10, 0.5)):
            report = load_summary(
                root / f"{prefix}_exact_50_f{factor}" / "scale_summary.json"
            )
            actual = float(report.get("satellite_resolution_deg", resolution))
            if not np.isclose(actual, resolution):
                raise SystemExit(
                    f"{prefix} f{factor}: metadata says {actual}°, expected {resolution}°"
                )
            reports[(mode, resolution)] = report
            rows.append(row(f"{mode} {resolution:g}°", report, resolution))

    fieldnames = list(rows[0])
    with (out / "footprint_ablation.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (out / "footprint_ablation.json").write_text(json.dumps(rows, indent=2) + "\n")

    metrics = [
        ("Withheld CRPSS (%)", "withheld_crpss_percent"),
        ("Full-field CRPSS (%)", "field_crpss_percent"),
        ("Footprint CRPSS (%)", "footprint_crpss_percent"),
        ("Sub-footprint CRPSS (%)", "subgrid_crpss_percent"),
        ("Subgrid RMSE gain (%)", "subgrid_rmse_gain_percent"),
        ("Subgrid Δr", "subgrid_correlation_gain"),
        ("Withheld cover90", "withheld_coverage90"),
    ]
    values = np.asarray([[item[key] for _, key in metrics] for item in rows], float)
    normalized = values.copy()
    for column in range(values.shape[1]):
        finite = np.isfinite(values[:, column])
        scale = np.nanmax(np.abs(values[finite, column])) if finite.any() else 1.0
        normalized[:, column] /= max(scale, 1e-12)
    fig, ax = plt.subplots(figsize=(15, 6.2), constrained_layout=True)
    image = ax.imshow(normalized, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(metrics)), [name for name, _ in metrics], rotation=25, ha="right")
    ax.set_yticks(range(len(rows)), [item["arm"] for item in rows])
    for i in range(len(rows)):
        for j in range(len(metrics)):
            value = values[i, j]
            ax.text(j, i, "--" if not np.isfinite(value) else f"{value:.3f}",
                    ha="center", va="center", fontsize=8)
    ax.set_title(
        "Exact-CHIRPS footprint-resolution ablation\n"
        "50 spread pseudo-gauges; common 120×120 fine-grid window; positive skill is better"
    )
    fig.colorbar(image, ax=ax, label="column-normalized directional value")
    fig.savefig(out / "fig_footprint_ablation_matrix.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    plot_metrics = metrics[:6]
    gauge = rows[0]
    for ax, (title, key) in zip(axes.flat, plot_metrics):
        for mode, color, marker in (
            ("Satellite only", "#f4a24c", "s"),
            ("Simultaneous", "#cf4961", "o"),
        ):
            series = [row(f"tmp", reports[(mode, r)], r)[key] for r in RESOLUTIONS]
            ax.plot(RESOLUTIONS, series, color=color, marker=marker, lw=2, label=mode)
        if key in {"withheld_crpss_percent", "field_crpss_percent"}:
            ax.axhline(gauge[key], color="#187dad", ls="--", label="Gauges only")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(title)
        ax.set_xlabel("Pseudo-satellite footprint (degrees)")
        ax.set_xticks(RESOLUTIONS)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "What exact 0.1°, 0.2°, and 0.5° footprints buy in the CHIRPS OSSE\n"
        "All arms use identical dates, priors, seeds, members, gauge geometry, and spatial coverage",
        fontsize=14,
    )
    fig.savefig(out / "fig_footprint_ablation.png", dpi=180)
    plt.close(fig)

    latex = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Arm & Withheld CRPSS & Field CRPSS & Footprint CRPSS & Subgrid CRPSS & Subgrid RMSE gain & $\\Delta r$ \\\\",
        "\\midrule",
    ]
    for item in rows:
        latex.append(
            f"{item['arm']} & {item['withheld_crpss_percent']:.1f} & "
            f"{item['field_crpss_percent']:.1f} & {item['footprint_crpss_percent']:.1f} & "
            f"{item['subgrid_crpss_percent']:.1f} & {item['subgrid_rmse_gain_percent']:.1f} & "
            f"{item['subgrid_correlation_gain']:.3f} \\\\"
        )
    latex.extend(["\\bottomrule", "\\end{tabular}"])
    (out / "table_footprint_ablation.tex").write_text("\n".join(latex) + "\n")
    print(f"wrote footprint-resolution paper artifacts to {out}")


if __name__ == "__main__":
    main()
