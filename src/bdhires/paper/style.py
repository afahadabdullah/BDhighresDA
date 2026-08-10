"""One matplotlib style for every paper figure, and data saved beside each one.

Two rules this module exists to enforce.

**Every figure ships the numbers behind it.** ``save_figure`` will not write a
figure without at least one named data table, and it writes those tables as CSV
next to the PDF. This is not bookkeeping for its own sake: a reviewer question
six months from now ("what was the wet-area fraction in panel c?") should be
answerable from a text file, not by re-running a GPU job whose inputs may have
moved. It also means a figure can be rebuilt, restyled or replotted by a
co-author without the pipeline.

**Provenance travels with the figure.** Each figure gets a JSON manifest
recording the script that made it, the input files it read, the git commit, and
the time. Figures that cannot say where they came from have caused real
confusion in this project already -- several PNGs in ``data/processed`` are now
of uncertain vintage.

Sizes follow the usual two-column journal geometry: 3.4 in for a single column,
7.0 in for a full-width figure. Setting them here rather than per-figure is what
keeps font sizes consistent once everything is scaled into the manuscript.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

FIGURE_WIDTH_ONE_COLUMN = 3.4
FIGURE_WIDTH_TWO_COLUMN = 7.0

# Chosen to stay distinguishable in greyscale and for the common forms of
# colour vision deficiency: the arms are usually ordered, so the palette runs
# dark-to-light rather than around a hue wheel.
PALETTE = {
    "background": "#8c8c8c",
    "gauges": "#1f6f8b",
    "satellite": "#e0a458",
    "combined": "#c1440e",
    "truth": "#111111",
    "null": "#b0b0b0",
    "accent": "#1a7f37",
    "warn": "#c1440e",
}

SEQUENTIAL = ["#08306b", "#2171b5", "#6baed6", "#c6dbef"]


def use_paper_style():
    """Apply the shared style and return the pyplot module."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.3,
        "pdf.fonttype": 42,      # editable text in Illustrator, not outlines
        "ps.fonttype": 42,
    })
    return plt


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def figure_path(out_dir: str | Path, number: str, slug: str) -> Path:
    """``fig03_station_folds`` under ``out_dir``, without an extension."""
    return Path(out_dir) / f"fig{number}_{slug}"


def _write_table(path: Path, table) -> dict:
    """Write one data table as CSV. Accepts a dict of columns or a list of rows."""
    if isinstance(table, dict):
        keys = list(table)
        columns = [np.asarray(table[k]).ravel() for k in keys]
        if not columns:
            return {"rows": 0, "columns": []}
        length = max(c.size for c in columns)
        if any(c.size not in (length, 1) for c in columns):
            raise ValueError(
                f"{path.name}: columns have inconsistent lengths "
                f"{ {k: np.asarray(v).size for k, v in table.items()} }"
            )
        columns = [np.repeat(c, length) if c.size == 1 else c for c in columns]
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(keys)
            for row in zip(*columns):
                writer.writerow(["" if (isinstance(v, float) and np.isnan(v)) else v
                                 for v in row])
        return {"rows": int(length), "columns": keys}

    rows = list(table)
    if not rows:
        return {"rows": 0, "columns": []}
    keys = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return {"rows": len(rows), "columns": keys}


def save_figure(figure, out_dir, number: str, slug: str, data: dict,
                sources=None, caption: str = "", formats=("pdf", "png")) -> Path:
    """Save a paper figure with its data and provenance.

    ``data`` maps a panel name to either a dict of equal-length columns or a
    list of row dicts. It is REQUIRED and may not be empty: a figure whose
    numbers cannot be written out is a figure nobody can check.
    """
    if not data:
        raise ValueError(
            f"fig{number} ({slug}) was given no data. Every paper figure must "
            f"ship the numbers behind it -- pass data={{'panel_a': {{...}}}}."
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = figure_path(out_dir, number, slug)

    written = []
    for extension in formats:
        target = stem.with_suffix(f".{extension}")
        figure.savefig(target)
        written.append(target.name)

    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    tables = {}
    for panel, table in data.items():
        target = data_dir / f"{stem.name}_{panel}.csv"
        tables[panel] = {"file": target.name, **_write_table(target, table)}

    manifest = {
        "figure": stem.name,
        "caption": caption,
        "files": written,
        "data": tables,
        "sources": [str(s) for s in (sources or [])],
        "generated_by": Path(sys.argv[0]).name or "python",
        "git_commit": _git_commit(),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (data_dir / f"{stem.name}_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=float) + "\n")

    print(f"[fig{number}] {stem.name}: " + ", ".join(written) +
          f"  (+{len(tables)} data table(s))")
    return stem
