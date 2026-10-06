#!/usr/bin/env python
"""Combined BMD + BWDB daily gauge table for one production period (no holdout).

Final-production counterpart of ``82_prepare_bmd_bwdb_may2022.py``.  That script
reads BMD only from the per-station 2020-2025 directory and always builds a
holdout fold.  The 2001-2020 reanalysis needs earlier BMD records and assimilates
every eligible gauge, so this script:

* reads BMD from the wide daily table ``Rainfall_daily_by_station_BMD.csv``
  (1975-2025), the only BMD source that covers 2001-2020 without a gap
* reads BWDB with the *same* reader and QC as script 82 (imported from it)
* writes the same ``combined_daily.csv`` / ``station_summary.csv`` schema, so
  ``87_superob_dense_gauges.py`` and ``28_simultaneous_method_sweep.py`` run
  unchanged
* writes NO holdout: every eligible station is assimilated.

For a 2020–2025 pilot without the wide history, --bmd-data-dir uses the
established per-station reader instead, with the same output schema and QC.
This optional source cannot serve the earlier historical production years.

Corrected production inputs use Rainfall_daily_by_station_BMD_corrected.csv
with BMD_production_station_catalog.csv and --bmd-catalog-only. Dates in this
file already incorporate the 2024+ correction and are never shifted here.

    python scripts/99_prepare_production_stations.py \
        --start 2005-04-01 --end 2005-06-30 \
        --out ROOT/stations/2005_q2/combined_daily.csv \
        --summary ROOT/stations/2005_q2/station_summary.csv \
        --report ROOT/stations/2005_q2/preparation_manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bdhires.bmd import (  # noqa: E402
    EXTRA_STATION_COORDS, MISSING_TOKENS, SENTINELS, STATION_ALIASES, _station_key,
    read_station_catalog, read_station_dir_bmd,
)

_spec = importlib.util.spec_from_file_location("prep82", ROOT / "scripts/82_prepare_bmd_bwdb_may2022.py")
prep82 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prep82)

# Column names of the wide daily table that differ from the catalogue spelling.
# Keys and values are ``_station_key`` forms.  Columns that match nothing are
# reported and skipped; they are never fuzzy-matched.
WIDE_ALIASES = {
    "ambagan": "ambaganctg", "chattogram": "chittagonj", "coxsbazar": "coxsbazar",
    "mcourt": "mcourt", "mymesingh": "mymensingh", "netrokuna": "natrakona",
    "sayedpur": "syedpur", "teltulia": "tetulia", "patuakhali": "pauakhali",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_wide_bmd(path: Path, catalog_csv: Path, start, end, max_mm: float = 1000.0,
                  catalog_only: bool = False):
    """Read wide rainfall as dated; a reviewed catalogue can disable legacy extras."""
    with Path(path).open(encoding="utf-8-sig") as handle:
        header = next(i for i, line in enumerate(handle) if line.split(",")[0].strip().lower() == "date")
    raw = pd.read_csv(path, skiprows=header, dtype=str, keep_default_na=False)
    raw.columns = [str(c).strip() for c in raw.columns]
    dates = pd.to_datetime(raw["Date"].str.strip(), errors="coerce")
    keep = dates.notna() & (dates >= start) & (dates <= end)
    raw, dates = raw.loc[keep], dates.loc[keep]
    catalog = read_station_catalog(catalog_csv)
    cat = {row["station_key"]: row for _, row in catalog.iterrows()}
    if not catalog_only:
        for key, info in EXTRA_STATION_COORDS.items():
            cat.setdefault(key, pd.Series({"station_id": info["station_id"], "catalog_name": info["catalog_name"],
                                           "lat": info["lat"], "lon": info["lon"]}))
    frames, unmatched, used = [], [], {}
    for column in raw.columns[1:]:
        key = _station_key(column.split(".")[0])
        key = WIDE_ALIASES.get(key, STATION_ALIASES.get(key, key))
        info = cat.get(key)
        if info is None:
            unmatched.append(column)
            continue
        token = raw[column].astype(str).str.strip()
        value = pd.to_numeric(token.where(~token.str.upper().isin(MISSING_TOKENS)), errors="coerce")
        value = value.mask(value.isin(SENTINELS) | (value < 0) | (value > max_mm))
        sid = int(info["station_id"])
        if sid in used:  # duplicated spelling of one station: fill gaps only
            used[sid]["precip_mm"] = used[sid]["precip_mm"].where(used[sid]["precip_mm"].notna(), value.to_numpy())
            continue
        used[sid] = pd.DataFrame({"station_id": sid, "name": str(info["catalog_name"]), "lat": float(info["lat"]),
                                  "lon": float(info["lon"]), "date": dates.to_numpy(), "precip_mm": value.to_numpy()})
        frames.append(used[sid])
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["station_id", "name", "lat", "lon", "date", "precip_mm"])
    return out, {"source": str(path), "format": "wide daily table", "stations_matched": len(frames),
                 "coordinate_policy": "catalogue only" if catalog_only else "catalogue plus legacy extras",
                 "date_policy": "input dates unchanged; no reader day shift",
                 "columns_unmatched_skipped": unmatched}


def read_bmd(args, start, end):
    """BMD records for [start, end] from the wide daily table (one source, 1975-2025).

    The original wide table was checked against the two readers used elsewhere: values are
    identical to the legacy station-month matrix (Jun-Sep 2017, 35 stations, 4,270
    station-days) and to the per-station 2020-2025 directory (Jun-Sep 2020, 4,514
    station-days), with no day shift in those historical comparisons. Corrected
    production dates are read as-is, including the supplied 2024+ correction.
    The legacy matrix ends during 2018 and the
    directory starts in 2020, so the wide table is the only source that covers
    2001-2020 without a gap.
    An explicit --bmd-data-dir is supported for bounded 2020–2025 pilots only.
    """
    directory = getattr(args, "bmd_data_dir", None)
    if directory:
        if getattr(args, "bmd_catalog_only", False):
            raise ValueError("--bmd-catalog-only is supported only with --bmd-wide")
        if start < pd.Timestamp("2020-01-01") or end > pd.Timestamp("2025-12-31"):
            raise ValueError("the 2020–2025 station-directory source cannot replace the wide BMD history outside those years")
        frame, qc = read_station_dir_bmd(directory, args.bmd_stations, start, end)
        return frame, [{"station_directory": qc}]
    frame, qc = read_wide_bmd(Path(args.bmd_wide), Path(args.bmd_stations), start, end,
                            catalog_only=getattr(args, "bmd_catalog_only", False))
    if frame.duplicated(["station_id", "date"]).any():
        raise ValueError("duplicate BMD station-days")
    return frame, [{"wide": qc}]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", required=True); p.add_argument("--end", required=True)
    p.add_argument("--bmd-wide", default="data/stations/Rainfall_daily_by_station_BMD.csv")
    p.add_argument("--bmd-data-dir", help="use existing per-station CSVs instead of the wide table, for 2020–2025 only")
    p.add_argument("--bmd-stations", default="data/stations/data_2020_2025/Stations.csv")
    p.add_argument("--bmd-catalog-only", action=argparse.BooleanOptionalAction, default=False,
                   help="use only explicit catalogue stations; do not add legacy fallback coordinates")
    p.add_argument("--bwdb-xlsx", default="data/stations/BWDB_Rainfall_2000_2025_corrected.xlsx")
    p.add_argument("--bwdb-max-mm", type=float, default=500.0)
    p.add_argument("--grid", default="bd")
    p.add_argument("--min-bmd", type=int, default=20, help="abort if fewer eligible BMD stations (guards a silent source gap)")
    p.add_argument("--min-bwdb", type=int, default=150, help="abort if fewer eligible BWDB stations")
    p.add_argument("--out", required=True); p.add_argument("--summary", required=True); p.add_argument("--report", required=True)
    args = p.parse_args()
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    if end < start:
        raise ValueError("--end must not precede --start")
    dates = pd.date_range(start, end, freq="D")

    bmd, bmd_qc = read_bmd(args, start, end)
    bmd = bmd.copy()
    bmd["station_id"] = "BMD_" + bmd["station_id"].astype(int).astype(str)
    bmd["source"] = "BMD"; bmd["accumulation_end_hour_utc"] = 0      # same convention as script 82
    bwdb, bwdb_qc = prep82.read_bwdb(Path(args.bwdb_xlsx), start, end, args.bwdb_max_mm)
    combined = pd.concat([bmd, bwdb], ignore_index=True)
    if combined.duplicated(["station_id", "date"]).any():
        raise ValueError("duplicate combined station-days")
    combined = combined.sort_values(["station_id", "date"]).reset_index(drop=True)
    summary = prep82.source_summary(combined, dates, args.grid)
    counts = summary.loc[summary["eligible_for_analysis"]].groupby("source")["station_id"].nunique().to_dict()
    if counts.get("BMD", 0) < args.min_bmd or counts.get("BWDB", 0) < args.min_bwdb:
        raise SystemExit(f"ERROR: too few eligible gauges for {args.start}..{args.end}: {counts} "
                         f"(need BMD>={args.min_bmd}, BWDB>={args.min_bwdb})")

    bmd_sources = ([path for path in sorted(Path(args.bmd_data_dir).glob("*.csv"))
                    if path.resolve() != Path(args.bmd_stations).resolve()
                    and not path.name.lower().endswith("stations.csv")]
                   if args.bmd_data_dir else [Path(args.bmd_wide)])
    inputs = [args.bwdb_xlsx, args.bmd_stations, *bmd_sources]
    for path in (args.out, args.summary, args.report):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.out, index=False, date_format="%Y-%m-%d")
    summary.to_csv(args.summary, index=False)
    report = {
        "mode": "production: every eligible station assimilated; no holdout",
        "period": {"start": str(start.date()), "end": str(end.date()), "days": int(len(dates))},
        "combined_daily": str(args.out), "station_summary": str(args.summary),
        "sources": {"BMD": {"qc": bmd_qc, "daily_window_utc": "[D-1 00:00, D 00:00]"},
                    "BWDB": {"qc": bwdb_qc, "daily_window_utc": "[D-1 03:00, D 03:00]"}},
        "eligible_station_counts": {k: int(v) for k, v in counts.items()},
        "eligibility_rule": "inside grid and >=50% daily coverage in the period (as script 82)",
        "input_sha256": {str(path): sha256(Path(path)) for path in inputs},
    }
    Path(args.report).write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({"period": report["period"], "eligible": report["eligible_station_counts"]}))


if __name__ == "__main__":
    main()
