#!/usr/bin/env python
"""Convert the legacy BMD station-month matrix to canonical daily observations.

Example for the May 2018 DA process check::

    python scripts/05_convert_bmd_legacy.py \
        --rainfall data/bmd/bmd.csv --stations data/bmd/Stations.csv \
        --start 2018-05-01 --end 2018-05-31 \
        --out data/processed/bmd_daily_may2018.csv \
        --summary data/processed/bmd_stations_may2018.csv \
        --report data/processed/bmd_qc_may2018.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bdhires.bmd import read_legacy_bmd, summarize_daily  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rainfall", required=True, help="legacy BMD bmd.csv")
    parser.add_argument("--stations", required=True, help="BMD Stations.csv catalogue")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--max-mm", type=float, default=1000.0)
    parser.add_argument("--out", required=True, help="canonical long-form CSV")
    parser.add_argument("--summary", required=True, help="per-station QC CSV")
    parser.add_argument("--report", required=True, help="machine-readable QC JSON")
    args = parser.parse_args()

    daily, report = read_legacy_bmd(
        args.rainfall,
        args.stations,
        start=args.start,
        end=args.end,
        max_mm=args.max_mm,
    )
    summary = summarize_daily(daily)
    for target in (args.out, args.summary, args.report):
        Path(target).parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(args.out, index=False, date_format="%Y-%m-%d")
    summary.to_csv(args.summary, index=False)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")

    print(
        f"wrote {args.out}: {report['valid_observations']} valid station-days, "
        f"{report['stations']} stations, {report['date_start']} to {report['date_end']}"
    )
    print(f"wrote {args.summary}")
    print(f"wrote {args.report}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
