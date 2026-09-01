#!/usr/bin/env python
"""Build an auditable combined BMD/BWDB daily table and constrained CV folds.

The two sources use the same *date labels* but not the same physical daily
window: BMD totals end at 00:00 UTC and BWDB/FFWC totals end at 03:00 UTC.
Both therefore use the existing CPC background calendar offset of -1 day, but
the mixed-source experiment records the residual three-hour support mismatch
rather than pretending the gauges have identical accumulation windows.

The fold files cover only gauges which can retain an assimilated neighbour
within the requested radius.  Stations without that support remain available
to the analysis in every fold but are not used for withheld-gauge scoring.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

import numpy as np
import pandas as pd

from bdhires.bmd import read_station_dir_bmd
from bdhires.grids import get_grid


_XLSX_MAIN = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_XLSX_DOC_REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_XLSX_PACKAGE_REL = "{http://schemas.openxmlformats.org/package/2006/relationships}"


def _xlsx_column_index(cell_reference: str) -> int:
    """Return zero-based column index from an XLSX reference such as ``AB14``."""
    letters = "".join(character for character in cell_reference if character.isalpha())
    value = 0
    for character in letters:
        value = value * 26 + ord(character.upper()) - ord("A") + 1
    return value - 1


def _read_xlsx_sheet_stdlib(path: Path, sheet_name: str) -> pd.DataFrame:
    """Read a plain XLSX worksheet when the compute environment lacks openpyxl.

    BWDB's delivered workbook contains ordinary XML worksheet cells (no macros,
    formulas, or rich formatting).  Reading this narrow interchange format here
    avoids installing packages into the frozen GH200 environment merely to
    ingest a static research input.
    """
    with ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall(f"{_XLSX_MAIN}si"):
                shared.append("".join(part.text or "" for part in item.iter(f"{_XLSX_MAIN}t")))
        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        relationships = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {
            relation.attrib["Id"]: relation.attrib["Target"]
            for relation in relationships.findall(f"{_XLSX_PACKAGE_REL}Relationship")
        }
        sheet_path = None
        for sheet in workbook.findall(f"{_XLSX_MAIN}sheets/{_XLSX_MAIN}sheet"):
            if sheet.attrib.get("name") == sheet_name:
                target = targets[sheet.attrib[f"{_XLSX_DOC_REL}id"]]
                target = target.lstrip("/")
                sheet_path = target if target.startswith("xl/") else "xl/" + target
                break
        if sheet_path is None:
            raise ValueError(f"{path} has no worksheet named {sheet_name!r}")
        root = ElementTree.fromstring(archive.read(sheet_path))
    rows: list[list[object]] = []
    width = 0
    for row in root.findall(f"{_XLSX_MAIN}sheetData/{_XLSX_MAIN}row"):
        values: dict[int, object] = {}
        for cell in row.findall(f"{_XLSX_MAIN}c"):
            column = _xlsx_column_index(cell.attrib["r"])
            cell_type = cell.attrib.get("t")
            value = cell.findtext(f"{_XLSX_MAIN}v")
            if cell_type == "s" and value is not None:
                parsed: object = shared[int(value)]
            elif cell_type == "inlineStr":
                parsed = "".join(part.text or "" for part in cell.iter(f"{_XLSX_MAIN}t"))
            elif value is None:
                parsed = None
            else:
                try:
                    parsed = float(value)
                    if parsed.is_integer():
                        parsed = int(parsed)
                except ValueError:
                    parsed = value
            values[column] = parsed
            width = max(width, column + 1)
        rows.append([values.get(column) for column in range(width)])
    if not rows:
        raise ValueError(f"{path}:{sheet_name} has no worksheet rows")
    # Rows before a later wide cell were initially shorter; pad them now.
    rows = [row + [None] * (width - len(row)) for row in rows]
    return pd.DataFrame(rows[1:], columns=[str(value).strip() for value in rows[0]])


def read_xlsx_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    """Prefer pandas/openpyxl, with a no-extra-dependency XLSX fallback."""
    try:
        return pd.read_excel(path, sheet_name=sheet_name)
    except ImportError as exc:
        if "openpyxl" not in str(exc):
            raise
        print(f"[BWDB] openpyxl unavailable; using standard-library XLSX reader for {sheet_name}")
        return _read_xlsx_sheet_stdlib(path, sheet_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bmd-data-dir", required=True)
    parser.add_argument("--bmd-stations", required=True)
    parser.add_argument("--bwdb-xlsx", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--out", required=True, help="Combined canonical daily CSV")
    parser.add_argument("--summary", required=True, help="Per-station QC CSV")
    parser.add_argument("--report", required=True, help="JSON provenance and fold manifest")
    parser.add_argument("--holdout-dir", required=True)
    parser.add_argument("--grid", default="bd")
    parser.add_argument("--holdout-folds", type=int, default=3)
    parser.add_argument(
        "--holdout-fraction", type=float, default=0.20,
        help="Fraction of all analysable stations withheld in each fold.",
    )
    parser.add_argument("--holdout-neighbor-km", type=float, default=20.0)
    parser.add_argument(
        "--coincident-group-km", type=float, default=5.0,
        help="Co-withhold BMD/BWDB pairs this close to prevent same-site leakage.",
    )
    parser.add_argument("--bwdb-max-mm", type=float, default=500.0)
    parser.add_argument("--seed", type=int, default=202205)
    return parser.parse_args()


def pairwise_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Great-circle distances with the diagonal set to infinity."""
    lat_r = np.radians(np.asarray(lat, float))
    lon_r = np.radians(np.asarray(lon, float))
    dlat = lat_r[:, None] - lat_r[None, :]
    dlon = lon_r[:, None] - lon_r[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_r)[:, None] * np.cos(lat_r)[None, :] * np.sin(dlon / 2) ** 2
    distance = 6371.0088 * 2 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    np.fill_diagonal(distance, np.inf)
    return distance


def read_bwdb(path: Path, start: pd.Timestamp, end: pd.Timestamp, max_mm: float) -> tuple[pd.DataFrame, dict]:
    """Read the supplied BWDB workbook without relying on its stale date fields."""
    stations = read_xlsx_sheet(path, "StationList")
    data = read_xlsx_sheet(path, "RainfallData")
    stations.columns = [str(column).strip() for column in stations.columns]
    data.columns = [str(column).strip() for column in data.columns]
    # ``Station`` is the workbook's label; accepting the older ``Station Name``
    # spelling keeps the reader compatible with a future BWDB export.
    name_column = "Station" if "Station" in stations.columns else "Station Name"
    required_station = {"Station ID", name_column, "Latitude", "Longitude"}
    if required_station - set(stations.columns):
        raise ValueError(f"StationList lacks {sorted(required_station - set(stations.columns))}")
    if "Date" not in data.columns:
        raise ValueError("RainfallData lacks Date")

    stations = stations.loc[:, ["Station ID", name_column, "Latitude", "Longitude"]].copy()
    stations = stations.rename(columns={name_column: "Station Name"})
    stations["Station ID"] = stations["Station ID"].astype(str).str.strip()
    stations["Latitude"] = pd.to_numeric(stations["Latitude"], errors="coerce")
    stations["Longitude"] = pd.to_numeric(stations["Longitude"], errors="coerce")
    stations = stations.dropna(subset=["Latitude", "Longitude"])
    stations = stations.drop_duplicates("Station ID", keep="first")
    # CL312 is a known transposition in the historical worksheet; the BWDB
    # real-time catalogue places it at 20.8925N, not 18.8925N.
    stations.loc[stations["Station ID"] == "CL312", "Latitude"] = 20.8925

    date_values = data["Date"]
    if pd.api.types.is_numeric_dtype(date_values):
        data["Date"] = pd.to_datetime(date_values, unit="D", origin="1899-12-30", errors="coerce").dt.normalize()
    else:
        data["Date"] = pd.to_datetime(date_values, errors="coerce").dt.normalize()
    data = data[(data["Date"] >= start) & (data["Date"] <= end)].copy()
    ids = [station_id for station_id in stations["Station ID"] if station_id in data.columns]
    if not ids:
        raise ValueError("no StationList IDs appear as RainfallData columns")
    long = data.melt(id_vars="Date", value_vars=ids, var_name="source_id", value_name="precip_mm")
    long["precip_mm"] = pd.to_numeric(long["precip_mm"], errors="coerce")
    invalid = (long["precip_mm"] < 0) | (long["precip_mm"] > max_mm)
    long.loc[invalid, "precip_mm"] = np.nan
    long = long.merge(stations, left_on="source_id", right_on="Station ID", how="inner", validate="many_to_one")
    out = pd.DataFrame(
        {
            "station_id": "BWDB_" + long["source_id"].astype(str),
            "name": long["Station Name"].astype(str).str.strip(),
            "lat": long["Latitude"].astype(float),
            "lon": long["Longitude"].astype(float),
            "date": long["Date"],
            "precip_mm": long["precip_mm"],
            "source": "BWDB",
            "accumulation_end_hour_utc": 3,
        }
    )
    if out.duplicated(["station_id", "date"]).any():
        raise ValueError("duplicate BWDB station-days after workbook parsing")
    return out, {
        "workbook": str(path), "stations_in_metadata": int(len(stations)),
        "stations_with_series": int(out["station_id"].nunique()),
        "values_over_max_removed": int((pd.to_numeric(long["precip_mm"], errors="coerce") > max_mm).sum()),
        "max_mm_threshold": float(max_mm), "cl312_latitude_corrected": 20.8925,
    }


def source_summary(frame: pd.DataFrame, dates: pd.DatetimeIndex, grid_name: str) -> pd.DataFrame:
    """Match ``load_stations`` eligibility, retaining enough information to audit it."""
    grid = get_grid(grid_name)
    lo, la, hi, ha = grid.bbox
    margin = grid.res / 2
    rows = []
    for station_id, group in frame.groupby("station_id", sort=True):
        values = group.set_index("date")["precip_mm"].reindex(dates)
        lat, lon = group[["lat", "lon"]].iloc[0]
        rows.append({
            "station_id": station_id, "source": group["source"].iloc[0], "name": group["name"].iloc[0],
            "lat": float(lat), "lon": float(lon), "accumulation_end_hour_utc": int(group["accumulation_end_hour_utc"].iloc[0]),
            "calendar_days": int(len(dates)), "n_obs": int(values.notna().sum()), "coverage": float(values.notna().mean()),
            "mean_mm": float(values.mean()) if values.notna().any() else np.nan,
            "max_mm": float(values.max()) if values.notna().any() else np.nan,
            "in_grid": bool(lo + margin < lon < hi - margin and la + margin < lat < ha - margin),
        })
    result = pd.DataFrame(rows)
    result["eligible_for_analysis"] = result["in_grid"] & (result["coverage"] >= 0.5)
    return result


def union_find_groups(meta: pd.DataFrame, distance: np.ndarray, coincident_km: float) -> list[list[int]]:
    """Group only cross-source near-duplicates, leaving same-network density intact."""
    n = len(meta)
    parent = list(range(n))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def join(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    source = meta["source"].to_numpy()
    for left in range(n):
        for right in range(left + 1, n):
            if source[left] != source[right] and distance[left, right] <= coincident_km:
                join(left, right)
    groups: dict[int, list[int]] = {}
    for index in range(n):
        groups.setdefault(find(index), []).append(index)
    return list(groups.values())


def make_folds(
    meta: pd.DataFrame, radius_km: float, coincident_km: float,
    folds: int, fraction: float, seed: int,
) -> tuple[list[list[str]], pd.DataFrame, dict]:
    """Partition supportable stations while proving a retained <=radius neighbour.

    Exact station-count fold capacities are used.  Randomized balanced packing
    is retried deterministically until every selected gauge has a retained
    neighbour after all co-located cross-source gauges in that fold are removed.
    """
    n = len(meta)
    distance = pairwise_km(meta["lat"], meta["lon"])
    nearest = distance.min(axis=1)
    groups = union_find_groups(meta, distance, coincident_km)
    station_group = np.empty(n, dtype=int)
    for group_index, members in enumerate(groups):
        station_group[members] = group_index

    # A group can be held out only if every member has a support neighbour
    # outside that entire co-located group.  Otherwise the group remains
    # assimilated in every fold.
    supportable_groups: list[list[int]] = []
    fixed_groups: list[list[int]] = []
    for members in groups:
        external = np.ones(n, dtype=bool)
        external[members] = False
        okay = all(np.any((distance[index] <= radius_km) & external) for index in members)
        (supportable_groups if okay else fixed_groups).append(members)
    candidates = [index for members in supportable_groups for index in members]
    target = int(round(n * fraction))
    if not 0 < fraction < 1:
        raise ValueError("--holdout-fraction must be strictly between 0 and 1")
    if target < 1 or target * folds > len(candidates):
        raise ValueError(f"only {len(candidates)} stations are supportable for {folds} folds")

    rng = np.random.default_rng(seed)
    assignment: list[list[int]] | None = None
    for attempt in range(2_000):
        available = list(supportable_groups)
        bins: list[list[int]] = []
        complete = True
        for _ in range(folds):
            held: list[int] = []
            while len(held) < target:
                choices = []
                for group_index, members in enumerate(available):
                    if len(held) + len(members) > target:
                        continue
                    proposed = held + members
                    retained = np.ones(n, dtype=bool)
                    retained[proposed] = False
                    if all(np.any((distance[index] <= radius_km) & retained) for index in proposed):
                        choices.append(group_index)
                if not choices:
                    complete = False
                    break
                # Prefer the group size that can exactly fill the remaining
                # slot; otherwise randomization makes folds geographically
                # independent while retaining a deterministic seed.
                remaining = target - len(held)
                exact = [index for index in choices if len(available[index]) == remaining]
                selected = int(rng.choice(exact or choices))
                held.extend(available.pop(selected))
            if not complete:
                break
            bins.append(held)
        if complete and all(len(held) == target for held in bins):
            assignment = bins
            break
    if assignment is None:
        raise RuntimeError(
            "could not construct the requested constrained 20% split after 2,000 attempts; "
            "increase the radius, reduce the coincident grouping radius, or inspect the station graph"
        )

    result = meta.copy()
    result["nearest_station_km"] = nearest
    result["coincident_group"] = station_group
    result["withheld_eligible"] = False
    result["holdout_fold"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    for fold, held in enumerate(assignment):
        result.loc[held, "withheld_eligible"] = True
        result.loc[held, "holdout_fold"] = fold
        retained = np.ones(n, dtype=bool)
        retained[held] = False
        result.loc[held, "retained_nearest_km"] = distance[np.asarray(held)][:, retained].min(axis=1)
    result["retained_nearest_km"] = result["retained_nearest_km"].astype(float)
    files = [sorted(meta.iloc[held]["station_id"].astype(str).tolist()) for held in assignment]
    manifest = {
        "holdout_folds": int(folds), "holdout_fraction_each_fold": float(fraction), "support_radius_km": float(radius_km),
        "coincident_cross_source_group_km": float(coincident_km),
        "analysis_stations": int(n), "withheld_eligible_stations": int(len(candidates)),
        "always_assimilated_not_scored_stations": int(n - len(candidates)),
        "fold_station_counts": [int(len(values)) for values in files],
        "folds_are_disjoint": True,
        "folds_cover_all_eligible_stations": False,
        "minimum_retained_neighbour_km": float(result.loc[result["withheld_eligible"], "retained_nearest_km"].min()),
        "maximum_retained_neighbour_km": float(result.loc[result["withheld_eligible"], "retained_nearest_km"].max()),
    }
    return files, result, manifest


def main() -> None:
    args = parse_args()
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    if end < start:
        raise ValueError("--end must not precede --start")
    dates = pd.date_range(start, end, freq="D")
    bmd, bmd_qc = read_station_dir_bmd(args.bmd_data_dir, args.bmd_stations, start, end)
    bmd = bmd.copy()
    bmd["station_id"] = "BMD_" + bmd["station_id"].astype(str)
    bmd["source"] = "BMD"
    bmd["accumulation_end_hour_utc"] = 0
    bwdb, bwdb_qc = read_bwdb(Path(args.bwdb_xlsx), start, end, args.bwdb_max_mm)
    combined = pd.concat([bmd, bwdb], ignore_index=True)
    if combined.duplicated(["station_id", "date"]).any():
        raise ValueError("duplicate combined station-days")
    combined = combined.sort_values(["station_id", "date"]).reset_index(drop=True)
    summary = source_summary(combined, dates, args.grid)
    analysis = summary.loc[summary["eligible_for_analysis"]].copy().reset_index(drop=True)
    folds, analysis_summary, fold_manifest = make_folds(
        analysis, args.holdout_neighbor_km, args.coincident_group_km,
        args.holdout_folds, args.holdout_fraction, args.seed,
    )
    summary = summary.merge(
        analysis_summary[["station_id", "nearest_station_km", "coincident_group", "withheld_eligible", "holdout_fold", "retained_nearest_km"]],
        on="station_id", how="left",
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    holdout_dir = Path(args.holdout_dir)
    holdout_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.out, index=False, date_format="%Y-%m-%d")
    summary.to_csv(args.summary, index=False)
    for fold, station_ids in enumerate(folds):
        (holdout_dir / f"fold{fold}.txt").write_text("\n".join(station_ids) + "\n")
    report = {
        "period": {"start": str(start.date()), "end": str(end.date()), "days": int(len(dates))},
        "combined_daily": str(args.out), "station_summary": str(args.summary), "holdout_dir": str(holdout_dir),
        "sources": {
            "BMD": {"qc": bmd_qc, "daily_window_utc": "[D-1 00:00, D 00:00]", "calendar_background_day_offset": -1},
            "BWDB": {"qc": bwdb_qc, "daily_window_utc": "[D-1 03:00, D 03:00]", "calendar_background_day_offset": -1},
        },
        "temporal_support_note": (
            "Date labels are aligned and both sources use background day D-1. BWDB aligns exactly "
            "with the existing 03 UTC prepared IMERG; BMD retains a three-hour support mismatch. "
            "This mixed-support May experiment is an evaluation, not a replacement production archive."
        ),
        "analysis_selection": fold_manifest,
        "source_station_counts": summary.groupby("source")["station_id"].nunique().astype(int).to_dict(),
        "source_analysis_counts": analysis.groupby("source")["station_id"].nunique().astype(int).to_dict(),
    }
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"out": args.out, "analysis_selection": fold_manifest}, indent=2))


if __name__ == "__main__":
    main()
