#!/usr/bin/env python3
"""Review downloaded extremes, then prepare a small four-stream attribution run.

Source agreement verifies copying/preparation, not the truth of a rainfall
measurement. Nearby disagreement never automatically masks or shifts a value.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_tail_helpers", ROOT/"scripts/107_surma_tail_diagnostics.py")
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
I = T.I
GROUP = "v2_event_attribution"
VARIANTS = ["background", "event_gauges_only", "event_imerg_only", I.BASELINE]


def local_record(item, archive):
    """Relocate only files within this downloaded archive; verify saved hashes."""
    parts = Path(item["path"]).parts
    # HPC plans use data/processed/surma_tail_pilot; archive root may be renamed.
    indices = [i for i, p in enumerate(parts) if p == "surma_tail_pilot"]
    path = archive.joinpath(*parts[indices[-1]+1:]) if indices else Path(item["path"])
    if not path.is_file() or I.sha(path) != item["sha256"]:
        raise ValueError("missing/changed downloaded input: " + str(path))
    return path


def flag_reasons(flags, station_id, date):
    if flags.empty: return ""
    match = flags[(flags.station_id == station_id) &
                  (flags.review_start <= date) & (flags.review_end >= date)]
    return " | ".join(sorted(set(match.reason.dropna().astype(str))))


def nearby(frame, station_id, date, lat, lon, bwdb_only=True):
    """Five nearest valid same-date BWDB records within 50 km, excluding self."""
    network = frame.station_id.str.startswith("BWDB_") if bwdb_only else pd.Series(True, index=frame.index)
    d = frame[(frame.date == date) & network &
              (frame.station_id != station_id) & frame.precip_mm.notna()].copy()
    la, lo = np.radians(d.lat.to_numpy()), np.radians(d.lon.to_numpy())
    a = np.sin((la-np.radians(lat))/2)**2
    a += np.cos(la)*np.cos(np.radians(lat))*np.sin((lo-np.radians(lon))/2)**2
    d["km"] = 12742.0176*np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return d[d.km <= 50].sort_values(["km", "station_id"]).head(5)


def source_frame(stations, days):
    """Use production aliases/QC, reading only needed BWDB dates into long form."""
    spec = importlib.util.spec_from_file_location("_production_stations", ROOT/"scripts/99_prepare_production_stations.py")
    P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
    bmd_path = stations/"Rainfall_daily_by_station_BMD_corrected.csv"
    catalog = stations/"BMD_production_station_catalog.csv"
    bwdb_path = stations/"BWDB_Rainfall_2000_2025_corrected.xlsx"
    bmd, _ = P.read_wide_bmd(bmd_path, catalog, min(days), max(days), catalog_only=True)
    bmd = bmd[bmd.date.isin(days)].copy()
    bmd.station_id = "BMD_"+bmd.station_id.astype(str)
    bmd["source"] = "BMD"
    meta = P.prep82.read_xlsx_sheet(bwdb_path, "StationList")
    wide = P.prep82.read_xlsx_sheet(bwdb_path, "RainfallData")
    meta.columns = meta.columns.str.strip(); wide.columns = wide.columns.str.strip()
    raw_dates = wide.Date
    dates = (pd.to_datetime(raw_dates, unit="D", origin="1899-12-30", errors="coerce")
             if pd.api.types.is_numeric_dtype(raw_dates) else pd.to_datetime(raw_dates, errors="coerce"))
    wide["Date"] = dates.dt.normalize()
    wide = wide[wide.Date.isin(days)]
    meta["Station ID"] = meta["Station ID"].astype(str).str.strip()
    meta = meta.drop_duplicates("Station ID").copy()
    meta["Latitude"] = pd.to_numeric(meta.Latitude, errors="coerce")
    meta["Longitude"] = pd.to_numeric(meta.Longitude, errors="coerce")
    meta.loc[meta["Station ID"] == "CL312", "Latitude"] = 20.8925
    ids = [s for s in meta["Station ID"] if s in wide.columns]
    long = wide.melt(id_vars="Date", value_vars=ids, var_name="Station ID", value_name="precip_mm")
    long.precip_mm = pd.to_numeric(long.precip_mm, errors="coerce")
    long.loc[(long.precip_mm < 0) | (long.precip_mm > 500), "precip_mm"] = np.nan
    long = long.merge(meta, on="Station ID", validate="many_to_one")
    bwdb = pd.DataFrame(dict(station_id="BWDB_"+long["Station ID"], date=long.Date,
        lat=long.Latitude, lon=long.Longitude, precip_mm=long.precip_mm, source="BWDB"))
    frame = pd.concat([bmd, bwdb], ignore_index=True)
    if frame.duplicated(["station_id", "date"]).any():
        raise ValueError("duplicate source station-day")
    return frame, [I.record(p) for p in (bmd_path, catalog, bwdb_path)]


def aggregation_summary(frame):
    rows = []
    for (role, aggregated), d in frame.groupby(["role", "aggregated"]):
        rows.append(dict(role=role, aggregated=bool(aggregated), n=len(d),
            original_mean_mm=float(d.precip_mm.mean()), assimilated_mean_mm=float(d.assimilated_mm.mean()),
            median_retained_fraction=float(d.assimilated_fraction_of_original.median()),
            mean_reduction_mm=float(d.original_minus_assimilated_mm.mean()),
            n_reduced_below50=int((d.assimilated_mm < 50).sum()),
            n_original_ge100=int((d.precip_mm >= 100).sum()),
            n_ge100_reduced_below100=int(((d.precip_mm >= 100) & (d.assimilated_mm < 100)).sum())))
    return pd.DataFrame(rows)


def review(args):
    archive, out = Path(args.pilot), Path(args.out_dir)
    if out.resolve() == archive.resolve() or archive.resolve() in out.resolve().parents:
        raise ValueError("write review separately from the downloaded archive")
    plan = json.loads((archive/"pilot_plan.json").read_text())
    summary = archive/"summary"
    inputs = [summary/(n+".csv") for n in ("aggregation_extremes", "heavy_events", "review_flags_for_pilot", "intensity_scores")]
    agg, heavy, flags, scores = [pd.read_csv(p) for p in inputs]
    for c in ("date",): agg[c] = pd.to_datetime(agg[c]); heavy[c] = pd.to_datetime(heavy[c])
    for c in ("review_start", "review_end"): flags[c] = pd.to_datetime(flags[c])
    original, assimilated, footprints = [], {}, {}
    held = set((archive/"holdout_ids.txt").read_text().splitlines())
    for w in plan["windows"]:
        paths = [local_record(item, archive) for item in w["files"]]
        path = next(p for p in paths if p.name == "original_qc.csv")
        original.append(pd.read_csv(path, parse_dates=["date"]))
        table = pd.read_csv(next(p for p in paths if p.name == "superob.csv"), parse_dates=["date"])
        assimilated[w["label"]] = table[~table.station_id.isin(held)]
        receipt = json.loads((archive/"runs"/w["label"]/"complete.json").read_text())
        if receipt["plan_sha256"] != I.sha(archive/"pilot_plan.json"):
            raise ValueError("completed result belongs to another plan")
        for item in receipt["files"]: local_record(item, archive)
        with np.load(archive/"runs"/w["label"]/"sweep.npz", allow_pickle=False) as z:
            coarse = z["raw_imerg_mm"]
            lat, lon = z["grid_lat"], z["grid_lon"]
            if len(lat) % coarse.shape[1] or len(lon) % coarse.shape[2]:
                raise ValueError("IMERG footprints do not tile archived model grid")
            coarse_lat = lat.reshape(coarse.shape[1], -1).mean(axis=1)
            coarse_lon = lon.reshape(coarse.shape[2], -1).mean(axis=1)
            for event in heavy[(heavy.window == w["label"]) & (heavy.variant == I.BASELINE)].to_dict("records"):
                d = np.flatnonzero(z["times"] == str(event["date"].date()))[0]
                s = np.flatnonzero(z["station_ids"] == event["station_id"])[0]
                y = np.abs(coarse_lat-z["station_lat"][s]).argmin()
                x = np.abs(coarse_lon-z["station_lon"][s]).argmin()
                footprints[(w["label"], event["station_id"], event["date"])] = float(coarse[d, y, x])
    original = pd.concat(original, ignore_index=True)
    if original.duplicated(["station_id", "date"]).any(): raise ValueError("overlapping pilot station-days")
    days = set(original.date)
    days |= {d+pd.Timedelta(days=shift) for d in list(days) for shift in (-1, 1)}
    # Include whole flagged months for meaningful monthly-source checks.
    for month in flags.get("month", pd.Series(dtype=str)).dropna().unique():
        period = pd.Period(month, "M"); days.update(pd.date_range(period.start_time, period.end_time.normalize()))
    sources, source_records = source_frame(Path(args.stations), sorted(days))
    lookup = sources.set_index(["station_id", "date"]).precip_mm
    # Also trace BWDB values to the uncorrected local compilation when present.
    # This is provenance corroboration; it is not an independent gauge archive.
    uncorrected_path = Path(args.stations)/"BWDB_Rainfall_2000_2025.xlsx"
    uncorrected = None
    if uncorrected_path.exists():
        spec = importlib.util.spec_from_file_location("_original_bwdb", ROOT/"scripts/99_prepare_production_stations.py")
        P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
        raw = P.prep82.read_xlsx_sheet(uncorrected_path, "RainfallData")
        raw.columns = raw.columns.str.strip()
        raw.Date = (pd.to_datetime(raw.Date, unit="D", origin="1899-12-30", errors="coerce")
                    if pd.api.types.is_numeric_dtype(raw.Date) else pd.to_datetime(raw.Date, errors="coerce"))
        raw.Date = raw.Date.dt.normalize()
        if raw.Date.duplicated().any(): raise ValueError("duplicate original BWDB date")
        uncorrected = raw.set_index("Date")
        source_records.append(I.record(uncorrected_path))
    baseline = heavy[heavy.variant == I.BASELINE].copy()
    targets = pd.concat([
        agg[["window", "role", "station_id", "date", "precip_mm"]].assign(kind="assimilated_original"),
        baseline[["window", "role", "station_id", "date", "observed_mm"]].rename(columns={"observed_mm": "precip_mm"}).assign(kind="withheld")], ignore_index=True)
    checks = []
    original_meta = original.groupby("station_id")[["lat", "lon"]].first()
    for r in targets.to_dict("records"):
        sid, date = r["station_id"], r["date"]
        value = lookup.get((sid, date), np.nan)
        lat, lon = original_meta.loc[sid, ["lat", "lon"]]
        neighbors = nearby(sources, sid, date, lat, lon)
        reasons = flag_reasons(flags, sid, date)
        # same-network BWDB neighbors share the 03 UTC support; BMD differs 3h.
        supported = len(neighbors) >= 2 and int((neighbors.precip_mm >= 50).sum()) >= 2
        used = nearby(assimilated[r["window"]], sid, date, lat, lon, bwdb_only=False)
        original_value, original_matches = np.nan, None
        if uncorrected is not None and sid.startswith("BWDB_"):
            source_id = sid.removeprefix("BWDB_")
            # CL9 is the reviewed one-day correction in create_bwdb_correct.py.
            original_date = date-pd.Timedelta(days=1) if source_id == "CL9" else date
            if source_id in uncorrected and original_date in uncorrected.index:
                original_value = pd.to_numeric(uncorrected.at[original_date, source_id], errors="coerce")
                original_matches = bool(np.isfinite(original_value) and abs(original_value-r["precip_mm"]) <= .001)
        checks.append({**r, "source_mm": value,
            "source_matches": bool(np.isfinite(value) and abs(value-r["precip_mm"]) <= .001),
            "qc_reasons": reasons, "neighbor_count": len(neighbors),
            "neighbor_median_mm": float(neighbors.precip_mm.median()) if len(neighbors) else np.nan,
            "neighbor_max_mm": float(neighbors.precip_mm.max()) if len(neighbors) else np.nan,
            "neighbors_ge50": int((neighbors.precip_mm >= 50).sum()),
            "neighbor_ids": "|".join(neighbors.station_id),
            "neighbor_values_mm": "|".join(f"{v:.2f}" for v in neighbors.precip_mm),
            "neighbor_distances_km": "|".join(f"{v:.2f}" for v in neighbors.km),
            "neighbor_supported": supported,
            "uncorrected_bwdb_mm": original_value,
            "uncorrected_bwdb_matches": original_matches,
            "assimilated_neighbor_ids": "|".join(used.station_id),
            "assimilated_neighbor_values_mm": "|".join(f"{v:.2f}" for v in used.precip_mm),
            "assimilated_neighbor_distances_km": "|".join(f"{v:.2f}" for v in used.km),
            "assimilated_neighbors_ge50": int((used.precip_mm >= 50).sum()),
            "source_previous_day_mm": lookup.get((sid, date-pd.Timedelta(days=1)), np.nan),
            "source_next_day_mm": lookup.get((sid, date+pd.Timedelta(days=1)), np.nan)})
    checks = pd.DataFrame(checks)
    flags_checked = []
    for f in flags.to_dict("records"):
        d = original[(original.station_id == f["station_id"]) & original.date.between(f["review_start"], f["review_end"])]
        for r in d.to_dict("records"):
            value = lookup.get((r["station_id"], r["date"]), np.nan)
            flags_checked.append(dict(station_id=r["station_id"], date=r["date"], reason=f["reason"],
                prepared_mm=r["precip_mm"], source_mm=value,
                source_matches=bool(np.isfinite(value) and np.isfinite(r["precip_mm"]) and abs(value-r["precip_mm"]) <= .001)))
    flag_check = pd.DataFrame(flags_checked)
    # Compare same station-days, not the marginal wet bins in isolation.
    comparison = baseline.merge(heavy[heavy.variant == "background"][["window", "station_id", "date", "ensemble_mean_mm"]],
        on=["window", "station_id", "date"], suffixes=("", "_background"), validate="one_to_one")
    comparison = comparison.merge(checks[checks.kind == "withheld"], on=["window", "role", "station_id", "date"], validate="one_to_one")
    comparison["analysis_minus_background_mm"] = comparison.ensemble_mean_mm-comparison.ensemble_mean_mm_background
    comparison["imerg_footprint_mm"] = [footprints[(r.window, r.station_id, r.date)] for r in comparison.itertuples()]
    eligible = comparison[(comparison.observed_mm >= 100) & comparison.source_matches &
        comparison.neighbor_supported & (comparison.qc_reasons == "")].copy()
    stats = aggregation_summary(agg)
    out.mkdir(parents=True, exist_ok=True)
    for name, table in [("aggregation_summary", stats), ("extreme_source_checks", checks),
                        ("flagged_source_checks", flag_check), ("withheld_event_review", comparison),
                        ("supported_event_candidates", eligible)]:
        table.to_csv(out/(name+".csv"), index=False, date_format="%Y-%m-%d")
    lines = ["# Event-level review", "", "Source agreement is with the supplied corrected archives, not independent verification of rainfall truth.",
        "No station-day has been masked or shifted. BMD/BWDB support still differs by three hours.", "",
        f"Extreme records checked: {len(checks)}; source mismatches/missing: {int((~checks.source_matches).sum())}.",
        f"Records overlapping existing QC flags: {int(checks.qc_reasons.ne('').sum())}.",
        f"Flag/source checks (monthly and daily flags may overlap): {len(flag_check)}; mismatches/missing: {int((~flag_check.source_matches).sum()) if len(flag_check) else 0}.",
        f"Uncorrected BWDB compilation available: {uncorrected is not None}; checked BWDB extreme records: {int(checks.uncorrected_bwdb_matches.notna().sum())}; mismatches/missing: {int(checks.uncorrected_bwdb_matches.eq(False).sum())}.",
        f"Source-matched, unflagged withheld >=100 mm events with at least two of five nearby BWDB gauges >=50 mm: {len(eligible)}.", "",
        "| Role | Merged | Events >=50 | Mean reduction mm | Median retained fraction | Reduced below50 | >=100 reduced below100 |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for r in stats.to_dict("records"):
        lines.append(f"| {r['role']} | {r['aggregated']} | {r['n']} | {r['mean_reduction_mm']:.2f} | {r['median_retained_fraction']:.3f} | {r['n_reduced_below50']} | {r['n_ge100_reduced_below100']}/{r['n_original_ge100']} |")
    lines += ["", "Aggregation rows concern retained inputs, not withheld skill. Multiple gauges may share the same cell/day; counts are original gauge-events, not unique storms.",
        "Averaging changes spatial support and is not itself evidence of a processing defect.",
        "Unflagged and neighbor-supported means a useful diagnostic candidate, not certified truth.",
        "Existing QC flags screen large discrepancies; unflagged events can still be wrong or localized."]
    (out/"review.md").write_text("\n".join(lines)+"\n")
    I.write_json(out/"review_provenance.json", {"status": "completed_source_review", "inputs": [I.record(p) for p in inputs]+source_records,
        "pilot_plan": I.record(archive/"pilot_plan.json"), "reader": I.record(__file__), "automatic_exclusions": 0,
        "candidate_file": I.record(out/"supported_event_candidates.csv")})
    print((out/"review.md").read_text())
    print(eligible[["window", "date", "station_id", "observed_mm", "ensemble_mean_mm_background", "ensemble_mean_mm", "neighbor_median_mm", "neighbors_ge50"]].round(2).to_string(index=False))


def prepare(args):
    parent, out = Path(args.pilot).resolve(), Path(args.out_dir)
    if out.resolve() == parent or (out/"pilot_plan.json").exists():
        raise ValueError("choose a fresh output directory")
    plan, changed = T.load_archive(parent)
    manifest = Path(args.windows); design = json.loads(manifest.read_text())
    review_dir = Path(args.review_dir)
    provenance = json.loads((review_dir/"review_provenance.json").read_text())
    if provenance["pilot_plan"]["sha256"] != I.sha(parent/"pilot_plan.json"):
        raise ValueError("source review belongs to another pilot")
    I.verify(provenance["inputs"]+[provenance["candidate_file"]])
    candidates = pd.read_csv(provenance["candidate_file"]["path"], keep_default_na=False)
    flag_path = next((Path(r["path"]) for r in provenance["inputs"] if Path(r["path"]).name == "review_flags_for_pilot.csv"), None)
    if flag_path is None: raise ValueError("source review must include the QC flag input")
    flags = pd.read_csv(flag_path, parse_dates=["review_start", "review_end"])
    if len(design["windows"]) > 4 or not design["windows"]: raise ValueError("select 1-4 small event windows")
    held = set(Path(plan["holdout"]).read_text().splitlines())
    out.mkdir(parents=True, exist_ok=True)
    qc_path = out/"qc/review_flags.csv"; qc_path.parent.mkdir(parents=True, exist_ok=True)
    flags.to_csv(qc_path, index=False, date_format="%Y-%m-%d")
    holdout = out/"holdout_ids.txt"; holdout.write_text("\n".join(sorted(held))+"\n")
    cases, labels, occupied = [], set(), set()
    for selection in design["windows"]:
        w = next(w for w in plan["windows"] if w["label"] == selection["parent_window"])
        start, end = pd.Timestamp(selection["start"]), pd.Timestamp(selection["end"])
        label = selection["label"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", label) or label in labels or not pd.Timestamp(w["start"]) <= start <= end <= pd.Timestamp(w["end"]) or (end-start).days > 4:
            raise ValueError("invalid, duplicate or oversized event window")
        days = set(pd.date_range(start, end))
        if occupied & days: raise ValueError("overlapping attribution windows would double-count station-days")
        overlaps = flags[(flags.window == w["label"]) & (flags.review_start <= end) & (flags.review_end >= start)]
        if len(overlaps): raise ValueError("selected window contains flagged input station-days; review those sources or choose other dates")
        labels.add(label); occupied |= days
        original = pd.read_csv(T.original_path(w), parse_dates=["date"])
        targets = candidates[(candidates.window == w["label"]) &
            candidates.date.between(str(start.date()), str(end.date()))]
        if targets.empty: raise ValueError("no source-reviewed, supported heavy events in selected window")
        events = []
        for target in targets.to_dict("records"):
            event = {"date": target["date"], "station_id": target["station_id"], "observed_mm": target["observed_mm"]}
            date = pd.Timestamp(event["date"])
            d = original[(original.station_id == event["station_id"]) & (original.date == date)]
            if event["station_id"] not in held or date not in days or len(d) != 1 or abs(float(d.precip_mm.iloc[0])-event["observed_mm"]) > .001:
                raise ValueError("selected event does not match frozen withheld input")
            if not target["source_matches"] or not target["neighbor_supported"] or target["qc_reasons"]:
                raise ValueError("source review evidence required for event selection")
            events.append({**event, "source_review": "matched_corrected_archive_and_neighbor_supported"})
        # Reuse the exact parent superobs and its error budget: recomputing a
        # 3-day budget would confound removal of observation streams.
        cases.append({**w, "label": label, "role": "retest", "start": str(start.date()), "end": str(end.date()),
            "days": len(days), "parent_window": w["label"], "events": events})
    inherited = {k: plan[k] for k in ("checkpoint", "config", "stats", "predictors", "fill_known_cpc_gaps", "members", "seed", "min_coverage", "held_stations", "retained_stations", "buffer_km")}
    shared = [I.record(p) for p in (holdout, parent/"pilot_plan.json", manifest, plan["checkpoint"], plan["config"], plan["stats"], __file__,
        ROOT/"scripts/107_surma_tail_diagnostics.py", ROOT/"scripts/106_surma_improvement.py", ROOT/"scripts/28_simultaneous_method_sweep.py",
        ROOT/"src/bdhires/da/guidance.py", ROOT/"src/bdhires/da/sampler.py")]
    shared += [I.record(qc_path), I.record(review_dir/"review_provenance.json"), provenance["candidate_file"]]+provenance["inputs"]
    I.write_json(out/"pilot_plan.json", {**inherited, "status": "prepared_research_pilot", "group": GROUP, "variants": VARIANTS,
        "windows": cases, "holdout": str(holdout.resolve()), "parent_pilot": str(parent), "shared_files": shared, "source_files": plan["source_files"],
        "qc_review": "source-matched, neighbor-supported event selection; no automatic exclusions",
        "excluded_station_days": plan.get("excluded_station_days", 0), "notes": T.NOTES,
        "parent_code_changes_read_only": changed, "selection": design})
    print(f"[prepared] {len(cases)} windows, {sum(w['days'] for w in cases)} days, four variants; {out}")


def summarize(args):
    T.diagnose(args, current=True)
    out = Path(args.out_dir); plan, _ = T.load_archive(out)
    events = pd.read_csv(out/"summary/heavy_events.csv")
    selected = []
    for w in plan["windows"]:
        for target in w["events"]:
            d = events[(events.window == w["label"]) & (events.date == target["date"]) & (events.station_id == target["station_id"])].copy()
            if set(d.variant) != set(VARIANTS): raise ValueError("selected event missing from comparison")
            means = d.set_index("variant").ensemble_mean_mm
            d["mean_increment_from_background_mm"] = d.ensemble_mean_mm-means["background"]
            d["combined_minus_gauges_only_mm"] = means[I.BASELINE]-means["event_gauges_only"]
            d["combined_minus_imerg_only_mm"] = means[I.BASELINE]-means["event_imerg_only"]
            selected.append(d)
    table = pd.concat(selected, ignore_index=True)
    table.to_csv(out/"summary/selected_event_attribution.csv", index=False)
    print(table[["date", "station_id", "variant", "observed_mm", "ensemble_mean_mm", "q95_mm", "mean_increment_from_background_mm"]].round(2).to_string(index=False))
    print("Stream differences are conditional effects; they need not add linearly. Event-selected retests do not estimate archive-wide skill.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["review", "prepare", "run", "summarize"])
    p.add_argument("--pilot", default="data/processed/surma_tail_pilot")
    p.add_argument("--out-dir")
    p.add_argument("--stations", default="data/stations")
    p.add_argument("--windows", default="configs/surma_event_windows.json")
    p.add_argument("--review-dir", default="data/processed/surma_event_review")
    p.add_argument("--task", type=int)
    args = p.parse_args()
    if args.out_dir is None:
        args.out_dir = "data/processed/surma_event_review" if args.stage == "review" else "data/processed/surma_event_attribution"
    if args.stage == "run": T.run(args, group=GROUP, variants=VARIANTS)
    else: globals()[args.stage](args)


if __name__ == "__main__":
    try: main()
    except (ValueError, OSError, KeyError, StopIteration) as exc:
        print("[event] failed: " + str(exc), file=sys.stderr); sys.exit(1)
