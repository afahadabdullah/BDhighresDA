#!/usr/bin/env python3
"""Recompute recoverable Bangladesh-only scores from supplied summary exports.

No inference or invented member/day data. Linear scores, RMS quantities and
coverage pool exact per-station sufficient statistics. Pearson r for the
two ensemble means uses monthly first/second moments plus summed squared
errors; the exporter verifies that those moments exactly reproduce the old
unfiltered correlations before using the country subset. Day-block intervals,
intensity bins, subgrid scores and full-grid temporal statistics require the
original archive and remain pending. Source exports are never overwritten.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bdhires.country import DEFAULT_BOUNDARY, read_boundary, points_inside, grid_mask

FINAL = "dense_s6_bwdb_r4"


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def moment_correlation(station_rows, monthly_rows):
    count = sum(int(r["n"]) for r in station_rows)
    if sum(int(r["n_days"]) for r in monthly_rows) != count:
        raise ValueError("monthly sufficient statistics do not cover the exact daily sample")
    sy = sum(int(r["n_days"]) * float(r["observed_mean_mm_day"]) for r in monthly_rows)
    sp = sum(int(r["n_days"]) * float(r["predicted_mean_mm_day"]) for r in monthly_rows)
    y2 = sum(int(r["n_days"]) * (float(r["observed_mean_mm_day"]) ** 2 + float(r["observed_daily_temporal_sd_mm"]) ** 2) for r in monthly_rows)
    p2 = sum(int(r["n_days"]) * (float(r["predicted_mean_mm_day"]) ** 2 + float(r["predicted_daily_temporal_sd_mm"]) ** 2) for r in monthly_rows)
    e2 = sum(int(r["n"]) * float(r["rmse"]) ** 2 for r in station_rows)
    covariance = (y2 + p2 - e2) / 2 - sy * sp / count
    denominator = np.sqrt((y2 - sy ** 2 / count) * (p2 - sp ** 2 / count))
    return float(covariance / denominator) if denominator > 0 else None


def pool(station_rows, monthly_rows):
    if not station_rows:
        raise ValueError("no station statistics inside country")
    n = sum(int(r["n"]) for r in station_rows)
    wet = sum(int(r["n_wet"]) for r in station_rows)
    linear = {k: sum(int(r["n"]) * float(r[k]) for r in station_rows) / n for k in ("crps", "mae", "bias", "coverage_90")}
    rms = {k: float(np.sqrt(sum(int(r["n"]) * float(r[k]) ** 2 for r in station_rows) / n)) for k in ("rmse", "spread")}
    conditional = {}
    for key, weight in (("dry_mae", lambda r: int(r["n"]) - int(r["n_wet"])), ("wet_mae", lambda r: int(r["n_wet"]))):
        denominator = sum(weight(r) for r in station_rows)
        conditional[key] = sum(weight(r) * float(r[key]) for r in station_rows if weight(r)) / denominator if denominator else None
    return {"n": n, "n_wet": wet, **linear, **rms, **conditional,
            "spread_skill": rms["spread"] / rms["rmse"], "correlation": moment_correlation(station_rows, monthly_rows)}


def catalogue(args):
    if args.station_catalog:
        return {r["station_id"]: (float(r["lat"]), float(r["lon"])) for r in read_csv(args.station_catalog)}, [args.station_catalog]
    import pandas as pd
    from bdhires.bmd import read_station_catalog, EXTRA_STATION_COORDS
    bmd = read_station_catalog(args.bmd_catalog)
    coords = {f"BMD_{r.station_id}": (r.lat, r.lon) for r in bmd.itertuples()}
    for row in EXTRA_STATION_COORDS.values():
        coords[f'BMD_{row["station_id"]}'] = (row["lat"], row["lon"])
    bwdb = pd.read_excel(args.bwdb_catalog, sheet_name="StationList")
    for _, row in bwdb.iterrows():
        station = str(row["Station ID"]).strip()
        point = (20.8925 if station == "CL312" else float(row["Latitude"]), float(row["Longitude"]))
        key = "BWDB_" + station
        if key in coords and coords[key] != point:
            raise ValueError("conflicting catalogue coordinates")
        coords[key] = point
    return coords, [args.bmd_catalog, args.bwdb_catalog, ROOT / "src/bdhires/bmd.py"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "paper1_cpcv2/superob-final")
    parser.add_argument("--output", type=Path, default=ROOT / "paper1_cpcv2/bangladesh/superob-final")
    parser.add_argument("--boundary-geojson", type=Path, default=DEFAULT_BOUNDARY)
    parser.add_argument("--station-catalog", type=Path, help="optional canonical CSV: station_id,lat,lon")
    parser.add_argument("--bmd-catalog", type=Path, default=ROOT / "data/stations/data_2020_2025/Stations.csv")
    parser.add_argument("--bwdb-catalog", type=Path, default=ROOT / "data/stations/BWDB_Rainfall_2000_2025_corrected.xlsx")
    args = parser.parse_args(argv)
    if args.input.resolve() == args.output.resolve():
        raise ValueError("country output must not overwrite supplied source exports")
    inputs = [args.input / "paper1_evaluation.json", args.input / "daily_withheld_scores.csv",
              args.input / "temporal_withheld_scores.csv", args.input / "gridded/long_term_withheld_station_scores.csv",
              args.input / "gridded/data/fig05_subgrid_case_fields.csv"]
    report = json.loads(inputs[0].read_text())
    if report["status"] != "withheld_evaluation_complete":
        raise ValueError("source must be a completed original evaluation")
    daily, temporal, products, fields = [read_csv(p) for p in inputs[1:]]
    country, region = read_boundary(args.boundary_geojson)
    coords, coordinate_inputs = catalogue(args)
    station_ids = sorted({r["label"] for r in daily if r["group"] == "station"})
    missing = set(station_ids) - set(coords)
    if missing:
        raise ValueError(f"missing original station coordinates: {sorted(missing)}")
    kept = {s for s in station_ids if points_inside(*coords[s], country)}
    outside = set(station_ids) - kept
    station_scores = [r for r in daily if r["group"] == "station"]
    for method in ("background", FINAL):
        all_stations = [r for r in station_scores if r["method"] == method]
        moments = [r for r in temporal if r["method"] == method and r["scale"] == "monthly"]
        reconstructed = pool(all_stations, moments)
        original = next(r for r in daily if r["group"] == "pooled" and r["method"] == method)
        for key in ("n", "crps", "rmse", "mae", "bias", "correlation", "spread", "coverage_90"):
            if not np.isclose(reconstructed[key], float(original[key]), rtol=1e-9, atol=1e-8):
                raise ValueError(f"sufficient statistics fail original-sample check: {method}/{key}")
    filtered_temporal = [r for r in temporal if r["station"] in kept]
    rows = []
    for group, label, ids in (("pooled", "all", kept), ("source", "BMD", {s for s in kept if not s.startswith("BWDB_")}),
                             ("source", "BWDB", {s for s in kept if s.startswith("BWDB_")})):
        for method in ("background", FINAL):
            stats = [r for r in station_scores if r["method"] == method and r["label"] in ids]
            moments = [r for r in filtered_temporal if r["method"] == method and r["station"] in ids and r["scale"] == "monthly"]
            rows.append({"scale": "daily", "group": group, "label": label, "method": method, **pool(stats, moments)})
    rows += [r for r in station_scores if r["label"] in kept]
    excluded_days = sum(int(r["n"]) for r in station_scores if r["label"] in outside and r["method"] == FINAL)
    counts = dict(report["counts"])
    counts["scored_station_days"] -= excluded_days
    counts["total_station_days"] = sum(len(np.arange(np.datetime64(entry["scope"]["start"]), np.datetime64(entry["scope"]["end"]) + 1)) * len(set(entry["scope"]["withheld_station_ids"]) & kept) for entry in report["archived_contracts"])
    counts["selection_station_days"] = counts["total_station_days"] - counts["scored_station_days"]
    counts["withheld_station_union"] = len(kept)
    region.update(excluded_station_ids=sorted(outside), excluded_scored_station_days=excluded_days,
                  included_station_ids=sorted(kept), station_coordinate_source="original input catalogues; archived NPZ coordinate identity pending full rerun")
    product_rows = []
    for source in dict.fromkeys(r["source"] for r in products):
        selected = [r for r in products if r["source"] == source and r["station_id"] in kept]
        n = sum(int(r["n"]) for r in selected)
        if n != counts["scored_station_days"]:
            raise ValueError("product table uses a different country sample")
        mse = sum(int(r["n"]) * float(r["rmse_mm"]) ** 2 for r in selected) / n
        product_rows.append({"source": source, "pooled_daily_n": n, "pooled_daily_rmse_mm": np.sqrt(mse),
                             "pooled_daily_mae_mm": sum(int(r["n"]) * float(r["mae_mm"]) for r in selected) / n,
                             "pooled_daily_bias_mm": sum(int(r["n"]) * float(r["bias_mm"]) for r in selected) / n,
                             "pooled_daily_correlation": next(r["correlation"] for r in rows if r["group"] == "pooled" and r["method"] == source) if source in ("background", FINAL) else ""})
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "daily_withheld_scores.csv", rows, list(daily[0]))
    write_csv(args.output / "temporal_withheld_scores.csv", filtered_temporal, list(temporal[0]))
    write_csv(args.output / "gridded/long_term_withheld_product_matrix.csv", product_rows)
    # Preserve the country-filtered sufficient statistics so product errors
    # can be compared by network without averaging station correlations.
    write_csv(args.output / "gridded/long_term_withheld_station_scores.csv",
              [r for r in products if r["station_id"] in kept])
    write_csv(args.output / "gridded/withheld_gauge_subgrid_anomalies.csv", [], ["method", "coarse_baseline", "correlation", "mse_skill_vs_no_subgrid"])
    write_csv(args.output / "station_country_audit.csv", [{"station_id": s, "lat": coords[s][0], "lon": coords[s][1], "inside_bangladesh": s in kept,
        "scored_station_days": next(int(r["n"]) for r in station_scores if r["label"] == s and r["method"] == FINAL)} for s in station_ids])
    for r in fields:
        inside = bool(points_inside(float(r["lat"]), float(r["lon"]), country))
        r["inside_bangladesh"] = inside
        if not inside:
            r["full_mm"], r["subgrid_mm"] = "", ""
    write_csv(args.output / "gridded/data/fig05_subgrid_case_fields.csv", fields)
    lats = sorted({float(r["lat"]) for r in fields}); lons = sorted({float(r["lon"]) for r in fields})
    region["country_grid_centres"] = int(grid_mask(lats, lons, country).sum())
    provenance = [{"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in [*inputs, *coordinate_inputs, args.boundary_geojson]]
    payload = {"status": "withheld_summary_recomputed", "evaluation_region": region,
               "contract": report["contract"], "profile": report["profile"], "periods": report["periods"],
               "counts": counts, "daily_scores": rows, "temporal_scores": filtered_temporal, "provenance": provenance,
               "archived_contracts": report["archived_contracts"], "paired_crps": [],
               "pending": ["equal-day block-bootstrap intervals", "period/intensity member-level scores",
                           "withheld subgrid anomaly scores", "full-grid daily/monthly/seasonal statistics",
                           "gridded-product pooled correlations", "archived station-coordinate identity",
                           "matched product errors by observed intensity",
                           "matched monthly/seasonal product errors at withheld gauges"],
               "case_selection": "retained 29 May 2024 illustrative case selected on original wider domain; country-only event reselection pending"}
    (args.output / "paper1_evaluation.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = ["# Bangladesh-only CPCv2 summary reevaluation", "", f"Scored: {counts['scored_station_days']} station-days; {len(kept)} original withheld sites; {counts['scored_dates']} dates.",
             f"Excluded: {', '.join(sorted(outside)) or 'none'}; {excluded_days} scored station-days.",
             "", "| Method | Fair CRPS | RMSE | MAE | Bias | r | Spread/RMSE | Coverage |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows[:2]:
        lines.append("| " + r["method"] + " | " + " | ".join(f'{r[k]:.3f}' for k in ("crps", "rmse", "mae", "bias", "correlation", "spread_skill", "coverage_90")) + " |")
    lines += ["", "Exact recomputation from verified per-station/month sufficient statistics; not a new ensemble inference run.", "", "Pending full archive rerun:", *["- " + item for item in payload["pending"]]]
    (args.output / "paper1_evaluation.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:8]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
