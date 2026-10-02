#!/usr/bin/env python3
"""Compare native daily IMERG against the same UTC day's 48 half hours on CPU.

Alternatively --audit-pilot inspects the inputs of a completed GPU pilot.
Neither mode changes source data, clips errors, or runs assimilation.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from bdhires.grids import BD
from bdhires.imerg import (_coarse_centres, _open_granule, _regional_array,
                          _require_mm_per_day, _require_mm_per_hour,
                          discover_imerg_files, discover_imerg_half_hourly_files)


def distribution(values):
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    result = {"size": int(array.size), "finite": int(finite.size),
              "negative": int((finite < 0).sum()), "above_1000": int((finite > 1000).sum())}
    for name, q in (("min", 0), ("median", 50), ("p95", 95), ("p99", 99), ("max", 100)):
        result[name] = float(np.percentile(finite, q)) if finite.size else None
    return result


def metadata(variable):
    keys = ("units", "long_name", "_FillValue", "missing_value", "valid_min", "valid_max",
            "valid_range", "scale_factor", "add_offset")
    def safe(value):
        if isinstance(value, list):
            return [safe(item) for item in value]
        if isinstance(value, float) and not np.isfinite(value):
            return str(value)
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value
    return {where: {key: safe(np.asarray(values[key]).tolist()) for key in keys if key in values}
            for where, values in (("attrs", variable.attrs), ("encoding", variable.encoding))}


def check_identity(path, manifest):
    expected = manifest.get("input_sha256", {}).get(str(path))
    if expected:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"pilot input changed after preparation: {path}")
    return "matches experiment manifest" if expected else "not recorded in experiment manifest"


def difference(a, b, mask):
    valid = np.isfinite(a) & np.isfinite(b) & mask
    av, bv = np.asarray(a)[valid], np.asarray(b)[valid]
    if not len(av):
        return {"n": 0, "mae": None, "rmse": None, "bias_daily_minus_halfhourly": None,
                "max_abs_difference": None, "within_0.001_mm_day": None}
    delta = av - bv
    return {"n": int(len(av)), "mae": float(np.abs(delta).mean()),
            "rmse": float(np.sqrt(np.mean(delta**2))),
            "bias_daily_minus_halfhourly": float(delta.mean()),
            "max_abs_difference": float(np.abs(delta).max()),
            "within_0.001_mm_day": int((np.abs(delta) <= .001).sum())}


def compare(day, daily_raw, halfhourly, output):
    day = date.fromisoformat(day)
    # The discovery helper labels windows by their END date. End at midnight
    # on D+1 to retrieve exactly D 00:00..23:30, matching the native daily file.
    end = (day + timedelta(days=1)).isoformat()
    daily_path = discover_imerg_files(daily_raw, day.isoformat(), day.isoformat())[0]
    files = discover_imerg_half_hourly_files(halfhourly, end, end, accumulation_end_hour_utc=0)
    lat, lon = _coarse_centres(BD, 2)
    with _open_granule(daily_path) as ds:
        _require_mm_per_day(ds, daily_path)
        native = {name: _regional_array(ds, name, lat, lon).astype(float)
                  for name in ("precipitation", "randomError", "precipitation_cnt", "randomError_cnt")
                  if name in ds}
        native_metadata = {name: metadata(ds[name]) for name in native}
    totals = np.zeros((len(lat), len(lon)))
    squared_errors = np.zeros_like(totals)
    p_count, e_count = np.zeros_like(totals, dtype=np.int16), np.zeros_like(totals, dtype=np.int16)
    halfhour_records = []
    for path in files:
        with _open_granule(path, required=frozenset({"precipitation", "randomError"})) as ds:
            _require_mm_per_hour(ds, path)
            p, e = (_regional_array(ds, name, lat, lon).astype(float)
                    for name in ("precipitation", "randomError"))
            record = {"path": str(path), "precipitation_mm_hour": distribution(p),
                      "random_error_mm_hour": distribution(e)}
            if not halfhour_records:
                record["metadata"] = {name: metadata(ds[name]) for name in ("precipitation", "randomError")}
        # Keep rainfall independent of error validity so a bad error cannot
        # silently alter the rainfall comparison. Zero rain/error remains valid.
        p_valid, e_valid = np.isfinite(p) & (p >= 0), np.isfinite(e) & (e >= 0)
        totals[p_valid] += .5 * p[p_valid]
        squared_errors[e_valid] += e[e_valid]**2
        p_count += p_valid
        e_count += e_valid
        halfhour_records.append(record)
    halfhour_total = np.where(p_count == 48, totals, np.nan)
    # Two different error definitions, both explicitly reported; neither is
    # silently substituted for the native daily error used in the original pilot.
    quadrature = np.where(e_count == 48, .5 * np.sqrt(squared_errors), np.nan)
    mean_squared = squared_errors / np.maximum(e_count, 1)
    documented_daily_error = np.where(e_count == 48, np.sqrt(24 * mean_squared), np.nan)
    rain_valid = (native["precipitation_cnt"] == 48) & (native["precipitation"] >= 0)
    error_valid = native["randomError"] >= 0
    if "randomError_cnt" in native:
        error_valid &= native["randomError_cnt"] == 48
    summary = {
        "utc_day": day.isoformat(), "region": "BD model box and halo, native 0.1 degree grid",
        "half_hourly_files": 48,
        "definitions": {"half_hourly_total_mm_day": "0.5 * sum(48 precipitation rates in mm/hour)",
                        "half_hourly_mean_mm_hour": "sum(48 rates) / 48; multiply by 24 to compare to native mm/day",
                        "half_hourly_quadrature_mm_day": "0.5 * sqrt(sum(48 squared error rates)); temporal independence",
                        "documented_daily_error_mm_day": "sqrt(24 * mean(squared error rates)); NASA catalogue formula",
                        "error_formula_source": "https://data.nasa.gov/dataset/gpm-imerg-final-precipitation-l3-1-day-0-1-degree-x-0-1-degree-v07-gpm-3imergdf-at-ges-dis-13ed8"},
        "daily_path": str(daily_path), "native_metadata": native_metadata,
        "native_distributions": {name: distribution(values) for name, values in native.items()},
        "half_hourly_precip_count": distribution(p_count), "half_hourly_error_count": distribution(e_count),
        "precip_count_mismatches": int((native["precipitation_cnt"] != p_count).sum()),
        "rainfall_daily_vs_half_hourly_total_mm_day": difference(native["precipitation"], halfhour_total, rain_valid),
        "error_daily_vs_half_hourly_quadrature_mm_day": difference(native["randomError"], quadrature, error_valid),
        "error_daily_vs_documented_daily_formula_mm_day": difference(native["randomError"], documented_daily_error, error_valid),
        "half_hourly_error_distribution_mm_day": distribution(quadrature),
        "half_hourly_files_detail": halfhour_records,
        "interpretation": "Same UTC window, V07B Final product, regional cells and rainfall units. "
                          "Rainfall and errors are checked independently; comparisons require 48 valid half hours. "
                          "Error definitions differ, so their difference alone does not establish a data defect."}
    if "randomError_cnt" in native:
        summary["error_count_mismatches"] = int((native["randomError_cnt"] != e_count).sum())
    fields = {"daily_precipitation": native["precipitation"], "half_hourly_total": halfhour_total,
              "rainfall_difference": np.where(rain_valid, native["precipitation"] - halfhour_total, np.nan),
              "daily_random_error": native["randomError"], "half_hourly_quadrature_error": quadrature,
              "documented_daily_error": documented_daily_error}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    xr.Dataset({name: (("lat", "lon"), values, {"units": "mm/day"}) for name, values in fields.items()},
               coords={"lat": lat, "lon": lon}, attrs={"utc_day": day.isoformat()}).to_netcdf(output / "fields.nc")
    (output / "comparison.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    lines = [f"# Native daily vs 48 half-hourly IMERG: {day} UTC", "", summary["interpretation"], "",
             "Differences are native daily minus half-hourly-derived; rainfall is in mm/day.", "",
             "| Comparison | Cells | MAE | RMSE | Mean difference | Maximum absolute difference |",
             "|---|---:|---:|---:|---:|---:|"]
    for key in ("rainfall_daily_vs_half_hourly_total_mm_day", "error_daily_vs_half_hourly_quadrature_mm_day",
                "error_daily_vs_documented_daily_formula_mm_day"):
        metric = summary[key]
        lines.append(f"| {key} | {metric['n']} | " + " | ".join(
            f"{metric[name]:.6f}" if metric[name] is not None else "NA"
            for name in ("mae", "rmse", "bias_daily_minus_halfhourly", "max_abs_difference")) + " |")
    lines += ["", "Rainfall cells agreeing within 0.001 mm/day: " + str(
        summary["rainfall_daily_vs_half_hourly_total_mm_day"]["within_0.001_mm_day"]),
        "", "Native daily randomError (mm/day): " + json.dumps(summary["native_distributions"]["randomError"]),
        "", "See comparison.json for native metadata, valid counts and error distributions."]
    (output / "comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[compare] results: {output}")
    return summary


def audit(root, daily_raw=None):
    root = Path(root)
    manifest = json.loads((root / "experiment.json").read_text())
    result = {"note": "Diagnostic only: no clipping, rescaling, input changes, or GPU sampling. "
              "The above_1000 threshold flags a tail for inspection; it is not a validity cutoff.",
              "prepared": {}, "raw_daily": []}
    prepared_errors = {}
    for label, name in (("reporting", "imerg_reporting_s04.nc"), ("calendar", "imerg_calendar_s04.nc")):
        path = root / name
        identity = check_identity(path, manifest)
        with xr.open_dataset(path) as ds:
            error = ds.randomError.values.copy()
            prepared_errors[label] = error
            result["prepared"][label] = {
                "path": str(path), "identity": identity,
                "precipitation_mm_day": distribution(ds.precipitation.values),
                "random_error_mm_day": distribution(error),
                "random_error_metadata": metadata(ds.randomError),
                "aggregation": ds.attrs.get("random_error_aggregation"),
                "spatial_sigma_scale": ds.attrs.get("random_error_scale_applied"),
                "counts": distribution(ds.precipitation_cnt.values)}
    a, b = prepared_errors["reporting"], prepared_errors["calendar"]
    if a.shape != b.shape:
        raise ValueError("prepared errors must have matching shapes")
    paired = np.isfinite(a) & np.isfinite(b) & (a > 0)
    result["calendar_to_reporting_error_ratio"] = distribution(b[paired] / a[paired])
    recorded = [Path(name) for name in manifest.get("input_sha256", {})
                if Path(name).name.startswith("3B-DAY.MS.MRG.3IMERG.") and name.endswith(".nc4")]
    if daily_raw is not None or not recorded:
        recorded = discover_imerg_files(daily_raw or "data/raw/imerg", "2022-04-30", "2022-05-04")
    if len(recorded) != 5:
        raise ValueError("audit requires the five original daily granules")
    lat, lon = _coarse_centres(BD, 2)
    for path in sorted(recorded):
        identity = check_identity(path, manifest)
        with _open_granule(path) as ds:
            arrays = {name: _regional_array(ds, name, lat, lon)
                      for name in ("precipitation", "randomError", "precipitation_cnt", "randomError_cnt")
                      if name in ds}
            record = {"path": str(path), "identity": identity,
                      "variables": {name: {"distribution": distribution(values), "metadata": metadata(ds[name])}
                                    for name, values in arrays.items()}}
        p, e, count = (arrays[name] for name in ("precipitation", "randomError", "precipitation_cnt"))
        accepted = np.isfinite(p) & np.isfinite(e) & (p >= 0) & (e >= 0) & (count >= 48)
        record["error_on_pilot_accepted_native_cells"] = distribution(e[accepted])
        if "randomError_cnt" in arrays:
            error_count = arrays["randomError_cnt"]
            record["accepted_precipitation_with_incomplete_error_count"] = int((accepted & (error_count < 48)).sum())
            record["accepted_precipitation_with_nonfinite_error_count"] = int((accepted & ~np.isfinite(error_count)).sum())
            record["error_with_full_error_count"] = distribution(e[accepted & (error_count == 48)])
        top = np.flatnonzero(accepted.ravel())
        top = top[np.argsort(e.ravel()[top])[-5:][::-1]]
        record["largest_accepted_errors"] = []
        for index in top:
            iy, ix = np.unravel_index(index, e.shape)
            record["largest_accepted_errors"].append({"lat": float(lat[iy]), "lon": float(lon[ix]),
                **{name: float(values[iy, ix]) if np.isfinite(values[iy, ix]) else None
                   for name, values in arrays.items()}})
        result["raw_daily"].append(record)
    output = root / "imerg_error_audit.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("IMERG randomError (mm/day):")
    for label, record in result["prepared"].items():
        print(label + ": " + json.dumps(record["random_error_mm_day"]))
    print("B/A error ratio: " + json.dumps(result["calendar_to_reporting_error_ratio"]))
    for record in result["raw_daily"]:
        print(Path(record["path"]).name + ": " + json.dumps(record["error_on_pilot_accepted_native_cells"]))
        if "accepted_precipitation_with_incomplete_error_count" in record:
            print("  accepted cells with error count <48: " + str(record["accepted_precipitation_with_incomplete_error_count"]))
    print(f"[audit] metadata, counts and largest errors: {output}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--day", default="2022-05-01", help="calendar UTC day to compare")
    parser.add_argument("--halfhourly", default="data/imerg_halfhourly/2022")
    parser.add_argument("--out", default=None, help="comparison output directory")
    parser.add_argument("--audit-pilot", action="store_true", help="audit a completed GPU pilot instead of comparing raw files")
    parser.add_argument("--root", default="data/processed/imerg_daily_window_pilot_production_stations")
    parser.add_argument("--daily-raw", help="daily granule directory; default data/raw/imerg, or manifest paths in audit mode")
    args = parser.parse_args()
    if args.audit_pilot:
        audit(args.root, args.daily_raw)
    else:
        compare(args.day, args.daily_raw or "data/raw/imerg", args.halfhourly,
                args.out or f"data/processed/imerg_daily_halfhourly_{args.day.replace('-', '')}")


if __name__ == "__main__":
    main()
