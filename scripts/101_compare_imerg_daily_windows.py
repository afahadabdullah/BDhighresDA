#!/usr/bin/env python3
"""One-member, five-day timing pilot using the final CPCv2 production method.

Compare the SAME gauge reports dated 2022-05-01..05, on the SAME background
dates 2022-04-30..05-04. Case A retains reporting labels and offset -1 with
03–03 UTC IMERG. Case B relabels all gauges -1 day and uses offset 0 with
native UTC daily IMERG. This changes satellite support and native error
aggregation, not the underlying predictor/gauge pairing.

Run in the existing GH200 environment; --dry-run needs only Python's stdlib.
See docs/IMERG_DAILY_WINDOW_PILOT.md for invocation and interpretation.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
METHOD = "dense_s6_bwdb_r4"
START, END = "2022-05-01", "2022-05-05"
PRODUCTION_STATION_ROOT = "data/processed/v2_bmd_bwdb_superob_2021_2024/stations/2022_may_sep"


def shift_day(value):
    return (date.fromisoformat(value) - timedelta(days=1)).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def run(script, *arguments):
    command = [sys.executable, "-u", str(REPO / "scripts" / script), *map(str, arguments)]
    print("[pilot] " + shlex.join(command), flush=True)
    subprocess.run(command, cwd=REPO, check=True)


def paths(args):
    root = Path(args.root)
    return {"root": root, "gauges": root / "gauges_reporting.csv",
            "shifted": root / "gauges_calendar.csv", "superobs": root / "superobs.json",
            "a": root / "imerg_reporting_s04.nc", "b": root / "imerg_calendar_s04.nc"}


def station_preparation_arguments(args, original, root):
    """Prefer the national wide table; use the established 2022 directory if absent."""
    for path in (args.bmd_catalog, args.bwdb):
        if not Path(path).is_file():
            raise FileNotFoundError(f"required gauge source missing: {path}")
    directory = args.bmd_data_dir
    if directory is None and not Path(args.bmd_wide).is_file():
        directory = "data/stations/data_2020_2025"
    if directory is not None:
        files = [path for path in Path(directory).glob("*.csv")
                 if path.resolve() != Path(args.bmd_catalog).resolve()
                 and not path.name.lower().endswith("stations.csv")]
        if not files:
            raise FileNotFoundError(
                f"no per-station BMD CSVs found in {directory}; provide --bmd-data-dir, "
                "--bmd-wide, or --stations pointing to an existing original combined gauge table")
        print(f"[pilot] using per-station BMD source: {directory}", flush=True)
        source = ["--bmd-data-dir", directory]
    else:
        print(f"[pilot] using wide BMD history: {args.bmd_wide}", flush=True)
        source = ["--bmd-wide", args.bmd_wide]
    return ["--start", START, "--end", END, *source,
            "--bmd-stations", args.bmd_catalog, "--bwdb-xlsx", args.bwdb,
            "--out", original, "--summary", root / "station_summary.csv",
            "--report", root / "station_preparation.json"]


def reuse_production_gauges(args, p):
    """Reuse the final seasonal station preparation, including its frozen error budget."""
    import numpy as np
    import pandas as pd
    from bdhires.grids import BD
    explicit = args.production_station_root is not None
    if not explicit and (args.stations or args.bmd_data_dir
                         or args.bmd_wide != "data/stations/Rainfall_daily_by_station_BMD.csv"):
        return None
    root = Path(args.production_station_root or PRODUCTION_STATION_ROOT)
    table = root / "superob_prod_0.25.csv"
    report = root / "superob_prod_0.25.json"
    if not explicit and not table.exists() and not report.exists():
        return None
    for path in (table, report):
        if not path.is_file():
            raise FileNotFoundError(f"archived production station input missing: {path}")
    budget = json.loads(report.read_text())
    recommendation = budget.get("recommended_representativeness") or {}
    representation = recommendation.get("superob_implied_representativeness")
    if representation is None or not np.isfinite(representation) or not 0 <= representation < 10:
        raise ValueError("archived production report must supply the measured superob representativeness")
    if budget.get("cell_deg") != .25 or budget.get("stations_held_out") != 0:
        raise ValueError("use the all-station 0.25° production table/report, not evaluation super-observations")
    frame = pd.read_csv(table, parse_dates=["date"], dtype={"station_id": str})
    required = {"station_id", "date", "lat", "lon", "precip_mm"}
    if not required <= set(frame) or frame.duplicated(["station_id", "date"]).any():
        raise ValueError("invalid archived production station schema or duplicate station-days")
    season = pd.date_range("2022-05-01", "2022-09-30")
    if frame.date.min() > season[0] or frame.date.max() < season[-1]:
        raise ValueError("production station table must cover the full May–September 2022 season")
    frame = frame.loc[frame.date.isin(season)].copy()
    if (frame.precip_mm.dropna() < 0).any():
        raise ValueError("archived production rainfall contains negative values")
    # Match the older GPU launcher's default 50% SEASONAL coverage filter.
    # Keep seasonally eligible stations even if they have no reports in these five days.
    meta = frame.groupby("station_id").first()
    coverage = frame.groupby("station_id").precip_mm.count() / len(season)
    lo, la, hi, ha = BD.bbox
    margin = BD.res / 2
    eligible = meta.index[(coverage >= .5) & meta.lat.between(la + margin, ha - margin, inclusive="neither")
                          & meta.lon.between(lo + margin, hi - margin, inclusive="neither")]
    if len(eligible) < 5:
        raise ValueError("fewer than five archived production stations survive the original seasonal filter")
    keys = pd.MultiIndex.from_product([eligible, pd.date_range(START, END)], names=["station_id", "date"])
    window = frame.set_index(["station_id", "date"]).reindex(keys).reset_index()
    # Preserve missing rainfall. Only static metadata is filled from the seasonal table.
    for name in ("lat", "lon", "name", "source", "accumulation_end_hour_utc"):
        if name in window:
            window[name] = window[name].fillna(window.station_id.map(meta[name]))
    window.to_csv(p["gauges"], index=False, date_format="%Y-%m-%d")
    p["superobs"].write_bytes(report.read_bytes())
    print(f"[pilot] reusing archived production stations: {table}; {len(eligible)} stations; R representativeness={representation}", flush=True)
    return {"budget": recommendation, "inputs": [table, report],
            "provenance": {"mode": "archived production super-observations",
                           "table": str(table), "report": str(report),
                           "eligibility_period": ["2022-05-01", "2022-09-30"],
                           "eligible_stations": len(eligible),
                           "preparation_stats_provenance": budget.get("stats_provenance"),
                           "note": "Preserves the archived error budget; its original preparation-statistics identity is recorded, not retroactively changed."}}


def download_daily(directory):
    """Fetch exactly five official daily granules, atomically; never a full year."""
    from bdhires.imerg import _open_granule, _require_mm_per_day
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    first, last = date.fromisoformat(shift_day(START)), date.fromisoformat(shift_day(END))
    for offset in range((last - first).days + 1):
        day = first + timedelta(days=offset)
        name = f"3B-DAY.MS.MRG.3IMERG.{day:%Y%m%d}-S000000-E235959.V07B.nc4"
        destination = directory / name
        if destination.is_file():
            with _open_granule(destination) as ds:
                _require_mm_per_day(ds, destination)
            continue
        url = f"https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGDF.07/{day:%Y/%m}/{name}"
        temporary = destination.with_suffix(".nc4.part")
        command = ["wget", "--auth-no-challenge=on", "--tries=5", "--timeout=120",
                   "--waitretry=5", "--retry-connrefused", "-q", "-O", str(temporary), url]
        cookies = Path.home() / ".urs_cookies"
        if cookies.is_file():
            command[1:1] = ["--load-cookies", str(cookies)]
        print(f"[pilot] downloading daily IMERG {day}", flush=True)
        subprocess.run(command, check=True)
        with _open_granule(temporary) as ds:
            _require_mm_per_day(ds, temporary)
        temporary.replace(destination)


def prepare_calendar_imerg(directory, output, *, error_mode="native", halfhourly=None,
                           validation_root=None, validation_inputs=None):
    import numpy as np
    import xarray as xr
    from bdhires.imerg import (_open_granule, _regional_array, load_imerg_daily,
                              validate_prepared_time_convention)
    if error_mode not in ("native", "verified-sum-squared"):
        raise ValueError(f"unknown daily error mode: {error_mode}")
    daily = load_imerg_daily(directory, shift_day(START), shift_day(END), min_count=48)
    error = daily.random_error.copy()
    aggregation = "native NASA daily product; raw values unchanged"
    reports = []
    counts_for_preparation = daily.count.copy()
    if error_mode == "verified-sum-squared":
        if halfhourly is None or validation_root is None:
            raise ValueError("verified sum-squared errors require half-hourly inputs and a validation output root")
        for i, day in enumerate(daily.time.astype("datetime64[D]").astype(str)):
            check_root = Path(validation_root) / day.replace("-", "")
            run("102_compare_imerg_daily_halfhourly.py", "--day", day,
                "--daily-raw", directory, "--halfhourly", halfhourly, "--out", check_root)
            report_path = check_root / "comparison.json"
            report = json.loads(report_path.read_text())
            check = report["error_encoding_check"]
            if report["utc_day"] != day or not check["all_cells_match"] or check["valid_cells"] == 0:
                raise ValueError(f"daily sum-squared error encoding not verified on {day}; inspect {report_path}")
            # Error validity is independent of precipitation validity. The native
            # reader's precipitation count alone cannot establish complete errors.
            with _open_granule(Path(directory) / daily.source_files[i]) as source:
                if "randomError_cnt" not in source:
                    raise ValueError(f"daily error counts required for verified conversion: {day}")
                counts = _regional_array(source, "randomError_cnt", daily.lat, daily.lon)
            usable = np.isfinite(daily.precipitation[i]) & np.isfinite(error[i])
            if not usable.any() or not np.all(counts[usable] == 48):
                raise ValueError(f"daily IMERG has incomplete error counts on usable pilot footprints: {day}")
            corrected = .5 * np.sqrt(error[i])
            with xr.open_dataset(check_root / "fields.nc") as source:
                quadrature = source.half_hourly_quadrature_error.values
            if not np.isfinite(quadrature[usable]).all() or not np.allclose(
                    corrected[usable], quadrature[usable], rtol=1e-5, atol=1e-5):
                raise ValueError(f"daily error conversion does not reproduce half-hourly quadrature on {day}")
            error[i] = corrected
            reports.append(str(report_path))
            if validation_inputs is not None:
                validation_inputs.extend([report_path, check_root / "fields.nc",
                                          *[Path(item["path"]) for item in report["half_hourly_files_detail"]]])
            print(f"[pilot] {day}: verified sum-squared errors on {check['valid_cells']} cells; "
                  "applied 0.5 * sqrt(raw daily randomError) before coarsening", flush=True)
        aggregation = ("0.5 * sqrt(raw daily sum of half-hourly squared error rates); "
                       "temporal-independence baseline; verified for every pilot UTC day")
        # Spatial NaN reductions must not hide an unusable native footprint.
        counts_for_preparation[~np.isfinite(daily.precipitation) | ~np.isfinite(error)] = 0
    # Native mm/day values are already daily mean rates * 24. Do not rescale.
    dataset = xr.Dataset(
        {"precipitation": (("time", "lat", "lon"), daily.precipitation, {"units": "mm/day"}),
         "randomError": (("time", "lat", "lon"), error,
                         {"units": "mm/day", "aggregation": aggregation}),
         "precipitation_cnt": (("time", "lat", "lon"), counts_for_preparation)},
        coords={"time": daily.time, "lat": daily.lat, "lon": daily.lon},
        attrs={"product": "GPM_3IMERGDF", "version": "V07B", "source_frequency": "daily",
               "bmd_accumulation_end_hour_utc": 0, "window_duration_hours": 24,
               "time_coordinate_semantics": "UTC calendar day; window start",
               "accumulation_window": "selected-day 00:00 UTC to next-day 00:00 UTC",
               "random_error_aggregation": aggregation,
               "daily_random_error_mode": error_mode,
               "error_encoding_validation_reports": json.dumps(reports),
               "quality_control": "all 48 half-hourly retrievals required",
               "source_files": json.dumps(list(daily.source_files))})
    validate_prepared_time_convention(dataset, allow_calendar_day=True)
    for values in (daily.precipitation, error):
        if not np.isfinite(values).reshape(5, -1).any(axis=1).all():
            raise ValueError("a native daily IMERG field has no valid regional footprints")
    dataset.to_netcdf(output)
    return [Path(directory) / name for name in daily.source_files]


def enforce_complete_footprints(path, *, calendar=False):
    """A coarsened observation must not hide an incomplete native footprint."""
    import numpy as np
    import xarray as xr
    from bdhires.imerg import validate_prepared_time_convention
    with xr.open_dataset(path) as source:
        ds = source.load()
    validate_prepared_time_convention(ds, allow_calendar_day=calendar)
    valid = (ds.precipitation_cnt >= 48) & np.isfinite(ds.precipitation) & np.isfinite(ds.randomError)
    valid &= (ds.precipitation >= 0) & (ds.randomError >= 0)
    ds["precipitation"] = ds.precipitation.where(valid)
    ds["randomError"] = ds.randomError.where(valid)
    if not np.isfinite(ds.precipitation.values).reshape(5, -1).any(axis=1).all():
        raise ValueError(f"no complete IMERG footprints on a pilot day: {path}")
    ds.to_netcdf(path)


def prepare(args):
    import numpy as np
    import pandas as pd
    import xarray as xr
    from bdhires.grids import BD
    p = paths(args)
    p["root"].mkdir(parents=True, exist_ok=True)
    for required in (args.ckpt, args.stats, args.config, args.data_zarr):
        if not Path(required).exists():
            raise FileNotFoundError(f"required existing project input missing: {required}")
    import importlib.util
    import torch
    spec = importlib.util.spec_from_file_location("_pilot_production", REPO / "scripts/100_surma_production.py")
    production = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(production)
    if sha(args.ckpt) != production.CHECKPOINT_SHA or sha(args.stats) != production.STATS_SHA:
        raise ValueError("use the evaluated CPCv2 checkpoint and statistics; input identity differs")
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    if str(checkpoint["cfg"]["data"]["stats"]) != str(args.stats):
        raise ValueError("--stats must match the checkpoint-bound statistics path")
    del checkpoint
    archived = reuse_production_gauges(args, p)
    original = None
    if archived:
        budget = archived["budget"]
    else:
        original = p["root"] / "original_gauges.csv"
        if args.stations:
            frame = pd.read_csv(args.stations, parse_dates=["date"], dtype={"station_id": str})
            required_columns = {"station_id", "lat", "lon", "date", "precip_mm"}
            if not required_columns <= set(frame):
                raise ValueError("--stations must be an original canonical daily gauge table")
            frame = frame.loc[frame.date.between(START, END)].copy()
            if frame.station_id.str.startswith("SOB_").any():
                raise ValueError("use --production-station-root for archived super-observations")
            frame.to_csv(original, index=False)
        else:
            run("99_prepare_production_stations.py",
                *station_preparation_arguments(args, original, p["root"]))
        frame = pd.read_csv(original, parse_dates=["date"], dtype={"station_id": str})
        if frame.duplicated(["station_id", "date"]).any():
            raise ValueError("duplicate original station-days")
        frame = frame.loc[frame.lat.between(float(BD.lat[0]), float(BD.lat[-1]))
                          & frame.lon.between(float(BD.lon[0]), float(BD.lon[-1]))].copy()
        finite = np.isfinite(frame.precip_mm) & (frame.precip_mm >= 0)
        frame.loc[~finite, "precip_mm"] = np.nan
        counts = frame.groupby("station_id").precip_mm.count()
        frame = frame.loc[frame.station_id.isin(counts[counts >= 3].index)].copy()
        if frame.empty:
            raise ValueError("no eligible original gauges in the pilot")
        frame.to_csv(original, index=False)
        run("87_superob_dense_gauges.py", "--stations", original, "--stats", args.stats,
            "--cell-deg", .25, "--out", p["gauges"], "--report", p["superobs"])
        budget = json.loads(p["superobs"].read_text())["recommended_representativeness"]
        if budget is None:
            raise ValueError("cannot establish measured super-observation error budget")
    gauges = pd.read_csv(p["gauges"], parse_dates=["date"], dtype={"station_id": str})
    gauges["original_report_date"] = gauges.date.dt.strftime("%Y-%m-%d")
    gauges["date"] -= pd.Timedelta(days=1)
    # Shift BOTH BMD and BWDB to preserve the production network and its exact values.
    gauges.to_csv(p["shifted"], index=False, date_format="%Y-%m-%d")

    aligned = args.aligned_imerg
    if not aligned:
        aligned = next((str(path) for path in (
            Path("data/processed/v2_confirmatory_2021_2024/imerg_s04/2022_may_sep.nc"),
            Path("data/processed/imerg_bd_aligned_20220501_20220531.nc")) if path.is_file()), None)
    if aligned:
        native_a = p["root"] / "imerg_reporting_subset.nc"
        run("43_subset_prepared_imerg.py", "--input", aligned, "--start", START, "--end", END,
            "--out", native_a)
    else:
        native_a = p["root"] / "imerg_reporting_subset.nc"
        if args.download_imerg:
            run("02_download_imerg_halfhourly.py", "--bmd-start", START, "--bmd-end", END,
                "--end-hour-utc", 3, "--out", args.halfhourly)
        run("08_prepare_imerg_observations.py", "--input", args.halfhourly,
            "--source-frequency", "half-hourly", "--start", START, "--end", END,
            "--min-count", 48, "--accumulation-end-hour-utc", 3, "--out", native_a,
            "--report", p["root"] / "imerg_reporting_qc.json")
    with xr.open_dataset(native_a) as ds:
        factor = 128 // ds.sizes["lat"]
        if factor == 8 and ds.sizes["lon"] == 16:
            if ds.attrs.get("observation_factor") != 8 or not np.isclose(
                    float(ds.attrs.get("required_error_corr_cells", -1)), .75):
                raise ValueError("existing S04 IMERG lacks the evaluated correlated-error metadata")
            ds.load().to_netcdf(p["a"])
        elif ds.sizes["lat"] == 64 and ds.sizes["lon"] == 64:
            run("44_coarsen_imerg_observations.py", "--input", native_a, "--factor", 8, "--out", p["a"])
        else:
            raise ValueError("--aligned-imerg must be native 0.1° or evaluated 0.4° IMERG")
    if args.download_imerg:
        download_daily(args.daily_raw)
        if args.daily_error_mode == "verified-sum-squared":
            # Calendar B needs Apr 30 00:00..May 5 00:00. Existing regional
            # files are skipped; an older 03 UTC archive may lack Apr 30's first six.
            run("02_download_imerg_halfhourly.py", "--bmd-start", START, "--bmd-end", END,
                "--end-hour-utc", 0, "--out", args.halfhourly)
    native_b = p["root"] / "imerg_calendar_native.nc"
    error_validation_inputs = []
    daily_files = prepare_calendar_imerg(args.daily_raw, native_b, error_mode=args.daily_error_mode,
        halfhourly=args.halfhourly, validation_root=p["root"] / "error_validation",
        validation_inputs=error_validation_inputs)
    run("44_coarsen_imerg_observations.py", "--input", native_b, "--factor", 8, "--out", p["b"])
    enforce_complete_footprints(p["a"])
    enforce_complete_footprints(p["b"], calendar=True)
    production.validate_imerg(p["a"], START, END, factor=8)
    inputs = [Path(args.ckpt), Path(args.stats), Path(args.config), p["gauges"], p["superobs"],
              p["shifted"], p["a"], p["b"], native_b, *daily_files, *error_validation_inputs]
    preparation_report = p["root"] / "station_preparation.json"
    if original:
        inputs.append(original)
    if not args.stations and not archived:
        inputs.append(preparation_report)
    if archived:
        inputs.extend(archived["inputs"])
    if aligned:
        inputs.append(Path(aligned))
    manifest = {"gauge_report_dates": [START, END],
                "calendar_background_dates": [shift_day(START), shift_day(END)],
                "members": 1, "method": METHOD, "data_zarr": args.data_zarr,
                "daily_random_error_mode": args.daily_error_mode,
                "daily_error_validation_days": [shift_day(START), shift_day(END)]
                    if args.daily_error_mode == "verified-sum-squared" else None,
                "statistics": args.stats, "checkpoint": args.ckpt,
                "representativeness": budget["superob_implied_representativeness"],
                "station_preparation": archived["provenance"] if archived else {
                    "mode": "rebuilt five-day super-observations", "eligibility_period": [START, END]},
                "case_a_seed": args.seed, "case_b_seed": args.seed + 1,
                "seed_note": "sweep seeds add observation archive index; +1 offsets B's one-day earlier labels",
                "input_sha256": {str(path): sha(path) for path in inputs},
                "caveats": ["One member; sensitivity only, no ensemble calibration or independent skill claim.",
                            "BMD support 00–00 UTC; BWDB support 03–03 UTC; both relabelled -1 day in B.",
                            ("Daily errors converted at native resolution using 0.5 * sqrt(raw sum squared rates); "
                             "verified against half-hourly quadrature for all five calendar days."
                             if args.daily_error_mode == "verified-sum-squared" else
                             "Daily raw randomError retained without encoding validation; this arm confounds timing and error interpretation."),
                            "CHIRPS is diagnostic only; common calendar-day reference used for comparison."]}
    write_json(p["root"] / "experiment.json", manifest)
    return manifest


def case_commands(args, representation):
    p = paths(args)
    common = ["--config", args.config, "--ckpt", args.ckpt, "--data-zarr", args.data_zarr,
              "--members", "1", "--group", "v2_bmd_bwdb_superob_winner",
              "--assimilate-all-stations", "--min-coverage", "0.0",
              "--set", "observations.imerg.factor=8",
              "--set", "observations.imerg.error_corr_cells=0.75",
              "--set", f"observations.gauges.representativeness={representation}"]
    cases = []
    for label, first, last, offset, seed, stations, imerg in (
        ("reporting", START, END, -1, args.seed, p["gauges"], p["a"]),
        ("calendar", shift_day(START), shift_day(END), 0, args.seed + 1, p["shifted"], p["b"])):
        command = common + ["--start", first, "--end", last, "--background-day-offset", str(offset),
                            "--seed", str(seed), "--stations", str(stations), "--imerg", str(imerg),
                            "--out", str(p["root"] / f"{label}.npz"),
                            "--report", str(p["root"] / f"{label}.json")]
        if label == "calendar":
            command.append("--allow-calendar-day-imerg")
        cases.append(command)
    return cases


def difference_metrics(a, b, mask=None):
    """Signed difference is B minus A; exclude missing footprints pairwise."""
    import numpy as np
    a, b = np.asarray(a, float), np.asarray(b, float)
    valid = np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        valid &= np.broadcast_to(mask, a.shape)
    av, bv = a[valid], b[valid]
    if not len(av):
        return {"n": 0, "mae": None, "rmse": None, "bias_b_minus_a": None, "correlation": None}
    delta = bv - av
    correlation = float(np.corrcoef(av, bv)[0, 1]) if len(av) > 1 and av.std() > 0 and bv.std() > 0 else None
    return {"n": int(len(av)), "mae": float(np.abs(delta).mean()),
            "rmse": float(np.sqrt(np.mean(delta ** 2))), "bias_b_minus_a": float(delta.mean()),
            "correlation": correlation, "max_abs_difference": float(np.abs(delta).max())}


def summarize(args):
    import csv
    import numpy as np
    import xarray as xr
    p = paths(args)
    manifest = json.loads((p["root"] / "experiment.json").read_text())
    for path, expected_hash in manifest["input_sha256"].items():
        if sha(path) != expected_hash:
            raise ValueError(f"pilot input changed after preparation: {path}")
    with np.load(p["root"] / "reporting.npz", allow_pickle=False) as source:
        a = {key: source[key] for key in source.files}
    with np.load(p["root"] / "calendar.npz", allow_pickle=False) as source:
        b = {key: source[key] for key in source.files}
    expected_a = np.arange(np.datetime64(START), np.datetime64(END) + np.timedelta64(1, "D"))
    expected_b = expected_a - np.timedelta64(1, "D")
    if not (np.array_equal(a["times"].astype("datetime64[D]"), expected_a)
            and np.array_equal(b["times"].astype("datetime64[D]"), expected_b)
            and np.array_equal(a["model_times"].astype("datetime64[D]"), expected_b)
            and np.array_equal(b["model_times"].astype("datetime64[D]"), expected_b)):
        raise ValueError("cases do not refer to the same physical background dates")
    for key in ("station_ids", "station_lat", "station_lon", "assim_idx", "eval_idx", "gauge_mm",
                "condition", "valid", "grid_lat", "grid_lon"):
        equal = np.array_equal(a[key], b[key]) if a[key].dtype.kind in "US" else np.array_equal(a[key], b[key], equal_nan=True)
        if not equal:
            raise ValueError(f"unmatched case input: {key}")
    scopes = []
    for label, dump in (("reporting", a), ("calendar", b)):
        report = json.loads((p["root"] / f"{label}.json").read_text())
        scope = report["scope"]
        scopes.append(scope)
        if scope["members"] != 1 or scope["checkpoint_stats"] != manifest["statistics"]:
            raise ValueError("pilot must use one member and the declared checkpoint-bound statistics")
        if scope["assimilate_all_stations"] is not True or len(dump["eval_idx"]):
            raise ValueError("this pilot requires the same all-station production setting")
        if dump[f"station_{METHOD}"].shape[1] != 1:
            raise ValueError("station archive does not contain exactly one member")
    for key in ("checkpoint", "checkpoint_data", "checkpoint_stats", "group", "config_overrides",
                "analysis_sampler_n_steps", "analysis_sampler_n_corrections", "analysis_sampler_heun"):
        if scopes[0][key] != scopes[1][key]:
            raise ValueError(f"unmatched sampler configuration: {key}")
    if not np.allclose(a["meanfield_background"], b["meanfield_background"], rtol=0, atol=1e-5, equal_nan=True):
        raise ValueError("background draws differ; predictor selection or random seeds are not matched")
    valid = a["valid"].astype(bool)
    af, bf = a[f"meanfield_{METHOD}"], b[f"meanfield_{METHOD}"]
    if not np.isfinite(af[:, valid]).all() or not np.isfinite(bf[:, valid]).all():
        raise ValueError("non-finite analysis rainfall on land")
    with xr.open_dataset(p["a"]) as source:
        ae = source.randomError.values.copy()
    with xr.open_dataset(p["b"]) as source:
        be = source.randomError.values.copy()
    rows = []
    for i, day in enumerate(expected_a.astype(str)):
        metrics = difference_metrics(af[i], bf[i], valid)
        rows.append({"gauge_report_date": day, "calendar_date": str(expected_b[i]), **metrics,
                     "analysis_a_land_mean_mm": float(af[i][valid].mean()),
                     "analysis_b_land_mean_mm": float(bf[i][valid].mean())})
    summary = {"scope": manifest, "matched_background_max_abs_difference":
               float(np.nanmax(np.abs(a["meanfield_background"] - b["meanfield_background"]))),
               "analysis_mm_day": difference_metrics(af, bf, valid),
               "five_day_total_mm": difference_metrics(af.sum(axis=0), bf.sum(axis=0), valid),
               "imerg_precipitation_mm_day": difference_metrics(a["raw_imerg_mm"], b["raw_imerg_mm"]),
               "imerg_random_error_mm_day": difference_metrics(ae, be),
               "assimilated_superob_fit_a": difference_metrics(a["gauge_mm"], a[f"station_{METHOD}"][:, 0]),
               "assimilated_superob_fit_b": difference_metrics(b["gauge_mm"], b[f"station_{METHOD}"][:, 0]),
               "analysis_vs_common_chirps_a": difference_metrics(b["chirps"], af, valid),
               "analysis_vs_common_chirps_b": difference_metrics(b["chirps"], bf, valid),
               "daily": rows,
               "interpretation": "Sensitivity of one paired draw; gauge fits are assimilated fits and CHIRPS is a common reference, not independent truth."}
    write_json(p["root"] / "comparison.json", summary)
    with (p["root"] / "daily_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(p["root"] / "comparison_fields.npz", gauge_report_dates=expected_a.astype(str),
                        calendar_dates=expected_b.astype(str), grid_lat=a["grid_lat"], grid_lon=a["grid_lon"],
                        valid=valid, reporting=af, calendar=bf, difference=bf - af,
                        background=a["meanfield_background"], common_chirps=b["chirps"])
    plot_comparison(p["root"], a, af, bf, valid, expected_a)
    lines = ["# Daily IMERG timing pilot: 1–5 May 2022 gauge records", "",
             "One member per day; matched CPC/ERA5 backgrounds, gauge values and random draws.", "",
             f"Station preparation: {manifest.get('station_preparation', {}).get('mode', 'not recorded')}; "
             f"gauge representativeness: {manifest.get('representativeness', 'not recorded')}.", "",
             f"Daily IMERG error mode: {manifest.get('daily_random_error_mode', 'native (legacy pilot)')}.", "",
             "Differences below are calendar IMERG (B) minus reporting-window IMERG (A).", "",
             "| Quantity | MAE | RMSE | Mean difference |", "|---|---:|---:|---:|"]
    for key in ("analysis_mm_day", "five_day_total_mm", "imerg_precipitation_mm_day", "imerg_random_error_mm_day"):
        m = summary[key]
        if m["n"]:
            lines.append(f"| {key} | {m['mae']:.3f} | {m['rmse']:.3f} | {m['bias_b_minus_a']:.3f} |")
    lines += ["", summary["interpretation"], "", *[f"- {text}" for text in manifest["caveats"]]]
    (p["root"] / "comparison.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary["analysis_mm_day"], indent=2))
    print(f"[pilot] results: {p['root'] / 'comparison.md'}", flush=True)


def plot_comparison(root, dump, af, bf, valid, days):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(6, 3, figsize=(11, 17), constrained_layout=True)
    extent = [dump["grid_lon"][0], dump["grid_lon"][-1], dump["grid_lat"][0], dump["grid_lat"][-1]]
    for i in range(6):
        left, right = (af[i], bf[i]) if i < 5 else (af.sum(axis=0), bf.sum(axis=0))
        scale = max(float(np.percentile(np.concatenate([left[valid], right[valid]]), 99)), 1.)
        delta = right - left
        bound = max(float(np.percentile(np.abs(delta[valid]), 99)), .1)
        for j, field in enumerate((left, right, delta)):
            im = axes[i, j].imshow(np.where(valid, field, np.nan), origin="lower", extent=extent,
                                   cmap="BrBG" if j == 2 else "YlGnBu", vmin=-bound if j == 2 else 0,
                                   vmax=bound if j == 2 else scale)
            if i < 5:
                report_date = str(np.datetime64(days[i], "D"))
                calendar_date = str(np.datetime64(days[i], "D") - np.timedelta64(1, "D"))
                title = (f"{report_date}: A reporting", f"{calendar_date}: B calendar",
                         f"{report_date} report: B − A")[j]
            else:
                title = f"Five-day total: {('A reporting', 'B calendar', 'B − A')[j]}"
            axes[i, j].set_title(title, fontsize=9)
            fig.colorbar(im, ax=axes[i, j], shrink=.8, label="mm/day" if i < 5 else "mm")
    fig.savefig(root / "comparison_maps.png", dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/processed/imerg_daily_window_pilot_may2022")
    parser.add_argument("--ckpt", default="runs/prior_h100_cpc_v2/best.pt")
    parser.add_argument("--stats", default="data/processed/stats_cpc_v2.json")
    parser.add_argument("--config", default="configs/da.yaml")
    parser.add_argument("--data-zarr", default="data/processed/bd_wide_cpc.zarr")
    station_mode = parser.add_mutually_exclusive_group()
    station_mode.add_argument("--stations", help="optional original canonical BMD/BWDB table; overrides archive auto-detection")
    station_mode.add_argument("--production-station-root", help="directory with existing superob_prod_0.25.csv/json; default archive auto-detected unless raw inputs explicitly selected")
    parser.add_argument("--bmd-wide", default="data/stations/Rainfall_daily_by_station_BMD.csv")
    parser.add_argument("--bmd-data-dir", help="explicit per-station BMD directory; automatically used if the default wide table is absent")
    parser.add_argument("--bmd-catalog", default="data/stations/data_2020_2025/Stations.csv")
    parser.add_argument("--bwdb", default="data/stations/BWDB_Rainfall_2000_2025_corrected.xlsx")
    parser.add_argument("--aligned-imerg", help="existing native or S04 reporting-window IMERG; auto-detects the May archive")
    parser.add_argument("--halfhourly", default="data/imerg_halfhourly/2022")
    parser.add_argument("--daily-raw", default="data/raw/imerg")
    parser.add_argument("--daily-error-mode", choices=("native", "verified-sum-squared"), default="native",
                        help="native preserves legacy raw errors; verified-sum-squared validates all five UTC days "
                             "against half hours and applies 0.5*sqrt(raw error) before coarsening")
    parser.add_argument("--download-imerg", action="store_true", help="fetch missing IMERG for this pilot only")
    parser.add_argument("--seed", type=int, default=202205)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true", help="explain dates and print the two sampler commands without IO")
    modes.add_argument("--prepare-only", action="store_true")
    modes.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args(argv)
    if args.production_station_root and args.bmd_data_dir:
        parser.error("choose --production-station-root or --bmd-data-dir, not both")
    if args.dry_run:
        print("A: gauges May 1–5; background Apr 30–May 4 (offset -1); IMERG 03–03 UTC.")
        print("B: SAME gauges relabelled Apr 30–May 4; background offset 0; daily IMERG 00–24 UTC.")
        print("One member; B seed = A seed + 1 to preserve per-day draws; backgrounds checked after sampling.")
        print(f"Daily error mode: {args.daily_error_mode}; verified mode checks Apr 30–May 4 before sampling.")
        for command in case_commands(args, "MEASURED_FROM_SHARED_SUPEROBS"):
            print(shlex.join([sys.executable, "scripts/28_simultaneous_method_sweep.py", *command]))
        return
    if args.summarize_only:
        summarize(args)
        return
    p = paths(args)
    # Re-run into a separate output root to keep sampled results tied to their inputs.
    if any((p["root"] / f"{label}.{suffix}").exists() for label in ("reporting", "calendar") for suffix in ("npz", "json")):
        raise ValueError("sampled or partial output already exists; use --summarize-only or a new --root")
    manifest = prepare(args)
    if args.prepare_only:
        return
    run("51_check_sqrt_da_gradient.py")
    for command in case_commands(args, manifest["representativeness"]):
        run("28_simultaneous_method_sweep.py", *command)
    summarize(args)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, ImportError, subprocess.CalledProcessError) as error:
        print(f"[pilot] failed: {error}", file=sys.stderr)
        raise SystemExit(1)
