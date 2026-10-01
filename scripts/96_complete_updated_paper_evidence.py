#!/usr/bin/env python3
"""Audit and recover evidence missing from the updated SURMA-Flow paper.

Default: inventory inputs and write a missing-evidence report. --run scores
the whole frozen test archive; it never trains, generates rainfall, or submits
jobs. --run-existing also invokes scripts 90 and 92 for their existing slots.
See docs/PAPER1_UPDATED_EVIDENCE.md for inputs and interpretation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from types import SimpleNamespace
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bdhires.country import DEFAULT_BOUNDARY, read_boundary, points_inside

FINAL = "dense_s6_bwdb_r4"
METHODS = ["background", FINAL]
ORDER = [FINAL, "background", "chirps", "imerg", "cpc"]
LABELS = {FINAL: "SURMA-Flow", "background": "Background", "chirps": "CHIRPS",
          "imerg": "IMERG 0.4 deg", "cpc": "CPC same day", "imerg_native": "IMERG 0.1 deg"}
EXPECTED_HASHES = {
    "checkpoint":"a04a3d9ae9109f905e06c32bfd55252daf1229d17c98b404e265064b89f210ea",
    "checkpoint_stats":"96b4de3862d931e0b4dc9f2e895b944a3d696a90187b592782771e65d5a8ce07"}


def module(number):
    path = next((ROOT / "scripts").glob(f"{number}_*.py"))
    spec = importlib.util.spec_from_file_location(f"_updated_paper_{number}", path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = result
    spec.loader.exec_module(result)
    return result


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite_json(value):
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def sample_field(field, lat, lon, station_lat, station_lon):
    """Bilinear sampling without turning missing or outside cells into rain=0.

    Missing neighbours with nonzero interpolation weight invalidate the sample.
    A zero-weight missing neighbour does not invalidate an exact grid-centre hit.
    """
    field = np.asarray(field, float)
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    if field.shape[-2:] != (len(lat), len(lon)) or min(len(lat), len(lon)) < 2:
        raise ValueError("field/grid shape mismatch or grid too small")
    yi, xi = np.argsort(lat), np.argsort(lon)
    lat, lon, field = lat[yi], lon[xi], field[..., yi, :][..., xi]
    if np.any(np.diff(lat) <= 0) or np.any(np.diff(lon) <= 0):
        raise ValueError("grid coordinates must be unique")
    sy, sx = np.asarray(station_lat, float), np.asarray(station_lon, float)
    y = np.clip(np.searchsorted(lat, sy, side="right") - 1, 0, len(lat) - 2)
    x = np.clip(np.searchsorted(lon, sx, side="right") - 1, 0, len(lon) - 2)
    wy, wx = (sy - lat[y]) / (lat[y + 1] - lat[y]), (sx - lon[x]) / (lon[x + 1] - lon[x])
    inside = np.isfinite(sy) & np.isfinite(sx) & (sy >= lat[0]) & (sy <= lat[-1]) & (sx >= lon[0]) & (sx <= lon[-1])
    shape = (*field.shape[:-2], len(sy))
    output, valid = np.zeros(shape), np.broadcast_to(inside, shape).copy()
    for dy, dx, weight in ((0, 0, (1-wy)*(1-wx)), (1, 0, wy*(1-wx)),
                           (0, 1, (1-wy)*wx), (1, 1, wy*wx)):
        values = field[..., y + dy, x + dx]
        valid &= np.isfinite(values) | (weight == 0)
        output += np.where(np.isfinite(values), values, 0) * weight
    return np.where(valid, output, np.nan)


def metrics(predicted, truth):
    error = predicted - truth
    return {"n": len(truth), "rmse_mm": float(np.sqrt(np.mean(error**2))),
            "mae_mm": float(np.mean(np.abs(error))), "bias_mm": float(np.mean(error)),
            "correlation": float(np.corrcoef(predicted, truth)[0, 1])
            if len(truth) > 1 and predicted.std() > 0 and truth.std() > 0 else None}


def validate_native_metadata(attrs):
    if (str(attrs.get("version")) != "V07B" or attrs.get("source_frequency") != "half-hourly"
            or int(attrs.get("bmd_accumulation_end_hour_utc",-1)) != 3
            or int(attrs.get("window_duration_hours",-1)) != 24):
        raise ValueError("native IMERG requires V07B half-hourly-derived 24-hour windows ending at 03 UTC")


def matched_samples(data, predictions, profile):
    dates = data["date"].astype("datetime64[D]")
    selection = (dates >= np.datetime64(profile["selection_start"])) & (dates <= np.datetime64(profile["selection_end"]))
    common = np.isfinite(data["truth"]) & ~selection
    primary_n = int(common.sum())
    for members in data["members"].values():
        common &= np.all(np.isfinite(members), axis=1)
    for predicted in predictions.values():
        if np.shape(predicted) != np.shape(data["truth"]):
            raise ValueError("prediction and original gauge sample shapes differ")
        common &= np.isfinite(predicted)
    if not common.any():
        raise ValueError("no common finite independent station-days")
    keep = {k: v[common] for k, v in data.items() if k != "members"}
    predictions = {k: v[common] for k, v in predictions.items()}
    return keep, predictions, {"primary_truth_n": primary_n, "matched_daily_n": int(common.sum()),
                               "daily_sample_attrition": primary_n - int(common.sum()),
                               "scored_dates": len(np.unique(keep["date"])),
                               "first_date": str(keep["date"].min()), "last_date": str(keep["date"].max())}


def product_rows(data, predictions, profile, counts, available_dates=None):
    """Daily and temporal comparisons across every available independent date."""
    rows, periods = [], []
    dates, stations, truth = data["date"], data["station"], data["truth"]
    archive_dates = np.unique(dates if available_dates is None else available_dates)

    def score(group, label, choose, observed=truth, values=predictions):
        if not np.any(choose):
            return
        for source, field in values.items():
            rows.append({"group": group, "label": label, "source": source, **counts,
                         **metrics(field[choose], observed[choose])})

    score("pooled", "all", np.ones(len(truth), bool))
    for group, labels in (("network", data["source"]), ("year", dates.astype("datetime64[Y]").astype(str)),
                          ("month", dates.astype("datetime64[M]").astype(str))):
        for label in np.unique(labels):
            score(group, str(label), labels == label)
    for low, high, label in ((0,1,"[0,1)"),(1,10,"[1,10)"),(10,25,"[10,25)"),
                             (25,50,"[25,50)"),(50,100,"[50,100)"),(100,np.inf,"[100,inf)")):
        score("intensity", label, (truth >= low) & (truth < high))
    station_scores = []
    for station in np.unique(stations):
        choose = stations == station
        for source, values in predictions.items():
            station_scores.append({"station_id": station, "source": source,
                                   "observed_mean_mm_day": float(truth[choose].mean()),
                                   "predicted_mean_mm_day": float(values[choose].mean()),
                                   **metrics(values[choose], truth[choose])})
    # Monthly groups count calendar days. Complete seasons never become
    # "complete" because only a shorter archive or smaller subset was supplied.
    for scale in ("monthly", "may_sep", "available_wet_season"):
        period_labels = dates.astype("datetime64[M]" if scale == "monthly" else "datetime64[Y]")
        obs_groups, pred_groups = [], {s: [] for s in predictions}
        for period in np.unique(period_labels):
            if scale == "monthly":
                start, stop = period.astype("datetime64[D]"), (period + np.timedelta64(1,"M")).astype("datetime64[D]")
            else:
                start, stop = np.datetime64(f"{period}-05-01"), np.datetime64(f"{period}-10-01")
            calendar = np.arange(start, stop)
            selected = (calendar >= np.datetime64(profile["selection_start"])) & (calendar <= np.datetime64(profile["selection_end"]))
            if scale == "available_wet_season":
                # Keep the archive denominator even if all products or gauges
                # are missing on a day; shared-sample attrition cannot improve
                # the coverage fraction or redefine a complete season.
                calendar = np.intersect1d(calendar[~selected], archive_dates)
            elif selected.any():
                continue
            if not len(calendar):
                continue
            required = len(calendar) if scale == "may_sep" else int(np.ceil(.8 * len(calendar)))
            for station in np.unique(stations):
                choose = (stations == station) & np.isin(dates, calendar)
                if int(choose.sum()) < required:
                    continue
                observed = float(truth[choose].mean())
                obs_groups.append(observed)
                for source, field in predictions.items():
                    predicted = float(field[choose].mean())
                    pred_groups[source].append(predicted)
                    periods.append({"scale": scale, "period": str(period), "station_id": station,
                                    "source": source, "n_days": int(choose.sum()), "eligible_days": len(calendar),
                                    "first_date": str(dates[choose].min()), "last_date": str(dates[choose].max()),
                                    "partial_season": scale == "available_wet_season" and len(calendar) < 153,
                                    "observed_mean_mm_day": observed, "predicted_mean_mm_day": predicted})
        score("temporal", scale, np.ones(len(obs_groups), bool), np.asarray(obs_groups),
              {s: np.asarray(v) for s,v in pred_groups.items()})
    return rows, station_scores, periods


def paired_intervals(data, predictions, block_days, resamples, seed):
    """Bootstrap pooled RMSE/MAE gains by resampling whole day blocks.

    All stations on a day move together. Recompute nonlinear RMSE in each
    resample. Station-day weighting matches the headline deterministic table.
    Bonferroni intervals cover the six primary product/metric contrasts for
    each block-width sensitivity analysis. No model-selection uncertainty.
    """
    dates = np.unique(data["date"])
    split = np.where(np.diff(dates).astype("timedelta64[D]").astype(int) > 1)[0] + 1
    segments = np.split(np.arange(len(dates)), split)
    day_index = np.searchsorted(dates, data["date"])
    n = np.bincount(day_index).astype(float)
    error = {s: p - data["truth"] for s,p in predictions.items()}
    sums = {s: (np.bincount(day_index, weights=e**2), np.bincount(day_index, weights=np.abs(e))) for s,e in error.items()}
    output = []
    for reference in [s for s in predictions if s != FINAL]:
        for width in block_days:
            rng = np.random.default_rng(seed)
            gains = np.empty((resamples,2))
            # Bounded memory: totals for at most 256 resamples at once.
            for start in range(0, resamples, 256):
                batch = min(256, resamples-start)
                totals = np.zeros((batch,5))
                for segment in segments:
                    length = len(segment); size = min(width,length)
                    starts = rng.integers(0,length,(batch,int(np.ceil(length/size))))
                    index = segment[((starts[...,None]+np.arange(size)).reshape(batch,-1)[:,:length]) % length]
                    values = [n, sums[reference][0], sums[FINAL][0], sums[reference][1], sums[FINAL][1]]
                    for col,v in enumerate(values):
                        totals[:,col] += v[index].sum(axis=1)
                gains[start:start+batch,0] = np.sqrt(totals[:,1]/totals[:,0]) - np.sqrt(totals[:,2]/totals[:,0])
                gains[start:start+batch,1] = (totals[:,3]-totals[:,4])/totals[:,0]
            for col, metric in enumerate(("rmse", "mae")):
                low, high = np.percentile(gains[:,col],[2.5,97.5])
                family = 6 if reference in ("chirps","imerg","cpc") else 0
                adj = np.percentile(gains[:,col], [100*.05/(2*family),100*(1-.05/(2*family))]) if family else [None,None]
                point = (np.sqrt(sums[reference][0].sum()/n.sum())-np.sqrt(sums[FINAL][0].sum()/n.sum())
                         if col == 0 else (sums[reference][1].sum()-sums[FINAL][1].sum())/n.sum())
                output.append({"reference": reference, "candidate": FINAL, "metric": metric,
                               "gain_mm_day": float(point), "ci_low": float(low), "ci_high": float(high),
                               "family_size": family, "family_ci_low": adj[0], "family_ci_high": adj[1],
                               "block_days": width, "n_resamples": resamples, "n_days": len(dates),
                               "n_segments": len(segments), "n": int(n.sum()), "seed": seed,
                               "weighting": "pooled station-days; all sites resampled together within day"})
    return output


def collect_predictions(data, scopes, root, periods, profile, country, cpc_override, native_paths):
    predictions = {m: data["members"][m].mean(axis=1) for m in METHODS}
    for product in ("chirps", "imerg"):
        predictions[product] = np.full(len(data["truth"]), np.nan)
    shared = module(55)
    first_grid = None
    for period in periods:
        base = root / "evaluation" / period
        with np.load(base.with_suffix(".npz"), allow_pickle=False) as dump:
            lat, lon, valid = dump["grid_lat"], dump["grid_lon"], dump["valid"].astype(bool)
            if first_grid is None:
                first_grid = lat.copy(), lon.copy(), valid.copy()
            elif not (np.array_equal(lat,first_grid[0]) and np.array_equal(lon,first_grid[1]) and np.array_equal(valid,first_grid[2])):
                raise ValueError("period grids or model validity masks differ")
            choose = data["period"] == period
            positions = np.flatnonzero(choose)
            times = dump["times"].astype("datetime64[D]")
            wanted = np.searchsorted(times,data["date"][choose])
            if not np.array_equal(times[wanted],data["date"][choose]):
                raise ValueError("product and validated gauge dates differ")
            fields = {"chirps": np.asarray(dump["chirps"],float),
                      "imerg": shared.upsample_coarse(np.asarray(dump["raw_imerg_mm"],float),8,valid.shape)}
            for product, field in fields.items():
                if field.shape != (len(times),len(lat),len(lon)):
                    raise ValueError("product shape and archive times differ")
                field = np.where(valid,field,np.nan)
                for day_index in np.unique(wanted):
                    rows = positions[wanted == day_index]
                    predictions[product][rows] = sample_field(field[day_index],lat,lon,data["station_lat"][rows],data["station_lon"][rows])
    lat,lon,valid = first_grid
    archive = {"time":np.unique(data["date"]), "lat":lat, "lon":lon, "valid":valid,
               "datasets":[SimpleNamespace(attrs={"scope":r["scope"]}) for r in scopes]}
    same_day = shared.load_same_day_cpc(archive,cpc_override)
    if same_day is None:
        raise FileNotFoundError("same-day CPC unavailable; supply --cpc-source-zarr pointing to the checkpoint packed store")
    predictions["cpc"] = np.full(len(data["truth"]),np.nan)
    for i,day in enumerate(archive["time"]):
        choose = data["date"] == day
        predictions["cpc"][choose] = sample_field(np.where(valid,same_day[0][i],np.nan),lat,lon,
                                                   data["station_lat"][choose],data["station_lon"][choose])
    native = None
    if native_paths:
        import xarray as xr
        for path in native_paths:
            with xr.open_dataset(path) as dataset:
                validate_native_metadata(dataset.attrs)
        native = module(81).load_native_imerg(native_paths,archive["time"])
        predictions["imerg_native"] = np.full(len(data["truth"]),np.nan)
        for i,day in enumerate(archive["time"]):
            choose = data["date"] == day
            predictions["imerg_native"][choose] = sample_field(native["values"][i],native["lat"],native["lon"],
                                                              data["station_lat"][choose],data["station_lon"][choose])
    return predictions, same_day[1], native


def overlap_audit(stations, config_path):
    """Curated, dated identities can confirm overlap; proximity only suggests it.

    No match is never proof of non-use, even for a purportedly complete list.
    Versions and dated provider inventories remain necessary external evidence.
    """
    records = json.loads(config_path.read_text())["inventories"] if config_path else []
    if any(r["product"] not in ("cpc","chirps","imerg") for r in records):
        raise ValueError("upstream product must be cpc, chirps or imerg")
    rows, sources = [], []
    for product in ("cpc","chirps","imerg"):
        inventories = [r for r in records if r["product"] == product]
        candidates = []
        for record in inventories:
            if record.get("membership") not in ("direct","gauge_adjustment") or not record.get("reference"):
                raise ValueError("upstream inventory requires membership and provider reference")
            path = Path(record["path"])
            if not path.is_absolute():
                path = config_path.parent / path
            values = read_csv(path)
            if not values or any(not r.get("start") or not r.get("end") or not r.get("station_id") for r in values):
                raise ValueError("upstream CSV needs station_id,start,end; dated membership must not be guessed")
            sources.append({"path":str(path.resolve()),"sha256":digest(path),"product":product,"reference":record["reference"]})
            candidates.extend((r,record) for r in values)
        for station in stations:
            matches, nearby, matched_dates = [], [], set()
            station_dates = np.asarray(station["dates"],dtype="datetime64[D]")
            for candidate, record in candidates:
                start,end = np.datetime64(candidate["start"]),np.datetime64(candidate["end"])
                if end < start:
                    raise ValueError("inventory membership end precedes start")
                overlaps = int(((station_dates >= start) & (station_dates <= end)).sum())
                if not overlaps:
                    continue
                if candidate.get("local_station_id") == station["station_id"] and candidate.get("evidence"):
                    matches.append((candidate,record,overlaps))
                    matched_dates.update(station_dates[(station_dates>=start)&(station_dates<=end)].astype(str))
                if candidate.get("lat") and candidate.get("lon"):
                    a,b = np.radians([station["lat"],station["lon"]]),np.radians([float(candidate["lat"]),float(candidate["lon"])])
                    h = np.sin((a[0]-b[0])/2)**2 + np.cos(a[0])*np.cos(b[0])*np.sin((a[1]-b[1])/2)**2
                    distance = 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(h,0,1)))
                    if distance <= 1.:
                        nearby.append(candidate["station_id"])
            rows.append({"local_station_id":station["station_id"],"product":product,
                         "status":"confirmed_overlap_on_dated_inventory" if matches else "coordinate_candidate_only" if nearby else "unknown",
                         "inventory_station_ids":";".join(sorted({m[0]["station_id"] for m in matches} or set(nearby))),
                         "membership":";".join(sorted({m[1]["membership"] for m in matches})),
                         "documented_station_day_matches":len(matched_dates),
                         "evidence":";".join(sorted({m[0]["evidence"] for m in matches})),
                         "note":"absence from supplied inventories does not establish upstream non-use"})
    return rows,sources


def summary_crosschecks(path, expected_sha=None):
    """Check country-summary claims recoverable without inventing daily arrays."""
    table_path = path/"gridded/long_term_withheld_station_scores.csv"
    report_path = path/"paper1_evaluation.json"
    if not table_path.is_file() or not report_path.is_file():
        return {"status":"missing_summary_inputs"}
    rows = read_csv(table_path); report = json.loads(report_path.read_text())
    if expected_sha and report.get("evaluation_region",{}).get("sha256") != expected_sha:
        raise ValueError("summary archive is not verified against the same Bangladesh boundary")
    by_source = {s:{r["station_id"]:r for r in rows if r["source"]==s} for s in ORDER}
    identities = {s:{station:int(r["n"]) for station,r in values.items()} for s,values in by_source.items()}
    if not all(ids==identities[FINAL] for ids in identities.values()):
        raise ValueError("summary product comparisons use different station counts")
    if sum(identities[FINAL].values()) != report["counts"]["scored_station_days"]:
        raise ValueError("summary station counts differ from country report")
    comparisons = {}
    for source in ("imerg","cpc","chirps"):
        difference = np.array([float(by_source[source][station]["rmse_mm"])-float(by_source[FINAL][station]["rmse_mm"]) for station in identities[FINAL]])
        comparisons[source] = {"stations_lower_analysis_rmse":int((difference>0).sum()),
                               "median_rmse_gain_mm_day":float(np.median(difference))}
    matrix = np.array([[float(by_source[s][station]["rmse_mm"]) for s in ORDER] for station in identities[FINAL]])
    strict_wins = int((matrix[:,0] < matrix[:,1:].min(axis=1)).sum())
    return {"status":"verified_country_summary", "station_days":sum(identities[FINAL].values()),
            "stations":len(identities[FINAL]),"strict_lowest_rmse_all_five":strict_wins,
            "comparison":comparisons,
            "median_station_correlation":{s:float(np.nanmedian([float(r["correlation"]) if r["correlation"] else np.nan for r in values.values()])) for s,values in by_source.items()},
            "minimum_station_days":min(identities[FINAL].values()),"maximum_station_days":max(identities[FINAL].values()),
            "source_sha256":digest(table_path),
            "not_recoverable_from_station_summaries":["pooled product correlation","paired product intervals","fine intensity bins","product monthly and seasonal scores"]}


def preparation_provenance(root,periods,logs):
    results = []
    for period in periods:
        path = root/"stations"/period/"superob_eval_0.25.json"
        if not path.is_file():
            results.append({"period":period,"status":"manifest_missing"})
            continue
        payload = json.loads(path.read_text())
        evidence = {k:v for k,v in payload.items() if "stats" in k.lower() or "transform" in k.lower()}
        results.append({"period":period,"path":str(path),"sha256":digest(path),
                        "status":"recorded_fields_require_review" if evidence else "historical_stats_identity_unrecorded",
                        "recorded_fields":evidence})
    candidates = []
    for path in logs or []:
        for line_number,line in enumerate(path.read_text(errors="replace").splitlines(),1):
            for match in re.finditer(r"(?:--stats\s+|\bSTATS=)([^\s]+)",line):
                candidate = match.group(1).strip("\"'")
                if "$" in candidate or "{" in candidate:
                    continue  # A default/template is not a measured execution record.
                candidates.append({"log":str(path),"log_sha256":digest(path),"line":line_number,
                                   "candidate_stats_path":candidate,"status":"pathname_candidate_only; historical content hash not proved"})
    return {"periods":results,"log_candidates":candidates,
            "note":"No statistics identity is inferred from a Slurm default, current file content or similar numerical values."}


def stations_from_summary(path, contract, profile):
    report_path = path / "paper1_evaluation.json"
    audit_path = path / "station_country_audit.csv"
    if not report_path.is_file() or not audit_path.is_file():
        return []
    report = json.loads(report_path.read_text())
    dates = {}
    for entry in report["archived_contracts"]:
        scope = entry["scope"]
        days = np.arange(np.datetime64(scope["start"]),np.datetime64(scope["end"])+np.timedelta64(1,"D"))
        days = days[(days < np.datetime64(profile["selection_start"])) | (days > np.datetime64(profile["selection_end"]))]
        for station in scope["withheld_station_ids"]:
            dates.setdefault(station,set()).update(days.astype(str))
    return [{"station_id":r["station_id"],"lat":float(r["lat"]),"lon":float(r["lon"]),
             "dates":sorted(dates.get(r["station_id"],[]))} for r in read_csv(audit_path)
            if r["inside_bangladesh"].lower() == "true"]


def write_tables(out, rows, paired):
    table = module(92).table
    f = lambda x: "unavailable" if x is None else f"{x:.3f}"
    daily = [r for r in rows if r["group"] == "pooled"]
    table(out/"tab_product_daily.tex",["Estimate","n","RMSE","MAE","Bias","Pooled r"],
          [[LABELS[r["source"]],r["n"],*[f(r[k]) for k in ("rmse_mm","mae_mm","bias_mm","correlation")]] for r in daily],
          "Identical original Bangladesh withheld station-days; original same-day CPC; deterministic scores only.")
    temporal = [r for r in rows if r["group"] == "temporal" and r["label"] != "available_wet_season"]
    if temporal:
        table(out/"tab_product_temporal.tex",["Scale","Estimate","Groups","RMSE","MAE","Bias"],
              [[r["label"],LABELS[r["source"]],r["n"],*[f(r[k]) for k in ("rmse_mm","mae_mm","bias_mm")]] for r in temporal],
              "Same eligible station-periods for all estimates. Monthly coverage >=80% calendar days; complete May-September seasons only. Means in mm/day, not ensemble totals.")
    table(out/"tab_product_paired.tex",["Reference","Metric","Block","Gain","95% low","95% high"],
          [[LABELS[r["reference"]],r["metric"],r["block_days"],*[f(r[k]) for k in ("gain_mm_day","ci_low","ci_high")]] for r in paired],
          "Positive pooled station-day gain favours SURMA-Flow. Gap-separated day blocks; nonlinear RMSE recomputed per resample. Family-adjusted intervals are in paired_product_scores.csv.")


def render_figures(out, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {FINAL:"#0B7A75","background":"#8A94A6","chirps":"#C8862A","imerg":"#7B5EA7","cpc":"#2F6FB0"}
    for group,labels,name in (("intensity",["[0,1)","[1,10)","[10,25)","[25,50)","[50,100)","[100,inf)"],"fig08_product_intensity"),
                              ("temporal",["monthly","may_sep"],"fig09_product_temporal")):
        fig,axes = plt.subplots(1,2,figsize=(10,3.6),layout="constrained")
        for ax,metric in zip(axes,("rmse_mm","mae_mm")):
            for source in ORDER:
                values = [next((r[metric] for r in rows if (r["group"],r["label"],r["source"]) == (group,label,source)),np.nan) for label in labels]
                ax.plot(np.arange(len(labels)),values,"o-",label=LABELS[source],color=colors[source])
            ax.set(xticks=np.arange(len(labels)),xticklabels=labels,ylabel="mm/day",title=metric.replace("_mm","").upper())
            ax.grid(alpha=.2)
        axes[0].legend(frameon=False,fontsize=8)
        fig.suptitle("Original Bangladesh withheld gauges; same sample for every method",fontsize=10)
        for suffix in ("pdf","png"):
            fig.savefig(out/f"{name}.{suffix}",bbox_inches="tight",dpi=200)
        plt.close(fig)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--contract",type=Path,default=ROOT/"configs/paper1_cpcv2_final.json")
    p.add_argument("--root",type=Path)
    p.add_argument("--summary",type=Path,default=ROOT/"paper1_cpcv2/bangladesh/superob-final")
    p.add_argument("--manuscript",type=Path,default=ROOT/"manuscript/BDhighresDA_arxiv.tex")
    p.add_argument("--boundary-geojson",type=Path,default=DEFAULT_BOUNDARY)
    p.add_argument("--out-dir",type=Path,default=ROOT/"output/paper1_updated_evidence")
    p.add_argument("--cpc-source-zarr",type=Path)
    p.add_argument("--native-imerg",type=Path,nargs="+",help="prepared 0.1-deg BMD 03 UTC window NetCDFs covering all archive days")
    p.add_argument("--upstream-inventories",type=Path,help="dated provider-station CSV inventory JSON; see documentation")
    p.add_argument("--history",type=Path,default=ROOT/"runs/prior_h100_cpc_v2/validation/history.jsonl")
    p.add_argument("--selection",type=Path)
    p.add_argument("--compute",type=Path)
    p.add_argument("--accounting-job-ids",nargs="+",help="explicit Slurm job IDs to read with sacct; raw accounting only, not fabricated stage timings")
    p.add_argument("--preparation-logs",type=Path,nargs="+",help="original execution logs to scan for recorded --stats paths")
    p.add_argument("--robustness",type=Path,nargs="+")
    p.add_argument("--run",action="store_true",help="score saved arrays; default is input audit only")
    p.add_argument("--run-existing",action="store_true",help="also run 90 and 92; requires --run")
    p.add_argument("--with-gridded",action="store_true",help="run 90's optional full gridded suite; requires --run-existing")
    p.add_argument("--no-figures",action="store_true")
    p.add_argument("--bootstrap",type=int,default=10000)
    p.add_argument("--block-days",type=int,nargs="+",default=[3,7])
    p.add_argument("--seed",type=int,default=20261001)
    args = p.parse_args(argv)
    if args.bootstrap < 1 or any(n < 1 for n in args.block_days):
        p.error("bootstrap and block widths must be positive")
    if args.run_existing and not args.run or args.with_gridded and not args.run_existing:
        p.error("--run-existing needs --run; --with-gridded needs --run-existing")
    if args.accounting_job_ids and any(not re.fullmatch(r"\d+(?:_\d+)?",s) for s in args.accounting_job_ids):
        p.error("accounting job IDs must be individual numeric or numeric_array IDs")
    return args


def main(argv=None):
    args = parse_args(argv)
    contract = json.loads(args.contract.read_text()); profile = contract["profiles"]["superob-final"]
    root = args.root or ROOT/profile["root"]; periods = list(contract["periods"])
    country,region = read_boundary(args.boundary_geojson)
    if region["sha256"] != contract["evaluation_region"]["boundary_sha256"]:
        raise ValueError("boundary differs from the final-paper contract")
    args.out_dir.mkdir(parents=True,exist_ok=True)
    generated = []
    inventory = module(90).inventory(root,periods,profile)
    unconfigured = [str(p) for p in (root/"evaluation").glob("*.npz") if p.stem not in periods]
    raw_missing = [r["path"] for r in inventory["required"] if not r["present"]]
    raw_reports = [root/"stations"/p/"combined_daily.csv" for p in periods]
    availability = {
        "P1_P4_original_station_tables":all(p.is_file() for p in raw_reports),
        "P2_P3_P9_BD1_member_archive":not raw_missing,
        "WG1_WG2_WG3_product_archive":not raw_missing,
        "WG4_upstream_inventory":bool(args.upstream_inventories and args.upstream_inventories.is_file()),
        "WG5_native_imerg":bool(args.native_imerg and all(p.is_file() for p in args.native_imerg)),
        "P5_selection":bool(args.selection and args.selection.is_file()),
        "P6_validation_history":args.history.is_file(),
        "P7_measured_compute":bool(args.compute and args.compute.is_file()),
        "P8_additional_holdouts":bool(args.robustness and len(args.robustness)>=2 and all((p/"paper1_evaluation.json").is_file() for p in args.robustness)),
        "BD2_full_gridded_archive":all(r["present"] for r in inventory["production_stores"]),
    }
    manifest = {"mode":"scoring" if args.run else "input_audit_only", "status":"pending_inputs",
                "manuscript_sha256":digest(args.manuscript) if args.manuscript.is_file() else None,
                "manuscript_source_present":args.manuscript.is_file(),"contract_sha256":digest(args.contract),
                "evaluation_region":region,"availability":availability,"inventory":inventory,
                "missing_original_station_tables":[str(p) for p in raw_reports if not p.is_file()],
                "scope":"every configured available archive period, not a selected year or two seasons",
                "selection_exclusion":[profile["selection_start"],profile["selection_end"]],
                "unconfigured_evaluation_files":unconfigured,"inputs":[],"outputs":[]}
    summary = summary_crosschecks(args.summary,region["sha256"])
    (args.out_dir/"summary_crosschecks.json").write_text(json.dumps(finite_json(summary),indent=2,allow_nan=False)+"\n")
    generated.append(args.out_dir/"summary_crosschecks.json")
    preparation = preparation_provenance(root,periods,args.preparation_logs)
    (args.out_dir/"preparation_stats_provenance.json").write_text(json.dumps(preparation,indent=2)+"\n")
    generated.append(args.out_dir/"preparation_stats_provenance.json")
    if args.accounting_job_ids:
        command = ["sacct","--noheader","--parsable2","--jobs",",".join(args.accounting_job_ids),
                   "--format","JobIDRaw,JobName,State,ElapsedRaw,AllocTRES,NodeList,Start,End"]
        accounting = subprocess.run(command,check=True,capture_output=True,text=True)
        if not accounting.stdout.strip():
            raise ValueError("sacct returned no records for the supplied job IDs")
        path = args.out_dir/"slurm_job_accounting.psv"
        path.write_text("JobIDRaw|JobName|State|ElapsedRaw|AllocTRES|NodeList|Start|End\n"+accounting.stdout)
        generated.append(path)
        manifest["accounting_note"] = "Raw allocation time; stage/checkpoint attribution still requires original logs or measured benchmark records."
    dates = np.unique(np.concatenate([np.arange(np.datetime64(start),np.datetime64(end)+np.timedelta64(1,"D")) for start,end in contract["periods"].values()]))
    full = np.arange(np.datetime64(f'{contract["test_years"][0]}-01-01'),np.datetime64(f'{contract["test_years"][1]+1}-01-01'))
    write_csv(args.out_dir/"test_date_coverage.csv",[{"date":str(day),"configured_archive":bool(day in dates),
              "selection_date":profile["selection_start"]<=str(day)<=profile["selection_end"],
              "raw_archive_present_locally":not raw_missing if day in dates else False} for day in full])
    generated.append(args.out_dir/"test_date_coverage.csv")
    manifest["coverage"] = {"configured_archive_days":len(dates),"days_outside_current_archive_contract":len(np.setdiff1d(full,dates)),
                            "note":"missing date rows are an extension plan, not evidence that a remote job has not run"}
    stations = stations_from_summary(args.summary,contract,profile)
    if args.run and unconfigured:
        raise ValueError("additional saved evaluation periods are present; extend and validate the frozen contract before claiming the whole available archive: "+", ".join(unconfigured))
    if args.run and not raw_missing:
        with tempfile.TemporaryDirectory(prefix="paper1-updated-",dir=args.out_dir.parent) as temp:
            stage = Path(temp)
            data,sources,scopes,warnings = module(90).load_samples(root,periods,contract,profile,METHODS)
            data,region = module(90).country_samples(data,args.boundary_geojson)
            manifest["evaluation_region"] = region; manifest["warnings"] = warnings
            manifest["inputs"] += sources
            manifest["checkpoint_identity"] = {}
            for role,expected in EXPECTED_HASHES.items():
                observed = [r["sha256"] for r in sources if r.get("role")==role]
                if observed and any(value != expected for value in observed):
                    raise ValueError(f"{role} file differs from the model-paper pinned SHA-256")
                manifest["checkpoint_identity"][role] = "current referenced file matches paper hash" if observed else "content unavailable; archived pathname alone is not a content identity"
            predictions,cpc_path,native = collect_predictions(data,scopes,root,periods,profile,country,args.cpc_source_zarr,args.native_imerg)
            # Primary five-way sample stays separate from the optional six-way
            # native comparison, so native missingness cannot change old rankings.
            matched,values,counts = matched_samples(data,{s:predictions[s] for s in ORDER},profile)
            rows,station_scores,temporal = product_rows(matched,values,profile,counts,dates)
            paired = paired_intervals(matched,values,args.block_days,args.bootstrap,args.seed)
            write_csv(stage/"withheld_product_strata.csv",rows)
            write_csv(stage/"withheld_product_station_scores.csv",station_scores)
            write_csv(stage/"withheld_product_temporal_samples.csv",temporal)
            write_csv(stage/"paired_product_scores.csv",paired)
            write_tables(stage,rows,paired)
            if not args.no_figures:
                render_figures(stage,rows)
            manifest["primary_comparison_counts"] = counts
            manifest["same_day_cpc_source"] = str(cpc_path)
            if native is not None:
                native_data,native_values,native_counts = matched_samples(data,predictions,profile)
                native_rows,_,_ = product_rows(native_data,native_values,profile,native_counts,dates)
                native_rows = [r for r in native_rows if r["group"] in ("pooled","network","year")]
                write_csv(stage/"native_imerg_comparison.csv",native_rows)
                native_paired = paired_intervals(native_data,{s:native_values[s] for s in (FINAL,"imerg","imerg_native")},args.block_days,args.bootstrap,args.seed)
                write_csv(stage/"native_imerg_paired.csv",native_paired)
                module(92).table(stage/"tab_native_imerg.tex",["Estimate","n","RMSE","MAE","Bias","r"],
                                 [[LABELS[r["source"]],r["n"],*["unavailable" if r[k] is None else f'{r[k]:.3f}' for k in ("rmse_mm","mae_mm","bias_mm","correlation")]] for r in native_rows if r["group"]=="pooled"],
                                 "Native 0.1-degree IMERG at original withheld gauges using BMD 03 UTC window labels. All six estimates rescored on the native comparison intersection; see native_comparison_counts for attrition.")
                manifest["native_comparison_counts"] = native_counts
                manifest["native_imerg_sources"] = [{"path":str(p),"sha256":digest(p)} for p in args.native_imerg]
            primary = np.isfinite(data["truth"]) & ((data["date"]<np.datetime64(profile["selection_start"])) | (data["date"]>np.datetime64(profile["selection_end"])))
            stations = [{"station_id":station,"lat":float(data["station_lat"][data["station"]==station][0]),
                         "lon":float(data["station_lon"][data["station"]==station][0]),
                         "dates":data["date"][(data["station"]==station)&primary].astype(str).tolist()} for station in np.unique(data["station"][primary])]
            # Publish only after every supplied product input passes validation.
            for path in stage.iterdir():
                path.replace(args.out_dir/path.name)
                generated.append(args.out_dir/path.name)
            manifest["status"] = "matched_product_scores_complete"
    overlaps,inventory_sources = overlap_audit(stations,args.upstream_inventories)
    if overlaps:
        write_csv(args.out_dir/"upstream_station_overlap.csv",overlaps)
        generated.append(args.out_dir/"upstream_station_overlap.csv")
        manifest["upstream_status_counts"] = {status:sum(r["status"]==status for r in overlaps) for status in {r["status"] for r in overlaps}}
    manifest["inputs"] += inventory_sources
    if args.run_existing and not raw_missing:
        command = [sys.executable,str(ROOT/"scripts/90_evaluate_cpcv2_paper1.py"),"--root",str(root),
                   "--out-dir",str(args.out_dir/"evaluation"),"--boundary-geojson",str(args.boundary_geojson),
                   "--contract",str(args.contract),"--bootstrap",str(args.bootstrap),
                   "--seed",str(args.seed),"--block-days",*[str(n) for n in args.block_days]]
        if args.with_gridded:command += ["--with-gridded"]
        if args.cpc_source_zarr:command += ["--cpc-source-zarr",str(args.cpc_source_zarr)]
        subprocess.run(command,check=True,cwd=ROOT)
        command = [sys.executable,str(ROOT/"scripts/92_complete_cpcv2_paper1.py"),"--root",str(root),
                   "--output",str(args.out_dir/"additional"),"--contract",str(args.contract),
                   "--boundary-geojson",str(args.boundary_geojson),"--history",str(args.history)]
        command += ["--seed",str(args.seed)]
        for flag,value in (("--selection",args.selection),("--compute",args.compute)):
            if value:command += [flag,str(value)]
        if args.robustness:command += ["--robustness",*[str(p) for p in args.robustness]]
        subprocess.run(command,check=True,cwd=ROOT)
    manifest["outputs"] = [{"path":p.name,"sha256":digest(p)} for p in generated]
    (args.out_dir/"evidence_manifest.json").write_text(json.dumps(finite_json(manifest),indent=2,allow_nan=False)+"\n")
    lines = ["# Updated-paper evidence audit","",f"Status: {manifest['status']}",
             "Scope: every available frozen test period; all May 2022 selection dates excluded.",
             "", "| Evidence | Inputs present (not scientific validation) |", "|---|---|"]
    lines += [f"| {key} | {'yes' if value else 'no'} |" for key,value in availability.items()]
    lines += ["", "See docs/PAPER1_UPDATED_EVIDENCE.md for the claim review, commands and input schemas.",
              "FP needs new sampling; WG4 needs dated provider station identities; P5/P6/P7 need original experiment/job records.",
              "No upstream overlap, scientific result, timing or ungenerated year is inferred from missing inputs."]
    (args.out_dir/"missing_evidence.md").write_text("\n".join(lines)+"\n")
    print(f"[updated-paper] {manifest['status']}; {len(raw_missing)} required archive inputs missing")
    print(f"[updated-paper] report: {args.out_dir/'missing_evidence.md'}")
    return 2 if args.run and raw_missing else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError,OSError,KeyError,subprocess.CalledProcessError) as error:
        print(f"[updated-paper] validation failed: {error}",file=sys.stderr)
        raise SystemExit(1)
