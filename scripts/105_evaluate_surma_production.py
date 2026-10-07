#!/usr/bin/env python3
"""CPU diagnostics of completed SURMA production; gauges are assimilated fits.

Read one quarterly Zarr at a time. Never regenerate fields or alter production
inputs. All figures have CSV source tables; an evaluation receipt is written
only after every requested quarter and figure succeeds.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bdhires.country import DEFAULT_BOUNDARY, read_boundary, grid_mask, points_inside, map_layer
from bdhires.paper import save_figure, use_paper_style

ANALYSIS = "dense_s6_bwdb_r4"
SOURCES = ("analysis", "background", "cpc", "imerg", "chirps")
LABELS = {"analysis": "SURMA analysis", "background": "Prior background",
          "cpc": "CPC (conditioning D−1)", "imerg": "IMERG S04 (assimilated)",
          "chirps": "CHIRPS (calendar D reference)"}
NOTES = [
    "Original BMD/BWDB gauges contributed to assimilated super-observations; their fits are not independent skill.",
    "Exact assimilated super-observation ensemble fits are also scored separately from saved station NPZs.",
    "BMD reporting support is [D-1 00:00,D 00:00] UTC; BWDB is [D-1 03:00,D 03:00] UTC.",
    "IMERG is the archived reporting-window S04 observation. CPC is archived conditioning from D-1, including recorded donor days.",
    "CHIRPS is the archived calendar-D reference; daily comparisons have different temporal support and are diagnostic only.",
    "IMERG enters assimilation, CPC conditions the prior, and CHIRPS is the training target; none is independent gridded truth.",
    "Grid domain statistics use cosine-latitude area weights and daily common finite support inside Bangladesh.",
    "Gauge scores use the same complete station-day sample across all products; pooling weights observed station-days equally.",
    "Bilinear point sampling requires finite contributing land cells; coastal samples with missing contributing cells are excluded.",
    "Monthly station totals require >=90% jointly valid days; totals are observed-day sums without missing-day inflation.",
    "Annual/monthly gridded totals require every day at a cell. Trends are descriptive slopes, not significance or attribution tests.",
    "Daily grid variability and extreme indices describe ensemble-mean rainfall, not member extremes or calibrated uncertainty.",
]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value):
    if isinstance(value, dict): return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray): return json_ready(value.tolist())
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, (float, np.floating)): return float(value) if np.isfinite(value) else None
    return value


def bilinear(field, lat, lon, station_lat, station_lon):
    """Sample (...,lat,lon); no extrapolation or filling missing land cells."""
    lat, lon = np.asarray(lat), np.asarray(lon)
    if not (np.all(np.diff(lat) > 0) and np.all(np.diff(lon) > 0)):
        raise ValueError("point sampling requires increasing coordinates")
    y = np.interp(station_lat, lat, np.arange(len(lat)))
    x = np.interp(station_lon, lon, np.arange(len(lon)))
    iy = np.minimum(np.floor(y).astype(int), len(lat)-2)
    ix = np.minimum(np.floor(x).astype(int), len(lon)-2)
    wy, wx = y-iy, x-ix
    result = np.zeros((*field.shape[:-2], len(y)), dtype=float)
    valid = np.ones(result.shape, dtype=bool)
    for dy, dx, weight in ((0, 0, (1-wy)*(1-wx)), (1, 0, wy*(1-wx)),
                           (0, 1, (1-wy)*wx), (1, 1, wy*wx)):
        values = field[..., iy+dy, ix+dx]
        valid &= (weight <= 1e-12) | np.isfinite(values)
        result += np.where(weight > 1e-12, np.nan_to_num(values)*weight, 0)
    inside = ((station_lat >= lat[0]) & (station_lat <= lat[-1]) &
              (station_lon >= lon[0]) & (station_lon <= lon[-1]))
    return np.where(valid & inside, result, np.nan)


def fair_crps(members, observed):
    """Fair CRPS of (member,station) ensembles, preserving missing values."""
    count = members.shape[0]
    ordered = np.sort(members, axis=0)
    weights = (2*np.arange(1, count+1)-count-1)[:, None]
    return np.mean(np.abs(members-observed), axis=0) - np.sum(ordered*weights, axis=0)/(count*(count-1))


def metrics(predicted, observed):
    keep = np.isfinite(predicted) & np.isfinite(observed)
    x, y = np.asarray(predicted)[keep], np.asarray(observed)[keep]
    if not len(x): return {"n": 0}
    diff = x-y
    corr = np.corrcoef(x, y)[0, 1] if len(x)>2 and x.std()>0 and y.std()>0 else np.nan
    wet_x, wet_y = x >= 1, y >= 1
    hit, miss, false = np.sum(wet_x & wet_y), np.sum(~wet_x & wet_y), np.sum(wet_x & ~wet_y)
    return {"n": len(x), "bias_mm": diff.mean(), "mae_mm": np.abs(diff).mean(),
            "rmse_mm": np.sqrt(np.mean(diff**2)), "correlation": corr,
            "wet_csi": hit/(hit+miss+false) if hit+miss+false else np.nan,
            "observed_mean_mm": y.mean(), "predicted_mean_mm": x.mean()}


def common_scores(frame, group_columns=()):
    """All sources use identical station-day rows, including ensemble columns."""
    frame = frame[np.isfinite(frame[["observed", *SOURCES]]).all(axis=1)]
    rows = []
    groups = frame.groupby(list(group_columns), observed=True) if group_columns else [((), frame)]
    for keys, selected in groups:
        if not isinstance(keys, tuple): keys = (keys,)
        base = dict(zip(group_columns, keys))
        for source in SOURCES:
            row = {**base, "product": source, **metrics(selected[source].to_numpy(), selected.observed.to_numpy())}
            if source in ("analysis", "background"):
                row.update(fair_crps_mm=selected[f"{source}_crps"].mean(),
                           interval90_coverage=selected[f"{source}_covered"].mean(),
                           rms_spread_mm=np.sqrt(np.mean(selected[f"{source}_spread"]**2)))
            rows.append(row)
    return pd.DataFrame(rows)


def monthly_station(frame, minimum=.9):
    frame = frame[np.isfinite(frame[["observed", *SOURCES]]).all(axis=1)].copy()
    frame["month"] = frame.date.dt.to_period("M").astype(str)
    keys = ["station_id", "network", "month"]
    values = frame.groupby(keys, observed=True)[["observed", *SOURCES]].sum()
    values["paired_days"] = frame.groupby(keys, observed=True).size()
    result = values.reset_index()
    result["calendar_days"] = pd.to_datetime(result.month).dt.days_in_month
    result["coverage"] = result.paired_days/result.calendar_days
    return result[result.coverage >= minimum].copy()


def block_mean(field, mask, factor):
    h, w = mask.shape
    if h % factor or w % factor: raise ValueError("grid does not divide into IMERG footprints")
    values = np.where(mask, field, np.nan).reshape(h//factor, factor, w//factor, factor)
    good = np.isfinite(values)
    count = good.sum(axis=(1, 3))
    return np.divide(np.where(good, values, 0).sum(axis=(1, 3)), count,
                     out=np.full(count.shape, np.nan), where=count > 0)


def area_mean(values, mask, lat):
    weights = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], mask.shape)
    good = mask & np.isfinite(values)
    if not good.any(): return np.nan
    return np.sum(values[good]*weights[good])/weights[good].sum()


class Accumulator:
    """Per-cell temporal moments/extremes, updated with daily common support."""
    def __init__(self, shape):
        self.count = np.zeros(shape, dtype=np.int32)
        self.total = np.zeros(shape); self.squared = np.zeros(shape)
        self.wet = np.zeros(shape); self.r10 = np.zeros(shape); self.r20 = np.zeros(shape)
        self.maximum = np.full(shape, -np.inf)
        self.dry_run = np.zeros(shape, dtype=np.int32); self.wet_run = self.dry_run.copy()
        self.cdd = self.dry_run.copy(); self.cwd = self.dry_run.copy()

    def add(self, field, valid):
        good = np.broadcast_to(valid, field.shape) & np.isfinite(field)
        x = np.where(good, field, 0)
        self.count += good; self.total += x; self.squared += x*x
        self.wet += good & (field >= 1); self.r10 += good & (field >= 10); self.r20 += good & (field >= 20)
        self.maximum = np.maximum(self.maximum, np.where(good, field, -np.inf))
        self.dry_run = np.where(good & (field < 1), self.dry_run+1, 0)
        self.wet_run = np.where(good & (field >= 1), self.wet_run+1, 0)
        self.cdd = np.maximum(self.cdd, self.dry_run); self.cwd = np.maximum(self.cwd, self.wet_run)

    def result(self, expected_days):
        mean = np.divide(self.total, self.count, out=np.full(self.total.shape, np.nan), where=self.count > 0)
        second = np.divide(self.squared, self.count, out=np.full(self.total.shape, np.nan), where=self.count > 0)
        complete = self.count == expected_days
        return {"count": self.count, "mean": mean, "daily_sd": np.sqrt(np.maximum(0, second-mean*mean)),
                "total": np.where(complete, self.total, np.nan),
                "wet_fraction": np.divide(self.wet, self.count, out=np.full(self.total.shape, np.nan), where=self.count > 0),
                **{name: np.where(complete, values, np.nan) for name, values in
                   (("rx1day", self.maximum), ("r10_days", self.r10), ("r20_days", self.r20),
                    ("cdd", self.cdd), ("cwd", self.cwd))}}


def require_final(root, first=None, last=None):
    status = load_module("_surma_status", ROOT/"scripts/104_surma_production_status.py")
    manifest = json.loads((root/"production_manifest.json").read_text())
    if manifest.get("status") != "complete" or manifest.get("members") != 30:
        raise ValueError("production final validation has not completed")
    archive_first, archive_last = int(manifest["start"][:4]), int(manifest["end"][:4])
    full_periods = status.quarters(archive_first, archive_last)
    shards = manifest.get("shards", [])
    labels = [s.get("period", {}).get("label") for s in shards]
    if (manifest["start"] != f"{archive_first}-01-01" or manifest["end"] != f"{archive_last}-12-31"
            or manifest.get("days") != sum(p["days"] for p in full_periods)
            or len(labels) != len(full_periods) or set(labels) != {p["label"] for p in full_periods}):
        raise ValueError("final receipt does not establish all unique requested quarters")
    first, last = first or archive_first, last or archive_last
    if not archive_first <= first <= last <= archive_last: raise ValueError("requested years outside production")
    periods = status.quarters(first, last)
    records = {s["period"]["label"]: s for s in shards}
    for period in periods:
        label = period["label"]
        receipt = json.loads((root/"validated"/f"{label}.json").read_text())
        if receipt != records[label] or not status.recorded_completion(root, period):
            raise ValueError(f"{label}: final receipt and completed artifacts disagree")
        for name, key in (("json", "report_sha256"), ("npz", "station_array_sha256")):
            path = root/"production_metadata"/f"{label}.{name}"
            if sha(path) != receipt.get(key): raise ValueError(f"{label}: station artifact checksum differs")
        prepared = root/"prepared"/f"{label}.json"
        if sha(prepared) != receipt.get("prepared_manifest_sha256"):
            raise ValueError(f"{label}: preparation provenance changed")
        preparation = json.loads(prepared.read_text())
        gauges = root/"stations"/label/"combined_daily.csv"
        recorded = [f for f in preparation.get("files", []) if Path(f["path"]).resolve() == gauges.resolve()]
        if len(recorded) != 1 or sha(gauges) != recorded[0].get("sha256"):
            raise ValueError(f"{label}: original gauge table differs from production preparation")
    return manifest, periods


def original_stations(root, label, times, lat, lon, country):
    folder = root/"stations"/label
    frame = pd.read_csv(folder/"combined_daily.csv", dtype={"station_id": str})
    summary = pd.read_csv(folder/"station_summary.csv", dtype={"station_id": str})
    eligible = summary.eligible_for_analysis.astype(str).str.lower().isin(("true", "1"))
    ids = set(summary.loc[eligible, "station_id"])
    frame.date = pd.to_datetime(frame.date)
    if frame.duplicated(["station_id", "date"]).any(): raise ValueError(f"{label}: duplicate original gauge days")
    metadata = frame.drop_duplicates("station_id").copy()
    inside = points_inside(metadata.lat.to_numpy(), metadata.lon.to_numpy(), country)
    inside &= ((metadata.lat >= lat[0]) & (metadata.lat <= lat[-1]) & (metadata.lon >= lon[0]) & (metadata.lon <= lon[-1]))
    metadata = metadata[inside & metadata.station_id.isin(ids)].sort_values("station_id")
    if metadata.empty: raise ValueError(f"{label}: no eligible Bangladesh gauges")
    observed = frame.pivot(index="date", columns="station_id", values="precip_mm").reindex(
        index=pd.to_datetime(times), columns=metadata.station_id).to_numpy(dtype=float)
    return metadata, observed


def process(root, periods, country, out):
    import zarr
    daily, agreement, station_parts, superob_rows = [], [], [], []
    months, years, source_paths, fallback_days = {}, {}, [], []
    baseline = None
    for number, period in enumerate(periods, 1):
        label = period["label"]; path = root/"gridded"/f"{label}.zarr"
        group = zarr.open_group(str(path), mode="r")
        times = np.asarray(group["time"][:]).astype("timedelta64[D]")+np.datetime64("1970-01-01")
        expected = np.arange(np.datetime64(period["start"]), np.datetime64(period["end"])+np.timedelta64(1, "D"))
        lat, lon = np.asarray(group["lat"][:]), np.asarray(group["lon"][:])
        mask = np.asarray(group["valid"][:], bool) & grid_mask(lat, lon, country)
        methods = np.asarray(group["method"][:]).astype(str).tolist()
        scope = dict(group.attrs["scope"])
        if (not np.array_equal(times, expected) or set(methods) != {"background", ANALYSIS}
                or group["precipitation"].shape[:3] != (2, len(times), 30)
                or scope.get("background_day_offset") != -1 or scope.get("assimilate_all_stations") is not True):
            raise ValueError(f"{label}: dates, ensemble or evaluated production scope differs")
        current = (lat, lon, mask)
        if baseline is None: baseline = current
        elif any(not np.array_equal(a, b) for a, b in zip(baseline, current)):
            raise ValueError(f"{label}: grid or validity mask changed")
        if not mask.any(): raise ValueError("empty Bangladesh land mask")
        shape = mask.shape
        im_shape = group["imerg"].shape[-2:]
        factor = shape[0]//im_shape[0]
        if factor < 1 or (im_shape[0]*factor, im_shape[1]*factor) != shape:
            raise ValueError("IMERG footprint dimensions mismatch")
        stations, observed = original_stations(root, label, times, lat, lon, country)
        slat, slon = stations.lat.to_numpy(), stations.lon.to_numpy()
        station_values = {s: np.full(observed.shape, np.nan, dtype=np.float32) for s in SOURCES}
        for s in ("analysis", "background"):
            for suffix in ("spread", "crps", "covered"):
                station_values[f"{s}_{suffix}"] = np.full(observed.shape, np.nan, dtype=np.float32)
        for index, day in enumerate(times):
            date_string = str(day); month, year = date_string[:7], date_string[:4]
            if month not in months: months[month] = Accumulator((len(SOURCES), *shape))
            if year not in years: years[year] = Accumulator((len(SOURCES), *shape))
            fields, spreads = {}, {}
            for s, method in (("analysis", ANALYSIS), ("background", "background")):
                mi = methods.index(method)
                fields[s] = np.asarray(group["ensemble_mean"][mi, index], float)
                spreads[s] = np.asarray(group["ensemble_std"][mi, index], float)
                members = np.asarray(group["precipitation"][mi, index], float)
                if not np.isfinite(members[:, mask]).all(): raise ValueError(f"{day}: nonfinite {s} land members")
                point = bilinear(members, lat, lon, slat, slon)
                station_values[s][index] = point.mean(axis=0)
                station_values[f"{s}_spread"][index] = point.std(axis=0, ddof=1)
                station_values[f"{s}_crps"][index] = fair_crps(point, observed[index])
                lo, hi = np.quantile(point, [.05, .95], axis=0)
                covered = ((observed[index] >= lo) & (observed[index] <= hi)).astype(float)
                covered[~np.isfinite(observed[index]) | ~np.isfinite(point).all(axis=0)] = np.nan
                station_values[f"{s}_covered"][index] = covered
            for s in ("cpc", "chirps"):
                fields[s] = np.asarray(group[s][index], float)
                station_values[s][index] = bilinear(fields[s], lat, lon, slat, slon)
            raw_imerg = np.asarray(group["imerg"][index], float)
            fields["imerg"] = np.repeat(np.repeat(raw_imerg, factor, axis=0), factor, axis=1)
            # IMERG is a footprint observation: sample its footprint, not an invented fine interpolation.
            # Cell edges are half a fine cell outside the coordinate centres.
            iy = np.clip(np.floor((slat-(lat[0]-(lat[1]-lat[0])/2))/(lat[1]-lat[0])).astype(int), 0, len(lat)-1)//factor
            ix = np.clip(np.floor((slon-(lon[0]-(lon[1]-lon[0])/2))/(lon[1]-lon[0])).astype(int), 0, len(lon)-1)//factor
            station_values["imerg"][index] = raw_imerg[iy, ix]
            stacked = np.stack([fields[s] for s in SOURCES])
            common = mask & np.isfinite(stacked).all(axis=0)
            if not common.any(): raise ValueError(f"{day}: no common finite product support inside Bangladesh")
            months[month].add(stacked, common); years[year].add(stacked, common)
            for s in SOURCES:
                values = fields[s][common]
                daily.append({"date": date_string, "product": s, "cells": int(common.sum()),
                    "domain_mean_mm": area_mean(fields[s], common, lat),
                    "spatial_sd_mm": float(values.std()) if len(values) else np.nan,
                    "spatial_p95_mm": float(np.quantile(values, .95)) if len(values) else np.nan,
                    "wet_area_fraction": area_mean((fields[s]>=1).astype(float), common, lat),
                    "domain_spread_mm": area_mean(spreads[s], common, lat) if s in spreads else np.nan})
            for reference in ("background", "cpc", "imerg", "chirps"):
                for scale in ("fine_grid", "imerg_footprint"):
                    a, b = fields["analysis"], fields[reference]
                    if scale == "imerg_footprint":
                        a, b = block_mean(a, common, factor), block_mean(b, common, factor)
                    else: a, b = a[common], b[common]
                    agreement.append({"date": date_string, "reference": reference, "scale": scale,
                                      **metrics(a.ravel(), b.ravel())})
        nday, nstation = observed.shape
        part = pd.DataFrame({"date": np.repeat(pd.to_datetime(times), nstation),
            "station_id": np.tile(stations.station_id, nday), "network": np.tile(stations.source, nday),
            "lat": np.tile(slat, nday), "lon": np.tile(slon, nday), "observed": observed.ravel(),
            **{s: values.ravel() for s, values in station_values.items()}})
        station_parts.append(part[np.isfinite(part.observed)].copy())
        with np.load(root/"production_metadata"/f"{label}.npz", allow_pickle=False) as saved:
            truth = saved["gauge_mm"]
            if (not np.array_equal(saved["times"].astype("datetime64[D]"), times)
                    or len(saved["eval_idx"]) or len(saved["assim_idx"]) != truth.shape[1]):
                raise ValueError(f"{label}: station archive is not all-station production")
            inside = points_inside(saved["station_lat"], saved["station_lon"], country)
            paired = np.isfinite(truth) & inside[None]
            ensemble = {s: saved[f"station_{m}"] for s, m in (("analysis", ANALYSIS), ("background", "background"))}
            for member in ensemble.values(): paired &= np.isfinite(member).all(axis=1)
            for s, member in ensemble.items():
                lo, hi = np.quantile(member, [.05, .95], axis=1)
                crps = fair_crps(member.transpose(1, 0, 2).reshape(30, -1), truth.ravel()).reshape(truth.shape)
                superob_rows.append({"quarter": label, "product": s,
                    **metrics(member.mean(axis=1)[paired], truth[paired]),
                    "fair_crps_mm": np.mean(crps[paired]),
                    "interval90_coverage": np.mean(((truth>=lo)&(truth<=hi))[paired])})
        source_paths.extend([path, root/"stations"/label/"combined_daily.csv",
                             root/"production_metadata"/f"{label}.npz"])
        fallback_days.extend(dict(group.attrs.get("cpc_background_qc", {})).get("missing_cpc_days", []))
        print(f"[evaluate] quarter {number}/{len(periods)} {label}: {len(times)} days; {nstation} original gauges", flush=True)
    station = pd.concat(station_parts, ignore_index=True)
    if not np.isfinite(station[["observed", *SOURCES]]).all(axis=1).any():
        raise ValueError("no jointly valid original gauge/product samples; inspect coordinates and products")
    daily = pd.DataFrame(daily); agreement = pd.DataFrame(agreement)
    daily.date = pd.to_datetime(daily.date)
    monthly = monthly_station(station)
    scores = common_scores(station, ("network",))
    station_scores = common_scores(station, ("network", "station_id", "lat", "lon"))
    station["year"] = station.date.dt.year; station["calendar_month"] = station.date.dt.month
    annual_scores = common_scores(station, ("network", "year"))
    seasonal_scores = common_scores(station, ("network", "calendar_month"))
    monthly_scores = []
    for network, selected in monthly.groupby("network"):
        for s in SOURCES:
            monthly_scores.append({"network": network, "product": s,
                                   **metrics(selected[s].to_numpy(), selected.observed.to_numpy())})
    tables = {"daily_domain": daily, "daily_product_agreement": agreement,
              "original_gauge_scores": scores, "station_scores": station_scores,
              "annual_gauge_scores": annual_scores, "seasonal_gauge_scores": seasonal_scores,
              "monthly_gauge_totals": monthly, "monthly_gauge_scores": pd.DataFrame(monthly_scores),
              "superobservation_fit": pd.DataFrame(superob_rows)}
    for name, table in tables.items(): table.to_csv(out/f"{name}.csv", index=False)
    # Large station-day tables go to gzip, retaining the full paired diagnostic sample.
    station.to_csv(out/"original_gauge_daily.csv.gz", index=False, compression="gzip")
    lat, lon, mask = baseline
    month_grids = {m: value.result(pd.Timestamp(m).days_in_month) for m, value in sorted(months.items())}
    year_grids = {y: value.result(366 if pd.Timestamp(f"{y}-12-31").is_leap_year else 365) for y, value in sorted(years.items())}
    return tables, station, month_grids, year_grids, lat, lon, mask, source_paths, fallback_days


def plot_all(out, tables, station, months, years, lat, lon, mask, country, sources):
    plt = use_paper_style()
    daily = tables["daily_domain"].copy()
    daily["month"] = daily.date.dt.to_period("M").astype(str)
    daily["year"] = daily.date.dt.year
    monthly = daily.groupby(["product", "month"], observed=True).agg(
        total_mm=("domain_mean_mm", "sum"), daily_sd_mm=("domain_mean_mm", "std"),
        wet_fraction=("wet_area_fraction", "mean"), spread_mm=("domain_spread_mm", "mean")).reset_index()
    monthly["calendar_month"] = pd.to_datetime(monthly.month).dt.month
    annual = daily.groupby(["product", "year"], observed=True).agg(
        total_mm=("domain_mean_mm", "sum"), daily_sd_mm=("domain_mean_mm", "std"),
        rx1_domain_mm=("domain_mean_mm", "max"), wet_fraction=("wet_area_fraction", "mean")).reset_index()
    monthly.to_csv(out/"monthly_domain.csv", index=False); annual.to_csv(out/"annual_domain.csv", index=False)
    figures = []

    def save(fig, number, slug, data, caption):
        fig.tight_layout()
        stem = save_figure(fig, out, number, slug, data=data, sources=sources, caption=caption)
        figures.append(stem.name); plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for s in SOURCES:
        selected = monthly[monthly["product"] == s]; dates = pd.to_datetime(selected.month)
        for ax, key in zip(axes, ("total_mm", "daily_sd_mm", "wet_fraction")):
            ax.plot(dates, selected[key], label=LABELS[s])
    for ax, title in zip(axes, ("Monthly Bangladesh mean total (mm)", "Within-month variability of daily domain mean (mm/day)", "Mean wet area fraction (>=1 mm/day)")): ax.set_ylabel(title)
    axes[0].legend(ncol=3); save(fig, "01", "monthly_domain_timeseries", {"monthly": monthly.to_dict("records")}, "Product agreement, varying common daily finite support; temporal support differs.")

    climatology = monthly.groupby(["product", "calendar_month"], observed=True).agg(
        mean_total_mm=("total_mm", "mean"), interannual_sd_mm=("total_mm", "std"),
        within_month_daily_sd_mm=("daily_sd_mm", "mean")).reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for s in SOURCES:
        selected = climatology[climatology["product"] == s]
        for ax, key in zip(axes, ("mean_total_mm", "interannual_sd_mm", "within_month_daily_sd_mm")):
            ax.plot(selected.calendar_month, selected[key], label=LABELS[s])
    for ax, title in zip(axes, ("Monthly climatology (mm/month)", "Interannual monthly SD (mm)", "Within-month daily SD (mm/day)")):
        ax.set(title=title, xlabel="Calendar month", xticks=range(1, 13))
    axes[0].legend(fontsize=6); save(fig, "02", "climatology_variability", {"climatology": climatology.to_dict("records")}, "Climatology of area-weighted domain means; uncertainty bands are not implied.")

    slopes = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for s in SOURCES:
        selected = annual[annual["product"] == s]
        for ax, key in zip(axes.ravel(), ("total_mm", "daily_sd_mm", "rx1_domain_mm", "wet_fraction")):
            ax.plot(selected.year, selected[key], marker=".", label=LABELS[s])
        slope = np.polyfit(selected.year, selected.total_mm, 1)[0] if len(selected)>1 else np.nan
        slopes.append({"product": s, "annual_total_slope_mm_per_year": slope})
    for ax, title in zip(axes.ravel(), ("Annual mean total (mm)", "Daily domain-mean SD (mm/day)", "Maximum daily domain mean (mm/day)", "Annual mean wet-area fraction")): ax.set(title=title, xlabel="Year")
    axes[0, 0].legend(fontsize=6); save(fig, "03", "annual_variability_extremes", {"annual": annual.to_dict("records"), "descriptive_trends": slopes}, "Annual changes describe ensemble means; slopes have no significance claim.")

    pooled = tables["original_gauge_scores"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for row_index, network in enumerate(("BMD", "BWDB")):
        subset = pooled[pooled.network == network].set_index("product").reindex(SOURCES)
        for ax, key in zip(axes[row_index], ("bias_mm", "rmse_mm", "correlation")):
            ax.bar(range(5), subset[key]); ax.set_xticks(range(5), SOURCES, rotation=25)
            ax.set_title(f"{network} contributing-gauge fit: {key}")
    save(fig, "04", "gauge_fit_metrics", {"paired_scores": pooled.to_dict("records")}, "Contributing gauges, identical paired station-days for all products; not held-out skill.")

    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    scatter_rows = []
    rng = np.random.default_rng(20261006)
    paired = station[np.isfinite(station[["observed", *SOURCES]]).all(axis=1)]
    for ri, network in enumerate(("BMD", "BWDB")):
        subset = paired[paired.network == network]
        sampled = subset.iloc[np.sort(rng.choice(len(subset), min(12000, len(subset)), replace=False))]
        for ci, s in enumerate(SOURCES):
            ax = axes[ri, ci]
            if len(sampled):
                ax.hexbin(sampled.observed, sampled[s], gridsize=40, bins="log", mincnt=1)
                bound = max(sampled.observed.max(), sampled[s].max()); ax.plot([0, bound], [0, bound], color="red", lw=.7)
            ax.set(title=f"{network}: {s}", xlabel="Gauge mm/day", ylabel="Product mm/day")
        scatter_rows.extend(sampled[["date", "station_id", "network", "observed", *SOURCES]].to_dict("records"))
    save(fig, "05", "gauge_daily_scatter", {"plotted_sample": scatter_rows}, "Seeded sample for display; scores use every common station-day.")

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    quantile_rows, exceedance_rows = [], []
    for ri, network in enumerate(("BMD", "BWDB")):
        subset = paired[paired.network == network]
        for s in SOURCES:
            probabilities = np.linspace(0, 1, 101)
            if subset.empty: continue
            observed = np.quantile(subset.observed, probabilities); product = np.quantile(subset[s], probabilities)
            axes[ri, 0].plot(observed, product, label=s)
            quantile_rows.extend({"network": network, "product": s, "quantile": p, "observed_mm": a, "product_mm": b}
                                 for p, a, b in zip(probabilities, observed, product))
        for s in ("observed", *SOURCES):
            wet = subset.loc[subset[s] >= 1, s].sort_values().to_numpy()
            if len(wet):
                thresholds = np.unique(np.quantile(wet, np.linspace(0, 1, 301)))
                exceedance = (len(wet)-np.searchsorted(wet, thresholds, side="left"))/len(wet)
                axes[ri, 1].plot(thresholds, exceedance, label="Gauge" if s=="observed" else s,
                                 color="black" if s=="observed" else None)
                exceedance_rows.extend({"network": network, "product": s, "threshold_mm": x,
                    "wet_samples": len(wet), "conditional_exceedance": p} for x, p in zip(thresholds, exceedance))
        if not subset.empty:
            bound = subset.observed.max(); axes[ri, 0].plot([0, bound], [0, bound], color="black", ls="--")
        axes[ri, 0].set(title=f"{network}: marginal QQ", xlabel="Gauge quantile mm/day", ylabel="Product quantile mm/day")
        axes[ri, 1].set(title=f"{network}: wet-day exceedance", xlabel="Rainfall mm/day", ylabel="P(rain >= x | rain >=1)", yscale="log")
        axes[ri, 1].legend(fontsize=6)
    save(fig, "06", "rainfall_distributions", {"quantiles": quantile_rows, "exceedance": exceedance_rows}, "Marginal distributions on identical station-days; empirical exceedance evaluated at up to 301 wet-day quantiles.")

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ri, network in enumerate(("BMD", "BWDB")):
        for s in SOURCES:
            selected = tables["seasonal_gauge_scores"]
            selected = selected[(selected.network == network) & (selected["product"] == s)]
            axes[ri, 0].plot(selected.calendar_month, selected.rmse_mm, label=s)
            selected = tables["annual_gauge_scores"]
            selected = selected[(selected.network == network) & (selected["product"] == s)]
            axes[ri, 1].plot(selected.year, selected.bias_mm, label=s)
        axes[ri, 0].set(title=f"{network}: seasonal fit RMSE", xlabel="Calendar month", ylabel="mm/day")
        axes[ri, 1].set(title=f"{network}: yearly fit bias", xlabel="Year", ylabel="mm/day")
    axes[0, 0].legend(fontsize=6); save(fig, "07", "gauge_seasonal_annual_fit", {"seasonal": tables["seasonal_gauge_scores"].to_dict("records"), "annual": tables["annual_gauge_scores"].to_dict("records")}, "Changing station networks and assimilation constrain interpretation of temporal changes.")

    maps = {}
    annual_names = ("total", "daily_sd", "wet_fraction", "rx1day", "r10_days", "r20_days", "cdd", "cwd")
    for name in annual_names:
        stack = np.stack([values[name] for values in years.values()])
        finite = np.isfinite(stack); count = finite.sum(axis=0)
        maps[name] = np.divide(np.where(finite, stack, 0).sum(axis=0), count,
                              out=np.full(stack.shape[1:], np.nan), where=count > 0)
    np.savez_compressed(out/"climatology_grids.npz", lat=lat, lon=lon, products=np.asarray(SOURCES), **maps)
    map_data = []
    yy, xx = np.meshgrid(lat, lon, indexing="ij")
    for index, s in enumerate(SOURCES):
        for name, fields in maps.items():
            map_data.extend({"product": s, "quantity": name, "lat": la, "lon": lo, "value": value}
                            for la, lo, value in zip(yy[mask], xx[mask], fields[index][mask]))
    for number, names, slug in (("08", ("total", "daily_sd", "wet_fraction"), "climatology_maps"),
                                 ("09", ("rx1day", "r20_days", "cdd"), "extremes_maps")):
        fig, axes = plt.subplots(3, 5, figsize=(16, 10))
        for ri, name in enumerate(names):
            values = maps[name][:, mask]; vmax = np.nanpercentile(values, 99) if np.isfinite(values).any() else 1
            for ci, s in enumerate(SOURCES):
                image = map_layer(axes[ri, ci], maps[name][ci], lat, lon, country, vmin=0, vmax=max(vmax, 1e-6))
                axes[ri, ci].set_title(f"{s}: {name}"); fig.colorbar(image, ax=axes[ri, ci], shrink=.65)
        save(fig, number, slug, {"grids": [r for r in map_data if r["quantity"] in names]}, "Mean annual indices of daily ensemble means; CPC/CHIRPS and S04 footprints have different spatial/temporal support.")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ss = tables["station_scores"]
    ss = ss[(ss["product"] == "analysis") & (ss.n >= 30)]
    for ax, key in zip(axes.ravel(), ("bias_mm", "rmse_mm", "correlation", "interval90_coverage")):
        # Blank Bangladesh outline, then coloured station points.
        map_layer(ax, np.full(mask.shape, np.nan), lat, lon, country)
        points = ax.scatter(ss.lon, ss.lat, c=ss[key], s=12, cmap="coolwarm" if key == "bias_mm" else "viridis", zorder=7)
        fig.colorbar(points, ax=ax, shrink=.7); ax.set_title(f"SURMA contributing gauge fit: {key}")
    save(fig, "10", "station_fit_maps", {"station_scores": ss.to_dict("records")}, "Stations with at least 30 paired days; intervals describe assimilated fit, not calibrated coverage.")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    ensemble_rows = []
    for network in ("BMD", "BWDB"):
        subset = paired[paired.network == network]
        for s in ("background", "analysis"):
            label = f"{network} {s}"
            spread = subset[f"{s}_spread"]
            if subset.empty: continue
            edges = np.unique(np.quantile(spread, np.linspace(0, 1, 9)))
            bins = pd.cut(spread, edges, include_lowest=True, duplicates="drop") if len(edges)>1 else pd.Series("constant", index=subset.index)
            for _, indices in subset.groupby(bins, observed=True).groups.items():
                selected = subset.loc[indices]
                row = {"network": network, "product": s, "n": len(selected),
                    "rms_spread_mm": np.sqrt(np.mean(selected[f"{s}_spread"]**2)),
                    "rmse_mm": np.sqrt(np.mean((selected[s]-selected.observed)**2)),
                    "coverage90": selected[f"{s}_covered"].mean()}
                ensemble_rows.append(row)
            rows = [r for r in ensemble_rows if r["network"]==network and r["product"]==s]
            axes[0].plot([r["rms_spread_mm"] for r in rows], [r["rmse_mm"] for r in rows], marker="o", label=label)
            axes[1].scatter([label], [subset[f"{s}_covered"].mean()])
            axes[2].scatter([label], [subset[f"{s}_crps"].mean()])
    axes[0].set(xlabel="RMS spread mm/day", ylabel="RMSE mm/day", title="Assimilated fit: spread/error"); axes[0].legend(fontsize=6)
    axes[1].axhline(.9, color="grey", ls="--"); axes[1].set_title("Assimilated fit: 90% interval coverage")
    axes[2].set_title("Assimilated fit: fair CRPS mm/day")
    for ax in axes[1:]: ax.tick_params(axis="x", rotation=30)
    save(fig, "11", "ensemble_fit_diagnostics", {"spread_bins": ensemble_rows, "network_scores": pooled.to_dict("records"), "exact_superob_fit": tables["superobservation_fit"].to_dict("records")}, "Assimilated intervals and CRPS cannot establish independent ensemble calibration.")

    agreement = tables["daily_product_agreement"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    agreement_summary = agreement.groupby(["reference", "scale"]).agg(
        mean_daily_rmse_mm=("rmse_mm", "mean"), mean_daily_correlation=("correlation", "mean")).reset_index()
    for scale in ("fine_grid", "imerg_footprint"):
        subset = agreement_summary[agreement_summary.scale == scale]
        axes[0].plot(subset.reference, subset.mean_daily_rmse_mm, marker="o", label=scale)
        axes[1].plot(subset.reference, subset.mean_daily_correlation, marker="o", label=scale)
    axes[0].set_title("Product agreement: average daily spatial RMSE (mm/day)")
    axes[1].set_title("Product agreement: average daily spatial correlation"); axes[0].legend()
    save(fig, "12", "product_agreement_scales", {"agreement": agreement_summary.to_dict("records")}, "Area cells are common finite support; repeated IMERG S04 cells do not provide independent fine-scale truth.")

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    monthly_gauge = tables["monthly_gauge_totals"]
    gauge_clim = monthly_gauge.copy(); gauge_clim["calendar_month"] = pd.to_datetime(gauge_clim.month).dt.month
    gauge_clim = gauge_clim.groupby(["network", "calendar_month"])[["observed", *SOURCES]].mean().reset_index()
    for ri, network in enumerate(("BMD", "BWDB")):
        selected = gauge_clim[gauge_clim.network == network]
        for s in ("observed", *SOURCES): axes[ri, 0].plot(selected.calendar_month, selected[s], label=s)
        subset = monthly_gauge[monthly_gauge.network == network]
        if len(subset): axes[ri, 1].hexbin(subset.observed, subset.analysis, gridsize=40, bins="log", mincnt=1)
        axes[ri, 0].set(title=f"{network}: paired station-month climatology", xlabel="Calendar month", ylabel="Observed-day total mm")
        axes[ri, 1].set(title=f"{network}: SURMA monthly fit", xlabel="Gauge total mm", ylabel="SURMA total mm")
    axes[0, 0].legend(fontsize=6)
    save(fig, "13", "gauge_monthly_climatology", {"climatology": gauge_clim.to_dict("records"), "paired_months": monthly_gauge.to_dict("records"), "monthly_scores": tables["monthly_gauge_scores"].to_dict("records")}, ">=90% common station-day coverage; changing network, sums are not scaled to fill missing days.")
    seasonal_maps, season_rows = {}, []
    for season, calendar_months in (("DJF", (12, 1, 2)), ("MAM", (3, 4, 5)),
                                   ("JJA", (6, 7, 8)), ("SON", (9, 10, 11))):
        layers = []
        for month in calendar_months:
            selected = np.stack([m["total"] for key, m in months.items() if int(key[-2:]) == month])
            good = np.isfinite(selected); count = good.sum(axis=0)
            layers.append(np.divide(np.where(good, selected, 0).sum(axis=0), count,
                out=np.full(selected.shape[1:], np.nan), where=count > 0))
        seasonal_maps[season] = np.sum(layers, axis=0)
        for si, s in enumerate(SOURCES):
            season_rows.extend({"season": season, "product": s, "lat": la, "lon": lo, "total_mm": val}
                               for la, lo, val in zip(yy[mask], xx[mask], seasonal_maps[season][si][mask]))
    fig, axes = plt.subplots(4, 5, figsize=(16, 12))
    for ri, (season, layer) in enumerate(seasonal_maps.items()):
        finite = layer[:, mask]; vmax = np.nanpercentile(finite, 99) if np.isfinite(finite).any() else 1
        for ci, s in enumerate(SOURCES):
            image = map_layer(axes[ri, ci], layer[ci], lat, lon, country, vmin=0, vmax=max(vmax, 1e-6))
            axes[ri, ci].set_title(f"{s}: {season}"); fig.colorbar(image, ax=axes[ri, ci], shrink=.65)
    save(fig, "14", "seasonal_climatology_maps", {"grids": season_rows}, "Sum of the three monthly climatological totals (mm/season); DJF is climatological, not a sequence of complete winter years.")

    annual_totals = np.stack([m["total"] for m in years.values()])
    complete = np.isfinite(annual_totals).all(axis=0)
    year_axis = np.asarray(list(years), float)
    if len(year_axis)>1:
        anomalies = year_axis-year_axis.mean()
        slope = np.sum(annual_totals*anomalies[:, None, None, None], axis=0)/np.sum(anomalies**2)
        variability = np.std(annual_totals, axis=0, ddof=1)
    else: slope = variability = np.full(annual_totals.shape[1:], np.nan)
    slope, variability = np.where(complete, slope, np.nan), np.where(complete, variability, np.nan)
    trend_rows = []
    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    for ri, (name, layer) in enumerate((("Interannual annual-total SD (mm)", variability), ("Annual-total slope (mm/year)", slope))):
        values = layer[:, mask]; limit = np.nanpercentile(np.abs(values), 99) if np.isfinite(values).any() else 1
        for ci, s in enumerate(SOURCES):
            image = map_layer(axes[ri, ci], layer[ci], lat, lon, country,
                             vmin=-max(limit, 1e-6) if ri else 0, vmax=max(limit, 1e-6), cmap="RdBu_r" if ri else "YlGnBu")
            axes[ri, ci].set_title(f"{s}: {name}"); fig.colorbar(image, ax=axes[ri, ci], shrink=.65)
            trend_rows.extend({"quantity": name, "product": s, "lat": la, "lon": lo, "value": val}
                             for la, lo, val in zip(yy[mask], xx[mask], layer[ci][mask]))
    save(fig, "15", "interannual_variability_trends", {"grids": trend_rows}, "Requires complete annual totals in every evaluated year. Slopes are descriptive; no significance, causal or independent skill claim.")
    # Save monthly maps for later custom plotting, without keeping daily fields.
    np.savez_compressed(out/"monthly_grids.npz", lat=lat, lon=lon, products=np.asarray(SOURCES),
                        months=np.asarray(list(months)), totals=np.stack([m["total"] for m in months.values()]))
    np.savez_compressed(out/"annual_grids.npz", lat=lat, lon=lon, products=np.asarray(SOURCES),
                        years=year_axis.astype(int), totals=annual_totals, total_slope=slope, interannual_sd=variability)
    return figures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/processed/brishti05_production_2001_2024"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/surma_production_2001_2024_diagnostics"))
    parser.add_argument("--start-year", type=int); parser.add_argument("--end-year", type=int)
    parser.add_argument("--boundary-geojson", type=Path, default=DEFAULT_BOUNDARY)
    args = parser.parse_args(argv)
    manifest, periods = require_final(args.root, args.start_year, args.end_year)
    country, boundary = read_boundary(args.boundary_geojson)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    receipt = args.out_dir/"evaluation.json"
    # A failed rerun cannot leave a stale success receipt behind.
    receipt.unlink(missing_ok=True)
    tables, station, months, years, lat, lon, mask, sources, fallback = process(args.root, periods, country, args.out_dir)
    figures = plot_all(args.out_dir, tables, station, months, years, lat, lon, mask, country, sources)
    report = {"status": "complete", "generated_utc": datetime.now(timezone.utc).isoformat(),
              "production_root": str(args.root.resolve()), "production_manifest_sha256": sha(args.root/"production_manifest.json"),
              "start": periods[0]["start"], "end": periods[-1]["end"], "days": sum(p["days"] for p in periods),
              "quarters": len(periods), "members": manifest["members"], "boundary": boundary,
              "figures": figures, "products": LABELS, "missing_cpc_days": fallback, "notes": NOTES,
              "gauge_scores": tables["original_gauge_scores"].to_dict("records")}
    receipt.write_text(json.dumps(json_ready(report), indent=2, allow_nan=False)+"\n")
    (args.out_dir/"README.md").write_text("# SURMA production diagnostics\n\n"+
        f"{report['start']} through {report['end']}; {report['days']:,} days; {len(figures)} figures (PNG/PDF).\n\n"+
        "\n".join(f"- {note}" for note in NOTES)+"\n\nFigures and their CSV inputs are recorded in `data/fig*_manifest.json`.\n")
    print(f"[done] {report['days']:,} days; {len(figures)} figures -> {args.out_dir}")


if __name__ == "__main__": main()
