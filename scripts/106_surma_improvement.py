#!/usr/bin/env python3
"""Separate QC and withheld-gauge pilot using existing SURMA production inputs.

No station source, checkpoint or production output is edited. QC flags are
review candidates, never automatic exclusions. Run --help for the four stages.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
BASELINE = "dense_s6_bwdb_r4"
VARIANTS = ["background", BASELINE, "improve_t115", "improve_t125",
            "improve_euler", "improve_noise015"]
NOTES = [
    "QC flags require source review; genuine localized extremes must not be removed merely for disagreement.",
    "Withheld gauges are excluded before aggregation and error-budget estimation, including their cell and distance buffer.",
    "Independence is from this assimilation only; upstream IMERG/CPC/CHIRPS station overlap remains unresolved.",
    "Historical windows overlap prior training years and are sensitivity checks, not unseen-year skill evidence.",
    "Default test windows were previously inspected in full-production diagnostics; they are reserved from this tuning, not a fresh blind benchmark.",
    "Noise is compared against an Euler control; initial draws and observation perturbations are paired, subsequent stochastic paths differ.",
    "90% field intervals exclude gauge measurement/representativeness uncertainty; coverage is diagnostic, not calibrated predictive coverage.",
    "No automatic winner or full-production submission: choose on tune cases, then assess that frozen choice on test cases.",
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def record(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def verify(records):
    for item in records:
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("input changed; use a fresh pilot directory: " + item["path"])


def windows(path):
    result = json.loads(Path(path).read_text())["windows"]
    seen, occupied = set(), set()
    for w in result:
        start, end = pd.Timestamp(w["start"]), pd.Timestamp(w["end"])
        if (not re.fullmatch(r"[A-Za-z0-9_-]+", w["label"]) or w["label"] in seen
                or w["role"] not in {"historical", "tune", "test"} or end < start
                or start.to_period("Q") != end.to_period("Q")
                or start != start.normalize() or end != end.normalize()):
            raise ValueError("invalid, duplicate or cross-quarter window: " + str(w))
        days = set(pd.date_range(start, end).strftime("%Y-%m-%d"))
        if occupied & days:
            raise ValueError("windows overlap")
        if days & set(pd.date_range("2022-05-01", "2022-05-31").strftime("%Y-%m-%d")):
            raise ValueError("exclude the original May 2022 configuration-selection month")
        seen.add(w["label"]); occupied |= days
        w["quarter"] = f"{start.year}_q{start.quarter}"
        w["days"] = len(days)
    if not result or not any(w["role"] == "tune" for w in result):
        raise ValueError("provide at least one tuning window")
    return result


def read_gauges(paths):
    frames = [pd.read_csv(p, dtype={"station_id": str}, parse_dates=["date"]) for p in paths]
    if not frames:
        raise ValueError("no archived combined_daily.csv files found; prepare production inputs first")
    frame = pd.concat(frames, ignore_index=True)
    required = {"station_id", "lat", "lon", "date", "precip_mm"}
    if not required <= set(frame):
        raise ValueError("canonical station table lacks " + str(required - set(frame)))
    if frame.station_id.isna().any() or (frame.station_id.str.strip() == "").any():
        raise ValueError("empty station identifiers")
    if frame.date.isna().any() or not (frame.date == frame.date.dt.normalize()).all():
        raise ValueError("invalid or non-daily station timestamps")
    if frame.duplicated(["station_id", "date"]).any():
        raise ValueError("duplicate original station-days; resolve before pilot preparation")
    if frame[["lat", "lon"]].isna().any().any() or not np.isfinite(frame[["lat", "lon"]]).all().all():
        raise ValueError("non-finite station coordinates")
    if ((frame.lat.abs() > 90) | (frame.lon.abs() > 180)).any():
        raise ValueError("station coordinates outside physical bounds")
    if (frame.groupby("station_id")[["lat", "lon"]].nunique() > 1).any().any():
        raise ValueError("station coordinates change between archived quarters; review station identity")
    if (frame.precip_mm < 0).any() or np.isinf(frame.precip_mm).any():
        raise ValueError("negative or infinite rainfall in canonical gauges")
    return frame.sort_values(["station_id", "date"]).reset_index(drop=True)


def distance(sites):
    lat = np.radians(sites.lat.to_numpy(float))
    lon = np.radians(sites.lon.to_numpy(float))
    a = np.sin((lat[:, None]-lat[None, :])/2)**2
    a += np.cos(lat[:, None])*np.cos(lat[None, :])*np.sin((lon[:, None]-lon[None, :])/2)**2
    return 12742.0176*np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def qc_tables(frame, radius=50):
    """Compare monthly sums on paired days; no filling or satellite-derived edits."""
    sites = frame.groupby("station_id")[["lat", "lon"]].first().sort_index()
    rain = frame.pivot(index="date", columns="station_id", values="precip_mm").reindex(columns=sites.index)
    dist = distance(sites)
    bwdb = np.array([s.startswith("BWDB_") for s in sites.index])
    monthly_rows, flags = [], []
    for j, sid in enumerate(sites.index):
        neighbors = np.flatnonzero(bwdb & (np.arange(len(sites)) != j) & (dist[j] <= radius))
        neighbors = neighbors[np.argsort(dist[j, neighbors])][:5]
        if not len(neighbors):
            continue
        focal = rain.iloc[:, j]
        for month, values in rain.iloc[:, neighbors].groupby(rain.index.to_period("M")):
            observed = focal.reindex(values.index)
            pairs = values.notna().mul(observed.notna(), axis=0)
            counts = pairs.sum()
            adequate = counts >= int(np.ceil(month.days_in_month*.8))
            comparison = values.where(pairs).sum(min_count=1)[adequate]
            station_totals = pairs.mul(observed.fillna(0), axis=0).sum()[adequate]
            if len(comparison) < 3:
                continue
            obs = float(station_totals.median())
            ref = float(comparison.median())
            row = {"station_id": sid, "month": str(month), "paired_station_total_mm": obs,
                   "nearby_bwdb_total_mm": ref, "ratio": obs/max(ref, 1.0),
                   "neighbors": len(comparison), "minimum_paired_days": int(counts[adequate].min()),
                   "neighbor_ids": "|".join(comparison.index), "radius_km": radius}
            monthly_rows.append(row)
            if obs >= 500 and obs >= 4*max(ref, 1.0):
                flags.append({**row, "reason": "monthly wet disagreement; inspect source", "automatic_exclusion": False})
        values = rain.iloc[:, neighbors]
        neighbor_count = values.notna().sum(axis=1)
        med = values.median(axis=1)
        suspect = (focal >= 200) & (neighbor_count >= 3) & (focal >= 4*med.clip(lower=1))
        for day in rain.index[suspect]:
            flags.append({"station_id": sid, "date": str(day.date()), "reason": "daily extreme disagreement; inspect source",
                          "observed_mm": float(focal.loc[day]), "nearby_bwdb_mm": float(med.loc[day]),
                          "neighbors": int(neighbor_count.loc[day]), "automatic_exclusion": False})
    monthly_columns = ["station_id", "month", "paired_station_total_mm", "nearby_bwdb_total_mm", "ratio",
                       "neighbors", "minimum_paired_days", "neighbor_ids", "radius_km"]
    return pd.DataFrame(monthly_rows, columns=monthly_columns), pd.DataFrame(flags, columns=None if flags else ["station_id", "reason", "automatic_exclusion"])


def qc(args):
    out = Path(args.out_dir)/"qc"; out.mkdir(parents=True, exist_ok=True)
    paths = sorted(Path(args.root).glob("stations/*/combined_daily.csv"))
    print(f"[qc] reading {len(paths)} archived quarterly original-gauge tables", flush=True)
    frame = read_gauges(paths)
    neighbors, flags = qc_tables(frame, args.neighbor_km)
    neighbors.to_csv(out/"nearby_monthly_comparisons.csv", index=False)
    flags.to_csv(out/"review_flags.csv", index=False)
    coverage = frame.groupby("station_id").agg(lat=("lat", "first"), lon=("lon", "first"),
                first=("date", "min"), last=("date", "max"), available_days=("precip_mm", "count"),
                maximum_mm=("precip_mm", "max"))
    coverage.to_csv(out/"station_coverage.csv")
    template = out/"reviewed_exclusions.csv"
    if not template.exists():
        pd.DataFrame(columns=["station_id", "start", "end", "reason"]).to_csv(template, index=False)
    write_json(out/"qc.json", {"status": "review_candidates_only", "rows": len(frame),
        "stations": len(coverage), "flags": len(flags), "files": [record(p) for p in paths], "notes": NOTES})
    print(f"[qc] {len(flags)} review flags; no observations changed. See {out}", flush=True)


def apply_exclusions(frame, path):
    frame = frame.copy()
    if not path:
        return frame, 0
    exclusions = pd.read_csv(path, dtype=str).fillna("")
    if not {"station_id", "start", "end", "reason"} <= set(exclusions):
        raise ValueError("reviewed exclusions need station_id,start,end,reason")
    count = 0
    for row in exclusions.to_dict("records"):
        start, end = pd.Timestamp(row["start"]), pd.Timestamp(row["end"])
        if not row["reason"].strip() or pd.isna(start) or pd.isna(end) or end < start:
            raise ValueError("exclusions require valid dates and a review reason")
        if row["station_id"] not in set(frame.station_id):
            raise ValueError("unknown exclusion station: " + row["station_id"])
        mask = (frame.station_id == row["station_id"]) & frame.date.between(start, end)
        count += int((mask & frame.precip_mm.notna()).sum())
        frame.loc[mask, "precip_mm"] = np.nan
    return frame, count


def cells(sites, size=.25):
    return np.floor(sites.lat/size).astype(int)*100000 + np.floor(sites.lon/size).astype(int)


def split_stations(sites, fraction, seed, buffer_km):
    """Choose cell clusters, exclude their whole cell and all nearby stations."""
    keys = cells(sites).to_numpy()
    rng = np.random.default_rng(seed)
    order = rng.permutation(np.unique(keys))
    chosen = []
    target = max(5, int(np.ceil(len(sites)*fraction)))
    for cell in order:
        chosen.extend(np.flatnonzero(keys == cell).tolist())
        if len(chosen) >= target:
            break
    held = np.unique(chosen)
    dist = distance(sites)
    blocked = np.isin(keys, keys[held]) | (dist[:, held].min(axis=1) < buffer_km)
    retained = np.flatnonzero(~blocked)
    if len(held) < 5 or len(retained) < 10:
        raise ValueError("too few withheld/retained stations; reduce holdout fraction or buffer")
    return set(sites.index[held]), set(sites.index[retained])


def run_script(name, *arguments):
    command = [sys.executable, "-u", str(ROOT/"scripts"/name), *map(str, arguments)]
    print("[pilot] " + " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


def prepare(args):
    from bdhires.grids import get_grid
    out, source = Path(args.out_dir), Path(args.root)
    plan_path = out/"pilot_plan.json"
    if plan_path.exists():
        raise ValueError("pilot_plan.json already exists; keep it immutable or choose a fresh --out-dir")
    cases = windows(args.windows)
    source_paths = sorted({source/"stations"/w["quarter"]/"combined_daily.csv" for w in cases})
    source_records = []
    for quarter in sorted({w["quarter"] for w in cases}):
        prepared = source/"prepared"/f"{quarter}.json"
        payload = json.loads(prepared.read_text())
        if payload["statistics_sha256"] != sha(args.stats):
            raise ValueError("pilot statistics differ from the production preparation transform")
        # Verify selected original tables and IMERG against production receipts.
        for path in (source/"stations"/quarter/"combined_daily.csv", source/"imerg_s04"/f"{quarter}.nc"):
            matches = [r for r in payload["files"] if Path(r["path"]).resolve() == path.resolve()]
            if len(matches) != 1 or sha(path) != matches[0]["sha256"]:
                raise ValueError("source differs from prepared production receipt: " + str(path))
            source_records.append(record(path))
        source_records.append(record(prepared))
    frame, excluded = apply_exclusions(read_gauges(source_paths), args.exclusions)
    grid = get_grid("bd"); lo, la, hi, ha = grid.bbox; margin = grid.res/2
    sites = frame.groupby("station_id")[["lat", "lon"]].first().sort_index()
    sites = sites[(sites.lon > lo+margin) & (sites.lon < hi-margin) &
                  (sites.lat > la+margin) & (sites.lat < ha-margin)]
    common = set(sites.index)
    for w in cases:
        subset = frame[frame.date.between(w["start"], w["end"])]
        count = subset.groupby("station_id").precip_mm.count()
        common &= set(count[count >= args.min_coverage*w["days"]].index)
    sites = sites.loc[sorted(common)]
    if len(sites) < 20:
        raise ValueError("fewer than 20 stations have adequate coverage in every window")
    held, retained = split_stations(sites, args.withhold, args.seed, args.buffer_km)
    out.mkdir(parents=True, exist_ok=True)
    holdout = out/"holdout_ids.txt"; holdout.write_text("\n".join(sorted(held)) + "\n")
    site_report = sites.copy()
    site_report["role"] = ["withheld" if s in held else "assimilated" if s in retained else "buffer_excluded" for s in sites.index]
    site_report.to_csv(out/"station_split.csv")
    predictor_receipt = source/"predictor_validation.json"
    predictor = json.loads(predictor_receipt.read_text())["predictors"]
    if not Path(predictor["path"]).exists():
        raise FileNotFoundError(predictor["path"])
    shared = [record(holdout), record(args.windows), record(args.ckpt), record(args.stats),
              record(args.config), record(predictor_receipt), record(__file__),
              record(ROOT/"scripts/28_simultaneous_method_sweep.py"), record(ROOT/"src/bdhires/da/sampler.py")]
    if args.exclusions:
        shared.append(record(args.exclusions))
    for w in cases:
        folder = out/"prepared"/w["label"]; folder.mkdir(parents=True, exist_ok=True)
        selected = frame[frame.station_id.isin(held | retained) & frame.date.between(w["start"], w["end"])]
        original = folder/"original_qc.csv"; selected.to_csv(original, index=False, date_format="%Y-%m-%d")
        table, budget = folder/"superob.csv", folder/"superob.json"
        run_script("87_superob_dense_gauges.py", "--stations", original, "--stats", args.stats,
                   "--holdout-ids", holdout, "--cell-deg", .25, "--protect-withheld-km", args.buffer_km,
                   "--fail-under-km", args.buffer_km, "--out", table, "--report", budget)
        superob_ids = pd.read_csv(table, usecols=["station_id"]).station_id.astype(str)
        if not (superob_ids.str.startswith("BWDB_") & ~superob_ids.isin(held)).any():
            raise ValueError("pilot has no unmerged assimilated BWDB gauges; the frozen BWDB R x4 arm would be inactive")
        representation = json.loads(budget.read_text())["recommended_representativeness"]
        if not representation or representation["superob_implied_representativeness"] is None:
            raise ValueError("insufficient co-reporting assimilated gauges for measured error budget")
        w.update(stations=str(table.resolve()), representation=representation["superob_implied_representativeness"],
                 imerg=str((source/"imerg_s04"/f"{w['quarter']}.nc").resolve()),
                 files=[record(original), record(table), record(budget)])
    plan = {"status": "prepared_research_pilot", "windows": cases, "variants": VARIANTS,
            "checkpoint": str(Path(args.ckpt).resolve()), "config": str(Path(args.config).resolve()),
            "stats": str(Path(args.stats).resolve()), "predictors": str(Path(predictor["path"]).resolve()),
            "fill_known_cpc_gaps": predictor.get("cpc_gap_policy") == "known_source_gaps_previous_day_cpc",
            "members": args.members, "seed": args.seed, "min_coverage": args.min_coverage,
            "holdout": str(holdout.resolve()), "held_stations": len(held), "retained_stations": len(retained),
            "buffer_km": args.buffer_km, "excluded_station_days": excluded,
            "qc_review": "explicit reviewed exclusions" if args.exclusions else "no reviewed exclusions supplied",
            "shared_files": shared, "source_files": source_records, "notes": NOTES}
    write_json(plan_path, plan)
    print(f"[prepared] {len(cases)} windows, {sum(w['days'] for w in cases)} days, {len(held)} withheld stations. {plan_path}")


def load_plan(args):
    plan = json.loads((Path(args.out_dir)/"pilot_plan.json").read_text())
    verify(plan["shared_files"])
    verify(plan["source_files"])
    return plan


def run(args):
    plan = load_plan(args)
    if args.task is not None and not 0 <= args.task < len(plan["windows"]):
        raise ValueError("--task outside pilot window range")
    cases = plan["windows"] if args.task is None else [plan["windows"][args.task]]
    run_script("51_check_sqrt_da_gradient.py")
    for w in cases:
        verify(w["files"])
        folder = Path(args.out_dir)/"runs"/w["label"]; folder.mkdir(parents=True, exist_ok=True)
        report, arrays, receipt = folder/"sweep.json", folder/"sweep.npz", folder/"complete.json"
        if receipt.exists():
            payload = json.loads(receipt.read_text())
            if payload["plan_sha256"] != sha(Path(args.out_dir)/"pilot_plan.json"):
                raise ValueError("completed run uses different plan")
            verify(payload["files"])
            print("[reuse] " + w["label"], flush=True); continue
        if report.exists() or arrays.exists():
            raise ValueError("partial pilot output; inspect and move aside before retry: " + str(folder))
        command = ["--config", plan["config"], "--ckpt", plan["checkpoint"], "--data-zarr", plan["predictors"],
                   "--stations", w["stations"], "--imerg", w["imerg"], "--start", w["start"], "--end", w["end"],
                   "--members", plan["members"], "--seed", plan["seed"], "--min-coverage", plan["min_coverage"],
                   "--holdout-station-ids-file", plan["holdout"], "--background-day-offset", -1,
                   "--group", "v2_production_improvement", "--set", "observations.imerg.factor=8",
                   "--set", "observations.imerg.error_corr_cells=0.75", "--set", "sampler.noise_scale=0.0",
                   "--set", "sampler.heun=true", "--set", f"observations.gauges.representativeness={w['representation']}",
                   "--out", arrays, "--report", report]
        if plan["fill_known_cpc_gaps"]:
            command.append("--fill-known-cpc-gaps")
        run_script("28_simultaneous_method_sweep.py", *command)
        scope = json.loads(report.read_text())["scope"]
        if (scope["start"] != w["start"] or scope["end"] != w["end"]
                or scope["members"] != plan["members"] or scope["assimilate_all_stations"]
                or scope["group"] != "v2_production_improvement"):
            raise ValueError("sampler scope differs from prepared pilot")
        if sha(scope["checkpoint_stats"]) != sha(plan["stats"]):
            raise ValueError("sampler checkpoint statistics differ from the super-observation transform")
        if set(scope["withheld_station_ids"]) != set(Path(plan["holdout"]).read_text().splitlines()):
            raise ValueError("withheld stations changed")
        write_json(receipt, {"plan_sha256": sha(Path(args.out_dir)/"pilot_plan.json"),
                            "files": [record(report), record(arrays)]})


def score_members(members, observed):
    """Physical-space scores on an already common, finite sample (member,N)."""
    if not observed.size:
        return {"n": 0}
    count = members.shape[0]; mean = members.mean(axis=0); diff = mean-observed
    ordered = np.sort(members, axis=0)
    fair = np.mean(np.abs(members-observed), axis=0)
    fair -= ((2*np.arange(1, count+1)-count-1)[:, None]*ordered).sum(axis=0)/(count*(count-1))
    lo, hi = np.quantile(members, [.05, .95], axis=0)
    result = {"n": int(observed.size), "observed_mean_mm": float(observed.mean()),
              "bias_mm": float(diff.mean()), "mae_mm": float(np.abs(diff).mean()),
              "rmse_mm": float(np.sqrt(np.mean(diff**2))), "fair_crps_mm": float(fair.mean()),
              "interval90_coverage": float(((observed >= lo) & (observed <= hi)).mean()),
              "rms_spread_mm": float(np.sqrt(members.var(axis=0, ddof=1).mean())),
              "mean_minus_median_mm": float((mean-np.median(members, axis=0)).mean())}
    result["correlation"] = float(np.corrcoef(mean, observed)[0, 1]) if mean.std() > 0 and observed.std() > 0 else None
    for threshold in (1, 50, 100):
        event = observed >= threshold; prediction = mean >= threshold
        hits, misses, false = int((event & prediction).sum()), int((event & ~prediction).sum()), int((~event & prediction).sum())
        probability = (members >= threshold).mean(axis=0)
        prefix = f"rain{threshold}_"
        result.update({prefix+"events": int(event.sum()), prefix+"hits": hits, prefix+"misses": misses,
                       prefix+"false_alarms": false, prefix+"brier": float(((probability-event)**2).mean()),
                       prefix+"pod": hits/(hits+misses) if hits+misses else None,
                       prefix+"csi": hits/(hits+misses+false) if hits+misses+false else None})
    return result


def summarize(args):
    plan = load_plan(args); rows, day_rows, pools = [], [], {}
    for w in plan["windows"]:
        folder = Path(args.out_dir)/"runs"/w["label"]
        receipt = json.loads((folder/"complete.json").read_text())
        if receipt["plan_sha256"] != sha(Path(args.out_dir)/"pilot_plan.json"):
            raise ValueError("run plan identity differs")
        verify(receipt["files"])
        with np.load(folder/"sweep.npz", allow_pickle=False) as z:
            if list(z["variant_names"]) != VARIANTS:
                raise ValueError("variant catalogue differs")
            expected = pd.date_range(w["start"], w["end"]).strftime("%Y-%m-%d").to_numpy()
            if not np.array_equal(z["times"], expected):
                raise ValueError("run dates incomplete or different")
            ix = z["eval_idx"]; observed = z["gauge_mm"][:, ix]
            arrays = {name: z["station_"+name][:, :, ix] for name in VARIANTS}
            common = np.isfinite(observed)
            for values in arrays.values():
                common &= np.isfinite(values).all(axis=1)
            if not common.any():
                raise ValueError("no common finite withheld samples: " + w["label"])
            ids = z["station_ids"][ix].astype(str)
            for network in ("ALL", "BMD", "BWDB"):
                selected = common if network == "ALL" else common & np.char.startswith(ids, network+"_")[None, :]
                for name, values in arrays.items():
                    members = np.moveaxis(values, 1, 0)[:, selected]; truth = observed[selected]
                    rows.append({"window": w["label"], "role": w["role"], "network": network, "variant": name, **score_members(members, truth)})
                    pools.setdefault((w["role"], network, name), []).append((members, truth))
                    for j, date in enumerate(z["times"]):
                        day_rows.append({"window": w["label"], "role": w["role"], "network": network, "date": str(date), "variant": name,
                                         **score_members(values[j][:, selected[j]], observed[j, selected[j]])})
    for (role, network, name), samples in pools.items():
        members = np.concatenate([s[0] for s in samples], axis=1)
        observed = np.concatenate([s[1] for s in samples])
        rows.append({"window": "POOLED", "role": role, "network": network, "variant": name, **score_members(members, observed)})
    out = Path(args.out_dir)/"summary"; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out/"withheld_scores.csv", index=False)
    pd.DataFrame(day_rows).to_csv(out/"paired_day_scores.csv", index=False)
    lines = ["# SURMA improvement pilot", "", "Sensitivity experiment; no automatic promotion to production.", "",
             "| Role | Network | Variant | N | Bias | RMSE | Fair CRPS | 90% field coverage |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        if r["window"] == "POOLED" and r["n"]:
            lines.append(f"| {r['role']} | {r['network']} | {r['variant']} | {r['n']} | {r['bias_mm']:.3f} | {r['rmse_mm']:.3f} | {r['fair_crps_mm']:.3f} | {r['interval90_coverage']:.3f} |")
    lines += ["", "Choose using tune rows; assess the frozen choice on test rows. Historical rows are in-training-year sensitivity.",
              "", "Heavy-rain counts, misses, false alarms and Brier scores are in withheld_scores.csv. Few extremes mean unresolved tail skill.",
              "", "Monthly/annual climate consistency, spatial extremes and hydrological testing remain release requirements.",
              "", *["- " + note for note in NOTES], "", f"QC status: {plan['qc_review']}; masked station-days: {plan['excluded_station_days']}."]
    (out/"comparison.md").write_text("\n".join(lines) + "\n")
    write_json(out/"summary.json", {"status": "complete_pilot", "plan_sha256": sha(Path(args.out_dir)/"pilot_plan.json"),
               "windows": len(plan["windows"]), "notes": NOTES})
    print("\n".join(lines[:len([r for r in rows if r['window'] == 'POOLED' and r['n']])+6]))
    print("[summary] " + str(out))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["qc", "prepare", "run", "summarize", "plan"])
    p.add_argument("--root", default="data/processed/brishti05_production_2001_2024")
    p.add_argument("--out-dir", default="data/processed/surma_improvement_pilot")
    p.add_argument("--windows", default="configs/surma_improvement_windows.json")
    p.add_argument("--ckpt", default="runs/prior_h100_cpc_v2/best.pt")
    p.add_argument("--stats", default="data/processed/stats_cpc_v2.json")
    p.add_argument("--config", default="configs/da.yaml")
    p.add_argument("--exclusions", help="reviewed station_id,start,end,reason CSV; masks only the pilot copy")
    p.add_argument("--members", type=int, default=30)
    p.add_argument("--seed", type=int, default=202205)
    p.add_argument("--min-coverage", type=float, default=.8)
    p.add_argument("--withhold", type=float, default=.1)
    p.add_argument("--buffer-km", type=float, default=20)
    p.add_argument("--neighbor-km", type=float, default=50)
    p.add_argument("--task", type=int, help="run one window by zero-based array index")
    args = p.parse_args()
    if args.members < 2 or not 0 < args.min_coverage <= 1 or not 0 < args.withhold < .5 or args.buffer_km < 0 or args.neighbor_km <= 0:
        p.error("invalid member count, coverage, holdout fraction or distances")
    if args.stage == "plan":
        cases = windows(args.windows)
        print(f"{len(cases)} windows; {sum(w['days'] for w in cases)} days; {args.members} members; 5 analysis arms + background")
        for i, w in enumerate(cases):
            print(f"{i:2d} {w['role']:10s} {w['start']}..{w['end']} {w['label']}")
    else:
        globals()[args.stage](args)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as exc:
        print("[improvement] failed: " + str(exc), file=sys.stderr)
        sys.exit(1)
