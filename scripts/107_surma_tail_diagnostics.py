#!/usr/bin/env python3
"""Diagnose saved rainfall tails and prepare a separate gauge-likelihood pilot.

No automatic station corrections. Completed archives can be read after code
updates; their plan, input data and completed-result hashes still must match.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_improvement_helpers", ROOT/"scripts/106_surma_improvement.py")
I = importlib.util.module_from_spec(spec); spec.loader.exec_module(I)
GROUP = "v2_tail_improvement"
VARIANTS = ["background", I.BASELINE, "tail_gauge_huber5", "tail_gauge_w125", "tail_gauge_w075"]
BINS = [("all", 0, np.inf), ("dry_lt1", 0, 1), ("rain1_10", 1, 10),
        ("rain10_50", 10, 50), ("rain50_100", 50, 100), ("rain_ge100", 100, np.inf)]
NOTES = [
    "Observations define rainfall bins; every variant uses identical finite station-days in each bin.",
    "An event in any member is a possibility diagnostic, not a deterministic detection or calibrated forecast.",
    "Field interval misses include point/grid and gauge uncertainty; probabilities require independent calibration checks.",
    "Aggregation diagnostics concern assimilated original gauges; they are not independent skill scores.",
    "QC flags are candidates only. Exclusions require original-source review and an explicit reason.",
    "Previously inspected test dates are now retest/development evidence, not a fresh blind benchmark.",
    "No automatic winner or production rerun. Assess dry-day false alarms as well as heavy-rain improvements.",
]


def load_archive(folder, strict_code=False):
    path = Path(folder)/"pilot_plan.json"
    plan = json.loads(path.read_text())
    # Updating diagnostic/sampler code does not invalidate already completed
    # results. Do not ignore data changes or reuse outputs to sample new code.
    code_changed = []
    for item in plan["shared_files"]:
        if Path(item["path"]).suffix == ".py" and not strict_code:
            if not Path(item["path"]).exists() or I.sha(item["path"]) != item["sha256"]:
                code_changed.append(item["path"])
        else:
            I.verify([item])
    I.verify(plan["source_files"])
    for w in plan["windows"]:
        I.verify(w["files"])
    return plan, code_changed


def window_arrays(folder, plan, w):
    run_dir = Path(folder)/"runs"/w["label"]
    receipt = json.loads((run_dir/"complete.json").read_text())
    if receipt["plan_sha256"] != I.sha(Path(folder)/"pilot_plan.json"):
        raise ValueError("completed result belongs to a different plan: " + w["label"])
    I.verify(receipt["files"])
    with np.load(run_dir/"sweep.npz", allow_pickle=False) as z:
        names = z["variant_names"].astype(str).tolist()
        if names != plan["variants"]:
            raise ValueError("variant catalogue differs")
        times = pd.date_range(w["start"], w["end"]).strftime("%Y-%m-%d").to_numpy()
        if not np.array_equal(z["times"], times):
            raise ValueError("saved dates differ or are incomplete")
        ix = z["eval_idx"]
        if set(z["station_ids"][ix].astype(str)) != held_ids(plan):
            raise ValueError("saved withheld station identities differ")
        observed = z["gauge_mm"][:, ix]
        arrays = {name: z["station_"+name][:, :, ix] for name in names}
        common = np.isfinite(observed)
        for values in arrays.values():
            if values.shape[:2] != (len(times), plan["members"]):
                raise ValueError("saved member dimensions differ")
            common &= np.isfinite(values).all(axis=1)
        if not common.any():
            raise ValueError("no common finite withheld observations")
        return times, z["station_ids"][ix].astype(str), observed, arrays, common


def conditional_scores(members, observed):
    result = I.score_members(members, observed)
    if not observed.size:
        return result
    lo, hi = np.quantile(members, [.05, .95], axis=0)
    result.update(interval90_width_mm=float((hi-lo).mean()),
                  below_interval_fraction=float((observed < lo).mean()),
                  above_interval_fraction=float((observed > hi).mean()),
                  member_max_mean_mm=float(members.max(axis=0).mean()),
                  member_q95_mean_mm=float(hi.mean()))
    for threshold in (50, 100):
        event = observed >= threshold
        probability = (members >= threshold).mean(axis=0)
        result[f"rain{threshold}_any_member_fraction"] = float((probability > 0).mean())
        result[f"rain{threshold}_probability_on_events"] = float(probability[event].mean()) if event.any() else None
        result[f"rain{threshold}_probability_on_nonevents"] = float(probability[~event].mean()) if (~event).any() else None
    return result


def original_path(w):
    matches = [f["path"] for f in w["files"] if Path(f["path"]).name == "original_qc.csv"]
    if len(matches) != 1:
        raise ValueError("expected one original_qc.csv for " + w["label"])
    return Path(matches[0])


def aggregation_rows(original, superobs, held):
    """Match by actual SOB identifiers/cells, never nearest unrelated stations."""
    frame = original[~original.station_id.isin(held)].copy()
    sites = frame.groupby("station_id")[["lat", "lon"]].first()
    key = I.cells(sites)
    all_ids = set(superobs.station_id.astype(str))
    mapping = {}
    for sid in sites.index:
        target = sid if sid in all_ids else "SOB_"+str(int(key.loc[sid]))
        if target not in all_ids:
            raise ValueError("cannot map assimilated original station to an actual super-observation: " + sid)
        mapping[sid] = target
    frame["assimilated_id"] = frame.station_id.map(mapping)
    counts = frame.groupby(["assimilated_id", "date"]).precip_mm.count().rename("reporting_cell_members")
    target = superobs[["station_id", "date", "precip_mm"]].rename(
        columns={"station_id": "assimilated_id", "precip_mm": "assimilated_mm"})
    result = frame.merge(target, on=["assimilated_id", "date"], how="left", validate="many_to_one")
    result = result.join(counts, on=["assimilated_id", "date"])
    result = result[result.precip_mm >= 50].copy()
    result["original_minus_assimilated_mm"] = result.precip_mm-result.assimilated_mm
    result["assimilated_fraction_of_original"] = result.assimilated_mm/result.precip_mm
    result["aggregated"] = result.assimilated_id.str.startswith("SOB_")
    return result


def shortlist(folder, plan, out):
    path = Path(folder)/"qc"/"review_flags.csv"
    # A next-round archive points back to the original QC candidate report.
    if not path.exists() and plan.get("parent_pilot"):
        path = Path(plan["parent_pilot"])/"qc"/"review_flags.csv"
    if not path.exists():
        return {"status": "qc_report_not_found", "path": str(path)}
    flags = pd.read_csv(path, dtype={"station_id": str})
    rows = []
    for w in plan["windows"]:
        original = pd.read_csv(original_path(w), usecols=["station_id"], dtype=str)
        for flag in flags.to_dict("records"):
            if flag["station_id"] not in set(original.station_id):
                continue
            day = flag.get("date")
            month = flag.get("month")
            if pd.notna(day):
                first = last = pd.Timestamp(day)
            elif pd.notna(month):
                period = pd.Period(month, "M"); first, last = period.start_time, period.end_time.normalize()
            else:
                continue
            start, end = max(first, pd.Timestamp(w["start"])), min(last, pd.Timestamp(w["end"]))
            if start <= end:
                rows.append({**flag, "window": w["label"], "window_role": w["role"],
                             "review_start": str(start.date()), "review_end": str(end.date())})
    table = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["station_id", "window", "review_start", "review_end"])
    table.to_csv(out/"review_flags_for_pilot.csv", index=False)
    return {"status": "review_candidates_only", "flags": len(table), "path": str(path), "sha256": I.sha(path)}


def diagnose(args, current=False):
    folder = Path(args.out_dir) if current else Path(args.pilot)
    out = Path(args.out_dir)/("summary" if current else "diagnostics_before")
    out.mkdir(parents=True, exist_ok=True)
    plan, changed_code = load_archive(folder)
    held = set(Path(plan["holdout"]).read_text().splitlines())
    scores, events, aggregations, pools = [], [], [], {}
    for w in plan["windows"]:
        print("[diagnose] " + w["label"], flush=True)
        times, ids, observed, arrays, common = window_arrays(folder, plan, w)
        for network in ("ALL", "BMD", "BWDB"):
            mask = common if network == "ALL" else common & np.char.startswith(ids, network+"_")[None, :]
            for label, lower, upper in BINS:
                selected = mask & (observed >= lower) & (observed < upper)
                for name, values in arrays.items():
                    members = np.moveaxis(values, 1, 0)[:, selected]; truth = observed[selected]
                    base = {"window": w["label"], "role": w["role"], "network": network, "rainfall_bin": label, "variant": name}
                    scores.append({**base, **conditional_scores(members, truth)})
                    pools.setdefault((w["role"], network, label, name), []).append((members, truth))
        for name, values in arrays.items():
            lo, hi = np.quantile(values, [.05, .95], axis=1)
            mean = values.mean(axis=1); maximum = values.max(axis=1)
            for day, site in np.argwhere(common & (observed >= 50)):
                events.append({"window": w["label"], "role": w["role"], "date": str(times[day]),
                    "station_id": ids[site], "network": ids[site].split("_")[0], "variant": name,
                    "observed_mm": float(observed[day, site]), "ensemble_mean_mm": float(mean[day, site]),
                    "q05_mm": float(lo[day, site]), "q95_mm": float(hi[day, site]),
                    "member_max_mm": float(maximum[day, site]),
                    "prob50": float((values[day, :, site] >= 50).mean()),
                    "prob100": float((values[day, :, site] >= 100).mean())})
        original = I.read_gauges([original_path(w)])
        superobs = I.read_gauges([w["stations"]])
        original = original[original.date.between(w["start"], w["end"])]
        superobs = superobs[superobs.date.between(w["start"], w["end"])]
        aggregation = aggregation_rows(original, superobs, held)
        aggregation["window"], aggregation["role"] = w["label"], w["role"]
        aggregations.append(aggregation)
    for (role, network, label, name), samples in pools.items():
        members = np.concatenate([m for m, y in samples], axis=1)
        observed = np.concatenate([y for m, y in samples])
        scores.append({"window": "POOLED", "role": role, "network": network, "rainfall_bin": label,
                       "variant": name, **conditional_scores(members, observed)})
    table = pd.DataFrame(scores)
    table.to_csv(out/"intensity_scores.csv", index=False)
    pd.DataFrame(events, columns=None if events else ["station_id", "date", "variant", "observed_mm"]).to_csv(out/"heavy_events.csv", index=False)
    pd.concat(aggregations, ignore_index=True).to_csv(out/"aggregation_extremes.csv", index=False)
    qc = shortlist(folder, plan, out)
    pooled = table[(table.window == "POOLED") & (table.rainfall_bin == "all") & (table.n > 0)]
    lines = ["# Rainfall-tail diagnostics", "", "Descriptive sensitivity; no automatic production change.", "",
             "| Role | Network | Variant | N | Bias | RMSE | CRPS | Field coverage |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in pooled.to_dict("records"):
        lines.append(f"| {r['role']} | {r['network']} | {r['variant']} | {r['n']} | {r['bias_mm']:.3f} | {r['rmse_mm']:.3f} | {r['fair_crps_mm']:.3f} | {r['interval90_coverage']:.3f} |")
    lines += ["", "## Conditional baseline diagnostics", "",
              "| Role | Observed rainfall bin | N | Bias | Field coverage | Below interval | Above interval | Mean member maximum |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    conditional = table[(table.window == "POOLED") & (table.network == "ALL") &
                        (table.variant == I.BASELINE) & (table.n > 0) &
                        table.rainfall_bin.isin(["dry_lt1", "rain50_100", "rain_ge100"])]
    for r in conditional.to_dict("records"):
        lines.append(f"| {r['role']} | {r['rainfall_bin']} | {r['n']} | {r['bias_mm']:.3f} | {r['interval90_coverage']:.3f} | {r['below_interval_fraction']:.3f} | {r['above_interval_fraction']:.3f} | {r['member_max_mean_mm']:.3f} |")
    lines += ["", "Read intensity_scores.csv for observed-intensity bias, interval width and lower/upper misses.",
              "Read heavy_events.csv for individual withheld events and member maxima/probabilities.",
              "Read aggregation_extremes.csv for original assimilated gauges versus actual super-observations.",
              "Read review_flags_for_pilot.csv for QC candidates affecting these inputs; no exclusions applied here.",
              "", f"Plan QC status: {plan.get('qc_review', 'unknown')}; masked station-days: {plan.get('excluded_station_days', 'unknown')}.",
              "", *["- "+note for note in NOTES]]
    (out/"comparison.md").write_text("\n".join(lines) + "\n")
    I.write_json(out/"diagnostics.json", {"status": "complete_diagnostics", "plan_sha256": I.sha(folder/"pilot_plan.json"),
        "reader": I.record(__file__), "historical_code_changes_read_only": changed_code, "qc_shortlist": qc, "notes": NOTES})
    print("[diagnostics] " + str(out/"comparison.md"))


def prepare(args):
    parent = Path(args.pilot).resolve(); out = Path(args.out_dir)
    if out.resolve() == parent or (out/"pilot_plan.json").exists():
        raise ValueError("choose a fresh output directory; never overwrite the original pilot")
    plan, changed_code = load_archive(parent)
    frames = I.read_gauges([original_path(w) for w in plan["windows"]])
    frame, masked = I.apply_exclusions(frames, args.exclusions)
    held = set(Path(plan["holdout"]).read_text().splitlines())
    out.mkdir(parents=True, exist_ok=True)
    holdout = out/"holdout_ids.txt"; holdout.write_text("\n".join(sorted(held)) + "\n")
    cases = []
    for previous in plan["windows"]:
        w = {k: previous[k] for k in ("label", "role", "start", "end", "quarter", "days")}
        if w["role"] == "test": w["role"] = "retest"
        selected = frame[frame.date.between(w["start"], w["end"])].copy()
        held_counts = selected[selected.station_id.isin(held)].groupby("station_id").precip_mm.count().reindex(sorted(held), fill_value=0)
        if (held_counts < plan["min_coverage"]*w["days"]).any():
            raise ValueError("reviewed exclusions leave insufficient withheld coverage; design a new split rather than silently changing it: " + w["label"])
        folder = out/"prepared"/w["label"]; folder.mkdir(parents=True, exist_ok=True)
        original = folder/"original_qc.csv"; selected.to_csv(original, index=False, date_format="%Y-%m-%d")
        table, budget = folder/"superob.csv", folder/"superob.json"
        I.run_script("87_superob_dense_gauges.py", "--stations", original, "--stats", plan["stats"],
            "--holdout-ids", holdout, "--cell-deg", .25, "--protect-withheld-km", plan["buffer_km"],
            "--fail-under-km", plan["buffer_km"], "--out", table, "--report", budget)
        report = json.loads(budget.read_text())["recommended_representativeness"]
        if not report or report["superob_implied_representativeness"] is None:
            raise ValueError("no measured error budget after reviewed exclusions")
        ids = pd.read_csv(table, usecols=["station_id"]).station_id.astype(str)
        if not (ids.str.startswith("BWDB_") & ~ids.isin(held)).any():
            raise ValueError("no retained unmerged BWDB gauges; frozen BWDB R x4 arm would be inactive")
        w.update(stations=str(table.resolve()), representation=report["superob_implied_representativeness"],
                 imerg=previous["imerg"], files=[I.record(original), I.record(table), I.record(budget)])
        cases.append(w)
    inherited = {k: plan[k] for k in ("checkpoint", "config", "stats", "predictors", "fill_known_cpc_gaps",
                 "members", "seed", "min_coverage", "held_stations", "retained_stations", "buffer_km")}
    shared = [I.record(path) for path in (holdout, parent/"pilot_plan.json", plan["checkpoint"], plan["stats"], plan["config"],
        __file__, ROOT/"scripts/106_surma_improvement.py", ROOT/"scripts/28_simultaneous_method_sweep.py",
        ROOT/"scripts/87_superob_dense_gauges.py", ROOT/"src/bdhires/da/sampler.py", ROOT/"src/bdhires/da/guidance.py")]
    if args.exclusions: shared.append(I.record(args.exclusions))
    I.write_json(out/"pilot_plan.json", {**inherited, "status": "prepared_research_pilot", "group": GROUP,
        "windows": cases, "variants": VARIANTS, "parent_pilot": str(parent), "holdout": str(holdout.resolve()),
        "shared_files": shared, "source_files": plan["source_files"],
        "excluded_station_days": plan.get("excluded_station_days", 0)+masked,
        "qc_review": "additional reviewed exclusion file supplied" if args.exclusions else "inherited QC; no additional reviewed exclusions",
        "notes": NOTES, "parent_code_changes_read_only": changed_code})
    print(f"[prepared] {len(cases)} windows; {len(held)} unchanged withheld stations; {masked} additional masked days; {out}")


def run(args, group=GROUP, variants=VARIANTS):
    out = Path(args.out_dir); plan, changed = load_archive(out, strict_code=True)
    if plan.get("group") != group or plan["variants"] != variants:
        raise ValueError("not a prepared tail pilot")
    if args.task is not None and not 0 <= args.task < len(plan["windows"]):
        raise ValueError("task outside window range")
    cases = plan["windows"] if args.task is None else [plan["windows"][args.task]]
    I.run_script("51_check_sqrt_da_gradient.py")
    tail_gradient_check()
    for w in cases:
        folder = out/"runs"/w["label"]; folder.mkdir(parents=True, exist_ok=True)
        report, arrays, receipt = folder/"sweep.json", folder/"sweep.npz", folder/"complete.json"
        if receipt.exists():
            window_arrays(out, plan, w)
            print("[reuse] " + w["label"], flush=True); continue
        if report.exists() or arrays.exists():
            raise ValueError("partial outputs; inspect/move aside before retry: " + str(folder))
        command = ["--config", plan["config"], "--ckpt", plan["checkpoint"], "--data-zarr", plan["predictors"],
            "--stations", w["stations"], "--imerg", w["imerg"], "--start", w["start"], "--end", w["end"],
            "--members", plan["members"], "--seed", plan["seed"], "--min-coverage", plan["min_coverage"],
            "--holdout-station-ids-file", plan["holdout"], "--background-day-offset", -1, "--group", group,
            "--set", "observations.imerg.factor=8", "--set", "observations.imerg.error_corr_cells=0.75",
            "--set", "sampler.noise_scale=0.0", "--set", "sampler.heun=true",
            "--set", f"observations.gauges.representativeness={w['representation']}", "--out", arrays, "--report", report]
        if plan["fill_known_cpc_gaps"]: command.append("--fill-known-cpc-gaps")
        I.run_script("28_simultaneous_method_sweep.py", *command)
        scope = json.loads(report.read_text())["scope"]
        if (scope["start"] != w["start"] or scope["end"] != w["end"] or scope["members"] != plan["members"]
                or scope["group"] != group or scope["assimilate_all_stations"]
                or set(scope["withheld_station_ids"]) != held_ids(plan)
                or I.sha(scope["checkpoint_stats"]) != I.sha(plan["stats"])):
            raise ValueError("sampler scope differs from frozen tail plan")
        I.write_json(receipt, {"plan_sha256": I.sha(out/"pilot_plan.json"), "files": [I.record(report), I.record(arrays)]})


def held_ids(plan):
    return set(Path(plan["holdout"]).read_text().splitlines())


def tail_gradient_check():
    """On-node autograd check of the new per-stream robust likelihood."""
    import torch
    from bdhires.da.guidance import GuidanceConfig, guidance_grad
    from bdhires.da.observation import CompositeObsOperator, PhysicalBilinearObsOperator, PhysicalBlockAverageObsOperator
    from bdhires.grids import Grid
    from bdhires.models.flow import RectifiedFlow
    from bdhires.transforms import PrecipTransform
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid = Grid("tail_gradient", 90., 23., 4, 4, .05)
    transform = PrecipTransform(kind="sqrt", mu=.4, sd=1.3)
    mask = np.ones((4, 4), dtype=np.float32)
    gauge = PhysicalBilinearObsOperator(grid, np.array([grid.lat[1]]), np.array([grid.lon[1]]), transform, valid=mask)
    satellite = PhysicalBlockAverageObsOperator(2, transform, valid=mask)
    operator = CompositeObsOperator([gauge, satellite], component_spread_cells=[0., 0.]).to(device)
    x = torch.full((2, 1, 4, 4), .5, device=device)
    t = torch.full((2,), .5, device=device)
    y = transform.forward(torch.full((2, 1, 5), 80., device=device))
    variance = torch.full((5,), .25, device=device)
    flow = RectifiedFlow()
    model = lambda state, time, cond: torch.zeros_like(state)
    gradients = []
    for delta in (3., torch.full((5,), 3., device=device), torch.tensor([5., 3., 3., 3., 3.], device=device)):
        cfg = GuidanceConfig(gamma=.001, huber_delta=delta)
        _, gradient = guidance_grad(x, t, model, flow, None, operator, y, variance, cfg)
        if not torch.isfinite(gradient).all().item():
            raise FloatingPointError("per-stream Huber produced non-finite gradients")
        gradients.append(gradient)
    if not torch.allclose(gradients[0], gradients[1], atol=1e-6, rtol=1e-6):
        raise ValueError("scalar and uniform-vector Huber do not agree")
    if torch.allclose(gradients[1], gradients[2], atol=1e-6, rtol=1e-6):
        raise ValueError("gauge Huber relaxation is inactive")
    print(f"[preflight] per-stream Huber: finite, scalar-equivalent control, active gauge relaxation on {device}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["diagnose", "prepare", "run", "summarize"])
    parser.add_argument("--pilot", default="data/processed/surma_improvement_pilot")
    parser.add_argument("--out-dir", default="data/processed/surma_tail_pilot")
    parser.add_argument("--exclusions", help="new reviewed station_id,start,end,reason CSV; do not edit the old plan's file")
    parser.add_argument("--task", type=int)
    args = parser.parse_args()
    if args.stage in {"diagnose", "summarize"}: diagnose(args, current=args.stage == "summarize")
    else: globals()[args.stage](args)


if __name__ == "__main__":
    try: main()
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as exc:
        print("[tail] failed: " + str(exc), file=sys.stderr); sys.exit(1)
