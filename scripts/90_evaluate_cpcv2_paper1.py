#!/usr/bin/env python3
"""Audit/evaluate saved CPCv2 test-year ensembles for the model paper.

No training or rainfall generation is performed. The default is the recorded
BMD+BWDB super-observation winner; --profile bmd-reference evaluates the older
five-fold BMD-only reference separately. Missing folds stop evaluation rather
than silently reducing the verification sample. --audit-only needs only Python's
standard library and reports missing inputs without producing skill scores.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "configs/paper1_cpcv2_final.json"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--profile", choices=("superob-final", "bmd-reference"),
                        default="superob-final")
    parser.add_argument("--root", type=Path, help="override the profile's archive root")
    parser.add_argument("--periods", nargs="+", help="explicit subset of archived periods")
    parser.add_argument("--comparators", nargs="*", default=[],
                        help="additional method keys present in every fold")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "output/paper1_cpcv2")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--with-gridded", action="store_true",
                        help="also run existing spectra/maps/subgrid evaluator on saved Zarrs")
    parser.add_argument("--cpc-source-zarr", type=Path)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--block-days", type=int, nargs="+", default=[3, 7])
    parser.add_argument("--seed", type=int, default=20261001)
    return parser.parse_args(argv)


def read_contract(args):
    contract = json.loads(args.contract.read_text())
    profile = contract["profiles"][args.profile]
    periods = args.periods or list(contract["periods"])
    if len(set(periods)) != len(periods) or any(p not in contract["periods"] for p in periods):
        raise ValueError("periods must be unique names from the frozen contract")
    if args.bootstrap < 1 or any(n < 1 for n in args.block_days):
        raise ValueError("bootstrap count and block widths must be positive")
    return contract, profile, periods, args.root or ROOT / profile["root"]


def inventory(root, periods, profile):
    required, optional = [], []
    for period in periods:
        if profile["layout"] == "five-fold":
            bases = [root / "cv" / period / f"fold{i}" for i in range(5)]
        else:
            bases = [root / "evaluation" / period]
        for base in bases:
            required.extend([base.with_suffix(".npz"), base.with_suffix(".json")])
        if profile.get("superob_cell_degrees") is not None:
            folder = root / "stations" / period
            required.extend([folder / "preparation_manifest.json", folder / "holdout/fold0.txt",
                             folder / "superob_eval_0.25.json"])
        optional.append(root / "gridded" / f"{period}.zarr")
    return {
        "root": str(root.resolve()),
        "required": [{"path": str(p), "present": p.is_file() and p.stat().st_size > 0}
                     for p in required],
        "production_stores": [{"path": str(p), "present": p.is_dir()} for p in optional],
    }


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance(path, hash_content=False):
    entry = {"path": str(path.resolve()), "bytes": path.stat().st_size,
             "mtime_ns": path.stat().st_mtime_ns}
    if hash_content:
        entry["sha256"] = sha256(path)
    return entry


def scoring_module():
    # Reuse the project's fair CRPS, calibration metrics and seasonal-gap-safe CI.
    spec = importlib.util.spec_from_file_location(
        "_paper1_scores", ROOT / "scripts/54_summarize_v2_confirmatory.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_fold(dump, report, contract, profile, period, fold, methods):
    import numpy as np

    scope = report["scope"]
    start, end = contract["periods"][period]
    expected = np.arange(np.datetime64(start, "D"),
                         np.datetime64(end, "D") + np.timedelta64(1, "D"))
    dates = dump["times"].astype("datetime64[D]")
    if not np.array_equal(dates, expected) or (scope["start"], scope["end"]) != (start, end):
        raise ValueError(f"{period}/fold{fold}: incomplete or mismatched dates")
    if any(int(str(day)[:4]) < contract["test_years"][0]
           or int(str(day)[:4]) > contract["test_years"][1] for day in dates):
        raise ValueError("evaluation dates must be in the configured test years")
    checkpoint_parts = Path(scope["checkpoint"]).parts
    if tuple(checkpoint_parts[-2:]) != ("prior_h100_cpc_v2", "best.pt"):
        raise ValueError(f"unexpected checkpoint: {scope['checkpoint']}")
    if scope.get("assimilate_all_stations") is not False:
        raise ValueError("independent scoring requires an explicitly held-out run")
    for key, expected_value in (("members", contract["members"]),
                                ("background_day_offset", contract["background_day_offset"]),
                                ("holdout_folds", profile["folds"]), ("holdout_fold", fold),
                                ("n_days", len(expected))):
        if scope.get(key) != expected_value:
            raise ValueError(f"{period}/fold{fold}: {key} differs from contract")
    if not np.array_equal(dump["model_times"].astype("datetime64[D]"),
                          dates + np.timedelta64(contract["background_day_offset"], "D")):
        raise ValueError("background records do not preserve the D-1 alignment")
    available = dump["variant_names"].astype(str).tolist()
    if any(name not in available for name in methods):
        raise ValueError(f"missing requested methods; available={available}")
    spec = report["variants"][profile["method"]]["spec"]
    for key, value in profile["expected_spec"].items():
        if spec.get(key) != value:
            raise ValueError(f"{period}/fold{fold}: method setting {key} differs")
    # All other method settings must also agree across files (checked by loader).
    overrides = dict(item.split("=", 1) for item in scope.get("config_overrides", []))
    if float(overrides.get("observations.imerg.factor", -1)) != contract["imerg_factor"]:
        raise ValueError("scope must explicitly record the frozen S04 factor")
    if float(overrides.get("observations.imerg.error_corr_cells", -1)) != 0.75:
        raise ValueError("scope must explicitly record S04 correlation length 0.75")
    if scope.get("precip_transform", {}).get("kind") != "sqrt":
        raise ValueError("the CPCv2 rainfall transform must be square root")
    station_ids = dump["station_ids"].astype(str)
    if len(station_ids) != len(set(station_ids)):
        raise ValueError("duplicate station identifiers")
    eval_idx, assim_idx = dump["eval_idx"], dump["assim_idx"]
    for label, indices in (("eval_idx", eval_idx), ("assim_idx", assim_idx)):
        if indices.dtype.kind not in "iu" or indices.ndim != 1:
            raise ValueError(f"{label} must be a vector of integer indices")
        if len(indices) != len(set(indices)) or np.any(indices < 0) or np.any(indices >= len(station_ids)):
            raise ValueError(f"invalid {label}")
    if not len(eval_idx) or set(eval_idx) & set(assim_idx):
        raise ValueError("withheld indices are empty or overlap assimilated indices")
    if set(eval_idx) | set(assim_idx) != set(range(len(station_ids))):
        raise ValueError("station indices must be partitioned between assimilation and verification")
    withheld = station_ids[eval_idx]
    if any(station.startswith("SOB_") for station in withheld):
        raise ValueError("verify original held-out gauges, not synthetic super-observations")
    if set(scope.get("withheld_station_ids", [])) != set(withheld):
        raise ValueError("scope and NPZ withheld station identifiers differ")
    if dump["gauge_mm"].shape != (len(dates), len(station_ids)):
        raise ValueError("gauge array shape does not match dates and stations")
    for method in methods:
        if dump[f"station_{method}"].shape != (len(dates), contract["members"], len(station_ids)):
            raise ValueError(f"{method}: ensemble shape does not match contract")
    return dates, station_ids, eval_idx, spec


def load_samples(root, periods, contract, profile, methods):
    import numpy as np

    pieces = {key: [] for key in ("date", "station", "period", "truth")}
    ensembles = {name: [] for name in methods}
    sources, scopes, warnings = [], [], []
    reference, reference_spec = None, None
    coordinates = {}
    for period in periods:
        seen, eligible = [], None
        for fold in range(profile["folds"]):
            base = (root / "cv" / period / f"fold{fold}" if profile["layout"] == "five-fold"
                    else root / "evaluation" / period)
            report_path, dump_path = base.with_suffix(".json"), base.with_suffix(".npz")
            report = json.loads(report_path.read_text())
            with np.load(dump_path, allow_pickle=False) as dump:
                dates, ids, indices, method_spec = validate_fold(
                    dump, report, contract, profile, period, fold, methods)
                scope = report["scope"]
                signature = {k: scope.get(k) for k in
                             ("checkpoint", "checkpoint_data", "checkpoint_stats", "seed",
                              "precip_transform", "group", "analysis_sampler_n_steps",
                              "analysis_sampler_n_corrections", "analysis_sampler_heun")}
                signature["config_overrides"] = sorted(
                    item for item in scope.get("config_overrides", [])
                    if not item.startswith("observations.gauges.representativeness="))
                if reference is None:
                    reference, reference_spec = signature, method_spec
                elif signature != reference or method_spec != reference_spec:
                    raise ValueError(f"{base}: checkpoint/data/seed/sampler/method contract differs")
                if eligible is not None and set(ids) != eligible:
                    raise ValueError(f"{period}: fold station sets differ")
                eligible = set(ids)
                # Synthetic cluster centroids can change with the reporting network
                # between periods; require fixed coordinates for scored original gauges.
                for index in indices:
                    station = ids[index]
                    coordinate = (float(dump["station_lat"][index]), float(dump["station_lon"][index]))
                    if station in coordinates and coordinates[station] != coordinate:
                        raise ValueError(f"{station}: station coordinates change")
                    coordinates[station] = coordinate
                withheld = ids[indices]
                seen.extend(withheld.tolist())
                pieces["date"].append(np.repeat(dates, len(indices)))
                pieces["station"].append(np.tile(withheld, len(dates)))
                pieces["period"].append(np.full(len(dates) * len(indices), period))
                pieces["truth"].append(dump["gauge_mm"][:, indices].reshape(-1).astype(float))
                for method in methods:
                    values = dump[f"station_{method}"][:, :, indices]
                    ensembles[method].append(np.moveaxis(values, 1, 2).reshape(-1, values.shape[1]))
                if profile.get("superob_cell_degrees"):
                    folder = root / "stations" / period
                    declared = set((folder / "holdout/fold0.txt").read_text().split())
                    manifest_path = folder / "superob_eval_0.25.json"
                    manifest = json.loads(manifest_path.read_text())
                    if declared != set(withheld) or manifest["stations_held_out"] != len(withheld):
                        raise ValueError(f"{period}: super-observation holdout manifest differs")
                    if manifest["cell_deg"] != profile["superob_cell_degrees"]:
                        raise ValueError(f"{period}: super-observation cell size differs")
                    prep = folder / "preparation_manifest.json"
                    preparation = json.loads(prep.read_text())
                    if not preparation.get("analysis_selection"):
                        raise ValueError(f"{period}: missing original holdout geometry provenance")
                    sources.extend([provenance(manifest_path, True), provenance(prep, True),
                                    provenance(folder / "holdout/fold0.txt", True)])
                    # The archived implementation applies r4 by BWDB_ id, not to SOB_ clusters.
                    assimilated = ids[dump["assim_idx"]]
                    warnings.append(f"{period}: R x4 applies to {sum(s.startswith('BWDB_') for s in assimilated)} "
                                    "unmerged BWDB_ records; SOB_ clusters use the shared superob error budget.")
                    warnings.append(f"{period}: archived superob manifests do not record the stats file used "
                                    "for the error budget; verify that transform against checkpoint stats before release.")
                sources.extend([provenance(dump_path), provenance(report_path, True)])
                scopes.append({"period": period, "fold": fold, "scope": scope, "method_spec": method_spec})
        if len(seen) != len(set(seen)):
            raise ValueError(f"{period}: station withheld more than once")
        if profile["layout"] == "five-fold" and set(seen) != eligible:
            raise ValueError(f"{period}: five folds do not exhaustively withhold every station")
    data = {key: np.concatenate(values) for key, values in pieces.items()}
    data["members"] = {name: np.concatenate(values).astype(float) for name, values in ensembles.items()}
    keys = [f"{day}|{station}" for day, station in zip(data["date"], data["station"])]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate withheld station-day across input files")
    data["source"] = np.where(np.char.startswith(data["station"], "BWDB_"), "BWDB", "BMD")
    # Pin available weights/stats by content rather than calling a mutable
    # best.pt pathname a permanent scientific identity. No model is loaded.
    for key in ("checkpoint", "checkpoint_stats"):
        recorded = Path(reference[key])
        candidates = [recorded] if recorded.is_absolute() else [
            ROOT / recorded, root.parent.parent.parent / recorded]
        artifact = next((p for p in candidates if p.is_file()), None)
        if artifact is None:
            warnings.append(f"{key}: archived pathname checked; content hash pending because {recorded} is unavailable.")
        else:
            sources.append({"role": key, **provenance(artifact, True)})
    return data, sources, scopes, warnings


def score_samples(data, methods, profile, scorer, block_days, resamples, seed):
    import numpy as np

    selected = ((data["date"] >= np.datetime64(profile["selection_start"])) &
                (data["date"] <= np.datetime64(profile["selection_end"])))
    common = np.isfinite(data["truth"])
    for name in methods:
        common &= np.isfinite(data["members"][name]).all(axis=1)
    keep = common & ~selected
    if not keep.any():
        raise ValueError("no finite independent station-days remain after selection exclusion")
    groups = [("pooled", "all", keep)]
    groups += [("period", p, keep & (data["period"] == p)) for p in np.unique(data["period"])]
    groups += [("source", s, keep & (data["source"] == s)) for s in np.unique(data["source"])]
    groups += [("station", s, keep & (data["station"] == s)) for s in np.unique(data["station"])]
    bins = [0, 1, 10, 25, 50, 100, np.inf]
    groups += [("intensity", f"[{lo},{hi})", keep & (data["truth"] >= lo) & (data["truth"] < hi))
               for lo, hi in zip(bins[:-1], bins[1:])]
    rows = []
    for group, label, mask in groups:
        for method in methods:
            result = scorer.metrics(data["members"][method][mask], data["truth"][mask])
            rows.append({"scale": "daily", "group": group, "label": label,
                         "method": method, **result})
    paired = []
    dates = np.unique(data["date"][keep])
    for candidate in methods[1:]:
        delta = (scorer.fair_crps(data["members"]["background"], data["truth"]) -
                 scorer.fair_crps(data["members"][candidate], data["truth"]))
        by_day = np.asarray([delta[keep & (data["date"] == d)].mean() for d in dates])
        for width in block_days:
            result = scorer.segmented_block_bootstrap(dates, by_day, width, resamples, seed)
            paired.append({"candidate": candidate, "reference": "background", **result,
                           "weighting": "equal days; each day averages its withheld station differences"})
    return rows, paired, keep, {
        "total_station_days": len(common), "selection_station_days": int(selected.sum()),
        "nonfinite_station_days_outside_selection": int((~common & ~selected).sum()),
        "scored_station_days": int(keep.sum()), "scored_dates": len(dates),
        "first_scored_date": str(dates[0]), "last_scored_date": str(dates[-1]),
    }


def temporal_scores(data, methods, profile):
    """Deterministic calendar means; daily member ids are not temporal trajectories."""
    import calendar
    import numpy as np

    rows = []
    months = data["date"].astype("datetime64[M]")
    selection_months = set(np.arange(np.datetime64(profile["selection_start"], "M"),
                                    np.datetime64(profile["selection_end"], "M") + 1).astype(str))
    periods = [("monthly", str(m), months == m, calendar.monthrange(int(str(m)[:4]), int(str(m)[5:7]))[1])
               for m in np.unique(months) if str(m) not in selection_months]
    for year in np.unique(data["date"].astype("datetime64[Y]").astype(str)):
        expected = np.arange(np.datetime64(f"{year}-05-01"), np.datetime64(f"{year}-10-01"))
        mask = (data["date"] >= expected[0]) & (data["date"] <= expected[-1])
        overlap_selection = ((expected >= np.datetime64(profile["selection_start"])) &
                             (expected <= np.datetime64(profile["selection_end"]))).any()
        if not overlap_selection and np.array_equal(np.unique(data["date"][mask]), expected):
            periods.append(("may_sep", year, mask, len(expected)))
    for scale, label, mask, requested_days in periods:
        for station in np.unique(data["station"][mask]):
            choose = mask & (data["station"] == station)
            finite = np.isfinite(data["truth"][choose])
            for method in methods:
                finite &= np.isfinite(data["members"][method][choose]).all(axis=1)
            if finite.sum() < np.ceil(0.8 * requested_days):
                continue
            truth = data["truth"][choose][finite]
            for method in methods:
                daily_mean = data["members"][method][choose][finite].mean(axis=1)
                rows.append({"scale": scale, "period": label, "station": station, "method": method,
                             "n_days": int(finite.sum()), "requested_days": requested_days,
                             "observed_mean_mm_day": float(truth.mean()),
                             "predicted_mean_mm_day": float(daily_mean.mean()),
                             "mean_error_mm_day": float(daily_mean.mean() - truth.mean()),
                             "observed_daily_temporal_sd_mm": float(truth.std()),
                             "predicted_daily_temporal_sd_mm": float(daily_mean.std())})
    return rows


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def json_ready(value):
    import numpy as np

    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def write_report(out, contract, profile_name, profile, rows, paired, counts):
    pooled = [r for r in rows if r["group"] == "pooled"]
    lines = ["# SURMA-Flow CPCv2: model-paper test-year evaluation", "",
             f"Final learned model: {contract['model_name']} (CPCv2 U-Net rectified flow).",
             f"Assimilation profile: `{profile_name}` / `{profile['method']}`.",
             f"Verification layout: {profile['layout']}; {profile['folds']} fold(s) per period.",
             f"Independent daily dates exclude {profile['selection_start']} through {profile['selection_end']}.",
             f"Scored: {counts['scored_station_days']} station-days over {counts['scored_dates']} dates.", "",
             "| Method | Fair CRPS | RMSE | MAE | Bias | Correlation | Spread/RMSE | 90% coverage |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in pooled:
        values = [row.get(k) for k in ("crps", "rmse", "mae", "bias", "correlation", "spread_skill", "coverage_90")]
        lines.append("| " + row["method"] + " | " + " | ".join(
            f"{v:.3f}" if v is not None else "—" for v in values) + " |")
    lines += ["", "Positive paired CRPS gain favours the candidate; intervals resample day blocks without crossing gaps."]
    for result in paired:
        lines.append(f"- {result['candidate']}, {result['block_days']}-day blocks: "
                     f"gain {result['difference']:.3f}, 95% CI [{result['ci_low']:.3f}, {result['ci_high']:.3f}] mm/day.")
    lines += ["", "Monthly scores exclude selection months and require 80% of calendar days. "
              "Seasonal scores require complete May–September coverage and exclude seasons containing selection dates.",
              "Temporal tables evaluate deterministic means and daily variability. Daily member indices do not establish "
              "a coherent ensemble of monthly or seasonal totals, so aggregate probabilistic scores are omitted.",
              "All-station maps diagnose fit and structure. Primary skill comes from original withheld gauges.",
              "CHIRPS is the training-target family, CPC conditions the prior, and IMERG is assimilated. "
              "Their field comparisons describe agreement.",
              "The single-holdout profile samples a constrained 20% network; it does not provide exhaustive five-fold evidence.",
              "The generated archive has seasonal gaps and currently stops in June 2024; it does not establish 2025 coverage."]
    (out / "paper1_evaluation.md").write_text("\n".join(lines) + "\n")


def run_gridded(args, root, periods, contract, profile, out):
    # Require saved all-station products, never regenerate them here.
    stores = [root / "gridded" / f"{p}.zarr" for p in periods]
    for path in stores:
        if not path.is_dir():
            raise FileNotFoundError(path)
        attrs_path = path / ".zattrs"
        if not attrs_path.is_file():
            raise ValueError(f"{path}: expected a completed Zarr v2 product")
        attrs = json.loads(attrs_path.read_text())
        if not attrs.get("complete") or attrs.get("schema") != "bdhires.physical_ensemble.v1":
            raise ValueError(f"{path}: production store is incomplete or has unknown schema")
        scope = attrs.get("scope", {})
        if scope.get("checkpoint") != contract["checkpoint"] and tuple(
                Path(scope.get("checkpoint", "")).parts[-2:]) != ("prior_h100_cpc_v2", "best.pt"):
            raise ValueError(f"{path}: production checkpoint is not CPCv2")
        if scope.get("members") != contract["members"] or not scope.get("assimilate_all_stations"):
            raise ValueError(f"{path}: unexpected production ensemble/observation contract")
        if (scope.get("start"), scope.get("end")) != tuple(contract["periods"][path.stem]):
            raise ValueError(f"{path}: production dates differ from the requested period")
        for key, value in profile["expected_spec"].items():
            if attrs.get("method_specs", {}).get(profile["method"], {}).get(key) != value:
                raise ValueError(f"{path}: production method setting {key} differs")
    command = [sys.executable, str(ROOT / "scripts/55_evaluate_v2_gridded_archive.py"),
               "--zarr", *map(str, stores), "--cv-root", str(root), "--cv-layout", profile["layout"],
               "--out-dir", str(out / "gridded"), "--factor", str(contract["imerg_factor"]),
               "--selection-daily-start", profile["selection_start"],
               "--selection-daily-end", profile["selection_end"]]
    if args.cpc_source_zarr:
        command.extend(["--cpc-source-zarr", str(args.cpc_source_zarr)])
    subprocess.run(command, check=True, cwd=ROOT)


def main(argv=None):
    args = parse_args(argv)
    contract, profile, periods, root = read_contract(args)
    out = args.out_dir / args.profile
    out.mkdir(parents=True, exist_ok=True)
    audit = inventory(root, periods, profile)
    missing = [r["path"] for r in audit["required"] if not r["present"]]
    audit.update(profile=args.profile, method=profile["method"], periods=periods,
                 contract_sha256=sha256(args.contract),
                 status="missing_inputs" if missing else "inputs_present_not_yet_validated",
                 missing=missing)
    (out / "input_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(f"[paper1] {audit['status']}; audit: {out / 'input_audit.json'}")
    if args.audit_only:
        return 0
    if missing:
        print(f"[paper1] missing {len(missing)} required files; evaluation not run", file=sys.stderr)
        return 2
    methods = list(dict.fromkeys(["background", profile["method"], *args.comparators]))
    scorer = scoring_module()
    data, sources, scopes, warnings = load_samples(root, periods, contract, profile, methods)
    rows, paired, _, counts = score_samples(data, methods, profile, scorer,
                                              args.block_days, args.bootstrap, args.seed)
    temporal = temporal_scores(data, methods, profile)
    write_csv(out / "daily_withheld_scores.csv", rows)
    write_csv(out / "paired_crps_intervals.csv", paired)
    write_csv(out / "temporal_withheld_scores.csv", temporal)
    write_report(out, contract, args.profile, profile, rows, paired, counts)
    payload = dict(status="withheld_evaluation_complete", contract=contract, profile=args.profile,
                   periods=periods, counts=counts, provenance=sources, archived_contracts=scopes,
                   warnings=warnings, daily_scores=rows, paired_crps=paired,
                   temporal_scores=temporal,
                   gridded_status="pending" if args.with_gridded else "not_requested",
                   temporal_probabilistic_scores="not evaluated: daily members are not validated temporal trajectories")
    (out / "paper1_evaluation.json").write_text(
        json.dumps(json_ready(payload), indent=2, allow_nan=False) + "\n")
    if args.with_gridded:
        run_gridded(args, root, periods, contract, profile, out)
        payload["gridded_status"] = "complete"
        (out / "paper1_evaluation.json").write_text(
            json.dumps(json_ready(payload), indent=2, allow_nan=False) + "\n")
    print(f"[paper1] withheld evaluation complete: {out / 'paper1_evaluation.md'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"[paper1] validation failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
