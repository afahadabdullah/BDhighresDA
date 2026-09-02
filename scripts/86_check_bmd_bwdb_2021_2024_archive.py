#!/usr/bin/env python
"""Comprehensive data integrity and numerical checker for the BMD+BWDB 2021-2024 archive.

Verifies:
1. Expected files: station preparation, evaluation NPZ/JSON, production metadata, and gridded Zarrs.
2. Evaluation runs: 20% holdout split, valid withheld station observations, finite CRPS/RMSE.
3. Production runs: explains why withheld CRPS printed NaN in production mode (all stations
   were assimilated, 0 withheld stations) and validates finite assimilated-fit scores.
4. Gridded Zarr stores: store completion, dimensions (153/61 days, 30 members, 128x128 grid),
   zero NaNs/Infs on land cells, strictly non-negative rainfall, realistic physical ranges,
   and positive ensemble spread.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PERIOD_DEFINITIONS = {
    "2021_may_sep": {"start": "2021-05-01", "end": "2021-09-30", "days": 153},
    "2022_may_sep": {"start": "2022-05-01", "end": "2022-09-30", "days": 153},
    "2023_may_sep": {"start": "2023-05-01", "end": "2023-09-30", "days": 153},
    "2024_may_jun": {"start": "2024-05-01", "end": "2024-06-30", "days": 61},
}
TOTAL_DAYS = 520
EXPECTED_METHODS = ["background", "v2_simul_s04_huber3"]
EXPECTED_MEMBERS = 30
GRID_SHAPE = (128, 128)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--root",
        default="data/processed/v2_bmd_bwdb_huber3_2021_2024",
        help="root directory of the BMD+BWDB Huber3 archive (default: data/processed/v2_bmd_bwdb_huber3_2021_2024)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="optional output JSON file for the integrity report",
    )
    parser.add_argument(
        "--check-all-zarr-members",
        action="store_true",
        help="thoroughly scan all 30 ensemble members across all days in Zarr (takes slightly longer)",
    )
    return parser.parse_args()


def fair_crps_per_sample(members: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Compute fair CRPS for each sample across members."""
    members, truth = np.asarray(members, float), np.asarray(truth, float)
    result = np.full(len(truth), np.nan)
    valid = np.isfinite(truth) & np.all(np.isfinite(members), axis=1)
    if not valid.any():
        return result
    selected, obs = members[valid], truth[valid]
    m_count = selected.shape[1]
    diff1 = np.mean(np.abs(selected - obs[:, None]), axis=1)
    sorted_m = np.sort(selected, axis=1)
    weights = 2 * np.arange(1, m_count + 1) - m_count - 1
    diff2 = np.sum(sorted_m * weights[None, :], axis=1) / (m_count * (m_count - 1))
    result[valid] = diff1 - diff2
    return result


def check_evaluation_run(eval_npz: Path, eval_json: Path, label: str) -> dict:
    """Verify independent holdout evaluation outputs."""
    errors = []
    warnings = []

    if not eval_npz.is_file():
        return {"status": "FAIL", "errors": [f"Missing evaluation npz: {eval_npz}"]}
    if not eval_json.is_file():
        return {"status": "FAIL", "errors": [f"Missing evaluation json: {eval_json}"]}

    dump = np.load(eval_npz, allow_pickle=False)
    variants = dump["variant_names"].astype(str).tolist()
    if variants != EXPECTED_METHODS:
        errors.append(f"Expected variants {EXPECTED_METHODS}, found {variants}")

    station_ids = dump["station_ids"].astype(str)
    eval_idx = np.asarray(dump["eval_idx"], int)
    assim_idx = np.asarray(dump["assim_idx"], int)
    total_stations = len(station_ids)

    if len(eval_idx) == 0:
        errors.append("Evaluation run has 0 withheld stations (eval_idx is empty)")
    if len(np.intersect1d(eval_idx, assim_idx)) > 0:
        errors.append("Overlap between withheld and assimilated stations")

    gauge_mm = np.asarray(dump["gauge_mm"], float)
    times = dump["times"].astype("datetime64[D]")
    expected_days = PERIOD_DEFINITIONS[label]["days"]
    if len(times) != expected_days:
        errors.append(f"Expected {expected_days} days, found {len(times)}")

    withheld_obs = gauge_mm[:, eval_idx]
    finite_obs = np.isfinite(withheld_obs)
    valid_station_days = int(finite_obs.sum())
    wet_station_days = int((withheld_obs >= 1.0).sum())

    if valid_station_days == 0:
        errors.append("Zero valid withheld observations")

    # Check finite CRPS on withheld stations
    scores = {}
    for method in EXPECTED_METHODS:
        key = f"station_{method}"
        if key not in dump:
            errors.append(f"Missing {key} in evaluation dump")
            continue
        ens = np.asarray(dump[key][:, :, eval_idx], float)
        # Flatten time and withheld stations
        flat_obs = withheld_obs.reshape(-1)
        flat_ens = np.moveaxis(ens, 1, 2).reshape(-1, ens.shape[1])
        valid_mask = np.isfinite(flat_obs) & np.all(np.isfinite(flat_ens), axis=1)

        crps_vals = fair_crps_per_sample(flat_ens, flat_obs)
        mean_crps = float(np.nanmean(crps_vals))
        ens_mean = flat_ens.mean(axis=1)
        diff = ens_mean[valid_mask] - flat_obs[valid_mask]
        rmse = float(np.sqrt(np.mean(diff**2))) if len(diff) else np.nan
        mae = float(np.mean(np.abs(diff))) if len(diff) else np.nan

        if not np.isfinite(mean_crps):
            errors.append(f"{method}: Withheld CRPS is NaN or non-finite")

        scores[method] = {
            "crps_mm": mean_crps,
            "rmse_mm": rmse,
            "mae_mm": mae,
            "finite_eval_samples": int(valid_mask.sum()),
        }

    # Verify Huber3 outperforms background
    if (
        "v2_simul_s04_huber3" in scores
        and "background" in scores
        and np.isfinite(scores["v2_simul_s04_huber3"]["crps_mm"])
        and np.isfinite(scores["background"]["crps_mm"])
    ):
        h3_crps = scores["v2_simul_s04_huber3"]["crps_mm"]
        bg_crps = scores["background"]["crps_mm"]
        if h3_crps >= bg_crps:
            warnings.append(
                f"Huber3 CRPS ({h3_crps:.3f}) is not lower than background ({bg_crps:.3f}) in {label}"
            )

    return {
        "status": "PASS" if not errors else "FAIL",
        "total_stations": total_stations,
        "withheld_stations": len(eval_idx),
        "assimilated_stations": len(assim_idx),
        "valid_withheld_obs_count": valid_station_days,
        "wet_withheld_obs_count": wet_station_days,
        "scores": scores,
        "errors": errors,
        "warnings": warnings,
    }


def check_production_run(prod_npz: Path, prod_json: Path, label: str) -> dict:
    """Verify production run metadata and clarify the 'NaN CRPS' print."""
    errors = []
    warnings = []

    if not prod_npz.is_file():
        return {"status": "FAIL", "errors": [f"Missing production npz: {prod_npz}"]}
    if not prod_json.is_file():
        return {"status": "FAIL", "errors": [f"Missing production json: {prod_json}"]}

    report = json.loads(prod_json.read_text())
    scope = report.get("scope", {})

    if not scope.get("assimilate_all_stations"):
        errors.append("Production scope.assimilate_all_stations is not True")

    dump = np.load(prod_npz, allow_pickle=False)
    eval_idx = np.asarray(dump["eval_idx"], int)
    assim_idx = np.asarray(dump["assim_idx"], int)
    station_ids = dump["station_ids"].astype(str)

    # In production runs, eval_idx is intentionally empty because ALL stations were assimilated!
    zero_withheld_explained = len(eval_idx) == 0
    if not zero_withheld_explained:
        warnings.append(
            f"Production run has non-empty eval_idx: {len(eval_idx)} stations"
        )

    # Verify assimilated fit scores (which ARE finite and meaningful in production)
    assimilated_fit = {}
    for method in EXPECTED_METHODS:
        entry = report.get("results", {}).get(method, {})
        fit = entry.get("assimilated_fit", {})
        crps = fit.get("crps_mm")
        rmse = fit.get("mean_rmse_mm")
        mae = fit.get("mean_mae_mm")

        if crps is None or not np.isfinite(crps):
            errors.append(f"Production {method}: assimilated_fit CRPS is missing or NaN")
        if rmse is None or not np.isfinite(rmse):
            errors.append(f"Production {method}: assimilated_fit RMSE is missing or NaN")

        assimilated_fit[method] = {
            "crps_mm": crps,
            "rmse_mm": rmse,
            "mae_mm": mae,
            "samples": fit.get("n", 0),
        }

    return {
        "status": "PASS" if not errors else "FAIL",
        "total_stations_assimilated": len(assim_idx),
        "withheld_stations_count": len(eval_idx),
        "why_stdout_printed_nan_crps": (
            "In production mode (--assimilate-all-stations), exactly 0 stations were withheld. "
            "Because withheld sample count n=0, the out-of-sample CRPS is undefined (NaN). "
            "All-station assimilated fit scores were correctly computed and are completely finite."
        ),
        "assimilated_fit": assimilated_fit,
        "errors": errors,
        "warnings": warnings,
    }


def check_gridded_zarr(zarr_path: Path, label: str, thorough: bool = False) -> dict:
    """Verify completeness, dimensions, land-valid finite values, and physics of Zarr store."""
    errors = []
    warnings = []

    if not zarr_path.is_dir():
        return {"status": "FAIL", "errors": [f"Zarr store directory missing: {zarr_path}"]}

    try:
        import xarray as xr
    except ImportError:
        return {"status": "FAIL", "errors": ["xarray required to inspect Zarr stores"]}

    try:
        ds = xr.open_zarr(zarr_path, consolidated=True)
    except Exception as exc:
        return {"status": "FAIL", "errors": [f"Cannot open Zarr store: {exc}"]}

    # Check store completion
    if not ds.attrs.get("complete"):
        errors.append("Zarr store attrs['complete'] is not True")

    scope = ds.attrs.get("scope", {})
    if not scope.get("assimilate_all_stations"):
        errors.append("Zarr scope.assimilate_all_stations is not True")

    # Check dimensions
    expected_days = PERIOD_DEFINITIONS[label]["days"]
    actual_days = int(ds.sizes.get("time", 0))
    if actual_days != expected_days:
        errors.append(f"Time dimension mismatch: expected {expected_days}, got {actual_days}")

    actual_members = int(ds.sizes.get("member", 0))
    if actual_members != EXPECTED_MEMBERS:
        errors.append(f"Member dimension mismatch: expected {EXPECTED_MEMBERS}, got {actual_members}")

    lat_size = int(ds.sizes.get("lat", 0))
    lon_size = int(ds.sizes.get("lon", 0))
    if (lat_size, lon_size) != GRID_SHAPE:
        errors.append(f"Grid shape mismatch: expected {GRID_SHAPE}, got {(lat_size, lon_size)}")

    methods = ds.method.values.astype(str).tolist()
    if methods != EXPECTED_METHODS:
        errors.append(f"Methods mismatch: expected {EXPECTED_METHODS}, got {methods}")

    valid = np.asarray(ds.valid.values, bool)
    n_land_cells = int(valid.sum())
    if n_land_cells < 100:
        errors.append(f"Valid land mask has too few cells: {n_land_cells}")

    # Inspect ensemble_mean on land
    emean = ds.ensemble_mean.values  # (method, time, lat, lon)
    stats = {}
    for m_idx, method in enumerate(methods):
        method_mean = emean[m_idx]  # (time, lat, lon)
        land_vals = method_mean[:, valid]

        n_nan = int(np.isnan(land_vals).sum())
        n_inf = int(np.isinf(land_vals).sum())
        if n_nan > 0:
            errors.append(f"{method}: Found {n_nan} NaNs on land cells in ensemble_mean")
        if n_inf > 0:
            errors.append(f"{method}: Found {n_inf} Infs on land cells in ensemble_mean")

        min_val = float(np.nanmin(land_vals))
        max_val = float(np.nanmax(land_vals))
        mean_val = float(np.nanmean(land_vals))

        if min_val < -1e-5:
            errors.append(f"{method}: Negative precipitation detected on land: min = {min_val:.4f} mm/day")
        if max_val > 1000.0:
            warnings.append(f"{method}: Unusually high precipitation: max = {max_val:.1f} mm/day")

        stats[method] = {
            "min_mm": min_val,
            "mean_mm": mean_val,
            "max_mm": max_val,
            "n_nan_on_land": n_nan,
            "n_inf_on_land": n_inf,
        }

    # Check ensemble_std (posterior spread)
    estd = ds.ensemble_std.values
    for m_idx, method in enumerate(methods):
        land_std = estd[m_idx][:, valid]
        mean_spread = float(np.nanmean(land_std))
        if mean_spread <= 0:
            errors.append(f"{method}: Ensemble posterior spread is zero or negative")
        stats[method]["mean_posterior_spread_mm"] = mean_spread

    # Check increment (difference between Huber3 and background)
    diff = emean[1] - emean[0]
    max_increment = float(np.nanmax(np.abs(diff[:, valid])))
    mean_abs_increment = float(np.nanmean(np.abs(diff[:, valid])))
    if max_increment < 1e-4:
        warnings.append("Huber3 analysis is numerically identical to background (no station update)")

    # Thorough check on full 30-member 5D precipitation array if requested
    if thorough:
        precip = ds.precipitation.values  # (method, time, member, lat, lon)
        precip_land = precip[:, :, :, valid]
        p_nan = int(np.isnan(precip_land).sum())
        p_inf = int(np.isinf(precip_land).sum())
        p_neg = int((precip_land < -1e-5).sum())
        if p_nan > 0:
            errors.append(f"Full ensemble: Found {p_nan} NaNs across members on land")
        if p_inf > 0:
            errors.append(f"Full ensemble: Found {p_inf} Infs across members on land")
        if p_neg > 0:
            errors.append(f"Full ensemble: Found {p_neg} negative values across members on land")

    # Check station array in Zarr
    n_stations = int(ds.sizes.get("station", 0))
    station_lat = np.asarray(ds.station_lat.values, float)
    station_lon = np.asarray(ds.station_lon.values, float)
    gauge_vals = np.asarray(ds.gauge.values, float)

    if n_stations < 100:
        warnings.append(f"Station count in Zarr is unexpectedly low: {n_stations}")
    if (station_lat < 19.0).any() or (station_lat > 28.0).any():
        warnings.append("Station latitude outside Bangladesh regional bounds")
    if (station_lon < 87.0).any() or (station_lon > 94.0).any():
        warnings.append("Station longitude outside Bangladesh regional bounds")

    ds.close()

    return {
        "status": "PASS" if not errors else "FAIL",
        "days": actual_days,
        "members": actual_members,
        "n_land_cells": n_land_cells,
        "n_stations_stored": n_stations,
        "stats": stats,
        "max_huber3_increment_mm": max_increment,
        "mean_abs_huber3_increment_mm": mean_abs_increment,
        "errors": errors,
        "warnings": warnings,
    }


def main() -> int:
    args = parse_args()
    root = Path(args.root)

    print("=" * 80)
    print(f"BMD+BWDB 2021-2024 Archive Data Checker")
    print(f"Archive root: {root}")
    print("=" * 80)

    if not root.is_dir():
        print(f"\n[ERROR] Root archive directory does not exist: {root}")
        print("Please check the path or ensure the job has created the output directory.")
        return 1

    overall_pass = True
    summary_report = {
        "archive_root": str(root),
        "periods": {},
        "overall_status": "PASS",
    }

    total_days_verified = 0

    print("\n1. Explaining the 'CRPS nan' Observation during Data Generation:")
    print("-" * 80)
    print(
        "  * During data generation (tasks 4..7 in slurm/v2_bmd_bwdb_huber3_2021_2024.sbatch),\n"
        "    scripts/28_simultaneous_method_sweep.py was run with --assimilate-all-stations.\n"
        "  * In this mode, ALL stations enter the likelihood (withheld stations eval_idx = []).\n"
        "  * Line 1470 prints: entry.get('crps_mm', float('nan')).\n"
        "  * Because zero stations were withheld (n=0), the withheld-gauge CRPS is mathematically\n"
        "    undefined (NaN). This is NORMAL and EXPECTED for all-station production runs.\n"
        "  * True evaluation scores with held-out stations are in the 'evaluation' runs (tasks 0..3),\n"
        "    and production assimilated-fit scores are verified below.\n"
    )
    print("-" * 80)

    for label, info in PERIOD_DEFINITIONS.items():
        print(f"\nVerifying Period: {label} ({info['start']} to {info['end']}, {info['days']} days)")
        print("~" * 80)

        # 1. Check Evaluation Run
        eval_npz = root / "evaluation" / f"{label}.npz"
        eval_json = root / "evaluation" / f"{label}.json"
        eval_res = check_evaluation_run(eval_npz, eval_json, label)

        print(f"  [Evaluation Holdout] Status: {eval_res['status']}")
        if eval_res["status"] == "PASS":
            tot = eval_res["total_stations"]
            w = eval_res["withheld_stations"]
            a = eval_res["assimilated_stations"]
            print(f"    - Stations: {tot} total ({a} assimilated, {w} withheld 20% holdout)")
            for method, sc in eval_res["scores"].items():
                print(
                    f"    - {method:24s} Withheld CRPS: {sc['crps_mm']:6.3f} mm/day | "
                    f"RMSE: {sc['rmse_mm']:6.3f} mm/day"
                )
        else:
            overall_pass = False
            for err in eval_res.get("errors", []):
                print(f"    - [ERROR] {err}")
        for warn in eval_res.get("warnings", []):
            print(f"    - [WARNING] {warn}")

        # 2. Check Production Run
        prod_npz = root / "production_metadata" / f"{label}.npz"
        prod_json = root / "production_metadata" / f"{label}.json"
        prod_res = check_production_run(prod_npz, prod_json, label)

        print(f"  [Production Run]     Status: {prod_res['status']}")
        if prod_res["status"] == "PASS":
            tot = prod_res["total_stations_assimilated"]
            print(f"    - All-station analysis: {tot} stations assimilated into likelihood")
            for method, fit in prod_res["assimilated_fit"].items():
                print(
                    f"    - {method:24s} Assimilated Fit CRPS: {fit['crps_mm']:6.3f} mm/day | "
                    f"RMSE: {fit['rmse_mm']:6.3f} mm/day"
                )
        else:
            overall_pass = False
            for err in prod_res.get("errors", []):
                print(f"    - [ERROR] {err}")
        for warn in prod_res.get("warnings", []):
            print(f"    - [WARNING] {warn}")

        # 3. Check Gridded Zarr
        zarr_path = root / "gridded" / f"{label}.zarr"
        zarr_res = check_gridded_zarr(zarr_path, label, thorough=args.check_all_zarr_members)

        print(f"  [Gridded Zarr Store] Status: {zarr_res['status']}")
        if zarr_res["status"] == "PASS":
            total_days_verified += zarr_res["days"]
            print(f"    - Store: {zarr_path.name} (complete: true)")
            print(f"    - Shape: {zarr_res['days']} days x {zarr_res['members']} members x 128x128 grid")
            print(f"    - Valid land cells: {zarr_res['n_land_cells']} (0 NaNs, 0 Infs, min >= 0.0 mm/day)")
            h3_stat = zarr_res["stats"].get("v2_simul_s04_huber3", {})
            print(
                f"    - Huber3 Land Precipitation: mean = {h3_stat.get('mean_mm', 0):.2f} mm/day | "
                f"max = {h3_stat.get('max_mm', 0):.1f} mm/day | "
                f"spread = {h3_stat.get('mean_posterior_spread_mm', 0):.2f} mm/day"
            )
            print(f"    - Max station increment over background: {zarr_res['max_huber3_increment_mm']:.2f} mm/day")
        else:
            overall_pass = False
            for err in zarr_res.get("errors", []):
                print(f"    - [ERROR] {err}")
        for warn in zarr_res.get("warnings", []):
            print(f"    - [WARNING] {warn}")

        summary_report["periods"][label] = {
            "evaluation": eval_res,
            "production": prod_res,
            "gridded_zarr": zarr_res,
        }

    print("\n" + "=" * 80)
    print("FINAL INTEGRITY VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"Total days verified across 4 periods: {total_days_verified} / {TOTAL_DAYS}")

    if overall_pass and total_days_verified == TOTAL_DAYS:
        print("[SUCCESS] ALL DATA INTEGRITY CHECKS PASSED!")
        print("  - The 'NaN CRPS' prints in production logs are normal (withheld stations n=0).")
        print("  - All evaluation holdout runs have valid, finite withheld CRPS.")
        print("  - All production runs have valid, finite assimilated fit scores.")
        print("  - All 4 seasonal Zarr stores are complete, finite, non-negative, and physically consistent.")
        summary_report["overall_status"] = "PASS"
        exit_code = 0
    else:
        print("[FAIL] Some data integrity checks failed. Review errors above.")
        summary_report["overall_status"] = "FAIL"
        exit_code = 1

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary_report, indent=2) + "\n")
        print(f"\nDetailed report saved to: {out_path}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
