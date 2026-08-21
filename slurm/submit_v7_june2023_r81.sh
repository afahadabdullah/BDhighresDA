#!/usr/bin/env bash
set -euo pipefail

# Prepare the exact June inputs and CPCv2 comparison contract, then submit five
# V7 cross-validation folds plus one all-station physical-ensemble production
# task. A dependent summary job pools the folds against the existing CPCv2 2023
# confirmation archive only after all six V7 tasks succeed; the launcher places
# that summary on a Grace GPU node as requested.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd -- "$SCRIPT_DIR/.." && pwd)"
mkdir -p logs

export V7_JUNE_ROOT="${V7_JUNE_ROOT:-data/processed/v7_june2023_r81_latest_latest}"
export V7_JUNE_CPC_ROOT="${V7_JUNE_CPC_ROOT:-data/processed/v2_confirmatory_2021_2024}"
export V7_JUNE_MEMBERS="${V7_JUNE_MEMBERS:-30}"
export V7_JUNE_STEPS="${V7_JUNE_STEPS:-50}"
export V7_JUNE_SEED="${V7_JUNE_SEED:-201805}"
export V7_JUNE_ARM_SET="${V7_JUNE_ARM_SET:-r81}"
export V7_JUNE_COMPARISON_LABEL="${V7_JUNE_COMPARISON_LABEL:-june2023_v7_r81_vs_cpcv2}"
export V7_JUNE_TOURNAMENT="${V7_JUNE_TOURNAMENT:-data/processed/v7_checkpoint_tournament_may03/20260821_1323}"
export V7_JUNE_MESO_CKPT="${V7_JUNE_MESO_CKPT:-$V7_JUNE_TOURNAMENT/source_checkpoints/latest_meso.pt}"
export V7_JUNE_ALLOC_CKPT="${V7_JUNE_ALLOC_CKPT:-$V7_JUNE_TOURNAMENT/source_checkpoints/latest_allocation.pt}"
export V7_JUNE_IMERG_SOURCE="${V7_JUNE_IMERG_SOURCE:-data/processed/imerg_bd_aligned_*.nc}"
export BMD_DATA_DIR="${BMD_DATA_DIR:-data/stations/data_2020_2025}"
export BMD_STATIONS="${BMD_STATIONS:-data/stations/data_2020_2025/Stations.csv}"

if [[ -n "${V7_JUNE_PREP_PYTHON:-}" ]]; then
    PREP_PYTHON="$V7_JUNE_PREP_PYTHON"
elif command -v python >/dev/null 2>&1; then
    PREP_PYTHON="python"
else
    PREP_PYTHON="python3"
fi
if ! command -v "$PREP_PYTHON" >/dev/null 2>&1 \
   || ! "$PREP_PYTHON" -c 'import numpy, pandas, xarray' >/dev/null 2>&1; then
    echo "ERROR: preparation Python needs numpy, pandas and xarray: $PREP_PYTHON" >&2
    echo "Run 'conda activate mytorch' first, or set V7_JUNE_PREP_PYTHON." >&2
    exit 1
fi

for required in "$V7_JUNE_MESO_CKPT" "$V7_JUNE_ALLOC_CKPT" "$BMD_STATIONS" \
                scripts/05_convert_bmd_dir.py scripts/43_subset_prepared_imerg.py \
                scripts/44_coarsen_imerg_observations.py \
                scripts/72_v7_two_stage_osse.py scripts/79_compare_v7_cpcv2_folds.py; do
    [[ -f "$required" ]] || { echo "ERROR: required file missing: $required" >&2; exit 1; }
done
[[ -d "$BMD_DATA_DIR" ]] || { echo "ERROR: missing $BMD_DATA_DIR" >&2; exit 1; }
[[ -e "$V7_JUNE_IMERG_SOURCE" ]] || compgen -G "$V7_JUNE_IMERG_SOURCE" >/dev/null || {
    echo "ERROR: no prepared monthly IMERG matches $V7_JUNE_IMERG_SOURCE" >&2
    exit 1
}

INPUT_DIR="$V7_JUNE_ROOT/inputs"
CONTRACT_DIR="$V7_JUNE_ROOT/comparison_contract"
mkdir -p "$INPUT_DIR" "$CONTRACT_DIR"
DAILY="$INPUT_DIR/bmd_daily_202306.csv"
if [[ ! -s "$DAILY" ]]; then
    PYTHONPATH="$PWD/src" "$PREP_PYTHON" -u scripts/05_convert_bmd_dir.py \
        --data-dir "$BMD_DATA_DIR" --stations "$BMD_STATIONS" \
        --start 2023-06-01 --end 2023-06-30 \
        --out "$DAILY" \
        --summary "$INPUT_DIR/bmd_stations_202306.csv" \
        --report "$INPUT_DIR/bmd_202306_qc.json"
else
    echo "Reusing $DAILY"
fi

IMERG="$INPUT_DIR/imerg_native_20230601_20230630.nc"
if [[ ! -s "$IMERG" ]]; then
    PYTHONPATH="$PWD/src" "$PREP_PYTHON" -u scripts/43_subset_prepared_imerg.py \
        --input "$V7_JUNE_IMERG_SOURCE" \
        --start 2023-06-01 --end 2023-06-30 \
        --out "$IMERG" --report "$INPUT_DIR/imerg_native_202306_qc.json"
else
    echo "Reusing $IMERG"
fi

if [[ "$V7_JUNE_ARM_SET" == "three_arm" ]]; then
    S04_IMERG="$INPUT_DIR/imerg_s04_20230601_20230630.nc"
    if [[ ! -s "$S04_IMERG" ]]; then
        PYTHONPATH="$PWD/src" "$PREP_PYTHON" -u \
            scripts/44_coarsen_imerg_observations.py \
            --input "$IMERG" --factor 8 --out "$S04_IMERG" \
            --report "$INPUT_DIR/imerg_s04_202306_qc.json"
    else
        echo "Reusing $S04_IMERG"
    fi
fi

# Freeze CPCv2's exact eligible pool and fold membership into plain-text files.
# The source dumps are May-Sep, but their June rows are audited again by the
# dependent comparison before a score is produced.
PYTHONPATH="$PWD/src" "$PREP_PYTHON" - "$V7_JUNE_CPC_ROOT" "$CONTRACT_DIR" <<'PY'
from pathlib import Path
import sys
import numpy as np

root, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
wanted = np.arange(np.datetime64("2023-06-01"), np.datetime64("2023-07-01"))
pool = None
held = []
for fold in range(5):
    path = root / "cv" / "2023_may_sep" / f"fold{fold}.npz"
    if not path.is_file():
        raise SystemExit(
            f"missing CPCv2 comparison input {path}; complete "
            "slurm/submit_v2_confirmatory_2021_2024.sh first"
        )
    with np.load(path, allow_pickle=False) as data:
        ids = np.asarray(data["station_ids"]).astype(str)
        times = np.asarray(data["times"]).astype("datetime64[D]")
        missing = np.setdiff1d(wanted, times)
        if missing.size:
            raise SystemExit(f"{path} lacks June dates {missing.astype(str).tolist()}")
        if "station_v2_simul_s04_ig010" not in data:
            raise SystemExit(f"{path} lacks the CPCv2 winning simultaneous arm")
        members = np.asarray(data["station_v2_simul_s04_ig010"]).shape[1]
        if members != 30:
            raise SystemExit(f"{path} has {members} members, expected 30")
        if pool is None:
            pool = ids
        elif set(ids.tolist()) != set(pool.tolist()):
            raise SystemExit(f"{path} has a different station pool")
        fold_ids = ids[np.asarray(data["eval_idx"], int)].astype(str)
    overlap = set(fold_ids.tolist()) & set(held)
    if overlap:
        raise SystemExit(f"CPCv2 fold {fold} repeats withheld IDs {sorted(overlap)}")
    held.extend(fold_ids.tolist())
    (out / f"cpcv2_2023_fold{fold}_withheld_ids.txt").write_text(
        "\n".join(fold_ids.tolist()) + "\n"
    )
if set(held) != set(pool.tolist()):
    raise SystemExit("CPCv2 folds do not partition the station pool")
(out / "cpcv2_2023_station_ids.txt").write_text("\n".join(pool.tolist()) + "\n")
print(f"Frozen {len(pool)} CPCv2 stations and five exact folds under {out}")
PY

echo "Submitting June 2023 V7 pilot"
echo "  six GPU tasks: five CPCv2-matched folds + one all-station Zarr"
echo "  members: $V7_JUNE_MEMBERS; steps: $V7_JUNE_STEPS; seed: $V7_JUNE_SEED"
echo "  arm set: $V7_JUNE_ARM_SET"
echo "  frozen meso: $V7_JUNE_MESO_CKPT"
echo "  frozen allocation: $V7_JUNE_ALLOC_CKPT"
echo "  output root: $V7_JUNE_ROOT"

array_result="$(sbatch --parsable --export=ALL "$@" slurm/v7_june2023_r81.sbatch)"
array_job="${array_result%%;*}"
summary_result="$(sbatch --parsable --dependency="afterok:${array_job}" \
    --partition="${V7_JUNE_SUMMARY_PARTITION:-grace}" \
    --gres="${V7_JUNE_SUMMARY_GRES:-gpu:1}" \
    --export=ALL "$@" slurm/v7_june2023_r81_summary.sbatch)"
summary_job="${summary_result%%;*}"

echo "submitted V7 GPU array: $array_job"
echo "submitted dependent comparison: $summary_job"
echo "monitor: squeue -u $USER"
echo "logs:    logs/bdhires-v7-jun23-${array_job}_*.out"
echo "Zarr:    $V7_JUNE_ROOT/gridded/june2023.zarr"
echo "scores:  $V7_JUNE_ROOT/comparison/${V7_JUNE_COMPARISON_LABEL}.md"
