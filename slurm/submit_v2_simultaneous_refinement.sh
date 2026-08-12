#!/usr/bin/env bash
set -euo pipefail

# Submit new CPC-v2 simultaneous arms and summarize them against the already
# completed best gauges, IMERG-only, and simultaneous S04 configurations.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd -- "$SCRIPT_DIR/.." && pwd)"
mkdir -p logs

export V2_REFINE_START="${V2_REFINE_START:-2022-05-01}"
export V2_REFINE_END="${V2_REFINE_END:-2022-05-10}"
export V2_REFINE_MEMBERS="${V2_REFINE_MEMBERS:-30}"
export V2_REFINE_SEED="${V2_REFINE_SEED:-201805}"
export V2_REFINE_ROOT="${V2_REFINE_ROOT:-data/processed/v2_simultaneous_refinement/ing2022_s04}"
export V2_REFINE_REFERENCE_ROOT="${V2_REFINE_REFERENCE_ROOT:-data/processed/v2_ingestion_triplet/ing2022_s04_g010_sqrtfix}"
export V2_REFINE_IMERG="${V2_REFINE_IMERG:-data/processed/imerg_prepared_ing2022/imerg_0p4deg_20220501_20220510.nc}"
export BMD_CKPT="${BMD_CKPT:-runs/prior_h100_cpc_v2/best.pt}"
export BMD_CONFIG="${BMD_CONFIG:-configs/da.yaml}"
export BMD_DATA_DIR="${BMD_DATA_DIR:-data/stations/data_2020_2025}"
export BMD_STATIONS="${BMD_STATIONS:-data/stations/data_2020_2025/Stations.csv}"
export BACKGROUND_DAY_OFFSET="${BACKGROUND_DAY_OFFSET:--1}"

for required in "$BMD_CKPT" "$BMD_CONFIG" "$BMD_STATIONS" "$V2_REFINE_IMERG"; do
    [[ -f "$required" ]] || { echo "ERROR: required file missing: $required"; exit 1; }
done
[[ -d "$BMD_DATA_DIR" ]] || { echo "ERROR: missing $BMD_DATA_DIR"; exit 1; }
for fold in 0 1 2 3 4; do
    for suffix in npz json; do
        required="$V2_REFINE_REFERENCE_ROOT/fold${fold}.${suffix}"
        [[ -s "$required" ]] || {
            echo "ERROR: completed S04 reference missing: $required"
            echo "Set V2_REFINE_REFERENCE_ROOT to the corrected triplet directory."
            exit 1
        }
    done
done
mkdir -p "$V2_REFINE_ROOT"

echo "Submitting CPC-v2 simultaneous S04 refinement tournament"
echo "  period: $V2_REFINE_START through $V2_REFINE_END"
echo "  members: $V2_REFINE_MEMBERS; checkpoint: $BMD_CKPT"
echo "  existing best controls: $V2_REFINE_REFERENCE_ROOT"
echo "  IMERG: $V2_REFINE_IMERG"
echo "  new outputs: $V2_REFINE_ROOT"
echo "  completed candidate folds are reused on resubmission"

array_result="$(sbatch --parsable --export=ALL "$@" \
    slurm/v2_simultaneous_refinement.sbatch)"
array_job="${array_result%%;*}"
summary_result="$(sbatch --parsable --dependency="afterok:${array_job}" \
    --export=ALL "$@" slurm/v2_simultaneous_refinement_summary.sbatch)"
summary_job="${summary_result%%;*}"

echo "submitted five-fold GPU array: $array_job"
echo "submitted dependent pooled summary: $summary_job"
echo "monitor: squeue -u $USER"
echo "logs:    logs/bdhires-v2-simul-refine-${array_job}_*.out"
echo "final:   $V2_REFINE_ROOT/refinement_selection.{md,json,png}"
echo "folds:   $V2_REFINE_ROOT/fold_plots/fold{0..4}_diagnostics.png"
