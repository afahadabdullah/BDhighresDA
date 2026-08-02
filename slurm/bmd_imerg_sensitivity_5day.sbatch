#!/bin/bash
#SBATCH --job-name=bdhires-imerg-sens
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --array=0-2%2
#SBATCH --output=logs/%x-%A_%a.out

# Three controlled satellite weights. Each GPU run generates its satellite
# posterior once, then evaluates 75 and 100 km gauge localizations on CPU.

set -euo pipefail

R_VALUES=(2 4 8)
R_VALUE="${R_VALUES[$SLURM_ARRAY_TASK_ID]}"
LABEL="aligned_r${R_VALUE}_l75-100"

export BMD_START="2018-05-01"
export BMD_END="2018-05-05"
export IMERG_DIR="${IMERG_DIR:-data/imerg}"
export IMERG_SOURCE_FREQUENCY="half-hourly"
export BMD_ACCUMULATION_END_HOUR_UTC="3"
export IMERG_MIN_COUNT="${IMERG_MIN_COUNT:-48}"
export IMERG_STRIDE="${IMERG_STRIDE:-3}"
export IMERG_R_MULTIPLIER="$R_VALUE"
export GAUGE_LOCALIZATION_KM="100"
export GAUGE_LOCALIZATIONS_KM="75,100"

export BMD_DAILY="data/processed/bmd_daily_sensitivity_${LABEL}.csv"
export BMD_STATION_SUMMARY="data/processed/bmd_stations_sensitivity_${LABEL}.csv"
export BMD_QC_REPORT="data/processed/bmd_qc_sensitivity_${LABEL}.json"
export IMERG_PREPARED="data/processed/imerg_bd_sensitivity_${LABEL}.nc"
export IMERG_QC="data/processed/imerg_bd_sensitivity_${LABEL}_qc.json"
export BMD_DUMP="data/processed/bmd_imerg_sensitivity_${LABEL}.npz"
export BMD_REPORT="data/processed/bmd_imerg_sensitivity_${LABEL}.json"
export BMD_DIAGNOSTICS="data/processed/bmd_imerg_sensitivity_${LABEL}_diagnostics.png"
export BMD_EVENTS="data/processed/bmd_imerg_sensitivity_${LABEL}_events.png"
export BMD_STATION_COMPARISON="data/processed/bmd_imerg_sensitivity_${LABEL}_station_comparison.png"
export BMD_SPATIAL="data/processed/bmd_imerg_sensitivity_${LABEL}_spatial.png"
export BMD_INTERCOMPARISON="data/processed/bmd_imerg_sensitivity_${LABEL}_intercomparison.png"
export BMD_EVALUATION="data/processed/bmd_imerg_sensitivity_${LABEL}_evaluation.json"

cd "$SLURM_SUBMIT_DIR"
echo "Aligned sensitivity case: IMERG extra R=${R_VALUE}x; localizations=75,100 km"
exec bash slurm/bmd_imerg_example.sbatch
