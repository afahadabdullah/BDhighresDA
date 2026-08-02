#!/usr/bin/env bash
set -euo pipefail

# Matched timing sensitivity: retain the same BMD/IMERG observation dates,
# station holdout and stochastic seeds as the aligned five-day run, but use
# the complete previous-day checkpoint prior (CPC, ERA5, residual base and
# seasonal encoding). Observation-date CHIRPS stays fixed as context only.
# Outputs are separate from the offset-0 run.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd -- "$SCRIPT_DIR/.." && pwd)"

export BMD_START="${BMD_START:-2018-05-01}"
export BMD_END="${BMD_END:-2018-05-05}"
export BACKGROUND_DAY_OFFSET="${BACKGROUND_DAY_OFFSET:--1}"
export IMERG_DIR="${IMERG_DIR:-data/imerg_halfhourly}"
export IMERG_SOURCE_FREQUENCY="${IMERG_SOURCE_FREQUENCY:-half-hourly}"
export BMD_ACCUMULATION_END_HOUR_UTC="${BMD_ACCUMULATION_END_HOUR_UTC:-3}"
export IMERG_MIN_COUNT="${IMERG_MIN_COUNT:-48}"
export IMERG_STRIDE="${IMERG_STRIDE:-3}"
export IMERG_R_MULTIPLIER="${IMERG_R_MULTIPLIER:-1.0}"
export GAUGE_LOCALIZATION_KM="${GAUGE_LOCALIZATION_KM:-150}"
export GAUGE_LOCALIZATIONS_KM="${GAUGE_LOCALIZATIONS_KM:-75,100}"

PREFIX="data/processed/bmd_imerg_aligned_offset_m1_20180501_05"
export BMD_DAILY="${BMD_DAILY:-data/processed/bmd_daily_aligned_20180501_05.csv}"
export BMD_STATION_SUMMARY="${BMD_STATION_SUMMARY:-data/processed/bmd_stations_aligned_20180501_05.csv}"
export BMD_QC_REPORT="${BMD_QC_REPORT:-data/processed/bmd_qc_aligned_20180501_05.json}"
export IMERG_PREPARED="${IMERG_PREPARED:-data/processed/imerg_bd_aligned_20180501_05.nc}"
export IMERG_QC="${IMERG_QC:-data/processed/imerg_bd_aligned_20180501_05_qc.json}"
export BMD_DUMP="${BMD_DUMP:-${PREFIX}.npz}"
export BMD_REPORT="${BMD_REPORT:-${PREFIX}.json}"
export BMD_DIAGNOSTICS="${BMD_DIAGNOSTICS:-${PREFIX}_diagnostics.png}"
export BMD_EVENTS="${BMD_EVENTS:-${PREFIX}_events.png}"
export BMD_STATION_COMPARISON="${BMD_STATION_COMPARISON:-${PREFIX}_station_comparison.png}"
export BMD_SPATIAL="${BMD_SPATIAL:-${PREFIX}_spatial.png}"
export BMD_INTERCOMPARISON="${BMD_INTERCOMPARISON:-${PREFIX}_intercomparison.png}"
export BMD_EVALUATION="${BMD_EVALUATION:-${PREFIX}_evaluation.json}"

mkdir -p logs
exec sbatch "$@" slurm/bmd_imerg_example.sbatch
