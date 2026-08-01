#!/usr/bin/env bash
set -euo pipefail

# Fast controlled gate. It runs five matched arms for May 1-5 only and keeps
# every output separate from the earlier unstable simultaneous experiment.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd -- "$SCRIPT_DIR/.." && pwd)"

export BMD_START="${BMD_START:-2018-05-01}"
export BMD_END="${BMD_END:-2018-05-05}"
export IMERG_STRIDE="${IMERG_STRIDE:-3}"
export IMERG_R_MULTIPLIER="${IMERG_R_MULTIPLIER:-1.0}"
export GAUGE_LOCALIZATION_KM="${GAUGE_LOCALIZATION_KM:-150}"
export GAUGE_LOCALIZATIONS_KM="${GAUGE_LOCALIZATIONS_KM:-75,100}"

export BMD_DAILY="${BMD_DAILY:-data/processed/bmd_daily_controlled_20180501_05.csv}"
export BMD_STATION_SUMMARY="${BMD_STATION_SUMMARY:-data/processed/bmd_stations_controlled_20180501_05.csv}"
export BMD_QC_REPORT="${BMD_QC_REPORT:-data/processed/bmd_qc_controlled_20180501_05.json}"
export IMERG_PREPARED="${IMERG_PREPARED:-data/processed/imerg_bd_controlled_20180501_05.nc}"
export IMERG_QC="${IMERG_QC:-data/processed/imerg_bd_controlled_20180501_05_qc.json}"
export BMD_DUMP="${BMD_DUMP:-data/processed/bmd_imerg_controlled_20180501_05.npz}"
export BMD_REPORT="${BMD_REPORT:-data/processed/bmd_imerg_controlled_20180501_05.json}"
export BMD_DIAGNOSTICS="${BMD_DIAGNOSTICS:-data/processed/bmd_imerg_controlled_20180501_05_diagnostics.png}"
export BMD_SPATIAL="${BMD_SPATIAL:-data/processed/bmd_imerg_controlled_20180501_05_spatial.png}"
export BMD_CHIRPS_SPATIAL="${BMD_CHIRPS_SPATIAL:-data/processed/bmd_imerg_controlled_20180501_05_chirps_spatial.png}"

mkdir -p logs
exec sbatch "$@" slurm/bmd_imerg_example.sbatch
