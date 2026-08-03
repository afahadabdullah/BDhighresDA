#!/bin/bash
# Reproduce the original successful CHIRPS OSSE design on a balanced JJA sample:
# one June, July and August nature-run day per year in 2021--2024.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd -- "$SCRIPT_DIR/.." && pwd)"
mkdir -p logs

OBS_ERROR="${OSSE_OBS_ERROR:-realistic}"

export OSSE_START="${OSSE_START:-2021-01-01}"
export OSSE_END="${OSSE_END:-2024-12-31}"
export OSSE_MONTHS="${OSSE_MONTHS:-6,7,8}"
export OSSE_DAYS="${OSSE_DAYS:-12}"
export OSSE_MEMBERS="${OSSE_MEMBERS:-16}"
export OSSE_NETWORKS="${OSSE_NETWORKS:-40}"
export OSSE_LAYOUT="${OSSE_LAYOUT:-spread}"
export OSSE_WITHHOLD="${OSSE_WITHHOLD:-0.2}"
export OSSE_PSEUDO_SATELLITE="${OSSE_PSEUDO_SATELLITE:-1}"
export OSSE_OBSERVATION_MODE="${OSSE_OBSERVATION_MODE:-combined}"
export OSSE_SATELLITE_STRIDE="${OSSE_SATELLITE_STRIDE:-1}"
export OSSE_SATELLITE_CORRELATION_CONTROL="${OSSE_SATELLITE_CORRELATION_CONTROL:-0}"
export OSSE_OBS_ERROR="$OBS_ERROR"
export OSSE_DUMP_OBS_ERROR="${OSSE_DUMP_OBS_ERROR:-$OBS_ERROR}"
export OSSE_OUT_DIR="${OSSE_OUT_DIR:-data/processed/osse_jja_2021_2024_${OBS_ERROR}}"

exec sbatch "$@" slurm/osse.sbatch
