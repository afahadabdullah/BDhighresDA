#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd -- "$SCRIPT_DIR/.." && pwd)"
mkdir -p logs

download_result="$(sbatch --parsable "$@" slurm/download_imerg_halfhourly_2017_may_sep.sbatch)"
download_job="${download_result%%;*}"
array_result="$(sbatch --parsable --dependency="afterok:${download_job}" "$@" \
    slurm/bmd_imerg_rotated_folds_2017_may_sep_smoke.sbatch)"
array_job="${array_result%%;*}"
summary_result="$(sbatch --parsable --dependency="afterok:${array_job}" "$@" \
    slurm/bmd_imerg_rotated_folds_2017_may_sep_smoke_summary.sbatch)"
summary_job="${summary_result%%;*}"

echo "submitted IMERG download/validation: ${download_job}"
echo "submitted dependent five-fold GPU process test: ${array_job}"
echo "submitted dependent CPU diagnostics: ${summary_job}"
echo "period: 2017-05-01 through 2017-09-30 (153 days)"
echo "default ensemble: 4 members; override with --export=ALL,BMD_MEMBERS=8"
echo "outputs: data/processed/bmd_imerg_smoke_2017_may_sep/"
