#!/usr/bin/env bash
#SBATCH --job-name=bdhires-pooled-summary
#SBATCH --partition=grace-cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module purge 2>/dev/null || true

export ENV_PREFIX="/home/afahad/nb/project/BDDA/envs/bdda-gh200"
PYTHON_BIN="$ENV_PREFIX/bin/python"
export PYTHONNOUSERSITE=1
unset PYTHONHOME
unset PYTHONPATH

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src"

srun --ntasks=1 "$PYTHON_BIN" -u scripts/22_summarize_multiyear_bmd_eval.py \
    --summaries data/processed/bmd_imerg_eval_2021_may_sep/rotated_summary.json \
                data/processed/bmd_imerg_eval_2022_may_sep/rotated_summary.json \
                data/processed/bmd_imerg_eval_2023_may_sep/rotated_summary.json \
                data/processed/bmd_imerg_eval_2024_may_jun/rotated_summary.json \
    --out-json data/processed/bmd_imerg_2021_2024_pooled_summary.json \
    --out-markdown data/processed/bmd_imerg_2021_2024_pooled_summary.md \
    --out-plot data/processed/bmd_imerg_2021_2024_pooled_summary.png

echo "Multi-year pooled summary complete."
echo "Markdown Table: data/processed/bmd_imerg_2021_2024_pooled_summary.md"
echo "JSON Data: data/processed/bmd_imerg_2021_2024_pooled_summary.json"
echo "Summary Plot: data/processed/bmd_imerg_2021_2024_pooled_summary.png"

