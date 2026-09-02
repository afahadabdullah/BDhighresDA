#!/usr/bin/env bash
set -euo pipefail

# Reuse the complete CPC-v2 gridded-evaluation/plotting suite with the
# super-obbed combined BMD+BWDB archive's single constrained holdout per season.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd -- "$SCRIPT_DIR/.." && pwd)"

export V2_CONFIRM_ROOT="${V2_BWDB_WINNER_ROOT:-data/processed/v2_bmd_bwdb_superob_2021_2024}"
export V2_EVAL_CV_ROOT="${V2_EVAL_CV_ROOT:-$V2_CONFIRM_ROOT}"
export V2_EVAL_CV_LAYOUT="single-holdout"
export V2_EVAL_SELECTION_DAILY_START="${V2_EVAL_SELECTION_DAILY_START:-2022-05-01}"
export V2_EVAL_SELECTION_DAILY_END="${V2_EVAL_SELECTION_DAILY_END:-2022-05-31}"
export V2_EVAL_TEXTURE_MEMBERS="${V2_EVAL_TEXTURE_MEMBERS:-5}"

exec bash slurm/submit_v2_gridded_evaluation.sh "$@"
