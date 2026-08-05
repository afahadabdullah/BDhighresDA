#!/usr/bin/env bash
# Submit real BMD + IMERG evaluation pipeline for all periods from 2021 through 2024.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd -- "$SCRIPT_DIR/.." && pwd)"
mkdir -p logs

BMD_MEMBERS="${BMD_MEMBERS:-16}"

echo "============================================================"
echo " Submitting Real BMD + IMERG Evaluation Pipeline (2021-2024)"
echo " Ensemble Members: ${BMD_MEMBERS}"
echo "============================================================"

# 1. Year 2021 May - September (Monsoon)
echo -e "\n[1/5] Submitting 2021 May--September..."
BMD_START="2021-05-01" BMD_END="2021-09-30" BMD_EVAL_LABEL="2021_may_sep" BMD_MEMBERS="$BMD_MEMBERS" \
  bash slurm/submit_bmd_imerg_eval.sh "$@"

# 2. Year 2022 May - September (Monsoon)
echo -e "\n[2/5] Submitting 2022 May--September..."
BMD_START="2022-05-01" BMD_END="2022-09-30" BMD_EVAL_LABEL="2022_may_sep" BMD_MEMBERS="$BMD_MEMBERS" \
  bash slurm/submit_bmd_imerg_eval.sh "$@"

# 3. Year 2023 May - September (Monsoon)
echo -e "\n[3/5] Submitting 2023 May--September..."
BMD_START="2023-05-01" BMD_END="2023-09-30" BMD_EVAL_LABEL="2023_may_sep" BMD_MEMBERS="$BMD_MEMBERS" \
  bash slurm/submit_bmd_imerg_eval.sh "$@"

# 4. Year 2024 May - June (Available Archive Period)
echo -e "\n[4/5] Submitting 2024 May--June..."
BMD_START="2024-05-01" BMD_END="2024-06-30" BMD_EVAL_LABEL="2024_may_jun" BMD_MEMBERS="$BMD_MEMBERS" \
  bash slurm/submit_bmd_imerg_eval.sh "$@"

# 5. Full Combined Multi-Year Record (2021-05-01 through 2024-06-30)
echo -e "\n[5/5] Submitting Full Multi-Year Record (2021-05-01 to 2024-06-30)..."
BMD_START="2021-05-01" BMD_END="2024-06-30" BMD_EVAL_LABEL="2021_2024_full" BMD_MEMBERS="$BMD_MEMBERS" \
  bash slurm/submit_bmd_imerg_eval.sh "$@"

echo -e "\nAll 2021-2024 evaluation runs submitted successfully!"
