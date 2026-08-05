#!/usr/bin/env bash
# Submit real BMD + IMERG evaluation pipeline for all periods from 2021 through 2024,
# and automatically schedule a dependent multi-year pooled summary job at the end.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd -- "$SCRIPT_DIR/.." && pwd)"
mkdir -p logs

BMD_MEMBERS="${BMD_MEMBERS:-16}"

echo "============================================================"
echo " Submitting Real BMD + IMERG Evaluation Pipeline (2021-2024)"
echo " Ensemble Members: ${BMD_MEMBERS}"
echo "============================================================"

# Helper function to submit an eval run and extract summary job ID
submit_period() {
    local start="$1" end="$2" label="$3"
    shift 3
    local output
    output="$(BMD_START="$start" BMD_END="$end" BMD_EVAL_LABEL="$label" BMD_MEMBERS="$BMD_MEMBERS" \
      bash slurm/submit_bmd_imerg_eval.sh "$@")"
    echo "$output" >&2
    local job_id
    job_id="$(echo "$output" | grep "^SUMMARY_JOB:" | cut -d':' -f2 | tr -d '[:space:]')"
    echo "$job_id"
}

# 1. Year 2021 May - September (Monsoon)
echo -e "\n[1/5] Submitting 2021 May--September..." >&2
job2021="$(submit_period "2021-05-01" "2021-09-30" "2021_may_sep" "$@")"

# 2. Year 2022 May - September (Monsoon)
echo -e "\n[2/5] Submitting 2022 May--September..." >&2
job2022="$(submit_period "2022-05-01" "2022-09-30" "2022_may_sep" "$@")"

# 3. Year 2023 May - September (Monsoon)
echo -e "\n[3/5] Submitting 2023 May--September..." >&2
job2023="$(submit_period "2023-05-01" "2023-09-30" "2023_may_sep" "$@")"

# 4. Year 2024 May - June (Available Archive Period)
echo -e "\n[4/5] Submitting 2024 May--June..." >&2
job2024="$(submit_period "2024-05-01" "2024-06-30" "2024_may_jun" "$@")"

# 5. Full Combined Multi-Year Record (2021-05-01 through 2024-06-30)
echo -e "\n[5/5] Submitting Full Multi-Year Record (2021-05-01 to 2024-06-30)..." >&2
jobfull="$(submit_period "2021-05-01" "2024-06-30" "2021_2024_full" "$@")"

# 6. Schedule dependent pooled multi-year summary job after all yearly summaries finish
dep_jobs="${job2021}:${job2022}:${job2023}:${job2024}"
echo -e "\n[Final] Submitting dependent multi-year pooled summary job (afterok:${dep_jobs})..." >&2

pooled_result="$(sbatch --parsable --dependency="afterok:${dep_jobs}" "$@" \
    slurm/bmd_imerg_multiyear_pooled_summary.sbatch)"
pooled_job="${pooled_result%%;*}"

echo "============================================================"
echo " Submitted Multi-Year Evaluation Suite + Pooled Summary!"
echo " Dependent Pooled Summary Job ID: ${pooled_job}"
echo " Markdown Output: data/processed/bmd_imerg_2021_2024_pooled_summary.md"
echo " JSON Output:     data/processed/bmd_imerg_2021_2024_pooled_summary.json"
echo "============================================================"
