#!/bin/bash
#SBATCH --job-name=bdhires-bmd-eval5
#SBATCH --partition=grace-cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out

# Re-evaluate completed five-day dumps against withheld BMD gauges. No GPU and
# no DA rerun are required. Missing sensitivity cases are skipped explicitly.
set -euo pipefail
module purge 2>/dev/null || true

export ENV_PREFIX="/home/afahad/nb/project/BDDA/envs/bdda-gh200"
PYTHON_BIN="$ENV_PREFIX/bin/python"
export PYTHONNOUSERSITE=1
unset PYTHONHOME
unset PYTHONPATH

[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: not found: $PYTHON_BIN"; exit 1; }
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/bdhires-mpl-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" data/processed logs

[[ "$(uname -m)" == "aarch64" ]] || {
    echo "ERROR: This job must run on an aarch64 Grace node."
    exit 1
}

PREFIXES=(
    "bmd_imerg_controlled_20180501_05"
    "bmd_imerg_sensitivity_r2_l75-100"
    "bmd_imerg_sensitivity_r4_l75-100"
    "bmd_imerg_sensitivity_r8_l75-100"
)
EVALUATIONS=()
for prefix in "${PREFIXES[@]}"; do
    dump="data/processed/${prefix}.npz"
    report="data/processed/${prefix}.json"
    if [[ ! -f "$dump" || ! -f "$report" ]]; then
        echo "SKIP $prefix: dump or report is not available"
        continue
    fi
    echo "Re-evaluating $prefix"
    srun --ntasks=1 "$PYTHON_BIN" -u scripts/16_plot_bmd_month_example.py \
        --dump "$dump" \
        --report "$report" \
        --out-evaluation "data/processed/${prefix}_evaluation.json" \
        --out-diagnostics "data/processed/${prefix}_diagnostics.png" \
        --out-events "data/processed/${prefix}_events.png" \
        --out-station-comparison "data/processed/${prefix}_station_comparison.png" \
        --out-spatial "data/processed/${prefix}_spatial.png" \
        --out-intercomparison "data/processed/${prefix}_intercomparison.png"
    EVALUATIONS+=("data/processed/${prefix}_evaluation.json")
done

if (( ${#EVALUATIONS[@]} == 0 )); then
    echo "ERROR: no completed five-day dumps were found"
    exit 1
fi

srun --ntasks=1 "$PYTHON_BIN" -u scripts/19_compare_bmd_5day_sensitivity.py \
    --evaluations "${EVALUATIONS[@]}" \
    --out-json data/processed/bmd_imerg_5day_method_selection.json \
    --out-plot data/processed/bmd_imerg_5day_method_selection.png

echo "Five-day withheld-BMD re-evaluation complete."
echo "Selection metrics: data/processed/bmd_imerg_5day_method_selection.json"
echo "Selection plot: data/processed/bmd_imerg_5day_method_selection.png"
