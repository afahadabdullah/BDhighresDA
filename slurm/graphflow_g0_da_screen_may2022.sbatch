#!/usr/bin/env bash
#SBATCH --job-name=graphflow-g0-da-screen
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out

# Broad GraphFlow-only DA method screen on one fixed May 1--10, 2022 fold.

set -euo pipefail
module purge 2>/dev/null || true

export ENV_PREFIX="${ENV_PREFIX:-/home/afahad/nb/project/BDDA/envs/bdda-gh200}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
unset PYTHONHOME
unset PYTHONPATH

[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: Python not found: $PYTHON_BIN"; exit 1; }
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
export PYTHONPATH="$PWD/src"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/graphflow-g0-da-screen-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" logs

[[ "$(uname -m)" == "aarch64" ]] || {
    echo "ERROR: this job must run on an aarch64 Grace/GH200 node" >&2
    exit 1
}

FOLD="${GRAPHFLOW_DA_SCREEN_FOLD:-0}"
[[ "$FOLD" =~ ^[0-4]$ ]] || { echo "ERROR: fold must be 0..4"; exit 1; }
START="2022-05-01"
END="2022-05-10"
ROOT="${GRAPHFLOW_DA_SCREEN_ROOT:-runs/graphflow_g0_multimesh/da_screen_may2022}"
CPC_ROOT="${GRAPHFLOW_CPC_ROOT:-data/processed/v2_simultaneous_refinement/ing2022_s04}"
IMERG="${GRAPHFLOW_IMERG:-data/processed/imerg_prepared_ing2022/imerg_0p4deg_20220501_20220510.nc}"
GRAPH_CKPT="${GRAPHFLOW_CKPT:-runs/prior_h100_cpc_graphflow_g0/best.pt}"
CPC_CKPT="${BMD_CKPT:-runs/prior_h100_cpc_v2/best.pt}"
STATIONS="$CPC_ROOT/fold${FOLD}_bmd.csv"
PREFIX="$ROOT/graphflow_g0_multimesh/fold${FOLD}"
SUMMARY="$ROOT/fold${FOLD}_screen"

for required in "$GRAPH_CKPT" "$CPC_CKPT" "$IMERG" "$STATIONS"; do
    [[ -s "$required" ]] || { echo "ERROR: required file missing: $required"; exit 1; }
done

if [[ -s "${SUMMARY}.json" && -s "${SUMMARY}.md" && -s "${SUMMARY}.png" ]]; then
    echo "GraphFlow DA screen and summary already complete: $SUMMARY"
    exit 0
fi
if [[ -e "${PREFIX}.npz" && ! -s "${PREFIX}.npz" ]] \
   || [[ -e "${PREFIX}.json" && ! -s "${PREFIX}.json" ]] \
   || [[ -e "${PREFIX}.npz" && ! -e "${PREFIX}.json" ]] \
   || [[ -e "${PREFIX}.json" && ! -e "${PREFIX}.npz" ]]; then
    echo "ERROR: partial DA output at $PREFIX; inspect it before retrying" >&2
    exit 1
fi

if [[ ! -s "${PREFIX}.npz" || ! -s "${PREFIX}.json" ]]; then
    "$PYTHON_BIN" -u scripts/51_check_sqrt_da_gradient.py
    echo "GraphFlow G0 broad DA screen"
    echo "  period: $START through $END; fold $((FOLD + 1))/5; 30 members"
    echo "  49 DA arms plus the common unguided background"
    echo "  categories: authority, temperature, gamma, spread, Huber, two-step EnSRF, stream controls"
    echo "  checkpoint: $GRAPH_CKPT"
    echo "  output: $PREFIX"

    "$PYTHON_BIN" -u scripts/graphflow_experiment.py da \
        --variant-set screen --model graphflow --fold "$FOLD" \
        --unet-ckpt "$CPC_CKPT" --graphflow-ckpt "$GRAPH_CKPT" \
        --stations "$STATIONS" --imerg "$IMERG" \
        --start "$START" --end "$END" --out "$ROOT"
else
    echo "Reusing completed DA dump ${PREFIX}.{npz,json}; rebuilding summary"
fi

"$PYTHON_BIN" -u scripts/29_summarize_method_sweep.py \
    --dump "${PREFIX}.npz" --report "${PREFIX}.json" \
    --baseline gfs_joint_base \
    --block-days 3 --n-resamples "${GRAPHFLOW_DA_SCREEN_BOOTSTRAPS:-10000}" \
    --seed 202209 --title "GraphFlow G0 one-fold DA screen" \
    --out-json "${SUMMARY}.json" \
    --out-markdown "${SUMMARY}.md" \
    --out-plot "${SUMMARY}.png"

echo "GraphFlow DA screen complete: ${SUMMARY}.{md,json,png}"
