#!/bin/bash
#SBATCH --job-name=bdhires-osse
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

# Observing-system simulation experiment: pseudo-gauges from CHIRPS, assimilated,
# scored on withheld stations.  See scripts/10_osse.py.
#
# Guided sampling takes the gradient THROUGH the network, so an analysis costs
# roughly 2-3x a background.  Budget accordingly before widening the sweep.

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

[[ "$(uname -m)" == "aarch64" ]] || {
    echo "ERROR: This job must run on an aarch64 Grace node."
    exit 1
}

CKPT="${OSSE_CKPT:-runs/prior_h100_v5/best.pt}"
CONFIG="${OSSE_CONFIG:-configs/da.yaml}"
DAYS="${OSSE_DAYS:-20}"
MEMBERS="${OSSE_MEMBERS:-16}"
MONTH="${OSSE_MONTH:-7}"
NETWORKS="${OSSE_NETWORKS:-10,25,50,100,200,bmd}"
OBS_ERROR="${OSSE_OBS_ERROR:-realistic,perfect}"
WITHHOLD="${OSSE_WITHHOLD:-0.4}"
FIGURE="${OSSE_FIGURE:-data/processed/osse.png}"
REPORT="${OSSE_REPORT:-data/processed/osse.json}"

[[ -f "$CKPT" ]] || { echo "ERROR: checkpoint not found: $CKPT"; exit 1; }

EXTRA=()
if [[ -n "${OSSE_DUMP:-}" ]]; then
    EXTRA+=(--dump "$OSSE_DUMP")
    [[ -n "${OSSE_DUMP_NETWORK:-}" ]] && EXTRA+=(--dump-network "$OSSE_DUMP_NETWORK")
    echo "Dumping raw ensembles to: $OSSE_DUMP"
fi
if [[ "${OSSE_TUNE:-0}" == "1" ]]; then
    EXTRA+=(--tune)
    echo "Mode: hyperparameter tuning"
fi

echo "Checkpoint: $CKPT"
echo "Networks: $NETWORKS   Obs error: $OBS_ERROR"
echo "Days: $DAYS   Members: $MEMBERS   Month: $MONTH   Withhold: $WITHHOLD"

srun --ntasks=1 "$PYTHON_BIN" -u scripts/10_osse.py \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --days "$DAYS" \
    --members "$MEMBERS" \
    --month "$MONTH" \
    --networks "$NETWORKS" \
    --obs-error "$OBS_ERROR" \
    --withhold "$WITHHOLD" \
    "${EXTRA[@]}" \
    --out-figure "$FIGURE" \
    --out-report "$REPORT"

# Verification suite from the dumped ensembles, if one was written.
if [[ -n "${OSSE_DUMP:-}" && -f "${OSSE_DUMP}" ]]; then
    echo
    echo "=============== DA verification suite ==============="
    srun --ntasks=1 "$PYTHON_BIN" -u scripts/11_da_diagnostics.py \
        --dump "$OSSE_DUMP" \
        --out-figure "${OSSE_DIAG_FIGURE:-data/processed/da_diagnostics.png}" \
        --out-report "${OSSE_DIAG_REPORT:-data/processed/da_diagnostics.json}"
fi

# Spatial impact maps from the same dump.
if [[ -n "${OSSE_DUMP:-}" && -f "${OSSE_DUMP}" ]]; then
    echo
    echo "=============== DA impact maps ==============="
    srun --ntasks=1 "$PYTHON_BIN" -u scripts/12_da_impact_maps.py \
        --dump "$OSSE_DUMP" \
        --cases "${OSSE_IMPACT_CASES:-3}" \
        --out-cases "${OSSE_IMPACT_CASES_FIGURE:-data/processed/da_impact_cases.png}" \
        --out-aggregate "${OSSE_IMPACT_FIGURE:-data/processed/da_impact_aggregate.png}" \
        --out-report "${OSSE_IMPACT_REPORT:-data/processed/da_impact.json}"
fi
