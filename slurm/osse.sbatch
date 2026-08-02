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

# Combined observing-system simulation experiment. CHIRPS supplies the nature
# truth, exact nested 0.1-degree pseudo-satellite footprints, and a synthetic
# 40-station network. The CPC-conditioned best checkpoint supplies the prior.
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

CKPT="${OSSE_CKPT:-runs/prior_h100_cpc/best.pt}"
CONFIG="${OSSE_CONFIG:-configs/da.yaml}"
DAYS="${OSSE_DAYS:-30}"
MEMBERS="${OSSE_MEMBERS:-16}"
MONTH="${OSSE_MONTH:-7}"
NETWORKS="${OSSE_NETWORKS:-40}"
LAYOUT="${OSSE_LAYOUT:-spread}"
OBS_ERROR="${OSSE_OBS_ERROR:-realistic}"
DUMP_OBS_ERROR="${OSSE_DUMP_OBS_ERROR:-${OBS_ERROR%%,*}}"
WITHHOLD="${OSSE_WITHHOLD:-0.2}"
PSEUDO_SATELLITE="${OSSE_PSEUDO_SATELLITE:-1}"
OBSERVATION_MODE="${OSSE_OBSERVATION_MODE:-}"
SATELLITE_STRIDE="${OSSE_SATELLITE_STRIDE:-1}"
FIGURE="${OSSE_FIGURE:-data/processed/osse_combined.png}"
REPORT="${OSSE_REPORT:-data/processed/osse_combined.json}"
DUMP="${OSSE_DUMP:-data/processed/osse_chirps_satellite_40stations.npz}"

[[ -f "$CKPT" ]] || { echo "ERROR: checkpoint not found: $CKPT"; exit 1; }

EXTRA=(--dump "$DUMP" --dump-obs-error "$DUMP_OBS_ERROR")
[[ -n "${OSSE_DUMP_NETWORK:-}" ]] && EXTRA+=(--dump-network "$OSSE_DUMP_NETWORK")
if [[ "$PSEUDO_SATELLITE" == "1" ]]; then
    EXTRA+=(--pseudo-satellite)
fi
[[ -n "$OBSERVATION_MODE" ]] && EXTRA+=(--observation-mode "$OBSERVATION_MODE")
EXTRA+=(--satellite-stride "$SATELLITE_STRIDE")
if [[ "${OSSE_SATELLITE_CORRELATION_CONTROL:-0}" == "1" ]]; then
    EXTRA+=(--satellite-correlation-control)
fi
echo "Dumping raw ensembles to: $DUMP"
if [[ "${OSSE_TUNE:-0}" == "1" ]]; then
    EXTRA+=(--tune)
    [[ -n "${OSSE_TUNE_NETWORK:-}" ]] && EXTRA+=(--tune-network "$OSSE_TUNE_NETWORK")
    [[ -n "${OSSE_TUNE_DAYS:-}" ]] && EXTRA+=(--tune-days "$OSSE_TUNE_DAYS")
    echo "Mode: hyperparameter tuning"
fi

echo "Checkpoint: $CKPT"
echo "Networks: $NETWORKS   Layout: $LAYOUT   Obs error: $OBS_ERROR"
echo "Dump observation-error case: $DUMP_OBS_ERROR"
echo "Days: $DAYS   Members: $MEMBERS   Month: $MONTH   Withhold: $WITHHOLD"
echo "Pseudo-satellite: $PSEUDO_SATELLITE (factor 2 = exact 0.1-degree footprints)"
echo "Observation mode: ${OBSERVATION_MODE:-legacy}   satellite stride: $SATELLITE_STRIDE"

srun --ntasks=1 "$PYTHON_BIN" -u scripts/10_osse.py \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --days "$DAYS" \
    --members "$MEMBERS" \
    --month "$MONTH" \
    --networks "$NETWORKS" \
    --station-layout "$LAYOUT" \
    --obs-error "$OBS_ERROR" \
    --withhold "$WITHHOLD" \
    "${EXTRA[@]}" \
    --out-figure "$FIGURE" \
    --out-report "$REPORT"

# Verification suite from the dumped ensembles, if one was written.
if [[ -f "$DUMP" ]]; then
    echo
    echo "=============== DA verification suite ==============="
    srun --ntasks=1 "$PYTHON_BIN" -u scripts/11_da_diagnostics.py \
        --dump "$DUMP" \
        --out-figure "${OSSE_DIAG_FIGURE:-data/processed/da_diagnostics.png}" \
        --out-report "${OSSE_DIAG_REPORT:-data/processed/da_diagnostics.json}"
fi

# Spatial impact maps from the same dump.
if [[ -f "$DUMP" ]]; then
    echo
    echo "=============== DA impact maps ==============="
    srun --ntasks=1 "$PYTHON_BIN" -u scripts/12_da_impact_maps.py \
        --dump "$DUMP" \
        --cases "${OSSE_IMPACT_CASES:-3}" \
        --out-cases "${OSSE_IMPACT_CASES_FIGURE:-data/processed/da_impact_cases.png}" \
        --out-aggregate "${OSSE_IMPACT_FIGURE:-data/processed/da_impact_aggregate.png}" \
        --out-report "${OSSE_IMPACT_REPORT:-data/processed/da_impact.json}"
fi

# Reconstruction maps and metric matrices. These explicitly separate the
# observed 0.1-degree footprint skill from unobserved 0.05-degree subgrid skill.
if [[ -f "$DUMP" ]]; then
    echo
    echo "=============== OSSE reconstruction + metric matrices ==============="
    srun --ntasks=1 "$PYTHON_BIN" -u scripts/14_osse_summary.py \
        --dump "$DUMP" \
        --cases "${OSSE_SUMMARY_CASES:-4}" \
        --out-reconstruction "${OSSE_RECONSTRUCTION_FIGURE:-data/processed/osse_reconstruction_maps.png}" \
        --out-matrix "${OSSE_MATRIX_FIGURE:-data/processed/osse_metric_matrix.png}" \
        --out-report "${OSSE_SCALE_REPORT:-data/processed/osse_scale_summary.json}"
fi

# Dedicated downscaling test. This removes every field's own 0.1-degree block
# mean and evaluates only the within-footprint 0.05-degree residual against
# CHIRPS. A coarse-only field is included as the null model.
if [[ -f "$DUMP" ]]; then
    echo
    echo "=============== OSSE 0.05-degree downscaling gate ==============="
    srun --ntasks=1 "$PYTHON_BIN" -u scripts/22_evaluate_osse_downscaling.py \
        --dump "$DUMP" \
        --minimum-valid-fraction "${OSSE_SUBGRID_VALID_FRACTION:-1.0}" \
        --out-figure "${OSSE_DOWNSCALING_FIGURE:-data/processed/osse_downscaling_diagnostics.png}" \
        --out-report "${OSSE_DOWNSCALING_REPORT:-data/processed/osse_downscaling_diagnostics.json}" \
        --out-spatial-data "${OSSE_DOWNSCALING_DATA:-data/processed/osse_downscaling_spatial.nc}"
fi
