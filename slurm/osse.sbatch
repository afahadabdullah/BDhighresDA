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
MONTHS="${OSSE_MONTHS:-}"
START="${OSSE_START:-}"
END="${OSSE_END:-}"
NETWORKS="${OSSE_NETWORKS:-40}"
BMD_STATIONS="${OSSE_BMD_STATIONS:-data/bmd/Stations.csv}"
LAYOUT="${OSSE_LAYOUT:-spread}"
HOLDOUT_LAYOUT="${OSSE_HOLDOUT_LAYOUT:-spread}"
OBS_ERROR="${OSSE_OBS_ERROR:-exact}"
DUMP_OBS_ERROR="${OSSE_DUMP_OBS_ERROR:-${OBS_ERROR%%,*}}"
WITHHOLD="${OSSE_WITHHOLD:-0.2}"
PSEUDO_SATELLITE="${OSSE_PSEUDO_SATELLITE:-1}"
OBSERVATION_MODE="${OSSE_OBSERVATION_MODE:-}"
SATELLITE_STRIDE="${OSSE_SATELLITE_STRIDE:-1}"
OUT_DIR="${OSSE_OUT_DIR:-}"
if [[ -n "$OUT_DIR" ]]; then
    mkdir -p "$OUT_DIR"
    FIGURE="${OSSE_FIGURE:-$OUT_DIR/osse_overview.png}"
    REPORT="${OSSE_REPORT:-$OUT_DIR/osse_metrics.json}"
    DUMP="${OSSE_DUMP:-$OUT_DIR/ensemble.npz}"
    DIAG_FIGURE="${OSSE_DIAG_FIGURE:-$OUT_DIR/da_diagnostics.png}"
    DIAG_REPORT="${OSSE_DIAG_REPORT:-$OUT_DIR/da_diagnostics.json}"
    IMPACT_CASES_FIGURE="${OSSE_IMPACT_CASES_FIGURE:-$OUT_DIR/da_impact_cases.png}"
    IMPACT_FIGURE="${OSSE_IMPACT_FIGURE:-$OUT_DIR/da_impact_aggregate.png}"
    IMPACT_REPORT="${OSSE_IMPACT_REPORT:-$OUT_DIR/da_impact.json}"
    RECONSTRUCTION_FIGURE="${OSSE_RECONSTRUCTION_FIGURE:-$OUT_DIR/osse_reconstruction.png}"
    MATRIX_FIGURE="${OSSE_MATRIX_FIGURE:-$OUT_DIR/osse_metric_matrix.png}"
    SCALE_REPORT="${OSSE_SCALE_REPORT:-$OUT_DIR/osse_scale_summary.json}"
    DOWNSCALING_FIGURE="${OSSE_DOWNSCALING_FIGURE:-$OUT_DIR/osse_downscaling.png}"
    DOWNSCALING_REPORT="${OSSE_DOWNSCALING_REPORT:-$OUT_DIR/osse_downscaling.json}"
    DOWNSCALING_DATA="${OSSE_DOWNSCALING_DATA:-$OUT_DIR/osse_downscaling_spatial.nc}"
    DOWNSCALING_CURVES="${OSSE_DOWNSCALING_CURVES:-$OUT_DIR/osse_downscaling_curves.npz}"
else
    FIGURE="${OSSE_FIGURE:-data/processed/osse_combined.png}"
    REPORT="${OSSE_REPORT:-data/processed/osse_combined.json}"
    DUMP="${OSSE_DUMP:-data/processed/osse_chirps_satellite_40stations.npz}"
    DIAG_FIGURE="${OSSE_DIAG_FIGURE:-data/processed/da_diagnostics.png}"
    DIAG_REPORT="${OSSE_DIAG_REPORT:-data/processed/da_diagnostics.json}"
    IMPACT_CASES_FIGURE="${OSSE_IMPACT_CASES_FIGURE:-data/processed/da_impact_cases.png}"
    IMPACT_FIGURE="${OSSE_IMPACT_FIGURE:-data/processed/da_impact_aggregate.png}"
    IMPACT_REPORT="${OSSE_IMPACT_REPORT:-data/processed/da_impact.json}"
    RECONSTRUCTION_FIGURE="${OSSE_RECONSTRUCTION_FIGURE:-data/processed/osse_reconstruction_maps.png}"
    MATRIX_FIGURE="${OSSE_MATRIX_FIGURE:-data/processed/osse_metric_matrix.png}"
    SCALE_REPORT="${OSSE_SCALE_REPORT:-data/processed/osse_scale_summary.json}"
    DOWNSCALING_FIGURE="${OSSE_DOWNSCALING_FIGURE:-data/processed/osse_downscaling_diagnostics.png}"
    DOWNSCALING_REPORT="${OSSE_DOWNSCALING_REPORT:-data/processed/osse_downscaling_diagnostics.json}"
    DOWNSCALING_DATA="${OSSE_DOWNSCALING_DATA:-data/processed/osse_downscaling_spatial.nc}"
    DOWNSCALING_CURVES="${OSSE_DOWNSCALING_CURVES:-data/processed/osse_downscaling_curves.npz}"
fi

[[ -f "$CKPT" ]] || { echo "ERROR: checkpoint not found: $CKPT"; exit 1; }

EXTRA=(--dump "$DUMP" --dump-obs-error "$DUMP_OBS_ERROR")
if [[ -n "$MONTHS" ]]; then
    EXTRA+=(--months "$MONTHS")
else
    EXTRA+=(--month "$MONTH")
fi
[[ -n "$START" ]] && EXTRA+=(--start "$START")
[[ -n "$END" ]] && EXTRA+=(--end "$END")
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
echo "BMD catalogue: $BMD_STATIONS   Holdout layout: $HOLDOUT_LAYOUT"
echo "Dump observation-error case: $DUMP_OBS_ERROR"
echo "Days: $DAYS   Members: $MEMBERS   Month(s): ${MONTHS:-$MONTH}   Withhold: $WITHHOLD"
echo "Date bounds: ${START:-checkpoint test start} to ${END:-checkpoint test end}"
echo "Pseudo-satellite: $PSEUDO_SATELLITE (factor 2 = exact 0.1-degree footprints)"
echo "Observation mode: ${OBSERVATION_MODE:-legacy}   satellite stride: $SATELLITE_STRIDE"

srun --ntasks=1 "$PYTHON_BIN" -u scripts/10_osse.py \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --days "$DAYS" \
    --members "$MEMBERS" \
    --networks "$NETWORKS" \
    --bmd-stations "$BMD_STATIONS" \
    --station-layout "$LAYOUT" \
    --holdout-layout "$HOLDOUT_LAYOUT" \
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
        --out-figure "$DIAG_FIGURE" \
        --out-report "$DIAG_REPORT"
fi

# Spatial impact maps from the same dump.
if [[ -f "$DUMP" ]]; then
    echo
    echo "=============== DA impact maps ==============="
    srun --ntasks=1 "$PYTHON_BIN" -u scripts/12_da_impact_maps.py \
        --dump "$DUMP" \
        --cases "${OSSE_IMPACT_CASES:-3}" \
        --out-cases "$IMPACT_CASES_FIGURE" \
        --out-aggregate "$IMPACT_FIGURE" \
        --out-report "$IMPACT_REPORT"
fi

# Reconstruction maps and metric matrices. These explicitly separate the
# observed 0.1-degree footprint skill from unobserved 0.05-degree subgrid skill.
if [[ -f "$DUMP" ]]; then
    echo
    echo "=============== OSSE reconstruction + metric matrices ==============="
    srun --ntasks=1 "$PYTHON_BIN" -u scripts/14_osse_summary.py \
        --dump "$DUMP" \
        --cases "${OSSE_SUMMARY_CASES:-4}" \
        --out-reconstruction "$RECONSTRUCTION_FIGURE" \
        --out-matrix "$MATRIX_FIGURE" \
        --out-report "$SCALE_REPORT"
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
        --out-figure "$DOWNSCALING_FIGURE" \
        --out-report "$DOWNSCALING_REPORT" \
        --out-spatial-data "$DOWNSCALING_DATA" \
        --out-curve-data "$DOWNSCALING_CURVES"
fi
