#!/usr/bin/env bash
#SBATCH --job-name=gfs-joint-iw125-da
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

# Dedicated single-variant DA run and spatial diagnostic generation for gfs_joint_iw125.
# Runs background + gfs_joint_iw125 on Fold 0 in ~15-30 minutes.

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
export MPLCONFIGDIR="${TMPDIR:-/tmp}/gfs-joint-iw125-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" logs

[[ "$(uname -m)" == "aarch64" ]] || {
    echo "ERROR: this job must run on an aarch64 Grace/GH200 node" >&2
    exit 1
}

FOLD="${GRAPHFLOW_DA_FOLD:-0}"
START="2022-05-01"
END="2022-05-10"
OUT_DIR="${OUT_DIR:-runs/graphflow_g0_multimesh/gfs_joint_iw125_da}"
CPC_ROOT="${GRAPHFLOW_CPC_ROOT:-data/processed/v2_simultaneous_refinement/ing2022_s04}"
IMERG="${GRAPHFLOW_IMERG:-data/processed/imerg_prepared_ing2022/imerg_0p4deg_20220501_20220510.nc}"
GRAPH_CKPT="${GRAPHFLOW_CKPT:-runs/prior_h100_cpc_graphflow_g0/best.pt}"
CPC_CKPT="${BMD_CKPT:-runs/prior_h100_cpc_v2/best.pt}"
STATIONS="$CPC_ROOT/fold${FOLD}_bmd.csv"

for required in "$GRAPH_CKPT" "$CPC_CKPT" "$IMERG" "$STATIONS"; do
    [[ -s "$required" ]] || { echo "ERROR: required file missing: $required"; exit 1; }
done

mkdir -p "$OUT_DIR"

echo "Running dedicated GraphFlow gfs_joint_iw125 DA and spatial diagnostics"
echo "  Period: $START to $END (Fold $FOLD)"
echo "  Checkpoint: $GRAPH_CKPT"
echo "  Output Directory: $OUT_DIR"

"$PYTHON_BIN" -u scripts/run_gfs_joint_iw125_da.py \
    --graphflow-ckpt "$GRAPH_CKPT" \
    --unet-ckpt "$CPC_CKPT" \
    --stations "$STATIONS" \
    --imerg "$IMERG" \
    --start "$START" \
    --end "$END" \
    --fold "$FOLD" \
    --members 30 \
    --out-dir "$OUT_DIR"

echo "All spatial diagnostics and summaries completed in $OUT_DIR"
