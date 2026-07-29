#!/bin/bash
#SBATCH --job-name=bdhires-B
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module purge
module load cuda/12.1 || true
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate bdhires

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

export OMP_NUM_THREADS=6
export NCCL_DEBUG=WARN
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

# Stage A (ERA5-only, 1981-2000) then Stage B (ERA5+IMERG, 2001-2018).
if [ ! -f runs/stageA/final.pt ]; then
  torchrun --standalone --nproc_per_node=2 scripts/train.py --config configs/train_stageA.yaml
fi
torchrun --standalone --nproc_per_node=2 scripts/train.py \
    --config configs/train_v100.yaml --init-from runs/stageA/final.pt
