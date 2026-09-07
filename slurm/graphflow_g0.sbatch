#!/usr/bin/env bash
#SBATCH --job-name=surma-graphflow-g0
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

# A separate experimental launcher. Default to engineering benchmarks, not training.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
export PYTHONNOUSERSITE=1
unset PYTHONHOME
export PYTHONPATH="$PWD/src"
GRAPHFLOW_PYTHON="${GRAPHFLOW_PYTHON:-/home/afahad/nb/project/BDDA/envs/bdda-gh200/bin/python}"
GRAPHFLOW_ACTION="${GRAPHFLOW_ACTION:-benchmark}"
[[ -x "$GRAPHFLOW_PYTHON" ]] || { echo "Missing Python: $GRAPHFLOW_PYTHON" >&2; exit 1; }
"$GRAPHFLOW_PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA required"; print(torch.cuda.get_device_name())'
case "$GRAPHFLOW_ACTION" in
    benchmark)
        srun "$GRAPHFLOW_PYTHON" scripts/benchmark_graphflow.py --device cuda \
            --batches 1 8 16 30 32 --repeats 5 --warmup 2 \
            --out runs/graphflow_g0_multimesh/benchmark_cuda.json
        ;;
    preflight)
        srun "$GRAPHFLOW_PYTHON" scripts/preflight_training.py \
            --config configs/train_h100_cpc_graphflow_g0.yaml --steps 3 \
            --out runs/graphflow_g0_multimesh/real_data_preflight.json
        ;;
    screen|full)
        srun "$GRAPHFLOW_PYTHON" scripts/graphflow_experiment.py train --stage "$GRAPHFLOW_ACTION"
        ;;
    control-screen)
        srun "$GRAPHFLOW_PYTHON" scripts/graphflow_experiment.py train --stage screen --baseline
        ;;
    *) echo "Unknown GRAPHFLOW_ACTION=$GRAPHFLOW_ACTION" >&2; exit 2 ;;
esac
