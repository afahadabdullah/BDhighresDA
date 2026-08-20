#!/usr/bin/env bash
#SBATCH --job-name=bdhires-v7-repro
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out

# Exact replay of the May 3 V7 diagnostic at data/processed/v7_osse/20260820_1356.
# It consumes the reference run's frozen checkpoints and BMD CSV rather than
# live best.pt files, then hashes both checkpoint pairs and compares all common
# withheld-gauge metrics.

set -euo pipefail
module purge 2>/dev/null || true

export ENV_PREFIX="${ENV_PREFIX:-/home/afahad/nb/project/BDDA/envs/bdda-gh200}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"
export PYTHONNOUSERSITE=1
unset PYTHONHOME
unset PYTHONPATH

[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: ARM Python not found: $PYTHON_BIN" >&2; exit 1; }
cd "${SLURM_SUBMIT_DIR:?submit this job with sbatch from the repository root}"
export PYTHONPATH="$PWD/src"
mkdir -p logs

REFERENCE_ROOT="${V7_REPRO_REFERENCE_ROOT:-data/processed/v7_osse/20260820_1356}"
REFERENCE_JSON="$REFERENCE_ROOT/v7_two_stage_real.json"
OUT="${V7_REPRO_OUT:-data/processed/v7_reproduction_may03/$(date +%Y%m%d_%H%M)}"
SEED="${V7_REPRO_SEED:-20220503}"
MESO_CKPT="$REFERENCE_ROOT/checkpoints/meso_frozen.pt"
ALLOC_CKPT="$REFERENCE_ROOT/checkpoints/allocation_frozen.pt"
DAILY="$REFERENCE_ROOT/bmd_daily.csv"

for required in "$REFERENCE_JSON" "$MESO_CKPT" "$ALLOC_CKPT" "$DAILY" \
                scripts/72_v7_two_stage_osse.py scripts/75_compare_v7_reproduction.py; do
    [[ -f "$required" ]] || { echo "ERROR: missing $required" >&2; exit 1; }
done

# Recover the actual observation file and fixed run parameters from the saved
# reference report, rather than relying on a currently-exported environment.
read -r START END MEMBERS STEPS OBSERVATIONS IMERG R_SWEEP <<< "$(
    "$PYTHON_BIN" -c '
import json, sys
r = json.load(open(sys.argv[1]))
arms = r["arms"]
first = next((arms[k]["days"][0]["date"] for k in arms if arms[k].get("days")), None)
if first is None:
    raise SystemExit("reference report has no evaluated date")
sweep = ",".join(f"{float(v):g}" for _, v in sorted(r.get("arm_imerg_r", {}).items(), key=lambda pair: float(pair[1])))
print(first, first, r["members"], r["n_steps"], r["observations"], r["satellite"], sweep)
' "$REFERENCE_JSON"
)"

[[ "$OBSERVATIONS" == "real" ]] || {
    echo "ERROR: reference is $OBSERVATIONS, not the requested real-BMD May 3 diagnostic" >&2
    exit 1
}
[[ -f "$IMERG" ]] || { echo "ERROR: reference IMERG file is missing: $IMERG" >&2; exit 1; }
[[ -n "$R_SWEEP" ]] || { echo "ERROR: reference report contains no IMERG R sweep" >&2; exit 1; }

# These are the fixed non-sweep arms in the original output.  The R sweep adds
# da_sim_r1,... and script 72 automatically retains background and da_meso.
ARMS="background,da_meso,da_fine,da_both"
mkdir -p "$OUT"

echo "Exact V7 May 3 reproduction"
echo "  reference: $REFERENCE_ROOT"
echo "  date: $START; members: $MEMBERS; steps: $STEPS; seed: $SEED"
echo "  frozen meso: $MESO_CKPT"
echo "  frozen allocation: $ALLOC_CKPT"
echo "  IMERG R sweep: $R_SWEEP"
echo "  output: $OUT"

srun --ntasks=1 "$PYTHON_BIN" -u scripts/72_v7_two_stage_osse.py \
    --meso-checkpoint "$MESO_CKPT" \
    --allocation-checkpoint "$ALLOC_CKPT" \
    --meso-archive data/processed/bd_wide_cpc_0p1.zarr \
    --meso-stats data/processed/stats_v7_meso.json \
    --subgrid-archive data/processed/v7/wide_v7.zarr \
    --stations "$DAILY" --imerg "$IMERG" \
    --start "$START" --end "$END" \
    --members "$MEMBERS" --n-steps "$STEPS" --seed "$SEED" \
    --observations real --min-coverage 0.8 \
    --arms "$ARMS" --imerg-r-sweep "$R_SWEEP" \
    --out "$OUT"

CANDIDATE_JSON="$OUT/v7_two_stage_real.json"
srun --ntasks=1 "$PYTHON_BIN" -u scripts/75_compare_v7_reproduction.py \
    --reference "$REFERENCE_JSON" --candidate "$CANDIDATE_JSON" \
    --out-json "$OUT/reproduction_audit.json" \
    --out-markdown "$OUT/reproduction_audit.md"

echo "Replay complete: $OUT/reproduction_audit.md"
