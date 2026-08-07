# v3 prior — train and test, end to end

Every command runs from the repo root on the GH200 login node.

## What is being fixed, in one paragraph

The v1 prior rains on **64.9%** of cells against CHIRPS's **45.9%**. Raining too
often costs most where true rainfall is lowest, and BMD gauges sit in the dry
half of the domain (CHIRPS ~6.7 mm/day at stations against 12.3 domain-wide), so
it surfaces as **+5.88 mm/day at stations** versus **+2.65 on the grid**. Two
causes: rectified flow is a *continuous* density while CHIRPS has ~54% of its
mass at exactly 0 mm (fixed by the hurdle head), and the residual target
`T(CHIRPS) − T(CPC)` makes "dry" a *moving* target against a base that is wet
over 84% of the domain (fixed by predicting the absolute field).

## The three runs

| tag | config | residual | hurdle | dry_weight | isolates |
|---|---|---|---|---|---|
| `v3` | `train_h100_cpc_v3.yaml` | absolute | yes | 2.0 | both fixes — **the headline run** |
| `v3_hurdle` | `train_h100_cpc_v3_hurdle_only.yaml` | residual | yes | 1.0 | the hurdle head alone |
| `v3_absolute` | `train_h100_cpc_v3_absolute_only.yaml` | absolute | no | 1.0 | the target change alone |

`cond_dropout: 0.1` in all three (v1 had 0.0). Identical across arms so it never
confounds the comparison, and it unlocks METHODOLOGY §3.3's unconditional-prior
ablation, which the v1 checkpoints cannot support at all.

No new stats file is needed. `stats_cpc.json` carries `precip_transform`
(mu = 1.194, sd = 2.179 on CHIRPS), which is what absolute mode uses; the
residual block in it is simply ignored when `residual_override: none`.

---

## Step 0 — unit tests (2 min, login node)

```bash
cd ~/nb/project/BDDA/BDhighresDA
PYTHONPATH=src python -m pytest tests/test_hurdle.py tests/test_method_sweep.py -q
```

Expect all green. `test_hurdle.py` covers the mask, the BCE term, the dry
weighting, and the ocean mask exclusion. These are skipped locally without
torch, so this is their first real execution.

## Step 1 — preflight each config (~5 min each, one GPU)

```bash
bash slurm/submit_preflight_training_cpc_v3_gh200.sh
bash slurm/submit_preflight_training_cpc_v3_absolute_gh200.sh
bash slurm/submit_preflight_training_cpc_v3_hurdle_gh200.sh
```

Runs 2 optimiser steps and writes normalisation diagnostics. Check before
launching anything long:

```bash
python - <<'PY'
import json
for tag in ("v3", "v3_absolute", "v3_hurdle"):
    d = json.load(open(f"data/processed/training_preflight_cpc_{tag}.json"))
    print(f"{tag:14s} {json.dumps(d)[:220]}")
PY
```

**What must be true:** the target standard deviation is near 1 (absolute mode
standardises against CHIRPS, not against the residual, so a value near 0.3 or 3
means the wrong stats file is being read), and for the two hurdle arms the loss
report carries a non-zero `hurdle` component.

## Step 2 — launch training

```bash
bash slurm/submit_train_cpc_v3_gh200.sh
bash slurm/submit_train_cpc_v3_absolute_gh200.sh
bash slurm/submit_train_cpc_v3_hurdle_gh200.sh

squeue -u $USER
```

150 epochs each. Outputs land in `runs/prior_h100_cpc_v3{,_absolute,_hurdle}/`.

## Step 3 — watch the metric that matters

```bash
tail -f logs/bdhires-train-*.out | grep -E "epoch|hurdle|wet_fraction"
```

The training line now prints `FM … coarse … hurdle …`. The one to watch is in
validation:

```bash
python - <<'PY'
import json, glob
for run in sorted(glob.glob("runs/prior_h100_cpc_v3*/validation/*.json"))[-6:]:
    d = json.load(open(run))
    cases = d.get("cases", [])
    if not cases: continue
    pred = sum(c["wet_fraction_pred"] for c in cases)/len(cases)
    obs  = sum(c["wet_fraction_obs"]  for c in cases)/len(cases)
    print(f"{run.split('/')[1]:28s} epoch {d.get('epoch','?'):>4}  "
          f"wet {pred:.3f} vs CHIRPS {obs:.3f}  ({pred-obs:+.3f})")
PY
```

**v1 sat at +0.19 and nobody noticed for 150 epochs**, because CRPS and the
q50/q99 quantiles are nearly blind to rain-no-rain miscounting. Anything inside
±0.05 by epoch 40 means the fix is working. If `v3` is still above +0.15 at
epoch 40, stop it — no amount of further training will close a structural gap,
and the ablations will tell you which half failed.

## Step 4 — re-run the prior audit on the new checkpoints

This is the direct before/after on the defect. Needs one short eval run per
checkpoint to produce dumps:

```bash
for tag in v3 v3_absolute v3_hurdle; do
  python scripts/15_bmd_month_example.py \
      --config configs/da.yaml \
      --ckpt runs/prior_h100_cpc_${tag}/best.pt \
      --stations data/processed/bmd_daily_2024_may_jun.csv \
      --imerg data/processed/bmd_imerg_eval_2024_may_jun/imerg_aligned_20240501_20240630.nc \
      --start 2024-05-01 --end 2024-06-30 \
      --members 16 --holdout-folds 5 --holdout-fold 0 \
      --background-day-offset -1 \
      --out  data/processed/v3_check_${tag}.npz \
      --report data/processed/v3_check_${tag}.json
done

for tag in v3 v3_absolute v3_hurdle; do
  echo "=== $tag ==="
  python scripts/32_prior_wet_bias_audit.py \
      --stats data/processed/stats_cpc.json \
      --dump data/processed/v3_check_${tag}.npz \
      --out-json data/processed/v3_prior_audit_${tag}.json
done
```

**The v1 baseline to beat**, from the same script on 400 days:

| quantity | v1 | target |
|---|--:|--:|
| wet fraction (grid) | 0.649 | 0.459 (CHIRPS) |
| bias vs CHIRPS (grid) | +2.65 | ~0 |
| bias vs CHIRPS (at stations) | +5.88 | ~0 |
| residual correlation | 0.372 | higher is better |

## Step 5 — check the Jensen gap shrank too

```bash
python scripts/31_jensen_bias_audit.py \
    --stats data/processed/stats_cpc.json \
    --observed-bias 10.30 --mean-observed 6.19 \
    --dump data/processed/v3_check_v3.npz \
    --out-json data/processed/v3_jensen.json
```

A drier, better-calibrated prior should have a *smaller* ensemble spread in
transformed space and therefore a smaller mean-vs-median gap. v1's background
gap was **+3.89 mm/day**. This is a secondary benefit, not the objective — do
not tune for it.

## Step 6 — the real comparison, against withheld gauges

```bash
sbatch --export=ALL,BMD_CKPT=runs/prior_h100_cpc_v3/best.pt \
       slurm/real_obs_confidence_may2024.sbatch
```

Produces the OSSE-style impact figure and the withheld-gauge table for the new
prior at three observation-confidence levels. **Read the median columns.** The
mean carries a per-arm Jensen inflation and is not comparable across arms.

## Step 7 — full multi-year evaluation, only if step 4 passed

```bash
BMD_CKPT=runs/prior_h100_cpc_v3/best.pt \
  bash slurm/submit_bmd_imerg_2021_2024_all.sh

python scripts/22_summarize_multiyear_bmd_eval.py   # add --help for its flags
```

---

## Decision gate

Promote a v3 checkpoint over v1 only if **both** hold on withheld gauges:

1. wet-fraction error is inside ±0.05 (v1: +0.19), and
2. median bias at stations improves on v1's +6.40 mm/day without CRPS getting
   worse.

If `v3_hurdle` alone clears the gate, keep the residual formulation — a zero
prediction reproducing CPC is a defensible floor worth retaining. If only
`v3_absolute` clears it, the hurdle head is unnecessary complexity and should be
dropped before it reaches the DA path. Report whichever single change is
sufficient; do not ship the bundle if half of it does the work.

## If something breaks

- **`out_channels` mismatch on resume** — a v1 checkpoint has 1 output channel
  and a hurdle model needs 2. Do not `--resume` across the change; use
  `--init-from`, which loads non-strictly and reports missing keys.
- **`hurdle` loss stays at 0.0** — `train.hurdle.enabled` is not being read;
  confirm with `python -c "import yaml;print(yaml.safe_load(open('configs/train_h100_cpc_v3.yaml'))['train']['hurdle'])"`.
- **wet fraction goes too far and the prior turns dry** — lower
  `train.dry_weight` from 2.0 toward 1.0 before touching `mask_threshold`; the
  weighting is the blunter instrument of the two.
