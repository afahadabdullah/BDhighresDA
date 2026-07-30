# Runbook: the v3 training run (residual target)

Everything from `git pull` to a scored checkpoint. Companion to
`docs/DIAGNOSIS_epoch119.md`.

## The three questions up front

**Do I need to retrain from scratch?** Yes. v3 changes both what the ERA5 input
channels *mean* (conditioning transform) and what the network *predicts* (the
residual target). No previous checkpoint is compatible with either.
`configs/train_h100.yaml` now writes to `runs/prior_h100_v3`, so this happens
automatically and `runs/prior_h100` survives as the before/after baseline.

**Do I delete the old scripts?** No. Nothing is superseded — the changes are edits
to existing files and are all in git. Do **not** delete `runs/prior_h100/` either;
Step 2 uses it to measure how much of the damage was the sampler alone.

**Do I rerun stats?** Yes — `06_compute_stats.py` only, writing to a **new file**
`stats_v3.json`. You do **not** repack the Zarr: the transform is applied at load
time, so ERA5 stays in raw physical units on disk. That saves a multi-day repack.

> ⚠️ Keep `stats.json` and `stats_v3.json` side by side. The old checkpoint needs
> the old file. `CondTransform.from_stats` returns the identity for statistics
> written before this change, so `runs/prior_h100` stays exactly reproducible.

---

## Architecture note — read this before running anything

`bdda-gh200` is an **aarch64** build and lives on the `grace` partition.
`gpulogin1` is **x86_64**. Running the env's interpreter on the login node fails
with:

```
cannot execute binary file: Exec format error
```

So every step that needs torch must go through SLURM to a Grace node, even
though the test suite itself is CPU-only. Steps 1 and 2 below need torch and/or
the packed store, so they are batch jobs too.

## Step 0 — pull and verify

```bash
cd /panfs/ccds02/nobackup/people/afahad/project/BDDA/BDhighresDA
git pull

bash slurm/submit_run_tests_gh200.sh
```

Then watch it:

```bash
squeue -u "$USER"
tail -f logs/bdhires-tests-*.out
```

Expect **52 passed, 15 skipped** from `tests/test_conditioning_fixes.py` and
`tests/test_progress_and_summary.py`. The skips are torch-dependent tests that
cannot run in the authoring environment, so on the cluster they should all
execute — **this is the first real check** of the residual code path.

The job also runs the full suite and the smoke test. Full-suite failures in
`test_era5_earthmover.py` / `test_pack_alignment.py` / `test_dem_download.py`
are pre-existing dependency gaps, not regressions; the v3 suite result is the
one that matters.

If you prefer an interactive shell:

```bash
srun --partition=grace --gres=gpu:1 --cpus-per-task=8 --mem=32G \
     --time=00:30:00 --pty bash
# then, on the compute node:
cd "$SLURM_SUBMIT_DIR" 2>/dev/null || cd /panfs/ccds02/nobackup/people/afahad/project/BDDA/BDhighresDA
export PYTHONPATH="$PWD/src" PYTHONNOUSERSITE=1
/home/afahad/nb/project/BDDA/envs/bdda-gh200/bin/python -m pytest tests/test_conditioning_fixes.py -q
```

## The provenance chain — why Steps 1a-1c all have to run

The repo enforces a checksum chain, and each link pins `git rev-parse HEAD`:

```
stats_v3.json
   └─> normalization_diagnostics.{json,png}   pins stats sha256 + git HEAD
          └─> training_preflight.json         pins normalization sha256 + git HEAD
                 └─> train_h100.sbatch        pins preflight sha256 + git HEAD
```

So any commit invalidates everything downstream of it. Submitting training after
a code change fails with:

```
AssertionError: repository changed after preflight; rerun the GH200 preflight
```

That is the guard working, not a bug — it is refusing to train on a repo that
differs from the one that was validated.

**Two consequences:**

1. Run Steps 1a → 1c → 3 in order, and **do not commit anything between 1b and
   3**. If you do, rerun from 1b.
2. The chain must be rebuilt against `stats_v3.json`, not the old `stats.json` —
   the normalization diagnostics hash the statistics file they were given.

## Step 1 — recompute statistics (~10 min)

`compute_stats.sbatch` already parameterises the output path, so no edit needed:

```bash
STATS_OUT=data/processed/stats_v3.json \
STATS_RESIDUAL=1 \
    bash slurm/submit_compute_stats.sh
tail -f logs/*compute*stats*.out
```

`STATS_RESIDUAL=1` switches the target to `T(CHIRPS) - T(ERA5_tp)` in
transformed space. The job prints a `residual target:` block; check
`encoded_std` is near 1 and `base_correlation` is sensible (that is how much of
CHIRPS the ERA5 base already explains).

**Check it worked** (pure stdlib, so the login node is fine here):

```bash
python -c "
import json
s = json.load(open('data/processed/stats_v3.json'))
print('transform:', s['cond_transform'])
for name, mean, sd in zip(s['cond_channels'], s['cond_mean'], s['cond_std']):
    print(f'  {name:12s} mean={mean:12.4f} sd={sd:12.4f}')
"
```

`cond_transform.kinds` must read `['log1p','none','sqrt','none','none','none']`.
`era5_tp` mean/sd should now be O(1) rather than O(0.001) with a huge spread.

## Step 1b — normalization diagnostics against the new statistics

```bash
NORM_STATS=data/processed/stats_v3.json \
NORM_FIGURE=data/processed/normalization_diagnostics_v3.png \
NORM_REPORT=data/processed/normalization_diagnostics_v3.json \
    bash slurm/submit_normalization_diagnostics.sh
```

This is where you get the first direct look at whether the conditioning fix
worked: the ERA5 `tp` histogram should now be roughly symmetric instead of a
spike at zero with a 20-sigma tail.

## Step 1c — rerun the training preflight

```bash
PREFLIGHT_NORM_REPORT=data/processed/normalization_diagnostics_v3.json \
    bash slurm/submit_preflight_training_gh200.sh
```

The preflight now reads the statistics path from the config instead of
hardcoding `stats.json`, so it validates the file training will actually load.
It runs two real optimiser steps on real data and records the checksums the
training job checks.

Confirm it passed before moving on:

```bash
python -c "
import json; r = json.load(open('data/processed/training_preflight.json'))
print('passed:', r['passed'], '| commit:', r['git_commit'][:8],
      '| val loss:', r['validation_loss'])
"
git rev-parse --short HEAD     # must match the commit above
```

## Step 2 — the free diagnostic, before spending any GPU-days

Re-plot the **old** checkpoint through the neutral background sampler. This
isolates how much of the bad figure was sampler settings versus training.

```bash
TEST_CKPT=runs/prior_h100/best.pt \
TEST_MAP_FIGURE=data/processed/test_prediction_maps_neutral.png \
TEST_METRICS_FIGURE=data/processed/test_prediction_metrics_neutral.png \
TEST_CASE_DIR=data/processed/test_prediction_cases_neutral \
TEST_REPORT=data/processed/test_prediction_report_neutral.json \
    bash slurm/submit_test_predictions.sh
```

(`test_predictions.sbatch` now takes `TEST_CKPT` / `TEST_CONFIG` instead of
hardcoding `runs/prior_h100/best.pt`.)

`configs/da.yaml` now has a `background_sampler` block (T=1.0, no correctors,
`schedule_power=1.0`) which this script picks up automatically.

For a clean comparison set `background_sampler.cfg_scale: 1.0` first — CFG is a
separate change. Then rerun at 1.5, 2.0, 3.0 and watch `spatial_correlation`.

**What to look for:** the q50 case had `bias +6.37 mm` on a `1.74 mm` day with
`14.2 mm` spread. Most of that should vanish. If it does, item 2 was the dominant
term.

> This step uses the OLD `stats.json`, which `configs/da.yaml` still points at.
> Leave it that way until the v3 checkpoint exists.

## Step 3 — launch the 250k-step run

```bash
mkdir -p logs
RESUME_IF_AVAILABLE=0 bash slurm/submit_train_gh200.sh
```

`slurm/train_h100.sbatch` now reads `out_dir` from the config instead of
hardcoding `runs/prior_h100`, and refuses to resume a checkpoint whose recorded
`data.stats` differs from the config's — so a stale `last.pt` can no longer be
silently resumed into a run with different input semantics.

To resume after a timeout, just resubmit without the flag:

```bash
bash slurm/submit_train_gh200.sh
```

580 epochs ≈ 251k steps at 433 steps/epoch.

## Step 4 — watch it improve

The sampled validation runs on two fixed held-out **July** days — a typical one
(q50) and a wet extreme (q99) — first at 10 completed epochs, then every 5th.
Same dates every time, so the curves are directly comparable.

```
runs/prior_h100_v3/validation/
  history.jsonl      one JSON row per evaluation, append-only
  epoch_0010.png     map panel: ERA5 | CHIRPS | ens mean | single member | error
  epoch_0015.png
  ...
  progress.png       CRPS / RMSE / bias / correlation / spread / coverage vs epoch
```

Live:

```bash
tail -f logs/bdhires-gh200-*.out | grep -E "sampled validation|new best"
```

Metric history without opening images (stdlib only, so the login node is fine):

```bash
python -c "
import json
for line in open('runs/prior_h100_v3/validation/history.jsonl'):
    r = json.loads(line)
    print(f\"epoch {r['epoch']:4d}  CRPS {r['mean_crps_mm']:6.3f}  \" +
          '  '.join(f\"{c['date']}: bias {c['bias_mm']:+6.2f} r {c['spatial_correlation']:.2f}\"
                    for c in r['cases']))
"
```

**Reading `progress.png`:**

| Panel | Healthy | Trouble |
|---|---|---|
| CRPS | falls, then flattens | rises after a minimum → overfitting |
| Bias | converges toward 0 | stuck positive on the q50 day → still climatology |
| Spatial correlation | rises above the ERA5 input's own r | plateaus below it → conditioning still weak |
| Spread | settles | collapsing toward 0 → CFG weight too high |
| 90% coverage | approaches 0.90 | far below → under-dispersive |

The correlation panel is the one that matters most: the epoch-119 failure was the
model scoring *below* its own input (0.46 vs 0.62). Crossing that line is the
first real sign the conditioning fix worked.

`best.pt` is now selected on **sampled CRPS**, not the flow-matching loss. The
checkpoint records `selected_by`, `crps` and `best_crps`.

Cost: ~1,920 network evaluations at 128×128 per check, ×115 checks. Tens of
seconds each on an H100. To trim, lower `validation.members` or `n_steps`; to
disable, set `validation.enabled: false`.

## Step 5 — score the new checkpoint

First update `configs/da.yaml:data.stats` to `data/processed/stats_v3.json` — the
v2 checkpoint needs the v2 conditioning. Revert it if you re-score the old run.

```bash
TEST_CKPT=runs/prior_h100_v3/best.pt \
TEST_MAP_FIGURE=data/processed/test_prediction_maps_v3.png \
TEST_METRICS_FIGURE=data/processed/test_prediction_metrics_v3.png \
TEST_CASE_DIR=data/processed/test_prediction_cases_v3 \
TEST_REPORT=data/processed/test_prediction_report_v3.json \
    bash slurm/submit_test_predictions.sh
```

---

## Config changes at a glance

| File | Key | Was | Now |
|---|---|---|---|
| `train_h100.yaml` | `data.stats` | `stats.json` | `stats_v3.json` |
| | `train.out_dir` | `runs/prior_h100` | `runs/prior_h100_v3` |
| | `train.epochs` | 120 | 580 |
| | `train.ckpt_every` | 5 | 5 (must divide `validation.every`) |
| | `data.min_valid_fraction` | — | 0.3 |
| | `validation.*` | — | new block |
| `da.yaml` | `sampler.schedule_power` | 2.0 | 1.0 |
| | `background_sampler` | — | new block, T=1.0, w=2.0 |

## Still not implemented

From the diagnosis, and each one changes what the retrain learns — worth doing
**before** Step 3, not after:

- DOY embedding into the FiLM `emb` vector (`models/unet.py`)
- Per-pixel day-of-year climatology channels
- Scattered validation years instead of 2019–2020
- Season-stratified batching
- Full-test evaluation harness with baselines (`scripts/evaluate.py`)
- ERA5/CHIRPS day-lag check
- 850 hPa moisture flux and 500 hPa omega predictors (needs a repack)

The first three are cheap. The lag check costs minutes and should be run
regardless.
