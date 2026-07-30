# Runbook: the v2 training run

Everything from `git pull` to a scored checkpoint. Companion to
`docs/DIAGNOSIS_epoch119.md`.

## The three questions up front

**Do I need to retrain from scratch?** Yes. The conditioning transform changes
what the ERA5 input channels *mean*, so the epoch-119 weights are fitted to a
different input distribution. Warm-starting would be worse than useless.
`configs/train_h100.yaml` now writes to `runs/prior_h100_v2`, so this happens
automatically and `runs/prior_h100` survives as the before/after baseline.

**Do I delete the old scripts?** No. Nothing is superseded — the changes are edits
to existing files and are all in git. Do **not** delete `runs/prior_h100/` either;
Step 2 uses it to measure how much of the damage was the sampler alone.

**Do I rerun stats?** Yes — `06_compute_stats.py` only, writing to a **new file**
`stats_v2.json`. You do **not** repack the Zarr: the transform is applied at load
time, so ERA5 stays in raw physical units on disk. That saves a multi-day repack.

> ⚠️ Keep `stats.json` and `stats_v2.json` side by side. The old checkpoint needs
> the old file. `CondTransform.from_stats` returns the identity for statistics
> written before this change, so `runs/prior_h100` stays exactly reproducible.

---

## Step 0 — pull and verify

```bash
cd /panfs/ccds02/nobackup/people/afahad/project/BDDA/BDhighresDA
git pull

export PYTHONPATH="$PWD/src"
ENV_PREFIX="/home/afahad/nb/project/BDDA/envs/bdda-gh200"

"$ENV_PREFIX/bin/python" -m pytest tests/test_conditioning_fixes.py -q
"$ENV_PREFIX/bin/python" scripts/smoke_test.py
```

Expect 26 passing tests. The 10 numpy-only ones already pass; the 16 needing
torch have not been executed anywhere yet, so **this is the first real check** —
do not skip it.

## Step 1 — recompute statistics (~10 min)

```bash
"$ENV_PREFIX/bin/python" scripts/06_compute_stats.py \
    --zarr data/processed/bd_wide.zarr \
    --train-years 1981 2018 \
    --transform log1p \
    --out data/processed/stats_v2.json
```

Or via SLURM, editing `--out` in `slurm/compute_stats.sbatch` first:

```bash
bash slurm/submit_compute_stats.sh
```

**Check it worked:**

```bash
"$ENV_PREFIX/bin/python" -c "
import json
s = json.load(open('data/processed/stats_v2.json'))
print('transform:', s['cond_transform'])
for name, mean, sd in zip(s['cond_channels'], s['cond_mean'], s['cond_std']):
    print(f'  {name:12s} mean={mean:12.4f} sd={sd:12.4f}')
"
```

`cond_transform.kinds` must read `['log1p','none','sqrt','none','none','none']`.
`era5_tp` mean/sd should now be O(1) rather than O(0.001) with a huge spread.

## Step 2 — the free diagnostic, before spending any GPU-days

Re-plot the **old** checkpoint through the neutral background sampler. This
isolates how much of the bad figure was sampler settings versus training.

```bash
"$ENV_PREFIX/bin/python" scripts/08_plot_test_predictions.py \
    --ckpt runs/prior_h100/best.pt \
    --config configs/da.yaml \
    --out-figure data/processed/test_prediction_maps_neutral.png \
    --out-metrics-figure data/processed/test_prediction_metrics_neutral.png \
    --out-report data/processed/test_prediction_report_neutral.json
```

`configs/da.yaml` now has a `background_sampler` block (T=1.0, no correctors,
`schedule_power=1.0`) which this script picks up automatically.

For a clean comparison set `background_sampler.cfg_scale: 1.0` first — CFG is a
separate change. Then rerun at 1.5, 2.0, 3.0 and watch `spatial_correlation`.

**What to look for:** the q50 case had `bias +6.37 mm` on a `1.74 mm` day with
`14.2 mm` spread. Most of that should vanish. If it does, item 2 was the dominant
term.

> This step uses the OLD `stats.json`, which `configs/da.yaml` still points at.
> Leave it that way until the v2 checkpoint exists.

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
runs/prior_h100_v2/validation/
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

Metric history without opening images:

```bash
"$ENV_PREFIX/bin/python" -c "
import json
for line in open('runs/prior_h100_v2/validation/history.jsonl'):
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

```bash
"$ENV_PREFIX/bin/python" scripts/08_plot_test_predictions.py \
    --ckpt runs/prior_h100_v2/best.pt \
    --config configs/da.yaml
```

First update `configs/da.yaml:data.stats` to `data/processed/stats_v2.json` — the
v2 checkpoint needs the v2 conditioning. Revert it if you re-score the old run.

---

## Config changes at a glance

| File | Key | Was | Now |
|---|---|---|---|
| `train_h100.yaml` | `data.stats` | `stats.json` | `stats_v2.json` |
| | `train.out_dir` | `runs/prior_h100` | `runs/prior_h100_v2` |
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
