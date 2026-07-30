# Diagnosis: why the epoch-119 background looks wrong

Checkpoint `runs/prior_h100/best.pt` — epoch 119, step 51,960, val_loss 0.1574.

> **Status:** items 1, 2, 3, 5 and 6 are implemented. Run order and open items are
> at the bottom, under "What to run next".

## The single most damning number

| case | ERA5 input spatial *r* | model ensemble-mean *r* |
|---|---|---|
| q50 2021-03-10 | −0.03 | 0.04 |
| q90 2024-08-17 | **0.62** | 0.46 |
| q99 2024-05-28 | **0.70** | 0.57 |

The model's output correlates **worse** with CHIRPS than the raw ERA5 field it is
conditioned on. A conditional generator that degrades its own predictor is not a
tuning problem — the conditioning path is weak and the sampler is adding noise on
top. Everything below follows from that.

Secondary symptom, same cause: **+6.4 mm bias on a 1.7 mm day, −6.4 mm on an
18.7 mm day.** Wet bias when dry, dry bias when wet = regression to climatology
plus variance inflation.

---

## Ordered by (expected payoff) / (cost)

### 1. ERA5 `tp` is fed raw and z-scored, while the target is log1p'd
`scripts/04_regrid_and_pack.py:418-435` stacks six raw ERA5 fields;
`scripts/06_compute_stats.py:65-67` z-scores every channel with a single global
mean/std.

Daily `tp` has skewness ~10. After global standardisation ~95% of days sit in a
sliver near −0.4 and monsoon days are +20σ outliers. A single input convolution
cannot extract a usable signal from that distribution — so the network falls back
on climatology, which is exactly what the maps show.

**Fix:** apply the *same* `log1p(p/eps)` transform to `era5_tp` before
standardising (and `sqrt` to `cape`). Recompute `stats.json`. Cheap, no
infrastructure change, largest expected win.

### 2. Sampler settings are inflating and decorrelating the ensemble
`configs/da.yaml:38` `prior_temperature: 1.25`, `n_corrections: 2`.

`src/bdhires/da/sampler.py:167-169` adds `kappa * x0_hat / max(t0, 1e-3)` to the
velocity for all `t >= 0.15`. At t=0.15 that factor is `0.2 / 0.15 = 1.33` — a
velocity perturbation the same order as the velocity itself. On top of that, two
Langevin corrector steps run at every one of the 50 levels.

In log space, inflated variance means the **ensemble mean in mm is biased high by
Jensen's inequality**. +6.4 mm bias with 14 mm spread on a 1.7 mm day is precisely
that signature.

Prior tempering is designed for the *guided* DA runs, where observations pull
members back. It should not be on for the unguided background.

**Fix:** regenerate the panels with `prior_temperature: 1.0`, `n_corrections: 0`,
`noise_scale: 0.0` before drawing any conclusion about the training. Also set
`schedule_power` to ≤ 1.0 — at 2.0, half the 50 steps land in t ∈ [0.75, 1] where
the field is already determined.

### 3. Trained with classifier-free guidance, sampled without it
`configs/train_h100.yaml:29` sets `cond_dropout: 0.10` and
`src/bdhires/models/flow.py:125-127` implements it — but the sampler never
performs a CFG combination. You are paying 10% of the training budget for an
unconditional branch you then throw away.

**Fix:** either set `cond_dropout: 0.0`, or actually use it:
`u = u_uncond + w * (u_cond - u_uncond)` with w ≈ 1.5–3. This is the standard,
direct fix for "the sample doesn't follow the condition."

### 4. Undertrained, and selected on a near-useless metric
38 train years ≈ 13.9k days / batch 32 = 433 steps/epoch × 120 = **52k steps**.
For a conditional flow model trained from scratch that is short; image-domain FM
models typically need 200–500k.

Worse, `best.pt` is chosen by masked flow-matching MSE at random *t*
(`scripts/train.py:192-209`). That loss is dominated by irreducible noise — the
gap between epoch 60 and epoch 119 is inside the sampling noise of the metric, so
"best" is close to arbitrary.

**Fix:** train to ≥ 250k steps, and select checkpoints on **sampled-field CRPS**
over a fixed validation subset, not on the FM loss.

### 5. Ocean fill encodes "moderate rain", not "no rain"
`src/bdhires/data/zarr_dataset.py:141`:

```python
x1 = self.transform.forward(target)[None] * mask[None]
```

`forward(0 mm)` is `-mu/sd`, a negative number — not 0. Multiplying by the mask
sets ocean cells to exactly 0.0 in transformed space, which inverts to
`eps * expm1(mu)` mm, i.e. a moderate rain rate. The loss masks those cells out,
but the 16×16 and 32×32 **global attention blocks mix them into the land field**,
and `sampler.py:197-198` re-imposes the same value at every integration step.

**Fix:** fill with `transform.forward(0.0)` rather than 0.0, and pass the validity
mask as an explicit conditioning channel.

### 6. A large fraction of random crops carry no gradient
`WIDE` spans 16.0–28.8 N (`src/bdhires/grids.py:75`); the southern third is Bay of
Bengal where CHIRPS is entirely NaN. Random 128×128 crops landing there have a
near-empty mask and contribute almost nothing while still consuming a full
forward/backward.

**Fix:** reject crops with valid fraction < 0.3 in `_crop_box`.

---

## Seasonality — yes, it matters, in four separate ways

### (a) The split itself is right; the validation window is too narrow
Contiguous year blocks (train 1981–2018 / val 2019–2020 / test 2021–2025) is the
correct choice — **never** split by random day. Day-to-day autocorrelation plus
CHIRPS's pentad-scale smoothing would leak test information into training.

But val = two years = two monsoon seasons. Checkpoint selection is then hostage to
whichever ENSO/IOD state 2019–2020 happened to be in.

**Better:** scatter the validation years across the record (e.g. 1988, 1996, 2004,
2012, 2018) so validation spans the full range of ENSO/IOD phases and the warming
trend; keep the test block contiguous and last (2021–2025).

Also note 1981–2018 → 2021–2025 spans a real trend in BD extreme rainfall *and*
changes in CHIRPS gauge density over time. Some test-period degradation is
non-stationarity, not model error.

### (b) Severe class imbalance in the training sample
~80% of Bangladesh's annual rainfall falls Jun–Sep. Roughly half of the 13.9k
training days are near-zero dry-season days that are trivially predictable, so
half the gradient budget is spent on the easy half of the problem.

Options, in order of safety:

1. **Season-stratified batching** — fix the monsoon / non-monsoon ratio within
   each batch. Preserves the marginal distribution, so the ensemble stays
   calibrated.
2. **Importance sampling by domain-mean intensity**, with loss reweighting to keep
   the estimator unbiased.
3. Naive oversampling of wet days *without* reweighting shifts the learned prior
   and will miscalibrate the ensemble. Avoid unless you accept that.

### (c) The model is barely told what season it is
The only seasonal signal is `sin/cos(DOY)` as two spatially constant maps
concatenated at the input conv (`zarr_dataset.py:118-123`). Those have to survive
four downsampling levels to reach the bottleneck. Meanwhile the FiLM path — the
`emb` vector that modulates every single ResBlock (`unet.py:45`) — carries **only
the flow time t**.

Two fixes, both high value:

1. Concatenate the DOY embedding onto `emb` so season FiLM-modulates every
   ResBlock. Small change, real effect.
2. **Add per-pixel day-of-year climatology channels** — smoothed climatological
   mean and 90th percentile CHIRPS, computed on training years only. The network
   then learns the *anomaly* rather than the field. In monsoon downscaling this is
   usually the single largest accuracy jump available.

### (d) Diagnose by season, not by three cherry-picked days
Three quantile-selected days tell you almost nothing, and the script itself says
so. Needed: full-test aggregates — CRPS, rank histogram, FSS at multiple scales,
spatial power spectra — each stratified JJAS / pre-monsoon (MAM) / dry (NDJF) and
by intensity bin.

---

## Three more things worth checking

**Verification framing is working against you.** You are scoring a generative
model by the RMSE and correlation of its *ensemble mean*. The mean of a
well-calibrated ensemble is supposed to be smooth and to lose fine-scale variance —
optimising for it will tune you into a deterministic blur, which defeats the point
of the flow model. Show individual members next to the mean, and score with CRPS
plus power spectra.

**There are no baselines.** Add bilinear ERA5, quantile-mapped ERA5, and a plain
deterministic UNet regression. If the flow model does not beat quantile-mapped
ERA5 on CRPS, nothing else in this document matters yet.

**Check the day alignment.** ERA5 `tp` is accumulated 00–24 UTC
(`scripts/00_download_era5.py:14`). Bangladesh is UTC+6 with a pronounced
nocturnal monsoon maximum, and CHIRPS's effective day boundary is not the same.
Cheap test: correlate domain-mean ERA5 tp against CHIRPS at lags −1, 0, +1 days
over the training period. If the maximum is not at lag 0, you are giving away
skill for free.

**The predictor set is thin.** Six surface fields (`tp, tcwv, cape, u10, v10,
msl`), no vertical structure. For monsoon precipitation, 850 hPa moisture flux
(q, u, v), 500 hPa omega, and ERA5's convective / large-scale precipitation split
carry far more information than u10/v10/msl.

---

## Do you need a better model?

**Not yet.** Items 1–3 are hours of work and two of them need no retraining at
all. Fix the conditioning transform, turn off prior tempering for the background,
and either drop or actually use CFG — then re-look at the maps.

If after that plus a ~250k-step run with climatology channels it still
underperforms, the architecture change worth making is **CorrDiff-style two-stage**
(Mardani et al., NVIDIA):

1. a deterministic UNet regression predicting the conditional mean, then
2. a diffusion / flow model over the **residual**.

It is the current standard for km-scale precipitation downscaling and is markedly
more sample-efficient than pure conditional diffusion — which matters a great deal
at ~14k training days, exactly your regime. Your existing `UNet` is reusable for
both stages, and the DA machinery in `src/bdhires/da/` applies unchanged to the
residual flow.

---

## What to run next

### Step 0 — confirm the fixes hold on a machine with torch

```bash
pytest tests/test_conditioning_fixes.py -q     # 20 tests
python scripts/smoke_test.py
```

The 10 numpy-only tests already pass; the 10 torch tests were written but could
not be executed here.

### Step 1 — free diagnostic, no retraining (do this first)

Recompute the statistics, then re-plot the **existing** epoch-119 checkpoint
through the neutral background sampler:

```bash
python scripts/06_compute_stats.py \
    --zarr data/processed/bd_wide.zarr \
    --train-years 1981 2018 --transform log1p \
    --out data/processed/stats_v2.json
```

> ⚠️ **Do not point the old checkpoint at `stats_v2.json`.** It was trained on raw
> z-scored conditioning, so it must keep using the old `stats.json` — the
> conditioning transform changes what the input channels *mean*. Keep both files.
> `CondTransform.from_stats` returns the identity for the old file, so the old
> checkpoint stays exactly reproducible.

To isolate the sampler effect on the old checkpoint, re-plot with the old stats:

```bash
python scripts/08_plot_test_predictions.py \
    --ckpt runs/prior_h100/best.pt --config configs/da.yaml
```

`configs/da.yaml` now carries a `background_sampler` block (T=1.0, no correctors,
`schedule_power=1.0`), which `08_plot_test_predictions.py` picks up
automatically. Set `cfg_scale: 1.0` there for the first run, since CFG changes
behaviour independently — then try 1.5, 2.0, 3.0.

**What to look for:** the +6.4 mm bias and 14 mm spread on the q50 dry day should
largely disappear. If they do, item 2 was the dominant term and the training run
is less broken than the maps suggested.

### Step 2 — the 250k-step retrain

```bash
python scripts/train.py --config configs/train_h100.yaml   # now points at stats_v2.json
```

Update `configs/train_h100.yaml:data.stats` to `data/processed/stats_v2.json`
before launching. 580 epochs ≈ 251k steps at 433 steps/epoch.

### Still open — worth doing before you spend the 250k steps

These were diagnosed but **not** implemented, and each one changes what the
retrain learns. Fixing them after the run means paying for the run twice.

| Item | Where | Effort |
|---|---|---|
| DOY embedding into the FiLM `emb` vector | `models/unet.py`, `data/zarr_dataset.py` | small |
| Per-pixel day-of-year climatology channels | `03_build_static.py` or a new script | medium |
| Scatter the validation years | `configs/train_h100.yaml` | trivial |
| Checkpoint selection on sampled CRPS | `scripts/train.py` | medium |
| Season-stratified batching | new sampler in `scripts/train.py` | medium |
| Full-test evaluation + baselines | `scripts/evaluate.py` | medium |
| ERA5/CHIRPS lag check | one-off script | small |
| Add 850 hPa moisture flux, 500 hPa omega | `00_download_era5.py`, repack | large |

The first three are cheap and high-value. The lag check should be run regardless —
it costs minutes and either confirms the alignment or recovers free skill.
