# Ingesting IMERG and gauges better — plan and decision gates

Written against the pooled 2021–2024 result (19,485 withheld station-days).
Ordered by what would change the conclusion, not by effort.

---

## 1. What the pooled result actually says

| | CRPS | RMSE | **bias** | **unbiased RMSE** | corr |
|---|---|---|---|---|---|
| Background | 9.456 | 22.94 | **+10.30** | 20.50 | 0.520 |
| Gauges only | 6.259 | 17.75 | −0.65 | 17.73 | 0.666 |
| IMERG only | 8.041 | 22.26 | **+9.88** | 19.95 | 0.593 |
| Simultaneous | 6.097 | 17.27 | +1.58 | 17.20 | 0.686 |

Three facts drive everything below.

1. **The background is wet by +10.3 mm/day** — 62% of its MAE and 20% of its
   MSE is a pure offset. Most of the headline "35% CRPS improvement" is the DA
   cleaning up after the prior, not the DA extracting information.
2. **IMERG removes 4% of that bias.** Gauges remove 106%. With the bias taken
   out, IMERG-only improves random error by 2.7% against 13% for gauges. As a
   standalone product the satellite arm is doing almost nothing.
3. **Fusion buys +2.6% CRPS and costs +7% MAE and 2.4× absolute bias.** The one
   clean, monotone IMERG win is correlation (0.666 → 0.686, and simultaneous >
   gauges-only in all four years independently). That is the honest framing:
   *the satellite constrains pattern, not amplitude.*

The four candidate causes, all of which the sweep separates:

- **IMERG is never bias-corrected.** `configs/da.yaml` has
  `bias_correction: null`; `slurm/bmd_imerg_example.sbatch` prints
  "native IMERG V07B is not bias-corrected". `METHODOLOGY.md` §4.4 calls
  skipping this "the main way this design goes wrong."
- **The likelihood is near-silent.** `error_corr_cells=3.0` with `stride=3`
  gives an R inflation of 2π·(3/3)² = **6.28×**, on top of stride-3 thinning
  that discards 8 of every 9 footprints. Effective σ_IMERG ≈ 0.91 against
  σ_gauge ≈ 0.27.
- **The arms are not controlled for tempering.** Background runs at
  `prior_temperature: 1.0`, all analysis arms at `1.25`. Inverting a convex
  transform over a broader ensemble lifts the mm-space *mean* without moving the
  median — the config comment already records T=1.25 causing "+6.4 mm bias on a
  1.7 mm day". Some unknown share of +9.88 may be this, not the satellite.
- **Joint guidance leaves station bullseyes.** Visible in the wet-day-frequency
  row of the intercomparison figure, in `Gauges only` and `Simultaneous` but
  never in `IMERG only`. They are *drying* discs on a near-saturated background,
  and they survive a 61-day mean — which is only possible if every gauge pulls
  the same direction on nearly every day. `METHODOLOGY.md` §6.4 predicted them.

---

## 2. Two diagnostics that come before the sweep

Neither needs a GPU. Both can invalidate the framing above.

**(a) Is the background biased against CHIRPS, or is CHIRPS biased against BMD?**
The prior's training target is CHIRPS. If the model reproduces CHIRPS and CHIRPS
is +10 mm against BMD gauges, this is a target/window problem — plausibly the
CHIRPS 00–00 UTC versus BMD 03–03 UTC window — and a data-prep fix. If the model
is +10 against CHIRPS itself, the prior needs retraining. Row 1 of the
intercomparison figure suggests the latter (Background is visibly wetter than
CHIRPS domain-wide) but it needs to be a number. **This decides whether the
paper's problem is DA or the prior.**

**(b) Why is `BACKGROUND_DAY_OFFSET=-1`?**
The background is conditioned on the *previous day's* ERA5/CPC, which is
consistent with the 0.520 correlation. The training split is
`train [1981,2018]`, `val [2019,2020]`, `test [2021,2025]`, so 2021–24 is
genuinely out of sample and no leakage precaution is needed. If the offset was a
precaution, rerunning at offset 0 is the cheapest available skill gain in the
whole project. The sweep exposes `--background-day-offset`; run the core group
at both.

---

## 2b. Matching the months across years

The first leave-one-year-out fit degraded 2024 (bias −0.545 → −1.101) while
improving 2021 and 2023. The cause is a stratification mismatch, not the
correction itself: the season bins are `MAM = (3,4,5)` and `JJAS = (6,7,8,9)`,
so **May and June fall in different bins**. 2021–23 run May–September, 2024 runs
May–June. The JJAS map applied to 2024's June was therefore fitted almost
entirely on July–August peak-monsoon intensities.

Two knobs now fix this, and a synthetic reproduction confirms the mechanism.
With a deliberately month-dependent bias, the 2024 residual bias is:

| Configuration | 2024 residual bias | May | June |
|---|--:|--:|--:|
| season bins, full spans (original) | +0.077 | −0.004 | **+0.161** |
| season bins, `--months 5 6` | +0.003 | — | — |
| `--season-mode month --months 5 6` | +0.003 | −0.004 | −0.002 |

The per-stratum breakdown localises the error exactly where predicted: May is
already clean, June carries all of it.

```bash
python scripts/27_fit_imerg_bias_correction.py --imerg <the four .nc> \
    --zarr data/processed/bd_wide_cpc.zarr --stats data/processed/stats_cpc.json \
    --grid bd --pool 5 --fit-error-model \
    --months 5 6 --season-mode month \
    --out data/processed/imerg_qm_loyo_mayjun.npz
```

**An important negative result, worth knowing before over-tuning this.** A
*monotone* bias is recovered by per-cell quantile mapping regardless of how
months are pooled — pooling changes the density of quantile knots, not the shape
of the recovered transfer function. Stratification only matters when the bias
*relationship itself* differs between months, which is physically plausible here
(warm-rain onset versus deep monsoon convection are different retrieval regimes)
but is not automatic. Read the new per-stratum lines in the fit log: if May and
June show similar raw-to-corrected behaviour, the stratification is not your
problem and finer binning will buy nothing.

Note also that **the sweep window is unaffected either way.** It runs May 1–5
2024, entirely inside MAM/M05, whose map was already fitted on May data from the
other three years. The reported 2024 degradation is a 61-day May+June statistic;
only the June half was contaminated.

## 3. The sweep

```bash
# fit the leave-one-year-out IMERG correction once (CPU, minutes)
python scripts/27_fit_imerg_bias_correction.py \
    --imerg data/processed/bmd_imerg_eval_2021_may_sep/imerg_aligned_20210501_20210930.nc \
            data/processed/bmd_imerg_eval_2022_may_sep/imerg_aligned_20220501_20220930.nc \
            data/processed/bmd_imerg_eval_2023_may_sep/imerg_aligned_20230501_20230930.nc \
            data/processed/bmd_imerg_eval_2024_may_jun/imerg_aligned_20240501_20240630.nc \
    --zarr data/processed/bd_wide_cpc.zarr --stats data/processed/stats_cpc.json \
    --grid bd --pool 5 --fit-error-model --out data/processed/imerg_qm_loyo.npz

# five-day screening sweep (one GPU)
bash slurm/submit_simultaneous_sweep_may2024.sh
SWEEP_GROUP=tempering bash slurm/submit_simultaneous_sweep_may2024.sh
SWEEP_GROUP=weighting bash slurm/submit_simultaneous_sweep_may2024.sh
```

`python scripts/28_simultaneous_method_sweep.py --list-variants` prints the
catalogue. Every arm shares the checkpoint, conditioning, holdout fold, prior
noise seed and observation perturbation seeds, so any difference between two
arms is the method and nothing else.

### What each group tests, and what would promote it

| Group | Hypothesis | Promote if |
|---|---|---|
| `core` | The two structural fixes — bias-corrected IMERG, and gauges applied by EnSRF after IMERG rather than jointly — beat production fusion. | Analysis bias moves toward zero **and** locality ratio drops toward 1 without CRPS getting worse. |
| `tempering` | A large share of the +9.88 mm IMERG-only bias is Jensen inflation from `prior_temperature: 1.25`, not the satellite. | `imerg_only_T100` bias is materially below `imerg_only`. If so, the production config must match T across arms or report the median. |
| `bias` | Quantile-mapping IMERG is sufficient on its own. | `imerg_only_bc` removes a large fraction of the background bias where `imerg_only` removed 4%. |
| `weighting` | IMERG is simply muted by the 6.28× correlation inflation and stride-3 thinning. | Any of `r0p25`, `dense` improves CRPS *and* correlation without a bias blow-up. If none do, the muting is not the problem and the R settings can stay. |
| `twostep` | Gaspari–Cohn tapering spreads gauge increments along meteorology instead of into discs. | Locality ratio near 1 while CRPS holds. This is the arm most likely to fix the figure. |

### Decision gate

`scripts/29_summarize_method_sweep.py` prints, for every arm, ΔCRPS against
`gauges_only` with a **paired circular block bootstrap** (block ≥ 3 days,
10,000 resamples). It promotes an arm only if central ΔCRPS improves *and*
absolute bias does not worsen by more than 0.5 mm/day.

**Promote at most two arms to the full 2021–2024 run.**

### Read the sweep correctly

Five days cannot resolve a 2% CRPS difference; the bootstrap interval is printed
so that a small central estimate cannot be mistaken for a result, and
`tests/test_method_sweep.py` asserts the interval stays wide at this sample size.
What five days *can* resolve is a 10 mm/day bias, a factor-of-two change in
increment locality, a wet-day frequency near 1, and an arm that diverges. Read
those columns.

---

## 4. Prior-side work the sweep cannot fix

The wet-day frequency panel shows the prior raining ≥1 mm almost everywhere,
almost every day. No assimilation scheme repairs that; it has to be fixed in the
prior. Three checkable causes, in order of suspicion:

1. **Residual-to-CPC floor.** The target is `T(CHIRPS) − T(base)` with
   `base = cpc_precip`. A coarse base is nonzero nearly everywhere, so the
   reconstruction inherits a nonzero floor after the inverse transform. Check the
   unconditional wet-day frequency of the decoded field against CHIRPS directly.
2. **Wet-day oversampling** (v2 checkpoint only). `wet_sampling` draws the
   wettest decile at 35% with no importance reweighting of the loss, which tilts
   the learned prior wet by construction. Confirm which checkpoint produced the
   pooled results.
3. **CPC regridding artefacts.** Row 3 of the intercomparison shows concentric
   rings around the orographic maximum in the CPC input that propagate into
   Background, IMERG-only and Simultaneous. The model is learning regridding
   structure as fine-scale skill. A referee will spot the rings.

Also note `cond_dropout: 0.0` in both training configs: the unconditional-prior
ablation described in `METHODOLOGY.md` §3.3 is **not available** from these
checkpoints. Either drop that claim from the paper or retrain with dropout.

---

## 5. Ordered next steps

1. Background-versus-CHIRPS bias at withheld stations (§2a). Decides the story.
2. Fit the leave-one-year-out IMERG map (script 27) and read its holdout
   diagnostics — raw versus corrected bias and wet fraction — before assimilating
   anything.
3. Run the `core` and `tempering` groups at offset −1 and offset 0.
4. Run `weighting` and `twostep` only if `core` has not already settled it.
5. Promote at most two arms to a full 2021–2024 rerun, and rerun **2024 as
   May–Sep** so the pooling is seasonally balanced — the current 2024 arm is
   May–Jun only, 2,196 station-days, and carries the largest reported IMERG gain.
6. Attach paired block-bootstrap intervals to the multi-year ΔCRPS. If the
   interval straddles zero, "indistinguishable" is the honest and publishable
   conclusion; the present 2.6% is not defensible without it.
7. Add non-generative baselines to the headline table — IDW on the same gauges,
   CPC bilinear, optimal interpolation. Nothing in the current tables shows the
   generative prior is necessary.

---

## Files

| Path | Role |
|---|---|
| `scripts/27_fit_imerg_bias_correction.py` | Leave-one-year-out IMERG→CHIRPS quantile map + empirical error model |
| `scripts/28_simultaneous_method_sweep.py` | The sweep. `--list-variants` for the catalogue |
| `scripts/29_summarize_method_sweep.py` | Ranking table, bootstrap intervals, bullseye diagnostic, figure |
| `slurm/simultaneous_method_sweep.sbatch` | Grace/GH200 job: observations → map → sweep → summary |
| `slurm/submit_simultaneous_sweep_may2024.sh` | Submission wrapper, May 1–5 2024 by default |
| `tests/test_method_sweep.py` | Unit tests for the quantile map, per-sample CRPS, bootstrap and locality diagnostic |
