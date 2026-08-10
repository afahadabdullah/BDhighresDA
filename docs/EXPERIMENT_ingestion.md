# Finding the best way to ingest IMERG and BMD

**Primary target:** withheld BMD stations. They are the only direct measurement
of rainfall in this system and the only defensible truth.

**Secondary target:** spatial structure and monthly means, judged against CPC,
IMERG *and* CHIRPS together. None of the three is correct — over May–June 2024
their daily correlations against BMD gauges were 0.76, 0.77 and 0.04
respectively — so the target is their *envelope*, not any one of them. A field
that sits outside all three is implausible; a field inside is not thereby
correct, only not obviously wrong.

**Fixed for this experiment:** the prior. `runs/prior_h100_cpc/best.pt` (V1).
The v3/v4 ablation is closed and recorded in `docs/ablation_v3.tex`; nothing
here retrains anything.

**Window:** 2022-05-01 to 2022-05-10, five rotated folds, 30 members.
Ten days x five folds is roughly 370 withheld station-days, matching the
May 2024 screen. Thirty members is chosen from the measurement in
`scripts/41`: fair CRPS is flat with ensemble size (drift -5% to +5% with no
trend) while the member-sampling interval narrows from ~3.0 mm/day at m=2 to
~0.3 at m=30 and 0 at m=50. Members buy precision, not skill, and 0.3 mm/day
is below the differences these arms are designed to produce.

---

## What is already known, so we do not repeat it

Four configurations were screened over 10 days of May 2024 with five folds
(300 withheld station-days). Every pairwise interval spanned zero except one:

| finding | evidence |
|---|---|
| stride, R multiplier, `prior_temperature` and `gamma` are **not separable** | 12 pairwise bootstrap intervals, all spanning zero |
| `s1r10T` is significantly **worse** | +2.60 CRPS in the satellite arm; mechanism identified below |
| combined ≤ gauges-only | 4.04 vs 4.08 CRPS (tie), 6.90 vs 6.24 MAE (worse), pattern 0.66 vs 0.74 (worse) |
| satellite-only is the weakest arm | MAE 10.05, bias +5.56, wet area 0.873 |
| every arm is outside the product wet-area envelope | 0.68–0.99 against a much lower product range |

Two mechanisms were identified and both matter for the design here.

**The R multiplier injects noise, it does not merely down-weight.**
`perturb_observations` draws each member's observation perturbation with
sd = √R. Raising R by 10× therefore multiplies the perturbation amplitude by
3.2×, and at stride 1 on the 0.1° grid the automatic correlation inflation is
already 56.5×, so ×10 gives 565× and perturbations 24× the nominal. That is why
`s1r10T` degrades rather than simply ignoring the satellite. **Consequence: R
inflation is not a usable strength knob on this system.** It is excluded here.

**IMERG has never been bias-corrected.** Every run has printed
`WARNING: no fitted bias correction in this bounded process run`, and
`observations.imerg.bias_correction` exists in the config but
`scripts/15_bmd_month_example.py` never reads it. The satellite has been
assimilated raw, with a measured +5.56 mm/day bias at the gauges, into a
Gaussian likelihood that assumes unbiasedness. This is the largest untested
defect and the first configuration below addresses it.

---

## The configurations

Two axes. Group A changes *what* and *where* is assimilated; Group B changes
*how strongly*. The earlier null result closed the **satellite** strength axis
only — stride and the R multiplier — and left the gauge side and the
gauge-to-satellite ratio untested, which is where the two streams actually
trade off.

Eleven arms are config-only and run with the code as it stands
(`slurm/submit_ingestion_experiment.sh`). Four need code that does not exist
yet and are listed but not submitted, so the experiment cannot half-execute.

### Group A — *what* and *where* is assimilated

| tag | method | change | code |
|---|---|---|---|
| `G` | gauges only | read out of any run with `scripts/42 --arm gauges`; **not submitted separately** | none |
| `RAW` | simultaneous, raw IMERG | the current baseline | none |
| `QM` | simultaneous, **bias-corrected** IMERG | quantile map from script 27 applied before assimilation | new |
| `S01` | IMERG at **0.1°** | `imerg.factor: 2` — the native footprint. This *is* `RAW`; named so the scale ladder reads cleanly, and **not submitted twice** | none |
| `S04` | IMERG at **0.4°** | `imerg.factor: 8` — assimilate only the large-scale component and let the prior supply fine structure | config |
| `S08` | IMERG at **0.8°** | `imerg.factor: 16` — the far end of the scale ladder | config |
| `MULTI` | IMERG at 0.1° **and** 0.4° together | both footprint sets in one likelihood | new |
| `GAP` | satellite **only away from gauges** | satellite R inflated within a radius of each assimilated gauge | new |
| `MEASR` | measured observation error | `sigma_obs`/`representativeness` from script 35 for both streams | config |
| `CPCOBS` | CPC assimilated as a third stream | 0.5° pseudo-satellite via script 34, alongside BMD and IMERG | config |

**The scale ladder is the point of `S01`/`S04`/`S08`.** IMERG's information is
concentrated at scales it can actually resolve; a 0.1° footprint asserts
structure where the retrieval is weakest and the prior is most useful. Running
three factors of the *same* product isolates observation scale from every other
choice, and pattern correlation has capped at 0.66–0.74 in every arm tried so
far, which is the symptom this addresses.

**`MULTI` double-counts on purpose, and that is the test.** Assimilating the
same field at two scales uses each observation twice, so the likelihood is
formally wrong. It is included because the *practical* question — does adding a
coarse constraint on top of the native one help — has a clear answer either way,
and because if it wins despite the double-counting that is itself informative
about where the satellite's usable information sits. Read it as an upper bound,
not as a defensible configuration.

**`CPCOBS` is not an independent observation and must be labelled as such.**
CPC conditions the prior (it is `cpc_precip` in the checkpoint's input channels)
*and* is itself a gauge analysis built from GTS station reports, which for
Bangladesh very likely include the BMD stations being assimilated and withheld.
Assimilating it therefore (a) reuses information the background already
contains, and (b) risks feeding withheld gauges back in through the side door,
which would inflate the scores in a way the holdout cannot detect. It is worth
running because a third spatial constraint is what a practitioner would reach
for, but its withheld-gauge numbers are **not** a clean skill claim and must not
be reported as one. See `scripts/34_make_cpc_pseudo_satellite.py`.

### Group B — *how strongly* each stream is ingested

The earlier null result on "strength" covered the **satellite** side only —
stride and the R multiplier. Gauge `sigma_obs` has been fixed at 0.10 with
`representativeness` 0.25 in every run to date, and the Desroziers diagnostic
says the analysis fits assimilated gauges about **12× harder than R permits**
(⟨(O−A)(O−B)⟩ = 0.014 against an assumed R of 0.178). The gauge axis and the
gauge-to-satellite *ratio* are therefore genuinely untested, and they are the
mechanism by which the two streams actually trade off.

Defaults for reference: gauges σ=0.10, rep=0.25; IMERG σ=0.35, rep=0.10.

| tag | gauge σ_obs | IMERG σ_obs | intent |
|---|---|---|---|
| `GW` | **0.05** | 0.35 | gauges trusted harder — does the over-fit help or hurt? |
| `GL` | **0.25** | 0.35 | gauges loosened toward the Desroziers-implied value |
| `GM` | **0.41** | 0.35 | gauge σ set to the *measured* representativeness (script 35) |
| `SW` | 0.10 | **0.20** | satellite trusted harder |
| `SL` | 0.10 | **1.00** | satellite loosened — down-weighted without inflating R |
| `RATIO` | **0.05** | **1.00** | maximal separation: gauges dominate, satellite is a weak prior nudge |

`SL` and `RATIO` down-weight the satellite through `sigma_obs` rather than the
R multiplier **on purpose**. Both enter R, but only the multiplier is applied
after the correlation inflation and therefore also scales the observation
perturbations drawn with sd = √R. Changing `sigma_obs` reaches the same
effective weight without the noise injection that made `s1r10T` fail. That
contrast — same weight, different mechanism — is itself a result worth having.

Plus one thinning contrast, `S1` (stride 1 instead of 3), to confirm the
earlier null on a new window rather than assume it transfers.

**What is actually submitted: eleven arms x five folds = 55 GPU jobs** at 10
days and 30 members — `RAW`, `S04`, `S08`, `GW`, `GL`, `GM`, `SW`, `SL`,
`RATIO`, `MEASR`, `S1`. `G` and `S01` are omitted because they are already
present in every run (`G` as the `gauges` arm, `S01` as `RAW`), and `QM`,
`GAP`, `MULTI` and `CPCOBS` are omitted because they need code.

### Why each arm

**`QM`** — a Gaussian likelihood with a biased observation moves the analysis to
the bias. If IMERG's +5.56 mm/day is the reason satellite assimilation hurts,
correcting it should show up immediately as bias moving toward zero in the
satellite arm. If it does not, the problem is the guidance rather than the
observation, and that is worth knowing before anything else is tried.

**The scale ladder** (`S01`, `S04`, `S08`) — pattern correlation caps at
0.66–0.74 across every arm tried. If skill improves as the footprint coarsens,
the satellite's usable information is at large scales and the fine footprints
have been injecting retrieval noise; if it degrades, the fine structure is real
and something else caps the correlation. Either outcome is a result.

**`GAP`** — the sharpest form of the "how do we combine them" question. Gauges
beat the satellite wherever both exist; the satellite's purpose is the ~4,200
km² per gauge the network cannot see. At present both are assimilated
everywhere, so the satellite competes with better information instead of
complementing it. If `GAP` beats `RAW`, the right combination rule is spatial
rather than a global weight.

**`MEASR`** — `sigma_obs` and `representativeness` are assumed (0.10/0.25 for
gauges, 0.35–0.60/0.30 for the satellite). Script 35 measured them from the
gauge network: σ_rep = 0.410 at cell scale, 0.419 at 0.1° footprints. Using
measured values is free and removes an assumption; it is not expected to be
decisive on its own.

---

## Evaluation

`scripts/42_select_best_config.py` already produces the ranked table and the
verdict. It reports, per configuration, pooled over all five folds:

* **Primary (gauges):** CRPS, MAE, wet-day MAE, bias, correlation at withheld
  stations.
* **Does the satellite help at all:** `--arm combined --vs-arm gauges`. Both
  arms come out of the *same dump file*, so they share the fold, the withheld
  stations, the days and the seeds, and the pairing is exact rather than merely
  fold-matched. This is the sharpest test available and the one the experiment
  exists to answer. It is also why `G` is not submitted as its own arm.
* **Which configuration:** paired bootstrap against `RAW`, matched fold by fold.
* **Secondary (structure):** spatial pattern correlation against each of CHIRPS,
  IMERG and CPC, and wet-area fraction against their envelope.

Run it for all three arms — `--arm combined`, `--arm gauges`, `--arm satellite`
— because the satellite arm isolates what the satellite contributes and the
combined arm shows what survives when gauges are also present.

Supporting figures, all gauge-referenced:

* `scripts/38` — daily→monthly aggregation, the systematic floor, per-station
  bias maps, Taylor summary, monthly means against all three products.
* `scripts/40` — power spectra against the product envelope, effective
  resolution, intensity quantiles.
* `scripts/37` — per-day maps of background, analysis and increment with gauges
  overlaid, for the winning configuration only.

### The decision rule, fixed in advance

1. An arm counts as better than `G` only if its paired bootstrap interval
   excludes zero. Point estimates do not decide anything — that is what produced
   the earlier rankings that reversed.
2. Among arms that pass (1), prefer the one whose wet-area fraction is inside
   the product envelope and whose pattern correlation is highest.
3. If no arm passes (1), the honest result is that satellite assimilation does
   not help at these settings, and `G` is the production configuration.

---

## What ten days can and cannot show

Ten days × five folds ≈ 370 withheld station-days, matching the May 2024
screen, which could **not** separate configurations differing by ~0.1 mm/day
CRPS. Nothing about this window changes that resolution limit.

It is the right sample anyway, *because these arms differ structurally rather
than marginally*. The scale ladder changes the observation footprint by factors
of four and eight; `RATIO` moves the two streams' relative weight by a factor of
400 in variance; `QM`, when it exists, changes the satellite bias by roughly
5 mm/day. Effects of that size are detectable at 370 station-days. What ten days
cannot do is rank two arms that come out close, and the plan for that outcome is
to promote the top two to May–June rather than to read the point estimates.

`scripts/42` prints `n_wet`. Below ~50 the window is too dry to conclude
anything, and the honest move is to extend the window rather than interpret it.

---

## Order of work

1. **Cut the prepared IMERG window** — `scripts/43_subset_prepared_imerg.py`
   slices `data/processed/imerg_bd_aligned_20220501_20220531.nc` down to
   2022-05-01..10. `scripts/15` compares the time axis to the checkpoint dates
   with `array_equal`, so the monthly file cannot be used directly; but it was
   written with the same `--source-frequency half-hourly --min-count 48
   --accumulation-end-hour-utc 3` a rebuild would use, so the accumulation is
   already right and only needs cutting. `submit_ingestion_experiment.sh` does
   this automatically, once, before submitting — per-arm generation is what
   caused the concurrent-write race that killed two folds of the last screen.
2. **Submit the eleven config-only arms** —
   `bash slurm/submit_ingestion_experiment.sh`.
3. **Evaluate** — `scripts/42` for `--arm combined`, `--arm gauges` and
   `--arm satellite`, then 38 and 40, then 37 for the winner.
4. **Then the code arms.** `QM` first: `scripts/15` never reads
   `observations.imerg.bias_correction`, so IMERG has been assimilated raw with
   a +5.56 mm/day bias in every experiment ever run, and that is the largest
   untested defect in the system. It needs the fit from script 27 (2021–2024,
   `--chirps-day-offset -1`; if the amplitude map is noise-dominated as in the
   earlier fit — interannual sd 1.348 against mean 0.586 — refit with
   `--frequency-only`, which keeps the drizzle cut and drops the unstable
   amplitude adjustment), then the application in script 15, with unit tests.
   `GAP` and `MULTI` follow.
