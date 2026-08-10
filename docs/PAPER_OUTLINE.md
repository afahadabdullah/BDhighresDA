# Paper outline: OSSE-led, real-data grounded

**Working title.** Generative rectified-flow priors for precipitation data
assimilation: what observation scale, not observation strength, controls.

**Framing decision.** OSSE-*led*, not OSSE-only. The OSSE establishes that the
method works when observation error is known; the real-data experiment shows
what limits it in practice. Either half alone is weaker — a pure OSSE invites
"does it work on real data?", and the real-data null alone cannot distinguish a
bad observation from a bad method.

**The reframe that does the most work.** An OSSE's power is that error is
*known*, not that it is *zero*. Perfect observations appear only as **null
models** (Claims A and B in `OSSE_DOWNSCALING_DESIGN.md`), never as an
assimilation assumption. Setting error to zero would discard the measured
σ_rep = 0.410 and would erase the paper's central finding, since with perfect
observations finer footprints are trivially better and the scale ladder
collapses.

Status key: **[have]** figure/data exists · **[run]** code exists, needs a run ·
**[new]** needs code · **[write]** prose only.

---

## 1. Introduction

Bangladesh: ~148,000 km², 38 BMD gauges, ~3,900 km² per gauge. Satellite
retrievals cover the gaps but carry bias and scale-dependent error. The
question is not whether to use both, but *how* to combine them.

Claim structure, stated up front so the reader knows what is and is not being
argued:

1. A generative prior can synthesise sub-grid rainfall structure the coarse
   input does not contain (Claim A).
2. The analysis places rain correctly *below* the satellite footprint, beating
   a null handed perfect footprint-mean information (Claim B).
3. Observation **scale** dominates observation **strength** in what the
   satellite contributes — and at native resolution the satellite can make the
   field worse than not assimilating it.

- **Fig 1** pipeline schematic — **[have]** `docs/figures/pipeline.png`

## 2. Method

Condensed from `docs/METHODOLOGY.md`, which is already written.

- 2.1 Rectified flow / stochastic interpolant prior; CFG via conditioning
  dropout — **[write]** (METHODOLOGY §3)
- 2.2 Observation operators: bilinear for gauges, exact block-average for
  satellite footprints — **[write]** (§4.1)
- 2.3 Likelihood and guidance; velocity↔score — **[write]** (§4.2, §3.4)
- 2.4 Precipitation transform and the atom at zero — **[write]** (§5)
- 2.5 **Observation error, measured not assumed.** Variogram (Matheron), nugget/
  sill/range, block dispersion γ̄(V,V), σ_rep = 0.410 transformed at cell
  scale, 0.419 at 0.1° footprints.
  - **Fig 2** variogram + fitted model — **[have]** `variogram.png`

  This subsection is a genuine differentiator. Most precipitation DA papers
  assume R; this one measures it.

## 3. Experimental design

- 3.1 OSSE construction: nature run, simulated gauges and footprints, imposed R
- 3.2 **Identical-twin mitigation — must be explicit.** The prior trains on
  1981–2018; the nature run is drawn from 2021–2025, a strict temporal holdout.
  Same *product* nonetheless, so the prior knows the truth's texture and
  intermittency, and results are optimistic in a way the paper must state.
  Fraternal-twin variant (nature run from bias-corrected IMERG rather than
  CHIRPS) — **[new]**, see §8.
- 3.3 Null models. Claim A null = 0.5° coarse input on the fine grid. Claim B
  null = the truth's own 0.1° block mean, upsampled: *perfect coarse
  information, zero sub-footprint structure.* Deliberately unfair to us.
- 3.4 Real-data design: 38 BMD stations, five rotated disjoint folds, 30
  members, fair (size-unbiased) CRPS, paired bootstrap over station-days.
  - **Fig 3** station map + fold assignment — **[have]** `station_map.png`

## 4. OSSE results

- **Fig 4** claims figure: A and B against their nulls —
  **[run]** `scripts/24_osse_paper_suite.py :: figure_claims`
- **Fig 5** RAPSD / effective resolution against truth —
  **[run]** `figure_spectra`
- **Fig 6** scale ladder in the OSSE — **[run]** `figure_scale_ladder`
- **Table 1** arm × metric matrix — **[run]** `write_latex` + `figure_matrix`
- **Fig 7** observation value: gauges vs satellite vs both —
  **[run]** `figure_observation_value`

**The one genuinely new OSSE experiment — [new].** Sweep the *satellite
observation error* at fixed everything else and locate the threshold at which
the satellite begins to add value over 30 gauges. This is the experiment that
explains the real-data null instead of merely reporting it: if real IMERG's
error sits on the wrong side of that threshold, the null is predicted rather
than discovered. Extends `scripts/26_osse_footprint_ablation.py`.

Secondary, and cheap once the sweep exists: vary **network density** (10/20/38
gauges). "How many gauges before the satellite stops contributing" is an
operational question for every gauge-sparse region, and it is answerable only
in an OSSE.

## 5. Real-data results

Source: `docs/ingestion_results.tex`, already written with every number.

- 5.1 Eleven arms, 380 withheld station-days, 178 wet. Ten of eleven
  combined-minus-gauges intervals contain zero. Gauges-only is production.
  - **Fig 8** ingestion matrix, three panels — **[have]** `ingestion_matrix.png`
- 5.2 **Stride 1 at native resolution is harmful**: +2.388 [+1.824, +2.956],
  replicated on two independent windows. Mechanism: `perturb_observations`
  draws with sd = √R, so inflating R injects noise rather than down-weighting.
  **This finding generalises well beyond precipitation** and deserves its own
  subsection rather than a footnote.
- 5.3 **Scale beats strength.** Pattern correlation 0.50 → 0.59 → 0.67 and wet
  area 0.922 → 0.875 → 0.820 as the footprint coarsens 0.1° → 0.4° → 0.8°,
  against a background pattern correlation of 0.60 — so 0.1° assimilation makes
  the field *worse than not assimilating*. Effective observation counts matched
  at 57/57/50, so this is scale and not quantity.
  - **Fig 9** structure vs footprint — **[run]** `scripts/40`, needs the 31-day
    confirmation (`slurm/submit_scale_ladder_31day.sh`, submitted)
- 5.4 Aggregation: daily → 5-day → monthly, systematic floor
  - **Fig 10** — **[have]** `error_vs_aggregation.png`, `aggregation_curve.png`

## 6. Why the satellite does not help in practice

The section that makes the two halves one paper. Three candidate explanations,
each testable against evidence already in hand:

1. **Prior wetness.** Background bias +8.37 mm/day at withheld stations, wet
   area 0.904, outside the CHIRPS/IMERG/CPC envelope. Assimilation removes most
   of the bias (+8.37 → +1.19 at best) but cannot repair a prior that rains
   almost everywhere. — **[have]**
2. **Uncorrected satellite bias.** IMERG assimilated raw at +5.56 mm/day into a
   Gaussian likelihood assuming unbiasedness;
   `observations.imerg.bias_correction` exists and is never read. — **[new]**,
   the `QM` arm. Cheapest remaining experiment and the largest untested defect.
3. **Target mismatch.** The analysis correlates −0.08 to +0.03 daily with
   CHIRPS — *the product the prior was trained on* — while CHIRPS itself
   correlates 0.29–0.56 with these gauges against IMERG's 0.71–0.78. — **[run]**
   `scripts/35` over the training years, scoring CHIRPS / IMERG / QM-IMERG
   against gauges on correlation **and wet-day frequency**.

Also here, because it is counter-intuitive and a reviewer will otherwise read
it as a failed measurement: **MEASR**, which uses the measured
representativeness, is among the *worst* arms. The measurement is not wrong —
the prior is. Given a background biased +8.37 mm/day, loosening the gauges
admits more of that bias, so the statistically correct observation error and
the skill-optimal one are different quantities.

## 7. Limitations

Written plainly rather than defensively; each already has its evidence.

- Identical-twin optimism in the OSSE (§3.2)
- Circularity in pattern correlation against IMERG, which is assimilated; CPC
  is cleaner but conditions the prior, so is not fully independent. Wet area and
  bias carry no circularity and agree.
- Scale ladder confounded with thinning — unavoidable (stride 3 at 0.8° leaves
  4 footprints) but its direction is known, since S1 shows stride 1 at 0.1° is
  catastrophic and the coarse stride-1 arms are nonetheless best.
- 38 gauges, one country, one season. The 31-day window is a superset of the
  10-day window: power, not independent replication. June 2022 is the
  independent test.
- Ensemble spread grows monotonically through sampling in all prior variants
  (r = +0.88 to +0.99) and remains unexplained — `ablation_v3.tex`.

## 8. What would change the conclusion

Short, concrete, and honest about cost:

- `QM` — bias-corrected IMERG. **[new]**, ~30 lines in `scripts/15` plus the
  fit from `scripts/27`. Largest untested defect.
- Retraining the prior on bias-corrected IMERG rather than CHIRPS. **[new]**,
  expensive, but §6.3 is the strongest argument in the paper for doing it.
- Fraternal-twin OSSE. **[new]**, removes the main reviewer objection.

---

## Target venues

*Journal of Hydrometeorology* or *QJRMS* if the framing is DA-methodological —
the √R perturbation result and the scale-over-strength result are both
transferable, and both are the kind of thing a DA audience will recognise as
useful negative knowledge. *Geoscientific Model Development* if the framing is
the system and its evaluation protocol. A pure OSSE with no real-data section
would land lower in all three.

## Critical path

1. Run the OSSE paper suite on existing OSSE output — Figs 4–7, Table 1. **[run]**
2. 31-day scale-ladder confirmation — Fig 9. **[submitted]**
3. Target comparison, `scripts/35` on training years — §6.3. **[run]**, minutes
4. `QM` — §6.2. **[new]**
5. Satellite-error threshold sweep — the new OSSE experiment. **[new]**
6. Fraternal twin, if reviewers or time demand it. **[new]**

Steps 1–3 use code that already exists and would establish whether the paper
holds together before any new code is written.
