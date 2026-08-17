# Reassessing the V3-SG workflow against what the data actually supports

Written after the alignment and intercomparison diagnostics. The goal is
unchanged: a ~0.05-degree daily precipitation reanalysis for Bangladesh from
satellite and local gauges. What has changed is the evidence about which inputs
carry the information that goal needs.

---

## 1. What the diagnostics established

| Finding | Evidence |
|---|---|
| CHIRPS daily carries little information shared with CPC | r = 0.330 daily, 0.431 at 3 days, 0.474 at 5, 0.559 at 10, **0.683 at 30** (2015–2020, 1965 days, 0.5-degree support) |
| It is not a registration bug | identity wins the flip/translation search (0.330 vs 0.309 for the best shift); identical result from the v2 and v4 archives |
| The prior reproduces its conditioning, not its target | model vs CPC 0.35, model vs IMERG after DA 0.45, model vs CHIRPS 0.03 |
| Gauge DA works | withheld-gauge CRPS 4.65 → 3.77 mm/day, a 19% improvement |
| Occurrence handling is biased | bias −0.85 → −2.6 mm/day as observations are added; dry MAE 2.8 → 1.0 while wet MAE barely moves |
| Blockiness is geometric, and fixable | block-constant base has an infinite seam index; the v5 conservative smooth base gives 1.16 with conservation unchanged |

The monotone rise from 0.33 to 0.68 does not isolate CHIRPS. CPC is a
gauge analysis over a very sparse network across NE India and Myanmar, so its
daily field is also heavily interpolated. **Both products are weak daily and
strong multi-day.** That joint weakness is the finding that matters.

---

## 2. The diagnosis

The prior's only precipitation input is CPC.

```
ERA5_DEFAULT = ("tcwv", "cape", "u10", "v10", "msl")     # no precipitation
IMERG                                                     # absent from the prior
```

So the emulator is asked to place today's rain at 0.05 degrees given a sparse
gauge interpolation at 0.5 degrees plus thermodynamic fields, and is scored
against a target whose daily component is disaggregated from pentads. Each of
those three legs is weak in the same dimension — **daily timing** — and the
weaknesses compound.

A useful way to see it: the model correlates with CPC at 0.35 and CPC
correlates with CHIRPS at 0.33. The model is doing roughly what its inputs
permit. The near-zero CHIRPS score is not primarily a training failure; it is
the pipeline delivering a daily answer without a daily observation anywhere in
it.

Meanwhile IMERG — a genuine half-hourly, passive-microwave-driven observation at
0.1 degrees — is held out of the prior entirely and used only in the
assimilation step. **The best daily observational constraint available is used
last and least.**

The exclusion follows a correct rule (IMERG must not be both a conditioning
channel and an assimilated observation) applied in a costly direction. The rule
does not require IMERG to be an observation. It requires a choice.

---

## 3. Proposed workflow

Make IMERG the daily backbone and assimilate only the gauges.

```
                 ┌─────────────────────────────────────────┐
   IMERG 0.1°    │  amount backbone, 0.1°                  │
   CPC 0.5°   ───▶  bias-corrected daily precipitation     │
   ERA5 tp      │  (observed daily timing)                 │
                 └───────────────────┬─────────────────────┘
                                     │  m at 0.1°
                                     ▼
   ERA5 dynamics  ┌──────────────────────────────────────┐
   terrain, coast │  stochastic subgrid emulator          │
   season       ──▶  0.1° → 0.05°, factor 2               │
                    │  conservative, ensemble             │
                 └───────────────────┬──────────────────┘
                                     │  x at 0.05°
                                     ▼
   BMD gauges  ──────────▶  DA on (m, z)  ──────▶  analysis ensemble
   (assimilated)                                   (withheld folds verify)
```

Four consequences, in order of importance.

### 3.1 The downscaling factor drops from 10× to 2×

Conserving on CPC's 0.5-degree support means predicting **100** fine cells from
one number. Conserving on IMERG's 0.1-degree support means predicting **4**.

This is not a marginal improvement. It changes the problem from "invent the
within-block field" to "split an observed cell four ways", which is close to
what terrain and the local gradient can actually determine. It also largely
dissolves the blockiness question: a 2×2 block boundary is a far smaller
discontinuity than a 10×10 one, and the v5 smooth base then has much less work
to do.

Grid arithmetic is clean: 0.1 / 0.05 = 2 exactly, and 0.5 / 0.1 = 5, so a
0.1-degree support still nests inside CPC's for diagnostics.

### 3.2 The amount field becomes observed rather than interpolated

CPC's 0.5-degree daily field over this domain is an interpolation across a
sparse network. IMERG sees the storms. Putting IMERG in the prior means the
emulator's "how much and roughly where" comes from an observation, and the
network's remaining job is the part it can actually learn.

Keep CPC as a **second** conditioning channel — it is gauge-based and therefore
carries information IMERG does not, particularly for bias. The two together are
better than either.

### 3.3 Add ERA5 total precipitation

It is already in the pipeline's reach and currently unused. ERA5 precipitation
is biased in the tropics, but its *timing and pattern* are dynamically
constrained by a full assimilation system, which is exactly the property CPC
lacks. It is also the only source of real daily timing before 2000.

This is the cheapest change on this list.

### 3.4 The assimilation set becomes gauges only

IMERG moves from observation to conditioning; BMD gauges are assimilated and,
through the existing five-fold scheme, verify. That keeps the no-double-counting
rule intact and preserves the only genuinely independent evidence in the system.

**Cost, stated plainly:** the "scale-aware division of observational authority"
claim weakens. With one observation type there is no IMERG-versus-gauge
authority contrast to measure, and the `S_m` / `C_m`-`C_z` machinery loses its
headline role. Retain the IMERG-assimilating configuration as a declared
ablation so that contrast is still measurable, but do not build the primary
product on it.

---

## 4. What CHIRPS becomes

Not a daily placement teacher. Under the proposed workflow the daily "where"
arrives from IMERG, and the emulator only has to add 0.1 → 0.05 detail — which
at 5 km over this terrain is substantially persistent: orographic gradients on
the Meghalaya and Chittagong barriers, coastal convergence, the river-plain
wet/dry geography.

CHIRPS resolves that structure well even though its daily timing does not
survive. So keep it as the target, and **change how it is scored**: as a
conditional distribution (spectra, variograms, wet-area fraction, orographic
ratio, extreme quantiles) rather than as a daily point match. Its poor daily
timing matters far less when it is no longer being asked to supply the daily
signal.

This also promotes `clim_ratio_null` from nuisance baseline to central
comparison, which is where the design's own claim ladder already pointed.

### On PERSIANN-CCS-CDR

Worth testing, not worth adopting on the strength of the paper. Its daily
timing is retrieval-driven (3-hourly IR, monthly-only bias correction), which is
the property CHIRPS lacks. Three cautions: its `-CPC` stream shares the NOAA/CPC
4-km merged IR archive with IMERG, so target and conditioning would be
correlated through a common source; IR-only retrievals systematically miss warm
orographic rain, which is a large share of the signal over exactly the terrain
of interest; and 0.5 / 0.04 is not an integer, so exact block aggregation to
CPC's support does not exist.

`scripts/63_product_intercomparison.py --alt-fine-glob` runs the identical
aggregation curve on it. Decide from the curve.

---

## 5. The scoping decision this forces

IMERG Final begins in 2000. That is a real trade and it is yours to make.

| | record | daily skill | what it is |
|---|---|---|---|
| **A — IMERG-conditioned** | 2000–present, ~19 training years | real daily observation | a high-skill modern reanalysis |
| **B — CPC/ERA5-conditioned** | 1981–present, ~38 training years | climatological + weak daily | a long record, honest that daily placement is unsupported |

Nineteen years is roughly 7,000 daily fields, which is ample. I would build **A**
as the primary product and keep **B** as an optional long extension with an
explicitly lower claim. Presenting both, with the difference measured rather
than asserted, is a stronger paper than either alone — and the aggregation curve
you now have is exactly the tool that quantifies the gap.

---

## 6. What to keep unchanged

Most of the machinery survives the reframe:

- the hierarchical `(m, z)` state, hard conservation, and the differentiable
  decoder — only the support changes, 0.5 → 0.1;
- the conservative smooth base (v5), which matters less at factor 2 but still
  removes a real artifact;
- the five-fold withheld-gauge verification and the neighboured holdout;
- the CRPS-based member-wise anomaly scoring and the day-block bootstrap;
- the frozen-encoding contract, schema versioning and the alignment guards;
- the claim ladder — it was written for exactly this situation.

---

## 7. Order of work

1. **Re-measure the model at 0.5 degrees.** Panel D now scores the model on the
   same support as its references. If the green bar is near 0.33, the current
   model already extracts what its conditioning permits and the diagnosis above
   is confirmed. If it is near 0.03, there is also a training problem. These
   lead in different directions and the run is free — no re-sampling needed.
2. **Run the PERSIANN curve** if the data is easy to obtain. One or two years
   over the domain is enough to see the shape.
3. **Rebuild the target archive on a 0.1-degree support** with IMERG, CPC and
   ERA5 total precipitation as conditioning. This is the substantive change.
4. **Retrain and re-run the pilot**, gauges-only as primary.
5. **Then** revisit the occurrence gate — the −2.6 mm/day dry bias is untouched
   by any of this and will be the largest remaining error term.

Step 1 costs nothing and should decide whether steps 3–4 are worth the compute.
