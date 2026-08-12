# CPC-v2 BMD/IMERG ingestion triplet

## Question

On the same May 1--10, 2022 development window used for CPC-v2 gauge-method
selection, compare three observation strategies with five disjoint BMD folds,
30 ensemble members, and identical prior/observation seeds:

| Arm | Configuration |
|---|---|
| `guided_s6_g010_t100` | selected gauge method: spread 6, gamma 0.01, temperature 1.0 |
| `v2_imerg_s04_t100` | IMERG only: S04 0.4-degree footprints, stride 1 |
| `v2_simultaneous_s04_t100` | one joint likelihood; spread only its gauge component |

An unguided `background` is also generated as the paired reference; it is not a
fourth ingestion strategy.

## Why S04

The earlier v1 ingestion experiment made withheld BMD gauges the primary
target. S04 had the lowest satellite-only CRPS (6.44, versus 6.46 for S08) and
the lowest simultaneous CRPS (4.77). S08 preserved product-scale structure
better, but selecting it would overturn the predefined primary target for a
secondary metric. Native-resolution stride 1 is excluded because it was
decisively harmful.

S04 requires four matched properties: the observation file is block-averaged
to 0.4 degrees, `observations.imerg.factor=8`, stride is 1, and
`error_corr_cells=0.75` preserves the physical correlation length. The launcher
prepares this file once before the array starts.

Raw IMERG is used because that is the configuration actually evaluated in the
earlier tournament. This run does not claim that raw IMERG is unbiased; the
leave-one-year-out quantile-mapped method remains a separate, unselected
experiment.

## Stream-specific spatial guidance

The gauge winner spreads its point-likelihood gradient by six 0.05-degree grid
cells. An S04 IMERG observation is already a 0.4-degree block average. Blurring
the sum of both gradients would therefore spread IMERG twice and would not be a
combination of the selected methods. The simultaneous arm evaluates the two
likelihood terms through the same denoised state, spreads only the gauge term,
then adds the gradients before the usual single norm clipping.
It preserves the selected per-stream softness: gamma 0.01 for gauges and the
S04/default gamma 0.001 for IMERG.

## Numerical preflight

The first submitted version exposed a masked-ocean autograd bug: CPC residual
bases can be NaN outside land, and the physical satellite operator applied its
nonlinear inverse before masking those cells. Although the matching IMERG
observations were excluded, backward propagation still evaluated `0 * NaN`,
which poisoned the gradient norm and then the analysis. Physical observation
operators now replace invalid cells before nonlinear transforms. Guidance also
fails on the first non-finite gradient, before clipping can contaminate a full
field.

The first full rerun then isolated a second issue to one ensemble member during
the simultaneous arm's Langevin correction. That code path had two uncontrolled
mechanisms: the adaptive step `delta = tau * dimension / ||score||^2` was
unbounded if prior and observation scores nearly cancelled, and correctors
perturbed masked ocean cells until the outer sampler restored the mask. The
failed run predates delta diagnostics, so their individual contributions cannot
be separated. The sampler now (1) computes the score norm over active land
dimensions, (2) caps delta at the configured `corrector_max_step=0.3`, and
(3) reapplies the land mask after every correction. Every fold report records
the cap count and maximum raw/applied delta under `sampler_diagnostics`; the
pooled report carries the aggregate so this safeguard cannot operate silently.
Treat cap activation above 1% of member-steps, or a much higher fraction in the
simultaneous arm than either single-stream arm, as a failed stability screen;
in that case the result needs a no-corrector sensitivity before publication.

A subsequent v1-to-v2 audit found the missing v2-specific mechanism. V1 used a
smooth `log1p` precipitation transform, whereas v2 uses `sqrt`. The physical
operators compute `T(average(T^-1(x)))`; at an exactly dry gauge stencil or
satellite footprint, automatic differentiation previously crossed the infinite
derivative of `sqrt(0)` after the inverse branch supplied a zero derivative.
This undefined `0 * inf` path explains why a single member could fail in either
observation component. The square-root transform now preserves its exact
forward values and explicitly selects the finite zero subgradient at zero.
Regression tests cover the transform itself and all-dry physical gauge and
block-average operators. Each GPU fold also runs
`scripts/51_check_sqrt_da_gradient.py` against its actual torch/CUDA backend
before assimilation. The land-mask and bounded-corrector changes remain valid
independent safeguards, but they are no longer presented as the complete
explanation of the v2 failures.

Before committing a full five-fold run, reproduce the failing fold through day
2 with all 30 members:

```bash
V2_INGEST_PREFLIGHT=1 V2_INGEST_END=2022-05-02 V2_INGEST_MEMBERS=30 \
  V2_INGEST_ROOT=data/processed/v2_ingestion_triplet/preflight \
  bash slurm/submit_v2_ingestion_triplet.sh
```

This prepares the matching two-day S04 file and submits fold 0 only. It does
not submit the five-fold summary, which correctly requires all folds.

## Run

From the cluster repository root:

```bash
git pull --ff-only origin main
bash slurm/submit_v2_ingestion_triplet.sh
```

IMERG preparation runs with `python` (or `python3`) from the active
submission-node environment. If that interpreter does not contain NumPy and
xarray, activate a compatible environment or set `V2_INGEST_PREP_PYTHON` to a
binary built for the submission node. The GPU job continues to use its
configured GH200 environment.

The launcher submits a five-fold GPU array and a dependent CPU summary. Outputs
are isolated at the path below. By default it first reproduces fold 0 through
May 2 with all 30 members—the exact case that previously failed—and holds the
full array on an `afterok` dependency. The preflight receives a dedicated
May 1--2 S04 file; it does not reuse the May 1--10 file because the loader
requires the observation dates to match the requested dates exactly. Set
`V2_INGEST_AUTO_PREFLIGHT=0` only after an identical commit has already passed
that preflight.

```text
preflight -> five-fold array -> pooled summary
```

Outputs are isolated at:

```text
data/processed/v2_ingestion_triplet/ing2022_s04_g010_sqrtfix/
  fold0.npz ... fold4.npz
  fold0.json ... fold4.json
  ingestion_selection.md
  ingestion_selection.json
  ingestion_selection.png
  fold_plots/fold0_diagnostics.png ... fold4_diagnostics.png
```

The primary paired test is
`CRPS(gauges-only) - CRPS(simultaneous)`. Positive values favour adding IMERG.
The 95% interval uses a circular three-day block bootstrap and keeps every
station from a weather day together. If the interval includes zero, the honest
conclusion is that the satellite contribution is unresolved on ten days.

## Matched v1 comparison

After the corrected five-fold v2 triplet and its summary finish, compare it
against both the previously reported v1 gauge result (`ing2022_RAW`) and the
same S04 simultaneous method under v1 (`ing2022_S04`) without rerunning DA:

```bash
bash slurm/submit_v1_v2_ingestion_comparison.sh
```

The CPU-only comparison aligns folds by their withheld station-ID sets rather
than trusting fold numbers. It rejects different station pools, dates, BMD
values, ensemble sizes, or assimilated complements. It reports paired
day-block intervals for v2 gauges versus v1 gauges, v2 simultaneous S04 versus
v1 simultaneous S04, and v2 simultaneous S04 versus the previously reported
v1 gauge DA. Outputs are written under:

```text
data/processed/v2_ingestion_triplet/ing2022_s04_g010_sqrtfix/v1_vs_v2/
  v1_vs_v2_selection.md
  v1_vs_v2_selection.json
  v1_vs_v2_selection.png
```
