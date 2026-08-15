# Version 3 subgrid model idea

## Purpose

The present CPC-v2 prior predicts precipitation directly on the 0.05-degree
grid. It substantially improves rainfall amount and withheld-gauge skill, but
it has not demonstrated that it can place rainfall correctly below the
0.5-degree support of CPC, its precipitation input. A fine output grid is not,
by itself, evidence of fine effective resolution.

The proposed model separates two different tasks:

1. predict **how much rain** falls in each coarse area; and
2. predict **how that rain is distributed** among the 0.05-degree cells inside
   the coarse area.

The intended scientific claim is deliberately limited to learning the
conditional distribution of **CHIRPS-like 0.05-degree patterns**. CHIRPS is a
gridded infrared--station product rather than physical truth at every 0.05-degree
cell. Independent BMD gauges remain necessary for real-data verification.

The main methodological contribution is the **scale-aware division of
observational authority**, not mass-conserving downscaling alone. CPC and ERA5
define a large-scale prior at CPC's native support, IMERG supplies independent
area-average evidence on a different support, BMD gauges supply point evidence,
and the learned allocation prior connects these constraints without pretending
that any one product is fine-scale truth. The deliberately mismatched 0.5-degree
prior and 0.4-degree satellite supports make this claim testable rather than
definitionally true.

> **Repository naming note.** `prior_h100_cpc_v3` is already used by the
> hurdle/absolute-target wet--dry ablation. To avoid overwriting or confusing
> that experiment, this design should use the explicit identifier
> `cpc_v3_subgrid` and an output directory such as
> `runs/prior_h100_cpc_v3_subgrid`. In a paper, it may be clearer to call this
> model **V3-SG**.

## One-sentence model description

V3-SG is a hierarchical, mass-conserving conditional flow model in which
coarse-amount and fine-allocation branches are pretrained separately and then
coupled into a joint multiscale flow; during DA, IMERG constrains overlapping
0.4-degree area means and BMD gauges constrain local fine-scale values.

## System overview

```text
CPC + ERA5 + season/static fields
                 |
                 v
       coarse amount model
                 |
                 v
       m: 0.5-degree rainfall ---------------+
                                             |
m + ERA5 + season/static fields + noise      |
                 |                           |
                 v                           v
       stochastic allocation model ----> reconstruction
                                             |
                                             v
                                  x: 0.05-degree ensemble

DA observations:
    IMERG S04 --------------> constrains overlapping 0.4-degree area means
    BMD point gauges -------> constrains x / within-block placement
```

The operational background uses no BMD or IMERG observations. Those products
enter only through the DA likelihood, preserving a clean distinction between
learned downscaling and observation-driven analysis.

The diagram shows the interpretable pretraining factorisation. After Phase 2,
the coarse and fine branches are coupled and fine-tuned as one multiscale joint
flow for operational background generation and DA; they are not treated as two
independent prior scores.

## Resolution and notation

- Fine target grid: 0.05 degrees.
- Prior coarse support: CPC's native 0.5 degrees, corresponding to 10 by 10
  fine cells on a CPC-aligned grid.
- DA satellite support: IMERG S04 at 0.4 degrees, represented by its own exact
  8 by 8 physical-space observation operator.
- `B10_cpc(x)`: conservative area-weighted 0.5-degree mean of `x` on the native
  CPC-aligned footprint grid.
- `B8_imerg(x)`: conservative area-weighted 0.4-degree mean on the prepared
  IMERG S04 footprint grid.
- `U(m)`: repeat or conservatively interpolate coarse field `m` to the fine
  grid.
- `m`: shorthand for the coarse hurdle latent (wetness logit plus positive
  amount latent), decoded to physical 0.5-degree rainfall amount.
- `z`: fine latent containing 0.05-degree allocation logits and wetness logits.
- `x = R(m, z)`: reconstructed physical 0.05-degree precipitation.

The prior hierarchy follows CPC's physical support rather than a previously
selected DA configuration. This gives a clean two-step claim: CPC is first
bias-corrected at the same 0.5-degree support, then downscaled from 0.5 to 0.05
degrees. IMERG remains an independent observation at its actual 0.4-degree
support. The 0.4-degree footprints need not coincide with the 0.5-degree latent
blocks: their differentiable observation operator acts on the reconstructed
fine field and can guide both `m` and `z`.

The current packed `cpc_precip` field was bilinearly interpolated onto the
0.05-degree grid. It must not be block-averaged and relabelled as native CPC for
this experiment. V3-SG preparation must retain native CPC cell centres and
edges and conservatively aggregate CHIRPS onto those exact footprints.

The `bd` lower edges (87.6 E, 20.3 N) are out of phase with the CPC grid, while
the `wide` upper edges do not close on complete CPC cells. Define a
V3-specific production domain on enclosing CPC edges and an inward-aligned
training domain that is an exact subset of the existing `wide` files:

```text
bd_cpc:    87.5--94.0 E, 20.0--27.0 N = (nlat, nlon) = (140, 130)
wide_cpc:  84.0--96.0 E, 16.0--28.0 N = (nlat, nlon) = (240, 240)
```

The inward choice discards only the northernmost and easternmost 0.8 degrees
of the legacy training halo. It retains the whole `bd_cpc` domain and the same
frozen 160-by-160 production canvas, closes exactly on 0.5-degree cells, and
reuses the existing 256-by-256 CHIRPS and DEM archives without resampling or
duplicating 44 years of input data.

All array shapes in this document use `(nlat, nlon)` order. Coordinate metadata
and dimension-name assertions remain mandatory at every file boundary.

Do not silently replace the existing project grids; store these as explicit
V3-SG grids with CPC cell-edge metadata. Model-B training crops must start at
fine-grid offsets divisible by 10. Their sizes must be divisible by both 10 and
`2^d`, where `d` is the number of U-Net downsamplings. With three downsamplings,
use a multiple of 40 such as 120 or 160 rather than the current arbitrary 128.
Production can use a CPC-aligned block halo to form an architecture-compatible
canvas and then retain the complete-block `bd_cpc` core. These are geometry
requirements, not scientific reasons to change the hierarchy to 0.4 degrees.

## Training data

### Fine target

The fine target is daily CHIRPS precipitation at 0.05 degrees in physical
mm/day. Training masks must continue to exclude ocean and invalid cells.

### Coarse target

For every training day, construct the target coarse field from the same CHIRPS
field on the native CPC footprint grid:

```text
m_truth = B10_cpc(CHIRPS_0.05)
```

This exact aggregation is essential. Regridded CPC must not be used as the
coarse training truth because CPC and CHIRPS differ in both magnitude and
pattern. The first model learns that correction explicitly.

### Predictor channels

Phase 0 should compare two deterministic predictor sets. The baseline retains
the current clean inputs:

- CPC precipitation and CPC validity for Model A;
- ERA5 TCWV, CAPE, U10, V10 and MSL;
- seasonal encoding; and
- the consistently prepared high-resolution static fields.

The maximal feasibility probe should also include the strongest physically
motivated placement predictors available before expensive generative training:

- adjacent-day atmospheric fields (`t-1`, `t`, `t+1`), since this is a
  reanalysis rather than a forecast;
- terrain-forced ascent `U10 dot grad(h)`, its magnitude and a smoothed version;
- low-level convergence and an onshore-wind component decayed by distance to
  coast;
- a training-period-only monthly CHIRPS allocation climatology; and
- if preparation is affordable, ERA5 vertical motion, 0--6 km shear and a
  CAPE-by-shear interaction.

The climatological allocation field is both a strong null and a candidate
conditioning channel. Supplying it does not guarantee improvement, so retain a
matched arm without it and measure the incremental daily information. All
climatologies must be built from training years only.

ERA5 is approximately 0.25 degree and therefore resolves roughly 2 by 2 values
inside a 0.5-degree CPC block. Predeclare the hypothesis that day-varying
placement skill will be strongest in the 0.25--0.5-degree band and will decline
below 0.25 degrees except where dynamic--static interactions such as terrain
forcing add information. A scale-resolved skill curve will test that mechanism.

IMERG should not be both a conditioning channel and an assimilated observation
in the same arm. A later native-IMERG or infrared-conditioning experiment must
either remove IMERG from the DA likelihood or explicitly model the dependence;
otherwise the same information is counted twice.

### Temporal split

Retain the existing independent split:

- training: 1981--2018;
- validation: 2019--2020;
- test/confirmation: 2021--2024.

The May 1--10, 2022 selection period must remain excluded from confirmatory
claims.

### Required feasibility checks before model training

Two checks are cheaper than training and can invalidate the intended endpoint:

1. **Gauge power.** Extract the existing variogram-based point-to-footprint
   representativeness estimates at factors 8 and 10 (rerun script 35 only if
   the report is unavailable). Use the actual fold/day/station dependence in a
   bootstrap or simulation to estimate the minimum detectable improvement in
   withheld-gauge subgrid-anomaly CRPS. The difference between two fitted
   representativeness variances is only a rough diagnostic, not a substitute
   for this power calculation.
2. **CHIRPS station leakage and static structure.** Audit whether BMD/GTS
   stations contribute to CHIRPS over the domain. Build a CHIRP (satellite-only)
   sensitivity where possible. In every case, compare against a monthly
   CHIRPS allocation-climatology null so a static ratio field cannot be called
   day-varying downscaling.

If independent gauge verification is too underpowered, it remains a valuable
secondary endpoint but cannot be the sole pass/fail gate. If the model cannot
beat the allocation-climatology null on held-out dates, the defensible result
is climatological disaggregation rather than daily subgrid prediction.

## Model A: coarse rainfall amount

The first conditional model represents

```text
p_theta(m_0.5 | CPC, ERA5, season, static)
```

Its target is `m_truth = B10_cpc(CHIRPS)`. It is responsible for:

- CPC-to-CHIRPS magnitude and bias correction;
- broad rainfall location;
- coarse wet/dry occurrence;
- calibrated coarse uncertainty.

Do not reuse the current 128-by-128 ADM-style U-Net unchanged. The native
`bd_cpc` field is only 14 by 13 coarse cells and `wide_cpc` is 24 by 24, so the
existing three-downsampling attention architecture is inappropriate. Begin
with a shallow two-level **coarse rectified flow over hurdle latents**: a
dequantised wetness logit and a transformed positive-amount latent. Both are
flow state channels with learned velocities; physical rainfall is decoded by
the hurdle rule. Reuse the target construction, dry/wet loss conventions and
calibration diagnostics from the existing CPC-v3 hurdle ablation, but do not
pretend its Bernoulli classification head is already a flow-state velocity.
A heteroscedastic Gaussian alone is not an acceptable precipitation distribution
because it cannot represent the atom at zero.

Construct the coarse occurrence-logit target with the same pinned dequantisation
principle used for the fine wetness target, and construct the amount latent only
for positive `m_truth` using the frozen square-root transform; dry blocks use
the pinned finite transform of zero, which the decoder ignores when occurrence
is dry. The hurdle decoder
maps the two continuous terminal channels back to an exact zero or nonnegative
amount, so Phase-1 samples and the coupled coarse state share an identical
representation.

If a stochastic coarse flow adds validation value, either use a purpose-built
shallow flow or train over a larger CPC-aligned South Asia/Bay-of-Bengal canvas
and crop to `bd_cpc`. Freeze this architecture choice in Phase 1 before coupling
Models A and B. Because Model A and the coupled coarse branch use the same
hurdle-latent state and velocity interface, coarse-branch weight transfer is
literal rather than aspirational. A deterministic hurdle baseline must be
retained so the value of stochastic coarse uncertainty is identifiable.

Square-root precipitation transformation remains the starting choice because
the CPC-v2 experiment showed that the complete sqrt-based package produced a
much better background. The coarse model still needs independent dry/wet and
tail diagnostics; transform choice should not be inferred from CRPS alone.

## Model B: stochastic subgrid allocation

The second conditional model represents

```text
p_phi(z_0.05 | m_0.5, ERA5, season, static)
```

It does not relearn the total rainfall amount. It learns the spatial allocation
within every 0.5-degree CPC footprint.

### Allocation target

Two equivalent parameterizations should be prototyped cheaply before the full
run:

1. **Zero-block-mean residual**

   ```text
   r_truth = CHIRPS_0.05 - U(B10_cpc(CHIRPS_0.05))
   ```

   This is easy to diagnose but needs a positivity repair after reconstruction.

2. **Positive block allocation (recommended)**

   The model predicts a wet mask and positive unnormalised cell weights. The
   weights are normalised within every coarse block. This guarantees positive
   precipitation and avoids an after-the-fact clipping step that changes the
   block total.

For the recommended parameterisation, make the latent target reproducible. In
each positive block define `w_truth` from the frozen rain threshold and set the
positive relative weight to `q_truth_i = x_i / m_block` on wet valid cells;
area-normalise it to remove round-off. Store an intensity logit such as
`log(q_truth_i + epsilon)` with its area-weighted wet-cell block mean removed,
because overall log-weight scale is unidentifiable after normalisation. Convert
the binary wet target to finite logits using a pinned dequantisation epsilon and
small seeded logistic noise. These intensity and wetness logits together are
`z_truth`. Save epsilon, threshold and dequantisation seed in the dataset
metadata; changing any of them defines a different target.

### Hard mass-conserving reconstruction

For a wet block, let `a_i` be the physical area of valid fine cell `i`,
`A = sum_i(a_i)`, and

```text
q_i = wet_i * softplus(z_i)
x_i = A * m_block * q_i / sum_j(a_j * q_j)
```

Then

```text
sum_i(a_i * x_i) / A = m_block
```

exactly. In a fully covered block there are 100 fine cells, but the area weights
still account for latitude. For a coastal or partially invalid block, use the
same valid-cell mask and area weights in `B10_cpc`, target construction and
reconstruction; exclude blocks that fail a predeclared valid-area threshold.
If `m_block` is zero, every valid fine cell is set to zero. If a sampled wet mask
is empty in a positive block, the implementation must deterministically activate
its highest-probability valid cell before normalisation.

The primary valid-area threshold is 0.50, matching the project's existing gauge
coverage convention, with 0.25 and 0.75 as geometry sensitivities. Freeze it
before inspecting which station names or scores are removed. Phase -1 must
report retained block, withheld-station and station-day counts overall and in
the coastal stratum for all three thresholds. Within-block latitude weighting
is numerically small over Bangladesh; consistent valid-cell masking is the
load-bearing part of coastal conservation.

The wet-mask head is important because positive weights alone make every fine
cell in a wet coarse block slightly wet. The target mask should be derived from
CHIRPS using a documented physical threshold and evaluated separately from
intensity. Wetness logits are part of `z`, not an external post-processing
variable. The fine branch must receive `m_block` explicitly, because the
expected wet-cell fraction depends strongly on block rainfall amount; predicting
wet fraction before cell placement is a reasonable implementation.

During joint DA, use an explicitly **hard-forward, soft-backward**
straight-through wetness estimator:

```text
w_soft = sigmoid(logit / tau)
w_hard = 1[w_soft >= 0.5]
w_st   = stop_gradient(w_hard - w_soft) + w_soft
```

The likelihood and reconstruction therefore see `w_hard`, including exact
zeros and exact area-weighted reprojection, while gradients follow `w_soft`.
The archived final field uses the same hard decoder and must equal the final
likelihood-forward field within numerical tolerance; a separate post-hoc
hardening operation is not allowed. This remains a declared biased-gradient
approximation to a discrete posterior, not an exact Bernoulli score. Freeze the
relaxation temperature on validation data and report sensitivity. Holding the
sampled background mask fixed during DA is also not allowed because a dry-cell
positive innovation would then have no occurrence pathway.

The normalisation denominator requires a documented positive floor, gradient
clipping and a finite-gradient test for marginally wet blocks. Hard block
normalisation remains the preferred starting design, but a soft-conservation
fallback is predeclared: predict the fine field with a conservation penalty and
apply exact projection during DA. Choose the fallback only if the hard model
shows conservation-scale seams, unstable joint-DA gradients or unacceptable
observation-fit damage from occurrence decoding.

### Why a generative model is required

The same coarse rainfall field can correspond to many valid fine patterns. A
deterministic model estimates their conditional mean and is expected to blur
or weaken convective structure. The allocation model therefore retains random
noise and produces an ensemble of plausible patterns. Spatial structure must
be evaluated on individual members as well as on the ensemble mean.

## Coupled operational prior

Separate Models A and B define the hierarchical background factorisation

```text
p(m, z | c) = p_theta(m | c) * p_phi(z | m, c)
```

where `c` contains CPC, ERA5, seasonal and static conditioning. The corresponding
joint score contains a cross term:

```text
grad_m log p(m,z|c) = grad_m log p_theta(m|c)
                      + grad_m log p_phi(z|m,c)
grad_z log p(m,z|c) = grad_z log p_phi(z|m,c)
```

A coarse flow and a conditional allocation flow trained separately do not
provide the second term merely by being run together. Autodifferentiating a
conditional velocity with respect to `m` is also not the same as differentiating
its log density. The final DA implementation must not silently drop this term.

The primary solution is a **coupled multiscale rectified flow** over the paired,
normalised state `s = (m, z)`. Here `m` includes the coarse hurdle occurrence
and positive-amount latents, while `z` includes allocation and fine wetness
logits. It has a shallow coarse branch, a fine allocation branch and cross-scale
feature exchange, and outputs both velocity components:

```text
u_psi(m_t, z_t, t | c) = (u_m, u_z)
```

Train it on the paired coarse hurdle-latent target decoded to `m_truth` and the
fine `z_truth`, initialising its branches from Models A and B. Its joint velocity
supplies the joint rectified-flow score, including
the learned amount--allocation dependence, while `R(m,z)` continues to enforce
physical conservation. Models A and B remain useful for oracle experiments,
initialisation and interpretable ablations; the coupled flow is the operational
background and DA prior.

If the coupled flow proves numerically infeasible, the correctness-preserving
fallback is posterior inference in the independent base-noise coordinates of
the two differentiable generators, with likelihood gradients propagated through
both complete transports. This is more expensive but retains the conditional
dependence. Making `z` independent of `m`, or dropping the cross term without a
quantified approximation study, is not an allowed production shortcut.

## Training sequence

### Phase -1: endpoint, geometry and nulls

Before fitting a V3-SG network:

1. complete the factor-8/factor-10 gauge-power and CHIRPS-leakage audits;
2. apply the predeclared rule selecting the gauge or held-out-CHIRPS primary
   endpoint;
3. derive and test `wide_cpc` and `bd_cpc`, including quantised crop origins;
   and
4. build the oracle and operational repeated-block, smooth and
   allocation-climatology nulls.

Failure of the alignment tests blocks training. An underpowered gauge endpoint
changes the claim ceiling and endpoint as declared below; it does not invalidate
the CHIRPS pattern-learning experiment.

The Phase -1 power audit must also simulate or bootstrap the achievable
precision for Gates 1--6: oracle and operational `Delta_CRPS`, V2
non-inferiority, coverage, spread--skill, retained-gain ratios and the authority
impulse contrast. A gate is
testable only if its confidence precision can resolve its frozen scientific
threshold on the planned confirmatory sample. If not, increase dates or members
where that actually improves precision; if adequate precision remains
impossible, mark the gate unresolved and lower the claim ceiling. Do **not**
widen a scientifically meaningful non-inferiority margin or acceptance band to
the observed minimum detectable effect merely to make the gate pass. This
decision is made once before Phase 0 and is never revisited after V3 results.

### Phase 0: oracle feasibility test

Train the allocation model using the exact target coarse amount as its input:

```text
input:  B10_cpc(CHIRPS), ERA5, season, static, noise
target: CHIRPS at 0.05 degrees
```

This answers the cleanest question: if the correct 0.5-degree amount is known,
can the model learn useful CHIRPS subgrid placement?

Run short deterministic residual probes with both the baseline and maximal
predictor sets. The stop decision is based on the maximal probe; the baseline
is an ablation that reveals how much the derived, adjacent-day and climatology
channels contribute. If the maximal probe cannot beat
`clim_ratio_null_oracle` in any predeclared terrain stratum, the available
predictors do not locate the daily
CHIRPS fine pattern. A full stochastic model may still reproduce the
distribution and spectra, but it cannot be expected to reconstruct the exact
daily placement.

Run a matched **maximal-minus-ERA5** probe and recompute the scale-resolved
skill curve. Attribute a 0.25--0.5-degree skill peak to ERA5 only if removing
ERA5 materially weakens that peak under the same split and null. If the peak
survives, interpret it as target-product or static-predictor structure rather
than evidence of an ERA5-resolved physical mechanism.

Also prototype a bounded-context allocation network using a 3-by-3 coarse-block
neighbourhood. Treat its many overlapping block-days as correlated examples,
not millions of independent samples. Promote it only if it matches the
field-level probe on held-out placement, cross-block object continuity and seam
diagnostics while materially reducing compute.

### Phase 1: coarse model

Train Model A from native CPC and ERA5 to `B10_cpc(CHIRPS)`. Validate its CRPS,
bias, wet/dry occurrence, Brier score and reliability, positive-amount tail,
calibration and spatial correlation at 0.5 degrees. Estimate the conditioning-
augmentation error distribution from held-out hurdle samples, retaining the dry
atom rather than fitting one Gaussian residual law. Save the coarse hurdle-
latent target transform and velocity-interface metadata required for literal
initialisation of the coupled coarse branch.

### Phase 2: oracle stochastic allocation

Train Model B using `m_truth`. Start with flow matching, the wet-mask loss and
hard reconstruction. Conditioning augmentation belongs in this phase rather
than in a later repair: randomly corrupt `m_truth` using an error distribution
estimated from Model A and pass the corruption level as a conditioning scalar.
Retain some clean `m_truth` examples so the oracle endpoint remains measurable.
The allocation branch keeps exactly this coarse-state-plus-corruption-level
interface after transfer into the coupled flow. In joint training the level is
`1-t` for the noisy coarse trajectory and zero for a clean coarse context; no
input channel or first-layer parameter is dropped or reinitialised. A pinned
batch must give identical Model-B and just-transferred joint fine velocities
before joint optimisation.
Do not initially add many texture losses: first determine what the correct
factorisation alone achieves.

### Phase 3: operational joint-flow coupling

Initialise the coupled multiscale flow from Models A and B and train it on paired
coarse amount and fine allocation states. Model A's held-out error distribution
calibrates the **sequential ablation's** conditioning augmentation; the
operational joint model instead observes its own noisy coarse state and the
matching `1-t` level directly. Joint training also presents a frozen fraction
of clean terminal coarse contexts with level zero. That trained mode is used by
the coupled oracle below and avoids treating naive trajectory replacement as an
exact conditional sampler. Verify that the coupled background
retains both Model A's coarse calibration and Model B's oracle placement skill.
Rerun Model A's complete Brier, reliability, positive-tail and CRPS diagnostics
on the coupled flow's decoded coarse marginal; calibration is not assumed to
survive weight transfer.
The oracle-to-operational loss quantifies how much signal is lost to coarse-input
error only when the comparison uses the coupled-flow oracle defined below.
Report the separate Model-B-oracle to coupled-flow-oracle difference as the
architecture/coupling cost; do not attribute it to coarse-input error.

Before any DA run, verify that the learned joint score changes allocation
statistics appropriately with `m`. A stress case that forces a large amount
increment must preserve the background relationship of wet fraction and
allocation entropy versus `m_block`. This test directly screens the cross term
that separate conditional flows would miss.

### Phase 4: optional spatial-loss refinement

Only if individual members remain deficient in fine-scale power, test a small
predeclared set of auxiliary losses:

- Laplacian-pyramid reconstruction at 0.05, 0.1 and 0.2 degrees;
- gradient or wet-boundary loss;
- batch radial-power-spectrum loss;
- variogram or patch-distribution loss.

These losses must not replace proper probabilistic scores. A sharp field can
have realistic texture while placing the rain incorrectly.

## Background generation

For each date and ensemble member:

1. Sample paired `(m,z)` from the coupled operational flow conditioned on CPC,
   ERA5 and the seasonal/static predictors.
2. Produce the fine wet mask from the coupled amount-aware head.
3. Reconstruct `x = R(m, z)` with exact block conservation.
4. Save `m`, `z` summaries and physical `x` so later DA and diagnostics can
   distinguish amount changes from allocation changes.

The sequential Model-A-then-Model-B generator remains an explicit ablation. It
must agree closely with the coupled flow before it can be used as a cheaper
unguided background substitute.

This is the unassimilated background and the only field used for a pure prior
downscaling claim.

## Hierarchical data assimilation

### DA state

The analysis state is the pair

```text
state = (m_0.5, z_0.05)
x_0.05 = R(m, z)
```

This notation suppresses the coarse and fine occurrence logits carried inside
the two branches. Both remain part of the guided continuous state; physical
zero rain is decoded only through the hurdle/relaxed-mask rules above.

The reconstruction must remain differentiable so observation gradients can
propagate to the correct latent component. The coupled flow supplies one joint
prior velocity and score for this multiscale state. The existing single-field
`sampler.assimilate` cannot obtain the required joint score by independently
calling Models A and B; `hierarchical_sampler.py` must evolve both branches and
apply their branch-specific normalisation and masks together.

### Observation operators

**IMERG S04**

```text
H_imerg(m, z) = B8_imerg(R(m,z))
```

IMERG therefore observes its true 0.4-degree area mean rather than being forced
onto the CPC hierarchy. Because the 0.4-degree IMERG and 0.5-degree CPC
footprints differ, its gradient legitimately contains information about both
the 0.5-degree amount `m` and some spatial allocation `z`. It still cannot
identify the complete 0.05-degree pattern inside an IMERG footprint.

The two regular grids repeat their relative overlap geometry every 2 degrees.
That does not inevitably create a stripe, but it can make the local information
content phase-dependent. Monitor total analysis-increment variance by phase
within this 2-degree cycle. An idealised uniform-innovation test must produce a
uniform interior increment;
phase-locked striping is a failed observation operator or sampler, not a
physical signal.

**BMD gauges**

```text
H_gauge(m, z) = bilinear_sample(R(m,z), station_locations)
```

The gauge likelihood constrains the physical 0.05-degree field at station
locations. Its gradient can alter the local allocation and, when supported by
the joint prior, the corresponding block amount.

### Simultaneous likelihood

The two streams enter one additive likelihood:

```text
log p(y | m, z)
    = log p(y_imerg | H_imerg(m, z), R_imerg)
    + log p(y_gauge | H_gauge(m, z), R_gauge)
```

The robust Huber-3 likelihood is the initial simultaneous configuration because
it performed best or near-best in the completed CPC-v2 experiments. IMERG and
BMD keep separate observation-error models and guidance schedules.

At every assimilated gauge and IMERG footprint, archive final `O-A` from three
decodes of the same final latent state: the counterfactual soft mask, the actual
hard-forward mask, and the saved/reloaded analysis. Standardise differences by
the effective observation standard deviation. The median absolute soft-to-hard
`O-A` change must be at most `0.10 sigma`, its 95th percentile at most
`0.50 sigma`, and the in-memory-hard versus reloaded-analysis difference at
most `1e-5 mm/day`. Select the relaxation temperature partly on these bounds,
not only gradient stability. If every validation temperature fails, use the
predeclared soft-conservation/continuous-occurrence fallback rather than
publishing a field different from the one whose fit was diagnosed.

Conceptually:

- CPC and ERA5 define the **0.5-degree background amount**;
- IMERG changes **0.4-degree observed area means**, guiding both nearby coarse
  amounts and the part of their allocation visible at that support;
- BMD changes **local 0.05-degree values and placement**;
- the learned joint prior propagates both constraints away from observations;
- hard reconstruction ensures that fine-cell adjustments remain consistent
  with the current, mutable 0.5-degree amount.

The final implementation must sample and guide `(m, z)` with the coupled joint
score. A sequential `m`-then-`z` path may be used only as a numerical smoke
test: with hard conservation, a gauge must be able to move `m` as well as
redistribute `z`, or an extreme point observation can be fitted only by creating
an implausible single-cell concentration. After assimilation, conditional wet
fraction and allocation entropy versus `m_block` must remain inside the
predeclared background reference envelope; conservation alone is insufficient.

## Experiment arms

The initial experiment should be deliberately compact:

| Arm | Coarse amount | Fine allocation | Observations | Question |
|---|---|---|---|---|
| `block_null_oracle` | exact CHIRPS 0.5 | repeated block mean | none | zero-subgrid oracle null |
| `smooth_null_oracle` | exact CHIRPS 0.5 | conservative smooth interpolation | none | benefit beyond removing block edges |
| `clim_ratio_null_oracle` | exact CHIRPS 0.5 | monthly CHIRPS allocation climatology | none | daily information beyond static climatology |
| `oracle_deterministic` | exact CHIRPS 0.5 | deterministic U-Net | none | predictable fine structure |
| `oracle_flow` | exact CHIRPS 0.5 | stochastic allocation flow | none | learnable CHIRPS distribution |
| `coupled_oracle` | coupled `m` branch clamped to exact CHIRPS 0.5 | coupled fine branch | none | architecture/coupling cost at exact amount |
| `smooth_null_operational` | same operational `m` members | conservative smooth interpolation | none | matched operational smooth null |
| `clim_ratio_null_operational` | same operational `m` members | monthly CHIRPS allocation climatology | none | matched operational climatology null |
| `operational_background` | coupled joint flow | coupled stochastic allocation | none | actual prior downscaling |
| `imerg_only` | coupled joint flow | coupled stochastic allocation | IMERG S04 | satellite area-mean correction |
| `gauges_only` | coupled joint flow | coupled stochastic allocation | BMD | point-driven amount/allocation correction |
| `simultaneous_huber3` | coupled joint flow | coupled stochastic allocation | IMERG + BMD | complete hierarchical DA |

The oracle arms diagnose the model and inputs; they are not deployable analyses.
For `coupled_oracle`, clamp the coarse state at every Euler/Heun/corrector stage
to its fixed linear-interpolant trajectory
`m_t = (1-t) * epsilon_m + t * m_truth`, using pinned noise, while sampling only
the fine branch, and provide the clean `m_truth` context with corruption level
zero through the conditioning mode used during joint training. The naive clamp
without that trained clean context is not accepted as a draw from
`p(z | m_1=m_truth)`.

Validate this approximation directly: take coarse fields produced by the joint
flow, condition the clean-context sampler on those fields, and compare its fine
distribution with the corresponding unclamped joint samples using allocation
entropy, wet fraction, scale-resolved residual variance and matched-coarse
`z` marginals. If the distributions disagree materially, use a back-and-forth
resampling conditional sampler and rerun this validation before retaining the
`coupled_oracle` gate.

### Frozen sampling and compute budget

Use 30 ensemble members for confirmatory CRPS, matching the completed V2
comparison. The coupled background and analysis use 50 Heun steps (99 velocity
evaluations) as the default. The simultaneous Huber-3 arm retains two Langevin
correctors per level only after the finite-gradient and joint-score stress tests
pass. Before Phase 3, run a pinned validation convergence check at 25, 50 and
100 Heun steps; retain 50 only if its mean CRPS differs from 100 by no more than
0.05 mm/day and its scale diagnostics remain inside Gate 3. Otherwise freeze
100 for all learned V3 arms before confirmation.

Smoke tests use at most 8 members and 25 steps and cannot support skill claims.
Null and oracle fields do not require five spatial folds; IMERG-only and
unguided backgrounds are generated once per date and reused. Only gauge-enabled
analyses require the five fold likelihoods. Cache shared coarse members,
conditioning tensors and observation operators so the arm table is not
implemented as an independent full rerun for every row.

## Evaluation

### CHIRPS-held-out evaluation

On held-out years, compare the oracle and operational backgrounds with CHIRPS
using:

- residual correlation after removing each field's own 0.5-degree CPC-aligned
  mean for the prior downscaling claim;
- the corresponding below-0.4-degree residual as a separate IMERG-support DA
  diagnostic;
- paired CRPS improvement against the repeated-block, smooth-interpolation and
  monthly allocation-climatology nulls;
- scale-resolved residual skill, with the predeclared expectation of strongest
  dynamic skill in the 0.25--0.5-degree band and a physically explainable
  decline below ERA5's approximately 0.25-degree support;
- Fractions Skill Score at multiple rain thresholds and spatial scales;
- radial power spectra and effective resolution;
- zonal and meridional spectra, including a check for power at the 0.5-degree
  conservation wavenumber;
- a seam index: mean absolute gradient across conservation-block boundaries
  divided by the same quantity in block interiors;
- variograms and residual variance by scale;
- wet-area fraction, object size and extreme-intensity distributions;
- ensemble CRPS, coverage, rank histograms and spread--skill;
- energy and variogram scores on fixed fine-grid patches as secondary proper
  scores of spatial dependence;
- individual-member texture and ensemble-mean coherent fraction.

For DA products, bin total physical-increment variance by phase in the 2-degree
IMERG/CPC overlap cycle. This remains a numerical-support/stripe diagnostic and
is separate from the observation-authority analysis below.

CHIRPS agreement demonstrates learning of the target product. It is not
independent proof of physical truth.

### Observation-authority attribution

The scale-aware division of observational authority is a designated primary
mechanistic result, not a latent-norm diagnostic. For background state
`(m_b,z_b)` and analysis state `(m_a,z_a)`, construct the four physical fields

```text
x_bb = R(m_b, z_b)      x_ab = R(m_a, z_b)
x_ba = R(m_b, z_a)      x_aa = R(m_a, z_a)
```

and use the symmetric two-factor decomposition

```text
C_m = 0.5 * [(x_ab - x_bb) + (x_aa - x_ba)]
C_z = 0.5 * [(x_ba - x_bb) + (x_aa - x_ab)]
C_m + C_z = x_aa - x_bb
```

This attributes the increment in mm/day while sharing the interaction fairly;
raw `m`- and `z`-latent norms are not comparable. Report signed maps, absolute
area-integrated shares, scale-resolved shares and distance from the assimilated
observation separately for `imerg_only`, `gauges_only` and
`simultaneous_huber3`.

Define the amount share in a fixed influence region as

```text
S_m = sum(|C_m|) / [sum(|C_m|) + sum(|C_z|)]
```

and pair the real-arm attribution with a controlled impulse-response experiment.
On the same representative held-out backgrounds, assimilate separately a
`+1 sigma` innovation at one IMERG footprint and one gauge, using the same
random seeds and no other observations. The primary authority contrast is
`S_m(IMERG impulse) - S_m(gauge impulse)`, with day/location block-bootstrap
uncertainty. A positive interval excluding zero is the designated evidence that
the area observation exercises relatively more amount authority than the point
observation; the complementary allocation result follows from the same
decomposition.

The magnitude of the allocation response to IMERG is a second required result.
Repeat the IMERG impulse with an otherwise identical 0.5-degree observation
operator aligned to the CPC blocks. Report whether
`1-S_m(IMERG 0.4-degree)` is materially larger than
`1-S_m(aligned 0.5-degree control)`. This control isolates allocation authority
created by the deliberate support mismatch from the nearly structural fact that
an area observation usually moves coarse amount more than a point gauge does.

The predeclared directional expectation is that IMERG produces the larger
amount contribution at coarse scales while its allocation contribution is
limited to structure identifiable from overlapping 0.4-degree means; gauges
produce the larger local allocation contribution, with a smaller block-amount
response permitted by the joint prior. The simultaneous arm should contain both
patterns without either observation stream erasing the other. This attribution
gets its own main-text figure. If these directions are not observed, report the
learned coupling honestly and remove or narrow the scale-aware-authority claim
in the Purpose and paper conclusions.

### Independent BMD verification

Use the established five matched folds. Every eligible station must be withheld
exactly once and excluded from every likelihood used to score it.

Report:

- daily, monthly and May--September CRPS, RMSE, bias and correlation;
- calibration and coverage;
- sub-footprint gauge anomalies on the 0.5-degree CPC support for the primary
  prior-downscaling claim, relative to matched CPC/CHIRPS baselines;
- the corresponding 0.4-degree IMERG-support anomalies as a separate DA-support
  diagnostic;
- paired anomaly CRPS improvement against the matched zero-subgrid and
  allocation-climatology nulls.

Positive withheld-gauge anomaly skill is the strongest available evidence that
the model places useful real-world subgrid structure rather than merely
imitating CHIRPS texture.

Report a matched-coarse-baseline anomaly variant and decompose error into
coarse-amount and within-footprint-placement components. Stratify subgrid
results by terrain gradient, distance to coast and rain-intensity class; these
are predeclared secondary analyses, not extra opportunities to redefine the
primary result.

### Primary endpoint and uncertainty

For every stochastic field, compute the predicted anomaly **member by member**:
fine-member value minus that member's area-weighted block mean on the designated
support. Score the resulting ensemble against the observed anomaly with CRPS.
A deterministic null is scored by the same formula, for which CRPS reduces to
absolute error. Operational nulls reuse the candidate's exact coarse-amount
members, so the comparison isolates allocation. Define the paired improvement
as

```text
Delta_CRPS = mean(CRPS_null - CRPS_candidate)
```

where positive values favour V3-SG. Ensemble-mean anomaly MSE is reported only
as a secondary, explicitly blur-favouring statistic. The same member-wise
anomaly and CRPS convention applies at gauges and at held-out CHIRPS grid cells.

Freeze the endpoint decision rule before model training. First define a
practically meaningful gauge-anomaly improvement using only existing v1/v2 and
observation-error information from confirmatory, non-selection dates only,
without looking at V3-SG scores. If the power audit shows that this improvement
is detectable with the available fold/day/station structure, the primary
endpoint is pooled five-fold withheld-gauge anomaly `Delta_CRPS` of
`operational_background` against
`clim_ratio_null_operational`.

If that practical effect is not detectable, the primary endpoint becomes
held-out CHIRPS daily residual-placement `Delta_CRPS` of
`operational_background` against `clim_ratio_null_operational`, both given the
same operational coarse-amount members. The withheld-gauge anomaly test becomes
the primary secondary endpoint, and the maximum claim is capped at **CHIRPS pattern
learning** unless independent evidence later resolves a real-data gain. This
fallback is selected by the pretraining power result, not by whether V3-SG wins.

The oracle comparison likewise gives `oracle_flow` and
`clim_ratio_null_oracle` the same exact CHIRPS 0.5-degree amount. Report
CHIRPS-, IMERG- and CPC-based anomaly definitions, but designate the one
supported by the power and leakage audit in advance. Confidence intervals must
preserve correlated weather days and station structure, using a day-block
bootstrap and a station-cluster sensitivity. All other spatial, spectral and
stratified diagnostics are secondary or exploratory.

### Production spatial fields

After method selection, generate one all-station analysis for maps, spatial
means, variability, spectra and product agreement. Never use this product for
independent station skill. The fold products remain the source of withheld
verification.

### V2 comparability

Freeze the existing five fold assignments, eligible-station IDs, scoring dates
and gauge QC rules for the V2--V3 comparison. Gauge scores sample both products
at exactly the same station coordinates and use only matched station--date
pairs; no fold re-optimisation is allowed.

The V3 `bd_cpc` grid contains the existing 128-by-128 `bd` grid as an exact
0.05-degree subarray: latitude indices `6:134` and longitude indices `2:130`.
Therefore gridded comparisons crop V3 to the old `bd` support without
interpolation. V3 IMERG files must be rebuilt on `bd_cpc` for assimilation, but
comparison diagnostics use matched physical footprints over the shared
interior. Archive both full V3 fields and the explicit `bd` comparison crop,
with coordinate assertions in the output metadata.

## Success gates

The following numerical thresholds are frozen before Phase 2 and are not tuned
on confirmatory results. The model supports a strong subgrid claim only if all
applicable gates hold:

1. Oracle `Delta_CRPS` against `clim_ratio_null_oracle` is positive and its
   two-sided 95% day-block-bootstrap interval excludes zero.
2. `coupled_oracle` `Delta_CRPS` against `clim_ratio_null_oracle` is positive
   with its 95% interval excluding zero. Operational `Delta_CRPS` against
   `clim_ratio_null_operational` is also positive with its interval excluding
   zero, and its point estimate retains at least one third of the
   `coupled_oracle` gain. The separate `oracle_flow` to `coupled_oracle`
   difference is reported as architecture/coupling cost and is not charged to
   coarse-amount error.
3. On held-out CHIRPS dates, the median individual-member radial-power ratio is
   within `[0.5, 2.0]` of CHIRPS in every predeclared octave from 0.05 to 0.5
   degrees, wet-area relative bias is no more than 20%, wet-day 95th- and
   99th-percentile ratios are within `[0.7, 1.3]`, 90% ensemble coverage is
   within `[0.85, 0.95]`, and the spread--RMSE ratio is within `[0.7, 1.3]`.
4. If the power audit declares the gauge anomaly endpoint adequately powered,
   independently withheld BMD anomaly `Delta_CRPS` against the designated
   matched null must be positive with its 95% interval excluding zero. If it is
   underpowered, gauge anomalies are supporting evidence and the claim ladder
   is capped one rung lower.
5. Against the frozen v2 simultaneous baseline on identical station--date
   pairs, the upper one-sided 95% bootstrap bound for
   `CRPS(V3-SG) - CRPS(v2)` is below the 0.10 mm/day non-inferiority margin.
   This margin is approximately 2% of the existing confirmatory V2 CRPS and is
   frozen before V3-SG results exist.
   The simultaneous analysis must also retain at least 80% of the background's
   positive anomaly `Delta_CRPS` against its matched climatology null.
6. For the scale-aware-authority claim, the controlled impulse-response
   contrast `S_m(IMERG) - S_m(gauge)` must be positive with its 95% interval
   excluding zero, the real single-stream arms must have the same direction,
   and `1-S_m(IMERG 0.4-degree)` must exceed the aligned 0.5-degree control by
   the frozen materiality threshold with its interval excluding zero.
   Failure does not erase rainfall skill, but it removes the claimed division
   of observational authority from the contribution.

If only texture and spectra improve, the correct claim is **stochastic texture
generation**. If CHIRPS residual placement improves but withheld-gauge anomaly
skill does not, the claim is **CHIRPS pattern learning**. Only improvement in
the independent gauge anomaly test supports **useful real-data subgrid
downscaling**.

## Proposed implementation layout

The exact paths may change during implementation, but the following separation
would keep the experiment auditable:

```text
configs/train_h100_cpc_v3_subgrid_coarse.yaml
configs/train_h100_cpc_v3_subgrid_allocation.yaml
configs/train_h100_cpc_v3_subgrid_joint.yaml
src/bdhires/models/hierarchical_subgrid.py
src/bdhires/data/subgrid_dataset.py
src/bdhires/da/hierarchical_sampler.py
scripts/56_build_chirps_subgrid_targets.py
scripts/57_train_subgrid_oracle.py
scripts/58_evaluate_subgrid_prior.py
```

Required tests should cover:

- exact area-weighted block conservation in physical units, including coastal
  partial blocks that pass the valid-area threshold;
- stable retained-station and station-day accounting at valid-area thresholds
  0.25, 0.50 and 0.75, reported separately for the coastal stratum;
- native CPC coordinate validation and exact CPC-edge alignment for both
  `wide_cpc` and `bd_cpc`;
- every training-crop origin is 0 modulo 10 and every crop size is divisible by
  both 10 and `2^d` for the chosen U-Net depth;
- nonnegative output and dry-block behavior;
- exact `z_truth` reproduction from the pinned wet threshold, epsilon and
  dequantisation seed;
- coastal/invalid block masking;
- differentiability through reconstruction;
- finite, bounded reconstruction gradients as `m_block` approaches zero,
  including a documented denominator floor and clipping rule;
- the empty-wet-mask fallback rate and its non-differentiable event handling;
- seam index near one and no axis-specific spectral spike at the conservation
  scale;
- IMERG likelihood acting through its independent 0.4-degree operator;
- a uniform full-coverage IMERG innovation producing a spatially uniform
  interior increment, with no 2-degree phase stripe;
- total analysis-increment variance remaining acceptably flat across the
  2-degree IMERG/CPC overlap phase in an idealised test;
- physical-space authority components satisfying `C_m + C_z = analysis -
  background` to numerical tolerance for every member, with separate outputs
  for IMERG-only, gauges-only and simultaneous arms;
- paired `+1 sigma` IMERG/gauge impulse experiments using identical backgrounds,
  seeds and fixed influence-region definitions;
- gauge likelihood acting on the reconstructed fine field;
- gauge likelihood gradients demonstrably reaching `m` as well as `z`;
- the coupled joint score changing wet fraction and allocation entropy
  consistently under a forced large `m` increment;
- `coupled_oracle` clamping the coarse interpolant after every Euler, Heun and
  corrector update, using the trained clean-context mode, recovering exact
  `m_truth` at the terminal state, and passing the matched-joint-distribution
  validation; naive replacement without clean context must fail explicitly;
- an explicit failure if a sampler attempts to combine separate Model-A and
  Model-B scores while omitting `grad_m log p_phi(z|m,c)`;
- the maximal-minus-ERA5 probe recomputing the identical scale-resolved metric
  and split used by the maximal Phase-0 probe;
- member-wise anomaly CRPS using each member's own block mean and returning
  absolute error for a one-member deterministic null;
- exact extraction of the legacy `bd` comparison crop from `bd_cpc` and
  unchanged fold/station IDs relative to V2;
- the coarse hurdle-latent decoder representing an exact dry atom and the
  conditional amount branch remaining nonnegative;
- literal Model-A-to-joint coarse-branch transfer with no unmatched intended
  parameters and identical pre-fine-tuning velocities on a pinned batch;
- literal Model-B-to-joint fine-branch transfer, including the retained coarse
  corruption-level channel, with identical pre-fine-tuning velocities on a
  pinned batch;
- a dry-cell positive innovation and wet-cell zero innovation both producing
  finite occurrence-logit gradients, followed by exact hard-mask conservation;
- wetness-relaxation temperature sensitivity frozen on validation data;
- final soft-mask, hard-forward and saved/reloaded `O-A` diagnostics enforcing
  the `0.10 sigma` median, `0.50 sigma` 95th-percentile and `1e-5 mm/day`
  serialization bounds;
- 25/50/100-step convergence metadata and enforcement of the frozen
  confirmatory step count;
- no IMERG double counting between conditioning and likelihood;
- exact reproducibility of the oracle and operational `smooth_null` and
  `clim_ratio_null` constructions from a pinned training-only CHIRPS version;
- deterministic reproduction from saved random seeds;
- output metadata identifying oracle, operational, CV and all-station scopes.

## Claim summary

The publishable model idea is not simply a sharper image generator. It is a
hierarchical probabilistic DA system with an explicit division of observational
authority:

> CPC and ERA5 define the large-scale prior, IMERG constrains rainfall amount at
> its independent 0.4-degree physical footprint, BMD gauges constrain local
> fine-scale values, and a CHIRPS-trained mass-conserving flow supplies
> calibrated uncertainty in the unresolved allocation between observations.

This framing matters because conditional diffusion for precipitation bias
correction and downscaling, and diffusion-prior posterior conditioning for
coarse climate information, already exist. V3-SG therefore should not claim
novelty for generative downscaling or posterior guidance alone. Its testable
contribution is the hierarchy of independent supports, joint amount/allocation
DA, and independent gauge-based evidence about effective subgrid resolution.

Relevant comparisons:

- [Aich et al. (2026), conditional diffusion for precipitation downscaling and
  bias correction](https://doi.org/10.5194/gmd-19-1791-2026)
- [Schmidt et al. (2025), probabilistic spatiotemporally coherent climate
  downscaling with a diffusion prior](https://doi.org/10.1038/s41612-025-01157-y)
