# BDhighresDA — Methodology

A 5 km daily precipitation reanalysis for Bangladesh. **ERA5 conditions a
generative prior; everything that actually measured rainfall — GPM IMERG and
BMD rain gauges — is assimilated as an observation at inference time.**

The approach follows Manshausen et al. (2025, *JAMES*) — a generative prior
over km-scale fields with observations assimilated zero-shot at sampling time —
with the diffusion prior replaced by a **conditional rectified-flow
(flow-matching) model**, which recent work finds gives better spatial skill at
about one third of the sampling cost (Wetherell 2026).

---

## 0. Pipeline overview — two phases, and only one of them sees observations

![BDhighresDA pipeline](figures/pipeline.svg)

**Phase 1 — training (offline, once).** ERA5 at 0.25° plus static fields
condition a flow-matching U-Net, which is trained against CHIRPS at 0.05° as
the target. That is the whole of training: coarse background in, 5 km
rainfall out. **No observation ever enters this phase.** What the network
learns is exactly `p(x_5km | ERA5)` — a downscaler, and nothing more.

**Phase 2 — inference (every day, no retraining).** For a given day, ERA5
conditions the sampler and the ODE is integrated from noise to a 5 km field.
Without observations, that gives the **background**: a plausible high-res
realisation consistent with ERA5. Turn observations on and the *same* frozen
network is used, but at every integration step the state is nudged by the
gradient of the observation likelihood — IMERG footprints and BMD gauges
together. That gives the **analysis**.

The separation is the point. Because the network never saw an observation,
you can add gauges, drop a satellite product, switch to IMERG Early for
near-real-time, or assimilate radar later — none of it requires retraining.
That is what "zero-shot data assimilation" means, and it is the property this
whole design exists to preserve.

---

## 1. The problem

| | |
|---|---|
| **Background** | ERA5, 0.25° (~28 km), 1940–present, hourly → daily |
| **Observation 1** | GPM IMERG Final, 0.1° (~10 km), 2000-06 → present, daily |
| **Observation 2** | BMD daily rain gauges, ~35 stations, 2020–2025 |
| **Prior training target** | CHIRPS v2.0 daily 0.05°, 1981–present |
| **Output** | 0.05° (~5 km) daily precipitation, 16-member ensemble |

ERA5 at 28 km cannot resolve the two features that dominate Bangladesh
rainfall: the Meghalaya/Shillong orographic barrier, which produces some of
the highest annual totals on Earth ~40 km north of the border, and the
mesoscale organisation of monsoon convection. IMERG sees *where* it rained at
10 km but carries known regime-dependent biases. The gauges are accurate but
sparse — roughly one per 4,200 km². The generative prior supplies the
fine-scale structure ERA5 lacks; IMERG constrains the pattern; the gauges
constrain the amplitude.

## 2. Why generative

A deterministic downscaler trained on MSE produces the *conditional mean*,
which for precipitation is over-smooth and systematically under-represents
extremes — exactly the values a flood-risk application needs. A generative
model samples from `p(x | ERA5)`, giving sharp fields and an ensemble that
quantifies the real ambiguity in the 28 km → 5 km map.

Assimilation then targets

```
p(x | y, ERA5)  ∝  p(x | ERA5) · p(y | x)
```

where `y` stacks the IMERG footprints and the gauge values. The prior handles
the first factor; the second is a Gaussian likelihood evaluated through a
differentiable observation operator. **The network is never trained on
observations.** This "zero-shot DA" property is the central claim of the
Manshausen paper and the main reason to prefer it over an end-to-end fusion
model, which must be retrained whenever the observing network changes — a
serious problem for a country whose gauge network has grown and shifted over
four decades, and for a satellite product that reissues major versions every
few years.

## 3. The prior

### 3.1 Conditioning: five ERA5 channels

The prior is conditioned on ERA5 and time-invariant fields. Nothing else — and
deliberately very little of ERA5.

| Channel | Question it answers |
|---|---|
| `tp` | How much rain did the background model itself produce? |
| `tcwv` | How much moisture is in the column? |
| `ivte` | How much moisture is being transported, |
| `ivtn` | and from which direction? |
| `cape` | Is the atmosphere unstable enough to convect it out? |

Plus three derived-for-free terms from the IVT pair — magnitude, and direction
as sin/cos so the network never sees the 0/360° discontinuity — and the
statics: sqrt-elevation, slope, land–sea mask, four lat/lon positional-encoding
channels, sin/cos day-of-year.

**Why so few.** With ~14,000 daily training samples, every additional channel
is capacity spent on something the network has to learn to ignore. These five
cover the actual causal chain for a daily rainfall total over the Bengal
delta: available moisture, its transport, the instability that converts it to
rain, and the background model's own estimate of the result. The monsoon flux
striking the Meghalaya barrier — which produces the domain's rainfall maximum —
is captured by the IVT pair together with the static orography.

They are also all **single-level**, so no ERA5 pressure-level request is
needed at all. The CDS download shrinks by roughly an order of magnitude,
which matters because the download queue, not training, is the long pole.

`ERA5 tp` deserves a note: it is a *model* field, the background's own
parameterised guess at rainfall, not a measurement. That is precisely why it
belongs in the prior and not in the likelihood.

**The extended set is an ablation, not a default.** `--extended` adds MSL,
t2m/d2m, CIN, convective precipitation, moisture-flux divergence, and
850/500 hPa `u/v/q/w` with derived shear. Turn them on only if validation CRPS
actually improves; report the comparison either way.

### 3.2 Rectified flow / stochastic interpolant

```
x_t = t·x₁ + (1−t)·x₀ ,   x₀ ~ N(0, I),   x₁ ~ p_data
```

The network `u_θ(x_t, t, c)` regresses the conditional velocity `x₁ − x₀`
(Lipman et al. 2023; Albergo et al. 2025), with `c` the conditioning stack
above concatenated on the channel axis.

**Training target `x₁` is CHIRPS 0.05° daily, 1981–2025.** Because the prior
never touches IMERG, it is not limited to the satellite era: it trains on the
full 44-year record in one stage, ~14,000 daily fields for 1981–2018 training.

### 3.3 Conditioning dropout gives two models for the price of one

10% of training samples have `c` zeroed. The same weights are therefore usable
as a **conditional downscaler** (pass `c`) and as an **unconditional prior**
`p(x₁)` (pass `c = None`) — the exact object Manshausen et al. train
separately. The latter gives you a clean ablation: unconditional prior +
observations only, isolating what ERA5 actually contributes.

### 3.4 From velocity to score, and back

With `x̂₁ = x_t + (1−t)u` and `x̂₀ = x_t − t·u`:

```
score(x_t) = ∇ₓ log p_t(x_t) = −x̂₀ / (1 − t)                       (A)
u(x_t)     = ( x_t + (1 − t)·score ) / t                            (B)
Δu         = ((1 − t) / t) · Δscore                                 (C)
```

Equation (C) is what lets the score-based DA guidance machinery drop straight
into a flow-matching sampler. Implemented in `src/bdhires/models/flow.py`,
unit-tested in `scripts/smoke_test.py`.

## 4. Observations

Both observing systems enter through one likelihood. There is no conditioning
path for either.

### 4.1 Operators

| Stream | n | Operator `H` | Notes |
|---|---|---|---|
| BMD gauges | ~35 | bilinear interpolation to lat/lon | `grid_sample`, differentiable |
| IMERG | ~3,500 | exact 2×2 block mean | 0.05° → 0.1° |

The IMERG operator is *exact*, not approximate: both the `wide` and `bd` grids
have edges on multiples of 0.1°, so each 0.05° cell nests perfectly inside one
IMERG footprint and the forward operator is a plain 2×2 average with no
interpolation error. (Verified numerically in the smoke test.)

Both operators act in **transformed** space (`log1p`, §5), so observations are
passed through the same transform as the target before comparison.

### 4.2 Likelihood and guidance

Following Rozet & Louppe (2024) Eq. (3) / Manshausen et al. Eq. (3):

```
p(y | x_t) = N( y | H(x̂₁),  R + Γ·(1−t)²/t² )
```

using `μ_t = t`, `σ_t = 1−t` for our interpolant. Observations are heavily
down-weighted early in the trajectory, when the state is mostly noise, and
approach their true error variance as `t → 1`. `Γ` is a scalar; Manshausen et
al. found `1e-3` better than the `1e-2` of the original SDA paper,
specifically for precipitation. The guidance gradient is taken **through the
network** (diffusion posterior sampling), so a guided sample costs ~2–3× an
unguided one — but that cost is independent of how many observations there
are, which is why the 3,500 IMERG footprints are essentially free on top of
the 35 gauges.

This is the direct analogue of the 3D-Var cost function

```
J = (x_b − x)ᵀB⁻¹(x_b − x) + (y − H(x))ᵀR⁻¹(y − H(x))
```

with the learned prior replacing the `B`-weighted background term — except the
prior is a full non-Gaussian distribution rather than a Gaussian with a
hand-tuned covariance, which is precisely why it can produce sharp,
non-Gaussian rainfall structure that 3D-Var and OI cannot.

### 4.3 Observation error

`R` is diagonal, with two contributions per stream:

- **measurement error** — small for gauges (~5%), larger and
  regime-dependent for IMERG
- **representativeness error** — the mismatch between what the instrument
  senses and what a model cell means. For a point gauge versus a 5 km cell
  average of daily convective rainfall this is the *dominant* term. For IMERG
  it is small, because `H` is exact.

Set them badly and the failure modes are opposite and both visible:
under-specify `R` and the analysis chases individual observations and grows
bullseyes; over-specify it and the observations do nothing. Tune on withheld
stations, and measure the IMERG term rather than guessing it
(`scripts/07_bias_correct_imerg.py --fit-error-model`).

The intended division of labour is **IMERG constrains the pattern, gauges
constrain the amplitude**, which in practice means σ_IMERG ≈ 3–5× σ_gauge.

### 4.4 IMERG must be de-biased first

A Gaussian likelihood assumes the observation is unbiased. IMERG is not: over
South Asia it over-detects light rain, and it underestimates orographic
rainfall along the Meghalaya barrier because passive-microwave retrievals miss
shallow warm-rain processes over land. A likelihood cannot discover and
correct that — it will faithfully pull the analysis toward the biased value.

So `scripts/07_bias_correct_imerg.py` fits, on the training years only, a
per-cell per-season quantile map from IMERG to CHIRPS with wet-day frequency
adaptation (IMERG's drizzle over-detection has to be removed before the
quantile map, or it propagates straight through). **Skipping this step is the
main way this design goes wrong.**

### 4.5 What this buys, relative to conditioning on IMERG

Feeding IMERG to the network as a predictor would also work, and a conditional
network could learn its biases implicitly. Assimilating it instead buys:

1. **20 extra years of training data** — the prior is not tied to the
   satellite era, so it trains on 1981–2018 in one stage rather than
   2001–2018, and the product can extend back to 1981 with gauges only.
2. **Version and latency independence** — swap IMERG V07 → V08, or Final
   (3.5-month latency) → Early (4-hour latency, larger `R`) for a
   near-real-time product, with no retraining.
3. **An inspectable weighting** — the satellite/gauge trade-off is an explicit
   number you tune and report, not something buried in network weights.
4. **Innovation diagnostics** — you get `y − H(x̂)` for IMERG, i.e. a map of
   where the satellite disagrees with the ERA5-informed prior. That is a
   publishable result on its own.

The cost is that the bias correction in §4.4 becomes mandatory, and that dense
observations can over-constrain the ensemble — see §6.

## 5. Precipitation transform

Daily rainfall has an atom at zero and a heavy tail. Both reference papers hit
tail problems from opposite directions: Manshausen et al. used log/exp and got
occasional unphysical extremes on inversion (their Appendix C); Wetherell
(2026) used sqrt and got a dry bias in the far tail.

`src/bdhires/transforms.py` implements `log1p`, `sqrt`, `cbrt` and `none`
behind one interface. **Default: `log1p` with ε = 0.1 mm** (roughly the BMD
reporting resolution), then standardisation on training-period statistics.
Treat the transform as a first-class ablation: run `log1p` vs `sqrt` and
compare the upper tail of the PDF and FSS at 100 mm/day.

## 6. Ensemble spread — designing against under-dispersion

**Every published generative-DA study reports under-dispersive ensembles.**
Manshausen et al. state it directly for their km-scale system and name the
mechanism: the Gaussian approximation used for the likelihood score is
mode-seeking. Assume this project will hit the same wall, and design against
it rather than patching it afterwards.

### 6.1 Where the spread is actually going

Five distinct mechanisms, worth separating because the fixes differ:

1. **The prior is under-dispersive.** Manshausen et al. Appendix C found
   generated states had lower variance than the training distribution. With
   ~14k training samples, strong EMA and a smooth denoiser, this is expected.
2. **Deterministic ODE sampling.** With the probability-flow ODE, *all*
   randomness comes from the `x₀` draw, and the learned flow is a smooth map
   that tends to contract that Gaussian onto the data manifold. Most
   generative-DA papers sample this way, and the toy test in
   `scripts/smoke_test.py` reproduces the contraction: a prior whose true sd
   is 0.50 samples at 0.43.
3. **Mode-seeking guidance.** Replacing the true `p(x₀|x_t)` with a Gaussian
   centred on the posterior mean pulls every member toward the same point.
4. **Unperturbed observations.** If all members assimilate the identical `y`,
   the analysis ensemble is too narrow — the exact same result as an EnKF with
   unperturbed observations, a textbook cause of variance collapse.
5. **Too many, too-confident observations.** ~3,500 IMERG footprints with a
   small `R` will pin the field almost everywhere and reduce the analysis to a
   deterministic downscaling of the satellite.

### 6.2 Fixes, in order of leverage

**(a) Perturbed observations — do this first.** Draw
`y_r = y + ε_r, ε_r ~ N(0, R)` independently per member. Free at inference,
statistically the right thing to do (each member is then a posterior draw
given a plausible realisation of the observation error), and it directly
removes mechanism 4. Implemented in
`bdhires.da.observation.perturb_observations`; on by default.

Crucially, the IMERG perturbations must be **spatially correlated**. Satellite
retrieval errors decorrelate over tens of kilometres; white noise on a 0.1°
field averages out over any neighbourhood and adds essentially no spread at
the scales anyone cares about. `error_corr_cells: 3.0` (~30 km) is the
starting point.

**(b) Prior temperature — the knob that actually inflates.** Sampling the
tempered density `p^(1/T)` means using `score/T`. Converting that to a
velocity through Eq. (B) gives, exactly,

```
u_T = u_θ + (1 − 1/T) · x̂₀ / t
```

— one extra term, monotone in `T`. Measured on the toy prior in the smoke
test (true sd 0.50): `T = 1.0 → 0.43`, `1.25 → 0.56`, `1.6 → 0.71`,
`2.0 → 0.84`. Note that `T = 1` already under-disperses, which is the
literature's finding reproduced in miniature.

Inflating the **prior** rather than the analysis is the right place to do it.
The observations then pull members back wherever they exist, so spread grows
where the field is unconstrained and stays tight where it is observed — which
is exactly the behaviour a reanalysis should have, and is not what post-hoc
inflation of the analysis gives you.

**(b′) SDE sampling — worth having, but not for this.** For any `g_t ≥ 0`,

```
dx = [ u_t(x) + (g_t²/2)·score_t(x) ] dt + g_t dW
```

has the **same marginals** as the probability-flow ODE (Albergo et al. 2025).
With `g_t² = 2η(1−t)` the drift correction collapses to `−η·x̂₀`, so the
update is `x ← x + dt·(u − η·x̂₀) + √(2η(1−t)dt)·z`. Its purpose is to stop
integration error and mode-seeking guidance compounding along a trajectory —
**not** to widen the ensemble, and in a small test here it slightly *narrowed*
it. Keep `η` as a tunable; do not rely on it for dispersion. Reporting this
honestly is worth more than an extra knob.

**(c) Background uncertainty from the ERA5 EDA.** ERA5 ships a 10-member
ensemble of data assimilations at 0.5°. Conditioning different analysis
members on different EDA members propagates *background* uncertainty into the
ensemble — a physically meaningful source that pure sampling noise cannot
represent, and which nobody in the generative-DA literature is currently
using. `scripts/00_download_era5.py --ensemble`; enable with
`ensemble.era5_eda: true`.

Together (a)–(c) give three separate, physically interpretable spread sources:
**downscaling ambiguity** (the x₀ draw, widened by `T`), **background error**
(EDA members), and **observation error** (perturbed obs). That decomposition
is itself a contribution — most published systems have only the first, and
even that only through the x₀ draw.

**(d) Loosen the guidance.** Larger `Γ` inflates the assumed likelihood
variance and softens the pull toward observations. `Γ ∈ {1e-4, 1e-3, 1e-2}` is
in the tuning grid. Correctly-sized `R`, especially the representativeness
term, does the same job more honestly.

**(e) Epistemic spread.** Keep U-Net dropout active at sampling time
(`inference_dropout`), or sample across several EMA checkpoints / training
seeds. Cheap, and it captures model uncertainty that trajectory noise cannot.

**(f) Post-hoc calibration — last resort.** `bdhires.eval.calibration` provides
multiplicative variance inflation and rank-based quantile recalibration.
Inflation fixes the second moment and nothing else; recalibration fixes the
distribution shape but needs a decent validation sample. **Always report the
uncalibrated numbers alongside.**

### 6.3 Measure it properly

Two things that make a calibrated ensemble look broken:

- **Forgetting the observation error.** When comparing spread against RMSE at
  gauges the relation is `MSE(mean, obs) ≈ spread² + σ_obs²`. Omit `σ_obs²` and
  a perfect ensemble looks under-dispersive. Likewise, rank histograms need
  observation error added to the members before ranking — Manshausen et al.
  Appendix D does this.
- **Averaging over intensity.** Under-dispersion is almost never uniform:
  generative priors are usually acceptable for light rain and badly
  under-dispersive for extremes, which is the regime a flood application cares
  about. `spread_skill_by_bin` reports it stratified by observed intensity.

`calibration_report()` returns overall and per-intensity spread/skill, the
rank histogram, a scalar flatness deviation, and the inflation factor that
would be needed — everything for the calibration figure in one call.

### 6.4 Also watch for

- **Assimilation bullseyes.** Manshausen et al. Figure C1 shows precipitation
  increments concentrated only at assimilated stations, because their prior
  was too dry so guidance only ever nudged upward. Plot the time-mean
  increment map; a healthy analysis spreads increments over coherent
  meteorological structures, not discs around gauges.
- **Ocean leakage.** CHIRPS is land-only. The valid mask is applied in the
  loss, the sampler and the output. Do not remove it.

## 7. Data volume

| Split | Period | Days | Purpose |
|---|---|---|---|
| Train | 1981-01 → 2018-12 | ~13,900 | prior |
| Validation | 2019 → 2020 | ~730 | early stopping, DA hyperparameters |
| Test / product | 2021 → 2025 | ~1,800 | evaluation with real BMD gauges |

Three mitigations for the fact that ~14k daily fields is small for a
generative model, all implemented:

1. **Random 128×128 crops of a 256×256 wide domain** (84–96.8°E, 16–28.8°N).
   Multiplies the effective sample count and exposes the model to a wider
   range of rainfall regimes — Bay of Bengal, Meghalaya, Arakan, Gangetic
   plain. Absolute position is fed in through static sin/cos channels so
   location-specific climatology remains learnable. No flips: they would
   destroy the orography–rainfall relationship.
2. **The full 44-year record**, available precisely because the prior does not
   depend on IMERG.
3. **Strong EMA (0.999), dropout 0.1, cosine LR.** Monitor validation
   flow-matching loss, and check unconditional samples against the CHIRPS
   climatology — the "climate of the model" diagnostic of Manshausen
   Appendix C.

If skill is still limited, the next lever is **more domain, not more years**:
extend the wide grid over South Asia, train one model, evaluate over
Bangladesh.

## 8. Time alignment

The most common silent bug in this kind of study.

- CHIRPS day D is 00–24 UTC.
- ERA5 `tp` is a *backward* hourly accumulation, so day D = sum of steps
  01:00(D) … 00:00(D+1). The packing script shifts by −1 h before resampling.
- IMERG `3IMERGDF` is already 00–24 UTC but stores a **rate in mm/hr** —
  multiply by 24.
- State variables are averaged over 00–24 UTC of day D.

Verify by correlating ERA5 `tp` with CHIRPS at lags −2…+2 days. The peak must
be at lag 0.

## 9. Experiment plan

1. **Pseudo-observations first.** Sample CHIRPS at the 35 BMD coordinates over
   the whole record and assimilate those. The true full field is known, so
   this validates the whole DA machinery and lets you tune `Γ`, `σ_obs` and
   `noise_scale` cleanly (Manshausen §4.1). Sweep gauge density
   (5/10/20/35/100) to quantify what a denser BMD network would buy — directly
   policy-relevant.
2. **Real gauges, 3-fold cross-validation.** Rotate the withheld third; report
   RMSE, MAE, CRPS, spread/skill overall and by intensity, and rank
   histograms at left-out stations. Manshausen et al.'s headline was **10%
   lower RMSE at left-out stations**; that is the number to beat.
3. **Ablations**: unconditional prior + obs (what does ERA5 add?); gauges only
   vs IMERG only vs both (what does each observing system add?); IMERG with
   and without bias correction; `log1p` vs `sqrt`; Γ; `noise_scale`; perturbed
   vs unperturbed observations; ensemble size 1/16/64.
4. **Baselines** at the same withheld stations: raw ERA5 bilinear, IMERG,
   CHIRPS itself, quantile-mapped ERA5, ordinary kriging of the gauges, and a
   deterministic U-Net. CHIRPS is a *strong* baseline over Bangladesh because
   it already blends gauges — beating it at withheld stations is the real bar.
   The honest framing is that this method adds (i) daily ensemble uncertainty,
   (ii) ERA5 dynamical information CHIRPS ignores, and (iii) the ability to
   assimilate observations CHIRPS never saw.
5. **Verification**: point scores at stations; FSS at 1/10/20/50/100 mm/day
   across 5–165 km neighbourhoods; SAL; POD/FAR/CSI/ETS; PDF and tail
   comparison; monsoon-onset composites; spatial bias and increment maps.

One caution worth checking early: CHIRPS already blends GTS gauges, some of
which are BMD stations. If a withheld station is inside CHIRPS, evaluating
against it is partly circular. Check the CHIRPS station list before fixing the
evaluation set.

## 10. Roadmap beyond precipitation

The architecture is multivariate-ready: add `t2m`/`tmax`/`tmin` output
channels and the same DA machinery assimilates temperature gauges with no code
change (Mishra South Asia 5 km is the natural target). Multivariate training
also buys physical consistency — Manshausen et al. showed their model learned
wind–rain relationships and could infer unobserved channels from observed
ones. Longer term: sub-daily via IMERG half-hourly, and a 4-D version where
the prior is over sequences rather than snapshots.

---

## References

- Manshausen, P., et al. (2025). Generative data assimilation of sparse weather
  station observations at kilometer scales. *JAMES*, 17, e2024MS004505.
  <https://doi.org/10.1029/2024MS004505> · <https://arxiv.org/abs/2406.16947>
- Rozet, S., & Louppe, G. (2024). Score-based data assimilation.
  <https://arxiv.org/abs/2306.10574>
- Lipman, Y., et al. (2023). Flow matching for generative modeling.
  <https://arxiv.org/abs/2210.02747>
- Albergo, M. S., Boffi, N. M., & Vanden-Eijnden, E. (2025). Stochastic
  interpolants: a unifying framework for flows and diffusions.
  <https://arxiv.org/abs/2303.08797>
- Wetherell, T. (2026). Flow matching for convective-scale precipitation
  downscaling. <https://arxiv.org/abs/2606.00281>
- Mardani, M., et al. (2024). Residual corrective diffusion modeling for
  km-scale atmospheric downscaling (CorrDiff).
  <https://arxiv.org/abs/2309.15214>
- Fotiadis, S., et al. (2024). Stochastic flow matching for resolving
  small-scale physics. <https://arxiv.org/abs/2410.19814>
- Chen, Y., et al. (2025). FlowDAS: a stochastic interpolant-based framework
  for data assimilation. NeurIPS 2025. <https://arxiv.org/abs/2501.16642>
- Karras, T., et al. (2022). Elucidating the design space of diffusion-based
  generative models (EDM). <https://arxiv.org/abs/2206.00364>
- Fortin, V., et al. (2014). Why should ensemble spread match the RMSE of the
  ensemble mean? *Journal of Hydrometeorology*, 15, 1708–1713.
- Zamo, M., & Naveau, P. (2018). Estimation of the continuous ranked
  probability score with limited information. *Mathematical Geosciences*, 50.
- Funk, C., et al. (2015). The climate hazards infrared precipitation with
  stations (CHIRPS). *Scientific Data*, 2, 150066.
- Huffman, G. J., et al. (2023). GPM IMERG Final Precipitation L3 1 day V07.
