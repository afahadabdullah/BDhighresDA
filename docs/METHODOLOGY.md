# BRISHTI-05 and SURMA-Flow methodology

**BRISHTI-05** is the public name for the Bangladesh daily precipitation
analysis produced by this repository. It means **Bangladesh Rainfall
Integration of Satellite, Hydrometeorological, and Terrestrial Information at
0.05°**; *brishti* is Bangla for rain. The model is **SURMA-Flow v1.0**:
**Score-guided Unified Rainfall Modeling and Assimilation with Rectified
Flow**.

The current released lineage is implemented under the historical machine key
`v2_simul_s04_ig010`. That key is retained in scripts, checkpoint paths, and
archived Zarr metadata solely for reproducibility. In publications, figures,
and user-facing data, call the model **SURMA-Flow v1.0** and the analysis
product **BRISHTI-05**.

BRISHTI-05 is a **daily 0.05° (about 5 km) ensemble analysis over Bangladesh**.
It combines a learned fine-scale prior conditioned on the coarse CPC rainfall
analysis and ERA5 atmospheric state with inference-time assimilation of BMD
gauges and GPM IMERG. It is an analysis product, not an observation and not a
gridded truth data set.

## 1. Product contract

| Item | Current BRISHTI-05 production contract |
|---|---|
| Fine grid | 0.05° Bangladesh grid, with a permanent land/valid mask |
| Public product | Daily physical precipitation ensemble, ensemble mean, and ensemble SD |
| Ensemble size | 30 members for the 2021–2024 production archive |
| Learned fine-scale target | CHIRPS daily precipitation at 0.05° |
| Prior conditioning | Original CPC daily precipitation at 0.5°, CPC validity, ERA5 state, static fields, and seasonal encoding |
| Inference-time observations | BMD daily gauges and prepared GPM IMERG Final |
| Selected production method | SURMA-Flow v1.0 simultaneous S04 analysis (legacy key `v2_simul_s04_ig010`) |
| Production archive | May–September 2021–2023 and May–June 2024 (520 BMD-labelled days) |

The product must be interpreted at the support of each datum. BMD is a point
measurement, CPC is a coarse conditioning analysis, IMERG is assimilated as a
0.4° footprint observation in the production method, and CHIRPS is the
fine-resolution training target. None of CPC, IMERG, or CHIRPS is designated
as gridded truth in real-data evaluation.

## 2. Separation of learning, conditioning, and assimilation

The three information roles are intentionally different.

1. **Training target — CHIRPS.** The flow model learns the distribution of
   fine-scale daily fields from CHIRPS. CHIRPS is never treated as independent
   verification for this model family.
2. **Conditioning — CPC and ERA5.** The frozen network receives CPC rainfall,
   CPC coverage, ERA5 atmospheric fields, static geography, and season. CPC is
   therefore part of the model background, not an observation added by the DA
   likelihood.
3. **Assimilation — BMD and IMERG.** At inference, the sampler adds likelihood
   guidance from BMD and IMERG. Neither BMD nor IMERG is fed as a learned
   conditioning channel or used as a training target.

This distinction is essential. BRISHTI-05 is not a purely ERA5-conditioned
downscaler, and it is not an end-to-end neural fusion model of all observing
systems. It is a CPC- and ERA5-informed generative prior whose posterior is
updated with available BMD and IMERG observations.

In compact notation, the analysis samples a field conditional on the
background inputs and observations: `p(fine rain | CPC, ERA5, static, BMD,
IMERG)`. The learned network provides the first part of that distribution;
the observation likelihood supplies the BMD and IMERG update during sampling.

## 3. SURMA-Flow v1.0 learned prior

### 3.1 Inputs and target

The SURMA-Flow v1.0 checkpoint (stored internally as `prior_h100_cpc_v2`) is
trained with the configuration in
[`configs/train_h100_cpc_v2.yaml`](../configs/train_h100_cpc_v2.yaml). Its
seven dynamic conditioning channels are:

| Source | Channels | Role |
|---|---|---|
| CPC Global Unified | `cpc_precip`, `cpc_valid` | Coarse rainfall amount and availability at the original 0.5° support |
| ERA5 | `tcwv`, `cape`, `u10`, `v10`, `msl` | Moisture, instability, low-level flow, and synoptic state |

Static terrain, land/valid-mask and positional features are packed with the
dynamic channels; sine/cosine day-of-year encoding is appended at sampling.
The packing code records the exact channel order in the Zarr and statistics
metadata, and inference reads that metadata from the checkpoint rather than
assuming a generic configuration.

The target is daily CHIRPS precipitation at 0.05°. CPC and CHIRPS
precipitation are transformed with the square-root transform before
normalisation. The network predicts the standardised residual between
transformed CHIRPS and transformed CPC; after sampling, that residual is added
back to the CPC baseline and inverse-transformed to physical mm/day. Thus a
zero residual reproduces the CPC background on the fine grid, while the flow
learns the distribution of plausible 0.05° corrections and texture.

This residual formulation means that BRISHTI-05 carries CPC's broad rainfall
constraint even before any BMD or IMERG assimilation. It also means CPC is
not an independent comparison product for the BRISHTI-05 background or
analysis.

### 3.2 Training procedure

SURMA-Flow v1.0 uses a conditional rectified-flow U-Net with multiscale
conditioning. It is trained on 1981–2018, validated on 2019–2020, and the
2021–2025 interval is reserved for the downstream BMD-era experiments. The
configuration uses 128×128 crops on the wide South Asia domain, wet-day and
wet-crop oversampling, dropout, EMA weights, and a logit-normal time sampling
schedule. A weak 0.5° consistency loss checks both the target and CPC-scale
amount during training.

The model represents a distribution, not a deterministic regression. Each
ensemble member starts from an independent noise draw and is integrated to a
physical 0.05° precipitation field. The learned flow is frozen before data
assimilation; no BMD or IMERG observation updates its weights.

## 4. Observation preparation and time contract

### 4.1 BMD gauges

BMD daily reports are converted from the canonical station files and screened
for coverage. For verification runs, stations are split into five spatially
distributed folds. Stations in the held-out fold do not enter the likelihood;
they are used only for scoring. In the all-station production archive every
eligible station is assimilated, so its subsequent fit is a diagnostic of
observation adherence, not independent validation.

### 4.2 IMERG

IMERG is prepared from half-hourly retrievals into the BMD 03:00–03:00 UTC
daily window. The files retain precipitation, random retrieval error, and
retrieval-count fields. The production launcher validates exact dates, units,
window end time, and the nested footprint grid before a GPU run starts.

For BRISHTI-05 production, the prepared IMERG field is coarsened to **S04**:
one 0.4° footprint is the exact mean of an 8×8 block of 0.05° cells. The
sampler uses every valid S04 footprint (stride 1). This is deliberately not
the native 0.1° IMERG grid used in some earlier experiments.

### 4.3 Dates

The archive uses the BMD observation label as its public daily date. For a BMD
label `D`:

- BMD and prepared IMERG represent the BMD window ending at 03 UTC on `D`.
- The CPC/ERA5/CHIRPS background record used to generate the analysis is
  `D − 1` (`background_day_offset = -1`).
- The saved CHIRPS field is therefore the `D − 1` conditioning/training-family
  field, whereas the prepared IMERG timestamp remains the BMD end date `D`.

This one-day contract is stored in run metadata and must be preserved in every
daily comparison, map title, and station table. A calendar-day product cannot
silently substitute for the BMD-aligned IMERG file.

## 5. Simultaneous data assimilation

The selected model configuration, **SURMA-Flow v1.0**, jointly assimilates BMD
gauges and S04 IMERG during the same rectified-flow sampling trajectory. Its
legacy machine key is `v2_simul_s04_ig010`. The observation operators are
differentiable:

- a physical bilinear interpolation from the 0.05° field to each BMD point;
- an exact physical 8×8 block mean from the 0.05° field to each S04 IMERG
  footprint.

The observation mismatch is evaluated in the checkpoint's transformed
(square-root) precipitation space. Guidance follows the likelihood gradient
through the frozen flow model at each sampling step. The method uses 50 Heun
steps with two bounded Langevin corrector updates in the guided sampler. The
production variant sets prior temperature to 1.0 and uses guidance gamma 0.01
for both the gauge and IMERG streams.

### 5.1 Gauge treatment

Gauges are **not** perfect observations. The production likelihood uses:

| Gauge component | Value | Interpretation |
|---|---:|---|
| Measurement SD | 0.10 | Transformed-space measurement uncertainty |
| Representativeness SD | 0.25 | Point gauge versus 0.05° cell-mean mismatch |
| Combined likelihood SD | about 0.269 | Square root of the two variance contributions |
| Gauge likelihood weight | 1.0 | Frozen 2021–2024 production setting |
| Gauge guidance spread | 6 fine cells | Spreads the point-gauge component over roughly 30–35 km |

The representativeness term is the dominant variance contribution. Each
ensemble member receives a separately perturbed gauge draw from this finite
error model. The gauge likelihood is therefore soft, rather than an exact
interpolation through station reports.

### 5.2 IMERG treatment

The S04 IMERG likelihood uses the supplied retrieval uncertainty plus a
transformed-space error floor and a representativeness component. In the
production run, `observations.imerg.factor=8` and
`observations.imerg.error_corr_cells=0.75` override the generic defaults. The
0.75-footprint correlation length prevents the dense S04 retrieval errors from
being treated as independent white noise. IMERG observation perturbations are
drawn separately for every ensemble member and retain that spatial
correlation.

The frozen `ig010` selection means IMERG and gauges use the same early-time
guidance softness (gamma 0.01). No fitted IMERG bias-correction map is applied
by that frozen production arm. This is a documented limitation: a Gaussian
likelihood is most defensible when its observation error and bias model are
well calibrated, so IMERG agreement is interpreted as adherence to an
assimilated product rather than independent skill.

### 5.3 What gauge-weight experiments mean

The production archive uses `gauge_weight = 1.0`. A weight `w` scales the
gauge likelihood authority by dividing its effective variance and gamma term
by `w`; it does not change the physical gauge error used to generate each
member's perturbed observation draw. Values above one force closer
assimilated-gauge fit, but may overfit point observations, create local
bullseyes, and degrade withheld-gauge CRPS. A zero-error “perfect gauge” is
not part of the real-data product because a point gauge is not an exact 5 km
area-average measurement and a zero variance likelihood is singular.

## 6. Ensemble interpretation

BRISHTI-05 ensemble members represent conditional downscaling ambiguity plus
the implemented observation perturbations. They do not yet include ERA5 EDA
background members or a full structural model-error ensemble. The ensemble
mean is useful for maps, but it can damp station-scale day-to-day variability;
that is a property to diagnose, not a reason to add arbitrary zero-mean noise.

Evaluation reports CRPS, ensemble spread divided by RMSE, empirical 90%
coverage, rank and intensity-stratified diagnostics where available. Add
observation uncertainty when interpreting rank and spread diagnostics. A
spread/RMSE value below one indicates under-dispersion; a value above one is
over-dispersion. The correct remedy must be selected on withheld stations,
not on stations that entered the likelihood.

The production sampler uses perturbed observations because assimilating the
same observation into every member is a known source of under-dispersion. Any
post-hoc spread calibration is a separately labelled product and must never
replace reporting of the raw BRISHTI-05 ensemble.

## 7. Verification and evidence hierarchy

### 7.1 Primary evidence: withheld BMD gauges

The primary real-data score is five-fold spatially rotated, withheld-gauge
verification. Every eligible station is held out once. Report CRPS first,
with MAE, bias, RMSE, correlation, coverage and spread diagnostics alongside.
For the frozen 2021–2024 confirmation archive, primary aggregate scores
exclude 2022-05-01 through 2022-05-10 because those days were used in
configuration selection. Monthly and seasonal aggregates also exclude the
corresponding selected period where appropriate.

All-station production scores must be labelled **assimilated fit**. They are
valuable for checking the likelihood and calibration, but cannot demonstrate
generalization.

### 7.2 Gridded comparisons are agreement, not truth

The archive compares BRISHTI-05 with three product families, each at its
native support and at a common coarse support:

| Product | Evidence role | Why it is not independent truth |
|---|---|---|
| BMD | Independent only when withheld | Production stations were assimilated |
| IMERG | Observation-adherence and unassimilated native-detail context | The same IMERG family was assimilated at S04 |
| CPC | Background/conditioning consistency | CPC conditioned the prior and is the residual baseline |
| CHIRPS | Fine-scale structural/training-family agreement | CHIRPS is the learned target |

Maps and spatial summaries use the Bangladesh ADM0 polygon intersected with
the permanent model-valid mask; all pixels outside that mask are missing and
rendered white. A native 0.1° IMERG comparison is informative about detail
not explicitly constrained by S04 assimilation, but it remains from the same
observation family. Original CPC is retained at 0.5° and is never falsely
presented as a 0.05° observation.

### 7.3 Fine-scale claims

An output grid at 0.05° does not by itself demonstrate correct 0.05°
information. BRISHTI-05 fine-scale analysis should be assessed through:

- exact decomposition into S04 footprint means and below-footprint residuals;
- member and ensemble-mean spectra, variograms, residual variance, and
  increment-locality diagnostics;
- agreement with CHIRPS residual structure, clearly labelled as
  training-family agreement; and, most importantly,
- withheld BMD sub-footprint anomaly tests against coarse-product baselines.

Positive fine-scale placement skill at withheld gauges is required before
claiming demonstrated real-data subgrid resolution. Texture alone, or
agreement with CHIRPS alone, is not enough.

## 8. Archived BRISHTI-05 data

The SURMA-Flow v1.0 production archive remains the authoritative BRISHTI-05
data lineage:

```text
data/processed/v2_confirmatory_2021_2024/
├── cv/                       five held-out station folds by period
├── gridded/                  all-station 30-member production ensembles
├── imerg_native/             prepared BMD-window IMERG
├── imerg_s04/                exact S04 IMERG used by production DA
├── production_metadata/      all-station run metadata
└── evaluation/               BRISHTI-05 descriptive and verification products
```

The gridded Zarr stores retain the historical method key
`v2_simul_s04_ig010`, the full physical ensemble in mm/day, ensemble mean,
ensemble SD, CPC conditioning field, CHIRPS field, S04 IMERG, BMD values, and
station-assimilation flags. They are machine-readable provenance for the
BRISHTI-05 release. Do not rename arrays in-place: map the historical method
key to **SURMA-Flow v1.0** in model descriptions and to **BRISHTI-05** in
user-facing analysis products.

The Bangladesh production-contract evaluator is documented in
[`EVALUATION_cpcv2_june2023_bangladesh.md`](EVALUATION_cpcv2_june2023_bangladesh.md).
It produces Bangladesh-masked maps, same-station daily-variability diagnostics,
native IMERG 0.1° and original CPC 0.5° comparisons, and conservative evidence
labels. The broader archive evaluator and its resolved-structure tests are
documented in [`EVALUATION_v2_gridded_archive.md`](EVALUATION_v2_gridded_archive.md).

## 9. Reproducibility

The frozen long-period production launcher is:

```bash
bash slurm/submit_v2_confirmatory_2021_2024.sh
```

It binds the run to `runs/prior_h100_cpc_v2/best.pt`, uses 30 members, derives
exact period-specific S04 IMERG files from the prepared BMD-window archive,
runs five withheld-station folds plus all-station production analyses, and
writes a dependent summary only after every task completes. The checkpoint's
training configuration determines its compatible Zarr, statistics,
transform, residual specification, and conditioning-channel order; inference
does not substitute those with an arbitrary current configuration.

Method-selection and diagnostic experiments are not silently promoted into the
archive. Any new BRISHTI-05 release must state its checkpoint hash, observation
settings, BMD station file, IMERG preparation path, date contract, ensemble
size, and whether every scored BMD station was withheld.

## References

- Manshausen, P., et al. (2025). Generative data assimilation of sparse weather
  station observations at kilometer scales. *JAMES*, 17, e2024MS004505.
  <https://doi.org/10.1029/2024MS004505>
- Rozet, S., & Louppe, G. (2024). Score-based data assimilation.
  <https://arxiv.org/abs/2306.10574>
- Lipman, Y., et al. (2023). Flow matching for generative modeling.
  <https://arxiv.org/abs/2210.02747>
- Albergo, M. S., Boffi, N. M., & Vanden-Eijnden, E. (2025). Stochastic
  interpolants: a unifying framework for flows and diffusions.
  <https://arxiv.org/abs/2303.08797>
- Funk, C., et al. (2015). The climate hazards infrared precipitation with
  stations (CHIRPS). *Scientific Data*, 2, 150066.
- Huffman, G. J., et al. (2023). GPM IMERG Final Precipitation L3 1 day V07.
