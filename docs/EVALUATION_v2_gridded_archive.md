# Evaluating the CPC-v2 gridded archive without treating CHIRPS as truth

## Scientific premise

There is no perfect gridded rainfall truth for this real-data experiment:

- BMD gauges directly measure point rainfall but have sparse spatial coverage.
- IMERG S04 is assimilated at 0.4° and therefore is not independent evidence
  for the simultaneous analyses.
- CPC conditions the prior and has coarse effective resolution.
- CHIRPS is the model's training target. It offers a 0.05° structural
  reference, but its spatial pattern can differ from IMERG and gauges and is
  not assumed to be truth.

The evaluator therefore does not produce one misleading “model versus CHIRPS”
score. It keeps three evidence classes separate.

All archived dates appear in the descriptive daily/monthly time series. The
aggregate daily method matrices exclude 2022-05-01 through 2022-05-10 because
those dates selected `ig010` and `huber3`; aggregate monthly matrices exclude
May 2022. This matches the confirmatory selection guard used by the pooled
withheld-gauge summary.

### 1. Independent gauge evidence

The five spatial folds provide withheld-gauge CRPS, bias, correlation,
dry/wet MAE and coverage. No held-out station enters the likelihood in the
fold where it is scored.

The stronger downscaling diagnostic is the **withheld-gauge sub-footprint
anomaly test**. For each held-out station and day:

```text
predicted anomaly = fine model at gauge − model's own 0.4° block mean
observed anomaly  = gauge amount − reference product's local 0.4° mean
```

The observed anomaly is calculated three ways, using CHIRPS, IMERG and CPC as
the coarse baseline. The evaluator reports correlation, RMSE, sign agreement,
variance ratio and MSE skill against a zero-subgrid null. A positive
`mse_skill_vs_no_subgrid` means the located fine-scale model anomaly explains
more gauge variation than assigning the whole footprint one value. Agreement
across all three baselines is substantially more credible than a result tied
to CHIRPS alone.

This gauge evidence appears only after all five CV folds for that season have
finished. The gridded-only diagnostics can be generated immediately after the
seasonal Zarr finishes.

### 2. Multi-product agreement

For CHIRPS, IMERG and CPC separately, the evaluator reports:

- mean daily spatial correlation on the native fine representation;
- daily correlation and centred RMSE after every field is reduced to the
  common 0.4° IMERG footprint;
- daily domain mean, spatial standard deviation, wet-area fraction and p95;
- daily and monthly mean posterior ensemble spread for the model methods;
- monthly mean amount and spatial pattern;
- within-month temporal variability and its spatial pattern.

These are labelled **agreement**, not skill. IMERG agreement tests observation
adherence, CPC agreement tests conditioning consistency, and CHIRPS agreement
tests similarity to the fine-resolution training product.

### 3. Reference-free resolved structure

The field is split exactly into its nested 0.4° block mean and residual:

```text
fine field = repeated 0.4° block mean + below-footprint residual
```

The residual has zero mean inside every valid 8×8 fine-cell block. The
evaluator measures:

- ensemble-mean and individual-member residual variance;
- the fraction of member texture coherent in the ensemble mean;
- the fraction of total spatial variance below 0.4°;
- residual gradient energy;
- member and ensemble-mean power spectra;
- member variograms;
- residual variance below 0.1°, 0.2°, 0.4° and 0.8°;
- subgrid increment relative to the background.

These establish whether the analysis truly resolves fine scales rather than
merely repeating IMERG footprints. They do not by themselves establish correct
placement: sharp random texture can pass an amplitude test. Placement evidence
comes from CHIRPS residual agreement and, more importantly, the withheld-gauge
anomaly test.

## Run after any season completes

The submission helper discovers every completed seasonal `.zarr`, submits one
evaluation per season, and—when at least two are available—also submits a
pooled evaluation:

```bash
git pull --ff-only origin main
bash slurm/submit_v2_gridded_evaluation.sh
```

To evaluate only one completed season:

```bash
bash slurm/submit_v2_gridded_evaluation.sh \
  data/processed/v2_confirmatory_2021_2024/gridded/2021_may_sep.zarr
```

The helper is safe to rerun as more seasons complete. Existing
`evaluation.json` products are reused. If a Zarr was evaluated before all five
CV folds for that period finished, refresh it once the folds exist:

```bash
V2_EVAL_FORCE=1 bash slurm/submit_v2_gridded_evaluation.sh \
  data/processed/v2_confirmatory_2021_2024/gridded/2021_may_sep.zarr
```

Disable the pooled job if desired:

```bash
V2_EVAL_POOL=0 bash slurm/submit_v2_gridded_evaluation.sh
```

## Outputs

Per-season output is written to:

```text
data/processed/v2_confirmatory_2021_2024/evaluation/<period>/
```

Pooled output is written under `evaluation/pooled_N_seasons/`. Each directory
contains:

- `evaluation.json` — strict machine-readable report and interpretation;
- `evaluation_matrix.csv` — combined methods × diagnostics table;
- `daily_domain.csv` — daily mean, spatial variability, wet area and p95;
- `monthly_domain.csv` — monthly amount and within-month variability;
- `product_matrix.csv` and `monthly_matrix.csv`;
- `subgrid_matrix.csv`;
- `withheld_gauge_matrix.csv`;
- `withheld_gauge_subgrid_anomalies.csv`;
- five PNG/PDF figures, each with source CSV and provenance manifest.

The figures are:

1. Existing method matrix extended with monthly and subgrid evidence, including
   gauge-anomaly MSE skill against a zero-subgrid null.
2. Daily/monthly mean and variability time series.
3. Subgrid evidence matrix.
4. Spectra, variograms and residual-variance scale ladder.
5. Full fields and below-0.4° residual maps for the most
   CHIRPS-subgrid-active archived day.

## Reading the result conservatively

A publishable real-data downscaling result should satisfy all three:

1. The method improves or preserves independent withheld-gauge value skill.
2. Its sub-footprint anomaly has positive MSE skill against the zero-subgrid
   null, preferably under all three product baselines.
3. Its member spectra and residual variance are plausible rather than strongly
   deficient or excessive, while a non-trivial fraction is coherent in the
   ensemble mean.

CHIRPS residual correlation without gauge-anomaly skill is product imitation,
not demonstrated real-world downscaling. Member-scale variance without either
form of placement agreement is generated texture, not demonstrated information.
