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

All archived dates appear in the descriptive daily/monthly/seasonal tables. The
aggregate daily method matrices exclude 2022-05-01 through 2022-05-10 because
those dates selected `ig010` and `huber3`; aggregate monthly matrices exclude
May 2022, and confirmatory May--September matrices exclude the complete 2022
season. This matches the confirmatory selection guard used by the pooled
withheld-gauge summary and prevents ten selected days from leaking into a
seasonal score.

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

The all-station production Zarr also permits a separate **assimilated-gauge
fit** diagnostic. Its 30-member fields are sampled at BMD coordinates with the
same bilinear grid geometry used by the likelihood. Those gauges entered the
production analysis and therefore show whether the likelihood was fitted, not
whether the method generalizes. They are never combined with or substituted
for the independent withheld-gauge scores.

### 2. Multi-product agreement

For CHIRPS, IMERG and CPC separately, the evaluator reports:

- mean daily spatial correlation on the native fine representation;
- daily correlation and centred RMSE after every field is reduced to the
  common 0.4° IMERG footprint;
- daily domain mean, spatial standard deviation, wet-area fraction and p95;
- daily and monthly mean posterior ensemble spread for the model methods;
- monthly mean amount and spatial pattern;
- within-month temporal variability and its spatial pattern.
- May--September mean and within-season daily temporal variability.

All sources are first reduced by exact block means for the common-support
daily/monthly/seasonal matrix. The plot therefore cannot reward a source just
for having more fine pixels. The saved native 0.05° maps remain available for
examining resolved spatial detail.

The word **variability** has one definition at each aggregation scale:

- daily: the day-to-day series of spatial standard deviation over the domain;
- monthly: the spatial field of daily temporal standard deviation within each
  calendar month;
- May--September: the spatial field of daily temporal standard deviation
  within each complete season.

These are precipitation variability diagnostics. Posterior ensemble spread is
uncertainty and is shown in separate columns and panels.

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

Run only the pooled analysis (avoid four additional per-season jobs):

```bash
V2_EVAL_PER_SEASON=0 V2_EVAL_FORCE=1 \
  bash slurm/submit_v2_gridded_evaluation.sh
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
- `seasonal_domain.csv` — each archived seasonal mean and temporal variability;
- `temporal_scale_grid_matrix.csv` — common-0.4° daily, monthly and
  May--September mean/variability agreement with all three products;
- `temporal_mean_variability_grids.npz` — native-grid monthly climatology and
  confirmatory May--September mean/temporal-SD fields behind the maps;
- `temporal_mean_variability_grid_summary.csv` — domain summaries of those maps;
- `product_matrix.csv` and `monthly_matrix.csv`;
- `subgrid_matrix.csv`;
- `withheld_gauge_matrix.csv`;
- `withheld_gauge_subgrid_anomalies.csv`;
- `gauge_temporal_scale_matrix.csv` — daily, monthly and seasonal results for
  independent withheld gauges and separately labelled assimilated-gauge fit;
- `long_term_withheld_product_matrix.csv` — pooled daily, mean station-temporal,
  and long-term station-mean correlation/RMSE/bias for every analysis, CHIRPS,
  IMERG and original same-day CPC against withheld BMD gauges. The evaluator
  infers the packed CPC store from `scope.checkpoint_data` (or accepts
  `--cpc-source-zarr`), selects target dates directly, and omits CPC if that
  source cannot be audited. It never substitutes the archived lagged CPC input.
  Every source is scored on the same common finite station-day sample;
- `long_term_withheld_station_scores.csv` — per-station temporal scores and
  observed/predicted long-term means behind that comparison;
- eleven PNG/PDF figures, each with source data and provenance manifest.

The figures are:

1. Existing method matrix extended with monthly and subgrid evidence, including
   gauge-anomaly MSE skill against a zero-subgrid null.
2. Daily/monthly mean and variability time series.
3. Subgrid evidence matrix.
4. Spectra, variograms and residual-variance scale ladder.
5. Full fields and below-0.4° residual maps for the most
   CHIRPS-subgrid-active archived day.
6. Calendar-month mean maps for every method and product.
7. Calendar-month daily temporal-variability maps.
8. Confirmatory May--September mean and within-season temporal-SD maps. For the
   pooled 2021--2024 archive these use complete 2021 and 2023 seasons; 2022 is
   selection-contaminated and 2024 has only May--June.
9. Common-support daily/monthly/May--September gridded agreement matrix.
10. Daily/monthly/May--September gauge matrix with independent withheld evidence
    and assimilated fit in separate rows.
11. Gauge-targeted comparison of all model products, CHIRPS, IMERG and original
    same-day CPC. It shows pooled daily correlation/RMSE/bias and mean temporal
    correlation across individual stations, plus spatial correlation/RMSE/bias
    across the long-term means of all withheld stations. The plotted
    across-station temporal correlation is averaged in Fisher-z space; the arithmetic mean and median
    are retained in the CSV.

## Run locally on a CPU node

Quote or explicitly list the Zarr paths; do not rely on an unset shell variable.
From the repository root:

```bash
mkdir -p logs
nohup env PYTHONPATH="$PWD/src" MPLBACKEND=Agg \
  python -u scripts/55_evaluate_v2_gridded_archive.py \
  --zarr \
    data/processed/v2_confirmatory_2021_2024/gridded/2021_may_sep.zarr \
    data/processed/v2_confirmatory_2021_2024/gridded/2022_may_sep.zarr \
    data/processed/v2_confirmatory_2021_2024/gridded/2023_may_sep.zarr \
    data/processed/v2_confirmatory_2021_2024/gridded/2024_may_jun.zarr \
  --cv-root data/processed/v2_confirmatory_2021_2024 \
  --out-dir data/processed/v2_confirmatory_2021_2024/evaluation/pooled_current \
  --factor 8 --texture-members 5 \
  > logs/v2-evaluation-local.out 2>&1 &
tail -f logs/v2-evaluation-local.out
```

Sampling the full production ensemble at assimilated gauges adds sequential
Zarr I/O but holds only one method/season field in memory at a time.

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
