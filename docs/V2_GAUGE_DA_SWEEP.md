# CPC-v2 gauges-only DA method tournament

## Question

The CPC-v2 background is already materially better than v1 at withheld BMD
gauges (CRPS 5.915 versus 7.547 over 2022-05-01--10).  The v1 DA settings are
therefore not automatically appropriate: a point update can erase a good prior
more easily than it repairs a poor one.  This tournament asks which update
mechanism adds value to **the fixed v2 background**.

May 1--10, 2022 is development data because it has already been inspected.  It
can screen mechanisms, but the selected method must be frozen and confirmed on
a longer, independent period.

## Core arms

All arms share checkpoint, dates, spatial fold, 30-member prior noise, gauge
perturbation seeds and observation-error configuration.

| Arm | Update |
|---|---|
| `background` | unguided v2 prior, temperature 1.0 |
| `guided_s0_t125` | current likelihood guidance; temperature 1.25, no spreading |
| `guided_s6_t125` | measured spread-6 candidate; temperature 1.25 |
| `guided_s0_t100` | no spreading, analysis temperature matched to background |
| `guided_s6_t100` | spread-6, analysis temperature matched to background |
| `ensrf_loc150` | serial EnSRF applied to the exact background ensemble, 150 km compact support |

The four guided arms are a 2x2 factorial, so temperature and spreading can be
identified rather than changed together.  EnSRF is a different DA method: it
uses localized ensemble covariance instead of blurring the likelihood gradient.

## Run

From the cluster repository root:

```bash
git pull --ff-only origin main
bash slurm/submit_v2_gauge_method_sweep.sh
```

Defaults are May 1--10, 2022, five disjoint BMD folds and 30 members using
`runs/prior_h100_cpc_v2/best.pt`.

Outputs are isolated from the previous v1/v2 comparison:

```text
data/processed/v2_gauge_da_sweep/ing2022_core/
  fold0.npz ... fold4.npz
  fold0.json ... fold4.json
  method_selection.md
  method_selection.json
  method_selection.png
```

The summary uses a paired circular bootstrap over 3-day blocks.  Every station
from a weather day stays in the same resample, so ten days are not misreported
as 380 independent events.

## Follow-up groups

Only run one after the core tournament identifies the relevant mechanism.

Spread sensitivity at matched temperature:

```bash
V2_SWEEP_GROUP=v2_gauges_spread \
V2_SWEEP_LABEL=ing2022_spread \
bash slurm/submit_v2_gauge_method_sweep.sh
```

EnSRF localization sensitivity (75, 150 and 300 km support):

```bash
V2_SWEEP_GROUP=v2_gauges_ensrf \
V2_SWEEP_LABEL=ing2022_ensrf \
bash slurm/submit_v2_gauge_method_sweep.sh
```

Config overrides are semicolon-separated and recorded in every report.  This is
how a later run should inject observation errors calibrated specifically under
`stats_cpc_v2.json`:

```bash
BMD_SET='observations.gauges.sigma_obs=X;observations.gauges.representativeness=Y' \
V2_SWEEP_LABEL=ing2022_core_calibratedR \
bash slurm/submit_v2_gauge_method_sweep.sh
```

Do not select `X` and `Y` from this CRPS table; estimate them independently with
`scripts/35_gauge_truth_error_budget.py`.

## Decision

Primary skill is fair CRPS at withheld BMD gauges.  The table also reports
dry/wet MAE, bias, correlation, 90% coverage, CPC pattern correlation, domain
wet fraction and increment locality.  A method is promoted only when its
central CRPS improves on the background and its absolute bias is not more than
0.5 mm/day worse.  At most two methods advance.

For a publication claim, rerun the promoted methods over an independent full
season with all five folds and keep the same day-block uncertainty calculation.
If no method beats the background there, background/no-DA is the selected v2
method.
