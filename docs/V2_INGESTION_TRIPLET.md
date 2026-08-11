# CPC-v2 BMD/IMERG ingestion triplet

## Question

On the same May 1--10, 2022 development window used for CPC-v2 gauge-method
selection, compare three observation strategies with five disjoint BMD folds,
30 ensemble members, and identical prior/observation seeds:

| Arm | Configuration |
|---|---|
| `guided_s6_t100` | selected gauge method: spread 6, temperature 1.0 |
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

## Run

From the cluster repository root:

```bash
git pull --ff-only origin main
bash slurm/submit_v2_ingestion_triplet.sh
```

The launcher submits a five-fold GPU array and a dependent CPU summary. Outputs
are isolated at:

```text
data/processed/v2_ingestion_triplet/ing2022_s04/
  fold0.npz ... fold4.npz
  fold0.json ... fold4.json
  ingestion_selection.md
  ingestion_selection.json
  ingestion_selection.png
```

The primary paired test is
`CRPS(gauges-only) - CRPS(simultaneous)`. Positive values favour adding IMERG.
The 95% interval uses a circular three-day block bootstrap and keeps every
station from a weather day together. If the interval includes zero, the honest
conclusion is that the satellite contribution is unresolved on ten days.
