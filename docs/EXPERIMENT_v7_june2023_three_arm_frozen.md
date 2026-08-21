# V7 three-arm June 2023 frozen-checkpoint comparison

This follow-up keeps the original frozen checkpoint pair and tests the three
closest simultaneous V7 configurations from the May 1-10 ingestion sweep:

| arm | IMERG support | May CRPS | difference from R81 |
|---|---:|---:|---:|
| `da_sim_r81` | native 0.1° | 3.802 | 0.000 |
| `da_sim_r27` | native 0.1° | 3.813 | +0.011 |
| `da_sim_s04_corr_g001_h3` | 0.4° S04 | 3.829 | +0.027 |

R27 is the near-tied native-weight alternative. The S04 arm is the closest
coarse-ingestion alternative and therefore tests a meaningfully different
satellite footprint rather than another tiny change to R.

The run explicitly ignores the later checkpoint tournament and uses:

```text
data/processed/v7_osse/20260820_1356/checkpoints/meso_frozen.pt
data/processed/v7_osse/20260820_1356/checkpoints/allocation_frozen.pt
```

As in the R81 pilot, it runs five exact CPCv2-matched station folds and one
all-station production analysis with 30 members and 50 steps per V7 stage. All
three physical ensembles are stored together in one method-indexed Zarr:

```text
data/processed/v7_june2023_three_arm_frozen/gridded/june2023.zarr
```

CPCv2 is not regenerated. The dependent comparison reuses the existing 2023
CPCv2 folds, pools all independently withheld station-days, and produces a
paired whole-day bootstrap interval for each V7 arm.

Submit from the repository root on Prism:

```bash
git pull --ff-only origin main
conda activate mytorch
bash slurm/submit_v7_june2023_three_arm_frozen.sh
```

The dependent summary is submitted on a Grace GPU node. Final scores are:

```text
data/processed/v7_june2023_three_arm_frozen/comparison/
june2023_v7_three_arm_frozen_vs_cpcv2.{md,json}
```
