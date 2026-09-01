# Combined BMD/BWDB CPC-v2 Huber3 archive, 2021–2024

This archive repeats the frozen CPC-v2 seasonal production contract with the
combined BMD and BWDB station network, but runs only the May-2022 winner:
`v2_simul_s04_huber3`.  `background` is retained solely as the matched
reference field and diagnostic control.

## Seasonal products

The archive covers May--September 2021, 2022 and 2023, and May--June 2024.
For every period it creates two products:

1. One deterministic random 20% BMD+BWDB holdout.  Every scored station has
   at least one station still assimilated within 15 km; direct BMD/BWDB
   co-located pairs within 5 km are held out together.
2. One all-station production Zarr.  This assimilates every eligible BMD and
   BWDB station and is for gridded maps and structural diagnostics, not
   independent gauge skill.

The 2022 May test selected Huber3.  Independent daily score summaries and
the gridded evaluator therefore exclude 2022-05-01 through 2022-05-31.

## Submit the archive

```bash
bash slurm/submit_v2_bmd_bwdb_huber3_2021_2024.sh
```

The dependent chain prepares station data, submits four held-out evaluation
tasks plus four all-station GPU tasks, then writes the compact confirmation
summary and fold diagnostics.

## Run the full gridded evaluation and plotting suite

After the production Zarr stores complete, run:

```bash
bash slurm/submit_v2_bmd_bwdb_huber3_2021_2024_evaluation.sh
```

This is the same evaluator used for the earlier CPC-v2 archive.  It produces
the method matrix, daily/monthly diagnostics, subgrid structure, spectra,
variograms, field maps, temporal-scale matrices, and independent-holdout vs
assimilated-fit gauge plots.  The one-holdout layout is stated in the output;
it is not treated as five-fold exhaustive verification.

All outputs live below:

```text
data/processed/v2_bmd_bwdb_huber3_2021_2024/
├── evaluation/<period>.{npz,json}        # constrained independent holdout
├── gridded/<period>.zarr                 # all-station production ensemble
├── summary/
│   ├── huber3_2021_2024_scores.{md,json,png}
│   ├── gridded_catalog.json
│   └── fold_plots/<period>_fold0_diagnostics.png
└── evaluation/<period>/fig{01..11}_*.png # full gridded evaluator
```
