# V7 R81 June 2023 production pilot

## Frozen choice

This pilot uses only `da_sim_r81` and the `latest_latest` checkpoint pair from
the May 3 tournament:

- meso: `data/processed/v7_checkpoint_tournament_may03/20260821_1323/source_checkpoints/latest_meso.pt`
- allocation: `data/processed/v7_checkpoint_tournament_may03/20260821_1323/source_checkpoints/latest_allocation.pt`

It does not read the live `runs/v7/*/best.pt` files. The archive uses 30 members,
50 sampling steps and seed 201805. Model and CHIRPS dates are May 31 through June
29; BMD, BMD-windowed IMERG and the saved archive time coordinate are June 1
through June 30.

## Products

The six-task array writes two distinct products under
`data/processed/v7_june2023_r81_latest_latest`:

1. Five withheld-gauge folds. The eligible station pool and each fold's
   withheld IDs are imported from the frozen CPCv2 2023 confirmation dumps.
   Every station is independently verified exactly once.
2. One all-station `bdhires.physical_ensemble.v1` Zarr. It retains all 30
   physical 0.05-degree members, plus the matched CPC condition, CHIRPS field,
   native 0.1-degree IMERG and BMD observations. This product is for maps and
   field statistics, not independent gauge verification.

The dependent summary subsets CPCv2's May-September 2023 folds to June, audits
dates, station coordinates, BMD values, held-out IDs and ensemble size, then
pools the five folds. Its paired uncertainty interval resamples whole days.

## Submit

From the repository root on Prism:

```bash
git pull --ff-only origin main
conda activate mytorch
bash slurm/submit_v7_june2023_r81.sh
```

The launcher prints both Slurm job IDs. Monitor with:

```bash
squeue -u "$USER"
tail -f logs/bdhires-v7-jun23-<ARRAY_JOB_ID>_*.out
```

Final products:

```text
data/processed/v7_june2023_r81_latest_latest/
├── cv/june2023/fold{0..4}/station_ensembles.npz
├── production/june2023/station_ensembles.npz
├── gridded/june2023.zarr
└── comparison/june2023_v7_r81_vs_cpcv2.{md,json}
```

## Gate before 2021-2023 production

Do not start the three-year V7 production run merely because all jobs finish.
First inspect the pooled R81-versus-CPCv2 CRPS, its day-bootstrap interval,
spread/RMSE, bias, failure logs and Zarr completeness. Once the June pilot passes
those checks, the same six-task pattern can be sharded by year for 2021-2023
without changing checkpoints, R81, member count, station-fold logic or archive
schema.
