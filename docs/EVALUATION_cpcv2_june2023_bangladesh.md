# CPCv2 June 2023 Bangladesh evaluation

This evaluation reads the already-completed all-station CPCv2 production Zarr
stores. It does **not** rerun data assimilation or regenerate CPCv2.

The target is `v2_simul_s04_ig010` for all 30 days of June 2023. Complete Junes
from the 2021, 2022, 2023 and 2024 seasonal stores are discovered at runtime;
the climatology excludes 2023, so the June 2023 anomaly is not compared against
a baseline containing itself. This short available-year average is descriptive
and is not called a 30-year climate normal.

All spatial scores and maps are restricted to the published Bangladesh ADM0
polygon intersected with the archive's model-valid mask. Outside pixels are
stored as missing and rendered white. The submission helper downloads and
snapshots the geoBoundaries `gbOpen` BGD ADM0 GeoJSON plus its metadata once.

## Evidence roles

- BMD: every production station entered the likelihood, so the station matrix
  measures assimilated fit, not independent verification.
- IMERG S04: entered the simultaneous likelihood, so agreement measures
  observation adherence.
- CHIRPS: the learned fine-grid target and a structural reference, not truth.
- CPC: loaded from the checkpoint-bound packed archive on the target day, not
  silently replaced by the lagged CPC conditioning field.

The output contains Bangladesh-only monthly-mean maps, within-June daily-SD
maps, daily time series, native and common-0.4-degree agreement matrices,
all-station gauge diagnostics, leave-2023-out June climatology maps, anomaly
maps, station-residual maps, CSV tables, an NPZ field bundle, PDFs and a
machine-readable JSON report. Boundary attribution: geoBoundaries `gbOpen`
Bangladesh ADM0, CC BY 4.0; the exact metadata snapshot is retained with the
downloaded geometry.

## Run

From the repository root on the Prism login node:

```bash
cd /home/afahad/project/BDDA/BDhighresDA
git pull --ff-only origin main
conda activate mytorch
bash slurm/submit_cpcv2_june2023_bangladesh_eval.sh
```

The default output is:

```text
data/processed/v2_confirmatory_2021_2024/evaluation/june2023_bangladesh_ig010
```

To rerun an existing report:

```bash
V2_JUNE_BGD_FORCE=1 bash slurm/submit_cpcv2_june2023_bangladesh_eval.sh
```
