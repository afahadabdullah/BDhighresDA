# BRISHTI-05 Bangladesh production-contract evaluation

**BRISHTI-05** means **Bangladesh Rainfall Integration of Satellite,
Hydrometeorological, and Terrestrial Information at 0.05°**. The name also
transliterates the Bangla word for rain. `v2_simul_s04_ig010` remains the frozen
method-lineage key in machine-readable files, but it is not the product name.

This evaluation reads the already-completed all-station production Zarr stores
from the legacy CPCv2 lineage. It does **not** rerun data assimilation.

The target is `v2_simul_s04_ig010` for the 30-field June 2023 production
contract. Produced-analysis and gridded-product dates are 2023-05-31 through
2023-06-29; their paired BMD observation labels are 2023-06-01 through
2023-06-30, exactly +1 day. Prepared IMERG uses the BMD 03 UTC window and is
therefore stamped with those BMD end-date labels, but is associated with the
preceding produced/model day. CHIRPS in the saved archive is already the D-1
conditioning field. Complete paired periods from the 2021, 2022, 2023
and 2024 seasonal stores are discovered at runtime. The climatology excludes
2023, so the anomaly is not compared against a baseline containing itself.
This short available-year average is descriptive, not a 30-year climate normal.

All spatial scores and maps are restricted to the published Bangladesh ADM0
polygon intersected with the archive's model-valid mask. Outside pixels are
stored as missing and rendered white. The submission helper downloads and
snapshots the geoBoundaries `gbOpen` BGD ADM0 GeoJSON plus its metadata once.

## Evidence roles

- BMD: every production station entered the likelihood, so the +1-day station
  matrix measures assimilated observation fit, not independent verification.
- IMERG: read from the prepared native 0.1-degree BMD-window files. This arm
  assimilated the same retrieval after coarsening to S04 (0.4 degrees), so the
  unassimilated 0.1-degree detail is informative but the product family is not
  independent.
- CHIRPS: the learned 0.05-degree analysis target and a structural comparison.
- CPC: read directly from original NOAA `precip.YYYY.nc` files and retained on
  its native 0.5-degree cells without fine-grid interpolation.

No gridded source is designated as truth. Native-support matrices area-average
the BRISHTI-05 field to 0.05, 0.1 or 0.5 degrees as appropriate. A second matrix
places every gridded analysis/product on the original CPC 0.5-degree cells.

The output contains Bangladesh-only monthly-mean maps, within-June daily-SD
maps, daily time series, native-support and common-0.5-degree agreement matrices,
all-station gauge diagnostics, leave-2023-out June climatology maps, anomaly
maps, station-residual maps, CSV tables, an NPZ field bundle, PDFs and a
machine-readable JSON report. Boundary attribution and licensing are copied
verbatim from the retained geoBoundaries metadata snapshot.

## Run

From the repository root on the Prism login node:

```bash
cd /home/afahad/project/BDDA/BDhighresDA
git pull --ff-only origin main
conda activate mytorch
bash slurm/submit_cpcv2_june2023_bangladesh_eval.sh
```

The default revised output is:

```text
data/processed/v2_confirmatory_2021_2024/evaluation/brishti05_june2023_native_refs
```

To rerun an existing report:

```bash
V2_JUNE_BGD_FORCE=1 bash slurm/submit_cpcv2_june2023_bangladesh_eval.sh
```
