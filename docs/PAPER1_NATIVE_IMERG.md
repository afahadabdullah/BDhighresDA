# Native 0.1-degree IMERG comparisons

The current manuscript uses `scripts/95_build_paper1_restructured_figures.py`.
Its IMERG scores and maps use the saved, assimilated 0.4-degree block means.
Native IMERG must be sampled and scored again: changing plot labels cannot
recover the 0.1-degree rainfall structure.

The new `scripts/110_build_native_imerg_paper_figures.py` renders native-product
comparisons from script 96's exports. It keeps the original figure folder and
script 95 intact, including any local manuscript-specific changes. Training
and assimilation do not need to run again. The method diagram and operator
description still correctly say **0.4 degrees**.

## Inputs

- Original withheld NPZ/JSONs and station manifests under
  `data/processed/v2_bmd_bwdb_superob_2021_2024`.
- Original same-day CPC packed store, normally `data/processed/bd_wide_cpc.zarr`.
- Actual prepared **native 0.1-degree V07B** NetCDFs for May–September
  2021–2023 and May–June 2024. Use the existing half-hourly-derived, 24-hour
  accumulations ending at 03 UTC on each reporting date. Script 08 prepares
  them; script 44's coarsened files are unsuitable.
- Existing Bangladesh gridded paper exports (compact seasonal arrays and
  the case-field CSV), available in the copied `paper1_updated_evidence_v2`
  bundle. No full production Zarr is needed by the new renderer.

The production preparation workflow also retains `imerg_native/*.nc`, and
monthly files named `imerg_bd_aligned_START_END.nc`. These can supply the
native inputs if they cover the requested dates and pass metadata checks.
Pass disjoint files; overlapping dates cause validation to stop.

## Run on the machine with the original archive

Create a text file containing one actual native NetCDF path per line. In the
existing scientific Python environment, from the repository root:

```bash
mapfile -t PAPER_NATIVE_FILES < /path/to/native_imerg_files.txt
python scripts/96_complete_updated_paper_evidence.py \
  --run --native-imerg "${PAPER_NATIVE_FILES[@]}" \
  --root data/processed/v2_bmd_bwdb_superob_2021_2024 \
  --cpc-source-zarr data/processed/bd_wide_cpc.zarr \
  --out-dir output/paper1_native_imerg
```

Run this scoring step on PRISM if those raw arrays are only there. Copy its
output directory back, including `evidence_manifest.json`, then render using
the already copied country-only gridded exports:

```bash
python scripts/110_build_native_imerg_paper_figures.py \
  --evidence output/paper1_native_imerg \
  --gridded paper1_updated_evidence_v2/evaluation/superob-final/gridded \
  --out-dir manuscript/figures_native_imerg
```

The existing `slurm/cpcv2_updated_paper_evidence.sbatch` also forwards
`PAPER1_NATIVE_IMERG_LIST` and `PAPER1_EVIDENCE_OUT`. It additionally runs the
broader ensemble evidence workflow; the direct command above performs just
the saved-array product evaluation.

On PRISM, `bdda-gh200/bin/python` is ARM/aarch64 and must run on a Grace
node. Calling it directly from an x86-64 node fails with `Exec format error`.
Use the current `(base)` Python for standard-library-only file-list creation;
submit scientific evaluation with the existing `grace-cpuonly` batch launcher.
After that job finishes, render on a compatible node:

```bash
srun --partition=grace-cpuonly \
  --nodes=1 --ntasks=1 --cpus-per-task=4 \
  --mem=16G --time=00:30:00 \
  env -u PYTHONHOME -u PYTHONPATH \
  PYTHONNOUSERSITE=1 MPLBACKEND=Agg \
  /home/afahad/nb/project/BDDA/envs/bdda-gh200/bin/python \
  scripts/110_build_native_imerg_paper_figures.py \
  --evidence output/paper1_native_imerg \
  --gridded output/paper1_native_imerg/evaluation/superob-final/gridded \
  --native-imerg-list output/native_imerg_files.txt \
  --out-dir manuscript/figures_native_imerg
```

## Outputs and interpretation

Script 96 exports pooled/network/year/month/intensity/temporal scores in
`native_imerg_comparison.csv`, station scores and coordinates in
`native_imerg_station_scores.csv`, temporal samples, paired intervals, a
native comparison table, and `native_imerg_grids.npz`. All six estimates,
including coarse IMERG for sensitivity analysis, share one finite station-day
intersection. Native missingness does not alter the existing five-way scores.
Both sample size and attrition are recorded. Native paired intervals include
every comparator; native IMERG's intervals are pointwise (the pre-existing
family-adjusted columns cover coarse IMERG, CPC and CHIRPS).

Script 110 plots SURMA-Flow, native IMERG, CPC, CHIRPS and background:

| Output | Replaces |
|---|---|
| `fig02_products` | Daily pooled and network product errors |
| `fig03_station_maps` | Station RMSE maps and product win counts |
| `fig04_native_product_years` | The product comparison panel of Figure 4 |
| `fig05_seasonal_maps` | Seasonal product maps |
| `fig06_native_time_scales` | The product mean-score panel of Figure 6 |
| `fig08_case` | The selected daily product maps |
| `fig08_product_intensity` | Product errors by observed intensity |

Yearly and temporal figures are standalone replacements for product panels;
they do not replace the accompanying ensemble-only panels. The renderer
checks current-run output hashes, native grid spacing, boundary identity,
seasonal years and case date. It draws IMERG on its own 0.1-degree grid;
it does not interpolate the 0.4-degree field to make a native-looking map.
Seasonal maps use complete non-selection May–September seasons (2021 and
2023). The actual date in the existing case CSV is authoritative. If it
differs from script 96's cached case (default 29 May 2024), script 110 reloads
only that day's field from the original native IMERG files and verifies its
source hash, timing metadata and grid. This does not rescore gauges or rerun
bootstrap calculations. Use `--native-imerg-list output/native_imerg_files.txt`
to remap source paths when needed; file contents must match the scored input
hashes. The copied case CSV is dated 1 July 2021, despite script 95's old
hardcoded 29 May 2024 caption. New case titles and provenance use the CSV date.

`native_comparison_numbers.json` and `native_figure_manifest.json` record
new scores, counts, map means, hashes and the distinction between comparison
at 0.1 degrees and assimilation at 0.4 degrees. Review these before updating
manuscript tables, abstract, percentages, station win counts, intensity
claims and conclusions. The old 0.4-degree headline gains do not establish
gains over the native product.
