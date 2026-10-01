# Bangladesh-only Paper 1 reevaluation

The verification region is the snapshotted geoBoundaries Bangladesh ADM0
polygon, not the model's rectangular computational domain. Grid centres and
original verification stations outside the polygon are excluded. Polygon
holes and islands are retained; station inclusion has numerical tolerance
only, with no geographic buffer. Map rasters are vector-clipped as well as
masked, with a black Bangladesh outline and pure white outside.

Boundary: `configs/geography/geoBoundaries-BGD-ADM0.geojson`, CC0 1.0,
boundary ID `BGD-ADM0-71232402`, represented year 2015, retrieved 2026-10-01.
SHA-256: `b556c52ad47050a751fe58b7e2636fd53fad54178ac73dba0b0d1b449e012e61`.
The adjacent metadata snapshot records the provider and fixed geometry URL.

## Recomputed from local exports

```bash
python scripts/94_recompute_paper1_bangladesh_exports.py
python scripts/91_build_cpcv2_paper1_figures.py
```

These write/read `paper1_cpcv2/bangladesh/superob-final/`, preserving the
supplied wider-domain `paper1_cpcv2/superob-final/` exports. Script 94 uses
the original BMD/BWDB catalogues and the preparation reader's documented
extra station coordinates and CL312 latitude correction. A canonical
`--station-catalog` CSV (`station_id,lat,lon`) can be supplied instead.
Catalogue provenance is hashed. Actual archived NPZ-coordinate identity
still requires the full archive rerun.

`BWDB_CL130` (24.8756°N, 92.3661°E) lies outside the polygon. It contributed
153 station-days in 2023 and 61 in 2024, all outside the selection month.
Removing it leaves **29,401 scored station-days, 132 withheld site IDs and
489 dates**. The temporal sample becomes 962 station-months and 120 complete
station-seasons.

| Metric | Background | SURMA-Flow |
|---|---:|---:|
| Fair CRPS (mm/day) | 8.768 | 6.258 |
| RMSE (mm/day) | 20.795 | 17.070 |
| MAE (mm/day) | 14.939 | 9.026 |
| Bias (mm/day) | 7.732 | -0.325 |
| Correlation | 0.525 | 0.657 |
| Spread/RMSE | 0.861 | 0.593 |
| 90% coverage | 0.843 | 0.727 |

The reconstruction is exact for the available sufficient statistics. Linear
scores use sample-weighted means; RMSE and spread pool squared quantities.
Coverage pools the reported station-level interval counts. Monthly daily
means and population standard deviations recover first/second moments;
summed daily squared errors recover cross moments for the two ensemble means.
The script first reproduces the original pooled metrics to numerical
precision, verifies sample denominators, and only then filters country sites.
It does not fabricate daily/member values or new confidence intervals.

The spatial case is the archived **29 May 2024** field, now masked and
clipped to Bangladesh. Its selection was on the wider domain; it remains
an illustration, not a newly selected Bangladesh extreme. Colour limits
now use inside-country values. The original physical-footprint residual
definition is retained for that illustrative export.

## Full saved-archive rerun on PRISM

After syncing the updated scripts and boundary snapshot to the compute
checkout, submit from the repository root:

```bash
mkdir -p logs
sbatch slurm/cpcv2_paper1_evaluation.sbatch
```

The launcher now defaults to Bangladesh's boundary and writes separate
`output/paper1_cpcv2_bangladesh/superob-final/` results. It validates the
original ensemble contract, excludes outside-country gauge arrays before
all daily/period/intensity/bootstrap/temporal scores, and forwards the same
boundary to script 55. That script masks all gridded field arrays before
domain statistics, filters withheld and assimilated-fit gauge sites, and
draws geographic country maps. This is post-processing only, not a rerun
of rainfall generation or assimilation.

The original archives must be present at
`data/processed/v2_bmd_bwdb_superob_2021_2024/`. Override `PAPER1_ROOT`,
`PAPER1_BOUNDARY`, `PAPER1_OUT`, `PAPER1_CPC_SOURCE` or `PYTHON_BIN` if needed.
All May 2022 remains excluded. Missing raw archives stop evaluation; a
summary-only result is not accepted as completed member-level verification.

Then generate the additional calibration/interpolation/paired-interval slots:

```bash
sbatch slurm/cpcv2_paper1_completion.sbatch
```

P9 now generates country-only three- and seven-day paired CRPS intervals
from the original inside-country member arrays. The completion workflow
also excludes outside stations from selection comparisons and IDW inputs,
and requires matching country masks for observing-system robustness tables.

## Evidence still pending the full archive

- Equal-day paired block-bootstrap intervals.
- Period scores affected by CL130 and fine rainfall-intensity bins.
- Withheld-gauge anomaly correlations/MSE skill.
- Full-grid daily, monthly and complete-season field statistics and
  country-specific illustrative-event reselection.
- Pooled CHIRPS/IMERG/CPC correlations on the new matched sample.
- Direct comparison of NPZ station coordinates with local catalogues.

The paper replaces the old intervals, fine intensity plot and subgrid plot
with explicit pending slots. Product correlations remain blank/pending;
averaging station correlations would not recover a pooled correlation.
Original earlier source/PDF/package backups have `_pre_Bangladesh` names.

Copy the full evaluated profile back to
`paper1_cpcv2/bangladesh/superob-final/`, regenerate script-91 figures,
copy completion artifacts to `manuscript/additional/`, review the new
scientific interpretation, recompile and package with script 93. This fills
figures/tables from real results; authors must still update pending prose.

## Withheld-gauge comparisons lead the paper

All main performance figures verify at original withheld Bangladesh gauges.
The daily product figure separates the pooled, BMD and BWDB samples for
all five methods; common station-day counts are checked before plotting.
Field maps, reference-dependent subgrid structure and model-only temporal
scatter are supporting appendix material.

Full reevaluation additionally writes `gridded/withheld_product_strata.csv`.
This compares model means, CHIRPS, IMERG and original same-day CPC on a
single daily finite intersection by network, year and observed intensity,
and on identical eligible station-period means at monthly/seasonal scales.
Original truth count, matched count and attrition are exported. The main
intensity/temporal product-comparison figures remain pending until this
CSV is generated on PRISM and script 91 is rerun. These calculations
reuse saved fields and do not launch training or generation.
