# Complete the CPCv2 model-paper evidence

For the updated eight-figure manuscript, use
[PAPER1_UPDATED_EVIDENCE.md](PAPER1_UPDATED_EVIDENCE.md) and script 96.
That workflow evaluates every available test period, adds paired product
comparisons and native IMERG, and records the current paper's evidence gaps.
See [PAPER1_REMAINING_EVIDENCE.md](PAPER1_REMAINING_EVIDENCE.md) for the nine
items still missing after the first full-archive run. Script 92 now also
exports daily model means, paired IDW intervals and IDW intensity gains.
The instructions below describe the existing script-92 completion slots.

The manuscript is `manuscript/BDhighresDA_arxiv.tex`. The reported results
come from `paper1_cpcv2/bangladesh/superob-final/`; additional evidence is explicitly
reserved in nine numbered slots plus four country-rerun/comparative figure slots. None contains fabricated measurements.
The historical 2000–2025 product remains a separate paper.

See [PAPER1_BANGLADESH.md](PAPER1_BANGLADESH.md) for the new country mask,
partial local reevaluation, and full saved-archive rerun. All verification
now excludes outside-country sites.

## Generate available and pending artifacts

Run from the project root with NumPy and Matplotlib installed in the existing
project environment. This is CPU post-processing, with no data download,
training, checkpoint loading, or rainfall generation.

```bash
python scripts/91_build_cpcv2_paper1_figures.py
python scripts/92_complete_cpcv2_paper1.py --audit-only
```

Script 91 reproduces four available result figures from the country-filtered
exports, including the six-panel pooled/BMD/BWDB product comparison.
Additional intensity, subgrid and all-product temporal panels are generated
only when their actual exports are present.
Script 92 needs the original NPZ/JSON archive for member-level diagnostics,
not just the copied summary directory. On PRISM, the default archive is
`data/processed/v2_bmd_bwdb_superob_2021_2024`. Run:

```bash
python scripts/92_complete_cpcv2_paper1.py \
  --root data/processed/v2_bmd_bwdb_superob_2021_2024 \
  --history runs/prior_h100_cpc_v2/validation/history.jsonl
```

This fills P1–P4, P9 and P6 if their original inputs are present. The raw gauge
inputs must be `stations/PERIOD/combined_daily.csv`, the actual preparation
outputs from `slurm/v2_bmd_bwdb_superob_2021_2024_prepare.sbatch`. Columns are
`station_id,lat,lon,date,precip_mm`. Super-observation CSVs cannot substitute
for original reports. Script 90's contract validation is reused before
calibration or IDW computation. IDW excludes all withheld identifiers,
checks withheld values/coordinates against the saved ensembles, and exports
the common finite sample and any attrition. Its power is fixed at two.

CPU batch launcher:

```bash
mkdir -p logs
sbatch slurm/cpcv2_paper1_completion.sbatch
```

Override `PAPER1_ROOT`, `PAPER1_HISTORY`, `PAPER1_SELECTION`, `PAPER1_COMPUTE`,
`PAPER1_OUTPUT`, or `PYTHON_BIN` as needed. `PAPER1_OUTPUT` is the artifact
directory, default `manuscript/additional`. Copy that directory back beside
the local manuscript and recompile; figures and tables replace their boxes
automatically. The batch job does not compile TeX or run inference.

| Slot | Required evidence | Generated artifact |
|---|---|---|
| P1 | Validated period NPZs and original combined CSVs | `fig_network.pdf`, `network_geometry.csv` |
| P2 | Validated withheld member arrays | `fig_calibration.pdf`, rank/reliability/coverage CSVs |
| P3 | Same arrays, original withheld rainfall | `tab_thresholds.tex`, `threshold_scores.csv` |
| P4 | Same arrays and original retained daily reports | `tab_interpolation.tex`, score and prediction CSVs |
| P5 | At least two matched May 2022 selection NPZ/JSON pairs | `tab_selection.tex`, `selection_scores.csv` |
| P6 | Original CPCv2 validation monitor JSONL | `fig_training.pdf`, `training_curve.csv` |
| P7 | Actual checkpoint-linked timing records | `tab_compute.tex`, `compute_summary.csv` |
| P8 | At least two completed audited script-90 evaluations using the same country mask | `tab_robustness.tex`, `robustness_scores.csv` |
| P9 | Original inside-country member arrays | `tab_country_intervals.tex`, `paired_country_crps.csv` |

Every run writes `completion_manifest.json`, with pending/generated status,
selection exclusion, input hashes, and generated-file hashes. Audit-only
checks file existence without scoring. Missing inputs are pending, while
available malformed or mismatched inputs are errors. `--require-complete`
returns exit code 2 if any slot is unavailable; a normal partial run succeeds
and reports what remains. An audit's `ready` means inputs exist, not that
their scientific validation has passed. Non-audit runs remove only this
script's known output files before regeneration, preventing stale fills.
Available artifacts are staged until all supplied inputs validate, so a bad
optional record leaves no partially validated fills.

## P5: selection comparison

Create a JSON file identifying *actual* saved selection experiments. Paths
are resolved from the working directory. Each prefix has both `.npz` and
`.json`, written by script 28. Labels should describe the differing settings;
read the archived specifications rather than inferring them from filenames.

```json
{
  "profiles": [
    {"label": "Raw retained network", "prefix": "PATH_TO_RAW_SELECTION_PREFIX", "method": "ACTUAL_ARM_KEY"},
    {"label": "Aggregated network", "prefix": "PATH_TO_SUPEROB_SELECTION_PREFIX", "method": "ACTUAL_ARM_KEY"},
    {"label": "Final superob + BWDB R x4", "prefix": "PATH_TO_FINAL_SELECTION_PREFIX", "method": "dense_s6_bwdb_r4"}
  ]
}
```

These are schema examples, not valid experiment paths. Then:

```bash
python scripts/92_complete_cpcv2_paper1.py --selection /path/to/selection_profiles.json
```

The exporter uses the shared finite station-day intersection and rejects
dates outside May 2022, differing original holdouts, truth, statistics,
member count or seeds, and non-CPCv2 checkpoints. A five-day sweep is labeled
as five days; it is not enlarged to a month. Differences involving several
settings do not constitute a one-factor ablation. Recovering a leaderboard
does not retroactively make those dates independent verification.

## P6: validation history

The project monitor already writes the expected JSONL schema: `epoch`,
`step`, `members`, `mean_crps_mm`, and `cases` containing `date`/`quantile`.
The plot requires fixed 2019–2020 cases and a fixed member count. Resolve
resumed-run duplicates from source logs rather than silently dropping them.
Confirm the evaluated checkpoint's actual selected epoch separately before
describing a plotted minimum as that checkpoint's identity.

## P7: measured computational cost

Create a timing JSON from original job accounting or a timed benchmark.
Do not substitute configured epochs, nominal job limits or synthetic times.
Required top-level keys are `checkpoint_sha256` and `measurements`.
The checksum must be
`a04a3d9ae9109f905e06c32bfd55252daf1229d17c98b404e265064b89f210ea`.

`measurements` contains exactly one row per `stage`: `training`, `background`,
and `analysis`. Each row requires `hardware` (short name), `gpus` (positive
integer), `wall_seconds` (measured positive number), and `record_source`
(specific job log/accounting record). Sampling rows additionally require:

```json
{"days": 10, "members": 30, "steps": 50, "correctors": 2, "grid": [128, 128]}
```

Here `days` must be the number actually timed; 10 is only a schema example.
Background requires `correctors: 0`; analysis requires `correctors: 2`.
Include guidance/observation processing in the analysis timing and document
initialization, warm-up, I/O, precision and measurement conditions in the
source record. Training time must cover the full training run, including
resumptions as applicable. Sampling GPU-hours and seconds per ensemble-day
are computed from actual duration. Then run with `--compute PATH.json`.
This script formats and validates measurements; it cannot recover missing
hardware timing from skill tables.

## P8: additional holdouts

Recover complete original BMD-reference archives and evaluate them separately:

```bash
python scripts/90_evaluate_cpcv2_paper1.py --profile bmd-reference \
  --root data/processed/v2_confirmatory_2021_2024 \
  --out-dir output/paper1_cpcv2
python scripts/92_complete_cpcv2_paper1.py --robustness \
  paper1_cpcv2/bangladesh/superob-final output/paper1_cpcv2/bmd-reference
```

The BMD-only five-fold experiment differs in network, guidance and excluded
selection dates. It is shown separately, not pooled with the final dense
profile or called exhaustive dense-network cross-validation. Existing
missing-input audits are rejected as results. New blocked/thinned-network
experiments require a new frozen split contract and actual ensemble runs;
post-processing cannot manufacture them. No such costly runs are launched
by these scripts.

## After new artifacts arrive

Review the derived scores and manifest, replace pending wording in the
surrounding interpretation with the observed findings, and rebuild the PDF
and source bundle. Automatic inclusion fills the figures/tables, not the
scientific conclusions. Keep any adverse baseline/calibration findings.
Do not retune against the independent test sample. Confirm author metadata,
funding, gauge reporting windows/upstream overlap, preparation-statistics
identity, and permitted archive/release identifiers before submission.

Package the revised editable source and available generated artifacts with:

```bash
python scripts/93_package_cpcv2_paper1.py
```

This writes `manuscript/SURMA_Flow_CPCv2_arxiv_source.zip` and verifies the
hashes of generated additions. It excludes private gauge records, weights
and large intermediate data. Compile the `.tex` from the extracted folder
with its relative figure paths intact.

## Main-result emphasis: original withheld gauges

The primary comparison is SURMA-Flow, unguided background, CHIRPS, the
archived 0.4-degree IMERG stream and original same-day CPC against identical
Bangladesh withheld station-days. The model-only temporal scatter and
illustrative field/subgrid maps are in the supporting appendix.

Script 94 also writes `gridded/long_term_withheld_station_scores.csv` after
country filtering. Script 91 verifies identical per-station sample counts
across all five methods before displaying RMSE/MAE by BMD and BWDB.
Pooled product correlations remain pending where unavailable; averaging
station correlations is not a replacement.

The full script-90/55 evaluation produces `gridded/withheld_product_strata.csv`
from original daily arrays. All products and model means use a single finite
intersection, with input/matched counts and attrition. This includes network,
year, observed-intensity and monthly/complete-season deterministic scores.
Same-day CPC is required; lagged conditioning CPC is never substituted.
Script 91 reads those rows to generate WG1 `fig08_product_intensity.pdf`
and WG2 `fig09_product_temporal.pdf`. These two main-text slots remain
pending locally. Monthly eligibility counts matched dates (80% calendar
coverage); seasons require all May--September dates. Every method shares
identical eligible station-period groups. No temporal ensemble scores are
generated. Script 92 alone cannot fill WG1/WG2 from model-member files:
run the full gridded evaluation as documented in PAPER1_BANGLADESH.md.
