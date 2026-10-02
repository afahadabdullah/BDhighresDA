# Daily IMERG timing pilot, 1–5 May 2022

Run `scripts/101_compare_imerg_daily_windows.py` in the existing GH200 project
environment. It prepares the observations, runs both cases sequentially, and
writes comparison tables and maps. Each case uses **one member on each of five
days**, the final CPCv2 checkpoint, and the production `dense_s6_bwdb_r4` method
(0.25° gauge super-observations, 0.4° IMERG, Huber 3, spread 6, BWDB variance
multiplier 4, 50 Heun steps and two analysis correctors). A matched unguided
background draw is also retained for each day.

| Input | A: existing reporting convention | B: UTC calendar days |
|---|---|---|
| Gauge records | Original reports labelled May 1–5 | The identical records relabelled April 30–May 4 |
| Background CPC/ERA5 | April 30–May 4, offset −1 | April 30–May 4, offset 0 |
| IMERG | 24-hour windows ending 03 UTC on May 1–5 | Native daily V07B for April 30–May 4, 00–24 UTC |
| Random draws | Base seed plus observation index | Base seed + 1 plus the previous day's index |
| CHIRPS comparison reference | April 30–May 4 | April 30–May 4 |

These are the **May 1–5 gauge reports**, not two different rainfall events.
Using May 1–5 calendar days in B would require May 2–6 gauge reports and would
move the experiment forward a day. The script shifts both BMD and BWDB to
preserve the existing production network and rainfall values. It computes the
station observations once and only relabels their dates for B. In archive-reuse
mode, station eligibility, centroids, rainfall values and the measured error
budget come from the older May–September production preparation. When rebuilding
from raw inputs, eligibility instead requires three valid reports in five days
and the error budget is measured over those five days. Those modes must be
distinguished when comparing this pilot to the historical production fields.

The current source readers declare BMD support as `[D−1 00 UTC, D 00 UTC]`
and BWDB as `[D−1 03 UTC, D 03 UTC]`. Current IMERG matches BWDB exactly;
calendar-day IMERG matches the relabelled BMD support exactly. Relabelling
does not eliminate BWDB's three-hour support difference. Mixed-source
super-observations retain this difference.

The NASA daily product is the mean of the valid half-hourly precipitation
rates multiplied by 24, already in mm/day. The pilot requires 48 valid
retrievals per native footprint in **both** cases; it does not multiply native
daily precipitation by 24 again. Daily `randomError` is retained as supplied
by NASA. Its aggregation differs from the half-hourly quadrature used in A,
so this tests the proposed full ingestion change, including uncertainty,
rather than time support alone. The report measures precipitation and
random-error differences separately.

Source: [NASA daily V07 product definition](https://data.nasa.gov/dataset/gpm-imerg-final-precipitation-l3-1-day-0-1-degree-x-0-1-degree-v07-gpm-3imergdf-at-ges-dis-13ed8).

## Run on PRISM

From the repository root, inspect without downloads or sampling:

```bash
python3 scripts/101_compare_imerg_daily_windows.py --dry-run
```

Submit the Slurm job from the repository root on the PRISM login node:

```bash
mkdir -p logs
sbatch slurm/imerg_daily_window_pilot.sbatch --download-imerg
```

If your allocation requires an account:

```bash
sbatch --account=g0609 slurm/imerg_daily_window_pilot.sbatch --download-imerg
```

The job requests one GH200 GPU on `grace`, eight CPUs, 64 GB of memory and
12 hours. It checks ARM architecture and CUDA availability, then prepares
the small input window, runs both GPU cases sequentially, and produces the
comparison. It uses the existing project environment without installing
packages. Arguments after the `.sbatch` filename go to the Python script:

```bash
sbatch slurm/imerg_daily_window_pilot.sbatch --download-imerg \
  --aligned-imerg data/processed/v2_confirmatory_2021_2024/imerg_s04/2022_may_sep.nc \
  --root data/processed/imerg_daily_window_pilot_may2022_run2
```

Omit `--download-imerg` when the five raw daily granules and the reporting-window
inputs already exist. Override `PYTHON_BIN` or `ENV_PREFIX` before submission
if the existing project environment is elsewhere. Monitor the returned job ID:

```bash
squeue -j JOBID
tail -f logs/imerg-daily-pilot-JOBID.out
```

For an interactive allocation instead:

```bash
srun --partition=grace --nodes=1 --ntasks=1 --gres=gpu:1 \
  --cpus-per-task=8 --mem=64G --time=04:00:00 \
  /home/afahad/nb/project/BDDA/envs/bdda-gh200/bin/python -u \
  scripts/101_compare_imerg_daily_windows.py --download-imerg
```

The resource time limits are requests, not measured runtimes. Earthdata login
and GES DISC access must already be configured through `~/.netrc`.

The script first reuses
`data/processed/v2_confirmatory_2021_2024/imerg_s04/2022_may_sep.nc`, or
`data/processed/imerg_bd_aligned_20220501_20220531.nc` if present. Set
`--aligned-imerg /path/to/prepared.nc` to choose another existing native or
S04 reporting-window archive. Otherwise it builds A from half-hours in
`data/imerg_halfhourly/2022`; `--download-imerg` acquires missing half-hours
for these five reporting days only. The same option fetches just **five
daily granules** for B, April 30–May 4, into `data/raw/imerg`.

Without `--download-imerg`, both sets of input files must already exist.
Native daily granules are cropped, validated, and coarsened to the evaluated
observation grid; no half-hourly re-accumulation is needed for B. The existing
production input loader remains strict by default; this pilot uses the
explicit `--allow-calendar-day-imerg` option with truthful calendar metadata.

Useful options:

- By default, when available, the pilot reuses
  `data/processed/v2_bmd_bwdb_superob_2021_2024/stations/2022_may_sep/superob_prod_0.25.csv`
  and its `.json` error report. It applies the older GPU job's 50% coverage
  threshold over the full May–September season, keeps its spatial centroids,
  subsets the five days, and preserves the original measured representativeness
  without re-aggregation or refitting. Stations that qualify for the season
  remain in the pilot even when they have missing reports in its five days.
- `--production-station-root /path/to/stations/2022_may_sep`: explicitly require
  the existing all-station production CSV and JSON. Missing companions fail
  clearly; evaluation super-observations with held-out stations are rejected.
  The original preparation-statistics provenance is copied into the experiment
  manifest, including any historical unresolved identity; reusing the table
  does not retroactively validate that identity.
- Explicit `--stations` or `--bmd-data-dir` selects rebuilding from raw data
  and overrides archive auto-detection. For strict production-input reuse:

  ```bash
  sbatch slurm/imerg_daily_window_pilot.sbatch --download-imerg \
    --production-station-root data/processed/v2_bmd_bwdb_superob_2021_2024/stations/2022_may_sep \
    --root data/processed/imerg_daily_window_pilot_production_stations
  ```

- When rebuilding from raw inputs, the pilot prefers the wide BMD history CSV when present. If it is absent,
  it automatically uses the existing per-station CSVs in
  `data/stations/data_2020_2025/`, through the same reader/QC as the established
  May 2022 experiment. `Stations.csv` alone is only a coordinate catalogue;
  daily station CSVs must also exist. Source files and hashes are recorded in
  `station_preparation.json`. This fallback is limited to 2020–2025 and cannot
  replace the historical source for 2001–2019 production.
- `--bmd-data-dir /path/to/data_2020_2025`: explicitly select the per-station
  source, even when the wide table exists. `--bmd-catalog` sets its catalogue.
- `--stations /path/to/combined_daily.csv`: original canonical BMD/BWDB
  station table. Dates are filtered to May 1–5 and gauges are aggregated once.
  Already aggregated `SOB_` tables are rejected.
- `--data-zarr /path/to/predictors.zarr`: relocated predictor store.
- `--ckpt`, `--stats`: evaluated artifact paths; hashes are checked and the
  statistics path must match checkpoint metadata.
- `--daily-raw`, `--halfhourly`: existing raw IMERG locations.
- `--prepare-only`: prepare all input files without sampling.
- `--summarize-only`: recompute reports and maps from the two saved cases.
- `--root`: separate experiment output root. Existing sampled or partial
  outputs are preserved; use a new root when rerunning sampling.

## Read the results

Default output: `data/processed/imerg_daily_window_pilot_may2022/`.

- `experiment.json`: exact dates, seeds, archived/rebuilt station mode, gauge error budget and
  SHA-256 identities of checkpoint, statistics and prepared observations.
- `reporting.npz/json`, `calendar.npz/json`: sampler results and metadata.
- `comparison.md/json`: pooled field differences, five-day total differences,
  IMERG precipitation/error differences, assimilated super-observation fits,
  and comparisons against one common CHIRPS calendar-day reference.
- `daily_comparison.csv`: daily field MAE, RMSE, mean difference, correlation,
  maximum difference and domain means.
- `comparison_maps.png`: daily and five-day A/B rainfall maps and B−A maps.
- `comparison_fields.npz`: paired analysis fields, differences, common
  background and common CHIRPS for follow-up plotting.

Maps and field metrics use the model's land-valid mask over the full BD box
and halo; they are not restricted to the Bangladesh country polygon.
The script checks the exact expected dates, identical gauge values and
coordinates, identical CPC fields, matched sampler settings, one-member
outputs, and matching background draws (absolute tolerance 1e−5 mm/day).
A failure in any of these checks invalidates the pairing and stops the report.

The existing sweep saves CHIRPS at its observation label, so A's original
`chirps` array is May 1–5 while B's is April 30–May 4. CHIRPS is not assimilated;
this pilot deliberately uses **B's calendar-day CHIRPS array for both cases**
in the comparison. It does not modify the original predictor archive.

With one member, differences are sensitivity of a single paired draw. Gauge
fits use assimilated observations and do not establish independent skill;
CHIRPS is a common reference, not independent truth. Small differences in
these five days motivate a larger pilot spanning intense storms, dry days
and seasons before replacing half-hourly IMERG in 2001–2024 production.

The older BMD-only archive (`v2_confirmatory_2021_2024`) used the per-station
BMD directory and catalogue. The later final combined archive
(`v2_bmd_bwdb_superob_2021_2024`) added the corrected BWDB workbook, aggregated
the full seasonal station table, and used `superob_prod_0.25.csv/json` in its
GPU production job. Its preparation launcher defaults to `stats.json` unless
overridden; the historical report is needed to establish the actual statistics
used. This pilot's raw rebuild explicitly uses `stats_cpc_v2.json`, while
archive mode preserves the stored historical budget and records its provenance.

## Compare one daily file with its 48 half hours

Before interpreting analysis differences as timing sensitivity, check raw
rainfall and errors on the **same UTC day**. This CPU comparison needs no
checkpoint, gauges, Slurm GPU allocation, or new daily downloads:

```bash
mkdir -p logs
sbatch slurm/imerg_daily_halfhourly_compare.sbatch --day 2022-05-01 \
  --daily-raw data/raw/imerg --halfhourly data/imerg_halfhourly/2022
```

The job uses `grace-cpuonly` and the established GH200 Python environment,
without requesting a GPU. On a compatible compute node, it can also be run
directly with that environment's Python and the same script arguments.

It compares native daily `precipitation` (mm/day) with `0.5 * sum(rate)`
from exactly 48 V07B Final half-hourly files for May 1 00:00–24:00 UTC.
The mean half-hourly rate is in mm/hour; multiply it by 24 to compare with
the daily file. The script keeps rainfall validity independent of error
validity and compares only cells with all 48 valid precipitation estimates.
Zero rainfall remains valid. The original reporting-window preparation ends
at 03 UTC and is not suitable for this direct calendar-day check.

Missing or duplicate intervals fail before producing results. If files are
missing, the existing downloader can fetch just that calendar day, expressed
as a window ending at midnight May 2; existing files are skipped:

```bash
/home/afahad/nb/project/BDDA/envs/bdda-gh200/bin/python \
  scripts/02_download_imerg_halfhourly.py \
  --bmd-start 2022-05-02 --bmd-end 2022-05-02 --end-hour-utc 0 \
  --out data/imerg_halfhourly/2022 --jobs 2
```

Run that direct downloader command on a compatible compute node. It is not
part of the comparison job and does not download an entire month.

Results are saved in `data/processed/imerg_daily_halfhourly_20220501/`:
`comparison.md/json` give rainfall/error differences, native units, fill and
scale metadata, valid-data counts, and half-hourly error distributions.
`fields.nc` saves the native rainfall, half-hourly total, difference, and
the separate error estimates. Errors are never clipped or silently replaced.
The error comparison includes both the existing temporal-independence
quadrature and a reconstruction of the distinct daily formula documented in
the [NASA dataset catalogue](https://data.nasa.gov/dataset/gpm-imerg-final-precipitation-l3-1-day-0-1-degree-x-0-1-degree-v07-gpm-3imergdf-at-ges-dis-13ed8).
Agreement of rainfall does not imply agreement of these error definitions.

To audit the original GPU pilot's raw and coarsened errors without resampling:

```bash
sbatch slurm/imerg_daily_halfhourly_compare.sbatch --audit-pilot \
  --root data/processed/imerg_daily_window_pilot_production_stations
```

This writes `imerg_error_audit.json`, recording regional error percentiles,
large values and their locations, precipitation/error counts (when provided),
units, missing-value metadata, scaling, and original manifest identities.
It does not modify pilot inputs. Values above 1000 mm/day are flagged for
inspection as a diagnostic tail, not declared invalid by an arbitrary cutoff.
An anomalously large error mismatch means the original two analyses cannot
be interpreted as a clean test of rainfall-window timing alone.
