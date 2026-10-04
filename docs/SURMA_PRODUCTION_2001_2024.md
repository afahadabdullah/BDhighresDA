# SURMA-Flow / BRISHTI-05 production, 2001–2024

The production workflow generates **8,766 daily fields, 30 members per day,
and 96 quarterly shards** from 2001-01-01 through 2024-12-31, including dry
seasons and leap days. It uses the final CPCv2 U-Net checkpoint and
`dense_s6_bwdb_r4`: 0.25° BMD/BWDB super-observations, 0.4° IMERG, Huber
delta 3, gauge spreading 6 cells and BWDB error variance multiplier 4.
Sampling uses 50 Heun steps and two analysis correctors. All eligible gauges
enter production; these fields do not replace the separately withheld
verification experiments.

## What data do we have?

The local source-file audit on 2026-10-02 found BMD and BWDB observations for
**every quarter in 2001–2024**, with at least 34 eligible BMD and 188 eligible
BWDB gauges in each quarter after the existing QC and 50% coverage criterion.
The 2001 Q1 preparation has 35 BMD / 229 BWDB gauges, and 2024 Q4 has
42 BMD / 265 BWDB gauges. These are checks of the local national source files;
the PRISM preflight repeats the checks on its actual copies and records hashes.

The desktop checkout has no learned checkpoint, CPCv2 statistics, packed
predictor Zarr, annual gridded sources or prepared IMERG months. Their status
on PRISM must be audited there. The copied paper results establish only the
old seasonal test windows, not full calendar-year input availability.

| Required input | Default location | Recovery |
|---|---|---|
| Evaluated CPCv2 weights | `runs/prior_h100_cpc_v2/best.pt` | Existing PRISM training artifact; pinned SHA-256 is checked. |
| Trained normalization/transform | `data/processed/stats_cpc_v2.json` | Existing evaluated artifact; preserve its content and checkpoint-bound path. |
| BMD daily history | `data/stations/Rainfall_daily_by_station_BMD.csv` | Existing private source covers early years as well as 2020–2024. |
| BMD coordinates | `data/stations/data_2020_2025/Stations.csv` | Existing catalogue and explicit aliases. |
| BWDB daily history | `data/stations/BWDB_Rainfall_2000_2025_corrected.xlsx` | Existing private workbook, same reader/QC as the model paper. |
| CPC + ERA5 predictors and static channels | `data/processed/bd_wide_cpc.zarr` | Reuse if complete; otherwise acquire annual sources and build a separate production predictor store. |
| Original static grid for rebuilding | `data/static/static_wide.nc` | Recover the grid used to pack the trained model's inputs. |
| Native, 03 UTC daily IMERG | `data/processed/imerg_bd_aligned_YYYYMMDD_YYYYMMDD.nc` | Missing months are downloaded from V07B half-hours and accumulated with script 08. |

## Audit PRISM first

To prioritize overnight half-hourly IMERG acquisition, use the dedicated
[resumable download job](IMERG_PRODUCTION_DOWNLOAD.md) before launching the
full production chain. It validates and prepares all 288 native months at the
paths this launcher reuses, with at most three concurrent NASA downloads.

From the PRISM repository root:

```bash
git pull --ff-only origin main
mkdir -p logs
bash slurm/submit_surma_production_2001_2024.sh --audit-only
```

After the audit job finishes:

```bash
cat data/processed/brishti05_production_2001_2024/input_inventory.json
```

The report checks the actual checkpoint/statistics identity, raw annual-file
inventory and all 288 IMERG months. The CPU job deeply validates any present
predictor store, including every requested daily predictor and the previous-day
boundary. Missing inputs remain explicit. An audit does not download or sample.
For a fast metadata inventory on a login node, use its native Python:

```bash
python3 scripts/100_surma_production.py audit
```

This fast mode reports file presence without claiming that NetCDF contents
have validated. Do not execute the ARM environment's Python on an x86 login
node; scientific work runs inside the CPU jobs.

## Credentials and environments

The existing GH200 project environment is used for preparation and sampling:
`/home/afahad/nb/project/BDDA/envs/bdda-gh200`. It already supports the final
model and scientific dependencies. Jobs do not install or update packages.

IMERG acquisition uses the project's existing Earthdata-authenticated
downloader. Configure your `~/.netrc` Earthdata entry and GES DISC access
before submitting missing-month downloads. Credentials are never included in
the audit output. The downloader requests `precipitation` and `randomError`
only for the Bangladesh analysis domain.

If ERA5 downloads are needed and the Earthmover Python 3.12 environment has
not yet been created, run the one-time setup:

```bash
bash slurm/setup_earthmover_env.sh
```

Wait for setup to complete before submitting acquisition. Override its Python
path with `ERA5_PYTHON` if installed elsewhere. Earthmover ERA5 and NOAA CPC
use public endpoints; their product descriptions are documented by the
[AWS ERA5 registry](https://registry.opendata.aws/earthmover-era5/),
[NOAA CPC catalogue](https://psl.noaa.gov/data/gridded/data.cpc.globalprecip.html)
and [NASA IMERG documentation](https://gpm.nasa.gov/resources/documents/imerg-v07-technical-documentation).

## Acquire and prepare inputs

```bash
bash slurm/submit_surma_production_2001_2024.sh --prepare-only
```

This submits a dependency chain:

1. **Fixed-input preflight:** verify model/statistics hashes, source files,
   credentials when downloads are needed, and gauge coverage in every quarter.
2. **Annual inputs:** reuse a complete packed predictor archive. If rebuilding
   is required, use the existing CPC, CHIRPS and ERA5 download scripts for
   2000–2024. Existing validated annual ERA5/CHIRPS files are reused.
3. **IMERG months:** validate/reuse prepared months; download missing months
   from half-hourly V07B, including the preceding-day intervals, and accumulate
   exact 24-hour windows ending at 03 UTC. Never substitute calendar-day IMERG.
4. **Predictor packing:** reuse the complete `bd_wide_cpc.zarr`, or pack a
   separate `ROOT/predictors.zarr`. Verify coordinates, seven static fields,
   required CPC/ERA5 channels, complete dates and finite predictors on every day.
5. **Quarterly preparation:** build original station tables, 0.25°
   super-observations and 0.4° IMERG with the established correlated-error
   aggregation. Preserve source hashes and measured representativeness.

The first production day needs CPC/ERA5 record **2000-12-31** because the
evaluated background-day offset is -1. The existing annual packer therefore
needs the 2000 annual source files if a separate production store is rebuilt.
CHIRPS is required by this packer's target/context schema; it is not an
additional assimilated stream. Weights and normalization are reused from
training, not refitted on the production period.

New production super-observation error budgets explicitly use the pinned
CPCv2 statistics. Their manifests record the bytes actually used. Historical
paper preparation-statistics identity remains a separate unresolved evidence
item; this run does not retroactively establish it.

The gauge windows retain the evaluated mixed-source convention: BMD ends at
00 UTC and BWDB at 03 UTC on date D, IMERG ends at 03 UTC, and background
predictors are D-1. The three-hour gauge-support difference remains explicit.

## Submit production

The full wrapper performs acquisition and preparation automatically before
sampling, so it can also be submitted directly:

```bash
bash slurm/submit_surma_production_2001_2024.sh
```

For an allocation that requires an account:

```bash
bash slurm/submit_surma_production_2001_2024.sh --account=g0609
```

The default concurrency is two annual/monthly input tasks per input array,
two quarterly preparation tasks, and two GH200 production tasks. The GPU
array has 96 tasks, each producing one quarter and retaining the background
and final analysis ensembles. An `afterok` dependency prevents sampling after
failed input stages. A final CPU job checks every shard and writes the full
archive receipt only after all 8,766 days exist and validate.

Inspect the commands without submitting anything:

```bash
bash slurm/submit_surma_production_2001_2024.sh --dry-run
```

Inspect jobs and logs with the IDs printed by the wrapper:

```bash
squeue -u "$USER"
tail -f logs/surma-prod-01-24-JOBID_TASKID.out
```

GPU jobs request 48 hours per quarter. Actual timing has not been measured
for a full calendar-year production run; inspect a pilot before increasing
concurrency. A one-year pilot exercises both wet and dry seasons:

```bash
SURMA_PROD_START_YEAR=2001 SURMA_PROD_END_YEAR=2001 \
SURMA_PROD_ROOT=data/processed/brishti05_production_pilot_2001 \
bash slurm/submit_surma_production_2001_2024.sh
```

This pilot uses its own output root. Metadata checks can reuse a wider existing
predictor archive. Keep model, statistics, raw inputs and predictor stores
unchanged while the dependency chain is running.

## Outputs and completion

```text
data/processed/brishti05_production_2001_2024/
  input_inventory.json             # optional audit report
  preflight.json                   # fixed-input hashes and gauge coverage
  gauge_coverage.json              # all-quarter station counts
  predictor_validation.json        # actual predictor path and validation
  predictors.zarr/                 # only if a separate store was needed
  stations/YEAR_qN/                # original reports, summary, superobs, provenance
  imerg_native/YEAR_qN.nc
  imerg_s04/YEAR_qN.nc
  prepared/YEAR_qN.json             # preparation/input hashes
  production_identity/YEAR_qN.json  # input identities at sampling start
  production_metadata/YEAR_qN.npz
  production_metadata/YEAR_qN.json
  gridded/YEAR_qN.zarr/
  validated/YEAR_qN.json
  production_manifest.json         # written only after all shards validate
```

Each field store has `precipitation(method,time,member,lat,lon)` and derived
`ensemble_mean`/`ensemble_std`. Select method `dense_s6_bwdb_r4` for the final
analysis; `background` is its reference. The grid is the existing 128×128
0.05° Bangladesh analysis box with its geographic halo. It retains the full
ensemble and declares which stations were assimilated. Country masks should
be applied to Bangladesh-only reporting and map exports.

Both field arrays alone total approximately 32.1 GiB before compression;
prepared inputs, predictor stores, raw half-hours and companion arrays need
additional space. Compression and actual download size vary.

Rerunning acquisition reuses validated months and complete predictors.
Quarterly preparation is regenerated and hashed. Completed sampled shards
are reused only after date/member/method/provenance/finite-field checks. A
partial sampled shard is preserved and reported as an error; inspect or move
it aside before retrying. Sampling currently restarts an interrupted quarter
rather than resuming within the quarter. If a dependency job fails, dependent
jobs remain blocked; resolve the logged input failure before resubmission.

## Overrides

Set these before invoking the wrapper:

| Variable | Purpose |
|---|---|
| `SURMA_PROD_ROOT` | Separate output root. |
| `SURMA_PROD_START_YEAR`, `SURMA_PROD_END_YEAR` | Subrange within 2001–2024. |
| `SURMA_PROD_DATA_ZARR` | Explicit complete/relocated predictor store, or separate store to build. |
| `SURMA_PROD_CKPT`, `SURMA_PROD_STATS` | Existing pinned model/statistics paths; statistics path must match checkpoint metadata. |
| `SURMA_PROD_BMD_WIDE`, `SURMA_PROD_BMD_CATALOG`, `SURMA_PROD_BWDB` | Actual private national source paths. |
| `SURMA_PROD_IMERG_RAW`, `SURMA_PROD_IMERG_DAILY` | Existing/downloaded half-hours and prepared monthly locations. |
| `SURMA_PROD_INPUT_CONCURRENCY`, `SURMA_PROD_CONCURRENCY` | CPU-input and GPU array concurrency. |
| `PYTHON_BIN`, `ENV_PREFIX`, `ERA5_PYTHON` | Existing project and Earthmover interpreters. |

The wrapper prints all job IDs. This document describes the runnable workflow;
the full PRISM data download and GPU production have not been executed from
the desktop checkout.
