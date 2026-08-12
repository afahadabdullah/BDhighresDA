# CPC-v2 frozen DA confirmation and gridded archive, 2021–2024

## Purpose

This is the long confirmation run for the configurations selected on
2022-05-01 through 2022-05-10. It has two distinct products:

1. **Held-out-gauge verification:** five spatial folds for each period. Every
   eligible BMD station is withheld exactly once and never enters the
   likelihood in the fold where it is scored.
2. **Gridded production archive:** every eligible BMD station is assimilated.
   These fields are for maps, monthly means, variability and product
   comparisons; they must not be used to claim independent gauge skill.

The requested 520 days are:

- 2021-05-01 through 2021-09-30 (153 days)
- 2022-05-01 through 2022-09-30 (153 days)
- 2023-05-01 through 2023-09-30 (153 days)
- 2024-05-01 through 2024-06-30 (61 days)

## Frozen methods

The archive contains the exact same five methods in this order:

1. `background`
2. `guided_s6_g010_t100` — selected gauges-only benchmark
3. `v2_simultaneous_s04_t100` — previously reported simultaneous benchmark
4. `v2_simul_s04_ig010` — primary frozen simultaneous candidate
5. `v2_simul_s04_huber3` — secondary robust candidate

This is a confirmation set, not another tuning grid. All methods use 30
members, the CPC-v2 checkpoint, a one-day background offset, S04 IMERG at
factor 8/stride 1, and the corrected sqrt-space likelihood.

## Independence from configuration selection

All requested dates remain in the Zarr archive. The primary daily verification
excludes 2022-05-01 through 2022-05-10 because those days selected `ig010` and
`huber3`. The primary monthly verification excludes all of May 2022 because a
monthly mean containing tuning days is not independent. The summary also emits
clearly labelled descriptive scores using every requested day.

## Submit

From the repository root on Prism:

```bash
git pull --ff-only origin main
bash slurm/submit_v2_confirmatory_2021_2024.sh
```

The launcher first makes one exact 0.4-degree S04 IMERG file per period from
the existing BMD-aligned seasonal files. It then submits one array:

- tasks 0–19: four periods × five held-out-station folds;
- tasks 20–23: four all-station gridded analyses;
- maximum four simultaneous GPU jobs by default.

A CPU summary job starts only if the entire array succeeds. Normal Slurm
options may be appended, for example:

```bash
bash slurm/submit_v2_confirmatory_2021_2024.sh --account=g0609
```

Monitor with the job IDs printed by the launcher:

```bash
squeue -u "$USER"
tail -f logs/bdhires-v2-confirm-<ARRAY_JOB_ID>_*.out
```

Defaults can be overridden with environment variables, including
`V2_CONFIRM_ROOT`, `V2_CONFIRM_MEMBERS`, `BMD_CKPT`, `BMD_DATA_DIR`, and
`V2_CONFIRM_NATIVE_2021` through `V2_CONFIRM_NATIVE_2024`.

## Outputs and comparison directory

Everything for this experiment is under:

```text
data/processed/v2_confirmatory_2021_2024/
├── cv/
│   ├── 2021_may_sep/fold{0..4}.{npz,json}
│   ├── 2022_may_sep/fold{0..4}.{npz,json}
│   ├── 2023_may_sep/fold{0..4}.{npz,json}
│   └── 2024_may_jun/fold{0..4}.{npz,json}
├── gridded/
│   ├── 2021_may_sep.zarr
│   ├── 2022_may_sep.zarr
│   ├── 2023_may_sep.zarr
│   └── 2024_may_jun.zarr
├── production_metadata/
├── imerg_native/
├── imerg_s04/
└── summary/
    ├── confirmatory_selection.md
    ├── confirmatory_selection.json
    ├── confirmatory_selection.png
    ├── gridded_catalog.json
    └── fold_plots/{period}_fold{0..4}_diagnostics.png
```

Use `summary/` for the formal method comparison. Use `gridded/` for monthly
maps and variability analyses. Do not compare this run against the earlier
ten-day directory as though all 520 days were an independent test; the summary
already applies the selection-period exclusion.

## Zarr schema

Each completed store is consolidated Zarr v2 with root schema
`bdhires.physical_ensemble.v1`. Important variables are:

| Variable | Dimensions | Meaning |
|:--|:--|:--|
| `precipitation` | method, time, member, lat, lon | full daily physical ensemble, mm/day |
| `ensemble_mean` | method, time, lat, lon | daily ensemble mean |
| `ensemble_std` | method, time, lat, lon | within-day posterior ensemble spread |
| `cpc` | time, lat, lon | checkpoint CPC input |
| `chirps` | time, lat, lon | matched CHIRPS target/product |
| `imerg` | time, imerg_lat, imerg_lon | assimilated S04 observation product |
| `gauge` | time, station | BMD observations |
| `assimilated_station` | station | whether a station entered this product |
| `valid` | lat, lon | permanent model-domain mask |

Ocean/permanently invalid cells are NaN in precipitation fields. A true zero
rainfall value is stored and decoded as zero, not as missing. The four stores
hold about 5.1 GB of raw float32 ensemble values before Zstandard compression,
plus summaries and inputs.

## Later monthly analysis

Select by passing a dictionary to xarray because `method` is also the name of
an optional `.sel()` argument:

```python
from pathlib import Path
import xarray as xr

root = Path("data/processed/v2_confirmatory_2021_2024/gridded")
paths = sorted(root.glob("*.zarr"))
variables = ["precipitation", "ensemble_mean", "ensemble_std", "cpc", "chirps", "imerg"]
parts = [xr.open_zarr(path, consolidated=True)[variables] for path in paths]
data = xr.concat(parts, dim="time", join="exact").sortby("time")

name = "v2_simul_s04_ig010"
daily_mean = data.ensemble_mean.sel({"method": name})
monthly_mean = daily_mean.resample(time="MS").mean("time")
monthly_daily_variability = daily_mean.resample(time="MS").std("time")
monthly_posterior_spread = (
    data.ensemble_std.sel({"method": name}).resample(time="MS").mean("time")
)
monthly_member_means = (
    data.precipitation.sel({"method": name}).resample(time="MS").mean("time")
)
```

These are different quantities: `monthly_daily_variability` measures temporal
variation of daily ensemble means, while `monthly_posterior_spread` measures
average within-day ensemble uncertainty. `monthly_member_means` retains the 30
members for probabilistic monthly comparisons.

## Safe restart behavior

Rerunning the submission command reuses a task only when all expected outputs
are present. A lone `.npz`, `.json`, Zarr directory, or `.zarr.incomplete`
directory is treated as a partial result and stops that task. Inspect it and
move it aside before resubmitting; the launcher never silently overwrites a
scientific product.
