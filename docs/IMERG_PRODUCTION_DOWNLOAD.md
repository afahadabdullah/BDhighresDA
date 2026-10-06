# Half-hourly IMERG for CPCv2 production, 2001–2024

The dedicated CPU job downloads regional **GPM_3IMERGHH V07B Final**
`precipitation` and `randomError` subsets, then runs the existing preparation
and production validators. It preserves the established **03:00–03:00 UTC**
reporting windows, rainfall integration and random-error quadrature. It does
not change gauge dates, the CPCv2 checkpoint, normalization or DA settings.

Acquisition now uses **cloud OPeNDAP** collection `C2723754847-GES_DISC`, as
listed by NASA's current CMR catalog, rather than the older
`gpm1/.../HTTP_services.cgi` subset service. The old request returned HTTP 404
for granules that still appear in the catalog; that response does not establish
a missing source file. The downloader reads the server's DMR metadata to learn
variable/group paths and dimension order, then requests native 64 × 64 cells
using `.dap.nc4?dap4.ce=...`. This is the
[cloud subset syntax recommended by GES DISC](https://forum.earthdata.nasa.gov/viewtopic.php?p=26549&sid=37f9a70a67831b9ad4f79ebaec86e5f2).

The complete calendar has 8,766 reporting days, 288 months and **420,768
half-hourly granules**. Source intervals begin **2000-12-31 03:00 UTC** and
finish with **2024-12-31 02:30 UTC**. January's previous-year boundary is stored
under the reporting year's raw directory, as in the production downloader.

## Submit on PRISM

From the existing repository, update the branch containing these scripts:

```bash
# Stop the previous failing coordinator if it is still running.
scancel 37938702
git pull --ff-only origin codex/imerg-daily-window-pilot
bash slurm/submit_imerg_production_download.sh --parallel-years --dry-run
bash slurm/submit_imerg_production_download.sh --parallel-years
```

Add `--account=YOUR_ACCOUNT` if required. Parallel mode submits a dependency
chain: **fresh one-granule probe → year array → final collection audit**.
The probe checks authenticated transfer, required variables, units and native
regional grid before releasing 24 year tasks (`--array=0-23%15`). At most
**15 years run simultaneously, with one transfer worker each**, as requested.
Override the year-task limit with `IMERG_YEAR_CONCURRENCY` (1..24). Each year
task has 2 CPUs, 12 GB memory and its own 24-hour wall limit; the probe and
collector use 8 CPUs and 24 GB. All jobs run on `grace-cpuonly` without a GPU.
The job uses the existing GH200 scientific environment:
`/home/afahad/nb/project/BDDA/envs/bdda-gh200/bin/python`. Override it with
`IMERG_PYTHON` if necessary; do not run this ARM interpreter on an x86 login
node. The job checks `numpy`, `xarray` and `netCDF4` before work. When `requests`
is installed, the downloader uses persistent HTTP sessions; otherwise it falls
back to the existing `wget` downloader without installing packages. Force a
transport with `--transport=requests` or `--transport=wget` if needed.

An existing Earthdata `~/.netrc` entry for `urs.earthdata.nasa.gov` and authorized
GES DISC/cloud OPeNDAP access are required. Preserve private credentials outside Git and use
`chmod 600 ~/.netrc`. Existing `~/.urs_cookies` are reused when readable;
otherwise workers establish fresh authenticated sessions. Cookies and
credentials are not written to progress reports.

The Requests transport retries an HTTP 401 once after clearing that worker's
in-memory session cookies, using the existing netrc credentials and redirect
handling. This covers a stale-cookie failure during either metadata access or
granule transfer. The shared `~/.urs_cookies` file is not modified. Persistent
401 responses and HTTP 403 still stop the worker without issuing a readiness
receipt. The log records `[auth]` when the cookie refresh is attempted; this
does not establish that session expiry caused a particular failure.

Cloud subset HTTP 404 responses receive the same five-attempt retry budget as
other transfer errors, with waits of 2, 4, 8 and 16 seconds. The log records
`[retry]` for each retry. A persistent 404 still fails that month; granules are
never replaced with zero rainfall or skipped to obtain readiness. A subset
404 alone does not demonstrate that the source granule is absent from CMR.

Omitting `--parallel-years` retains one coordinator with at most
**three serial download workers in total**, 8 CPUs, 24 GB memory and a 24-hour
wall limit. `IMERG_DOWNLOAD_CONNECTIONS` controls this coordinator's workers;
`IMERG_YEAR_CONCURRENCY` independently controls the year array's task limit.
With the Requests transport, each worker reuses its HTTP session across
granules and months. For conservative transfer concurrency matching
[GES DISC's three-connection guidance](https://forum.earthdata.nasa.gov/viewtopic.php?p=23888&sid=976eaeb9c72fd601c7f52271ce7fec9c),
set `IMERG_YEAR_CONCURRENCY=3`; the requested default is now 15 active transfers.
Do not run the old IMERG download arrays alongside this job: their connections
would add to this total. Submit one workflow at a time. Shared archive and
exclusive reporting-year locks permit disjoint year jobs and prevent
overlapping downloads, including overlap with the older coordinator.

Optional examples:

```bash
# A single month, using the same preparation/validation path.
bash slurm/submit_imerg_production_download.sh --month=2022-05

# Fewer connections, selected years, or a different wall limit.
IMERG_START_YEAR=2021 IMERG_END_YEAR=2024 IMERG_YEAR_CONCURRENCY=2 \
  bash slurm/submit_imerg_production_download.sh --parallel-years --time=18:00:00
```

Raise an existing array's task limit without resubmitting or changing the
collector's dependency:

```bash
scontrol update JobId=37938704 ArrayTaskThrottle=15
```

Default paths can be overridden with `SURMA_PROD_IMERG_RAW`,
`SURMA_PROD_IMERG_DAILY` and `IMERG_DOWNLOAD_STATE`. Use the same raw/prepared
overrides when subsequently submitting the CPCv2 production launcher.

## Follow progress and resume

The submitter prints the job ID and exact log path:

```bash
tail -f logs/imerg-hh-production-JOBID.out
cat data/processed/imerg_download_2001_2024/status.json
tail -f data/processed/imerg_download_2001_2024/logs/2022-05.out

# In year-array mode (task index = YEAR - IMERG_START_YEAR):
tail -f logs/imerg-hh-year-ARRAYID_21.out  # 2022 for the default 2001 start
cat data/processed/imerg_download_2001_2024/years/2022/status.json
cat data/processed/imerg_download_2001_2024/probe.json
```

The top-level status is `probing` until the array's final collector issues the
full readiness receipt. Follow individual year reports while the array runs.
If the probe fails, the array cannot start; inspect the probe job log and
`probe.out` before fixing credentials/access and resubmitting. If a year task
fails or times out, the collector cannot issue readiness. Cancel any old
pending dependency jobs before resubmitting the same workflow. Valid files
from completed or partially completed years are reused.

Public CMR metadata and URL/schema construction are verified locally. The
authenticated cloud transfer is checked by the PRISM probe, where Earthdata
credentials are available; a local dry run does not certify remote transfer.

`status.json` reports newly downloaded granules, raw/prepared cache credits,
validated/failed months and elapsed time. After 100 new downloads it estimates
the remaining download time at the measured rate. Unchecked caches are counted
as remaining work until inspected; preparation and queue time are additional,
and future NASA response times can change.

For an empty cache, 12-hour completion needs about **9.7 granules/second** in
aggregate, before preparation. Overnight completion is therefore a target,
not a guarantee. Prepared months and raw granules already on PRISM can reduce
the transfer workload substantially.

After a timeout, network failure or interrupted job, **rerun the same submit
command**. Valid prepared months undergo the existing production validator
before being skipped. Raw cache files are screened by NetCDF/HDF signature;
missing or invalid responses are downloaded to `.part` files and atomically
promoted. Transient failures are retried up to five times with backoff. The
Requests transport stops further acquisition on HTTP 401/403; the wget fallback
stops on an explicit authentication failure. Fix Earthdata access before retrying.

Each completed month is prepared with all 48 half-hours required per valid
footprint, validated for dates, 03 UTC support, native grid and rainfall/error
fields, then promoted to its canonical filename. A corrupt raw file that
passes the signature check can still fail scientific preparation: inspect
that month's log, remove the identified corrupt granule and resubmit. Failed
months yield a nonzero job exit and no readiness receipt. Raw files are retained.

## Production handoff

Outputs use the existing production paths:

```text
data/imerg_halfhourly/YEAR/3B-HHR....V07B.HDF5.SUB.nc4
data/processed/imerg_bd_aligned_YYYYMMDD_YYYYMMDD.nc
data/processed/imerg_bd_aligned_YYYYMMDD_YYYYMMDD_qc.json
data/processed/imerg_download_2001_2024/IMERG_READY.json
```

The readiness receipt is issued only after every **selected** month validates.
A `--month` test produces a receipt for that selection, not the full archive.
In array mode, each year writes `years/YEAR/IMERG_READY.json`; the final
collector verifies all year receipts and revalidates every monthly file before
writing the top-level receipt. A missing, failed or mismatched year blocks it.
Before the full production run, check the default full-period receipt:

```bash
python3 - <<'PY'
import json
with open('data/processed/imerg_download_2001_2024/IMERG_READY.json') as handle:
    report = json.load(handle)
assert report['status'] == 'ready'
assert report['start'] == '2001-01-01' and report['end'] == '2024-12-31'
assert len(report['months']) == 288
assert all(month['status'] == 'validated' for month in report['months'].values())
print('All 288 IMERG months ready for CPCv2 production')
PY

bash slurm/submit_surma_production_2001_2024.sh --audit-only
# Review input_inventory.json after the audit finishes, then:
bash slurm/submit_surma_production_2001_2024.sh
```

The existing launcher validates and reuses these months. IMERG readiness covers
IMERG only: the trained CPCv2 weights/statistics, full-history national gauges,
static grid and CPC/ERA5 predictors must also pass the production preflight.
See [the complete production runbook](SURMA_PRODUCTION_2001_2024.md).
