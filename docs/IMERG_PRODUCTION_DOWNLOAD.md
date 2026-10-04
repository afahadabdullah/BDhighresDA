# Half-hourly IMERG for CPCv2 production, 2001–2024

The dedicated CPU job downloads regional **GPM_3IMERGHH V07B Final**
`precipitation` and `randomError` subsets, then runs the existing preparation
and production validators. It preserves the established **03:00–03:00 UTC**
reporting windows, rainfall integration and random-error quadrature. It does
not change gauge dates, the CPCv2 checkpoint, normalization or DA settings.

The complete calendar has 8,766 reporting days, 288 months and **420,768
half-hourly granules**. Source intervals begin **2000-12-31 03:00 UTC** and
finish with **2024-12-31 02:30 UTC**. January's previous-year boundary is stored
under the reporting year's raw directory, as in the production downloader.

## Submit on PRISM

From the existing repository, update the branch containing these scripts:

```bash
git pull --ff-only origin codex/imerg-daily-window-pilot
bash slurm/submit_imerg_production_download.sh --dry-run
bash slurm/submit_imerg_production_download.sh
```

Add `--account=YOUR_ACCOUNT` if required. This submits **one CPU job**, with
8 CPUs, 24 GB memory and a 24-hour wall limit on `grace-cpuonly`. There is no
GPU reservation. The job uses the existing GH200 scientific environment:
`/home/afahad/nb/project/BDDA/envs/bdda-gh200/bin/python`. Override it with
`IMERG_PYTHON` if necessary; do not run this ARM interpreter on an x86 login
node. The job checks `numpy`, `xarray` and `netCDF4` before work. When `requests`
is installed, the downloader uses persistent HTTP sessions; otherwise it falls
back to the existing `wget` downloader without installing packages. Force a
transport with `--transport=requests` or `--transport=wget` if needed.

An existing Earthdata `~/.netrc` entry for `urs.earthdata.nasa.gov` and authorized
GES DISC access are required. Preserve private credentials outside Git and use
`chmod 600 ~/.netrc`. Existing `~/.urs_cookies` are reused when readable;
otherwise workers establish fresh authenticated sessions. Cookies and
credentials are not written to progress reports.

The coordinator runs at most **three serial download workers in total**.
With the Requests transport, each worker reuses its HTTP session across
granules and months. This follows
[GES DISC's published three-connection maximum](https://forum.earthdata.nasa.gov/viewtopic.php?p=23888&sid=976eaeb9c72fd601c7f52271ce7fec9c).
Do not run the old IMERG download arrays alongside this job: their connections
would add to this total. An advisory lock prevents two copies of the new
coordinator from using the same raw archive simultaneously.

Optional examples:

```bash
# A single month, using the same preparation/validation path.
bash slurm/submit_imerg_production_download.sh --month=2022-05

# Fewer connections, selected years, or a different wall limit.
IMERG_START_YEAR=2021 IMERG_END_YEAR=2024 IMERG_DOWNLOAD_CONNECTIONS=2 \
  bash slurm/submit_imerg_production_download.sh --time=18:00:00
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
```

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
