#!/usr/bin/env python3
"""Resumable, globally bounded IMERG half-hourly acquisition and preparation.

Three serial HTTP workers share one coordinator; scientific NetCDF work runs
in subprocesses because the underlying libraries are not thread safe.
"""
from __future__ import annotations

import argparse
import concurrent.futures
from contextlib import ExitStack
from dataclasses import replace
from datetime import date
import fcntl
import importlib.util
import json
import netrc
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import quote, urlencode, urlparse
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
BBOX = "20.3,87.6,26.7,94.0"
CLOUD = "https://opendap.earthdata.nasa.gov"
COLLECTION = "C2723754847-GES_DISC"
LOCAL = threading.local()


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


HH = load_script("_imerg_hh_download", "02_download_imerg_halfhourly.py")
PROD = load_script("_imerg_production", "100_surma_production.py")


def prepared_path(folder, month):
    return Path(folder) / (
        "imerg_bd_aligned_" + month["start"].replace("-", "") + "_"
        + month["end"].replace("-", "") + ".nc"
    )


def requests_for(month):
    return [HH.request_for(value, BBOX) for value in HH.bmd_interval_starts(
        date.fromisoformat(month["start"]), date.fromisoformat(month["end"]), 3)]


def cloud_base(request):
    identifier = HH.DATASET + ":" + request.source_name
    return f"{CLOUD}/collections/{COLLECTION}/granules/{quote(identifier, safe='')}"


def schema_from_dmr(document):
    """Read actual variable paths/dimension order rather than assuming a Grid group."""
    root = ET.fromstring(document)
    variables = {}
    def walk(group, prefix=""):
        for child in group:
            tag = child.tag.rsplit("}", 1)[-1]
            name = child.attrib.get("name", "")
            if tag == "Group":
                walk(child, prefix + "/" + name)
            elif name in ("precipitation", "randomError", "lat", "lon", "time"):
                dimensions = [item.attrib.get("name", "").rsplit("/", 1)[-1]
                              for item in child if item.tag.rsplit("}", 1)[-1] == "Dim"]
                if tag not in ("Dimension", "Attribute"):
                    variables[name] = (prefix + "/" + name, dimensions)
    walk(root)
    for name in ("precipitation", "randomError", "lat", "lon", "time"):
        if name not in variables:
            raise ValueError("Cloud IMERG metadata lacks " + name)
    for name in ("precipitation", "randomError"):
        if sorted(variables[name][1]) != ["lat", "lon", "time"]:
            raise ValueError("Unsupported IMERG dimensions: " + name)
    for name in ("lat", "lon", "time"):
        if variables[name][1] != [name]:
            raise ValueError("Unsupported IMERG coordinate dimensions: " + name)
    return variables


def cloud_request(request, schema):
    # Global V07B centres: -179.95 + 0.1*i; -89.95 + 0.1*j.
    # Native BD cells: 87.65..93.95 E and 20.35..26.65 N, 64 x 64.
    slices = {"time": "[0:1:0]", "lon": "[2676:1:2739]", "lat": "[1103:1:1166]"}
    projections = []
    for name in ("precipitation", "randomError", "lat", "lon", "time"):
        path, dimensions = schema[name]
        projections.append(path + "".join(slices[dimension] for dimension in dimensions))
    return replace(request, url=cloud_base(request) + ".dap.nc4?" + urlencode({"dap4.ce": ";".join(projections)}))


def resolve_cloud_request(request, args):
    if not hasattr(LOCAL, "schema"):
        metadata_url = cloud_base(request) + ".dmr"
        if getattr(args, "transport", "requests") == "wget":
            with tempfile.TemporaryDirectory(dir=args.state) as folder:
                path = Path(folder) / "schema.xml"
                command = ["wget", "--auth-no-challenge=on", "--tries=3", "--timeout=120", "-q", "-O", str(path), metadata_url]
                cookies = Path.home() / ".urs_cookies"
                if cookies.is_file():
                    command[1:1] = ["--load-cookies", str(cookies)]
                subprocess.run(command, check=True)
                document = path.read_bytes()
        else:
            if not hasattr(LOCAL, "session"):
                LOCAL.session = make_session()
            with LOCAL.session.get(metadata_url, timeout=(30, 180)) as response:
                if response.status_code in (401, 403):
                    raise AuthenticationError("Earthdata denied cloud OPeNDAP metadata access; check ~/.netrc and application authorization")
                response.raise_for_status()
                document = response.content
        LOCAL.schema = schema_from_dmr(document)
    return cloud_request(request, LOCAL.schema)


def month_count(month):
    return ((date.fromisoformat(month["end"]) - date.fromisoformat(month["start"])).days + 1) * 48


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


class AuthenticationError(RuntimeError):
    pass


def make_session():
    import requests
    from http.cookiejar import MozillaCookieJar

    try:
        credentials = netrc.netrc().authenticators("urs.earthdata.nasa.gov")
    except (OSError, netrc.NetrcParseError):
        raise AuthenticationError("Cannot read Earthdata ~/.netrc; check its format and permissions") from None
    if not credentials:
        raise AuthenticationError("Add the Earthdata urs.earthdata.nasa.gov entry to ~/.netrc")

    class EarthdataSession(requests.Session):
        def rebuild_auth(self, prepared_request, response):
            # Keep authentication only on the two explicitly trusted HTTPS hosts.
            target = urlparse(prepared_request.url)
            if target.scheme != "https" or target.hostname not in (
                    "urs.earthdata.nasa.gov", urlparse(HH.SERVICE).hostname, urlparse(CLOUD).hostname):
                prepared_request.headers.pop("Authorization", None)
            else:
                prepared_request.prepare_auth(self.auth)

    session = EarthdataSession()
    session.auth = (credentials[0], credentials[2])
    session.headers["User-Agent"] = "BDhighresDA-IMERG-production/1.0"
    session.mount("https://", requests.adapters.HTTPAdapter(
        pool_connections=2, pool_maxsize=1, pool_block=True, max_retries=0))
    cookies = Path.home() / ".urs_cookies"
    if cookies.is_file():
        jar = MozillaCookieJar(str(cookies))
        try:
            jar.load(ignore_discard=True)
            session.cookies.update(jar)
        except (OSError, ValueError):
            pass  # A fresh authenticated session can obtain its own cookies.
    return session


def download_one(request, output, session, retries=5):
    """Promote only complete binary responses; safely retry transient failures."""
    destination = output / request.output_name
    if HH.valid_netcdf(destination):
        return "cached"
    temporary = destination.with_name(destination.name + ".part")
    for attempt in range(retries):
        try:
            with session.get(request.url, stream=True, timeout=(30, 180)) as response:
                if response.status_code in (401, 403):
                    raise AuthenticationError(
                        f"Earthdata HTTP {response.status_code}; check ~/.netrc and GES DISC authorization")
                if response.status_code == 404:
                    final_url = getattr(response, "url", None)
                    responding_host = urlparse(final_url if isinstance(final_url, str) else request.url).hostname
                    raise FileNotFoundError("HTTP 404 from " + str(responding_host)
                                            + " for " + request.output_name
                                            + "; endpoint/subset failure does not establish a missing archive granule")
                response.raise_for_status()
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            if not HH.valid_netcdf(temporary):
                raise RuntimeError("GES DISC returned a non-NetCDF response")
            temporary.replace(destination)
            return "downloaded"
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            if isinstance(exc, (AuthenticationError, FileNotFoundError)):
                raise
            if attempt + 1 == retries:
                # Do not print response bodies, credentials or authentication URLs.
                raise RuntimeError(f"{request.output_name}: failed after {retries} attempts ({type(exc).__name__})") from None
            time.sleep(min(60, 2 ** (attempt + 1)))


def download_wget(request, output):
    """Keep the existing authenticated transport available without installation."""
    try:
        result = HH.download_one(request, output, Path.home() / ".urs_cookies")
        return "cached" if result == "skipped" else result
    except subprocess.CalledProcessError as exc:
        if exc.returncode == 6:  # wget's username/password authentication failure
            raise AuthenticationError("wget authentication failed; check Earthdata ~/.netrc") from None
        raise RuntimeError(f"wget failed for {request.output_name} (exit {exc.returncode})") from None
    except Exception as exc:
        raise RuntimeError(f"wget failed for {request.output_name} ({type(exc).__name__})") from None


def scientific_command(arguments, log):
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    command = [sys.executable, "-u", *map(str, arguments)]
    log.write("[command] " + " ".join(command) + "\n")
    log.flush()
    subprocess.run(command, check=True, cwd=ROOT, env=environment, stdout=log, stderr=log)


def validate_month(path, month, log):
    scientific_command([Path(__file__), "--validate-file", path,
                        "--validate-start", month["start"], "--validate-end", month["end"]], log)


def process_month(args, month, progress, stop):
    key = month["start"][:7]
    output = prepared_path(args.daily, month)
    log_path = Path(args.state) / "logs" / (key + ".out")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", buffering=1) as log:
        if stop.is_set():
            raise RuntimeError("Stopped after an Earthdata authentication failure")
        if output.is_file():
            try:
                validate_month(output, month, log)
            except subprocess.CalledProcessError:
                log.write("Existing prepared month failed validation; rebuilding from half-hours.\n")
            else:
                progress("prepared_cache", month_count(month))
                return {"status": "validated", "reused": True, "file": str(output)}
        raw = Path(args.raw) / month["start"][:4]
        raw.mkdir(parents=True, exist_ok=True)
        granules = requests_for(month)
        manifest = raw / f"_urls_bmd_{month['start'].replace('-', '')}_{month['end'].replace('-', '')}_end03utc.txt"
        # The old OTF URL is deliberately excluded: discover cloud variable paths
        # once per worker, then write the actual DAP4 URLs as downloads are needed.
        with manifest.open("w") as urls:
            urls.write("# Cloud DAP4 subset URLs used for missing granules\n")
        for index, request in enumerate(granules, 1):
            if stop.is_set():
                raise RuntimeError("Stopped after an Earthdata authentication failure")
            if HH.valid_netcdf(raw / request.output_name):
                result = "cached"
            else:
                request = resolve_cloud_request(request, args)
                with manifest.open("a") as urls:
                    urls.write(request.url + "\n")
                if getattr(args, "transport", "requests") == "wget":
                    result = download_wget(request, raw)
                else:
                    result = download_one(request, raw, LOCAL.session)
            progress(result, 1)
            if index == 1 or index % 100 == 0 or index == len(granules):
                log.write(f"[{index}/{len(granules)}] {result}: {request.output_name}\n")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.stem + ".pending.nc")
        qc = output.with_name(output.stem + "_qc.json")
        temporary_qc = qc.with_name(qc.name + ".pending")
        try:
            scientific_command([ROOT / "scripts/08_prepare_imerg_observations.py",
                                "--input", raw, "--source-frequency", "half-hourly",
                                "--start", month["start"], "--end", month["end"],
                                "--min-count", 48, "--accumulation-end-hour-utc", 3,
                                "--out", temporary, "--report", temporary_qc], log)
            validate_month(temporary, month, log)
            # The preparation script's QC references the raw granules, not this filename.
            temporary_qc.replace(qc)
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
            temporary.with_suffix(temporary.suffix + ".tmp").unlink(missing_ok=True)
            temporary_qc.unlink(missing_ok=True)
        return {"status": "validated", "reused": False, "file": str(output)}


def run(args, months):
    raw = Path(args.raw)
    raw.mkdir(parents=True, exist_ok=True)
    # Shared root lock excludes older single-coordinator versions, which held
    # it exclusively. Exclusive reporting-year locks permit disjoint year jobs.
    with ExitStack() as stack:
        try:
            root_lock = stack.enter_context((raw / ".production_download.lock").open("a"))
            fcntl.flock(root_lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            for year in sorted({month["start"][:4] for month in months}):
                folder = raw / year
                folder.mkdir(parents=True, exist_ok=True)
                lock = stack.enter_context((folder / ".production_download.lock").open("a"))
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another production IMERG downloader already owns this raw archive") from None
        if getattr(args, "probe_only", False):
            return probe(args, months)
        if getattr(args, "collect_years", False):
            return collect_years(args, months)
        return run_locked(args, months)


def run_locked(args, months):
    state = Path(args.state)
    state.mkdir(parents=True, exist_ok=True)
    ready = state / "IMERG_READY.json"
    ready.unlink(missing_ok=True)
    required = sum(month_count(month) for month in months)
    summary = {"status": "running", "start": months[0]["start"], "end": months[-1]["end"],
               "required_granules": required, "connections": args.connections,
               "transport": getattr(args, "transport", "requests"), "source": "cloud-opendap",
               "downloaded": 0, "cached": 0, "prepared_cache": 0, "months": {}}
    mutex = threading.Lock()
    started = time.monotonic()
    stop = threading.Event()

    def progress(kind, count):
        with mutex:
            summary[kind] += count

    def snapshot():
        with mutex:
            elapsed = time.monotonic() - started
            summary["elapsed_seconds"] = round(elapsed, 1)
            remaining = required - sum(summary[k] for k in ("downloaded", "cached", "prepared_cache"))
            summary["remaining_unchecked_or_missing_granules"] = remaining
            rate = summary["downloaded"] / elapsed if elapsed else 0
            summary["downloaded_granules_per_second"] = round(rate, 4)
            # Conservative until remaining caches are inspected; preparation is additional.
            summary["download_eta_hours_at_current_rate"] = round(remaining / rate / 3600, 2) if summary["downloaded"] >= 100 else None
            write_json(state / "status.json", summary)
            valid = sum(value["status"] == "validated" for value in summary["months"].values())
            print(f"[progress] months {valid}/{len(months)} validated; new {summary['downloaded']}; "
                  f"raw cached {summary['cached']}; prepared cached granules {summary['prepared_cache']}; "
                  f"unchecked/missing {remaining}; download ETA at current rate "
                  f"{summary['download_eta_hours_at_current_rate']} hours (preparation additional)", flush=True)

    snapshot()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.connections) as executor:
        futures = {executor.submit(process_month, args, month, progress, stop): month for month in months}
        pending = set(futures)
        while pending:
            done, pending = concurrent.futures.wait(pending, timeout=30,
                                                    return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                key = futures[future]["start"][:7]
                try:
                    result = future.result()
                except Exception as exc:
                    if isinstance(exc, AuthenticationError):
                        stop.set()
                    result = {"status": "failed", "error": str(exc),
                              "log": str(state / "logs" / (key + ".out"))}
                    failure_log = Path(result["log"])
                    failure_log.parent.mkdir(parents=True, exist_ok=True)
                    with failure_log.open("a") as log:
                        log.write(f"[failed] {key}: {exc}\n")
                    print(f"[failed] {key}: {exc}; see {result['log']}", file=sys.stderr, flush=True)
                with mutex:
                    summary["months"][key] = result
            snapshot()
    failed = [key for key, value in summary["months"].items() if value["status"] != "validated"]
    summary["status"] = "failed" if failed else "ready"
    snapshot()
    if failed:
        raise RuntimeError(f"{len(failed)} months failed. Rerun the same submission to resume; no IMERG_READY receipt issued.")
    write_json(ready, summary)
    print(f"[ready] All {len(months)} selected IMERG months validated: {ready}", flush=True)


def probe(args, months):
    """Fresh authenticated transfer and scientific check before releasing an array."""
    state = Path(args.state)
    state.mkdir(parents=True, exist_ok=True)
    (state / "IMERG_READY.json").unlink(missing_ok=True)
    write_json(state / "status.json", {"status": "probing", "source": "cloud-opendap"})
    request = resolve_cloud_request(requests_for(months[0])[0], args)
    with tempfile.TemporaryDirectory(dir=state) as folder:
        output = Path(folder)
        if args.transport == "wget":
            download_wget(request, output)
        else:
            download_one(request, output, LOCAL.session)
        with (state / "probe.out").open("a") as log:
            scientific_command([Path(__file__), "--validate-granule", output / request.output_name], log)
    write_json(state / "probe.json", {"status": "validated", "source": "cloud-opendap",
                                     "granule": request.source_name, "subset_url": request.url})
    print("[probe] Fresh cloud subset passed regional variable/unit/grid validation", flush=True)


def collect_years(args, months):
    """Audit every year receipt and prepared file before the full-period handoff."""
    state = Path(args.state)
    state.mkdir(parents=True, exist_ok=True)
    ready = state / "IMERG_READY.json"
    ready.unlink(missing_ok=True)
    reports = {}
    for year in sorted({month["start"][:4] for month in months}):
        path = state / "years" / year / "IMERG_READY.json"
        report = json.loads(path.read_text())
        if (report.get("status") != "ready" or report.get("source") != "cloud-opendap"
                or report.get("start") != year + "-01-01" or report.get("end") != year + "-12-31"):
            raise ValueError("Incomplete year receipt: " + str(path))
        reports[year] = report
    merged = {}
    with (state / "collection_validation.out").open("a") as log:
        for month in months:
            key = month["start"][:7]
            recorded = reports[key[:4]]["months"].get(key, {})
            path = prepared_path(args.daily, month)
            if recorded.get("status") != "validated" or Path(recorded.get("file", "")).resolve() != path.resolve():
                raise ValueError("Missing/mismatched year result: " + key)
            validate_month(path, month, log)
            merged[key] = recorded
    summary = {"status": "ready", "source": "cloud-opendap", "start": months[0]["start"],
               "end": months[-1]["end"], "required_granules": sum(month_count(month) for month in months),
               "months": merged, "year_receipts": list(reports)}
    write_json(state / "status.json", summary)
    write_json(ready, summary)
    print(f"[ready] All {len(months)} year-array months validated: {ready}")


def validate_granule(path):
    import numpy as np
    from bdhires.imerg import _open_granule, _regional_array, _require_mm_per_hour, _coarse_centres
    from bdhires.grids import BD
    lat, lon = _coarse_centres(BD, 2)
    with _open_granule(Path(path), required=frozenset({"precipitation", "randomError"})) as ds:
        _require_mm_per_hour(ds, Path(path))
        for name in ("precipitation", "randomError"):
            values = _regional_array(ds, name, lat, lon)
            if not np.any(np.isfinite(values) & (values >= 0)):
                raise ValueError("No valid regional values in " + name)
    print("Validated regional half-hourly granule " + str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2001)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--month", action="append", help="Optional YYYY-MM selection within the year range")
    parser.add_argument("--connections", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument("--transport", choices=("auto", "requests", "wget"), default="auto",
                        help="auto reuses Requests sessions when installed, otherwise uses the existing wget downloader")
    parser.add_argument("--raw", default="data/imerg_halfhourly")
    parser.add_argument("--daily", default="data/processed")
    parser.add_argument("--state", default="data/processed/imerg_download_2001_2024")
    parser.add_argument("--dry-run", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--probe-only", action="store_true")
    mode.add_argument("--collect-years", action="store_true")
    parser.add_argument("--validate-granule", help=argparse.SUPPRESS)
    parser.add_argument("--validate-file", help=argparse.SUPPRESS)
    parser.add_argument("--validate-start", help=argparse.SUPPRESS)
    parser.add_argument("--validate-end", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.validate_granule:
        sys.path.insert(0, str(ROOT / "src"))
        validate_granule(args.validate_granule)
        return
    if args.validate_file:
        PROD.validate_imerg(Path(args.validate_file), args.validate_start, args.validate_end)
        print("Validated " + args.validate_file)
        return
    if args.transport == "auto":
        args.transport = "requests" if importlib.util.find_spec("requests") else "wget"
    if not 2001 <= args.start_year <= args.end_year <= 2024:
        parser.error("Year range must be within 2001..2024")
    months = PROD.monthly(args.start_year, args.end_year)
    if args.collect_years and args.month:
        parser.error("--collect-years requires complete years, without --month")
    if args.month:
        available = {month["start"][:7] for month in months}
        if not set(args.month) <= available:
            parser.error("--month must be YYYY-MM within the requested years")
        months = [month for month in months if month["start"][:7] in args.month]
    granules = sum(month_count(month) for month in months)
    first, last = requests_for(months[0])[0], requests_for(months[-1])[-1]
    print(f"IMERG V07B Final: {len(months)} months, {granules:,} half-hours; bbox {BBOX}")
    print(f"Interval starts: {first.start} through {last.start} UTC; reporting windows end 03 UTC")
    print(f"At most {args.connections} concurrent downloads; valid raw/prepared caches reused")
    print(f"HTTP transport: {args.transport}")
    print(f"Source: cloud OPeNDAP collection {COLLECTION}; DAP4 regional NetCDF4 subsets")
    if args.dry_run:
        present = sum(prepared_path(args.daily, month).is_file() for month in months)
        print(f"Dry run: {present} prepared monthly files present (contents not validated); no writes/downloads")
        return
    if args.collect_years:
        run(args, months)
        return
    if args.transport == "wget" and not shutil.which("wget"):
        parser.error("wget is unavailable; select an existing Python environment with requests installed")
    if args.transport == "requests" and not importlib.util.find_spec("requests"):
        parser.error("requests is unavailable in this interpreter; use --transport=auto or --transport=wget")
    run(args, months)


if __name__ == "__main__":
    main()
