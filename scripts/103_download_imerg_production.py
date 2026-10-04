#!/usr/bin/env python3
"""Resumable, globally bounded IMERG half-hourly acquisition and preparation.

Three serial HTTP workers share one coordinator; scientific NetCDF work runs
in subprocesses because the underlying libraries are not thread safe.
"""
from __future__ import annotations

import argparse
import concurrent.futures
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
import threading
import time
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
BBOX = "20.3,87.6,26.7,94.0"
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
                    "urs.earthdata.nasa.gov", urlparse(HH.SERVICE).hostname):
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
                    raise FileNotFoundError("GES DISC granule not found: " + request.output_name)
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
        manifest.write_text("\n".join(request.url for request in granules) + "\n")
        for index, request in enumerate(granules, 1):
            if stop.is_set():
                raise RuntimeError("Stopped after an Earthdata authentication failure")
            if HH.valid_netcdf(raw / request.output_name):
                result = "cached"
            elif getattr(args, "transport", "requests") == "wget":
                result = download_wget(request, raw)
            else:
                if not hasattr(LOCAL, "session"):
                    LOCAL.session = make_session()
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
    # An OS lock is released on exit/Slurm termination. It prevents duplicate
    # coordinators on the same raw archive, even with different state folders.
    with (raw / ".production_download.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another production IMERG downloader already owns this raw archive") from None
        return run_locked(args, months)


def run_locked(args, months):
    state = Path(args.state)
    state.mkdir(parents=True, exist_ok=True)
    ready = state / "IMERG_READY.json"
    ready.unlink(missing_ok=True)
    required = sum(month_count(month) for month in months)
    summary = {"status": "running", "start": months[0]["start"], "end": months[-1]["end"],
               "required_granules": required, "connections": args.connections,
               "transport": getattr(args, "transport", "requests"),
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
    parser.add_argument("--validate-file", help=argparse.SUPPRESS)
    parser.add_argument("--validate-start", help=argparse.SUPPRESS)
    parser.add_argument("--validate-end", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.validate_file:
        PROD.validate_imerg(Path(args.validate_file), args.validate_start, args.validate_end)
        print("Validated " + args.validate_file)
        return
    if args.transport == "auto":
        args.transport = "requests" if importlib.util.find_spec("requests") else "wget"
    if not 2001 <= args.start_year <= args.end_year <= 2024:
        parser.error("Year range must be within 2001..2024")
    months = PROD.monthly(args.start_year, args.end_year)
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
    if args.dry_run:
        present = sum(prepared_path(args.daily, month).is_file() for month in months)
        print(f"Dry run: {present} prepared monthly files present (contents not validated); no writes/downloads")
        return
    if args.transport == "wget" and not shutil.which("wget"):
        parser.error("wget is unavailable; select an existing Python environment with requests installed")
    if args.transport == "requests" and not importlib.util.find_spec("requests"):
        parser.error("requests is unavailable in this interpreter; use --transport=auto or --transport=wget")
    run(args, months)


if __name__ == "__main__":
    main()
