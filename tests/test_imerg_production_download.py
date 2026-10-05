"""Acquisition boundaries, atomic downloads, resume gates and Slurm submission."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_download_production_test", ROOT / "scripts/103_download_imerg_production.py")
DL = importlib.util.module_from_spec(spec)
spec.loader.exec_module(DL)
MONTH = {"start": "2022-05-01", "end": "2022-05-31"}
BINARY = b"\x89HDF\r\n\x1a\n" + b"x" * 1024
DMR = '''<Dataset xmlns="http://xml.opendap.org/ns/DAP/4.0#">
<Float32 name="precipitation"><Dim name="/time"/><Dim name="/lon"/><Dim name="/lat"/></Float32>
<Float32 name="randomError"><Dim name="/time"/><Dim name="/lon"/><Dim name="/lat"/></Float32>
<Float32 name="lat"><Dim name="/lat"/></Float32>
<Float32 name="lon"><Dim name="/lon"/></Float32>
<Int32 name="time"><Dim name="/time"/></Int32></Dataset>'''


def response(body=BINARY, status=200):
    result = Mock(status_code=status)
    result.__enter__ = Mock(return_value=result)
    result.__exit__ = Mock(return_value=False)
    result.iter_content.return_value = [body]
    return result


class DownloadTests(unittest.TestCase):
    def test_cloud_subset_uses_catalog_identity_actual_paths_and_native_cells(self):
        request = DL.requests_for(MONTH)[0]
        schema = DL.schema_from_dmr(DMR)
        cloud = DL.cloud_request(request, schema)
        parsed = urlparse(cloud.url)
        self.assertEqual(parsed.hostname, "opendap.earthdata.nasa.gov")
        self.assertIn("C2723754847-GES_DISC", parsed.path)
        self.assertIn("GPM_3IMERGHH.07%3A", parsed.path)
        self.assertTrue(parsed.path.endswith(".HDF5.dap.nc4"))
        ce = parse_qs(parsed.query)["dap4.ce"][0]
        self.assertIn("/precipitation[0:1:0][2676:1:2739][1103:1:1166]", ce)
        self.assertIn("/randomError[0:1:0][2676:1:2739][1103:1:1166]", ce)
        self.assertIn("/lat[1103:1:1166]", ce)
        self.assertNotIn("HTTP_services.cgi", cloud.url)
        self.assertEqual(cloud.output_name, request.output_name)

    def test_cloud_metadata_accepts_group_paths_and_dimension_order_changes(self):
        inner = DMR.split('>', 1)[1].rsplit('</Dataset>', 1)[0]
        document = '<Dataset xmlns="http://xml.opendap.org/ns/DAP/4.0#"><Group name="Grid">' + inner + '</Group></Dataset>'
        cloud = DL.cloud_request(DL.requests_for(MONTH)[0], DL.schema_from_dmr(document))
        ce = parse_qs(urlparse(cloud.url).query)["dap4.ce"][0]
        self.assertIn("/Grid/precipitation", ce)
        document = DMR.replace('<Dim name="/lon"/><Dim name="/lat"/>', '<Dim name="/lat"/><Dim name="/lon"/>')
        cloud = DL.cloud_request(DL.requests_for(MONTH)[0], DL.schema_from_dmr(document))
        self.assertIn("/precipitation[0:1:0][1103:1:1166][2676:1:2739]", parse_qs(urlparse(cloud.url).query)["dap4.ce"][0])
        with self.assertRaisesRegex(ValueError, "lacks randomError"):
            DL.schema_from_dmr(DMR.replace('name="randomError"', 'name="otherError"'))

    def test_cloud_schema_is_fetched_once_per_worker(self):
        request = DL.requests_for(MONTH)[0]
        for name in ("schema", "session"):
            if hasattr(DL.LOCAL, name): delattr(DL.LOCAL, name)
        session = Mock()
        metadata = response()
        metadata.content = DMR.encode()
        session.get.return_value = metadata
        args = SimpleNamespace(transport="requests")
        try:
            with patch.object(DL, "make_session", return_value=session):
                DL.resolve_cloud_request(request, args)
                DL.resolve_cloud_request(DL.requests_for(MONTH)[1], args)
            self.assertEqual(session.get.call_count, 1)
            self.assertTrue(session.get.call_args[0][0].endswith(".dmr"))
        finally:
            for name in ("schema", "session"):
                if hasattr(DL.LOCAL, name): delattr(DL.LOCAL, name)

    def test_full_calendar_and_year_boundary(self):
        months = DL.PROD.monthly(2001, 2024)
        self.assertEqual(len(months), 288)
        self.assertEqual(sum(DL.month_count(month) for month in months), 420768)
        self.assertEqual(str(DL.requests_for(months[0])[0].start), "2000-12-31 03:00:00")
        self.assertEqual(str(DL.requests_for(months[-1])[-1].start), "2024-12-31 02:30:00")
        feb = DL.PROD.monthly(2004, 2004)[1]
        self.assertEqual(DL.month_count(feb), 29 * 48)
        previous = DL.requests_for(months[0])[-1].start
        following = DL.requests_for(months[1])[0].start
        self.assertEqual((following - previous).total_seconds(), 1800)

    def test_atomic_download_and_cached_file_avoids_http(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            request = DL.requests_for(MONTH)[0]
            session = Mock()
            session.get.return_value = response()
            self.assertEqual(DL.download_one(request, folder, session), "downloaded")
            self.assertEqual((folder / request.output_name).read_bytes(), BINARY)
            self.assertFalse(list(folder.glob("*.part")))
            self.assertEqual(DL.download_one(request, folder, session), "cached")
            self.assertEqual(session.get.call_count, 1)

    def test_nonbinary_response_retries_and_never_promotes(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(DL.time, "sleep"):
            folder = Path(temp)
            request = DL.requests_for(MONTH)[0]
            session = Mock()
            session.get.return_value = response(b"<html>login failed</html>")
            with self.assertRaisesRegex(RuntimeError, "failed after 2 attempts"):
                DL.download_one(request, folder, session, retries=2)
            self.assertEqual(session.get.call_count, 2)
            self.assertFalse(list(folder.iterdir()))

    def test_transient_failure_then_success_and_auth_failure_stops_retry(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(DL.time, "sleep"):
            request = DL.requests_for(MONTH)[0]
            session = Mock()
            session.get.side_effect = [TimeoutError(), response()]
            self.assertEqual(DL.download_one(request, Path(temp), session), "downloaded")
            (Path(temp) / request.output_name).unlink()
            session.get.side_effect = None
            session.get.return_value = response(status=401)
            with self.assertRaises(DL.AuthenticationError):
                DL.download_one(request, Path(temp), session)
            self.assertEqual(session.get.call_count, 3)

    def test_wget_fallback_preserves_resume_and_handles_auth_failure(self):
        request = DL.requests_for(MONTH)[0]
        with patch.object(DL.HH, "download_one", return_value="skipped"):
            self.assertEqual(DL.download_wget(request, Path("fixture")), "cached")
        with patch.object(DL.HH, "download_one", side_effect=subprocess.CalledProcessError(6, "wget")):
            with self.assertRaises(DL.AuthenticationError):
                DL.download_wget(request, Path("fixture"))

    def test_existing_prepared_month_must_validate_before_skip(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            args = SimpleNamespace(daily=folder, raw=folder / "raw", state=folder / "state")
            path = DL.prepared_path(folder, MONTH)
            path.write_bytes(BINARY)
            progress = Mock()
            with patch.object(DL, "validate_month") as validate, patch.object(DL, "make_session") as session:
                result = DL.process_month(args, MONTH, progress, DL.threading.Event())
            validate.assert_called_once()
            session.assert_not_called()
            self.assertTrue(result["reused"])
            progress.assert_called_once_with("prepared_cache", 31 * 48)

    def test_actual_48_granule_preparation_is_compatible_with_production(self):
        import numpy as np
        import xarray as xr
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            args = SimpleNamespace(daily=folder / "daily", raw=folder / "raw", state=folder / "state")
            day = {"start": "2022-05-01", "end": "2022-05-01"}
            raw = args.raw / "2022"
            raw.mkdir(parents=True)
            dataset = xr.Dataset({
                "precipitation": (("lat", "lon"), np.ones((64, 64), np.float32), {"units": "mm/hr"}),
                "randomError": (("lat", "lon"), np.full((64, 64), 2, np.float32), {"units": "mm/hr"}),
            }, coords={"lat": 20.3 + .1 * (np.arange(64) + .5), "lon": 87.6 + .1 * (np.arange(64) + .5)})
            for request in DL.requests_for(day):
                dataset.to_netcdf(raw / request.output_name)
            with patch.object(DL, "make_session") as session:
                result = DL.process_month(args, day, Mock(), DL.threading.Event())
            session.assert_not_called()
            path = DL.prepared_path(args.daily, day)
            self.assertEqual(result["file"], str(path))
            DL.PROD.validate_imerg(path, day["start"], day["end"])
            with xr.open_dataset(path) as output:
                np.testing.assert_allclose(output.precipitation, 24)
                np.testing.assert_allclose(output.randomError, np.sqrt(48), rtol=1e-6)
                np.testing.assert_array_equal(output.precipitation_cnt, 48)
                self.assertEqual(str(output.time.values[0])[:19], "2022-05-01T03:00:00")
            self.assertTrue(path.with_name(path.stem + "_qc.json").is_file())
            self.assertFalse(list(args.daily.glob("*.pending*")))

    def test_exclusive_archive_lock_rejects_second_coordinator(self):
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(raw=Path(temp))
            with (args.raw / ".production_download.lock").open("a") as lock:
                DL.fcntl.flock(lock, DL.fcntl.LOCK_EX | DL.fcntl.LOCK_NB)
                with patch.object(DL, "run_locked") as run:
                    with self.assertRaisesRegex(RuntimeError, "already owns"):
                        DL.run(args, [MONTH])
                run.assert_not_called()

    def test_year_locks_allow_disjoint_jobs_and_reject_overlapping_years(self):
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(raw=Path(temp))
            folder = args.raw / "2022"
            folder.mkdir()
            with (folder / ".production_download.lock").open("a") as lock:
                DL.fcntl.flock(lock, DL.fcntl.LOCK_EX | DL.fcntl.LOCK_NB)
                with patch.object(DL, "run_locked") as run:
                    DL.run(args, DL.PROD.monthly(2021, 2021))
                    self.assertEqual(run.call_count, 1)
                    with self.assertRaisesRegex(RuntimeError, "already owns"):
                        DL.run(args, DL.PROD.monthly(2022, 2022))

    def test_year_collection_requires_matching_receipts_and_revalidates_files(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            args = SimpleNamespace(state=folder / "state", daily=folder / "daily")
            months = DL.PROD.monthly(2022, 2022)
            report = {"status": "ready", "source": "cloud-opendap", "start": "2022-01-01", "end": "2022-12-31",
                      "months": {m["start"][:7]: {"status": "validated", "file": str(DL.prepared_path(args.daily, m))} for m in months}}
            receipt = args.state / "years/2022/IMERG_READY.json"
            DL.write_json(receipt, report)
            with patch.object(DL, "validate_month") as validate:
                DL.collect_years(args, months)
            self.assertEqual(validate.call_count, 12)
            self.assertTrue((args.state / "IMERG_READY.json").is_file())
            del report["months"]["2022-12"]
            DL.write_json(receipt, report)
            with patch.object(DL, "validate_month"):
                with self.assertRaisesRegex(ValueError, "Missing/mismatched"):
                    DL.collect_years(args, months)
            self.assertFalse((args.state / "IMERG_READY.json").exists())

    def test_probe_always_fetches_and_validates_a_fresh_subset(self):
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(state=Path(temp), transport="requests")
            DL.LOCAL.session = Mock()
            try:
                with patch.object(DL, "resolve_cloud_request", return_value=DL.requests_for(MONTH)[0]), \
                     patch.object(DL, "download_one") as download, patch.object(DL, "scientific_command") as validate:
                    DL.probe(args, [MONTH])
                download.assert_called_once()
                self.assertIn("--validate-granule", validate.call_args[0][0])
                self.assertEqual(json.loads((args.state / "probe.json").read_text())["status"], "validated")
                self.assertFalse((args.state / "IMERG_READY.json").exists())
            finally:
                del DL.LOCAL.session

    def test_preparation_failure_preserves_old_file_and_removes_pending(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            args = SimpleNamespace(daily=folder, raw=folder / "raw", state=folder / "state")
            path = DL.prepared_path(folder, MONTH)
            path.write_bytes(b"old file")
            with patch.object(DL, "validate_month", side_effect=subprocess.CalledProcessError(1, "validate")), \
                 patch.object(DL, "requests_for", return_value=[]), \
                 patch.object(DL, "scientific_command", side_effect=subprocess.CalledProcessError(1, "prepare")):
                with self.assertRaises(subprocess.CalledProcessError):
                    DL.process_month(args, MONTH, Mock(), DL.threading.Event())
            self.assertEqual(path.read_bytes(), b"old file")
            self.assertFalse(list(folder.glob("*.pending*")))

    def test_ready_receipt_requires_every_month_success_and_is_invalidated_on_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            args = SimpleNamespace(state=folder / "state", raw=folder / "raw", connections=3)
            def success(args, month, progress, stop):
                progress("prepared_cache", DL.month_count(month))
                return {"status": "validated", "reused": True, "file": "fixture.nc"}
            with patch.object(DL, "process_month", side_effect=success):
                DL.run(args, [MONTH])
            receipt = args.state / "IMERG_READY.json"
            self.assertEqual(json.loads(receipt.read_text())["status"], "ready")
            with patch.object(DL, "process_month", side_effect=RuntimeError("missing granule")):
                with self.assertRaisesRegex(RuntimeError, "months failed"):
                    DL.run(args, [MONTH])
            self.assertFalse(receipt.exists())
            self.assertEqual(json.loads((args.state / "status.json").read_text())["status"], "failed")

    def test_auth_failure_stops_other_work_and_never_issues_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            args = SimpleNamespace(state=folder / "state", raw=folder / "raw", connections=1)
            def failure(args, month, progress, stop):
                if stop.is_set():
                    raise RuntimeError("stopped")
                raise DL.AuthenticationError("Earthdata denied access")
            with patch.object(DL, "process_month", side_effect=failure):
                with self.assertRaises(RuntimeError):
                    DL.run(args, [MONTH])
            self.assertFalse((args.state / "IMERG_READY.json").exists())

    def test_three_workers_bound_global_concurrency(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            args = SimpleNamespace(state=folder / "state", raw=folder / "raw", connections=3)
            mutex = DL.threading.Lock()
            active = 0
            peak = 0
            def work(args, month, progress, stop):
                nonlocal active, peak
                with mutex:
                    active += 1
                    peak = max(peak, active)
                DL.time.sleep(.02)
                progress("prepared_cache", DL.month_count(month))
                with mutex:
                    active -= 1
                return {"status": "validated", "file": "fixture.nc"}
            with patch.object(DL, "process_month", side_effect=work):
                DL.run(args, DL.PROD.monthly(2022, 2022))
            self.assertEqual(peak, 3)

    def test_slurm_one_job_no_array_and_rejects_excess_connections(self):
        script = ROOT / "slurm/submit_imerg_production_download.sh"
        result = subprocess.run(["bash", str(script), "--dry-run", "--account=test"], cwd=ROOT,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("sbatch"), 2)  # command and .sbatch suffix
        self.assertNotIn("--array", result.stdout)
        self.assertIn("--connections 3", result.stdout)
        self.assertIn("--start-year 2001 --end-year 2024", result.stdout)
        result = subprocess.run(["bash", str(script), "--dry-run"], cwd=ROOT, capture_output=True, text=True,
                                env={**os.environ, "IMERG_DOWNLOAD_CONNECTIONS": "4"})
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("sbatch", result.stdout)

    def test_year_array_is_probe_gated_capped_and_followed_by_audit(self):
        result = subprocess.run(["bash", str(ROOT / "slurm/submit_imerg_production_download.sh"),
                                 "--parallel-years", "--dry-run"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(result.stdout.strip().splitlines()), 3)
        self.assertIn("--probe-only", result.stdout)
        self.assertIn("--dependency=afterok:DRY_PROBE --array=0-23%3", result.stdout)
        self.assertIn("--year-array-worker 2001 2024", result.stdout)
        self.assertIn("--dependency=afterok:DRY_ARRAY", result.stdout)
        self.assertIn("--collect-years", result.stdout)

    def test_year_array_worker_forwards_exact_year_and_single_connection(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / ".netrc").write_text("# placeholder, no credentials used by this test\n")
            fake = folder / "python"
            fake.write_text('#!/bin/bash\nif [[ "$1" == "-c" ]]; then exit 0; fi\nprintf "ARG:%s\\n" "$@"\n')
            fake.chmod(0o755)
            env = {**os.environ, "HOME": temp, "IMERG_PYTHON": str(fake),
                   "SLURM_SUBMIT_DIR": str(ROOT), "SLURM_JOB_ID": "123", "SLURM_ARRAY_TASK_ID": "21"}
            result = subprocess.run(["bash", str(ROOT / "slurm/imerg_production_download.sbatch"),
                                     "--year-array-worker", "2001", "2024", "fixture-state",
                                     "--connections", "3"], cwd=ROOT, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("ARG:--start-year\nARG:2022", result.stdout)
            self.assertIn("ARG:--end-year\nARG:2022", result.stdout)
            self.assertIn("ARG:--connections\nARG:1", result.stdout)
            self.assertIn("ARG:fixture-state/years/2022", result.stdout)

    def test_failed_scheduler_submission_is_not_reported_as_success(self):
        with tempfile.TemporaryDirectory() as temp:
            fake = Path(temp) / "sbatch"
            fake.write_text("#!/bin/sh\nexit 17\n")
            fake.chmod(0o755)
            result = subprocess.run(["bash", str(ROOT / "slurm/submit_imerg_production_download.sh")], cwd=ROOT,
                                    capture_output=True, text=True,
                                    env={**os.environ, "PATH": temp + os.pathsep + os.environ["PATH"]})
            self.assertEqual(result.returncode, 17)
            self.assertNotIn("Download job:", result.stdout)


if __name__ == "__main__":
    unittest.main()
