"""Production calendar, prerequisite gates and scheduler dependencies."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_surma_production',ROOT/'scripts/100_surma_production.py')
PROD=importlib.util.module_from_spec(spec);spec.loader.exec_module(PROD)


class ProductionTests(unittest.TestCase):
    def test_calendar_covers_all_days_including_leap_days(self):
        blocks=PROD.periods(2001,2024)
        self.assertEqual(len(blocks),96)
        days=[d for p in blocks for d in PROD.dates(p['start'],p['end'])]
        self.assertEqual(len(days),8766)
        self.assertEqual(len(set(days)),8766)
        self.assertEqual(days,PROD.dates('2001-01-01','2024-12-31'))
        self.assertIn('2024-02-29',days)
        self.assertEqual(len(PROD.monthly(2001,2024)),288)

    def test_predictor_planner_requires_previous_year_and_cpc_channels(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            meta={'schema_version':1,'complete':True,'completed_years':list(range(2000,2025)),
                  'cond_channels':PROD.CHANNELS}
            (folder/'.zattrs').write_text(json.dumps(meta))
            self.assertTrue(PROD.predictor_metadata_ready(folder,2001,2024))
            meta['completed_years'].remove(2000)
            (folder/'.zattrs').write_text(json.dumps(meta))
            self.assertFalse(PROD.predictor_metadata_ready(folder,2001,2024))
            meta['completed_years'].append(2000);meta['cond_channels']=PROD.CHANNELS[1:]
            (folder/'.zattrs').write_text(json.dumps(meta))
            self.assertFalse(PROD.predictor_metadata_ready(folder,2001,2024))

    def test_deep_predictor_audit_rejects_missing_previous_day(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            meta={'schema_version':1,'complete':True,'completed_years':[2000,2001],'cond_channels':PROD.CHANNELS}
            (folder/'.zattrs').write_text(json.dumps(meta))
            fake={'time':np.asarray(PROD.dates('2001-01-01','2001-12-31'),dtype='datetime64[ns]')}
            fake=type('Store',(dict,),{'attrs':meta})(fake)
            with patch.dict(sys.modules,{'zarr':SimpleNamespace(open_group=lambda *a,**k:fake)}):
                with self.assertRaisesRegex(ValueError,'previous-day boundary'):
                    PROD.validate_predictors(folder,2001,2001)

    def test_deep_predictor_audit_checks_every_day_and_float32_grid(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            meta={'schema_version':1,'complete':True,'completed_years':[2000,2001],'cond_channels':PROD.CHANNELS}
            (folder/'.zattrs').write_text(json.dumps(meta))
            class Conditions:
                def __init__(self): self.calls=[]
                def __getitem__(self,index):
                    self.calls.append(index)
                    return np.ones((7,256,256),np.float32)
            conditions=Conditions()
            fake=type('Store',(dict,),{'attrs':meta})({
                'time':np.asarray(PROD.dates('2000-12-31','2001-12-31'),dtype='datetime64[ns]'),
                'valid':np.ones((256,256),np.uint8), 'static':np.zeros((7,256,256),np.float32),
                'lat':(16+.05*(np.arange(256)+.5)).astype(np.float32),
                'lon':(84+.05*(np.arange(256)+.5)).astype(np.float32),'cond':conditions})
            fake['target']=np.broadcast_to(np.float32(0),(366,256,256))
            with patch.dict(sys.modules,{'zarr':SimpleNamespace(open_group=lambda *a,**k:fake)}):
                result=PROD.validate_predictors(folder,2001,2001)
            self.assertEqual(result['days'],366)
            self.assertEqual(len(conditions.calls),366)

    def test_month_validation_rejects_wrong_window_and_incomplete_dates(self):
        class Array:
            def __init__(self,value,units=None):
                self.values=np.asarray(value); self.attrs={'units':units} if units else {}
        class Dataset(dict):
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def __getattr__(self,key):return self[key]
        ds=Dataset({
            'time':Array(np.asarray(PROD.dates('2001-01-01','2001-01-31'),dtype='datetime64[D]')),
            'lat':Array(20.3+.1*(np.arange(64)+.5)), 'lon':Array(87.6+.1*(np.arange(64)+.5)),
            'precipitation':Array(np.ones((31,64,64)),'mm/day'),
            'randomError':Array(np.ones((31,64,64)),'mm/day'),
            'precipitation_cnt':Array(np.full((31,64,64),48))})
        ds.attrs={'source_frequency':'half-hourly','bmd_accumulation_end_hour_utc':3}
        ds.sizes={'lat':64,'lon':64}
        with patch.dict(sys.modules,{'xarray':SimpleNamespace(open_dataset=lambda p:ds)}):
            PROD.validate_imerg(Path('fixture.nc'),'2001-01-01','2001-01-31')
            ds.attrs['bmd_accumulation_end_hour_utc']=0
            with self.assertRaisesRegex(ValueError,'ends its accumulation'):
                PROD.validate_imerg(Path('fixture.nc'),'2001-01-01','2001-01-31')
            ds.attrs['bmd_accumulation_end_hour_utc']=3
            ds['time'].values=ds['time'].values[:-1]
            with self.assertRaisesRegex(ValueError,'dates incomplete'):
                PROD.validate_imerg(Path('fixture.nc'),'2001-01-01','2001-01-31')

    def test_reusing_predictors_does_not_redownload_annual_sources(self):
        args=SimpleNamespace(start_year=2001,end_year=2024,task=0)
        with patch.object(PROD,'choose_data',return_value=Path('fixture')), \
             patch.object(PROD,'predictor_metadata_ready',return_value=True), \
             patch.object(PROD,'run') as run:
            PROD.download_year(args)
            run.assert_not_called()

    def test_existing_month_must_validate_before_skip(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);path=folder/'imerg_bd_aligned_20010101_20010131.nc';path.write_bytes(b'invalid fixture')
            args=SimpleNamespace(start_year=2001,end_year=2001,task=0,imerg_daily=folder)
            with patch.object(PROD,'validate_imerg',side_effect=ValueError('bad window')),patch.object(PROD,'run') as run:
                with self.assertRaisesRegex(ValueError,'bad window'): PROD.download_month(args)
                run.assert_not_called()

    def test_full_submission_has_download_and_gpu_dependencies(self):
        result=subprocess.run(['bash',str(ROOT/'slurm/submit_surma_production_2001_2024.sh'),'--dry-run'],
                              capture_output=True,text=True,cwd=ROOT)
        self.assertEqual(result.returncode,0,result.stderr)
        commands=result.stderr
        self.assertEqual(commands.count('sbatch --parsable'),7)
        self.assertIn('--array=0-287%2',commands)
        self.assertEqual(commands.count('--array=0-95%2'),2)
        self.assertIn('--dependency=afterok:DRY_JOB:DRY_JOB',commands)
        self.assertIn(' finalize',commands)
        self.assertNotIn('unbound variable',commands)

    def test_audit_does_not_schedule_downloads_or_gpu_jobs(self):
        result=subprocess.run(['bash',str(ROOT/'slurm/submit_surma_production_2001_2024.sh'),'--dry-run','--audit-only'],
                              capture_output=True,text=True,cwd=ROOT)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(result.stderr.count('sbatch --parsable'),1)
        self.assertIn(' audit',result.stderr)
        self.assertNotIn('--array',result.stderr)

    def test_invalid_concurrency_or_years_fail_before_submission(self):
        for override in ({'SURMA_PROD_CONCURRENCY':'0'},{'SURMA_PROD_START_YEAR':'2000'}):
            result=subprocess.run(['bash',str(ROOT/'slurm/submit_surma_production_2001_2024.sh'),'--dry-run'],
                                  capture_output=True,text=True,cwd=ROOT,env={**os.environ,**override})
            self.assertNotEqual(result.returncode,0)
            self.assertNotIn('sbatch --parsable',result.stderr)

    def test_failed_scheduler_submission_stops_chain(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);fake=folder/'sbatch';fake.write_text('#!/bin/sh\nexit 17\n');fake.chmod(0o755)
            result=subprocess.run(['bash',str(ROOT/'slurm/submit_surma_production_2001_2024.sh'),'--audit-only'],
                                  capture_output=True,text=True,cwd=ROOT,env={**os.environ,'PATH':str(folder)+':'+os.environ['PATH']})
            self.assertEqual(result.returncode,17)
            self.assertNotIn('Input audit:',result.stdout)


if __name__=='__main__':unittest.main()
