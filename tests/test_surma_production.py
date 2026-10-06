"""Production calendar, prerequisite gates and scheduler dependencies."""
import importlib.util
import io
from contextlib import redirect_stdout
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
    def test_source_check_distinguishes_variables_empty_files_and_matching_receipts(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            packed=folder/'packed.zarr'; packed.mkdir()
            (packed/'.zattrs').write_text(json.dumps({'schema_version':1,'complete':False,
                'completed_years':[2024], 'cond_channels':['cpc_precip','era5_tcwv']}))
            (packed/'cond').mkdir(); (packed/'cond'/'.zarray').write_text('{}')
            (folder/'precip.2024.nc').write_bytes(b'raw CPC fixture')
            (folder/'era5_daily_2024.nc').touch()  # empty files cannot be credited
            months=PROD.monthly(2024,2024)
            first,second=[folder/f"imerg_bd_aligned_{m['start'].replace('-','')}_{m['end'].replace('-','')}.nc" for m in months[:2]]
            first.write_bytes(b'prepared fixture'); second.write_bytes(b'prepared fixture')
            PROD.write_json(folder/'years/2024/status.json',{'months':{
                '2024-01':{'status':'validated','file':str(first)},
                '2024-02':{'status':'validated','file':str(folder/'different.nc')},
                '2024-03':{'status':'validated','file':str(folder/'absent.nc')}}})
            args=SimpleNamespace(start_year=2024,end_year=2025,root=folder,data_zarr=str(packed),
                static=folder/'static.nc',cpc=folder,era5=folder,chirps=folder,
                imerg_daily=folder,imerg_state=folder)
            output=io.StringIO()
            with patch.object(PROD,'fixed_inputs',return_value={}),redirect_stdout(output):
                PROD.source_check(args)
            rows={line.split()[0]:[cell.lstrip('✓✗') for cell in line.split()[1:]] for line in output.getvalue().splitlines()
                  if line.split() and line.split()[0] in ('2023','2024','2025')}
            self.assertEqual(rows['2024'],['P1','RAW','P1','MISS','MISS','MISS','MISS','MISS','2/12','1/12'])
            self.assertEqual(rows['2025'],['MISS']*8+['0/12','0/12'])
            self.assertEqual(rows['2023'][-2:],['--','--'])
            self.assertIn('not deep validation',output.getvalue())

    def test_source_check_accepts_2025_without_scientific_packages_or_writes(self):
        with tempfile.TemporaryDirectory() as temp:
            command=[sys.executable,'-S',str(ROOT/'scripts/100_surma_production.py'),
                     'source-check','--start-year','2024']
            result=subprocess.run(command,cwd=temp,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('Source years checked: 2024..2025',result.stdout)
            self.assertIn('2025 ',result.stdout)
            self.assertEqual(list(Path(temp).iterdir()),[])
            result=subprocess.run([sys.executable,'-S',str(ROOT/'scripts/100_surma_production.py'),
                'production','--end-year','2025','--task','0'],cwd=temp,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('production years must be within 2001..2024',result.stderr)

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

    def test_predictor_failures_list_all_dates_without_relaxing_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            meta={'schema_version':1,'complete':True,'completed_years':[2000,2001],'cond_channels':PROD.CHANNELS}
            (folder/'.zattrs').write_text(json.dumps(meta))
            calls=[]
            class Conditions:
                def __getitem__(self,index):
                    calls.append(index)
                    values=np.ones((7,256,256),np.float32)
                    # Partial coverage remains legitimate; entire missing days fail.
                    values[1,:128]=0
                    if index in (2,5): values[1]=0
                    if index==8: values[2,0,0]=np.nan
                    return values
            fake=type('Store',(dict,),{'attrs':meta})({
                'time':np.asarray(PROD.dates('2000-12-31','2001-12-31'),dtype='datetime64[ns]'),
                'valid':np.ones((256,256),np.uint8),'static':np.zeros((7,256,256),np.float32),
                'lat':(16+.05*(np.arange(256)+.5)).astype(np.float32),
                'lon':(84+.05*(np.arange(256)+.5)).astype(np.float32),
                'cond':Conditions(),'target':np.broadcast_to(np.float32(0),(366,256,256))})
            with patch.dict(sys.modules,{'zarr':SimpleNamespace(open_group=lambda *a,**k:fake)}):
                with self.assertRaises(PROD.PredictorValidationError) as caught:
                    PROD.validate_predictors(folder,2001,2001)
            result=caught.exception.report
            self.assertEqual(result['cpc_unavailable_dates'],['2001-01-02','2001-01-05'])
            self.assertEqual(result['issues'][-1]['channels'],['era5_tcwv'])
            self.assertEqual(result['status'],'invalid_requires_recovery')
            self.assertEqual(len(calls),366)

    def test_known_cpc_gap_opt_in_preserves_native_mask_and_rejects_other_defects(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            meta={'schema_version':1,'complete':True,'completed_years':[2003,2004],'cond_channels':PROD.CHANNELS}
            (folder/'.zattrs').write_text(json.dumps(meta))
            times=np.asarray(PROD.dates('2003-12-31','2004-12-31'),dtype='datetime64[ns]')
            class Conditions:
                unknown=False; placeholder=0.; bad_era5=False
                def __getitem__(self,index):
                    values=np.ones((7,256,256),np.float32)
                    day=str(times[index].astype('datetime64[D]'))
                    if day=='2004-09-10' or (self.unknown and day=='2004-09-11'):
                        values[0]=self.placeholder;values[1]=0
                        if self.bad_era5: values[2,0,0]=np.nan
                    return values
            conditions=Conditions()
            fake=type('Store',(dict,),{'attrs':meta})({
                'time':times,'valid':np.ones((256,256),np.uint8),'static':np.zeros((7,256,256),np.float32),
                'lat':(16+.05*(np.arange(256)+.5)).astype(np.float32),
                'lon':(84+.05*(np.arange(256)+.5)).astype(np.float32),
                'cond':conditions,'target':np.broadcast_to(np.float32(0),(367,256,256))})
            with patch.dict(sys.modules,{'zarr':SimpleNamespace(open_group=lambda *a,**k:fake)}):
                with self.assertRaises(PROD.PredictorValidationError):
                    PROD.validate_predictors(folder,2004,2004)
                result=PROD.validate_predictors(folder,2004,2004,allow_known_cpc_gaps=True)
                self.assertEqual(result['accepted_cpc_gap_dates'],['2004-09-10'])
                self.assertEqual(result['status'],'validated_all_days_with_known_cpc_gaps')
                conditions.unknown=True
                with self.assertRaises(PROD.PredictorValidationError) as caught:
                    PROD.validate_predictors(folder,2004,2004,True)
                self.assertEqual(caught.exception.report['issues'][0]['date'],'2004-09-11')
                conditions.unknown=False;conditions.placeholder=2.
                with self.assertRaises(PROD.PredictorValidationError):
                    PROD.validate_predictors(folder,2004,2004,True)
                conditions.placeholder=0.;conditions.bad_era5=True
                with self.assertRaises(PROD.PredictorValidationError) as caught:
                    PROD.validate_predictors(folder,2004,2004,True)
                self.assertEqual(caught.exception.report['issues'][0]['channels'],['era5_tcwv'])

    def test_known_gap_reuse_rejected_when_raw_cpc_now_has_coverage(self):
        args=SimpleNamespace(start_year=2004,end_year=2004,cpc='raw',allow_known_cpc_gaps=True)
        result={'status':'validated_all_days_with_known_cpc_gaps','accepted_cpc_gap_dates':['2004-09-10'],
                'cpc_unavailable_dates':['2004-09-10'],'issues':[]}
        with patch.object(PROD,'validate_predictors',return_value=result) as validate, \
             patch.object(PROD,'diagnose_cpc_gaps',return_value=[
                 {'date':'2004-09-10','status':'raw_has_coverage_packed_does_not'}]):
            with self.assertRaisesRegex(PROD.PredictorValidationError,'raw CPC has coverage'):
                PROD.validate_production_predictors(args,'packed')
        validate.assert_called_once_with('packed',2004,2004,True)

    def test_cpc_gap_flags_use_next_reporting_date_and_respect_quarter(self):
        predictor={'accepted_cpc_gap_dates':sorted(PROD.KNOWN_CPC_GAPS),
                   'cpc_gap_policy':'known_source_gaps_native_mask'}
        q3=PROD.cpc_background_qc({'start':'2004-07-01','end':'2004-09-30'},predictor)
        self.assertEqual(q3['missing_cpc_days'],[
            {'background_date':'2004-09-10','reporting_date':'2004-09-11'}])
        q1=PROD.cpc_background_qc({'start':'2007-01-01','end':'2007-03-31'},predictor)
        self.assertEqual(q1['missing_cpc_days'],[
            {'background_date':'2007-02-26','reporting_date':'2007-02-27'}])
        q4=PROD.cpc_background_qc({'start':'2004-10-01','end':'2004-12-31'},predictor)
        self.assertEqual(q4['missing_cpc_days'],[])

    def test_cpu_job_forwards_only_explicit_cpc_gap_opt_in(self):
        with tempfile.TemporaryDirectory() as temp:
            python=Path(temp)/'python';python.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n');python.chmod(0o755)
            for value in ('0','1','bad'):
                env={**os.environ,'SLURM_SUBMIT_DIR':str(ROOT),'PYTHON_BIN':str(python),
                     'SURMA_PROD_ALLOW_KNOWN_CPC_GAPS':value}
                result=subprocess.run(['bash',str(ROOT/'slurm/surma_production_inputs.sbatch'),'audit'],
                                      env=env,cwd=ROOT,capture_output=True,text=True)
                self.assertEqual(result.returncode,2 if value=='bad' else 0,result.stderr)
                self.assertEqual('--allow-known-cpc-gaps' in result.stdout,value=='1')

    def test_failed_deep_audit_saves_inventory_and_predictor_diagnostics(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            args=SimpleNamespace(start_year=2001,end_year=2001,root=folder,report=None,
                era5=folder,cpc=folder,chirps=folder,imerg_daily=folder,deep=True)
            failure={'status':'invalid_requires_recovery','path':'fixture.zarr',
                'cpc_unavailable_dates':['2001-01-02','2001-01-05'],
                'issues':[{'date':day,'reason':'CPC conditioning entirely unavailable'}
                          for day in ('2001-01-02','2001-01-05')]}
            with patch.object(PROD,'choose_data',return_value=folder/'fixture.zarr'), \
                 patch.object(PROD,'predictor_metadata_ready',return_value=True), \
                 patch.object(PROD,'fixed_inputs',return_value={}), \
                 patch.object(PROD,'diagnose_cpc_gaps',return_value=[{'status':'raw_source_also_unavailable'}]), \
                 patch.object(PROD,'validate_predictors',side_effect=PROD.PredictorValidationError(failure)), \
                 redirect_stdout(io.StringIO()):
                with self.assertRaises(PROD.PredictorValidationError): PROD.audit(args)
            report=json.loads((folder/'input_inventory.json').read_text())
            self.assertEqual(report['status'],'failed')
            self.assertEqual(report['predictors'],failure)
            self.assertEqual(report['predictors']['cpc_source_diagnostics'],[{'status':'raw_source_also_unavailable'}])
            self.assertEqual(len(report['imerg_months']),12)

    def test_cpc_diagnostic_failure_preserves_original_gate_and_report(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            args=SimpleNamespace(start_year=2001,end_year=2001,root=folder,report=None,
                era5=folder,cpc=folder,chirps=folder,imerg_daily=folder,deep=True)
            failure={'status':'invalid_requires_recovery','cpc_unavailable_dates':['2001-01-02'],
                'issues':[{'date':'2001-01-02','reason':'CPC conditioning entirely unavailable'}]}
            with patch.object(PROD,'choose_data',return_value=folder), \
                 patch.object(PROD,'predictor_metadata_ready',return_value=True), \
                 patch.object(PROD,'fixed_inputs',return_value={}), \
                 patch.object(PROD,'validate_predictors',side_effect=PROD.PredictorValidationError(failure)), \
                 patch.object(PROD,'diagnose_cpc_gaps',side_effect=OSError('raw CPC unreadable')), \
                 redirect_stdout(io.StringIO()):
                with self.assertRaises(PROD.PredictorValidationError): PROD.audit(args)
            report=json.loads((folder/'input_inventory.json').read_text())
            self.assertEqual(report['status'],'failed')
            self.assertIn('CPC conditioning entirely unavailable',report['error'])
            self.assertEqual(report['predictors']['cpc_source_diagnostic_error'],'raw CPC unreadable')

    def test_cpc_source_diagnostics_distinguish_raw_and_packed_gaps(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);(folder/'precip.2001.nc').write_bytes(b'fixture')
            class Field:
                def isel(self,time): self.index=time[0];return self
                def sel(self,**kwargs): return self
            field=Field()
            class Dataset:
                time=SimpleNamespace(values=np.asarray(['2001-01-02','2001-01-05'],dtype='datetime64[ns]'))
                def __enter__(self): return self
                def __exit__(self,*args): pass
                def __getitem__(self,name): return field
            def interpolate(data,*args):
                return None,np.full((1,2,2),data.index,dtype=np.float32)
            packer=SimpleNamespace(_rename_coords=lambda raw:raw,
                interpolate_cpc_condition=interpolate,
                WIDE=SimpleNamespace(lat_min=16,lat_max=28.8,lon_min=84,lon_max=96.8,lat=[],lon=[]))
            fake_spec=SimpleNamespace(loader=SimpleNamespace(exec_module=lambda mod:None))
            modules={'xarray':SimpleNamespace(open_dataset=lambda path:Dataset()),
                     'zarr':SimpleNamespace(open_group=lambda *a,**k:{'valid':np.ones((2,2))})}
            with patch.dict(sys.modules,modules), \
                 patch.object(importlib.util,'spec_from_file_location',return_value=fake_spec), \
                 patch('importlib.util.module_from_spec',return_value=packer):
                rows=PROD.diagnose_cpc_gaps(folder/'packed',folder,
                    ['2001-01-02','2001-01-05','2001-01-06','2002-01-01'])
            self.assertEqual([row['status'] for row in rows],[
                'raw_source_also_unavailable','raw_has_coverage_packed_does_not',
                'raw_source_date_missing_or_duplicated','raw_source_file_missing'])
            self.assertEqual(rows[1]['raw_supported_land_cells'],4)

    def test_deep_fixed_input_failure_also_leaves_report(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            args=SimpleNamespace(start_year=2001,end_year=2001,root=folder,report=None,
                era5=folder,cpc=folder,chirps=folder,imerg_daily=folder,deep=True)
            with patch.object(PROD,'choose_data',return_value=folder), \
                 patch.object(PROD,'predictor_metadata_ready',return_value=False), \
                 patch.object(PROD,'fixed_inputs',side_effect=[{},ValueError('checkpoint channels differ')]), \
                 redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError,'checkpoint channels differ'): PROD.audit(args)
            report=json.loads((folder/'input_inventory.json').read_text())
            self.assertEqual(report['status'],'failed')
            self.assertEqual(report['error'],'checkpoint channels differ')

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
        for override in ({'SURMA_PROD_CONCURRENCY':'0'},{'SURMA_PROD_START_YEAR':'2000'},
                         {'SURMA_PROD_ALLOW_KNOWN_CPC_GAPS':'bad'}):
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
