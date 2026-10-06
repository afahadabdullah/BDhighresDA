"""Completion accounting must exclude failed, duplicate and unpersisted work."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_production_status',ROOT/'scripts/104_surma_production_status.py')
STATUS=importlib.util.module_from_spec(spec);spec.loader.exec_module(STATUS)


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data))


def completed(root,p):
    period={k:p[k] for k in ('label','start','end')}
    fields=root/'gridded'/f"{p['label']}.zarr"
    scope={'start':p['start'],'end':p['end'],'members':30}
    record={'status':'complete','period':period,'days':p['days'],'field_store':str(fields)}
    write(root/'validated'/f"{p['label']}.json",record)
    write(fields/'.zattrs',{'complete':True,'schema':'bdhires.physical_ensemble.v1','scope':scope})
    write(root/'production_metadata'/f"{p['label']}.json",{'scope':scope})
    (root/'production_metadata'/f"{p['label']}.npz").write_bytes(b'fixture archive')
    return record


class StatusTests(unittest.TestCase):
    def test_calendar_and_script_need_no_scientific_environment_or_writes(self):
        periods=STATUS.quarters(2001,2024)
        self.assertEqual(len(periods),96)
        self.assertEqual(sum(p['days'] for p in periods),8766)
        self.assertEqual(periods[12]['days'],91)  # leap-year Q1
        with tempfile.TemporaryDirectory() as temp:
            result=subprocess.run([sys.executable,'-S',str(ROOT/'scripts/104_surma_production_status.py'),'--json'],
                                  cwd=temp,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            s=json.loads(result.stdout)['summary']
            self.assertEqual(s['validated_days'],0)
            self.assertEqual(s['states'],{'UNKNOWN':96})
            self.assertEqual(list(Path(temp).iterdir()),[])

    def test_complete_receipt_requires_matching_artifacts_and_period(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);p=STATUS.quarters(2004,2004)[0]
            completed(root,p)
            self.assertTrue(STATUS.recorded_completion(root,p))
            arrays=root/'production_metadata'/f"{p['label']}.npz"
            arrays.unlink()
            self.assertFalse(STATUS.recorded_completion(root,p))
            completed(root,p)
            write(root/'gridded'/f"{p['label']}.zarr/.zattrs",{'complete':False})
            self.assertFalse(STATUS.recorded_completion(root,p))
            receipt=completed(root,p);receipt['period']['start']='2004-01-02'
            write(root/'validated'/f"{p['label']}.json",receipt)
            self.assertFalse(STATUS.recorded_completion(root,p))

    def test_failed_logged_days_and_slurm_completion_without_receipt_do_not_count(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);logs=root/'logs';logs.mkdir()
            p=STATUS.quarters(2001,2001)
            completed(root,p[0])
            jobs={
                '100_0':{'job':'100_0','state':'RUNNING'}, # already validated; no double count
                '100_1':{'job':'100_1','state':'FAILED'},
                '101_1':{'job':'101_1','state':'RUNNING'}, # newer retry
                '100_2':{'job':'100_2','state':'FAILED'},
                '100_3':{'job':'100_3','state':'COMPLETED'}}
            lines={'100_0':'[sweep] day 90/90 2001-03-31 (45 s/day)',
                   '100_1':'[sweep] day 20/91 2001-04-20 (45 s/day)',
                   '101_1':'[sweep] day 3/91 2001-04-03 (46 s/day)',
                   '100_2':'[sweep] day 40/92 2001-08-09 (45 s/day)',
                   '100_3':'[sweep] day 92/92 2001-12-31 (45 s/day)'}
            for job,line in lines.items(): (logs/f'surma-prod-01-24-{job}.out').write_text(line)
            with patch.object(STATUS,'slurm_records',return_value=(jobs,[])):
                report=STATUS.snapshot(root,2001,2001,logs,['100','101'])
            s=report['summary']
            self.assertEqual(s['validated_quarters'],1)
            self.assertEqual(s['validated_days'],90)
            self.assertEqual(s['active_logged_days'],3)
            self.assertEqual(s['known_sampling_days'],93)
            self.assertEqual(s['validated_analysis_member_days'],2700)
            self.assertEqual(report['quarters'][1]['job'],'101_1')
            self.assertTrue(any('Slurm COMPLETED' in note for note in report['notes']))

    def test_stale_log_period_and_malformed_receipt_are_ignored(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);p=STATUS.quarters(2001,2001)[0]
            log=root/'log';log.write_text('[sweep] day 10/90 2004-01-10 (45 s/day)')
            self.assertEqual(STATUS.log_progress(log,p)['days'],0)
            (root/'validated').mkdir();(root/'validated/2001_q1.json').write_text('{unfinished')
            self.assertFalse(STATUS.recorded_completion(root,p))

    def test_live_queue_overrides_accounting_and_missing_slurm_is_reported(self):
        responses=[SimpleNamespace(returncode=0,stdout='100_0|PENDING|0:0\n100_1|FAILED|1:0\n',stderr=''),
                   SimpleNamespace(returncode=0,stdout='100_0|RUNNING|None\n100_2|PENDING|JobArrayTaskLimit\n',stderr='')]
        with patch.object(STATUS.subprocess,'run',side_effect=responses):
            jobs,notes=STATUS.slurm_records(['100'])
        self.assertEqual(jobs['100_0']['state'],'RUNNING')
        self.assertEqual(jobs['100_1']['exit_code'],'1:0')
        self.assertEqual(jobs['100_2']['reason'],'JobArrayTaskLimit')
        self.assertEqual(notes,[])
        with patch.object(STATUS.subprocess,'run',side_effect=FileNotFoundError('Slurm unavailable')):
            jobs,notes=STATUS.slurm_records(['100'])
        self.assertEqual(jobs,{})
        self.assertEqual(len(notes),2)

    def test_final_receipt_requires_all_unique_validated_quarters(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);records=[completed(root,p) for p in STATUS.quarters(2001,2001)]
            manifest={'status':'complete','start':'2001-01-01','end':'2001-12-31','days':365,
                      'members':30,'shards':records}
            write(root/'production_manifest.json',manifest)
            with patch.object(STATUS,'slurm_records',return_value=({},[])):
                self.assertTrue(STATUS.snapshot(root,2001,2001,root,[])['summary']['final_receipt_complete'])
                manifest['shards']=[records[0]]*4;write(root/'production_manifest.json',manifest)
                self.assertFalse(STATUS.snapshot(root,2001,2001,root,[])['summary']['final_receipt_complete'])


if __name__=='__main__': unittest.main()
