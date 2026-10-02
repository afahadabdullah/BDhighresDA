"""Evidence identity, new IDW exports and honest remaining-gap inventory."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from bdhires.paper_evidence import checkpoint_metadata


def script(number):
    path=next((ROOT/'scripts').glob(str(number)+'_*.py'))
    spec=importlib.util.spec_from_file_location('_remaining_test_'+str(number),path)
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    return mod


class RemainingEvidenceTests(unittest.TestCase):
    def test_collector_preserves_verified_files_and_rejects_changed_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); source=root/'source'; source.mkdir(); (source/'additional').mkdir()
            data=source/'scores.csv'; data.write_text('n,rmse\n10,2\n')
            audit=script(97); collector=script(98)
            (source/'evidence_manifest.json').write_text(json.dumps({'outputs':[{'path':data.name,'sha256':audit.digest(data)}]}))
            (source/'additional/completion_manifest.json').write_text('{"outputs":[]}')
            out=root/'out'; out.mkdir()
            records=collector.retain_evidence(source,out,audit)
            self.assertEqual((out/'verified_evidence/scores.csv').read_bytes(),data.read_bytes())
            self.assertEqual(len(records),3)
            data.write_text('altered')
            with self.assertRaisesRegex(ValueError,'changed'):
                collector.retain_evidence(source,out,audit)
            with self.assertRaisesRegex(ValueError,'escapes'):
                collector.safe_artifact(source,'../outside.csv')

    def test_baseline_inputs_cannot_include_withheld_gauges(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp); raw=folder/'raw.csv'; dump=folder/'eval.npz'
            raw.write_text('station_id,lat,lon,date,precip_mm\nR,23,90,2021-05-01,10\nW,23.1,90,2021-05-01,12\nOUT,0,0,2021-05-01,8\n')
            np.savez(dump,station_ids=np.array(['W','OUT']),eval_idx=np.array([0,1]),station_lat=np.array([23.1,0.]),station_lon=np.array([90.,0.]))
            helper=script(92); collector=script(98)
            data={'period':np.array(['period']), 'date':np.array(['2021-05-01']),
                  'station':np.array(['W']), 'truth':np.array([12.]),
                  'members':{helper.FINAL:np.ones((1,30)), 'background':np.zeros((1,30))}}
            with patch.object(collector,'points_inside',side_effect=lambda lat,lon,country:lat>20):
                inputs, targets=collector.baseline_period(data,np.array([True]),'period',raw,dump,{},helper)
                self.assertEqual(inputs['station_ids'].tolist(),['R'])
                self.assertEqual(targets['station_ids'].tolist(),['W'])
                self.assertEqual(targets['analysis_members_mm'].shape,(1,30))
                self.assertNotIn('truth_mm',inputs)
                data['truth'][0]=999
                with self.assertRaisesRegex(ValueError,'rainfall differ'):
                    collector.baseline_period(data,np.array([True]),'period',raw,dump,{},helper)

    def test_collector_logs_are_candidates_not_measurements(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp); log=folder/'job.out'
            log.write_text('Starting\n--stats original/stats.json\nelapsed 20 seconds\n')
            records=script(98).log_evidence([log],folder,script(97))
            value=json.loads((folder/records[0]['path']).read_text())
            self.assertEqual([r['line'] for r in value['matches']],[2,3])
            self.assertEqual(value['status'],'candidate_requires_stage_and_run_attribution')

    def test_review_tracks_new_baseline_and_publication_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            rows=script(97).review(Path(temp))
            self.assertEqual({r.get('slot') for r in rows if r['item']>9},{'KR','META'})

    def test_idw_interval_and_intensity_share_daily_means(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);mod=script(92)
            truth=[0.,5.,20.,40.,70.,150.]
            rows=[{'date':f'2021-05-{i+1:02d}','station_id':'BMD_A','period':'2021_may_sep',
                   'truth_mm':v,'idw_mm':v+3,'analysis_mean_mm':v+1} for i,v in enumerate(truth)]
            mod.build_idw_comparisons(rows,folder,resamples=50)
            intervals=mod.read_csv(folder/'paired_idw_intervals.csv')
            self.assertEqual(len(intervals),4)
            for row in intervals:
                self.assertAlmostEqual(float(row['gain_mm_day']),2.)
                self.assertAlmostEqual(float(row['ci_low']),2.)
                self.assertEqual(int(row['n']),6)
            strata=mod.read_csv(folder/'idw_intensity_scores.csv')
            self.assertEqual(len(strata),12)
            self.assertEqual({float(r['rmse_gain_mm_day']) for r in strata},{2.})
            with self.assertRaisesRegex(ValueError,'analysis_mean_mm'):
                mod.build_idw_comparisons([{k:v for k,v in r.items() if k!='analysis_mean_mm'} for r in rows],folder)
            with self.assertRaisesRegex(ValueError,'selection'):
                mod.build_idw_comparisons([{**rows[0],'date':'2022-05-05'}],folder)

    def test_checkpoint_epoch_comes_from_pinned_content(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'best.pt';path.write_bytes(b'synthetic checkpoint identity only')
            expected=hashlib.sha256(path.read_bytes()).hexdigest()
            result=checkpoint_metadata(path,expected,loader=lambda p:{'epoch':49,'step':100,'selected_by':'sampled_crps','weights':'ema'})
            self.assertEqual(result['completed_epoch'],50)
            self.assertEqual(result['epoch_zero_based'],49)
            self.assertEqual(result['sha256'],expected)
            with self.assertRaisesRegex(ValueError,'SHA-256'):
                checkpoint_metadata(path,'wrong',loader=lambda p:{'epoch':39})
            self.assertEqual(checkpoint_metadata(path,expected,loader=lambda p:{})['status'],'epoch_unrecorded')
            with self.assertRaisesRegex(ValueError,'nonnegative'):
                checkpoint_metadata(path,expected,loader=lambda p:{'epoch':-1})

    def test_future_preparation_hash_describes_bytes_actually_read(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'stats.json';content=b'{"precip_transform":{"kind":"fixture-only"}}'
            path.write_bytes(content)
            fake=SimpleNamespace(PrecipTransform=SimpleNamespace(from_dict=lambda d:SimpleNamespace(forward=lambda x:x)))
            with patch.dict(sys.modules,{'bdhires.transforms':fake}):
                forward,record=script(87).load_transform(str(path),with_provenance=True)
            self.assertEqual(record['sha256'],hashlib.sha256(content).hexdigest())
            self.assertEqual(record['precip_transform'],{'kind':'fixture-only'})
            self.assertEqual(record['status'],'recorded_at_preparation')
            self.assertEqual(forward(3),3)

    def test_gridded_input_presence_is_not_generated_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            (folder/'evidence_manifest.json').write_text(json.dumps({'availability':{'BD2_full_gridded_archive':True}}))
            row=next(r for r in script(97).review(folder) if r['item']==4)
            self.assertTrue(row['archive_present_in_hpc_audit'])
            self.assertEqual(row['status'],'gridded_outputs_missing')

    def test_export_hash_mismatch_is_not_completed_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);path=folder/'native_imerg_comparison.csv';path.write_text('changed')
            manifest={'outputs':[{'path':path.name,'sha256':'wrong'}]}
            with self.assertRaisesRegex(ValueError,'differs'):
                script(97).manifest_artifact(folder,manifest,path.name)

    def test_extension_plan_contains_all_missing_calendar_dates(self):
        contract=json.loads((ROOT/'configs/paper1_cpcv2_final.json').read_text())
        plan=script(97).extension_plan(contract)
        self.assertEqual(plan['total_days'],1306)
        wet2024=next(r for r in plan['periods'] if r['year']==2024 and r['season']=='wet_may_sep')
        self.assertEqual((wet2024['start'],wet2024['end'],wet2024['days']),('2024-07-01','2024-09-30',92))
        self.assertEqual(sum(r['days'] for r in plan['periods'] if r['year']==2025),365)
        self.assertTrue(plan['plan_only'])

    def test_selection_discovery_never_substitutes_a_different_arm(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            for name,method,day in [('base','dense_s6_bwdb_r4','2022-05-01'),('superob','dense_s6_bwdb_r4','2022-05-02'),('wrong','another_arm','2022-05-01'),('testyear','dense_s6_bwdb_r4','2023-05-01')]:
                np.savez(folder/(name+'.npz'),times=np.array([day]),**{'station_'+method:np.zeros((1,30,2))})
                (folder/(name+'.json')).write_text(json.dumps({'scope':{'assimilate_all_stations':False}}))
            status=script(97).discover_selection(folder,folder/'candidates.json')
            self.assertEqual(status['found'],2)
            self.assertEqual({p['label'] for p in json.loads((folder/'candidates.json').read_text())['profiles']},{'base','superob'})
            self.assertEqual(status['status'],'candidates_need_script92_matching_validation')


if __name__=='__main__':unittest.main()
