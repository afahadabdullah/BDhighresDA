import ast
from dataclasses import dataclass, replace
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('_ingestion_test', ROOT/'scripts/109_surma_ingestion.py')
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)


def scores():
    rows = []
    for split in ('development', 'retest'):
        for method in M.METHODS:
            for network in ('ALL', 'BMD', 'BWDB'):
                for bin in ('all', 'dry', 'heavy', 'extreme'):
                    rows.append(dict(window='POOLED', split=split, method=method, network=network, bin=bin,
                        n=100, fair_crps_mm=10., rmse_mm=20., bias_mm=2. if bin == 'dry' else -20.,
                        rain50_brier=.05, interval90_coverage=.8))
    return pd.DataFrame(rows)


def improve(table, method, factor=.9):
    mask = table.method == method
    table.loc[mask, 'fair_crps_mm'] *= factor
    table.loc[mask, 'rmse_mm'] *= factor
    table.loc[mask, 'bias_mm'] *= factor


class IngestionTests(unittest.TestCase):
    def test_no_improvement_retains_current(self):
        result = M.choose(scores())
        self.assertEqual(result['selected'], M.BASE)
        self.assertIsNone(result['candidate'])

    def test_chooses_development_winner_then_checks_retest(self):
        table = scores(); improve(table, 'local_025_s3', .85); improve(table, 'fine_010_s3', .9)
        self.assertEqual(M.choose(table)['selected'], 'local_025_s3')
        table.loc[(table.method == 'local_025_s3') & (table.split == 'retest'), 'fair_crps_mm'] = 11.
        result = M.choose(table)
        self.assertEqual(result['candidate'], 'local_025_s3')
        self.assertEqual(result['selected'], M.BASE)  # Never search the audit for a runner-up.
        self.assertTrue(result['audit_failures'])

    def test_dry_false_alarms_network_harm_and_small_tails_block_selection(self):
        for column, value, extra in [('bias_mm', 5., lambda t: t.bin == 'dry'),
                ('fair_crps_mm', 12., lambda t: t.network == 'BMD'),
                ('n', 2, lambda t: t.bin == 'heavy'),
                ('fair_crps_mm', 12., lambda t: t.bin == 'extreme'),
                ('rain50_brier', .1, lambda t: t.bin == 'all')]:
            with self.subTest(column=column):
                table = scores(); improve(table, 'fine_010_s3')
                table.loc[(table.method == 'fine_010_s3') & extra(table), column] = value
                self.assertEqual(M.choose(table)['selected'], M.BASE)

    def test_station_validation_protects_withheld_values_and_buffer(self):
        original = pd.DataFrame(dict(station_id=['BMD_1', 'BWDB_2'],
            date=pd.to_datetime(['2020-01-01']*2), lat=[23., 23.5], lon=[90., 90.], precip_mm=[120., 30.]))
        M.check_station_table(original, original.copy(), {'BMD_1'}, 20., .8, 1)
        changed = original.copy(); changed.loc[0, 'precip_mm'] = 100.
        with self.assertRaises(AssertionError): M.check_station_table(original, changed, {'BMD_1'}, 20., .8, 1)
        changed = original.copy(); changed.loc[1, 'lat'] = 23.01
        with self.assertRaisesRegex(ValueError, 'buffer'): M.check_station_table(original, changed, {'BMD_1'}, 20., .8, 1)

    def test_scores_use_only_withheld_finite_observations(self):
        obs = np.array([[0., 100.], [np.nan, 50.]])
        members = np.stack([obs, obs], axis=1)
        members[1, :, 0] = 90.
        result = pd.DataFrame(M.score_rows(members, obs, np.array(['BMD_1', 'BWDB_2']), 'w', 'development', M.BASE))
        self.assertEqual(result[(result.network == 'ALL') & (result.bin == 'all')].iloc[0]['n'], 3)
        self.assertEqual(result[(result.network == 'ALL') & (result.bin == 'heavy')].iloc[0]['n'], 2)
        self.assertEqual(result[(result.network == 'ALL') & (result.bin == 'all')].iloc[0].rmse_mm, 0.)

    def test_partial_run_is_preserved_before_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); path = root/'local/runs/example'; path.mkdir(parents=True)
            (path/'sweep.npz').write_bytes(b'partial')
            plan = dict(tasks=[dict(kind='local', task=0, label='example')])
            with patch.object(M, 'load', return_value=plan), patch.object(M.T, 'run') as run:
                M.run(SimpleNamespace(out_dir=str(root), task=0))
                run.assert_called_once()
            archived = list(path.parent.glob('example.partial-*'))
            self.assertEqual(len(archived), 1)
            self.assertEqual((archived[0]/'sweep.npz').read_bytes(), b'partial')

    def test_sampler_variants_only_change_names_notes_and_gauge_spread(self):
        tree = ast.parse((ROOT/'scripts/28_simultaneous_method_sweep.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Variant')
        ns = dict(dataclass=dataclass, replace=replace, __name__=__name__)
        exec(compile(ast.Module(body=[cls], type_ignores=[]), 'variants', 'exec'), ns)
        base = ns['Variant'](name=M.I.BASELINE, huber_delta=3., gauge_component_spread_cells=6.,
            gauge_guidance_gamma=.01, imerg_guidance_gamma=.001, secondary_r_multiplier=4.)
        ns.update(_IMPROVEMENT_BASE=base, CORE=[ns['Variant'](name='background', streams='none')])
        for name in ('V2_INGESTION_LOCAL', 'V2_INGESTION_FINE'):
            node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
            exec(compile(ast.Module(body=[node], type_ignores=[]), 'variants', 'exec'), ns)
        self.assertEqual(replace(ns['V2_INGESTION_FINE'][1], name=base.name, note=base.note), base)
        self.assertEqual(replace(ns['V2_INGESTION_LOCAL'][1], name=base.name, note=base.note, gauge_component_spread_cells=6.), base)

    def test_submission_chains_summary_and_does_not_duplicate_active_jobs(self):
        import contextlib
        import io
        import json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(out_dir=str(root), concurrency=4)
            plan = dict(tasks=[dict(kind='local', task=0, label='example')])
            sub = dict(windows=[dict(label='example')])
            with patch.object(M, 'ROOT', root), patch.object(M, 'load', return_value=plan), \
                 patch.object(M.T, 'load_archive', return_value=(sub, [])), \
                 patch.object(M.subprocess, 'check_output', side_effect=['42\n', '43\n']) as sbatch, \
                 patch.object(M.subprocess, 'run') as queue, contextlib.redirect_stdout(io.StringIO()):
                M.submit(args)
                self.assertIn('--array=0%4', sbatch.call_args_list[0].args[0])
                self.assertIn('--dependency=afterok:42', sbatch.call_args_list[1].args[0])
                self.assertEqual(json.loads((root/'submission.json').read_text())['summary'], '43')
                for state in ('42|42_0|RUNNING|None\nN/A|43|PENDING|Dependency', 'N/A|43|RUNNING|None'):
                    queue.return_value = SimpleNamespace(returncode=0, stdout=state, stderr='')
                    M.submit(args)
                self.assertEqual(sbatch.call_count, 2)

    def test_prepare_and_summarize_end_to_end_without_gpu(self):
        # Exercise actual aggregation, hash receipts, sample alignment and selection.
        import contextlib
        import io
        import json
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); parent = root/'parent'; parent.mkdir()
            holdout = parent/'held.txt'; holdout.write_text('BMD_1\nBWDB_1\n')
            shared = []
            for name in ('model.pt', 'stats.json', 'config.yaml'):
                p = parent/name; p.write_text('fixture'); shared.append(M.I.record(p))
            variants = ['background', M.I.BASELINE]
            cases = []
            for index in range(8):
                folder = parent/'prepared'/str(index); folder.mkdir(parents=True)
                dates = pd.date_range(f'20{10+index}-06-01', periods=15)
                original = pd.DataFrame([dict(station_id=sid, date=date, lat=lat, lon=lon, precip_mm=rain)
                    for date in dates for sid, lat, lon, rain in [('BMD_1', 23., 90., 0.),
                        ('BWDB_1', 23.1, 90., 100.), ('BWDB_A', 23.61, 90.41, 60.),
                        ('BWDB_B', 23.62, 90.42, 70.)]])
                original.to_csv(folder/'original_qc.csv', index=False)
                original.to_csv(folder/'superob.csv', index=False)
                (folder/'superob.json').write_text(json.dumps(dict(recommended_representativeness=dict(implied_representativeness=.7))))
                cases.append(dict(label=str(index), role='tune' if index < 4 else 'retest',
                    start=str(dates[0].date()), end=str(dates[-1].date()), days=15, quarter=f'20{10+index}_q2',
                    stations=str(folder/'superob.csv'), imerg='unused', representation=.5,
                    files=[M.I.record(folder/name) for name in ('original_qc.csv', 'superob.csv', 'superob.json')]))
            plan = dict(status='prepared_research_pilot', windows=cases, variants=variants,
                checkpoint=str(parent/'model.pt'), stats=str(parent/'stats.json'), config=str(parent/'config.yaml'),
                predictors='unused', members=3, seed=123, min_coverage=.8, held_stations=2, retained_stations=2,
                buffer_km=20., fill_known_cpc_gaps=False, holdout=str(holdout), shared_files=shared, source_files=[])
            M.I.write_json(parent/'pilot_plan.json', plan)
            def completed(folder, plan, w, new=False):
                dest = folder/'runs'/w['label']; dest.mkdir(parents=True)
                obs = np.tile([0., 100.], (15, 1))
                prior = np.repeat((obs*.3+10)[:, None, :], 3, axis=1)
                base = np.repeat((obs*.5+10)[:, None, :], 3, axis=1)
                good = np.repeat((obs*.9+1)[:, None, :], 3, axis=1)
                arrays = {'station_'+v: prior if v == 'background' else good if new and v == 'ingest_s3' and folder.name == 'local' else base for v in plan['variants']}
                np.savez(dest/'sweep.npz', variant_names=plan['variants'], times=pd.date_range(w['start'], w['end']).strftime('%Y-%m-%d').to_numpy(dtype=str),
                    station_ids=['BMD_1', 'BWDB_1'], eval_idx=np.array([0, 1]), gauge_mm=obs, **arrays)
                M.I.write_json(dest/'sweep.json', dict(scope=dict(background_day_offset=-1, seed=123, assimilate_all_stations=False)))
                M.I.write_json(dest/'complete.json', dict(plan_sha256=M.I.sha(folder/'pilot_plan.json'),
                    files=[M.I.record(dest/p) for p in ('sweep.npz', 'sweep.json')]))
            for w in cases: completed(parent, plan, w)
            out = root/'comparison'; args = SimpleNamespace(out_dir=str(out), pilot=str(parent))
            def run_script(name, *arguments):
                subprocess.run([sys.executable, str(ROOT/'scripts'/name), *map(str, arguments)],
                    check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            with patch.object(M.I, 'run_script', side_effect=run_script), contextlib.redirect_stdout(io.StringIO()):
                M.prepare(args)
                M.prepare(args)  # idempotent; immutable plan reused
            prepared = M.load(out, strict=True)
            self.assertEqual(len(prepared['tasks']), 16)
            for kind in M.GROUPS:
                sub, _ = M.T.load_archive(out/kind, strict_code=True)
                for w in sub['windows']:
                    self.assertEqual(w['representation'], .7 if kind == 'fine' else .5)
                    completed(out/kind, sub, w, new=True)
            with contextlib.redirect_stdout(io.StringIO()): M.summarize(args)
            decision = json.loads((out/'summary/selection.json').read_text())
            self.assertEqual(decision['selected'], 'local_025_s3')
            self.assertFalse(decision['recipe']['production_ready'])
            # Editing original gauge data must invalidate the frozen comparison.
            Path(cases[0]['stations']).write_text('changed')
            with self.assertRaises(ValueError): M.summarize(args)


if __name__ == '__main__': unittest.main()
