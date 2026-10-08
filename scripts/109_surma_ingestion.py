#!/usr/bin/env python3
"""One bounded ingestion comparison: prepare, submit, resume, and select.

Run `python scripts/109_surma_ingestion.py launch` on PRISM. The frozen CPC-v2
model and reporting-window IMERG are reused. Selection concerns four recipes,
not a globally optimal method or an independently verified production product.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import getpass
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('_ingestion_tail', ROOT/'scripts/107_surma_tail_diagnostics.py')
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
I = T.I
BASE = 'current_025_s6'
METHODS = {
    BASE: dict(cell_deg=.25, spread_cells=6, error_policy='parent_superob'),
    'local_025_s3': dict(cell_deg=.25, spread_cells=3, error_policy='parent_superob'),
    'fine_010_s6': dict(cell_deg=.10, spread_cells=6, error_policy='parent_point_floor'),
    'fine_010_s3': dict(cell_deg=.10, spread_cells=3, error_policy='parent_point_floor'),
}
GROUPS = {
    'local': ('v2_ingestion_local', ['background', 'ingest_s3']),
    'fine': ('v2_ingestion_fine', ['background', 'ingest_s6', 'ingest_s3']),
}
POLICY = dict(development_crps_gain=.02, development_heavy_crps_gain=.05,
    max_rmse_increase=.01, max_network_crps_increase=.02,
    max_dry_bias_increase_mm=.5, max_brier_increase=.002,
    max_coverage_drop=.03, min_heavy_records=20, min_dry_records=30, min_extreme_records=5)
NOTES = [
    'Four predeclared ingestion recipes; checkpoint, dates, holdouts, IMERG, sampler and likelihood weights fixed.',
    'Historical/tune windows select one candidate; test/retest windows only accept or reject that candidate. No audit-driven runner-up.',
    'All dates were previously inspected: this is development/retest evidence, not fresh independent confirmation.',
    'Prior and satellite seeds are matched. Changing gauge tables changes observation perturbations; gauge draws are not fully paired across meshes.',
    'Fine cells use at least the parent point-gauge representativeness; finer averaging does not automatically earn smaller errors.',
    'BWDB error multiplier remains four; finer cells change how many BWDB observations remain unmerged.',
    'Withheld values, station identities and spatial buffers are preserved. No automatic masks or date shifts.',
    'Point-gauge extremes and grid rainfall have different support; interval coverage excludes observation uncertainty.',
    'The thresholds are practical screening rules, not significance tests. No automatic production rerun.',
]


def load(out, strict=False):
    plan = json.loads((Path(out)/'ingestion_plan.json').read_text())
    if plan['methods'] != METHODS or plan['policy'] != POLICY:
        raise ValueError('plan design differs from this script; use its original code')
    I.verify(plan['inputs'])
    if strict: I.verify(plan['code'])
    return plan


def check_station_table(original, table, held, buffer_km, min_coverage, days):
    """Reject changed withheld truth, moving sites, insufficient coverage or leakage."""
    cols = ['station_id', 'date', 'lat', 'lon', 'precip_mm']
    def truth(frame):
        return frame[frame.station_id.isin(held)][cols].sort_values(['station_id', 'date']).reset_index(drop=True)
    pd.testing.assert_frame_equal(truth(original), truth(table), check_dtype=False, atol=1e-8, rtol=0)
    if table.duplicated(['station_id', 'date']).any(): raise ValueError('duplicate station-days')
    if (table.groupby('station_id')[['lat', 'lon']].nunique() > 1).any().any():
        raise ValueError('station coordinates move within a window')
    counts = table.groupby('station_id').precip_mm.count()
    if (counts < min_coverage*days).any():
        raise ValueError('a recipe would silently drop low-coverage stations')
    sites = table.groupby('station_id')[['lat', 'lon']].first().sort_index()
    distance = I.distance(sites)
    mask = sites.index.isin(held)
    if set(sites.index[mask]) != held or not (~mask).any(): raise ValueError('station split changed')
    if distance[np.ix_(mask, ~mask)].min() + 1e-8 < buffer_km:
        raise ValueError('recipe violates withheld buffer')


def prepare(args):
    out = Path(args.out_dir).resolve(); parent = Path(args.pilot).resolve()
    if (out/'ingestion_plan.json').exists():
        plan = load(out, strict=True)
        if plan['parent'] != str(parent): raise ValueError('existing plan uses a different parent')
        print('[reuse] frozen ingestion plan'); return
    if out.exists() and any(out.iterdir()):
        raise ValueError('partial preparation found; inspect it and choose a fresh --out-dir')
    plan, changed = T.load_archive(parent)
    if len(plan['windows']) != 8 or I.BASELINE not in plan['variants']:
        raise ValueError('use the completed eight-window surma_tail_pilot, not the selected-event pilot')
    held = T.held_ids(plan)
    for record in plan['shared_files']:
        if Path(record['path']).name in {'sampler.py', 'guidance.py', 'observation.py'}:
            I.verify([record])  # Never reuse a baseline after a core DA change.
    # Read all baseline receipts before spending time preparing candidates.
    receipts = []
    for w in plan['windows']:
        T.window_arrays(parent, plan, w)
        report = json.loads((parent/'runs'/w['label']/'sweep.json').read_text())['scope']
        if report['background_day_offset'] != -1 or report['seed'] != plan['seed'] or report['assimilate_all_stations']:
            raise ValueError('baseline timing, random seed or holdout differs')
        receipts += [I.record(parent/'runs'/w['label']/f) for f in ('complete.json', 'sweep.json', 'sweep.npz')]
    roles = {w['role'] for w in plan['windows']}
    if not roles & {'tune', 'historical'} or not roles & {'test', 'retest'}:
        raise ValueError('both development and retest windows required')
    out.mkdir(parents=True, exist_ok=True)
    code = [I.record(ROOT/p) for p in ('scripts/109_surma_ingestion.py', 'scripts/107_surma_tail_diagnostics.py',
        'scripts/106_surma_improvement.py', 'scripts/87_superob_dense_gauges.py',
        'scripts/28_simultaneous_method_sweep.py', 'src/bdhires/da/sampler.py',
        'src/bdhires/da/guidance.py', 'src/bdhires/da/observation.py')]
    tasks, inputs = [], [I.record(parent/'pilot_plan.json'), *receipts]
    for kind, (group, variants) in GROUPS.items():
        folder = out/kind; folder.mkdir()
        cases = []
        for index, previous in enumerate(plan['windows']):
            w = dict(previous)
            if kind == 'fine':
                dest = folder/'prepared'/w['label']; dest.mkdir(parents=True)
                original = I.read_gauges([T.original_path(w)])
                table, budget = dest/'superob.csv', dest/'superob.json'
                # The existing script's geometry protections do not require Torch.
                I.run_script('87_superob_dense_gauges.py', '--stations', T.original_path(w),
                    '--holdout-ids', plan['holdout'], '--cell-deg', .10,
                    '--protect-withheld-km', plan['buffer_km'], '--fail-under-km', plan['buffer_km'],
                    '--out', table, '--report', budget)
                fine = I.read_gauges([table])
                check_station_table(original, fine, held, plan['buffer_km'], plan['min_coverage'], w['days'])
                old_budget = next(Path(r['path']) for r in previous['files'] if Path(r['path']).name == 'superob.json')
                point_error = json.loads(old_budget.read_text())['recommended_representativeness']['implied_representativeness']
                if point_error is None or not np.isfinite(point_error): raise ValueError('missing parent point-error budget')
                w.update(stations=str(table), representation=max(float(point_error), float(previous['representation'])),
                    files=[*previous['files'], I.record(table), I.record(budget)])
            cases.append(w)
            tasks.append(dict(kind=kind, task=index, label=w['label']))
        subplan = {**plan, 'group': group, 'variants': variants, 'windows': cases,
            'parent_pilot': str(parent), 'shared_files': [*plan['shared_files'], *code], 'notes': NOTES}
        # Archive code changes are allowed only for reading the parent. New jobs
        # pin the current code; immutable non-code source records are retained.
        subplan['shared_files'] = [r for r in plan['shared_files'] if Path(r['path']).suffix != '.py'] + code
        I.write_json(folder/'pilot_plan.json', subplan)
        inputs.append(I.record(folder/'pilot_plan.json'))
    I.write_json(out/'ingestion_plan.json', dict(parent=str(parent), methods=METHODS, policy=POLICY,
        tasks=tasks, inputs=inputs, code=code, notes=NOTES, parent_code_changes_read_only=changed))
    print(f'[prepared] {len(tasks)} GPU tasks; three new recipes; completed baseline reused; {out}')


def run(args):
    out = Path(args.out_dir); plan = load(out, strict=True)
    if args.task is None or not 0 <= args.task < len(plan['tasks']): raise ValueError('valid --task required')
    task = plan['tasks'][args.task]; group, variants = GROUPS[task['kind']]
    run_folder = out/task['kind']/'runs'/task['label']
    if run_folder.exists() and not (run_folder/'complete.json').exists():
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
        archived = run_folder.with_name(run_folder.name+'.partial-'+stamp)
        run_folder.rename(archived)
        print(f'[resume] preserved incomplete attempt: {archived}', flush=True)
    T.run(SimpleNamespace(out_dir=str(out/task['kind']), task=task['task']), group=group, variants=variants)


def score_rows(members, observed, ids, label, split, method):
    rows = []
    for network in ('ALL', 'BMD', 'BWDB'):
        net = np.ones(observed.shape, bool) if network == 'ALL' else np.broadcast_to(np.char.startswith(ids, network+'_'), observed.shape)
        for name, low, high in [('all', 0, np.inf), ('dry', 0, 1), ('heavy', 50, np.inf), ('extreme', 100, np.inf)]:
            mask = net & np.isfinite(observed) & (observed >= low) & (observed < high)
            values = np.moveaxis(members, 1, 0)[:, mask]
            rows.append(dict(window=label, split=split, method=method, network=network, bin=name,
                **I.score_members(values, observed[mask])))
    return rows


def gate(table, method, split):
    d = table[(table.window == 'POOLED') & (table.split == split)]
    def get(m, network='ALL', bin='all'):
        rows = d[(d.method == m) & (d.network == network) & (d.bin == bin)]
        if len(rows) != 1: raise ValueError('missing or duplicate score row')
        return rows.iloc[0]
    base, candidate = get(BASE), get(method)
    reasons = []
    checks = [('all CRPS', candidate.fair_crps_mm <= base.fair_crps_mm*(1-POLICY['development_crps_gain'] if split == 'development' else 1)),
        ('RMSE', candidate.rmse_mm <= base.rmse_mm*(1+POLICY['max_rmse_increase'])),
        ('50mm Brier', candidate.rain50_brier <= base.rain50_brier+POLICY['max_brier_increase']),
        ('field coverage', candidate.interval90_coverage >= base.interval90_coverage-POLICY['max_coverage_drop'])]
    for bin, minimum in [('dry', POLICY['min_dry_records']), ('heavy', POLICY['min_heavy_records'])]:
        b, c = get(BASE, bin=bin), get(method, bin=bin)
        if min(b.n, c.n) < minimum:
            reasons.append(f'insufficient {bin} records'); continue
        if bin == 'dry': checks.append(('dry bias', abs(c.bias_mm) <= abs(b.bias_mm)+POLICY['max_dry_bias_increase_mm']))
        else:
            checks.append(('heavy CRPS', c.fair_crps_mm <= b.fair_crps_mm*(1-POLICY['development_heavy_crps_gain'] if split == 'development' else 1)))
            checks.append(('heavy bias', abs(c.bias_mm) <= abs(b.bias_mm)))
    b, c = get(BASE, bin='extreme'), get(method, bin='extreme')
    if min(b.n, c.n) < POLICY['min_extreme_records']:
        reasons.append('insufficient >=100mm records')
    else:
        checks += [('extreme CRPS', c.fair_crps_mm <= b.fair_crps_mm),
                   ('extreme bias', abs(c.bias_mm) <= abs(b.bias_mm))]
    for network in ('BMD', 'BWDB'):
        b, c = get(BASE, network), get(method, network)
        checks.append((network+' CRPS', min(b.n, c.n) >= 20 and c.fair_crps_mm <= b.fair_crps_mm*(1+POLICY['max_network_crps_increase'])))
    reasons += [name for name, ok in checks if not ok]
    return reasons


def choose(table):
    dev = {m: gate(table, m, 'development') for m in METHODS if m != BASE}
    eligible = [m for m, failures in dev.items() if not failures]
    if not eligible:
        return dict(selected=BASE, candidate=None, status='retain_current', development_failures=dev, audit_failures=[])
    scores = table[(table.window == 'POOLED') & (table.split == 'development') & (table.network == 'ALL') & (table.bin == 'all')].set_index('method')
    candidate = min(eligible, key=lambda m: (scores.at[m, 'fair_crps_mm'], list(METHODS).index(m)))
    failed = gate(table, candidate, 'retest')
    return dict(selected=BASE if failed else candidate, candidate=candidate,
        status='retain_current' if failed else 'candidate_passed_retest', development_failures=dev, audit_failures=failed)


def summarize(args):
    out = Path(args.out_dir); plan = load(out, strict=True)
    parent = Path(plan['parent']); baseline, _ = T.load_archive(parent)
    subplans = {kind: T.load_archive(out/kind)[0] for kind in GROUPS}
    rows, pools = [], {}
    for index, w in enumerate(baseline['windows']):
        times, ids, observed, old, common = T.window_arrays(parent, baseline, w)
        arrays = {BASE: old[I.BASELINE]}
        for kind in GROUPS:
            other_times, other_ids, other_obs, values, _ = T.window_arrays(out/kind, subplans[kind], subplans[kind]['windows'][index])
            if set(other_ids) != set(ids) or not np.array_equal(times, other_times): raise ValueError('evaluation identities/dates differ')
            order = [list(other_ids).index(s) for s in ids]
            if not np.allclose(observed, other_obs[:, order], equal_nan=True): raise ValueError('withheld truth changed')
            values = {k: v[:, :, order] for k, v in values.items()}
            if not np.allclose(old['background'], values['background'], atol=1e-4, rtol=1e-5, equal_nan=True):
                raise ValueError('background control changed: cached baseline comparison is invalid')
            arrays['local_025_s3' if kind == 'local' else 'fine_010_s3'] = values['ingest_s3']
            if kind == 'fine': arrays['fine_010_s6'] = values['ingest_s6']
        for values in arrays.values():
            if not np.isfinite(values[:, :, np.isfinite(observed).any(axis=0)]).all():
                raise ValueError('non-finite predictions; refusing a changing evaluation sample')
        split = 'development' if w['role'] in ('historical', 'tune') else 'retest'
        for method, values in arrays.items():
            rows += score_rows(values, observed, ids, w['label'], split, method)
            pools.setdefault((split, method), []).append((values, observed, ids))
    for (split, method), entries in pools.items():
        ids = entries[0][2]
        if any(not np.array_equal(ids, e[2]) for e in entries): raise ValueError('withheld order changes across windows')
        rows += score_rows(np.concatenate([e[0] for e in entries]), np.concatenate([e[1] for e in entries]), ids, 'POOLED', split, method)
    table = pd.DataFrame(rows); dest = out/'summary'; dest.mkdir(exist_ok=True)
    table.to_csv(dest/'scores.csv', index=False)
    choice = choose(table)
    chosen_kind = 'fine' if choice['selected'].startswith('fine') else 'local'
    recipe = {**METHODS[choice['selected']], 'imerg': 'existing half-hourly reporting-window S04',
        'background_day_offset': -1, 'checkpoint': baseline['checkpoint'], 'config': baseline['config'],
        'stats': baseline['stats'], 'seed': baseline['seed'], 'members': baseline['members'],
        'base_variant': I.BASELINE, 'gauge_component_spread_cells': METHODS[choice['selected']]['spread_cells'],
        'sampling_group': 'v2_tail_improvement' if choice['selected'] == BASE else GROUPS[chosen_kind][0],
        'sampling_variant': I.BASELINE if choice['selected'] == BASE else 'ingest_s6' if choice['selected'].endswith('s6') else 'ingest_s3',
        'representation_by_window': {w['label']: w['representation'] for w in subplans[chosen_kind]['windows']},
        'production_error_policy': 'For fine cells, use max(parent point-gauge error, parent superob error), estimated from assimilated inputs only; for current cells retain the existing superob error.',
        'production_ready': False, 'reason': 'Development/retest selection; production preparation must use this recipe and validate its error budgets.'}
    I.write_json(dest/'selection.json', {**choice, 'recipe': recipe, 'policy': POLICY, 'notes': NOTES,
        'plan': I.record(out/'ingestion_plan.json'), 'scores': I.record(dest/'scores.csv')})
    selected = table[(table.window == 'POOLED') & (table.network == 'ALL') & (table.bin == 'all')]
    lines = ['# Ingestion method selection', '', f"Decision: **{choice['selected']}** ({choice['status']}).", '',
        '| Split | Method | N | Bias mm | RMSE mm | Fair CRPS mm | Field coverage |', '|---|---|---:|---:|---:|---:|---:|']
    for r in selected.itertuples():
        lines.append(f'| {r.split} | {r.method} | {r.n} | {r.bias_mm:.3f} | {r.rmse_mm:.3f} | {r.fair_crps_mm:.3f} | {r.interval90_coverage:.3f} |')
    lines += ['', 'Development gate failures: '+json.dumps(choice['development_failures']),
        'Chosen candidate retest failures: '+json.dumps(choice['audit_failures']), '', *['- '+n for n in NOTES]]
    (dest/'comparison.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines)); print(f'[selection] {dest}/selection.json')


def submit(args):
    out = Path(args.out_dir).resolve(); plan = load(out, strict=True)
    receipt_path = out/'submission.json'
    if receipt_path.exists():
        previous = json.loads(receipt_path.read_text())
        jobs = [str(previous[k]) for k in ('array', 'summary') if previous.get(k)]
        active = subprocess.run(['squeue', '--noheader', '--user', getpass.getuser(), '--format=%F|%i|%T|%r'], capture_output=True, text=True)
        if active.returncode: raise ValueError('cannot verify previous Slurm jobs: '+active.stderr.strip())
        rows = [line.split('|', 3) for line in active.stdout.splitlines() if line.strip()]
        ours = [r for r in rows if r[0].strip() in jobs or r[1].strip() in jobs]
        if any(r[0].strip() == str(previous['array']) for r in ours):
            print('[already submitted] GPU array is active: '+str(previous['array'])); return
        for r in ours:
            if r[2].strip() == 'PENDING' and r[3].strip() == 'DependencyNeverSatisfied':
                subprocess.run(['scancel', str(previous['summary'])], check=True)
                print('[resume] cancelled obsolete dependent summary '+str(previous['summary']))
            else:
                print('[already submitted] summary is active: '+str(previous['summary'])); return
    pending = []
    for index, task in enumerate(plan['tasks']):
        folder = out/task['kind']; sub, _ = T.load_archive(folder, strict_code=True)
        w = sub['windows'][task['task']]
        if (folder/'runs'/w['label']/'complete.json').exists(): T.window_arrays(folder, sub, w)
        else: pending.append(index)
    if not pending:
        summarize(args); return
    env = os.environ.copy()
    env.update(SURMA_IMPROVE_OUT=str(out), SURMA_IMPROVE_SCRIPT='scripts/109_surma_ingestion.py')
    (ROOT/'logs').mkdir(exist_ok=True)
    def sbatch(*parts):
        job = subprocess.check_output(['sbatch', '--parsable', '--export=ALL', *parts], cwd=ROOT, env=env, text=True).strip().split(';')[0]
        if not job.isdigit(): raise ValueError('unexpected sbatch response: '+job)
        return job
    array = sbatch('--job-name=surma-ingest', '--array='+','.join(map(str, pending))+f'%{args.concurrency}', 'slurm/surma_improvement.sbatch')
    receipt = dict(array=array, summary=None, tasks=pending)
    I.write_json(receipt_path, receipt)
    receipt['summary'] = sbatch('--job-name=surma-ingest-summary', '--dependency=afterok:'+array, 'slurm/surma_improvement_summary.sbatch')
    I.write_json(receipt_path, receipt)
    print(f"GPU array: {array}; pending tasks: {len(pending)}; concurrency: {args.concurrency}\nDependent selection: {receipt['summary']}\nOutput: {out}/summary/comparison.md")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['launch', 'prepare', 'run', 'summarize', 'submit'], nargs='?', default='launch')
    p.add_argument('--pilot', default='data/processed/surma_tail_pilot')
    p.add_argument('--out-dir', default='data/processed/surma_ingestion_selection')
    p.add_argument('--task', type=int)
    p.add_argument('--concurrency', type=int, default=4)
    args = p.parse_args()
    if args.concurrency < 1: p.error('--concurrency must be positive')
    if args.stage == 'launch': prepare(args); submit(args)
    else: globals()[args.stage](args)


if __name__ == '__main__':
    try: main()
    except (ValueError, OSError, KeyError, AssertionError, StopIteration, subprocess.CalledProcessError) as exc:
        print('[ingestion] failed: '+str(exc), file=sys.stderr); sys.exit(1)
