#!/usr/bin/env python3
"""Collect Paper 1 evidence for transfer from PRISM without rerunning inference.

Preserves verified existing results, retrieves log/accounting evidence, and can
export retained-only baseline inputs after the original archive passes script
90's contract checks. Missing records stay pending. No download or submission.
The output can contain restricted gauge observations: do not publish it as code.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from bdhires.country import read_boundary, points_inside, DEFAULT_BOUNDARY


def module(number):
    path = next((ROOT/'scripts').glob(str(number)+'_*.py'))
    spec = importlib.util.spec_from_file_location('_collect_'+str(number), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def safe_artifact(folder, relative):
    path = (folder/relative).resolve()
    if not path.is_relative_to(folder.resolve()):
        raise ValueError('artifact escapes source folder: '+relative)
    return path


def retain_evidence(source, out, audit):
    """Copy only manifest-backed tables/metadata, verifying before and after copy."""
    records = []
    for parent, name in ((source, 'evidence_manifest.json'),
                         (source/'additional', 'completion_manifest.json')):
        manifest = audit.load_json(parent/name)
        entries = list(manifest.get('outputs', []))
        if parent == source:
            entries += manifest.get('gridded_result_files', [])
        # Manifests are retained as provenance, not independently authenticated.
        entries.append({'path': name, 'sha256': audit.digest(parent/name)})
        for entry in entries:
            path = safe_artifact(parent, entry['path'])
            if path.suffix not in ('.csv', '.json', '.tex', '.md'):
                continue
            if not path.is_file() or audit.digest(path) != entry['sha256']:
                raise ValueError('missing or changed exported evidence: '+str(path))
            relative = path.relative_to(source.resolve())
            target = out/'verified_evidence'/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            if audit.digest(target) != entry['sha256']:
                raise ValueError('copied evidence hash differs: '+str(target))
            records.append({'source': str(path), 'path': str(target.relative_to(out)),
                            'sha256': entry['sha256']})
    return records


def log_evidence(paths, out, audit):
    """Extract relevant lines with hashes, not invented stage timings/provenance."""
    pattern = re.compile(r'--stats\b|stats_provenance|checkpoint|best\.pt|epoch|wall.?time|elapsed|seconds|GPU|CUDA|SLURM_JOB_ID|Submitted batch job', re.I)
    records = []
    for index, path in enumerate(paths):
        matches = []
        with path.open(errors='replace') as stream:
            for number, line in enumerate(stream, 1):
                if pattern.search(line):
                    matches.append({'line': number, 'text': line.rstrip()})
        record = {'source': str(path.resolve()), 'sha256': audit.digest(path),
                  'status': 'candidate_requires_stage_and_run_attribution', 'matches': matches}
        target = out/f'log_evidence_{index:03d}.json'
        write_json(target, record)
        records.append({'path': target.name, 'source': record['source'], 'sha256': audit.digest(target),
                        'matched_lines': len(matches)})
    return records


def baseline_period(data, keep, period, raw_path, dump_path, country, helper):
    """Separate fitting observations from withheld targets for one period."""
    coords, values = helper.raw_station_data(raw_path)
    coords = {s: p for s, p in coords.items() if points_inside(*p, country)}
    with np.load(dump_path, allow_pickle=False) as dump:
        ids = dump['station_ids'].astype(str)
        # Exclude ALL original withheld IDs, even those outside the country.
        excluded = set(ids[dump['eval_idx']])
        for i in dump['eval_idx']:
            if ids[i] in coords and not np.allclose(coords[ids[i]],
                    [dump['station_lat'][i], dump['station_lon'][i]], rtol=0, atol=1e-5):
                raise ValueError('original and archived coordinates differ')
    retained = sorted(set(coords)-excluded)
    if not retained:
        raise ValueError('no retained original gauges for '+period)
    indices = np.flatnonzero(keep & (data['period'] == period))
    days = np.unique(data['date'][indices]).astype(str)
    for i in indices:
        key = (str(data['date'][i]), data['station'][i])
        if key[1] not in excluded or key[1] not in coords:
            raise ValueError('target is not an original Bangladesh withheld gauge')
        if not np.isclose(values.get(key, np.nan), data['truth'][i], rtol=0, atol=1e-4):
            raise ValueError('original and archived withheld rainfall differ')
    inputs = {'dates': days, 'station_ids': np.asarray(retained),
              'lat_lon': np.asarray([coords[s] for s in retained]),
              'rain_mm': np.asarray([[values.get((d, s), np.nan) for s in retained] for d in days])}
    targets = {'dates': data['date'][indices].astype(str), 'station_ids': data['station'][indices],
               'lat_lon': np.asarray([coords[s] for s in data['station'][indices]]),
               'truth_mm': data['truth'][indices],
               'analysis_members_mm': data['members'][helper.FINAL][indices],
               'background_members_mm': data['members']['background'][indices]}
    return inputs, targets


def prepare_baseline(root, contract, boundary, out, audit):
    scorer, helper = module(90), module(92)
    periods = list(contract['periods']); profile = contract['profiles']['superob-final']
    required = scorer.inventory(root, periods, profile)['required']
    paths = [Path(r['path']) for r in required]
    paths += [root/'stations'/p/'combined_daily.csv' for p in periods]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        return {'status': 'original_archive_required', 'missing': missing}
    country, region = read_boundary(boundary)
    if region['sha256'] != contract['evaluation_region']['boundary_sha256']:
        raise ValueError('country boundary differs from paper contract')
    data, _, _, warnings = scorer.load_samples(root, periods, contract, profile, ['background', helper.FINAL])
    data, _ = scorer.country_samples(data, boundary)
    keep = helper.independent_mask(data, profile)
    generated = []
    for period in periods:
        inputs, targets = baseline_period(data, keep, period,
            root/'stations'/period/'combined_daily.csv', root/'evaluation'/f'{period}.npz', country, helper)
        for role, arrays in (('retained_inputs', inputs), ('verification_only', targets)):
            path = out/f'{period}_{role}.npz'
            np.savez_compressed(path, **arrays)
            generated.append({'path': path.name, 'sha256': audit.digest(path), 'role': role})
    return {'status': 'inputs_exported_not_a_baseline_result', 'outputs': generated,
            'inputs': [{'path': str(p), 'sha256': audit.digest(p)} for p in paths],
            'station_days': int(keep.sum()), 'dates': len(np.unique(data['date'][keep])),
            'evaluation_region': region, 'warnings': warnings,
            'rules': ['Fit and tune using retained inputs or separate training/selection data only.',
                      'Never use verification_only files for fitting covariance or calibrating variance.',
                      'Daily marginal ensembles do not define coherent monthly ensemble trajectories.',
                      'Specify covariance, precipitation transform and uncertainty calibration before scoring.',
                      'Score on matched withheld station-days and report attrition; these files contain no kriging result.']}


def main(argv=None):
    audit = module(97)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence-dir', type=Path, default=audit.default_evidence())
    p.add_argument('--out-dir', type=Path, default=ROOT/'output/paper1_information_bundle')
    p.add_argument('--contract', type=Path, default=ROOT/'configs/paper1_cpcv2_final.json')
    p.add_argument('--root', type=Path, help='original PRISM archive root')
    p.add_argument('--logs', type=Path, nargs='+', help='actual executed preparation/training/sampling logs')
    p.add_argument('--accounting-job-ids', nargs='+', help='explicit Slurm IDs; retrieve read-only sacct records')
    p.add_argument('--selection-root', type=Path, help='directory with original May profile NPZ/JSON pairs')
    p.add_argument('--prepare-baseline-inputs', action='store_true')
    p.add_argument('--boundary-geojson', type=Path, default=DEFAULT_BOUNDARY)
    args = p.parse_args(argv)
    if args.out_dir.exists():
        p.error('use a new --out-dir so old or incomplete collections cannot masquerade as this run')
    if not (args.evidence_dir/'evidence_manifest.json').is_file():
        p.error('supply the completed export with --evidence-dir')
    if args.accounting_job_ids and any(not re.fullmatch(r'\d+(?:_\d+)?', v) for v in args.accounting_job_ids):
        p.error('accounting job IDs must be individual numeric or numeric_array IDs')
    contract = audit.load_json(args.contract)
    rows = audit.review(args.evidence_dir)
    args.out_dir.mkdir(parents=True)
    report = {'status': 'collecting', 'source': str(args.evidence_dir.resolve()),
              'contract_sha256': audit.digest(args.contract), 'evidence': rows,
              'note': 'Private research bundle; presence/candidate records are not new scientific results.'}
    manifest = args.out_dir/'collection_manifest.json'
    write_json(manifest, report)
    report['retained_files'] = retain_evidence(args.evidence_dir, args.out_dir, audit)
    report['logs'] = log_evidence(args.logs or [], args.out_dir, audit)
    if args.accounting_job_ids:
        command = ['sacct', '--noheader', '--parsable2', '--jobs', ','.join(args.accounting_job_ids),
                   '--format', 'JobIDRaw,JobName,State,ElapsedRaw,AllocTRES,NodeList,Start,End']
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        if not result.stdout.strip():
            raise ValueError('sacct returned no records')
        path = args.out_dir/'slurm_job_accounting.psv'
        path.write_text('JobIDRaw|JobName|State|ElapsedRaw|AllocTRES|NodeList|Start|End\n'+result.stdout)
        report['accounting'] = {'path': path.name, 'sha256': audit.digest(path), 'command': command,
                                'note': 'Allocation elapsed time is not measured training/sampling stage time.'}
    if args.selection_root:
        report['selection'] = audit.discover_selection(args.selection_root, args.out_dir/'selection_candidates.json')
    if args.prepare_baseline_inputs:
        report['baseline'] = prepare_baseline(args.root or ROOT/contract['profiles']['superob-final']['root'],
                                             contract, args.boundary_geojson, args.out_dir, audit)
    write_json(args.out_dir/'test_extension_plan.json', audit.extension_plan(contract))
    measurements = [{'stage': stage, 'hardware': None, 'gpus': None, 'wall_seconds': None, 'record_source': None,
                     **({'days': None, 'members': 30, 'steps': 50, 'correctors': 2 if stage == 'analysis' else 0,
                         'grid': [128, 128]} if stage != 'training' else {})}
                    for stage in ('training', 'background', 'analysis')]
    write_json(args.out_dir/'compute.template.json', {'checkpoint_sha256': audit.PINNED, 'measurements': measurements})
    write_json(args.out_dir/'publication_metadata.template.json', {
        'funding_award_and_required_wording': None, 'compute_allocation_and_acknowledgement': None,
        'author_affiliations_confirmed': None, 'code_release_commit_or_tag': None,
        'checkpoint_public_url_or_access_statement': None, 'results_public_url_or_access_statement': None,
        'gauge_data_access_and_redistribution_statement': None})
    report['status'] = 'collection_complete_missing_science_still_pending'
    report['generated_files'] = [{'path': str(path.relative_to(args.out_dir)), 'sha256': audit.digest(path)}
                                  for path in sorted(args.out_dir.rglob('*')) if path.is_file() and path != manifest]
    write_json(manifest, report)
    print('[paper1-collection] '+str(manifest))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        print('[paper1-collection] failed: '+str(error), file=sys.stderr)
        raise SystemExit(1)
