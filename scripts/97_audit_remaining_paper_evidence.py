#!/usr/bin/env python3
"""Review the nine remaining gaps from an existing exported evidence folder.

Reads copied results without overwriting their manifests. Writes a recovery
checklist and a calendar-extension plan, never fabricated measurements. Can
also identify actual May 2022 selection arrays for script 92 to validate.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from bdhires.paper_evidence import checkpoint_metadata
PINNED = 'a04a3d9ae9109f905e06c32bfd55252daf1229d17c98b404e265064b89f210ea'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def manifest_artifact(folder, manifest, name):
    record = next((r for r in manifest.get('outputs', []) if r['path']==name), None)
    path = folder/name
    if not record or not path.is_file():
        return False
    if digest(path) != record['sha256']:
        raise ValueError('export differs from its manifest: '+str(path))
    return True


def discover_selection(folder, output, method='dense_s6_bwdb_r4'):
    """Find the same actual arm across saved profiles; script 92 checks matching.

    Never substitute a different method or synthesize a ranking from filenames.
    Prefixes alone are not evidence: inspect original scope and saved arrays.
    """
    profiles = []
    for report_path in sorted(folder.glob('*.json')):
        prefix = report_path.with_suffix(''); dump_path = Path(str(prefix)+'.npz')
        if not dump_path.is_file():
            continue
        report = load_json(report_path); scope = report.get('scope', {})
        if scope.get('assimilate_all_stations') is not False:
            continue
        with np.load(dump_path, allow_pickle=False) as dump:
            if 'station_'+method not in dump.files:
                continue
            days = dump['times'].astype('datetime64[D]')
            if not len(days) or np.any(days < np.datetime64('2022-05-01')) or np.any(days > np.datetime64('2022-05-31')):
                continue
        profiles.append({'label':prefix.name, 'prefix':str(prefix.resolve()), 'method':method})
    if len(profiles)<2:
        return {'status':'insufficient_saved_profiles', 'found':len(profiles)}
    output.write_text(json.dumps({'profiles':profiles},indent=2)+'\n')
    return {'status':'candidates_need_script92_matching_validation', 'found':len(profiles),
            'configuration':str(output), 'note':'May selection dates only. No winner inferred; comparisons can fail if holdouts, seeds or statistics differ.'}


def extension_plan(contract):
    archive = set()
    for start, end in contract['periods'].values():
        archive.update(np.arange(np.datetime64(start),np.datetime64(end)+np.timedelta64(1,'D')).astype(str))
    rows = []
    for year in range(contract['test_years'][0],contract['test_years'][1]+1):
        for season,start,stop in (('dry_jan_apr',f'{year}-01-01',f'{year}-05-01'),
                                  ('wet_may_sep',f'{year}-05-01',f'{year}-10-01'),
                                  ('dry_oct_dec',f'{year}-10-01',f'{year+1}-01-01')):
            missing=np.asarray([d for d in np.arange(start,stop,dtype='datetime64[D]') if str(d) not in archive])
            if not len(missing):continue
            segments=np.split(missing,np.where(np.diff(missing).astype('timedelta64[D]').astype(int)>1)[0]+1)
            for days in segments:
                rows.append({'year':year,'season':season,'start':str(days[0]),'end':str(days[-1]),'days':len(days),
                             'status':'outside_current_archive_contract; inspect remote inventory before sampling'})
    return {'checkpoint_sha256':PINNED,'plan_only':True,'periods':rows,'total_days':sum(r['days'] for r in rows),
            'requirements':['Confirm raw BMD/BWDB, CPC and prepared 03 UTC IMERG coverage.',
                            'Freeze extension contract and original-gauge holdouts before scoring.',
                            'Keep May 2022 excluded; keep the selected model and DA profile fixed.',
                            'Prepare in a separate archive, recording statistics content hashes.',
                            'Generate withheld-gauge arrays as well as production grids; all-station fit is not verification.',
                            'Validate and merge with all existing test periods; do not relabel partial seasons as complete.']}


def review(export):
    primary=load_json(export/'evidence_manifest.json')
    child=load_json(export/'additional/completion_manifest.json')
    native=all(manifest_artifact(export,primary,name) for name in ('native_imerg_comparison.csv','native_imerg_paired.csv'))
    paired=all(manifest_artifact(export/'additional',child,name) for name in ('paired_idw_intervals.csv','idw_intensity_scores.csv'))
    overlap_path=export/'upstream_station_overlap.csv'; statuses={}
    if manifest_artifact(export,primary,'upstream_station_overlap.csv'):
        with overlap_path.open(newline='') as stream:
            for row in csv.DictReader(stream):
                statuses[row['status']]=statuses.get(row['status'],0)+1
    grid_records=primary.get('gridded_result_files',[])
    grid_hashes_match=all((export/r['path']).is_file() and digest(export/r['path'])==r['sha256'] for r in grid_records)
    if grid_records and not grid_hashes_match:raise ValueError('gridded export files missing or changed')
    grid_verified=grid_hashes_match and {'subgrid_matrix.csv','withheld_gauge_subgrid_anomalies.csv'} <= {Path(r['path']).name for r in grid_records}
    prep=load_json(export/'preparation_stats_provenance.json') if manifest_artifact(export,primary,'preparation_stats_provenance.json') else {}
    historical=all(any(isinstance(v,dict) and v.get('status')=='recorded_at_preparation' and v.get('sha256') for v in r.get('recorded_fields',{}).values()) for r in prep.get('periods',[])) and bool(prep.get('periods'))
    checkpoint=load_json(export/'checkpoint_metadata.json') if manifest_artifact(export,primary,'checkpoint_metadata.json') else {}
    if checkpoint.get('status')=='verified_checkpoint_epoch' and checkpoint.get('sha256')!=PINNED:
        raise ValueError('checkpoint metadata differs from the paper identity')
    return [
        {'item':1,'evidence':'Native 0.1-degree IMERG','status':'generated' if native else 'prepared_native_files_required','next':'Supply PAPER1_NATIVE_IMERG_LIST; rerun script 96. Do not substitute assimilated 0.4-degree IMERG.'},
        {'item':2,'evidence':'Upstream verification-gauge overlap','status':'documented_overlap_available_nonmatches_unknown' if statuses.get('confirmed_overlap_on_dated_inventory') else 'dated_provider_inventories_required','status_counts':statuses,'next':'Supply documented provider/version inventories and ID crosswalks; absence is not proof of non-use.'},
        {'item':3,'evidence':'Paired IDW interval and intensity gains','status':'generated' if paired else 'daily_model_means_export_required','next':'Rerun updated script 92 from original arrays; it now exports daily model means and both comparisons.'},
        {'item':4,'evidence':'Sub-0.4-degree withheld skill','status':'outputs_available_require_scientific_review' if grid_verified else 'gridded_outputs_missing','archive_present_in_hpc_audit':primary.get('availability',{}).get('BD2_full_gridded_archive',False),'next':'PAPER1_FULL_GRIDDED=1 with the CPU launcher; copy evaluation/superob-final/gridded and retain its hashes.'},
        {'item':5,'evidence':'May 2022 profile selection','status':'generated' if manifest_artifact(export/'additional',child,'selection_scores.csv') else 'saved_selection_experiments_required','next':'Use --discover-selection-root on actual profiles, then script 92 --selection; only matched original May arrays establish a ranking.'},
        {'item':6,'evidence':'Compute and evaluated checkpoint epoch','compute_status':'generated' if manifest_artifact(export/'additional',child,'compute_summary.csv') else 'measured_stage_timings_required','epoch_status':checkpoint.get('status','checkpoint_metadata_required'),'next':'Script 96 reads epoch from the hash-verified checkpoint. Supply checkpoint-linked measured timings; validation-curve minimum is not checkpoint identity.'},
        {'item':7,'evidence':'Five-fold BMD and sparse holdouts','status':'comparison_available_needs_design_review' if manifest_artifact(export/'additional',child,'robustness_scores.csv') else 'complete_original_fold_archives_required','next':'Audit/run script 90 --profile bmd-reference. Missing folds require generation. Sparse-network experiments need their own frozen contracts and audited outputs.'},
        {'item':8,'evidence':'Historical preparation statistics','status':'recorded_at_original_preparation' if historical else 'historical_identity_unresolved','next':'Recover original executed logs or archived hashes. New script 87 records the actual transform/hash prospectively; never rewrite old provenance as if measured then.'},
        {'item':9,'evidence':'Remaining 2021-2025 dates','status':'archive_extension_required','next':'Review test_extension_plan.json and remote inventory; missing dates require new preparation and sampling.'}]


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence-dir',type=Path,default=ROOT/'paper1_updated_evidence')
    p.add_argument('--out-dir',type=Path,default=ROOT/'output/paper1_remaining_evidence')
    p.add_argument('--contract',type=Path,default=ROOT/'configs/paper1_cpcv2_final.json')
    p.add_argument('--discover-selection-root',type=Path,help='actual directory of saved profile .npz/.json pairs')
    p.add_argument('--checkpoint',type=Path,help='inspect this actual pinned checkpoint in the existing Torch environment')
    p.add_argument('--idw-samples',type=Path,help='new daily IDW/model export; old exports without analysis means are rejected')
    args=p.parse_args(argv);args.out_dir.mkdir(parents=True,exist_ok=True)
    rows=review(args.evidence_dir);contract=load_json(args.contract)
    report={'source':str(args.evidence_dir),'remaining_items':rows}
    if args.discover_selection_root:
        report['selection_discovery']=discover_selection(args.discover_selection_root,args.out_dir/'selection_candidates.json')
    if args.checkpoint:
        report['checkpoint']=checkpoint_metadata(args.checkpoint,PINNED)
        (args.out_dir/'checkpoint_metadata.json').write_text(json.dumps(report['checkpoint'],indent=2)+'\n')
    if args.idw_samples:
        completion=load_json(args.idw_samples.parent/'completion_manifest.json')
        if not manifest_artifact(args.idw_samples.parent,completion,args.idw_samples.name):
            raise ValueError('IDW daily samples must be hashed in their original completion_manifest.json')
        if completion.get('evaluation_region',{}).get('sha256') != contract['evaluation_region']['boundary_sha256']:
            raise ValueError('IDW daily export does not establish the same Bangladesh boundary')
        spec=importlib.util.spec_from_file_location('_idw_completion',ROOT/'scripts/92_complete_cpcv2_paper1.py')
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        mod.build_idw_comparisons(mod.read_csv(args.idw_samples),args.out_dir)
        report['idw_recovery']={'status':'generated','daily_input_sha256':digest(args.idw_samples)}
    (args.out_dir/'test_extension_plan.json').write_text(json.dumps(extension_plan(contract),indent=2)+'\n')
    (args.out_dir/'remaining_evidence.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Remaining paper evidence','', '| Item | Evidence | Status |','|---|---|---|']
    lines += [f"| {r['item']} | {r['evidence']} | {r.get('status',r.get('compute_status',''))} |" for r in rows]
    lines += ['',* [f"{r['item']}. {r['next']}" for r in rows]]
    (args.out_dir/'remaining_evidence.md').write_text('\n'.join(lines)+'\n')
    print('[remaining-evidence] '+str(args.out_dir/'remaining_evidence.md'))
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,OSError,KeyError) as error:
        print('[remaining-evidence] validation failed: '+str(error),file=sys.stderr)
        raise SystemExit(1)
