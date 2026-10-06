#!/usr/bin/env python3
"""Print read-only SURMA production progress using standard-library Python."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import subprocess


def quarters(first, last):
    result=[]
    for year in range(first,last+1):
        for q,month in enumerate((1,4,7,10),1):
            start=date(year,month,1)
            stop=date(year+1,1,1) if q==4 else date(year,month+3,1)
            result.append({'label':f'{year}_q{q}','start':start.isoformat(),
                           'end':(stop-timedelta(days=1)).isoformat(),'days':(stop-start).days})
    return result


def read_json(path):
    try: return json.loads(Path(path).read_text())
    except (OSError,ValueError): return None


def matching_period(record, period):
    return isinstance(record,dict) and record.get('period')=={k:period[k] for k in ('label','start','end')}


def recorded_completion(root, period):
    """Receipt plus matching completed artifacts; no heavy array/data re-audit."""
    label=period['label']
    receipt=read_json(root/'validated'/f'{label}.json')
    if receipt is None: return False
    if not matching_period(receipt,period) or receipt.get('status')!='complete' or receipt.get('days')!=period['days']:
        return False
    fields=root/'gridded'/f'{label}.zarr'
    if not isinstance(receipt.get('field_store'),str) or Path(receipt['field_store']).resolve()!=fields.resolve(): return False
    meta=read_json(fields/'.zattrs')
    report=read_json(root/'production_metadata'/f'{label}.json')
    arrays=root/'production_metadata'/f'{label}.npz'
    if (not isinstance(meta,dict) or not isinstance(report,dict) or not arrays.is_file()
            or arrays.stat().st_size==0 or meta.get('complete') is not True
            or meta.get('schema')!='bdhires.physical_ensemble.v1'): return False
    scope=report.get('scope',{})
    return (isinstance(scope,dict) and scope.get('start')==period['start'] and scope.get('end')==period['end']
            and scope.get('members')==30 and meta.get('scope')==scope)


def slurm_records(job_ids):
    if not job_ids: return {}, ['No Slurm jobs queried; pass --jobs for live job/log progress.']
    records={}; notes=[]
    commands=[['sacct','-X','-n','-P','-j',','.join(job_ids),
               '--format=JobID%80,State%40,ExitCode'],
              ['squeue','--array','-h','-j',','.join(job_ids),'-o','%i|%T|%R']]
    for command in commands:
        try:
            response=subprocess.run(command,capture_output=True,text=True,timeout=20)
        except (OSError,subprocess.TimeoutExpired) as exc:
            notes.append(f'{command[0]} unavailable: {exc}'); continue
        if response.returncode:
            notes.append(f'{command[0]}: {response.stderr.strip() or "query failed"}'); continue
        for line in response.stdout.splitlines():
            fields=[value.strip() for value in line.split('|')]
            if len(fields)<3 or not re.fullmatch(r'\d+_\d+',fields[0]): continue
            job,state,value=fields[:3]
            row=records.setdefault(job,{'job':job})
            row['state']=state.split()[0].rstrip('+')
            row['exit_code' if command[0]=='sacct' else 'reason']=value
    return records,notes


def log_progress(path, period):
    try:
        with Path(path).open('rb') as handle:
            handle.seek(0,2);size=handle.tell();handle.seek(max(0,size-65536))
            tail=handle.read().decode('utf-8',errors='replace')
    except OSError: return {'days':0,'seconds_per_day':None}
    matches=re.findall(r'\[sweep\] day (\d+)/(\d+) (\d{4}-\d{2}-\d{2}) \(([\d.]+) s/day\)',tail)
    for done,total,day,pace in reversed(matches):
        done,total=int(done),int(total)
        if (total==period['days'] and 1<=done<=total and
                day==(date.fromisoformat(period['start'])+timedelta(days=done-1)).isoformat()):
            return {'days':done,'seconds_per_day':float(pace)}
    return {'days':0,'seconds_per_day':None}


def snapshot(root, first, last, logs, job_ids):
    root=Path(root);periods=quarters(first,last)
    jobs,notes=slurm_records(job_ids)
    latest={}
    for job,row in sorted(jobs.items(),key=lambda item:int(item[0].split('_')[0])):
        index=int(job.split('_')[1])
        if 0<=index<len(periods): latest[index]=row
    rows=[]
    for index,p in enumerate(periods):
        receipt=read_json(root/'prepared'/f"{p['label']}.json")
        prepared=matching_period(receipt,p) and receipt.get('status')=='validated_preparation'
        complete=recorded_completion(root,p)
        job=latest.get(index,{})
        progress=log_progress(Path(logs)/f"surma-prod-01-24-{job['job']}.out",p) if job else {'days':0,'seconds_per_day':None}
        row={**p,'task':index,'prepared':bool(prepared),'validated':complete,
             'job':job.get('job'),'state':job.get('state','UNKNOWN'),
             'exit_code':job.get('exit_code'),'reason':job.get('reason'),
             'logged_days':progress['days'],'seconds_per_day':progress['seconds_per_day']}
        if not complete and row['state']=='COMPLETED':
            notes.append(f"{p['label']}: Slurm COMPLETED but no matching validated output; inspect its log/artifacts.")
        if (root/'validated'/f"{p['label']}.json").exists() and not complete:
            notes.append(f"{p['label']}: completion receipt/artifacts missing, unreadable or inconsistent.")
        rows.append(row)
    total_days=sum(p['days'] for p in periods)
    complete_days=sum(p['days'] for p in rows if p['validated'])
    active=[p for p in rows if not p['validated'] and p['state'] in ('RUNNING','COMPLETING','SUSPENDED')]
    active_days=sum(p['logged_days'] for p in active)
    summary={'total_quarters':len(rows),'prepared_quarters':sum(p['prepared'] for p in rows),
             'validated_quarters':sum(p['validated'] for p in rows),'total_days':total_days,
             'validated_days':complete_days,'completion_percent':100*complete_days/total_days,
             'remaining_unvalidated_days':total_days-complete_days,
             'active_logged_days':active_days,'known_sampling_days':complete_days+active_days,
             'known_sampling_percent':100*(complete_days+active_days)/total_days,
             'validated_analysis_member_days':complete_days*30,
             'total_analysis_member_days':total_days*30,'states':dict(Counter(p['state'] for p in rows))}
    manifest=read_json(root/'production_manifest.json')
    shards=manifest.get('shards',[]) if isinstance(manifest,dict) else []
    expected={k:f'{first}-01-01' if k=='start' else f'{last}-12-31' for k in ('start','end')}
    summary['final_receipt_complete']=(isinstance(manifest,dict) and manifest.get('status')=='complete'
        and all(manifest.get(k)==v for k,v in expected.items()) and manifest.get('days')==total_days
        and manifest.get('members')==30 and summary['validated_quarters']==len(rows)
        and isinstance(shards,list) and len(shards)==len(rows)
        and all(isinstance(s,dict) and isinstance(s.get('period'),dict) for s in shards)
        and {s['period'].get('label') for s in shards}=={p['label'] for p in rows})
    return {'checked_utc':datetime.now(timezone.utc).isoformat(),'root':str(root),
            'scope':expected,'summary':summary,'quarters':rows,'notes':notes}


def print_report(report, details=False):
    s=report['summary']
    print(f"SURMA PRODUCTION STATUS: {report['scope']['start']} .. {report['scope']['end']}")
    print(f"Prepared quarters:             {s['prepared_quarters']}/{s['total_quarters']}")
    print(f"Validated completed quarters:  {s['validated_quarters']}/{s['total_quarters']}")
    print(f"Validated completed days:      {s['validated_days']:,}/{s['total_days']:,} ({s['completion_percent']:.1f}%)")
    print(f"Remaining unvalidated days:    {s['remaining_unvalidated_days']:,}")
    print(f"Logged days in active jobs:    {s['active_logged_days']:,} (not yet validated)")
    print(f"Known sampling progress:       {s['known_sampling_days']:,}/{s['total_days']:,} ({s['known_sampling_percent']:.1f}%; includes active logs)")
    print(f"Completed analysis member-days:{s['validated_analysis_member_days']:>10,}/{s['total_analysis_member_days']:,}")
    print('Latest job states:             '+', '.join(f'{state}={count}' for state,count in sorted(s['states'].items())))
    print('Final production receipt:      '+('COMPLETE' if s['final_receipt_complete'] else 'pending/unverified'))
    print('\nYEAR  QUARTERS   VALIDATED DAYS  ACTIVE LOGGED DAYS')
    for year in sorted({p['start'][:4] for p in report['quarters']}):
        rows=[p for p in report['quarters'] if p['start'].startswith(year)]
        days=sum(p['days'] for p in rows if p['validated']);total=sum(p['days'] for p in rows)
        logged=sum(p['logged_days'] for p in rows if not p['validated'] and p['state'] in ('RUNNING','COMPLETING','SUSPENDED'))
        print(f"{year}    {sum(p['validated'] for p in rows)}/4        {days:3}/{total}              {logged:3}")
    visible=[p for p in report['quarters'] if details or (not p['validated'] and p['state'] not in ('PENDING','UNKNOWN'))]
    if visible:
        print('\nTASK QUARTER  JOB                  STATE                 LOGGED DAYS  S/DAY')
        for p in visible:
            state='VALIDATED' if p['validated'] else p['state']
            pace=f"{p['seconds_per_day']:.0f}" if p['seconds_per_day'] is not None else '--'
            print(f"{p['task']:3}  {p['label']:8} {p['job'] or '--':20} {state:21} {p['logged_days']:3}/{p['days']:<3}      {pace:>5}")
    for note in report['notes']: print('NOTE: '+note)
    print('\nRecorded completion; arrays are not re-audited. Active logged days are excluded from completion percentage.')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default=os.environ.get('SURMA_PROD_ROOT','data/processed/brishti05_production_2001_2024'))
    p.add_argument('--start-year',type=int,default=2001);p.add_argument('--end-year',type=int,default=2024)
    p.add_argument('--logs',default='logs');p.add_argument('--jobs',nargs='+',default=[])
    p.add_argument('--details',action='store_true');p.add_argument('--json',action='store_true')
    args=p.parse_args(argv)
    if not 2001<=args.start_year<=args.end_year<=2024: p.error('years must be within 2001..2024')
    jobs=[job for value in args.jobs for job in value.split(',')]
    if any(not re.fullmatch(r'\d+',job) for job in jobs): p.error('--jobs requires numeric parent array IDs')
    report=snapshot(args.root,args.start_year,args.end_year,args.logs,jobs)
    if args.json: print(json.dumps(report,indent=2,allow_nan=False))
    else: print_report(report,args.details)


if __name__=='__main__': main()
