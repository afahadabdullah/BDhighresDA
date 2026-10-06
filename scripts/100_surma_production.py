#!/usr/bin/env python3
"""Audit, acquire, prepare and validate full-calendar CPCv2 production inputs.

Use slurm/submit_surma_production_2001_2024.sh for the dependency chain.
Metadata-only audit uses the standard library and runs on a login node. Deep
audit and preparation use the existing scientific environment on CPU nodes.
Never recomputes model normalization statistics or overwrites sampled output.
"""
from __future__ import annotations

import argparse
import calendar
from datetime import date, timedelta
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_SHA = 'a04a3d9ae9109f905e06c32bfd55252daf1229d17c98b404e265064b89f210ea'
STATS_SHA = '96b4de3862d931e0b4dc9f2e895b944a3d696a90187b592782771e65d5a8ce07'
CHANNELS = ['cpc_precip', 'cpc_valid', 'era5_tcwv', 'era5_cape', 'era5_u10', 'era5_v10', 'era5_msl']
FINAL = 'dense_s6_bwdb_r4'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def dates(start, end):
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    return [(first+timedelta(days=i)).isoformat() for i in range((last-first).days+1)]


def periods(first, last):
    return [{'label': f'{year}_q{q}', 'start': f'{year}-{month:02d}-01',
             'end': f'{year}-{month+2:02d}-{calendar.monthrange(year,month+2)[1]:02d}'}
            for year in range(first,last+1) for q,month in enumerate((1,4,7,10),1)]


def monthly(first, last):
    return [{'start': f'{year}-{month:02d}-01',
             'end': f'{year}-{month:02d}-{calendar.monthrange(year,month)[1]:02d}'}
            for year in range(first,last+1) for month in range(1,13)]


def attrs(path):
    meta = Path(path)/'.zattrs'
    return json.loads(meta.read_text()) if meta.is_file() else {}


def predictor_metadata_ready(path, first, last):
    """Fast planner hint; deep audit checks dates, channels and actual arrays."""
    meta = attrs(path)
    return (meta.get('schema_version') == 1 and meta.get('complete') is True
            and set(range(first-1,last+1)) <= set(meta.get('completed_years', []))
            and set(CHANNELS) <= set(meta.get('cond_channels', [])))


def choose_data(args):
    if args.data_zarr:
        return Path(args.data_zarr)
    reference = Path('data/processed/bd_wide_cpc.zarr')
    return reference if predictor_metadata_ready(reference,args.start_year,args.end_year) else Path(args.root)/'predictors.zarr'


def run(script, *arguments, python=None):
    command = [str(python or sys.executable), '-u', str(ROOT/'scripts'/script), *map(str,arguments)]
    print('[production] '+ ' '.join(command), flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


def era5_file_ready(folder, year):
    import importlib.util
    path=Path(folder)/f'era5_daily_{year}.nc'
    if not path.is_file(): return False
    spec=importlib.util.spec_from_file_location('_production_era5',ROOT/'scripts/00_download_era5.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    try: mod.validate_year(path,year)
    except (ValueError,OSError): return False
    return True


def validate_imerg(path, start, end, factor=2):
    import numpy as np
    import xarray as xr
    sys.path.insert(0,str(ROOT/'scripts'))
    import importlib.util
    spec = importlib.util.spec_from_file_location('_prepared_imerg', ROOT/'scripts/43_subset_prepared_imerg.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    with xr.open_dataset(path) as ds:
        mod.validate_source(ds, Path(path), False)
        actual = ds.time.values.astype('datetime64[D]').astype(str).tolist()
        if actual != dates(start,end):
            raise ValueError('IMERG daily dates incomplete or duplicated: '+str(path))
        step=.05*factor; n=128//factor
        if (ds.sizes.get('lat')!=n or ds.sizes.get('lon')!=n
                or not np.allclose(ds.lat.values,20.3+step*(np.arange(n)+.5),rtol=0,atol=2e-3)
                or not np.allclose(ds.lon.values,87.6+step*(np.arange(n)+.5),rtol=0,atol=2e-3)):
            raise ValueError('IMERG grid is not nested on the checkpoint BD domain: '+str(path))
        if factor == 8 and (int(ds.attrs.get('observation_factor',0)) != 8
                            or abs(float(ds.attrs.get('required_error_corr_cells',0))-.75)>1e-6):
            raise ValueError('IMERG coarsening/error contract differs: '+str(path))
        for name in ('precipitation','randomError'):
            finite = np.isfinite(ds[name].values)
            if not np.all(finite.reshape(len(actual),-1).any(axis=1)):
                raise ValueError('IMERG has an empty daily field: '+str(path))
        if np.any(np.isfinite(ds.precipitation.values) & (ds.precipitation_cnt.values<48)):
            raise ValueError('IMERG finite precipitation lacks all 48 half-hours: '+str(path))


def validate_predictors(path, first, last):
    import numpy as np
    import zarr
    store = zarr.open_group(str(path), mode='r')
    if not predictor_metadata_ready(path,first,last):
        raise ValueError('predictor metadata does not establish complete requested years')
    names = list(store.attrs['cond_channels'])
    missing = set(CHANNELS)-set(names)
    if missing:
        raise ValueError('missing conditioning channels: '+str(sorted(missing)))
    times = np.asarray(store['time'][:],dtype='datetime64[ns]').astype('datetime64[D]')
    required = np.asarray(dates(f'{first-1}-12-31',f'{last}-12-31'), dtype='datetime64[D]')
    selected = np.flatnonzero((times>=required[0]) & (times<=required[-1]))
    if not np.array_equal(times[selected],required):
        raise ValueError('predictors lack requested dates or the previous-day boundary')
    if any(name not in store for name in ('cond','target','static','valid','lat','lon')):
        raise ValueError('packed predictor/context/static arrays are missing')
    valid = np.asarray(store['valid'][:])>.5
    if valid.shape != (256,256) or not valid.any():
        raise ValueError('predictors must use the checkpoint WIDE 256x256 grid')
    if (not np.allclose(store['lat'][:],16+.05*(np.arange(256)+.5),rtol=0,atol=5e-6)
            or not np.allclose(store['lon'][:],84+.05*(np.arange(256)+.5),rtol=0,atol=5e-6)):
        raise ValueError('predictor grid coordinates differ from checkpoint WIDE')
    static=np.asarray(store['static'][:])
    if static.shape!=(7,256,256) or not np.isfinite(static).all():
        raise ValueError('seven finite checkpoint static channels are required')
    # Check all required days, one at a time, without loading the archive in RAM.
    channels = [names.index(name) for name in CHANNELS]
    for index in selected:
        values = np.asarray(store['cond'][int(index)])[channels]
        if not np.isfinite(values[:,valid]).all():
            raise ValueError('non-finite predictors on '+str(times[index]))
        if not np.any(values[CHANNELS.index('cpc_valid')][valid] > 0):
            raise ValueError('CPC conditioning entirely unavailable on '+str(times[index]))
        if not np.isfinite(np.asarray(store['target'][int(index)])[valid]).all():
            raise ValueError('packed CHIRPS context is incomplete on '+str(times[index]))
    return {'status':'validated_all_days','path':str(path), 'first_background_date':str(required[0]),
            'last_date':str(required[-1]), 'days':len(required), 'selected_channels':CHANNELS}


def fixed_inputs(args, deep=False):
    result = {}
    for name,path,expected in [('checkpoint',args.ckpt,CHECKPOINT_SHA),('statistics',args.stats,STATS_SHA)]:
        p = Path(path)
        if not p.is_file():
            result[name]={'path':str(p),'status':'missing; recover evaluated artifact, do not retrain/recompute'}
        else:
            digest = sha(p)
            result[name]={'path':str(p),'sha256':digest,'status':'verified' if digest==expected else 'identity_mismatch'}
    for name,path in [('bmd_wide',args.bmd_wide),('bmd_catalog',args.bmd_stations),('bwdb',args.bwdb)]:
        p=Path(path); result[name]={'path':str(p),'status':'present' if p.is_file() else 'missing',
                                  'sha256':sha(p) if p.is_file() else None}
    if deep and result['checkpoint']['status']=='verified':
        import torch
        checkpoint=torch.load(args.ckpt,map_location='cpu',weights_only=True)
        cfg=checkpoint['cfg']['data']
        if cfg.get('cond_channels') != CHANNELS or str(cfg.get('stats')) != str(args.stats):
            raise ValueError('use the checkpoint-bound statistics path and conditioning channels')
        result['checkpoint']['completed_epoch']=int(checkpoint['epoch'])+1
    return result


def source_check(args):
    """Print a read-only source inventory; no scientific imports or Slurm job."""
    mode=getattr(args,'color','auto')
    color=(mode=='always' or (mode=='auto' and sys.stdout.isatty()
           and 'NO_COLOR' not in os.environ and os.environ.get('TERM')!='dumb'))

    def mark(text,ok,width=0,right=False):
        label=('✓' if ok else '✗')+str(text)
        label=label.rjust(width) if right else label.ljust(width)
        return f'\033[{32 if ok else 31}m{label}\033[0m' if color else label

    def cell(text):
        if text=='--': return text.rjust(9)
        if '/' in text:
            count,total=map(int,text.split('/'))
            ok=count==total
        else:
            ok=text=='RAW' or text.startswith('P')
        return mark(text,ok,9,right=True)

    def present(path):
        path=Path(path)
        return path.is_file() and path.stat().st_size>0

    candidates=([Path(args.data_zarr)] if args.data_zarr else
                [Path('data/processed/bd_wide_cpc.zarr'),Path(args.root)/'predictors.zarr'])
    packed=[]
    print('SOURCE CHECK: file presence and recorded metadata; not deep validation')
    print(f'Source years checked: {args.start_year}..{args.end_year}; '
          f'{args.start_year-1} is preceding-year predictor context')
    print('\nShared inputs:')
    for name,record in fixed_inputs(args).items():
        label=mark(record['status'],record['status'] in ('verified','present'),58)
        print(f"  {name:14} {label} {record['path']}")
    exists=present(args.static)
    print(f"  {'static grid':14} {mark('PRESENT' if exists else 'MISSING',exists,58)} {args.static}")
    print('  Station file presence does not establish daily/yearly gauge coverage.')
    print('\nPacked source candidates:')
    for index,path in enumerate(candidates,1):
        try:
            meta=attrs(path)
            if not isinstance(meta,dict): raise ValueError('invalid packed metadata object')
            years=meta.get('completed_years',[])
            channels=meta.get('cond_channels',[])
            if not isinstance(years,list) or not isinstance(channels,list):
                raise ValueError('invalid completed_years/cond_channels metadata')
            packed.append((index,path,meta,set(years),set(channels)))
            print(f'  {mark(f"P{index}",meta.get("complete") is True)}: {path}; complete={meta.get("complete",False)}; '
                  f'recorded years={years}; channels={channels}')
        except (OSError,ValueError,TypeError) as exc:
            print(f'  {mark(f"P{index}",False)}: {path}; unreadable metadata ({type(exc).__name__})')

    def packed_source(year,array,channel=None):
        for index,path,meta,years,channels in packed:
            if (meta.get('schema_version')==1 and year in years
                    and (channel is None or channel in channels)
                    and (path/array/'.zarray').is_file()):
                return f'P{index}'
        return None

    header=f"{'YEAR':<6}"+''.join(f'{name:>9}' for name in
        ('CPC_P','CPC_V','TCWV','CAPE','U10','V10','MSL','CHIRPS','IM_FILES','IM_REC'))
    print('\n'+header)
    print('-'*len(header))
    totals=[0,0]
    for year in range(args.start_year-1,args.end_year+1):
        raw_cpc=present(Path(args.cpc)/f'precip.{year}.nc')
        raw_era5=present(Path(args.era5)/f'era5_daily_{year}.nc')
        raw_chirps=present(Path(args.chirps)/f'chirps_wide_{year}.nc')
        cells=[packed_source(year,'cond',channel) or ('RAW' if
               (raw_cpc if channel.startswith('cpc_') else raw_era5) else 'MISS')
               for channel in CHANNELS]
        cells.append(packed_source(year,'target') or ('RAW' if raw_chirps else 'MISS'))
        files=recorded=0
        if year>=args.start_year:
            report=Path(args.imerg_state)/'years'/str(year)/'status.json'
            try:
                state=json.loads(report.read_text()) if report.is_file() else {}
                if not isinstance(state,dict): raise ValueError('invalid IMERG report object')
                records=state.get('months',{})
                if not isinstance(records,dict): records={}
            except (OSError,ValueError):
                records={}
                print(f'  NOTE: unreadable IMERG report: {report}')
            for month in monthly(year,year):
                path=Path(args.imerg_daily)/f"imerg_bd_aligned_{month['start'].replace('-','')}_{month['end'].replace('-','')}.nc"
                exists=present(path)
                files+=exists
                record=records.get(month['start'][:7],{})
                if (exists and isinstance(record,dict) and record.get('status')=='validated'
                        and isinstance(record.get('file'),str) and record['file']
                        and Path(record['file']).resolve()==path.resolve()):
                    recorded+=1
            totals[0]+=files; totals[1]+=recorded
            cells.extend((f'{files}/12',f'{recorded}/12'))
        else:
            cells.extend(('--','--'))
        print(f'{year:<6}'+''.join(cell(value) for value in cells))
    print(f'\n{mark(" present/complete",True)}; {mark(" missing/incomplete or identity mismatch",False)}; -- = not applicable.')
    print('P1/P2 = that variable/year is recorded in packed metadata; arrays not revalidated.')
    print('RAW = nonempty annual source file exists; variables/dates/values not inspected.')
    print('MISS = neither a recorded packed source nor the expected raw file was found.')
    print('IM_FILES = prepared monthly files present; IM_REC = matching validation records, not a fresh audit.')
    print(f'IMERG totals: {totals[0]}/{12*(args.end_year-args.start_year+1)} files; '
          f'{totals[1]} matching month validation records.')
    print('\nRaw source patterns:')
    print(f'  CPC:    {args.cpc}/precip.YEAR.nc -> cpc_precip; cpc_valid is derived during packing')
    print(f'  ERA5:   {args.era5}/era5_daily_YEAR.nc -> tcwv, cape, u10, v10, msl')
    print(f'  CHIRPS: {args.chirps}/chirps_wide_YEAR.nc -> precipitation target/context')
    print(f'  IMERG:  {args.imerg_daily}/imerg_bd_aligned_START_END.nc -> precipitation, randomError')
    print(f'  IMERG validation reports: {args.imerg_state}/years/YEAR/status.json')


def audit(args):
    selected=choose_data(args)
    raw=[]
    for year in range(args.start_year-1,args.end_year+1):
        row={'year':year}
        for name,path in [('era5',Path(args.era5)/f'era5_daily_{year}.nc'),
                          ('cpc',Path(args.cpc)/f'precip.{year}.nc'),
                          ('chirps',Path(args.chirps)/f'chirps_wide_{year}.nc')]:
            row[name]={'path':str(path),'present':path.is_file() and path.stat().st_size>0}
        raw.append(row)
    imerg=[]
    for p in monthly(args.start_year,args.end_year):
        path=Path(args.imerg_daily)/f"imerg_bd_aligned_{p['start'].replace('-','')}_{p['end'].replace('-','')}.nc"
        row={**p,'path':str(path),'status':'present_unvalidated' if path.is_file() else 'download_and_prepare_required'}
        if path.is_file() and args.deep:
            try: validate_imerg(path,p['start'],p['end']); row['status']='validated'
            except (ValueError,OSError) as exc: row.update(status='invalid_requires_recovery',reason=str(exc))
        imerg.append(row)
    report={'scope':{'start':f'{args.start_year}-01-01','end':f'{args.end_year}-12-31',
                     'days':len(dates(f'{args.start_year}-01-01',f'{args.end_year}-12-31')),
                     'quarterly_tasks':len(periods(args.start_year,args.end_year)), 'members':30},
            'fixed_inputs':fixed_inputs(args,args.deep),'predictors':{'path':str(selected),
            'status':'metadata_ready_requires_deep_validation' if predictor_metadata_ready(selected,args.start_year,args.end_year) else 'pack_or_complete_required'},
            'annual_sources':raw,'imerg_months':imerg,
            'notes':['CPC/ERA5/CHIRPS annual sources are unnecessary to reacquire if the existing packed predictors validate.',
                     '2000-12-31 conditioning is needed for 2001-01-01 with background offset -1.',
                     'CHIRPS is needed by the existing packer/archive context, not assimilated as an observation.',
                     'BMD and BWDB daily support remains different by three hours, matching the evaluated method.',
                     'All eligible original gauges are assimilated in production; these grids are not withheld verification.']}
    if args.deep and predictor_metadata_ready(selected,args.start_year,args.end_year):
        report['predictors']=validate_predictors(selected,args.start_year,args.end_year)
    write_json(args.report or Path(args.root)/'input_inventory.json',report)
    print(json.dumps({'scope':report['scope'],'fixed_inputs':report['fixed_inputs'],
                      'predictors':report['predictors'],
                      'imerg_months_missing_or_invalid':sum(r['status'] in ('download_and_prepare_required','invalid_requires_recovery') for r in imerg),
                      'imerg_months_present_unvalidated':sum(r['status']=='present_unvalidated' for r in imerg)},indent=2))


def preflight(args):
    fixed=fixed_inputs(args,deep=True)
    if any(v['status'] not in ('verified','present') for v in fixed.values()):
        raise ValueError('required evaluated model/statistics or gauge source missing/mismatched: '+str(fixed))
    selected=choose_data(args)
    if not predictor_metadata_ready(selected,args.start_year,args.end_year):
        if not Path(args.static).is_file():
            raise ValueError('recover the original data/static/static_wide.nc before packing')
        era_python=os.environ.get('ERA5_PYTHON','/home/afahad/nb/project/BDDA/envs/bdda-earthmover/bin/python')
        need_era=any(not era5_file_ready(args.era5,y) for y in range(args.start_year-1,args.end_year+1))
        if need_era and not Path(era_python).is_file():
            raise ValueError('ERA5 download environment missing; run bash slurm/setup_earthmover_env.sh or set ERA5_PYTHON')
    missing=[]
    for m in monthly(args.start_year,args.end_year):
        path=Path(args.imerg_daily)/f"imerg_bd_aligned_{m['start'].replace('-','')}_{m['end'].replace('-','')}.nc"
        if not path.is_file(): missing.append(str(path))
    if missing and not (Path.home()/'.netrc').is_file():
        raise ValueError('missing IMERG months require Earthdata credentials in ~/.netrc; no credentials are printed')
    gauges=audit_gauges(args)
    write_json(Path(args.root)/'preflight.json',{'status':'fixed_inputs_verified',
               'fixed_inputs':fixed,'predictor_path':str(selected),'missing_imerg_months':len(missing),
               'gauge_coverage':gauges})


def audit_gauges(args):
    """Check every quarter of the actual national source files before sampling."""
    import importlib.util
    import pandas as pd
    spec=importlib.util.spec_from_file_location('_production_stations',ROOT/'scripts/99_prepare_production_stations.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    start,end=pd.Timestamp(f'{args.start_year}-01-01'),pd.Timestamp(f'{args.end_year}-12-31')
    bmd,bmd_qc=mod.read_bmd(args,start,end);bmd['source']='BMD'
    bwdb,_=mod.prep82.read_bwdb(Path(args.bwdb),start,end,500.)
    rows=[]
    for frame in (bmd,bwdb):
        if frame.empty: raise ValueError('national gauge source has no requested observations')
        frame=frame.loc[frame['lat'].between(20.325,26.675) & frame['lon'].between(87.625,93.975)].copy()
        if frame.empty: raise ValueError('national gauge source has no stations inside model domain')
        frame['quarter']=frame['date'].dt.to_period('Q')
        source=str(frame['source'].iloc[0])
        counts=frame.groupby(['quarter','station_id'])['precip_mm'].count()
        for q in pd.period_range(start,end,freq='Q'):
            n=(q.end_time.normalize()-q.start_time).days+1
            stations=int((counts.loc[q]>=.5*n).sum()) if q in counts.index.levels[0] else 0
            if stations < {'BMD':20,'BWDB':150}[source]:
                raise ValueError(f'{q}: too few eligible {source} gauges ({stations}); inspect original coverage')
            rows.append({'quarter':str(q),'source':source,'days':n,'eligible_stations':stations})
    result={'status':'all_quarters_passed','quarter_counts':rows,
            'bmd_source_qc':bmd_qc,
            'minimum_by_source':{s:min(r['eligible_stations'] for r in rows if r['source']==s) for s in ('BMD','BWDB')},
            'source_hashes':{str(p):sha(p) for p in (args.bmd_wide,args.bmd_stations,args.bwdb)}}
    write_json(args.report or Path(args.root)/'gauge_coverage.json',result)
    return result


def download_year(args):
    selected=choose_data(args)
    if predictor_metadata_ready(selected,args.start_year,args.end_year):
        print('[production] reusing packed predictors; annual download unnecessary'); return
    year=args.start_year-1+args.task
    if not args.start_year-1 <= year <= args.end_year: raise ValueError('annual task out of range')
    run('02b_download_cpc.py','--start',year,'--end',year,'--out',args.cpc,'--require-complete')
    run('01_download_chirps.py','--start',year,'--end',year,'--out',args.chirps)
    if era5_file_ready(args.era5,year):
        print('[production] validated existing ERA5 year '+str(year)); return
    era_python=os.environ.get('ERA5_PYTHON','/home/afahad/nb/project/BDDA/envs/bdda-earthmover/bin/python')
    run('00_download_era5.py','--start',year,'--end',year,'--out',args.era5,'--workers',2,python=era_python)


def download_month(args):
    p=monthly(args.start_year,args.end_year)[args.task]
    out=Path(args.imerg_daily)/f"imerg_bd_aligned_{p['start'].replace('-','')}_{p['end'].replace('-','')}.nc"
    if out.is_file():
        validate_imerg(out,p['start'],p['end'])
        print('[production] validated existing '+str(out)); return
    raw=Path(args.imerg_raw)/p['start'][:4]
    run('02_download_imerg_halfhourly.py','--bmd-start',p['start'],'--bmd-end',p['end'],
        '--end-hour-utc',3,'--bbox','20.3,87.6,26.7,94.0','--jobs',2,'--out',raw)
    run('08_prepare_imerg_observations.py','--input',raw,'--source-frequency','half-hourly',
        '--start',p['start'],'--end',p['end'],'--min-count',48,'--accumulation-end-hour-utc',3,
        '--out',out,'--report',out.with_name(out.stem+'_qc.json'))
    validate_imerg(out,p['start'],p['end'])


def pack(args):
    selected=choose_data(args)
    if not predictor_metadata_ready(selected,args.start_year,args.end_year):
        if selected.exists() and attrs(selected).get('start_year') != args.start_year-1:
            raise ValueError('existing predictor store has a different range; choose a separate --data-zarr')
        if not Path(args.static).is_file():
            raise ValueError('missing trained static grid; recover data/static/static_wide.nc')
        run('04_regrid_and_pack.py','--start',args.start_year-1,'--end',args.end_year,
            '--era5',args.era5,'--chirps',args.chirps,'--cpc',args.cpc,'--static',args.static,'--out',selected)
    result=validate_predictors(selected,args.start_year,args.end_year)
    fixed=fixed_inputs(args,deep=True)
    if any(v['status'] not in ('verified','present') for v in fixed.values()):
        raise ValueError('required fixed inputs missing or identity differs: '+str(fixed))
    write_json(Path(args.root)/'predictor_validation.json',{'predictors':result,'fixed_inputs':fixed})


def prepare(args):
    preflight_record=json.loads((Path(args.root)/'preflight.json').read_text())
    for f in preflight_record['fixed_inputs'].values():
        if sha(f['path'])!=f['sha256']: raise ValueError('fixed production input changed after preflight: '+f['path'])
    p=periods(args.start_year,args.end_year)[args.task]
    folder=Path(args.root)/'stations'/p['label']; folder.mkdir(parents=True,exist_ok=True)
    run('99_prepare_production_stations.py','--start',p['start'],'--end',p['end'],
        '--bmd-wide',args.bmd_wide,'--bmd-stations',args.bmd_stations,'--bwdb-xlsx',args.bwdb,
        '--bmd-catalog-only' if getattr(args,'bmd_catalog_only',True) else '--no-bmd-catalog-only',
        '--out',folder/'combined_daily.csv','--summary',folder/'station_summary.csv',
        '--report',folder/'preparation_manifest.json')
    run('87_superob_dense_gauges.py','--stations',folder/'combined_daily.csv','--stats',args.stats,
        '--cell-deg',.25,'--out',folder/'superob_prod_0.25.csv','--report',folder/'superob_prod_0.25.json')
    native=Path(args.root)/'imerg_native'/f"{p['label']}.nc"
    coarse=Path(args.root)/'imerg_s04'/f"{p['label']}.nc"
    native.parent.mkdir(parents=True,exist_ok=True); coarse.parent.mkdir(parents=True,exist_ok=True)
    # Pass exactly the overlapping months, not every file in the archive.
    inputs=[Path(args.imerg_daily)/f"imerg_bd_aligned_{m['start'].replace('-','')}_{m['end'].replace('-','')}.nc"
            for m in monthly(args.start_year,args.end_year) if p['start']<=m['start']<=p['end']]
    run('43_subset_prepared_imerg.py','--input',*inputs,'--start',p['start'],'--end',p['end'],
        '--out',native,'--report',native.with_name(native.stem+'_qc.json'))
    run('44_coarsen_imerg_observations.py','--input',native,'--factor',8,'--out',coarse,
        '--report',coarse.with_name(coarse.stem+'_qc.json'))
    validate_imerg(coarse,p['start'],p['end'],8)
    files=[folder/'combined_daily.csv',folder/'preparation_manifest.json',folder/'superob_prod_0.25.csv',
           folder/'superob_prod_0.25.json',coarse,Path(args.stats)]
    write_json(Path(args.root)/'prepared'/f"{p['label']}.json",{'status':'validated_preparation','period':p,
               'statistics_sha256':sha(args.stats),'files':[{'path':str(f),'sha256':sha(f)} for f in files]})


def production(args):
    p=periods(args.start_year,args.end_year)[args.task]
    prepared=json.loads((Path(args.root)/'prepared'/f"{p['label']}.json").read_text())
    for f in prepared['files']:
        if sha(f['path'])!=f['sha256']: raise ValueError('prepared input changed: '+f['path'])
    if prepared['statistics_sha256']!=STATS_SHA or sha(args.ckpt)!=CHECKPOINT_SHA:
        raise ValueError('frozen model/statistics identity differs')
    predictor=json.loads((Path(args.root)/'predictor_validation.json').read_text())
    data_zarr=predictor['predictors']['path']
    prefix=Path(args.root)/'production_metadata'/p['label']; prefix.parent.mkdir(parents=True,exist_ok=True)
    fields=Path(args.root)/'gridded'/f"{p['label']}.zarr"
    paths=[prefix.with_suffix('.json'),prefix.with_suffix('.npz'),fields,Path(str(fields)+'.incomplete')]
    if all(f.exists() for f in paths[:3]):
        verify_shard(args,p); print('[production] verified existing '+p['label']); return
    if any(f.exists() for f in paths):
        raise ValueError('partial production output; preserve/inspect before retrying: '+p['label'])
    write_json(Path(args.root)/'production_identity'/f"{p['label']}.json",{
        'checkpoint_sha256':sha(args.ckpt),'statistics_sha256':sha(args.stats),
        'prepared_manifest_sha256':sha(Path(args.root)/'prepared'/f"{p['label']}.json"),
        'predictor_validation_sha256':sha(Path(args.root)/'predictor_validation.json')})
    folder=Path(args.root)/'stations'/p['label']
    report=json.loads((folder/'superob_prod_0.25.json').read_text())
    representation=report['recommended_representativeness']['superob_implied_representativeness']
    if representation is None or not 0<=float(representation)<10:
        raise ValueError('unavailable measured super-observation representativeness')
    run('51_check_sqrt_da_gradient.py')
    run('28_simultaneous_method_sweep.py','--config','configs/da.yaml','--ckpt',args.ckpt,'--data-zarr',data_zarr,
        '--stations',folder/'superob_prod_0.25.csv','--imerg',Path(args.root)/'imerg_s04'/f"{p['label']}.nc",
        '--start',p['start'],'--end',p['end'],'--background-day-offset',-1,'--members',30,
        '--holdout-folds',1,'--holdout-fold',0,'--group','v2_bmd_bwdb_superob_winner',
        '--set','observations.imerg.factor=8','--set','observations.imerg.error_corr_cells=0.75',
        '--set',f'observations.gauges.representativeness={representation}',
        '--seed',202205,'--out',prefix.with_suffix('.npz'),'--report',prefix.with_suffix('.json'),
        '--assimilate-all-stations','--fields-zarr',fields)
    verify_shard(args,p)


def verify_shard(args,p):
    import numpy as np
    import zarr
    prefix=Path(args.root)/'production_metadata'/p['label']
    identity=json.loads((Path(args.root)/'production_identity'/f"{p['label']}.json").read_text())
    if (identity['checkpoint_sha256']!=CHECKPOINT_SHA or identity['statistics_sha256']!=STATS_SHA
            or identity['prepared_manifest_sha256']!=sha(Path(args.root)/'prepared'/f"{p['label']}.json")
            or identity['predictor_validation_sha256']!=sha(Path(args.root)/'predictor_validation.json')):
        raise ValueError('production input provenance differs: '+p['label'])
    report=json.loads(prefix.with_suffix('.json').read_text()); scope=report['scope']
    if (scope['start']!=p['start'] or scope['end']!=p['end'] or scope['members']!=30
            or scope['assimilate_all_stations'] is not True or scope['n_withheld_stations']!=0
            or scope['background_day_offset']!=-1 or scope['group']!='v2_bmd_bwdb_superob_winner'
            or scope['analysis_sampler_n_steps']!=50 or scope['analysis_sampler_n_corrections']!=2
            or scope['analysis_sampler_heun'] is not True or scope['seed']!=202205):
        raise ValueError('production scope differs: '+p['label'])
    predictor=json.loads((Path(args.root)/'predictor_validation.json').read_text())
    if scope['checkpoint_data']!=predictor['predictors']['path'] or scope['checkpoint_stats']!=args.stats:
        raise ValueError('production used different predictor/statistics paths: '+p['label'])
    store=zarr.open_group(str(Path(args.root)/'gridded'/f"{p['label']}.zarr"),mode='r')
    if store.attrs.get('complete') is not True or store.attrs.get('schema')!='bdhires.physical_ensemble.v1':
        raise ValueError('incomplete production store: '+p['label'])
    actual=(np.asarray(store['time'][:]).astype('timedelta64[D]')+np.datetime64('1970-01-01')).astype(str).tolist()
    if actual!=dates(p['start'],p['end']): raise ValueError('production dates differ: '+p['label'])
    methods=np.asarray(store['method'][:]).astype(str).tolist()
    if set(methods)!={'background',FINAL} or store['precipitation'].shape!=(2,len(actual),30,128,128):
        raise ValueError('production fields/member dimensions differ: '+p['label'])
    valid=np.asarray(store['valid'][:])>.5
    for index in range(len(actual)):
        if not np.isfinite(np.asarray(store['precipitation'][:,index])[:,:,valid]).all():
            raise ValueError('production contains non-finite land rainfall: '+actual[index])
    spec=store.attrs['method_specs'][FINAL]
    contract=json.loads((ROOT/'configs/paper1_cpcv2_final.json').read_text())
    expected=contract['profiles']['superob-final']['expected_spec']
    if any(spec.get(k)!=v for k,v in expected.items()):
        raise ValueError('analysis method differs from evaluated profile: '+p['label'])
    if store.attrs['scope']!=scope:
        raise ValueError('field and station report scopes differ: '+p['label'])
    with np.load(prefix.with_suffix('.npz'),allow_pickle=False) as dump:
        if dump['times'].astype('datetime64[D]').astype(str).tolist()!=actual or len(dump['eval_idx'])!=0:
            raise ValueError('station archive does not establish full all-station production')
        expected_background=(np.asarray(actual,dtype='datetime64[D]')-np.timedelta64(1,'D')).astype(str).tolist()
        if (dump['model_times'].astype('datetime64[D]').astype(str).tolist()!=expected_background
                or set(dump['assim_idx'].tolist())!=set(range(len(dump['station_ids'])))):
            raise ValueError('station archive background dates or assimilation membership differ')
    write_json(Path(args.root)/'validated'/f"{p['label']}.json",{'status':'complete','period':p,
               'checkpoint_sha256':sha(args.ckpt),'report_sha256':sha(prefix.with_suffix('.json')),
               'station_array_sha256':sha(prefix.with_suffix('.npz')),
               'prepared_manifest_sha256':sha(Path(args.root)/'prepared'/f"{p['label']}.json"),
               'field_store':str(Path(args.root)/'gridded'/f"{p['label']}.zarr"),'days':len(actual)})


def finalize(args):
    records=[]
    for p in periods(args.start_year,args.end_year):
        verify_shard(args,p)
        records.append(json.loads((Path(args.root)/'validated'/f"{p['label']}.json").read_text()))
    write_json(Path(args.root)/'production_manifest.json',{'status':'complete','model':'SURMA-Flow v1.0 CPCv2',
        'start':f'{args.start_year}-01-01','end':f'{args.end_year}-12-31','days':sum(r['days'] for r in records),
        'members':30,'checkpoint_sha256':CHECKPOINT_SHA,'statistics_sha256':STATS_SHA,'shards':records})


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['source-check','audit','audit-gauges','preflight','choose-data','download-year','download-month','pack','prepare','production','finalize'])
    p.add_argument('--start-year',type=int,default=2001);p.add_argument('--end-year',type=int)
    p.add_argument('--root',default='data/processed/brishti05_production_2001_2024')
    p.add_argument('--data-zarr'); p.add_argument('--ckpt',default='runs/prior_h100_cpc_v2/best.pt')
    p.add_argument('--stats',default='data/processed/stats_cpc_v2.json')
    p.add_argument('--bmd-wide',default='data/stations/Rainfall_daily_by_station_BMD_corrected.csv')
    p.add_argument('--bmd-stations',default='data/stations/BMD_production_station_catalog.csv')
    p.add_argument('--bmd-catalog-only',action=argparse.BooleanOptionalAction,default=True,
                   help='use only reviewed catalogue coordinates (default); legacy fallback requires explicit opt-in')
    p.add_argument('--bwdb',default='data/stations/BWDB_Rainfall_2000_2025_corrected.xlsx')
    p.add_argument('--era5',default='data/raw/era5');p.add_argument('--cpc',default='data/raw/cpc')
    p.add_argument('--chirps',default='data/raw/chirps');p.add_argument('--static',default='data/static/static_wide.nc')
    p.add_argument('--imerg-raw',default='data/imerg_halfhourly');p.add_argument('--imerg-daily',default='data/processed')
    p.add_argument('--imerg-state',default=os.environ.get('IMERG_DOWNLOAD_STATE','data/processed/imerg_download_2001_2024'),
                   help='year download reports used by source-check')
    p.add_argument('--color',choices=['auto','always','never'],default='auto',
                   help='source-check colors (auto uses terminal detection; NO_COLOR disables auto)')
    p.add_argument('--task',type=int);p.add_argument('--deep',action='store_true');p.add_argument('--report')
    args=p.parse_args(argv)
    if args.end_year is None: args.end_year=2025 if args.stage=='source-check' else 2024
    last_allowed=2025 if args.stage=='source-check' else 2024
    if not 2001<=args.start_year<=args.end_year<=last_allowed:
        p.error(f'{args.stage} years must be within 2001..{last_allowed}')
    limits={'download-year':args.end_year-args.start_year+2,'download-month':12*(args.end_year-args.start_year+1),
            'prepare':4*(args.end_year-args.start_year+1),'production':4*(args.end_year-args.start_year+1)}
    if args.stage in limits and (args.task is None or not 0<=args.task<limits[args.stage]):
        p.error('missing/out-of-range --task for '+args.stage)
    if args.stage=='choose-data': print(choose_data(args)); return
    globals()[args.stage.replace('-','_')](args)


if __name__=='__main__':
    try: main()
    except (ValueError,OSError,KeyError,ImportError,subprocess.CalledProcessError) as error:
        print('[production] failed: '+str(error),file=sys.stderr);raise SystemExit(1)
