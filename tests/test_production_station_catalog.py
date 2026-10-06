"""Reviewed gauge coordinates must not be undone by legacy fallbacks."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('_production_station_catalog', ROOT/'scripts/99_prepare_production_stations.py')
PREP = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PREP)
spec = importlib.util.spec_from_file_location('_production_catalog_driver', ROOT/'scripts/100_surma_production.py')
PROD = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PROD)


def test_reviewed_catalog_keeps_corrected_dates_and_blocks_legacy_tetulia(tmp_path):
    catalog = tmp_path/'catalog.csv'
    catalog.write_text('StationNumber,Station,Latitude,Longitude\n37,Rajarhat,25.80,89.55\n43,Aricha,23.84,89.78\n')
    rainfall = tmp_path/'corrected.csv'
    rainfall.write_text('Date,Rajarhat,Aricha,Teltulia\n2024-01-02,10,20,30\n2024-01-03,11,21,31\n')
    args = SimpleNamespace(bmd_wide=rainfall,bmd_stations=catalog,bmd_catalog_only=True)
    frame, qc = PREP.read_bmd(args,pd.Timestamp('2024-01-02'),pd.Timestamp('2024-01-03'))
    assert set(frame.station_id)=={37,43}
    assert frame.loc[frame.station_id==37,'lon'].tolist()==[89.55,89.55]
    assert frame.loc[frame.station_id==43,'date'].dt.strftime('%Y-%m-%d').tolist()==['2024-01-02','2024-01-03']
    assert frame.loc[frame.station_id==43,'precip_mm'].tolist()==[20,21]
    assert qc[0]['wide']['columns_unmatched_skipped']==['Teltulia']
    assert qc[0]['wide']['coordinate_policy']=='catalogue only'
    legacy, _ = PREP.read_wide_bmd(rainfall,catalog,pd.Timestamp('2024-01-02'),pd.Timestamp('2024-01-03'))
    assert 42 in set(legacy.station_id)


def test_production_passes_catalog_policy_to_quarterly_preparation(tmp_path):
    PROD.write_json(tmp_path/'preflight.json',{'fixed_inputs':{}})
    args=SimpleNamespace(start_year=2024,end_year=2024,task=0,root=tmp_path,
                         bmd_wide='corrected.csv',bmd_stations='reviewed.csv',bwdb='bwdb.xlsx')
    with patch.object(PROD,'run',side_effect=RuntimeError('captured')) as run:
        try:
            PROD.prepare(args)
        except RuntimeError as error:
            assert str(error)=='captured'
    command=run.call_args.args
    assert '--bmd-catalog-only' in command
    assert 'corrected.csv' in command and 'reviewed.csv' in command


def test_production_defaults_to_corrected_files_and_catalog_only():
    with patch.object(PROD,'audit') as audit:
        PROD.main(['audit'])
    args=audit.call_args.args[0]
    assert args.bmd_wide=='data/stations/Rainfall_daily_by_station_BMD_corrected.csv'
    assert args.bmd_stations=='data/stations/BMD_production_station_catalog.csv'
    assert args.bmd_catalog_only is True
