from pathlib import Path

import numpy as np

from bdhires.bmd import read_legacy_bmd, spread_folds, spread_holdout


def test_legacy_bmd_conversion_handles_report_header_aliases_and_missing(tmp_path: Path):
    stations = tmp_path / "Stations.csv"
    stations.write_text(
        "StationNumber,Station,StationId,Latitude,Longitude\n"
        "1,Chittagonj,0,22.27,91.82\n"
        "2,Pauakhali,0,22.33,90.33\n"
    )
    rainfall = tmp_path / "bmd.csv"
    prefix = "\n".join(["," * 33] * 7)
    days = ",".join(str(day) for day in range(1, 32))
    row1 = ["Chittagong", "2018", "2"] + ["1", "***", "3"] + ["0"] * 25 + [""] * 3
    row2 = ["Patuakhali", "2018", "2"] + ["0"] * 28 + [""] * 3
    rainfall.write_text(
        prefix + "\n" + f"Stati,Year,Month,{days}\n" + ",".join(row1) + "\n" + ",".join(row2) + "\n"
    )

    daily, report = read_legacy_bmd(
        rainfall, stations, start="2018-02-01", end="2018-02-28"
    )

    assert len(daily) == 56
    assert daily["station_id"].nunique() == 2
    assert daily["date"].max().strftime("%Y-%m-%d") == "2018-02-28"
    assert np.isnan(daily.loc[daily["name"] == "Chittagong", "precip_mm"].iloc[1])
    assert report["valid_observations"] == 55
    assert report["missing_observations"] == 1
    assert len(report["station_aliases"]) == 2


def test_spread_holdout_is_deterministic_and_separated():
    lat = np.array([0.0, 0.0, 1.0, 1.0, 0.5])
    lon = np.array([0.0, 1.0, 0.0, 1.0, 0.5])
    selected = spread_holdout(lat, lon, 4)

    assert selected.tolist() == spread_holdout(lat, lon, 4).tolist()
    assert len(np.unique(selected)) == 4
    assert 4 not in selected


def test_spread_folds_are_deterministic_balanced_and_exhaustive():
    lat = np.repeat(np.arange(5, dtype=float), 6)
    lon = np.tile(np.arange(6, dtype=float), 5)

    folds = spread_folds(lat, lon, n_splits=5)

    assert [fold.tolist() for fold in folds] == [
        fold.tolist() for fold in spread_folds(lat, lon, n_splits=5)
    ]
    assert [len(fold) for fold in folds] == [6] * 5
    assert np.array_equal(np.sort(np.concatenate(folds)), np.arange(30))
    assert sum(len(np.unique(fold)) for fold in folds) == 30
