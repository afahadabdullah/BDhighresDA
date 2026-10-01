"""Whole-archive evidence, product intersections and dated inventory checks."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("updated_evidence",ROOT/"scripts/96_complete_updated_paper_evidence.py")
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)
PROFILE = {"selection_start":"2022-05-01","selection_end":"2022-05-31"}


def fixture():
    days = np.concatenate([np.arange("2021-05-01","2021-10-01",dtype="datetime64[D]"),
                           np.arange("2022-05-01","2022-10-01",dtype="datetime64[D]"),
                           np.arange("2023-05-01","2023-10-01",dtype="datetime64[D]"),
                           np.arange("2024-05-01","2024-07-01",dtype="datetime64[D]")])
    truth = (np.arange(len(days)*2)%100).astype(float)
    data = {"date":np.repeat(days,2),"station":np.tile(["BMD_A","BWDB_B"],len(days)),
            "source":np.tile(["BMD","BWDB"],len(days)),"truth":truth,
            "station_lat":np.full(len(truth),23.),"station_lon":np.full(len(truth),90.),
            "members":{"background":np.repeat((truth+2)[:,None],30,axis=1),
                       mod.FINAL:np.repeat((truth+.5)[:,None],30,axis=1)}}
    predictions = {source:truth+error for source,error in zip(mod.ORDER,(.5,2,3,4,5))}
    return data,predictions


class UpdatedEvidenceTests(unittest.TestCase):
    def test_all_four_years_and_every_month_use_shared_samples(self):
        data,predicted = fixture()
        kept,values,counts = mod.matched_samples(data,predicted,PROFILE)
        rows,station_rows,periods = mod.product_rows(kept,values,PROFILE,counts)
        self.assertEqual(counts["scored_dates"],489)
        self.assertEqual(counts["matched_daily_n"],978)
        self.assertEqual({r["label"] for r in rows if r["group"]=="year"},{"2021","2022","2023","2024"})
        self.assertEqual(len({r["label"] for r in rows if r["group"]=="month"}),16)
        for year,days in (("2021",153),("2022",122),("2023",153),("2024",61)):
            self.assertEqual({r["n"] for r in rows if r["group"]=="year" and r["label"]==year},{2*days})
        self.assertEqual({r["n"] for r in rows if r["group"]=="temporal" and r["label"]=="monthly"},{32})
        self.assertEqual({r["n"] for r in rows if r["group"]=="temporal" and r["label"]=="may_sep"},{4})
        available = [r for r in periods if r["scale"]=="available_wet_season" and r["period"]=="2024"]
        self.assertTrue(all(r["partial_season"] and r["eligible_days"]==61 for r in available))
        self.assertEqual(len(station_rows),10)

    def test_native_attrition_does_not_change_five_way_comparison(self):
        data,predicted = fixture()
        five,_,five_counts = mod.matched_samples(data,predicted,PROFILE)
        predicted["imerg_native"] = data["truth"]+1
        predicted["imerg_native"][0] = np.nan
        six,_,six_counts = mod.matched_samples(data,predicted,PROFILE)
        self.assertEqual(len(five["truth"])-len(six["truth"]),1)
        self.assertEqual(five_counts["daily_sample_attrition"],0)
        self.assertEqual(six_counts["daily_sample_attrition"],1)

    def test_complete_season_requires_complete_matched_dates(self):
        data,predicted = fixture()
        predicted["chirps"][0] = np.nan
        kept,values,counts = mod.matched_samples(data,predicted,PROFILE)
        rows,_,_ = mod.product_rows(kept,values,PROFILE,counts)
        self.assertEqual({r["n"] for r in rows if r["group"]=="temporal" and r["label"]=="may_sep"},{3})

    def test_partial_season_coverage_uses_archive_not_remaining_product_dates(self):
        data,predicted = fixture()
        archive_dates = np.unique(data["date"])
        unavailable = (data["date"] >= np.datetime64("2024-05-01")) & (data["date"] < np.datetime64("2024-05-15"))
        predicted["chirps"][unavailable] = np.nan
        kept,values,counts = mod.matched_samples(data,predicted,PROFILE)
        _,_,periods = mod.product_rows(kept,values,PROFILE,counts,archive_dates)
        # 47/61 < 80%; missing whole days must not shrink the denominator to 47.
        self.assertFalse(any(r["scale"]=="available_wet_season" and r["period"]=="2024" for r in periods))

    def test_bilinear_missing_values_are_not_dry_rainfall(self):
        field = np.array([[1.,np.nan],[3.,5.]])
        result = mod.sample_field(field,[0,1],[0,1],[0,.5,2],[0,.5,0])
        self.assertEqual(result[0],1.)
        self.assertTrue(np.isnan(result[1]) and np.isnan(result[2]))
        descending = np.array([[5.,3.],[np.nan,1.]])
        self.assertEqual(mod.sample_field(descending,[1,0],[1,0],[0],[0])[0],1.)

    def test_native_product_window_and_version_must_be_explicit(self):
        valid = {"version":"V07B","source_frequency":"half-hourly","bmd_accumulation_end_hour_utc":3,"window_duration_hours":24}
        mod.validate_native_metadata(valid)
        for key,value in (("version","V06"),("source_frequency","daily"),("bmd_accumulation_end_hour_utc",0),("window_duration_hours",48)):
            with self.assertRaisesRegex(ValueError,"native IMERG"):
                mod.validate_native_metadata({**valid,key:value})

    def test_paired_rmse_uses_pooled_squared_errors_and_gap_segments(self):
        data = {"date":np.array(["2021-05-01","2021-05-01","2021-05-02","2021-05-10"],dtype="datetime64[D]"),"truth":np.zeros(4)}
        predictions = {mod.FINAL:np.array([0.,4.,1.,3.]),"cpc":np.array([3.,5.,2.,1.])}
        rows = mod.paired_intervals(data,predictions,[3,7],200,7)
        expected = np.sqrt(np.mean(predictions["cpc"]**2))-np.sqrt(np.mean(predictions[mod.FINAL]**2))
        for row in rows:
            self.assertEqual(row["n_segments"],2)
            self.assertEqual(row["family_size"],6)
            if row["metric"]=="rmse":self.assertAlmostEqual(row["gain_mm_day"],expected)
            self.assertLessEqual(row["family_ci_low"],row["ci_low"])
            self.assertGreaterEqual(row["family_ci_high"],row["ci_high"])
        self.assertEqual(rows,mod.paired_intervals(data,predictions,[3,7],200,7))

    def test_inventory_identity_dates_proximity_and_unknown(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            mod.write_csv(folder/"inventory.csv",[
                {"station_id":"provider-A","local_station_id":"BMD_A","start":"2021-01-01","end":"2021-06-01","evidence":"provider crosswalk","lat":23,"lon":90},
                {"station_id":"provider-A","local_station_id":"BMD_A","start":"2021-06-02","end":"2021-12-31","evidence":"provider crosswalk","lat":23,"lon":90},
                {"station_id":"provider-B","local_station_id":"","start":"2021-01-01","end":"2021-12-31","evidence":"","lat":24,"lon":91}])
            config = folder/"inventories.json"
            config.write_text(json.dumps({"inventories":[{"product":"cpc","path":"inventory.csv","membership":"direct","reference":"provider-list"}]}))
            stations = [{"station_id":s,"lat":lat,"lon":lon,"dates":["2021-06-01","2021-06-02"]}
                        for s,lat,lon in (("BMD_A",23,90),("BWDB_B",24,91),("BMD_C",25,92))]
            rows,_ = mod.overlap_audit(stations,config)
            cpc = {r["local_station_id"]:r for r in rows if r["product"]=="cpc"}
            self.assertEqual(cpc["BMD_A"]["status"],"confirmed_overlap_on_dated_inventory")
            self.assertEqual(cpc["BMD_A"]["documented_station_day_matches"],2)
            self.assertEqual(cpc["BWDB_B"]["status"],"coordinate_candidate_only")
            self.assertEqual(cpc["BMD_C"]["status"],"unknown")
            self.assertTrue(all(r["status"]=="unknown" for r in rows if r["product"]=="imerg"))

    def test_audit_missing_inputs_never_advertises_old_scores(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp);output = folder/"out";output.mkdir()
            (output/"paired_product_scores.csv").write_text("stale prior run\n")
            self.assertEqual(mod.main(["--root",str(folder/"absent"),"--summary",str(folder/"no_summary"),"--out-dir",str(output)]),0)
            manifest = json.loads((output/"evidence_manifest.json").read_text())
            self.assertEqual(manifest["status"],"pending_inputs")
            self.assertNotIn("paired_product_scores.csv",{r["path"] for r in manifest["outputs"]})
            coverage = mod.read_csv(output/"test_date_coverage.csv")
            self.assertEqual(sum(r["configured_archive"]=="True" for r in coverage),520)
            self.assertEqual(sum(r["selection_date"]=="True" for r in coverage),31)

    def test_validated_archive_to_product_tables_without_production_zarr(self):
        fixture_spec = importlib.util.spec_from_file_location(
            "paper1_evaluation_fixture", ROOT/"tests/test_cpcv2_paper1.py")
        fixture_module = importlib.util.module_from_spec(fixture_spec)
        fixture_spec.loader.exec_module(fixture_module)
        PaperEvaluationTests = fixture_module.PaperEvaluationTests
        base = PaperEvaluationTests("test_common_samples_and_selection_exclusion");base.setUp()
        try:
            lat,lon = np.linspace(20,24,8),np.linspace(89,93,8)
            field = np.broadcast_to((np.arange(30)%10)[:,None,None],(30,8,8)).astype(float)
            base.dump.update(grid_lat=lat,grid_lon=lon,valid=np.ones((8,8),bool),chirps=field+2,raw_imerg_mm=(np.arange(30)%10+3)[:,None,None].astype(float))
            base.write_fixture()
            base.contract["evaluation_region"]["boundary_sha256"] = mod.digest(base.boundary)
            contract = base.root/"contract.json";contract.write_text(json.dumps(base.contract))
            original = mod.module;shared = original(55)
            fake = SimpleNamespace(upsample_coarse=shared.upsample_coarse,
                                   load_same_day_cpc=lambda archive,override:(field+4,base.root/"same_day_cpc.zarr"))
            with patch.object(mod,"module",side_effect=lambda number:fake if number==55 else original(number)):
                result = mod.main(["--run","--no-figures","--bootstrap","20","--root",str(base.root),
                                   "--contract",str(contract),"--boundary-geojson",str(base.boundary),
                                   "--summary",str(base.root/"no_summary"),"--out-dir",str(base.root/"out")])
            self.assertEqual(result,0)
            manifest = json.loads((base.root/"out/evidence_manifest.json").read_text())
            self.assertEqual(manifest["primary_comparison_counts"]["matched_daily_n"],60)
            self.assertEqual(manifest["status"],"matched_product_scores_complete")
            self.assertTrue((base.root/"out/tab_product_daily.tex").is_file())
            rows = mod.read_csv(base.root/"out/withheld_product_strata.csv")
            self.assertEqual({int(r["n"]) for r in rows if r["group"]=="pooled"},{60})
        finally:base.tearDown()


if __name__ == "__main__":unittest.main()
