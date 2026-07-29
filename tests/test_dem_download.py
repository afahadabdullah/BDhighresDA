from __future__ import annotations

import importlib.util
from pathlib import Path


def load_downloader():
    path = Path(__file__).resolve().parents[1] / "scripts" / "03_download_dem.py"
    spec = importlib.util.spec_from_file_location("download_dem", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_copernicus_tile_names() -> None:
    module = load_downloader()
    assert module.tile_stem(23, 90) == "Copernicus_DSM_COG_30_N23_00_E090_00_DEM"
    assert module.tile_stem(-7, -12) == "Copernicus_DSM_COG_30_S07_00_W012_00_DEM"


def test_wide_candidate_tile_extent() -> None:
    module = load_downloader()
    tiles = module.candidate_tiles()
    stems = [stem for stem, _ in tiles]
    assert len(tiles) == 169
    assert stems[0] == "Copernicus_DSM_COG_30_N16_00_E084_00_DEM"
    assert stems[-1] == "Copernicus_DSM_COG_30_N28_00_E096_00_DEM"
