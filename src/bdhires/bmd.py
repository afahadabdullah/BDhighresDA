"""Readers and reproducible splits for legacy BMD daily rainfall files.

The historical BMD archive used by this project is not a conventional table:
each row is one station-month and columns ``1`` through ``31`` contain daily
totals.  Seven report-title rows precede the header, ``***`` means missing,
and the station spellings differ slightly from the companion coordinate file.
This module converts that source into the repository's canonical long form
without changing the original measurements.
"""

from __future__ import annotations

import calendar
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd


MISSING_TOKENS = {"", "***", "NA", "N/A", "NULL", "NONE"}
SENTINELS = {-999.0, -9999.0, -99.9, 999.0, 9999.0}

# Explicit aliases are safer than fuzzy matching scientific station records.
# Keys and values use ``_station_key`` normalization.
STATION_ALIASES = {
    "chittagong": "chittagonj",
    "patuakhali": "pauakhali",
    "sydpur": "syedpur",
}


def _station_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


def _header_row(path: str | Path) -> int:
    """Return the zero-based row containing ``Stati,Year,Month,...``."""
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for index, row in enumerate(csv.reader(handle)):
            if len(row) >= 3 and [cell.strip().lower() for cell in row[:3]] == [
                "stati",
                "year",
                "month",
            ]:
                return index
    raise ValueError(f"could not find the BMD header row in {path}")


def read_station_catalog(path: str | Path) -> pd.DataFrame:
    """Read the companion BMD station catalogue with stable station numbers."""
    frame = pd.read_csv(path)
    columns = {str(column).strip().lower(): column for column in frame.columns}
    required = {"stationnumber", "station", "latitude", "longitude"}
    missing = required - set(columns)
    if missing:
        raise ValueError(f"{path} is missing catalogue columns: {sorted(missing)}")

    out = pd.DataFrame(
        {
            "station_id": pd.to_numeric(frame[columns["stationnumber"]], errors="raise"),
            "catalog_name": frame[columns["station"]].astype(str).str.strip(),
            "lat": pd.to_numeric(frame[columns["latitude"]], errors="raise"),
            "lon": pd.to_numeric(frame[columns["longitude"]], errors="raise"),
        }
    )
    if out["station_id"].duplicated().any():
        duplicate = out.loc[out["station_id"].duplicated(), "station_id"].tolist()
        raise ValueError(f"duplicate StationNumber values in {path}: {duplicate}")
    if out[["lat", "lon"]].duplicated().any():
        raise ValueError(f"duplicate station coordinates in {path}")
    out["station_id"] = out["station_id"].astype(int)
    out["station_key"] = out["catalog_name"].map(_station_key)
    if out["station_key"].duplicated().any():
        duplicate = out.loc[out["station_key"].duplicated(), "catalog_name"].tolist()
        raise ValueError(f"duplicate normalized station names in {path}: {duplicate}")
    return out


def read_legacy_bmd(
    rainfall_csv: str | Path,
    station_csv: str | Path,
    start: str | np.datetime64 | None = None,
    end: str | np.datetime64 | None = None,
    max_mm: float = 1000.0,
) -> tuple[pd.DataFrame, dict]:
    """Convert the legacy station-month matrix to canonical daily long form.

    Returns a dataframe with columns
    ``station_id,name,lat,lon,date,precip_mm`` and an auditable QC report.
    Missing observations are retained as NaN on valid calendar days.
    """
    header = _header_row(rainfall_csv)
    raw = pd.read_csv(rainfall_csv, skiprows=header, dtype=str, keep_default_na=False)
    raw.columns = [str(column).strip() for column in raw.columns]
    required = {"Stati", "Year", "Month"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"{rainfall_csv} is missing columns: {sorted(missing)}")

    day_columns = [str(day) for day in range(1, 32)]
    missing_days = set(day_columns) - set(raw.columns)
    if missing_days:
        raise ValueError(f"{rainfall_csv} is missing day columns: {sorted(missing_days)}")

    raw["station_source_name"] = raw["Stati"].astype(str).str.strip()
    raw["year"] = pd.to_numeric(raw["Year"], errors="raise").astype(int)
    raw["month"] = pd.to_numeric(raw["Month"], errors="raise").astype(int)
    long = raw.melt(
        id_vars=["station_source_name", "year", "month"],
        value_vars=day_columns,
        var_name="day",
        value_name="precip_source",
    )
    long["day"] = pd.to_numeric(long["day"], errors="raise").astype(int)
    long["date"] = pd.to_datetime(
        {"year": long["year"], "month": long["month"], "day": long["day"]},
        errors="coerce",
    )
    invalid_calendar = int(long["date"].isna().sum())
    long = long[long["date"].notna()].copy()

    source = long["precip_source"].astype(str).str.strip()
    token_upper = source.str.upper()
    known_missing = token_upper.isin(MISSING_TOKENS)
    numeric = pd.to_numeric(source.where(~known_missing), errors="coerce")
    unknown_tokens = sorted(source[~known_missing & numeric.isna()].unique().tolist())
    if unknown_tokens:
        raise ValueError(
            f"unrecognized precipitation tokens in {rainfall_csv}: {unknown_tokens[:20]}"
        )
    sentinel = numeric.isin(SENTINELS)
    negative = numeric < 0
    too_large = numeric > max_mm
    long["precip_mm"] = numeric.mask(sentinel | negative | too_large)

    catalog = read_station_catalog(station_csv)
    long["station_key_source"] = long["station_source_name"].map(_station_key)
    long["station_key"] = long["station_key_source"].replace(STATION_ALIASES)
    merged = long.merge(catalog, on="station_key", how="left", validate="many_to_one")
    unmatched = sorted(merged.loc[merged["station_id"].isna(), "station_source_name"].unique())
    if unmatched:
        raise ValueError(f"BMD stations without coordinates: {unmatched}")

    merged["station_id"] = merged["station_id"].astype(int)
    merged["name"] = merged["station_source_name"]
    if start is not None:
        merged = merged[merged["date"] >= pd.Timestamp(start)]
    if end is not None:
        merged = merged[merged["date"] <= pd.Timestamp(end)]
    if merged.empty:
        raise ValueError(f"no BMD observations remain in requested period {start} to {end}")
    if merged.duplicated(["station_id", "date"]).any():
        duplicate = merged.loc[
            merged.duplicated(["station_id", "date"], keep=False), ["station_id", "date"]
        ].head(20)
        raise ValueError(f"duplicate BMD station-days:\n{duplicate.to_string(index=False)}")

    out = merged[["station_id", "name", "lat", "lon", "date", "precip_mm"]].copy()
    out = out.sort_values(["station_id", "date"]).reset_index(drop=True)
    finite = out["precip_mm"].notna()
    aliases = (
        merged.loc[
            merged["station_key_source"] != merged["station_key"],
            ["station_source_name", "catalog_name"],
        ]
        .drop_duplicates()
        .sort_values("station_source_name")
    )
    report = {
        "rainfall_source": str(rainfall_csv),
        "station_source": str(station_csv),
        "header_row_1based": header + 1,
        "requested_start": None if start is None else str(pd.Timestamp(start).date()),
        "requested_end": None if end is None else str(pd.Timestamp(end).date()),
        "date_start": str(out["date"].min().date()),
        "date_end": str(out["date"].max().date()),
        "stations": int(out["station_id"].nunique()),
        "calendar_station_days": int(len(out)),
        "valid_observations": int(finite.sum()),
        "missing_observations": int((~finite).sum()),
        "coverage": float(finite.mean()),
        "wet_fraction_of_valid": float((out.loc[finite, "precip_mm"] > 0).mean()),
        "mean_mm_per_valid_day": float(out.loc[finite, "precip_mm"].mean()),
        "maximum_mm": float(out.loc[finite, "precip_mm"].max()),
        "invalid_calendar_cells_dropped": invalid_calendar,
        "sentinel_values_removed": int(sentinel.sum()),
        "negative_values_removed": int(negative.sum()),
        "values_over_max_removed": int(too_large.sum()),
        "max_mm_threshold": float(max_mm),
        "station_aliases": aliases.to_dict(orient="records"),
        "scope_note": (
            "Coverage is over station-days present in the legacy station-month rows; "
            "stations whose archive ended before the requested month are absent."
        ),
    }
    return out, report


EXTRA_STATION_COORDS = {
    "dimla": {"catalog_name": "Dimla", "lat": 26.12833, "lon": 88.92500, "station_id": 36},
    "rajarhat": {"catalog_name": "Rajarhat", "lat": 25.80500, "lon": 89.66833, "station_id": 37},
    "gopalgonj": {"catalog_name": "Gopalgonj", "lat": 23.00722, "lon": 89.83333, "station_id": 38},
    "natrakona": {"catalog_name": "Natrakona", "lat": 24.87750, "lon": 90.72750, "station_id": 39},
    "nikli": {"catalog_name": "Nikli", "lat": 24.31667, "lon": 90.91667, "station_id": 40},
    "tarash": {"catalog_name": "Tarash", "lat": 24.43333, "lon": 89.36667, "station_id": 41},
    "tetulia": {"catalog_name": "Tetulia", "lat": 26.58333, "lon": 88.55000, "station_id": 42},
}


def read_station_dir_bmd(
    data_dir: str | Path,
    catalog_csv: str | Path,
    start: str | np.datetime64 | None = None,
    end: str | np.datetime64 | None = None,
    max_mm: float = 1000.0,
) -> tuple[pd.DataFrame, dict]:
    """Convert per-station CSV files in a directory to canonical daily long form.

    Expects a folder of CSV files where each file contains daily station data
    (e.g., columns ``Datetime``/``date`` and ``Rainfall``/``precip_mm``) and a
    catalog CSV (e.g., ``Stations.csv``).

    Returns a dataframe with columns ``station_id,name,lat,lon,date,precip_mm``
    and an auditable QC report.
    """
    data_dir = Path(data_dir)
    catalog_csv = Path(catalog_csv)
    if not data_dir.is_dir():
        raise ValueError(f"data directory does not exist: {data_dir}")
    if not catalog_csv.is_file():
        raise ValueError(f"catalog CSV does not exist: {catalog_csv}")

    catalog = read_station_catalog(catalog_csv)

    csv_files = [
        path for path in data_dir.glob("*.csv")
        if path.resolve() != catalog_csv.resolve() and not path.name.lower().endswith("stations.csv")
    ]
    if not csv_files:
        raise ValueError(f"no station CSV files found in {data_dir}")

    catalog_map = {row["station_key"]: row for _, row in catalog.iterrows()}

    records = []
    invalid_calendar = 0
    sentinels_count = 0
    negative_count = 0
    too_large_count = 0
    unmatched_files = []

    for fpath in sorted(csv_files):
        fname_base = fpath.stem
        key_source = _station_key(fname_base)
        key = STATION_ALIASES.get(key_source, key_source)

        st_info = catalog_map.get(key)
        if st_info is None:
            for cat_key, row in catalog_map.items():
                if key in cat_key or cat_key in key:
                    st_info = row
                    break
        if st_info is None and key in EXTRA_STATION_COORDS:
            st_info = pd.Series(EXTRA_STATION_COORDS[key])
        if st_info is None:
            unmatched_files.append(fpath.name)
            continue


        raw = pd.read_csv(fpath, dtype=str, keep_default_na=False)
        cols = {str(c).strip().lower(): c for c in raw.columns}

        date_col = cols.get("datetime") or cols.get("date")
        val_col = cols.get("rainfall") or cols.get("precip_mm") or cols.get("rain")
        if not date_col or not val_col:
            continue

        df = pd.DataFrame()
        df["date"] = pd.to_datetime(raw[date_col].str.strip(), errors="coerce")
        bad_dates = int(df["date"].isna().sum())
        invalid_calendar += bad_dates
        df = df[df["date"].notna()].copy()

        source_vals = raw.loc[df.index, val_col].astype(str).str.strip()
        token_upper = source_vals.str.upper()
        known_missing = token_upper.isin(MISSING_TOKENS)
        numeric = pd.to_numeric(source_vals.where(~known_missing), errors="coerce")

        sentinel = numeric.isin(SENTINELS)
        negative = numeric < 0
        too_large = numeric > max_mm

        sentinels_count += int(sentinel.sum())
        negative_count += int(negative.sum())
        too_large_count += int(too_large.sum())

        df["precip_mm"] = numeric.mask(sentinel | negative | too_large)
        df["station_id"] = int(st_info["station_id"])
        df["name"] = str(st_info["catalog_name"])
        df["lat"] = float(st_info["lat"])
        df["lon"] = float(st_info["lon"])

        records.append(df)

    if not records:
        raise ValueError(f"no matching station data parsed from {data_dir}")

    merged = pd.concat(records, ignore_index=True)

    if start is not None:
        merged = merged[merged["date"] >= pd.Timestamp(start)]
    if end is not None:
        merged = merged[merged["date"] <= pd.Timestamp(end)]
    if merged.empty:
        raise ValueError(f"no BMD observations remain in requested period {start} to {end}")

    if merged.duplicated(["station_id", "date"]).any():
        duplicate = merged.loc[
            merged.duplicated(["station_id", "date"], keep=False), ["station_id", "date"]
        ].head(20)
        raise ValueError(f"duplicate BMD station-days:\n{duplicate.to_string(index=False)}")

    out = merged[["station_id", "name", "lat", "lon", "date", "precip_mm"]].copy()
    out = out.sort_values(["station_id", "date"]).reset_index(drop=True)
    finite = out["precip_mm"].notna()

    report = {
        "data_dir": str(data_dir),
        "station_source": str(catalog_csv),
        "requested_start": None if start is None else str(pd.Timestamp(start).date()),
        "requested_end": None if end is None else str(pd.Timestamp(end).date()),
        "date_start": str(out["date"].min().date()),
        "date_end": str(out["date"].max().date()),
        "stations": int(out["station_id"].nunique()),
        "calendar_station_days": int(len(out)),
        "valid_observations": int(finite.sum()),
        "missing_observations": int((~finite).sum()),
        "coverage": float(finite.mean()),
        "wet_fraction_of_valid": float((out.loc[finite, "precip_mm"] > 0).mean()),
        "mean_mm_per_valid_day": float(out.loc[finite, "precip_mm"].mean()),
        "maximum_mm": float(out.loc[finite, "precip_mm"].max()),
        "invalid_calendar_cells_dropped": invalid_calendar,
        "sentinel_values_removed": sentinels_count,
        "negative_values_removed": negative_count,
        "values_over_max_removed": too_large_count,
        "max_mm_threshold": float(max_mm),
        "unmatched_files": unmatched_files,
    }
    return out, report



def summarize_daily(frame: pd.DataFrame) -> pd.DataFrame:
    """Return per-station availability and rainfall summaries."""
    rows = []
    for station_id, group in frame.groupby("station_id", sort=True):
        values = group["precip_mm"]
        finite = values.notna()
        valid = values[finite]
        rows.append(
            {
                "station_id": int(station_id),
                "name": str(group["name"].iloc[0]),
                "lat": float(group["lat"].iloc[0]),
                "lon": float(group["lon"].iloc[0]),
                "date_start": str(group["date"].min().date()),
                "date_end": str(group["date"].max().date()),
                "calendar_days": int(len(group)),
                "n_obs": int(finite.sum()),
                "coverage": float(finite.mean()),
                "wet_fraction": float((valid > 0).mean()) if len(valid) else np.nan,
                "mean_mm": float(valid.mean()) if len(valid) else np.nan,
                "p95_mm": float(valid.quantile(0.95)) if len(valid) else np.nan,
                "max_mm": float(valid.max()) if len(valid) else np.nan,
                "total_mm": float(valid.sum()) if len(valid) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def spread_holdout(lat: np.ndarray, lon: np.ndarray, count: int) -> np.ndarray:
    """Choose geographically spread withheld stations by farthest-point sampling."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if lat.shape != lon.shape or lat.ndim != 1:
        raise ValueError("lat and lon must be same-length one-dimensional arrays")
    if not 1 <= count < len(lat):
        raise ValueError("holdout count must be between 1 and n_stations - 1")
    x = lon * np.cos(np.radians(float(lat.mean())))
    points = np.column_stack([lat, x])
    centroid = points.mean(axis=0)
    selected = [int(np.argmax(np.sum((points - centroid) ** 2, axis=1)))]
    while len(selected) < count:
        distance = np.min(
            np.sum((points[:, None, :] - points[np.asarray(selected)][None, :, :]) ** 2, axis=2),
            axis=1,
        )
        distance[np.asarray(selected)] = -np.inf
        selected.append(int(np.argmax(distance)))
    return np.sort(np.asarray(selected, dtype=int))


def nearest_neighbour_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Great-circle distance from each station to its closest OTHER station.

    Used to describe how supportable a station is: a withheld gauge with a
    neighbour 20 km away can in principle be reconstructed from nearby
    observations, while one 120 km from anything can only be reconstructed from
    the prior. Scoring both together, without saying which is which, mixes a
    test of the assimilation with a test of the prior.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if lat.shape != lon.shape or lat.ndim != 1:
        raise ValueError("lat and lon must be same-length one-dimensional arrays")
    if lat.size < 2:
        return np.full(lat.size, np.inf)
    return _pairwise_km(lat, lon).min(axis=1)


def max_bearing_gap_deg(
    lat: np.ndarray, lon: np.ndarray, k: int = 6
) -> np.ndarray:
    """Largest angular gap between the bearings to a station's ``k`` neighbours.

    This detects stations at the EDGE of a network, which is a different defect
    from being far from everything. A station in the interior has neighbours in
    every direction, so its bearings are spread around the circle and the
    largest gap between consecutive ones is small. A station on the boundary has
    neighbours only on the inward side, so one gap approaches or exceeds
    180 degrees -- everything beyond the boundary is missing.

    That matters for verification because an interior gauge can be reconstructed
    by interpolation while a boundary gauge can only be reconstructed by
    extrapolation, which is a far harder problem and one the prior, not the
    assimilation, ends up answering.

    Returns degrees in ``[0, 360]``; ``360`` for a station with no neighbours.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    n = lat.size
    if n < 2:
        return np.full(n, 360.0)

    scale = np.cos(np.radians(float(lat.mean())))
    y = lat
    x = lon * scale
    out = np.empty(n, dtype=float)
    for i in range(n):
        dy = y - y[i]
        dx = x - x[i]
        distance = np.hypot(dy, dx)
        distance[i] = np.inf
        take = np.argsort(distance)[: min(k, n - 1)]
        bearings = np.sort(np.degrees(np.arctan2(dy[take], dx[take])) % 360.0)
        if bearings.size == 0:
            out[i] = 360.0
            continue
        # Circular gaps, including the wrap from the last bearing to the first.
        gaps = np.diff(bearings)
        wrap = 360.0 - (bearings[-1] - bearings[0])
        out[i] = float(max(gaps.max() if gaps.size else 0.0, wrap))
    return out


def neighbored_holdout(
    lat: np.ndarray,
    lon: np.ndarray,
    count: int,
    radius_km: float = 75.0,
    max_gap_deg: float = 200.0,
    neighbours: int = 6,
) -> np.ndarray:
    """Spread holdout restricted to stations that HAVE a nearby neighbour.

    ``spread_holdout`` seeds at the station farthest from the centroid and then
    samples farthest-point, so it selects the most ISOLATED and most PERIPHERAL
    stations by construction -- Sylhet, Teknaf, Tetulia and the like. That makes
    the verification set maximally adversarial for reasons that have nothing to
    do with the assimilation method.

    Two distinct defects are excluded here, because they fail differently:

    **Isolation.** A withheld gauge with no neighbour inside the background
    error correlation length cannot be reconstructed from observations at all,
    so its score measures the prior. Excluded by ``radius_km``.

    **Edge.** A gauge on the network boundary has neighbours only on the inward
    side, so reconstructing it is EXTRAPOLATION rather than interpolation --
    a harder problem, and again mostly a test of the prior. Excluded by
    ``max_gap_deg``, the largest angular gap between bearings to its
    ``neighbours`` nearest stations: an interior station has neighbours all
    around and a small maximum gap, a boundary station has one gap approaching
    or exceeding 180 degrees.

    The selection is then spread among the survivors, so the withheld set stays
    geographically distributed rather than clustered.

    HONESTY REQUIREMENT. This raises apparent skill, and reporting it alone
    would be cherry-picking. It is defensible only when described for what it
    is -- skill at stations the analysis could plausibly reconstruct -- and it
    should be reported ALONGSIDE the unrestricted holdout, not instead of it.
    Callers are expected to record each withheld station's nearest-neighbour
    distance so the score can be stratified after the fact.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if not 1 <= count < len(lat):
        raise ValueError("holdout count must be between 1 and n_stations - 1")

    nearest = nearest_neighbour_km(lat, lon)
    gap = max_bearing_gap_deg(lat, lon, k=neighbours)
    supported = nearest <= float(radius_km)
    interior = gap <= float(max_gap_deg)
    eligible = np.flatnonzero(supported & interior)
    if eligible.size < count:
        raise ValueError(
            f"only {eligible.size} of {lat.size} stations qualify "
            f"({int(supported.sum())} have a neighbour within {radius_km:g} km, "
            f"{int(interior.sum())} are interior at a {max_gap_deg:g} deg gap "
            f"threshold), which is fewer than the {count} to be withheld. "
            f"Lower the withheld fraction, raise --holdout-neighbor-km, or "
            f"relax --holdout-max-gap-deg. Do NOT fall back silently: "
            f"reintroducing isolated or peripheral stations is the exact bias "
            f"this function exists to remove."
        )

    # Farthest-point spread among the eligible subset only.
    x = lon * np.cos(np.radians(float(lat.mean())))
    points = np.column_stack([lat, x])[eligible]
    centroid = points.mean(axis=0)
    selected = [int(np.argmax(np.sum((points - centroid) ** 2, axis=1)))]
    while len(selected) < count:
        distance = np.min(
            np.sum((points[:, None, :] - points[np.asarray(selected)][None, :, :]) ** 2,
                   axis=2),
            axis=1,
        )
        distance[np.asarray(selected)] = -np.inf
        selected.append(int(np.argmax(distance)))
    chosen = eligible[np.asarray(selected, dtype=int)]

    # A station's support must come from gauges that REMAIN IN THE ANALYSIS.
    # Eligibility above was computed against all stations, so withholding a
    # cluster can strip the very neighbours that made its members eligible --
    # each supporting the next, all removed together. Repair by swapping the
    # worst-supported pick for the best-supported unused candidate until every
    # withheld station has an ASSIMILATED neighbour inside the radius.
    all_pairs = _pairwise_km(lat, lon)
    unused = [int(i) for i in eligible if i not in set(chosen.tolist())]
    for _ in range(len(eligible) + 1):
        held = set(int(i) for i in chosen)
        support = np.array([
            np.min([all_pairs[i, j] for j in range(lat.size) if j not in held])
            if lat.size - len(held) else np.inf
            for i in chosen
        ])
        worst = int(np.argmax(support))
        if support[worst] <= float(radius_km):
            break
        if not unused:
            raise ValueError(
                f"cannot withhold {count} stations and still leave every one of "
                f"them an assimilated neighbour within {radius_km:g} km. The "
                f"eligible pool ({eligible.size}) is too small relative to the "
                f"withheld count. Lower the withheld fraction or raise the "
                f"radius; a holdout whose members removed each other's support "
                f"is the bias this function exists to avoid."
            )
        replacement_scores = []
        for candidate in unused:
            trial = set(held) - {int(chosen[worst])} | {candidate}
            others = [j for j in range(lat.size) if j not in trial]
            replacement_scores.append(
                min(all_pairs[candidate, j] for j in others) if others else np.inf
            )
        best = int(np.argmin(replacement_scores))
        unused.append(int(chosen[worst]))
        chosen[worst] = unused.pop(best)
    return np.sort(chosen)


def _pairwise_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Full great-circle distance matrix, diagonal set to infinity."""
    phi = np.radians(np.asarray(lat, float))
    lam = np.radians(np.asarray(lon, float))
    dphi = phi[:, None] - phi[None, :]
    dlam = lam[:, None] - lam[None, :]
    a = (np.sin(dphi / 2.0) ** 2
         + np.cos(phi[:, None]) * np.cos(phi[None, :]) * np.sin(dlam / 2.0) ** 2)
    distance = 6371.0088 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    np.fill_diagonal(distance, np.inf)
    return distance


def spread_folds(
    lat: np.ndarray, lon: np.ndarray, n_splits: int = 5
) -> list[np.ndarray]:
    """Partition stations into deterministic, geographically spread folds.

    Stations are visited in a farthest-point ordering. Each next station is
    assigned to one of the currently smallest folds where it maximizes its
    distance from stations already assigned to that fold. Fold sizes therefore
    differ by at most one and every station is withheld exactly once.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if lat.shape != lon.shape or lat.ndim != 1:
        raise ValueError("lat and lon must be same-length one-dimensional arrays")
    if not 2 <= n_splits <= len(lat):
        raise ValueError("n_splits must be between 2 and n_stations")

    x = lon * np.cos(np.radians(float(lat.mean())))
    points = np.column_stack([lat, x])
    centroid = points.mean(axis=0)
    order = [int(np.argmax(np.sum((points - centroid) ** 2, axis=1)))]
    while len(order) < len(points):
        distance = np.min(
            np.sum(
                (points[:, None, :] - points[np.asarray(order)][None, :, :]) ** 2,
                axis=2,
            ),
            axis=1,
        )
        distance[np.asarray(order)] = -np.inf
        order.append(int(np.argmax(distance)))

    folds: list[list[int]] = [[] for _ in range(n_splits)]
    for station in order:
        smallest = min(len(fold) for fold in folds)
        candidates = [
            index for index, fold in enumerate(folds) if len(fold) == smallest
        ]
        scores = []
        for index in candidates:
            if not folds[index]:
                scores.append(float("inf"))
            else:
                delta = points[np.asarray(folds[index])] - points[station]
                scores.append(float(np.min(np.sum(delta**2, axis=1))))
        chosen = candidates[int(np.argmax(scores))]
        folds[chosen].append(station)

    return [np.sort(np.asarray(fold, dtype=int)) for fold in folds]


def month_lengths(year: int) -> dict[int, int]:
    """Small public helper used by audit scripts and tests."""
    return {month: calendar.monthrange(year, month)[1] for month in range(1, 13)}
