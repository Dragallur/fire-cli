"""Fire danger indices from the Copernicus Emergency Management Service (CEMS).

Dataset: "Fire danger indices historical data from the Copernicus Emergency
Management Service" (``cems-fire-historical-v1``), derived from ERA5 reanalysis,
daily, global, 1940-present, hosted on the Early Warning Data Store (EWDS).

Before the first download, accept the dataset licence once at
https://ewds.climate.copernicus.eu/datasets/cems-fire-historical-v1?tab=download
"""

from __future__ import annotations

import logging
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Protocol

import requests

from fire_cli import config
from fire_cli.query import BBox, DateRange

log = logging.getLogger(__name__)

EWDS_URL = "https://ewds.climate.copernicus.eu/api"
DATASET = "cems-fire-historical-v1"
SYSTEM_VERSION = "4_1"


@dataclass(frozen=True)
class Index:
    name: str  # name in the EWDS request
    system: str
    description: str


# Short names used throughout fire-cli -> EWDS variable names.
INDICES: dict[str, Index] = {
    # Canadian Forest Fire Weather Index System
    "fwi": Index("fire_weather_index", "Canadian FWI", "Fire Weather Index"),
    "ffmc": Index("fine_fuel_moisture_code", "Canadian FWI", "Fine Fuel Moisture Code"),
    "dmc": Index("duff_moisture_code", "Canadian FWI", "Duff Moisture Code"),
    "dc": Index("drought_code", "Canadian FWI", "Drought Code"),
    "isi": Index("initial_fire_spread_index", "Canadian FWI", "Initial Spread Index"),
    "bui": Index("build_up_index", "Canadian FWI", "Build-Up Index"),
    "dsr": Index("fire_daily_severity_rating", "Canadian FWI", "Daily Severity Rating"),
    # Australian McArthur Mark 5
    "ffdi": Index("fire_danger_index", "McArthur Mark 5", "Forest Fire Danger Index"),
    "df": Index("drought_factor", "McArthur Mark 5", "Drought Factor"),
    "kbdi": Index("keetch_byram_drought_index", "McArthur Mark 5", "Keetch-Byram Drought Index"),
    # US National Fire Danger Rating System
    "bi": Index("burning_index", "US NFDRS", "Burning Index"),
    "erc": Index("energy_release_component", "US NFDRS", "Energy Release Component"),
    "ic": Index("ignition_component", "US NFDRS", "Ignition Component"),
    "sc": Index("spread_component", "US NFDRS", "Spread Component"),
}

GRIDS = {"0.25": "0.25/0.25", "0.5": "0.5/0.5"}


class RetrieveClient(Protocol):
    def retrieve(self, name: str, request: dict, target: str | None = None): ...


def make_client(key: str | None = None) -> RetrieveClient:
    import cdsapi

    # quiet=False keeps the queue status messages ("accepted", "running") in the log. The client
    # adds its own stream handler to loggers without one, which would print every message twice.
    for name in ("cdsapi", "ecmwf.datastores.legacy_client"):
        logger = logging.getLogger(name)
        if not logger.handlers:
            logger.addHandler(logging.NullHandler())
    return cdsapi.Client(url=EWDS_URL, key=key or config.ewds_api_key(), quiet=False, progress=False)


@lru_cache(maxsize=1)
def _consolidated_months() -> frozenset[tuple[int, int]]:
    """(year, month) pairs covered by the quality-controlled consolidated dataset.

    Recent months only exist in the provisional "intermediate" dataset.
    """
    coll = requests.get(f"{EWDS_URL}/catalogue/v1/collections/{DATASET}", timeout=30)
    coll.raise_for_status()
    href = next(link["href"] for link in coll.json()["links"] if link.get("rel") == "constraints")
    resp = requests.get(href, timeout=30)
    resp.raise_for_status()
    months = set()
    for combo in resp.json():
        if "consolidated_dataset" in combo.get("dataset_type", []):
            months.update((int(y), int(m)) for y in combo["year"] for m in combo["month"])
    return frozenset(months)


def dataset_type_for(year: int, month: int) -> str:
    try:
        consolidated = _consolidated_months()
    except (requests.RequestException, KeyError, StopIteration) as exc:
        log.warning("Could not read EWDS constraints (%s); assuming consolidated data", exc)
        return "consolidated_dataset"
    return "consolidated_dataset" if (year, month) in consolidated else "intermediate_dataset"


def resolve_indices(names: Iterable[str]) -> list[str]:
    names = [n.lower() for n in names]
    unknown = [n for n in names if n not in INDICES]
    if unknown:
        raise ValueError(f"Unknown index {unknown}; choose from {sorted(INDICES)}")
    return sorted(set(names), key=list(INDICES).index)


def build_request(bbox: BBox, year: int, month: int, days: list[int], indices: list[str], grid: str) -> dict:
    return {
        "product_type": "reanalysis",
        "variable": [INDICES[i].name for i in indices],
        "dataset_type": dataset_type_for(year, month),
        "system_version": SYSTEM_VERSION,
        "year": [f"{year}"],
        "month": [f"{month:02d}"],
        "day": [f"{d:02d}" for d in days],
        "grid": GRIDS[grid],
        "area": bbox.as_cds_area(),
        "data_format": "netcdf",
    }


def request_area(bbox: BBox, grid: str) -> BBox:
    """The area actually requested from EWDS for ``bbox``.

    An area smaller than one grid cell can come back empty, and EWDS places the grid at the
    requested west/north edge, so half a cell more on every side makes the cells cover the area.
    """
    g = float(grid)
    b = bbox.buffered(g)
    return BBox(max(-180.0, b.west - g / 2), max(-90.0, b.south - g / 2),
                min(180.0, b.east + g / 2), min(90.0, b.north + g / 2))


@dataclass(frozen=True)
class MonthRequest:
    year: int
    month: int
    request: dict
    stem: str  # cache file name without extension

    def cached_files(self, out_dir: Path) -> list[Path]:
        return sorted(out_dir.glob(f"{self.stem}*.nc"))


def default_cache() -> Path:
    return config.cache_dir() / "cems"


def plan_months(bbox: BBox, dates: DateRange, indices: Iterable[str], grid: str = "0.25") -> list[MonthRequest]:
    """One EWDS request per calendar month, with the cache file name it is stored under."""
    if grid not in GRIDS:
        raise ValueError(f"grid must be one of {sorted(GRIDS)}")
    indices = resolve_indices(indices)
    area = request_area(bbox, grid)
    plan = []
    for year, month, days in dates.months():
        request = build_request(area, year, month, days, indices, grid)
        # The dataset type is part of the name so provisional data is refetched once consolidated.
        kind = request["dataset_type"].split("_")[0]
        stem = f"cems_{'-'.join(indices)}_{grid}_{area.slug()}_{kind}_{year}-{month:02d}_{days[0]:02d}-{days[-1]:02d}"
        plan.append(MonthRequest(year, month, request, stem))
    return plan


def download_indices(
    bbox: BBox,
    dates: DateRange,
    indices: Iterable[str] = ("fwi",),
    *,
    grid: str = "0.25",
    out_dir: Path | None = None,
    client: RetrieveClient | None = None,
    overwrite: bool = False,
    workers: int = 3,
) -> list[Path]:
    """Download daily fire danger indices as NetCDF, one request per calendar month.

    Up to ``workers`` monthly requests wait in the EWDS queue at the same time.
    Returns the NetCDF paths in chronological order. Files already present in
    ``out_dir`` are reused unless ``overwrite`` is set; the EWDS key is only
    needed for months that are not cached.
    """
    from concurrent.futures import ThreadPoolExecutor

    out_dir = Path(out_dir or default_cache())
    out_dir.mkdir(parents=True, exist_ok=True)

    def fetch_month(m: MonthRequest) -> list[Path]:
        existing = m.cached_files(out_dir)
        if existing and not overwrite:
            log.info("Using cached %s", existing[0].name)
            return existing
        log.info("Requesting %s %d-%02d (%s) from EWDS", ",".join(m.request["variable"]), m.year, m.month,
                 m.request["dataset_type"])
        tmp = out_dir / f"{m.stem}.part"
        (client or make_client()).retrieve(DATASET, m.request, str(tmp))
        log.info("Downloaded %d-%02d", m.year, m.month)
        return _finalise(tmp, out_dir / f"{m.stem}.nc")

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(fetch_month, plan_months(bbox, dates, indices, grid)))
    return [p for month_paths in results for p in month_paths]


def _finalise(tmp: Path, target: Path) -> list[Path]:
    """Move the download into place, unpacking it if the server sent a zip of NetCDFs."""
    if not zipfile.is_zipfile(tmp):
        tmp.replace(target)
        return [target]
    out = []
    with zipfile.ZipFile(tmp) as zf:
        for i, member in enumerate(m for m in zf.namelist() if m.endswith(".nc")):
            dest = target.with_name(f"{target.stem}_{i}.nc")
            dest.write_bytes(zf.read(member))
            out.append(dest)
    tmp.unlink()
    return out


# GRIB short names seen in EWDS NetCDF files -> fire-cli names.
GRIB_SHORT_NAMES = {"fwinx": "fwi", "fdimrk": "ffdi"}


def _short_name(var) -> str | None:
    """fire-cli name for a NetCDF variable, from its GRIB short name or its long name."""
    if var.name in GRIB_SHORT_NAMES:
        return GRIB_SHORT_NAMES[var.name]
    long_name = str(var.attrs.get("long_name", "")).lower()
    matches = [short for short, idx in INDICES.items() if idx.description.lower() in long_name]
    return matches[0] if len(matches) == 1 else None


def open_indices(paths: Iterable[Path]):
    """Open downloaded NetCDFs as one ``xarray.Dataset`` with ``time``, ``latitude``, ``longitude``.

    Variables are renamed to fire-cli index names (``fwi``, ``ffdi``, ...) where
    recognised, longitudes are normalised to [-180, 180) and latitudes sorted
    north to south.
    """
    import xarray as xr

    datasets = []
    for p in paths:
        ds = xr.open_dataset(p)
        if "valid_time" in ds.dims or "valid_time" in ds.coords:
            ds = ds.rename({"valid_time": "time"})
        ds = ds.drop_vars([v for v in ("number", "surface", "step") if v in ds.coords])
        datasets.append(ds)
    ds = xr.merge(datasets) if len(datasets) > 1 else datasets[0]

    ds = ds.rename({v: short for v, short in ((v, _short_name(ds[v])) for v in ds.data_vars) if short and short != v})

    if float(ds.longitude.max()) > 180:
        ds = ds.assign_coords(longitude=((ds.longitude + 180) % 360) - 180)
    ds = ds.sortby("longitude").sortby("latitude", ascending=False).sortby("time")
    return ds
