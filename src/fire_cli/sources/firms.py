"""Active-fire detections from NASA FIRMS (Fire Information for Resource Management System).

Uses the FIRMS area API, which returns at most 5 days per request. Historical
dates come from the quality-controlled Standard Processing (SP) archive and
recent dates from the Near Real-Time (NRT) feed; the split point is read from
the FIRMS data availability endpoint.
"""

from __future__ import annotations

import io
import logging
import os
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from fire_cli import config
from fire_cli.query import BBox, DateRange

log = logging.getLogger(__name__)

FIRMS_URL = "https://firms.modaps.eosdis.nasa.gov"
MAX_DAYS_PER_REQUEST = 5
# NRT files for the last few days can still be updated, so they are never cached.
NRT_SETTLE_DAYS = 3
# How long the product availability list is trusted before asking FIRMS again (when a key is set).
AVAILABILITY_MAX_AGE_S = 24 * 3600


@dataclass(frozen=True)
class Sensor:
    sp: str | None  # Standard Processing product, None if not available
    nrt: str


SENSORS: dict[str, Sensor] = {
    "viirs-snpp": Sensor("VIIRS_SNPP_SP", "VIIRS_SNPP_NRT"),
    "viirs-noaa20": Sensor("VIIRS_NOAA20_SP", "VIIRS_NOAA20_NRT"),
    "viirs-noaa21": Sensor("VIIRS_NOAA21_SP", "VIIRS_NOAA21_NRT"),
    "modis": Sensor("MODIS_SP", "MODIS_NRT"),
    "landsat": Sensor(None, "LANDSAT_NRT"),  # US and Canada only
}


class FirmsError(RuntimeError):
    pass


class FirmsClient:
    """FIRMS API client. The MAP_KEY is only needed once a request is actually sent, so
    fully cached jobs work without one."""

    def __init__(self, map_key: str | None = None, session: requests.Session | None = None,
                 base_url: str = FIRMS_URL, cache: Path | None = None):
        self._map_key = map_key or os.environ.get("FIRMS_MAP_KEY")
        self.session = session or requests.Session()
        self.base_url = base_url.rstrip("/")
        self.cache = Path(cache) if cache else None
        self._availability: dict[str, tuple[date, date]] | None = None

    @property
    def has_key(self) -> bool:
        return bool(self._map_key)

    @property
    def map_key(self) -> str:
        return self._map_key or config.firms_map_key()  # raises MissingCredentialsError

    def _get(self, path: str) -> str:
        url = f"{self.base_url}/api/{path}"
        for attempt in range(4):
            resp = self.session.get(url, timeout=120)
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < 3:
                wait = 2**attempt * 5
                log.warning("FIRMS returned %s, retrying in %ss", resp.status_code, wait)
                time.sleep(wait)
                continue
            if resp.status_code >= 400:
                raise FirmsError(f"FIRMS returned HTTP {resp.status_code}: {resp.text.strip()[:300]}")
            return resp.text
        raise AssertionError("unreachable")

    def availability(self) -> dict[str, tuple[date, date]]:
        """``{product: (min_date, max_date)}`` for every FIRMS product.

        Cached for a day; an older copy is used when there is no key or FIRMS can't be reached.
        """
        if self._availability is None:
            path = self.cache / "availability.csv" if self.cache else None
            cached = path is not None and path.exists()
            fresh = cached and time.time() - path.stat().st_mtime < AVAILABILITY_MAX_AGE_S
            text = None
            if cached and (fresh or not self.has_key):
                text = path.read_text()
            else:
                try:
                    text = self._get(f"data_availability/csv/{self.map_key}/ALL")
                    _check_csv(text, "data_id")
                except requests.RequestException:
                    if not cached:
                        raise
                    log.warning("Could not reach FIRMS; using the cached product availability")
                    text = path.read_text()
                else:
                    if path is not None:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(text)
            df = pd.read_csv(io.StringIO(text))
            self._availability = {
                row.data_id: (date.fromisoformat(row.min_date), date.fromisoformat(row.max_date))
                for row in df.itertuples()
            }
        return self._availability

    def plan(self, sensor: str, dates: DateRange) -> list[tuple[str, DateRange]]:
        """Split ``dates`` into ``(product, sub-range)`` pairs, preferring SP over NRT."""
        if sensor not in SENSORS:
            raise ValueError(f"Unknown sensor {sensor!r}; choose from {sorted(SENSORS)}")
        products = SENSORS[sensor]
        avail = self.availability()
        plan: list[tuple[str, DateRange]] = []
        sp_end = None
        if products.sp and products.sp in avail:
            sp_start, sp_end = avail[products.sp]
            if part := dates.clip(sp_start, sp_end):
                plan.append((products.sp, part))
        if products.nrt in avail:
            nrt_start, nrt_end = avail[products.nrt]
            if sp_end:
                nrt_start = max(nrt_start, sp_end + timedelta(days=1))
            if part := dates.clip(nrt_start, nrt_end):
                plan.append((products.nrt, part))

        covered = sum(p.days for _, p in plan)
        if covered < dates.days:
            log.warning("%s has no data for %d of %d requested days", sensor, dates.days - covered, dates.days)
        return plan

    def area(self, product: str, bbox: BBox, chunk: DateRange) -> pd.DataFrame:
        """One area API call (at most 5 days)."""
        if chunk.days > MAX_DAYS_PER_REQUEST:
            raise ValueError(f"FIRMS allows at most {MAX_DAYS_PER_REQUEST} days per request")
        text = self._get(f"area/csv/{self.map_key}/{product}/{bbox.as_firms()}/{chunk.days}/{chunk.start.isoformat()}")
        _check_csv(text, "latitude")
        return pd.read_csv(io.StringIO(text))


def _check_csv(text: str, first_column: str) -> None:
    # FIRMS reports errors (bad key, bad parameters, quota) as plain text with HTTP 200.
    if not text.lstrip().startswith(first_column):
        raise FirmsError(f"Unexpected FIRMS response: {text.strip()[:300]}")


@dataclass(frozen=True)
class Chunk:
    sensor: str
    product: str
    dates: DateRange
    path: Path  # cache file

    @property
    def cached(self) -> bool:
        return self.path.exists()


def plan_chunks(bbox: BBox, dates: DateRange, sensors: Iterable[str], client: FirmsClient, cache: Path) -> list[Chunk]:
    """Every area request the job needs, with the cache file it is (or will be) stored in."""
    return [
        Chunk(sensor, product, chunk, cache / f"{product}_{bbox.slug()}_{chunk.start}_{chunk.end}.csv")
        for sensor in dict.fromkeys(sensors)
        for product, part in client.plan(sensor, dates)
        for chunk in part.chunks(MAX_DAYS_PER_REQUEST)
    ]


def default_cache() -> Path:
    return config.cache_dir() / "firms"


def download_detections(
    bbox: BBox,
    dates: DateRange,
    sensors: Iterable[str] = ("viirs-snpp",),
    *,
    client: FirmsClient | None = None,
    cache: Path | None = None,
):
    """Fetch active-fire detections as a point ``GeoDataFrame`` in EPSG:4326.

    Adds ``sensor``, ``product`` and a UTC ``acq_datetime`` column to the FIRMS
    attributes. Settled chunks are cached as CSV under ``cache``.
    """
    import geopandas as gpd

    cache = Path(cache or default_cache())
    cache.mkdir(parents=True, exist_ok=True)
    client = client or FirmsClient(cache=cache)
    settled_before = date.today() - timedelta(days=NRT_SETTLE_DAYS)

    frames = []
    for chunk in plan_chunks(bbox, dates, sensors, client, cache):
        if chunk.cached:
            df = pd.read_csv(chunk.path)
        else:
            log.info("Requesting %s %s..%s from FIRMS", chunk.product, chunk.dates.start, chunk.dates.end)
            df = client.area(chunk.product, bbox, chunk.dates)
            if chunk.dates.end < settled_before:
                df.to_csv(chunk.path, index=False)
        if not df.empty:
            frames.append(df.assign(sensor=chunk.sensor, product=chunk.product))

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["latitude", "longitude"])
    if not df.empty:
        hhmm = df["acq_time"].astype(int).astype(str).str.zfill(4)
        df["acq_datetime"] = pd.to_datetime(df["acq_date"] + " " + hhmm, format="%Y-%m-%d %H%M", utc=True)
        # MODIS confidence is 0-100, VIIRS is l/n/h; keep one dtype so the column can be written to GIS formats.
        for col in ("confidence", "version"):
            if col in df:
                df[col] = df[col].astype(str)
        df = df.drop_duplicates()
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326")
