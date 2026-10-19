"""OpenStreetMap raster tiles mosaicked into a georeferenced GeoTIFF basemap.

Follows the OSM tile usage policy (https://operations.osmfoundation.org/policies/tiles/):
identifying User-Agent, sequential requests, local tile cache, and a hard cap on
tiles per request -- no bulk downloading. Maps must credit "© OpenStreetMap contributors".
"""

from __future__ import annotations

import io
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests

from fire_cli import __version__, config
from fire_cli.query import BBox

log = logging.getLogger(__name__)

OSM_TILES = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
ATTRIBUTION = "© OpenStreetMap contributors"
USER_AGENT = f"fire-cli/{__version__} (wildfire data tool; python-requests)"
TILE_SIZE = 256
MAX_LAT = 85.05112878  # Web Mercator limit
MAX_ZOOM = 16
DEFAULT_MAX_TILES = 64
HARD_MAX_TILES = 250
CACHE_DAYS = 30


@dataclass(frozen=True)
class TileRange:
    zoom: int
    x0: int
    x1: int  # inclusive
    y0: int
    y1: int  # inclusive

    @property
    def count(self) -> int:
        return (self.x1 - self.x0 + 1) * (self.y1 - self.y0 + 1)


def _tile_xy(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    lat = max(-MAX_LAT, min(MAX_LAT, lat))
    n = 2**zoom
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)
    return min(max(x, 0), n - 1), min(max(y, 0), n - 1)


def tile_range(bbox: BBox, zoom: int) -> TileRange:
    x0, y0 = _tile_xy(bbox.west, bbox.north, zoom)
    x1, y1 = _tile_xy(bbox.east, bbox.south, zoom)
    return TileRange(zoom, x0, x1, y0, y1)


def choose_zoom(bbox: BBox, max_tiles: int = DEFAULT_MAX_TILES) -> int:
    """Highest zoom whose mosaic stays within ``max_tiles``."""
    zoom = 0
    for z in range(MAX_ZOOM + 1):
        if tile_range(bbox, z).count > max_tiles:
            break
        zoom = z
    return zoom


def _mercator_bounds(tr: TileRange) -> tuple[float, float, float, float]:
    """(left, bottom, right, top) of the tile range in EPSG:3857 metres."""
    extent = 2 * math.pi * 6378137
    size = extent / 2**tr.zoom
    left = -extent / 2 + tr.x0 * size
    right = -extent / 2 + (tr.x1 + 1) * size
    top = extent / 2 - tr.y0 * size
    bottom = extent / 2 - (tr.y1 + 1) * size
    return left, bottom, right, top


class TileFetcher:
    def __init__(self, url: str = OSM_TILES, cache: Path | None = None, session: requests.Session | None = None):
        self.url = url
        self.cache = Path(cache or config.cache_dir() / "tiles")
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.downloaded = 0

    def _cache_path(self, z: int, x: int, y: int) -> Path:
        host = self.url.split("/")[2]
        return self.cache / host / str(z) / str(x) / f"{y}.png"

    def get(self, z: int, x: int, y: int) -> bytes:
        path = self._cache_path(z, x, y)
        if path.exists() and time.time() - path.stat().st_mtime < CACHE_DAYS * 86400:
            return path.read_bytes()
        resp = self.session.get(self.url.format(z=z, x=x, y=y), timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"Tile server returned HTTP {resp.status_code} for {z}/{x}/{y}: {resp.text[:200]}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(resp.content)
        self.downloaded += 1
        return resp.content


def download_osm_basemap(
    bbox: BBox,
    out_path: Path,
    *,
    zoom: int | None = None,
    max_tiles: int = DEFAULT_MAX_TILES,
    url: str = OSM_TILES,
    fetcher: TileFetcher | None = None,
) -> Path:
    """Write an RGB GeoTIFF (EPSG:3857) of map tiles covering ``bbox``.

    ``zoom`` defaults to the most detailed level that fits in ``max_tiles``.
    """
    import rasterio
    from PIL import Image
    from rasterio.transform import from_bounds

    max_tiles = min(max_tiles, HARD_MAX_TILES)
    zoom = choose_zoom(bbox, max_tiles) if zoom is None else zoom
    tr = tile_range(bbox, zoom)
    if tr.count > max_tiles:
        raise ValueError(
            f"Zoom {zoom} needs {tr.count} tiles (limit {max_tiles}). "
            "Lower the zoom or leave it unset to choose one automatically."
        )
    fetcher = fetcher or TileFetcher(url)
    log.info("Fetching %d map tiles at zoom %d", tr.count, zoom)

    width = (tr.x1 - tr.x0 + 1) * TILE_SIZE
    height = (tr.y1 - tr.y0 + 1) * TILE_SIZE
    mosaic = np.zeros((3, height, width), dtype=np.uint8)
    for x in range(tr.x0, tr.x1 + 1):
        for y in range(tr.y0, tr.y1 + 1):
            img = Image.open(io.BytesIO(fetcher.get(zoom, x, y))).convert("RGB")
            if img.size != (TILE_SIZE, TILE_SIZE):
                img = img.resize((TILE_SIZE, TILE_SIZE))
            row, col = (y - tr.y0) * TILE_SIZE, (x - tr.x0) * TILE_SIZE
            mosaic[:, row : row + TILE_SIZE, col : col + TILE_SIZE] = np.asarray(img).transpose(2, 0, 1)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff", "width": width, "height": height, "count": 3, "dtype": "uint8",
        "crs": "EPSG:3857", "transform": from_bounds(*_mercator_bounds(tr), width, height),
        "compress": "deflate", "tiled": True, "photometric": "RGB",
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mosaic)
        dst.update_tags(TIFFTAG_COPYRIGHT=ATTRIBUTION, source=url.split("/{z}")[0], zoom=str(zoom))
    return out_path
