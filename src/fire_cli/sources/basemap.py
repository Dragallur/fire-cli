"""Reference map layers from Natural Earth (public domain, 1:10m scale).

Gives the fire layers some geographic context (borders, lakes, rivers, towns)
in GIS software and in rendered animation frames, without needing a tile service.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import requests

from fire_cli import config
from fire_cli.query import BBox

log = logging.getLogger(__name__)

NATURAL_EARTH_URL = "https://naciscdn.org/naturalearth/10m"

# name -> (Natural Earth dataset, attribute columns to keep)
LAYERS: dict[str, tuple[str, list[str]]] = {
    "countries": ("cultural/ne_10m_admin_0_countries", ["name", "adm0_a3"]),
    "lakes": ("physical/ne_10m_lakes", ["name"]),
    "rivers": ("physical/ne_10m_rivers_lake_centerlines", ["name"]),
    "places": ("cultural/ne_10m_populated_places_simple", ["name", "featurecla", "pop_max"]),
}


def _fetch(dataset: str, cache: Path) -> Path:
    path = cache / f"{dataset.split('/')[-1]}.zip"
    if not path.exists():
        log.info("Downloading Natural Earth %s", path.stem)
        resp = requests.get(f"{NATURAL_EARTH_URL}/{dataset}.zip", timeout=120)
        resp.raise_for_status()
        tmp = path.with_suffix(".part")
        tmp.write_bytes(resp.content)
        tmp.replace(path)
    return path


def download_basemap(
    bbox: BBox,
    layers: Iterable[str] = tuple(LAYERS),
    *,
    margin: float = 0.25,
    cache: Path | None = None,
) -> dict:
    """Natural Earth layers clipped to ``bbox`` grown by ``margin`` (fraction of its size).

    Returns ``{layer name: GeoDataFrame}`` in EPSG:4326; layers with no features
    in the area are left out.
    """
    import geopandas as gpd
    from shapely.geometry import box

    cache = Path(cache or config.cache_dir() / "naturalearth")
    cache.mkdir(parents=True, exist_ok=True)

    dx = max((bbox.east - bbox.west) * margin, 0.1)
    dy = max((bbox.north - bbox.south) * margin, 0.1)
    extent = BBox(
        max(-180.0, bbox.west - dx), max(-90.0, bbox.south - dy),
        min(180.0, bbox.east + dx), min(90.0, bbox.north + dy),
    )
    window = (extent.west, extent.south, extent.east, extent.north)

    out = {}
    for name in layers:
        if name not in LAYERS:
            raise ValueError(f"Unknown basemap layer {name!r}; choose from {sorted(LAYERS)}")
        dataset, columns = LAYERS[name]
        gdf = gpd.read_file(_fetch(dataset, cache), bbox=window)
        if gdf.empty:
            continue
        gdf.columns = [c if c == "geometry" else c.lower() for c in gdf.columns]
        gdf = gdf[[c for c in columns if c in gdf.columns] + ["geometry"]]
        out[name] = gpd.clip(gdf, box(*window)).to_crs(4326)
    return out
