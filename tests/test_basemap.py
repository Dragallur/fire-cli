import zipfile

import geopandas as gpd
import pytest
from shapely.geometry import Point, box

from fire_cli.query import BBox
from fire_cli.sources import basemap


@pytest.fixture
def cache(tmp_path):
    """Pre-populate the cache with tiny stand-ins for the Natural Earth zips."""
    shp = tmp_path / "src"
    shp.mkdir()
    countries = gpd.GeoDataFrame(
        {"NAME": ["Inside", "Far away"], "ADM0_A3": ["INS", "FAR"], "POP_EST": [1, 2]},
        geometry=[box(10, 45, 20, 55), box(100, 0, 110, 10)], crs=4326,
    )
    places = gpd.GeoDataFrame({"name": ["Outside"], "featurecla": ["x"], "pop_max": [5]}, geometry=[Point(-50, 0)], crs=4326)
    for gdf, stem in [(countries, "ne_10m_admin_0_countries"), (places, "ne_10m_populated_places_simple")]:
        gdf.to_file(shp / f"{stem}.shp")
        with zipfile.ZipFile(tmp_path / f"{stem}.zip", "w") as zf:
            for f in shp.glob(f"{stem}.*"):
                zf.write(f, f.name)
    return tmp_path


def test_clips_to_area_and_keeps_selected_columns(cache):
    layers = basemap.download_basemap(BBox(14, 50, 15, 51), ["countries", "places"], margin=0, cache=cache)
    assert list(layers) == ["countries"]  # no places in the area
    countries = layers["countries"]
    assert list(countries["name"]) == ["Inside"]
    assert list(countries.columns) == ["name", "adm0_a3", "geometry"]
    # margin=0 still leaves the minimum 0.1 degree border
    assert countries.total_bounds.tolist() == pytest.approx([13.9, 49.9, 15.1, 51.1])


def test_unknown_layer(cache):
    with pytest.raises(ValueError, match="Unknown basemap layer"):
        basemap.download_basemap(BBox(14, 50, 15, 51), ["roads"], cache=cache)
