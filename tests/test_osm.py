import io

import pytest
import rasterio
from PIL import Image

from fire_cli.query import BBox
from fire_cli.sources import osm


class FakeFetcher:
    def __init__(self):
        self.calls = []

    def get(self, z, x, y):
        self.calls.append((z, x, y))
        buf = io.BytesIO()
        Image.new("P", (256, 256), color=x % 256).save(buf, format="PNG")
        return buf.getvalue()


def test_tile_range_known_values():
    # Prague at zoom 10 is tile x=553, y=346
    tr = osm.tile_range(BBox(14.42, 50.08, 14.43, 50.09), 10)
    assert (tr.x0, tr.y0, tr.count) == (553, 346, 1)


def test_choose_zoom_respects_limit():
    bbox = BBox(14.0, 50.7, 14.6, 51.0)
    z = osm.choose_zoom(bbox, max_tiles=64)
    assert osm.tile_range(bbox, z).count <= 64 < osm.tile_range(bbox, z + 1).count


def test_mosaic_is_georeferenced(tmp_path):
    bbox = BBox(14.0, 50.7, 14.6, 51.0)
    fetcher = FakeFetcher()
    path = osm.download_osm_basemap(bbox, tmp_path / "osm.tif", zoom=10, fetcher=fetcher)
    tr = osm.tile_range(bbox, 10)
    assert len(fetcher.calls) == tr.count
    with rasterio.open(path) as ds:
        assert ds.crs.to_epsg() == 3857
        assert ds.count == 3 and ds.width == 256 * (tr.x1 - tr.x0 + 1)
        assert "OpenStreetMap" in ds.tags()["TIFFTAG_COPYRIGHT"]
        # The mosaic covers the requested area
        from pyproj import Transformer

        t = Transformer.from_crs(4326, 3857, always_xy=True)
        left, bottom = t.transform(bbox.west, bbox.south)
        right, top = t.transform(bbox.east, bbox.north)
        assert ds.bounds.left <= left and ds.bounds.right >= right
        assert ds.bounds.bottom <= bottom and ds.bounds.top >= top


def test_explicit_zoom_over_limit_raises(tmp_path):
    with pytest.raises(ValueError, match="tiles"):
        osm.download_osm_basemap(BBox(0, 40, 20, 55), tmp_path / "x.tif", zoom=10, fetcher=FakeFetcher())
