from datetime import date

import pytest

from fire_cli.query import BBox, DateRange
from fire_cli.sources import firms

AVAILABILITY = """data_id,min_date,max_date
MODIS_NRT,2026-06-01,2026-10-18
MODIS_SP,2000-11-01,2026-05-31
VIIRS_SNPP_NRT,2026-04-01,2026-10-18
VIIRS_SNPP_SP,2012-01-20,2026-03-31
"""

VIIRS_CSV = """latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight
50.1,14.2,330.5,0.4,0.4,{day},134,N,VIIRS,n,2,290.1,5.2,D
"""

MODIS_CSV = """latitude,longitude,brightness,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_t31,frp,daynight
50.2,14.3,320.0,1.0,1.0,{day},1030,T,MODIS,77,6.1NRT,295.0,12.0,D
"""


class FakeResponse:
    def __init__(self, text, status_code=200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class FakeSession:
    def __init__(self):
        self.urls = []

    def get(self, url, timeout=None):
        self.urls.append(url)
        if "data_availability" in url:
            return FakeResponse(AVAILABILITY)
        if "/area/csv/" in url:
            product, _bbox, _days, start = url.split("/area/csv/")[1].split("/")[1:]
            template = MODIS_CSV if product.startswith("MODIS") else VIIRS_CSV
            return FakeResponse(template.format(day=start))
        return FakeResponse("not found", 404)


@pytest.fixture
def client():
    return firms.FirmsClient(map_key="KEY", session=FakeSession())


def test_plan_splits_between_sp_and_nrt(client):
    plan = client.plan("viirs-snpp", DateRange.parse("2026-03-25", "2026-04-05"))
    assert plan == [
        ("VIIRS_SNPP_SP", DateRange(date(2026, 3, 25), date(2026, 3, 31))),
        ("VIIRS_SNPP_NRT", DateRange(date(2026, 4, 1), date(2026, 4, 5))),
    ]


def test_plan_unknown_sensor(client):
    with pytest.raises(ValueError):
        client.plan("goes", DateRange.parse("2026-01-01"))


def test_download_chunks_and_merges_sensors(client, tmp_path):
    bbox = BBox(12, 48.5, 19, 51)
    gdf = firms.download_detections(
        bbox, DateRange.parse("2023-07-01", "2023-07-12"), ["viirs-snpp", "modis"], client=client, cache=tmp_path
    )
    area_calls = [u for u in client.session.urls if "/area/" in u]
    # 12 days -> 3 chunks per sensor
    assert len(area_calls) == 6
    assert "/VIIRS_SNPP_SP/12,48.5,19,51/5/2023-07-01" in area_calls[0]
    assert set(gdf["sensor"]) == {"viirs-snpp", "modis"}
    assert gdf.crs.to_epsg() == 4326
    assert gdf["confidence"].map(type).eq(str).all()
    assert str(gdf["acq_datetime"].iloc[0]) == "2023-07-01 01:34:00+00:00"

    # Settled chunks were cached, so a second run makes no area calls.
    client.session.urls.clear()
    firms.download_detections(
        bbox, DateRange.parse("2023-07-01", "2023-07-12"), ["viirs-snpp", "modis"], client=client, cache=tmp_path
    )
    assert not [u for u in client.session.urls if "/area/" in u]


def test_gdf_writes_to_geopackage(client, tmp_path):
    gdf = firms.download_detections(
        BBox(12, 48.5, 19, 51), DateRange.parse("2023-07-01"), ["viirs-snpp", "modis"], client=client, cache=tmp_path
    )
    out = tmp_path / "fires.gpkg"
    gdf.to_file(out)
    assert out.exists()


def test_error_text_raises(tmp_path):
    class BadKeySession(FakeSession):
        def get(self, url, timeout=None):
            return FakeResponse("Invalid MAP_KEY.", 401)

    c = firms.FirmsClient(map_key="BAD", session=BadKeySession())
    with pytest.raises(firms.FirmsError, match="Invalid MAP_KEY"):
        c.availability()
