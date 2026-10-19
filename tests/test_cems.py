import threading
import zipfile

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from fire_cli.query import BBox, DateRange
from fire_cli.sources import cems


@pytest.fixture(autouse=True)
def offline_constraints(monkeypatch):
    monkeypatch.setattr(cems, "_consolidated_months", lambda: frozenset({(2023, 7)}))


def _fake_netcdf(path, days, lon):
    times = pd.to_datetime([f"2023-07-{d:02d}" for d in days])
    lat = np.array([49.0, 50.0, 51.0])
    data = np.random.rand(len(times), len(lat), len(lon)).astype("float32")
    ds = xr.Dataset(
        {"fwinx": (("valid_time", "latitude", "longitude"), data)},
        coords={"valid_time": times, "latitude": lat, "longitude": lon},
    )
    ds.to_netcdf(path)


class FakeClient:
    # download_indices calls retrieve() from worker threads; netCDF/HDF5 writes are not thread-safe.
    _lock = threading.Lock()

    def __init__(self, lon=(12.0, 13.0), zipped=False):
        self.requests = []
        self.lon = np.array(lon)
        self.zipped = zipped

    def retrieve(self, name, request, target=None):
        with self._lock:
            self._retrieve(name, request, target)

    def _retrieve(self, name, request, target):
        self.requests.append((name, request))
        days = [int(d) for d in request["day"]]
        if not self.zipped:
            _fake_netcdf(target, days, self.lon)
            return
        nc = target + ".nc"
        _fake_netcdf(nc, days, self.lon)
        with zipfile.ZipFile(target, "w") as zf:
            zf.write(nc, "data_0.nc")


def test_one_request_per_month_with_correct_types(tmp_path):
    client = FakeClient()
    paths = cems.download_indices(
        BBox(12, 48.5, 19, 51), DateRange.parse("2023-07-30", "2023-08-02"), ["FWI", "ffdi"],
        out_dir=tmp_path, client=client,
    )
    assert len(paths) == 2
    (_, july), (_, aug) = sorted(client.requests, key=lambda r: r[1]["month"])
    assert july["variable"] == ["fire_weather_index", "fire_danger_index"]
    assert july["day"] == ["30", "31"] and july["dataset_type"] == "consolidated_dataset"
    assert aug["day"] == ["01", "02"] and aug["dataset_type"] == "intermediate_dataset"
    # half a 0.25° cell of padding on every side
    assert july["area"] == [51.125, 11.875, 48.375, 19.125]

    # Cached on second call
    cems.download_indices(
        BBox(12, 48.5, 19, 51), DateRange.parse("2023-07-30", "2023-08-02"), ["fwi", "ffdi"],
        out_dir=tmp_path, client=client,
    )
    assert len(client.requests) == 2


def test_unknown_index():
    with pytest.raises(ValueError, match="Unknown index"):
        cems.resolve_indices(["fwi", "nope"])


def test_zip_response_is_unpacked(tmp_path):
    paths = cems.download_indices(
        BBox(12, 48.5, 19, 51), DateRange.parse("2023-07-01"), out_dir=tmp_path, client=FakeClient(zipped=True)
    )
    assert len(paths) == 1 and paths[0].suffix == ".nc"
    assert not list(tmp_path.glob("*.part"))


def test_open_indices_normalises_coords(tmp_path):
    paths = cems.download_indices(
        BBox(12, 48.5, 19, 51), DateRange.parse("2023-07-01", "2023-07-02"),
        out_dir=tmp_path, client=FakeClient(lon=(350.0, 359.0, 1.0)),
    )
    ds = cems.open_indices(paths)
    assert "time" in ds.dims
    assert list(ds.longitude.values) == [-10.0, -1.0, 1.0]
    assert ds.latitude.values[0] > ds.latitude.values[-1]


def test_open_indices_uses_fire_cli_names(tmp_path):
    lat, lon = [50.0], [14.0]
    t = pd.to_datetime(["2023-07-01"])
    ds = xr.Dataset(
        {
            "fwinx": (("valid_time", "latitude", "longitude"), [[[1.0]]]),
            "ffmcode": (("valid_time", "latitude", "longitude"), [[[2.0]]], {"long_name": "Fine fuel moisture code"}),
            "mystery": (("valid_time", "latitude", "longitude"), [[[3.0]]], {"long_name": "Something else"}),
        },
        coords={"valid_time": t, "latitude": lat, "longitude": lon},
    )
    ds.to_netcdf(tmp_path / "a.nc")
    out = cems.open_indices([tmp_path / "a.nc"])
    assert set(out.data_vars) == {"fwi", "ffmc", "mystery"}
