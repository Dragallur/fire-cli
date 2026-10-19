import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
import xarray as xr
from click.testing import CliRunner
from PIL import Image

from fire_cli import styles
from fire_cli.animate import animate
from fire_cli.cli import main
from fire_cli.job import Job
from fire_cli.render import Outputs, render


@pytest.fixture
def outdir(tmp_path):
    """A small fetch output: 3 days of FWI on a 0.25° grid and a few detections."""
    job = Job(name="t", description="Test fire", bbox=[14.0, 50.0, 15.0, 50.5],
              start=pd.Timestamp("2023-07-01").date(), end=pd.Timestamp("2023-07-03").date(), output=str(tmp_path))
    (tmp_path / "job.yaml").write_text(job.to_yaml())

    lat = np.array([50.5, 50.25, 50.0])
    lon = np.array([14.0, 14.25, 14.5, 14.75, 15.0])
    fwi = np.stack([np.full((3, 5), v, dtype="float32") for v in (3.0, 25.0, 60.0)])
    fwi[0, 0, 0] = np.nan
    xr.Dataset({"fwi": (("time", "latitude", "longitude"), fwi)},
               coords={"time": pd.date_range("2023-07-01", periods=3), "latitude": lat, "longitude": lon}
               ).to_netcdf(tmp_path / "indices.nc")

    fires = pd.DataFrame({
        "latitude": [50.22, 50.22, 50.33, 49.0],   # last one is outside the area
        "longitude": [14.52, 14.52, 14.67, 14.5],
        "acq_date": ["2023-07-02", "2023-07-02", "2023-07-03", "2023-07-03"],
        "frp": [10.0, 5.0, 2.0, 1.0],
    })
    gpd.GeoDataFrame(fires, geometry=gpd.points_from_xy(fires.longitude, fires.latitude), crs=4326).to_file(
        tmp_path / "fires.gpkg", layer="fires")
    return tmp_path


def test_index_rasters(outdir):
    files = {f.name: f for f in render(outdir)}
    assert {"fwi_daily.tif", "fwi_max.tif", "fwi_mean.tif", "fires_count.tif"} <= set(files)
    with rasterio.open(files["fwi_daily.tif"]) as ds:
        assert ds.count == 3 and ds.descriptions == ("2023-07-01", "2023-07-02", "2023-07-03")
        assert ds.crs.to_epsg() == 4326
        # cell centres at 0.25° steps -> edges half a cell outside
        assert ds.bounds.left == pytest.approx(13.875) and ds.bounds.top == pytest.approx(50.625)
        assert ds.read(2)[0, 0] == 25.0
    with rasterio.open(files["fwi_max.tif"]) as ds:
        assert ds.read(1)[0, 0] == 60.0
    assert "EFFIS" not in files["fwi_daily.tif"].with_suffix(".qml").read_text()
    assert 'colorRampType="DISCRETE"' in files["fwi_daily.tif"].with_suffix(".qml").read_text()


def test_fire_rasters(outdir):
    files = {f.name: f for f in render(outdir, fire_res=0.1)}
    with rasterio.open(files["fires_count.tif"]) as ds:
        count = ds.read(1)
        assert count.sum() == 3  # the point outside the area is dropped
        assert count.max() == 2
    with rasterio.open(files["fires_first_day.tif"]) as ds:
        first = ds.read(1)
        assert sorted(first[first > 0].tolist()) == [2, 3]  # day 2 and day 3 of the interval
    with rasterio.open(files["fires_frp.tif"]) as ds:
        assert np.nanmax(ds.read(1)) == pytest.approx(15.0)


def test_outputs_load_from_job_file(outdir):
    assert Outputs.load(outdir / "job.yaml").dir == outdir


def test_animation_gif_has_one_frame_per_day(outdir):
    out = animate(outdir, fps=4, width_px=500, basemap="none", hold_last=0)
    with Image.open(out) as im:
        assert im.n_frames == 3
        assert im.info["duration"] == 250


def test_animation_mp4(outdir):
    out = animate(outdir, out=outdir / "a.mp4", width_px=500, basemap="none")
    assert out.stat().st_size > 1000


def test_animation_unknown_index(outdir):
    with pytest.raises(ValueError, match="not downloaded"):
        animate(outdir, index="ffdi", basemap="none")


def test_danger_class_labels():
    fwi = styles.DANGER_CLASSES["fwi"]
    assert [fwi.label_for(v) for v in (0, 5.2, 21.2, 49.9, 80)] == ["Very low", "Low", "Moderate", "Very high", "Extreme"]


def test_cli_render_and_animate(outdir):
    r = CliRunner().invoke(main, ["render", str(outdir), "--stats", "max"])
    assert r.exit_code == 0, r.output
    assert not (outdir / "rasters" / "fwi_mean.tif").exists()
    r = CliRunner().invoke(main, ["animate", str(outdir), "--basemap", "none", "--width", "500", "-o",
                                  str(outdir / "x.gif")])
    assert r.exit_code == 0, r.output
    assert (outdir / "x.gif").exists()


def test_render_and_animate_create_output_folders(outdir, tmp_path):
    files = render(outdir, out_dir=tmp_path / "new" / "nested")
    assert files and all(f.parent == tmp_path / "new" / "nested" for f in files)
    out = animate(outdir, out=tmp_path / "a" / "b" / "anim.gif", width_px=400, basemap="none")
    assert out.exists()


def test_missing_input_path_explains_both_paths(tmp_path):
    r = CliRunner().invoke(main, ["render", str(tmp_path / "nope")])
    assert r.exit_code == 2
    assert "does not exist" in r.output and "-o" in r.output
