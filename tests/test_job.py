from datetime import date

import pytest
from click.testing import CliRunner

from fire_cli import job as jobmod
from fire_cli.cli import main
from fire_cli.job import Job, JobError, estimate, load_job


@pytest.fixture(autouse=True)
def keys(monkeypatch):
    monkeypatch.setenv("FIRMS_MAP_KEY", "x")
    monkeypatch.setenv("EWDS_API_KEY", "y")


def test_examples_load_and_validate():
    for name in jobmod.example_names():
        job = load_job(example=name)
        job.validate()
        assert job.description


def test_job_file_overridden_by_options(tmp_path):
    (tmp_path / "area.geojson").write_text(
        '{"type":"FeatureCollection","features":[{"type":"Feature","properties":{},'
        '"geometry":{"type":"Point","coordinates":[15,50]}}]}'
    )
    f = tmp_path / "job.yaml"
    f.write_text("name: mine\narea_file: area.geojson\nstart: 2023-07-01\nindices: fwi, ffdi\nsensors: []\n")
    job = load_job(f, overrides={"end": "2023-07-03", "indices": None})
    assert job.indices == ["fwi", "ffdi"]
    assert job.sensors == []
    assert job.dates.days == 3
    # area_file is resolved relative to the job file; a single point gets a small box
    assert job.area.west == pytest.approx(14.95) and job.area.north == pytest.approx(50.05)

    job = load_job(f, overrides={"bbox": [1, 2, 3, 4]})
    assert job.area_file is None and job.area.east == 3


def test_example_then_override():
    job = load_job(example="ceske-svycarsko-2022", overrides={"sensors": ["modis"], "start": "2022-07-24"})
    assert job.sensors == ["modis"]
    assert job.start == date(2022, 7, 24)
    assert job.bbox == [13.95, 50.75, 14.65, 51.0]


def test_unknown_setting_is_reported(tmp_path):
    f = tmp_path / "job.yaml"
    f.write_text("bbox: [1, 2, 3, 4]\nstart: 2023-01-01\nsensor: [modis]\n")
    with pytest.raises(JobError, match="sensor"):
        load_job(f)


def test_yaml_roundtrip():
    job = load_job(example="australia-2019-20")
    assert Job.from_dict(__import__("yaml").safe_load(job.to_yaml())) == job


def test_estimate_warnings_and_errors(monkeypatch):
    big = estimate(load_job(example="australia-2019-20"))
    assert any("Large area" in w for w in big.warnings)
    assert big.minutes[1] > 60

    small = load_job(overrides={"bbox": [14.2, 50.8, 14.3, 50.9], "start": "2010-01-01", "sensors": ["viirs-snpp"]})
    est = estimate(small)
    assert any("smaller than one" in w for w in est.warnings)
    assert any("viirs-snpp has no data before" in w for w in est.warnings)
    assert not est.errors

    monkeypatch.delenv("FIRMS_MAP_KEY")
    assert any("FIRMS key" in e for e in estimate(small).errors)


def test_osm_zoom_over_tile_limit_is_an_error():
    est = estimate(load_job(example="australia-2019-20", overrides={"osm_zoom": 12}))
    assert any("tiles" in e for e in est.errors)


def test_cli_dry_run_and_hints():
    runner = CliRunner()
    r = runner.invoke(main, ["fetch", "--example", "ceske-svycarsko-2022", "--dry-run"])
    assert r.exit_code == 0, r.output
    assert "EWDS request" in r.output and "FIRMS request" in r.output

    r = runner.invoke(main, ["fetch"])
    assert r.exit_code == 2 and "--example ceske-svycarsko-2022" in r.output

    r = runner.invoke(main, ["fetch", "--bbox", "10,0,5,1", "--start", "2023-01-01"])
    assert r.exit_code == 2 and "West must be less than east" in r.output

    r = runner.invoke(main, ["examples", "australia-2019-20"])
    assert r.exit_code == 0 and r.output.startswith("# Australian")


def test_cli_runs_job_and_reports_failures(monkeypatch, tmp_path):
    calls = []

    def fake_run(job):
        calls.append(job)
        return jobmod.Result(outputs={"fires": tmp_path / "fires.gpkg"}, failures={"indices": "queue timeout"})

    monkeypatch.setattr("fire_cli.cli.run", fake_run)
    r = CliRunner().invoke(main, ["fetch", "--bbox", "14,50,15,51", "--start", "2023-07-01", "-o", str(tmp_path), "-y"])
    assert r.exit_code == 1
    assert "queue timeout" in r.output and "Rerun" in r.output
    assert calls[0].output == str(tmp_path)


AVAILABILITY = "data_id,min_date,max_date\nVIIRS_SNPP_SP,2012-01-20,2026-03-31\nVIIRS_SNPP_NRT,2026-04-01,2026-10-18\n"


def _cache_everything(job):
    """Write the FIRMS availability list and an empty file for every request the job makes."""
    from fire_cli.sources import cems, firms

    (firms.default_cache()).mkdir(parents=True, exist_ok=True)
    (firms.default_cache() / "availability.csv").write_text(AVAILABILITY)
    client = firms.FirmsClient(cache=firms.default_cache())
    for chunk in firms.plan_chunks(job.area, job.dates, job.sensors, client, firms.default_cache()):
        chunk.path.write_text("latitude,longitude,acq_date,acq_time,confidence,version,frp\n")
    cems.default_cache().mkdir(parents=True, exist_ok=True)
    for m in cems.plan_months(job.area, job.dates, job.indices, job.grid):
        (cems.default_cache() / f"{m.stem}.nc").write_bytes(b"")


@pytest.fixture
def no_keys(monkeypatch, tmp_path):
    monkeypatch.delenv("FIRMS_MAP_KEY", raising=False)
    monkeypatch.delenv("EWDS_API_KEY", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))  # no ~/.cdsapirc


def test_fully_cached_job_needs_no_keys(no_keys):
    job = load_job(overrides={"bbox": [14, 50.7, 14.6, 51], "start": "2023-07-01", "end": "2023-07-12",
                              "indices": ["fwi"], "sensors": ["viirs-snpp"], "osm": False})
    _cache_everything(job)
    est = estimate(job)
    assert not est.errors
    assert sum("all cached" in line for line in est.lines) == 2
    assert est.minutes == (0, 0)


def test_missing_parts_need_keys(no_keys):
    job = load_job(overrides={"bbox": [14, 50.7, 14.6, 51], "start": "2023-07-01", "end": "2023-07-12",
                              "indices": ["fwi"], "sensors": ["viirs-snpp"], "osm": False})
    _cache_everything(job)
    longer = load_job(overrides={**{k: getattr(job, k) for k in ("bbox", "indices", "sensors", "osm")},
                                 "start": "2023-07-01", "end": "2023-07-20"})
    est = estimate(longer)
    # FIRMS: 01-05, 06-10 are cached; 11-15, 16-20 are new (11-12 was cached as part of a shorter chunk).
    assert any("No FIRMS key, and 2 request(s)" in e for e in est.errors)
    assert any("~4 FIRMS request(s)" in line and "2 to download" in line for line in est.lines)
    # EWDS caches per requested day range, so July 1-20 is a new request even though 1-12 is cached.
    assert any("No EWDS key, and 1 month(s)" in e for e in est.errors)

    # Only the indices missing: FIRMS needs no key.
    shorter_fires = load_job(overrides={**{k: getattr(job, k) for k in ("bbox", "sensors", "osm")},
                                        "indices": ["fwi", "ffdi"], "start": "2023-07-01", "end": "2023-07-12"})
    errors = estimate(shorter_fires).errors
    assert len(errors) == 1 and "EWDS" in errors[0]


def test_nothing_cached_and_no_keys(no_keys):
    est = estimate(load_job(example="ceske-svycarsko-2022"))
    assert any("No FIRMS key" in e for e in est.errors)
    assert any("No EWDS key" in e for e in est.errors)


def test_cached_download_without_key(no_keys):
    from fire_cli.sources import firms

    job = load_job(overrides={"bbox": [14, 50.7, 14.6, 51], "start": "2023-07-01", "end": "2023-07-03",
                              "sensors": ["viirs-snpp"]})
    _cache_everything(job)
    chunk = next(iter(firms.default_cache().glob("VIIRS_SNPP_SP_*.csv")))
    chunk.write_text("latitude,longitude,acq_date,acq_time,confidence,version,frp\n50.8,14.3,2023-07-02,130,n,2,4.0\n")
    gdf = firms.download_detections(job.area, job.dates, job.sensors)
    assert len(gdf) == 1
