"""A download job: what to fetch, a pre-flight estimate, and running it.

A job comes from (lowest to highest priority) a built-in example, a YAML job
file, and command line options.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field, fields
from datetime import date, timedelta
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from fire_cli import config
from fire_cli.query import BBox, DateRange
from fire_cli.sources import cems, firms, osm

log = logging.getLogger(__name__)

# First day of data for each FIRMS sensor (standard processing archive, or NRT if there is no archive).
FIRMS_FIRST_DAY = {
    "viirs-snpp": date(2012, 1, 20),
    "viirs-noaa20": date(2018, 4, 1),
    "viirs-noaa21": date(2023, 3, 1),
    "modis": date(2000, 11, 1),
    "landsat": date(2022, 6, 20),
}
CEMS_FIRST_DAY = date(1940, 1, 1)

# Rough timings used for the estimate. EWDS time depends on its queue and grows with
# the number of fields (days x indices) per request, not with the area.
# Measured Oct 2026: processing takes ~2 min, but requests waited 1-2 h in the queue and
# ran one at a time per user, so the upper bound assumes no parallelism.
EWDS_MINUTES_PER_REQUEST = (10, 120)
FIRMS_SECONDS_PER_REQUEST = (1, 5)
FIRMS_SECONDS_PER_REQUEST_BIG = (10, 60)
FIRMS_TRANSACTIONS_PER_10_MIN = 5000
# Above this (square degrees x days) FIRMS responses get large; e.g. České Švýcarsko ~4, Australia ~25 000.
FIRMS_BIG_JOB = 2000


class JobError(ValueError):
    pass


@dataclass
class Job:
    name: str = "fire-job"
    description: str = ""
    bbox: list[float] | None = None
    area_file: str | None = None
    start: date | None = None
    end: date | None = None
    indices: list[str] = field(default_factory=lambda: ["fwi"])
    grid: str = "0.25"
    sensors: list[str] = field(default_factory=lambda: ["viirs-snpp"])
    basemap: bool = True
    osm: bool = True
    osm_zoom: int | None = None
    output: str | None = None
    render: bool = False          # write GeoTIFFs after downloading
    animation: str | None = None  # e.g. animation.gif or animation.mp4, relative to the output directory

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Job":
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise JobError(f"Unknown job setting(s) {unknown}; valid settings are {sorted(known)}")
        data = dict(data)
        for key in ("start", "end"):
            if isinstance(data.get(key), str):
                data[key] = date.fromisoformat(data[key])
        for key in ("indices", "sensors"):
            if isinstance(data.get(key), str):
                data[key] = [s.strip() for s in data[key].split(",") if s.strip()]
            elif data.get(key) is None and key in data:
                data[key] = []
        if data.get("grid") is not None:
            data["grid"] = f"{float(data['grid']):g}"
        return cls(**data)

    def to_yaml(self) -> str:
        data = asdict(self)
        for key in ("start", "end"):
            if data[key]:
                data[key] = data[key].isoformat()
        return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)

    # -- resolved values ------------------------------------------------------

    @property
    def area(self) -> BBox:
        if self.area_file:
            return BBox.from_file(self.area_file)
        if self.bbox:
            if len(self.bbox) != 4:
                raise JobError("bbox needs four numbers: [west, south, east, north]")
            return BBox(*map(float, self.bbox))
        raise JobError("No area given: set bbox or area_file (try `fire-cli examples` for templates)")

    @property
    def dates(self) -> DateRange:
        if not self.start:
            raise JobError("No start date given")
        return DateRange(self.start, self.end or self.start)

    @property
    def output_dir(self) -> Path:
        return Path(self.output or Path("fire-cli-output") / self.name)

    def validate(self) -> None:
        self.area, self.dates  # noqa: B018 - raise early on a bad area or date
        cems.resolve_indices(self.indices)
        if self.grid not in cems.GRIDS:
            raise JobError(f"grid must be one of {sorted(cems.GRIDS)}")
        bad = [s for s in self.sensors if s not in firms.SENSORS]
        if bad:
            raise JobError(f"Unknown sensor(s) {bad}; choose from {sorted(firms.SENSORS)}")
        if self.dates.start > date.today():
            raise JobError("The interval starts in the future")


def example_names() -> list[str]:
    return sorted(p.name.removesuffix(".yaml") for p in resources.files("fire_cli.examples").iterdir() if p.name.endswith(".yaml"))


def example_text(name: str) -> str:
    path = resources.files("fire_cli.examples") / f"{name}.yaml"
    if not path.is_file():
        raise JobError(f"No example {name!r}; available: {', '.join(example_names())}")
    return path.read_text(encoding="utf-8")


def load_job(job_file: str | Path | None = None, example: str | None = None, overrides: dict | None = None) -> Job:
    """Merge example < job file < overrides (None values in ``overrides`` are ignored)."""
    data: dict[str, Any] = {}
    if example:
        data.update(yaml.safe_load(example_text(example)) or {})
    if job_file:
        job_file = Path(job_file)
        try:
            loaded = yaml.safe_load(job_file.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise JobError(f"{job_file} is not valid YAML: {exc}") from exc
        if not isinstance(loaded, dict):
            raise JobError(f"{job_file} should contain settings like 'bbox:' and 'start:'")
        # A relative area_file is relative to the job file, not the current directory.
        if loaded.get("area_file") and not Path(loaded["area_file"]).is_absolute():
            loaded["area_file"] = str(job_file.parent / loaded["area_file"])
        data.update(loaded)
    for key, value in (overrides or {}).items():
        if value is not None:
            data[key] = value
    if "bbox" in (overrides or {}) and overrides["bbox"] is not None:
        data.pop("area_file", None)  # an explicit --bbox wins over the job file's area
    return Job.from_dict(data)


# -- estimate -----------------------------------------------------------------


@dataclass
class Estimate:
    lines: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    minutes: tuple[float, float] = (0.0, 0.0)

    def add_minutes(self, low: float, high: float) -> None:
        self.minutes = (self.minutes[0] + low, self.minutes[1] + high)


def fmt_minutes(m: float) -> str:
    if m < 1:
        return "<1 min"
    if m < 90:
        return f"{m:.0f} min"
    return f"{m / 60:.1f} h"


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def estimate(job: Job) -> Estimate:
    """Requests, size and time the job needs, plus warnings about limits and coverage."""
    est = Estimate()
    bbox, dates = job.area, job.dates
    today = date.today()
    est.lines.append(
        f"Area {bbox.west:g},{bbox.south:g} → {bbox.east:g},{bbox.north:g} "
        f"({bbox.east - bbox.west:.2f}° × {bbox.north - bbox.south:.2f}°), "
        f"{dates.start} → {dates.end} ({dates.days} days)"
    )
    if dates.end > today:
        est.warnings.append(f"The interval ends in the future; only data up to {today} can exist.")

    if job.indices:
        indices = cems.resolve_indices(job.indices)
        plan = cems.plan_months(bbox, dates, indices, job.grid)
        missing = [m for m in plan if not m.cached_files(cems.default_cache())]
        g = float(job.grid)
        cells = (math.ceil(max(bbox.east - bbox.west, g) / g) + 1) * (math.ceil(max(bbox.north - bbox.south, g) / g) + 1)
        size = cells * dates.days * len(indices) * 4
        lo, hi = EWDS_MINUTES_PER_REQUEST
        est.add_minutes(math.ceil(len(missing) / 3) * lo, len(missing) * hi)
        status = "all cached" if not missing else f"{len(missing)} to download" if len(missing) < len(plan) else ""
        est.lines.append(
            f"Fire danger indices ({', '.join(indices)}): {len(plan)} EWDS request(s)"
            + (f" ({status})" if status else "")
            + f", up to {len(indices) * 31} fields each (limit 3720), ~{cells} grid cells, ~{_fmt_bytes(size)}"
        )
        if missing:
            est.warnings.append(
                f"EWDS requests wait in a shared queue: {lo} min to {hi // 60} h each, and often run one at a "
                "time. The rest of the job finishes first; you can stop and rerun later (finished months are "
                "cached) and follow the queue at https://ewds.climate.copernicus.eu/requests"
            )
        if dates.start < CEMS_FIRST_DAY:
            est.warnings.append(f"Fire danger indices start on {CEMS_FIRST_DAY}.")
        if dates.end > today - timedelta(days=3):
            est.warnings.append("The newest fire danger data lags a few days behind today; the last days may be missing.")
        if bbox.east - bbox.west < g or bbox.north - bbox.south < g:
            est.warnings.append(
                f"The area is smaller than one {g}° grid cell (~{g * 111:.0f} km); "
                "the indices will be a single coarse value for the whole area."
            )
        # Only months that still have to be downloaded need a key.
        if missing and not (config.ewds_api_key() or Path("~/.cdsapirc").expanduser().exists()):
            est.errors.append(
                f"No EWDS key, and {len(missing)} month(s) are not cached: export EWDS_API_KEY=<token> "
                "(from https://ewds.climate.copernicus.eu/profile) and accept the dataset licence on its Download tab."
            )

    if job.sensors:
        volume = (bbox.east - bbox.west) * (bbox.north - bbox.south) * dates.days
        lo, hi = FIRMS_SECONDS_PER_REQUEST_BIG if volume > FIRMS_BIG_JOB else FIRMS_SECONDS_PER_REQUEST
        client = firms.FirmsClient(cache=firms.default_cache())
        try:
            chunks = firms.plan_chunks(bbox, dates, job.sensors, client, firms.default_cache())
            total, n = len(chunks), sum(not c.cached for c in chunks)
        except (config.MissingCredentialsError, firms.FirmsError, OSError) as exc:
            # No key and nothing cached yet (or FIRMS unreachable): estimate from the interval alone.
            log.debug("FIRMS plan unavailable: %s", exc)
            total = n = math.ceil(dates.days / firms.MAX_DAYS_PER_REQUEST) * len(job.sensors)
        est.add_minutes(n * lo / 60, n * hi / 60)
        if n == 0:
            detail = "all cached"
        else:
            detail = (f"{n} to download, " if n < total else "") + f"{fmt_minutes(n * lo / 60)} – {fmt_minutes(n * hi / 60)}"
        est.lines.append(
            f"Active fires ({', '.join(job.sensors)}): ~{total} FIRMS request(s) of up to "
            f"{firms.MAX_DAYS_PER_REQUEST} days, {detail}"
        )
        if volume > FIRMS_BIG_JOB and n:
            est.warnings.append(
                "Large area × long interval: expect hundreds of thousands of detections, "
                "slow FIRMS responses and a GeoPackage of hundreds of MB. Narrow the area or "
                "dates, or use a single VIIRS sensor, if you don't need all of it."
            )
        if n > FIRMS_TRANSACTIONS_PER_10_MIN / 10:
            est.warnings.append(
                f"FIRMS allows {FIRMS_TRANSACTIONS_PER_10_MIN} transactions per 10 minutes per key and large "
                "requests count as several; the download pauses and retries if the limit is hit."
            )
        for sensor in job.sensors:
            first = FIRMS_FIRST_DAY[sensor]
            if dates.end < first:
                est.warnings.append(f"{sensor} has no data before {first}; it will return nothing.")
            elif dates.start < first:
                est.warnings.append(f"{sensor} data starts on {first}; earlier days will be empty.")
        if "landsat" in job.sensors:
            est.warnings.append("Landsat detections cover only the USA and Canada.")
        if n and not client.has_key:
            recent = " (the last few days are never cached, NRT data still changes)" if (
                dates.end >= today - timedelta(days=firms.NRT_SETTLE_DAYS)) else ""
            est.errors.append(
                f"No FIRMS key, and {n} request(s) are not cached{recent}: export FIRMS_MAP_KEY=<key> "
                "(free at https://firms.modaps.eosdis.nasa.gov/api/map_key/)."
            )

    if job.basemap:
        est.lines.append("Basemap: Natural Earth borders, lakes, rivers, towns (~10 MB once, then cached)")
    if job.osm:
        zoom = job.osm_zoom if job.osm_zoom is not None else osm.choose_zoom(bbox)
        tiles = osm.tile_range(bbox, zoom).count
        est.lines.append(f"OpenStreetMap background: {tiles} tiles at zoom {zoom}")
        if tiles > osm.DEFAULT_MAX_TILES:
            est.errors.append(
                f"osm_zoom {zoom} needs {tiles} tiles, more than the {osm.DEFAULT_MAX_TILES} allowed by the OSM "
                "tile policy. Lower osm_zoom or leave it empty to choose automatically."
            )
        est.add_minutes(0, tiles * 0.5 / 60)

    return est


# -- run ------------------------------------------------------------------------


@dataclass
class Result:
    outputs: dict[str, Path] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)


def run(job: Job) -> Result:
    """Download everything the job asks for into ``job.output_dir``.

    Each source is independent: a failing source is reported and the others still run.
    """
    out = job.output_dir
    out.mkdir(parents=True, exist_ok=True)
    bbox, dates = job.area, job.dates
    result = Result()
    (out / "job.yaml").write_text(job.to_yaml(), encoding="utf-8")
    result.outputs["job"] = out / "job.yaml"
    gpkg = out / "fires.gpkg"

    def step(name: str, fn) -> None:
        try:
            fn()
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 - report and continue with the other sources
            log.error("%s failed: %s", name, exc)
            log.debug("Traceback", exc_info=True)
            result.failures[name] = str(exc)

    def fetch_indices() -> None:
        paths = cems.download_indices(bbox, dates, job.indices, grid=job.grid)
        ds = cems.open_indices(paths).sel(time=slice(str(dates.start), str(dates.end)))
        path = out / "indices.nc"
        ds.to_netcdf(path)
        result.outputs["indices"] = path

    def fetch_fires() -> None:
        gdf = firms.download_detections(bbox, dates, job.sensors)
        log.info("%d fire detections", len(gdf))
        gdf.to_file(gpkg, layer="fires")
        result.outputs["fires"] = gpkg

    def fetch_basemap() -> None:
        from fire_cli.sources import basemap

        for name, layer in basemap.download_basemap(bbox).items():
            layer.to_file(gpkg, layer=f"basemap_{name}")
        result.outputs["basemap"] = gpkg

    def fetch_osm() -> None:
        result.outputs["osm"] = osm.download_osm_basemap(bbox, out / "osm_basemap.tif", zoom=job.osm_zoom)

    # Quick sources first so their results are on disk while EWDS is queued.
    if job.basemap:
        step("basemap", fetch_basemap)
    if job.osm:
        step("osm", fetch_osm)
    if job.sensors:
        step("fires", fetch_fires)
    if job.indices:
        step("indices", fetch_indices)

    if job.render:
        def do_render() -> None:
            from fire_cli.render import render

            files = render(out)
            if files:
                result.outputs["rasters"] = files[0].parent
        step("render", do_render)
    if job.animation:
        def do_animate() -> None:
            from fire_cli.animate import animate

            result.outputs["animation"] = animate(out, out=out / job.animation)
        step("animation", do_animate)
    return result
