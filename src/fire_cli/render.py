"""Turn a fetch output directory into GIS-ready GeoTIFFs.

Every GeoTIFF gets a QGIS ``.qml`` style next to it, which QGIS applies
automatically when the file is opened.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from fire_cli import styles
from fire_cli.job import Job, load_job
from fire_cli.query import BBox, DateRange

log = logging.getLogger(__name__)

STATS = ("max", "mean", "min")
VIIRS_PIXEL_DEG = 0.00375  # ~375 m, the VIIRS I-band footprint
MAX_FIRE_GRID = 1000       # cells along the longer side of the fire rasters


@dataclass
class Outputs:
    """Everything `fire-cli fetch` wrote to an output directory."""

    dir: Path
    job: Job

    @classmethod
    def load(cls, path: str | Path) -> "Outputs":
        """``path`` is an output directory or a job file pointing to one."""
        path = Path(path).expanduser()
        if not path.exists():
            raise FileNotFoundError(
                f"{path} does not exist. PATH is the data to use: a `fire-cli fetch` output folder or its job "
                "file. Choose where files are written with -o (missing folders are created)."
            )
        if path.is_file():
            job = load_job(path)
            return cls(job.output_dir, job)
        job_file = path / "job.yaml"
        if not job_file.exists():
            raise FileNotFoundError(f"{path} has no job.yaml; run `fire-cli fetch` first")
        job = load_job(job_file)
        return cls(path, job)

    @property
    def bbox(self) -> BBox:
        return self.job.area

    @property
    def dates(self) -> DateRange:
        return self.job.dates

    @property
    def gpkg(self) -> Path:
        return self.dir / "fires.gpkg"

    @property
    def osm(self) -> Path | None:
        p = self.dir / "osm_basemap.tif"
        return p if p.exists() else None

    def indices(self):
        """Fire danger indices as an xarray Dataset, or None if not downloaded."""
        from fire_cli.sources import cems

        p = self.dir / "indices.nc"
        if not p.exists():
            return None
        ds = cems.open_indices([p])
        return ds.sel(time=slice(str(self.dates.start), str(self.dates.end)))

    def fires(self):
        """Fire detections as a GeoDataFrame, or None if not downloaded."""
        import geopandas as gpd
        import pyogrio

        if not self.gpkg.exists() or "fires" not in pyogrio.list_layers(self.gpkg)[:, 0]:
            return None
        return gpd.read_file(self.gpkg, layer="fires")

    def basemap(self) -> dict:
        import geopandas as gpd
        import pyogrio

        if not self.gpkg.exists():
            return {}
        names = [n for n in pyogrio.list_layers(self.gpkg)[:, 0] if n.startswith("basemap_")]
        return {n.removeprefix("basemap_"): gpd.read_file(self.gpkg, layer=n) for n in names}


def _write(path: Path, data: np.ndarray, transform, *, nodata, descriptions=None, qml: str | None = None,
           tags: dict | None = None) -> Path:
    import rasterio

    if data.ndim == 2:
        data = data[None]
    profile = {
        "driver": "GTiff", "width": data.shape[2], "height": data.shape[1], "count": data.shape[0],
        "dtype": data.dtype, "crs": "EPSG:4326", "transform": transform, "nodata": nodata,
        "compress": "deflate", "tiled": data.shape[1] >= 256 and data.shape[2] >= 256,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        for i, desc in enumerate(descriptions or [], start=1):
            dst.set_band_description(i, desc)
        if tags:
            dst.update_tags(**tags)
    if qml:
        path.with_suffix(".qml").write_text(qml, encoding="utf-8")
    return path


def _grid_transform(lat: np.ndarray, lon: np.ndarray):
    from rasterio.transform import from_origin

    res_x = float(abs(lon[1] - lon[0])) if len(lon) > 1 else 0.25
    res_y = float(abs(lat[1] - lat[0])) if len(lat) > 1 else res_x
    return from_origin(float(lon[0]) - res_x / 2, float(lat[0]) + res_y / 2, res_x, res_y)


def _index_style(name: str, values: np.ndarray) -> str:
    if name in styles.DANGER_CLASSES:
        return styles.qml_classes(styles.DANGER_CLASSES[name])
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return styles.qml_gradient(0, 1)
    return styles.qml_gradient(float(finite.min()), float(np.percentile(finite, 98)))


def render_indices(outputs: Outputs, out_dir: Path, *, indices: list[str] | None = None,
                   stats: tuple[str, ...] = ("max", "mean"), per_day: bool = False) -> list[Path]:
    """Per index: a multi-band daily GeoTIFF (one band per day), period statistics,
    and optionally one GeoTIFF per day."""
    ds = outputs.indices()
    if ds is None:
        log.warning("No indices.nc in %s; skipping fire danger rasters", outputs.dir)
        return []
    names = indices or list(ds.data_vars)
    missing = [n for n in names if n not in ds.data_vars]
    if missing:
        raise ValueError(f"Index {missing} not in {outputs.dir / 'indices.nc'} (has {list(ds.data_vars)})")

    transform = _grid_transform(ds.latitude.values, ds.longitude.values)
    days = [str(t)[:10] for t in ds.time.values]
    written = []
    for name in names:
        cube = ds[name].transpose("time", "latitude", "longitude").values.astype("float32")
        qml = _index_style(name, cube)
        tags = {"index": name, "source": "CEMS fire danger indices (ERA5), Copernicus EWDS"}
        written.append(_write(out_dir / f"{name}_daily.tif", cube, transform, nodata=np.nan,
                              descriptions=days, qml=qml, tags=tags))
        with np.errstate(all="ignore"):
            for stat in stats:
                if stat not in STATS:
                    raise ValueError(f"Unknown statistic {stat!r}; choose from {STATS}")
                fn = {"max": np.nanmax, "mean": np.nanmean, "min": np.nanmin}[stat]
                written.append(_write(out_dir / f"{name}_{stat}.tif", fn(cube, axis=0).astype("float32"),
                                      transform, nodata=np.nan, descriptions=[f"{stat} {days[0]}..{days[-1]}"],
                                      qml=qml, tags=tags))
        if per_day:
            for i, day in enumerate(days):
                written.append(_write(out_dir / name / f"{name}_{day}.tif", cube[i], transform, nodata=np.nan,
                                      descriptions=[day], qml=qml, tags=tags))
    return written


def fire_grid(bbox: BBox, res: float | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """Cell edges (lon, lat-descending) for fire rasters; default resolution ~375 m,
    coarser for large areas."""
    if res is None:
        res = max(VIIRS_PIXEL_DEG, max(bbox.east - bbox.west, bbox.north - bbox.south) / MAX_FIRE_GRID)
    nx = max(1, int(np.ceil((bbox.east - bbox.west) / res)))
    ny = max(1, int(np.ceil((bbox.north - bbox.south) / res)))
    lon_edges = bbox.west + np.arange(nx + 1) * res
    lat_edges = bbox.north - np.arange(ny + 1) * res
    return lon_edges, lat_edges, res


def render_fires(outputs: Outputs, out_dir: Path, *, res: float | None = None) -> list[Path]:
    """Detection count, summed fire radiative power, and day of first detection per cell."""
    from rasterio.transform import from_origin

    gdf = outputs.fires()
    if gdf is None or gdf.empty:
        log.warning("No fire detections in %s; skipping fire rasters", outputs.dir)
        return []
    bbox, dates = outputs.bbox, outputs.dates
    lon_edges, lat_edges, res = fire_grid(bbox, res)
    transform = from_origin(bbox.west, bbox.north, res, res)
    ny, nx = len(lat_edges) - 1, len(lon_edges) - 1

    # Row/column of every detection; points outside the box are dropped.
    col = np.floor((gdf["longitude"].to_numpy() - bbox.west) / res).astype(int)
    row = np.floor((bbox.north - gdf["latitude"].to_numpy()) / res).astype(int)
    inside = (col >= 0) & (col < nx) & (row >= 0) & (row < ny)
    col, row = col[inside], row[inside]
    flat = row * nx + col

    count = np.bincount(flat, minlength=ny * nx).reshape(ny, nx).astype("int32")
    frp = np.bincount(flat, weights=gdf["frp"].to_numpy()[inside], minlength=ny * nx).reshape(ny, nx)
    frp = np.where(count > 0, frp, np.nan).astype("float32")

    day_no = (gdf["acq_date"].map(lambda d: (np.datetime64(d) - np.datetime64(dates.start)).astype(int)) + 1)
    first = np.full(ny * nx, np.iinfo("int16").max, dtype="int32")
    np.minimum.at(first, flat, day_no.to_numpy()[inside].astype("int32"))
    first = np.where(first == np.iinfo("int16").max, 0, first).reshape(ny, nx).astype("int16")

    tags = {"source": "NASA FIRMS active fire detections", "resolution_deg": f"{res:g}"}
    finite_frp = frp[np.isfinite(frp)]
    return [
        _write(out_dir / "fires_count.tif", count, transform, nodata=0, descriptions=["detections"],
               qml=styles.qml_gradient(1, max(2, float(np.percentile(count[count > 0], 98))), integer=True), tags=tags),
        _write(out_dir / "fires_frp.tif", frp, transform, nodata=np.nan, descriptions=["sum of FRP (MW)"],
               qml=styles.qml_gradient(0, float(np.percentile(finite_frp, 98)) if finite_frp.size else 1), tags=tags),
        _write(out_dir / "fires_first_day.tif", first, transform, nodata=0,
               descriptions=[f"day of first detection (1 = {dates.start})"],
               qml=styles.qml_gradient(1, dates.days, styles.BLUES, integer=True), tags={**tags, "day_1": str(dates.start)}),
    ]


@dataclass
class RenderResult:
    files: list[Path] = field(default_factory=list)


def render(path: str | Path, *, indices: list[str] | None = None, stats: tuple[str, ...] = ("max", "mean"),
           per_day: bool = False, fire_res: float | None = None, out_dir: str | Path | None = None) -> list[Path]:
    outputs = Outputs.load(path)
    out = Path(out_dir or outputs.dir / "rasters")
    files = render_indices(outputs, out, indices=indices, stats=stats, per_day=per_day)
    files += render_fires(outputs, out, res=fire_res)
    return files
