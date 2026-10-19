# fire-cli

Download fire danger indices and active-fire detections for any area and time
interval, and turn them into GIS-ready rasters (GeoTIFF) and animations.

## Data sources

| Source | What | Coverage | Access |
|---|---|---|---|
| [CEMS fire danger indices](https://ewds.climate.copernicus.eu/datasets/cems-fire-historical-v1) (ERA5-based) | Canadian FWI system, McArthur FFDI, US NFDRS | Global, daily, 1940–present, 0.25° | Copernicus EWDS API |
| [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/) | Active-fire detections (VIIRS, MODIS, Landsat) | Global, MODIS 2000+, VIIRS 2012+ | FIRMS area API |
| [Natural Earth](https://www.naturalearthdata.com/) | Basemap: countries, lakes, rivers, towns | Global, 1:10m | Direct download, no key |
| [OpenStreetMap](https://www.openstreetmap.org/) tiles | Detailed background map | Global | Tile server, no key |

Available indices: `fwi ffmc dmc dc isi bui dsr` (Canadian), `ffdi df kbdi`
(McArthur Mark 5), `bi erc ic sc` (US NFDRS).

Sensors: `viirs-snpp viirs-noaa20 viirs-noaa21 modis landsat`.

## Setup

```sh
python3 -m virtualenv .venv   # or python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

Credentials (both free):

1. **EWDS**: register at https://ewds.climate.copernicus.eu, copy your API token
   from your profile, and **accept the licence** on the dataset's *Download* tab.
   `export EWDS_API_KEY=<token>`
2. **FIRMS**: request a MAP_KEY at https://firms.modaps.eosdis.nasa.gov/api/map_key/.
   `export FIRMS_MAP_KEY=<key>`

Raw downloads are cached under `~/.cache/fire-cli` (override with `FIRE_CLI_CACHE`).
Keys are only needed for parts that are not cached yet: a fully cached job, and
`render`/`animate`, run without any keys. The last 3 days of FIRMS near-real-time
data are never cached because FIRMS still updates them.

## Command line

```sh
fire-cli examples                                   # ready-made jobs
fire-cli fetch --example ceske-svycarsko-2022 --dry-run   # plan, time estimate, warnings
fire-cli fetch --example ceske-svycarsko-2022       # Czech wildfire, July 2022 (small)
fire-cli fetch --example australia-2019-20          # Black Summer 2019/20 (large, hours)
fire-cli fetch --bbox 14,50.7,14.6,51 --start 2022-07-20 --end 2022-08-05 -i fwi -s viirs-snpp
fire-cli render fire-cli-output/ceske-svycarsko-2022     # GeoTIFFs + QGIS styles
fire-cli animate fire-cli-output/ceske-svycarsko-2022    # day-by-day GIF
fire-cli animate fire-cli-output/australia-2019-20 -o black-summer.mp4 --fps 6
fire-cli info                                       # indices, sensors, API key status
```

`render` and `animate` take a fetch output directory or its job file. Setting
`render: true` and `animation: animation.gif` in the job file runs them right
after `fetch`.

### Job files

Everything can also live in a YAML job file. Print an example as a template,
edit it, and pass it to `fetch`; command line options still override it:

```sh
fire-cli examples ceske-svycarsko-2022 > my-fire.yaml
fire-cli fetch my-fire.yaml
fire-cli fetch my-fire.yaml --start 2022-07-24 --sensors modis
```

```yaml
name: my-fire
bbox: [13.95, 50.75, 14.65, 51.0]   # west, south, east, north; or area_file: area.geojson
start: 2022-07-20
end: 2022-08-10
indices: [fwi, ffdi]                # [] to skip
grid: 0.25
sensors: [viirs-snpp, modis]        # [] to skip
basemap: true                       # Natural Earth layers
osm: true                           # OpenStreetMap background GeoTIFF
osm_zoom: null                      # null = automatic
output: fire-cli-output/my-fire
```

### Output

```
fire-cli-output/<name>/
  job.yaml          the resolved job, to rerun or share
  fires.gpkg        layers: fires, basemap_countries, basemap_lakes, basemap_rivers, basemap_places
  osm_basemap.tif   OpenStreetMap background (EPSG:3857)
  indices.nc        daily fire danger indices (NetCDF, opens in QGIS)
  rasters/          from `fire-cli render`, each .tif with a .qml style QGIS loads automatically
    fwi_daily.tif     one band per day (band names are the dates); likewise for every index
    fwi_max.tif       period maximum (also _mean, _min with --stats; one file per day with --per-day)
    fires_count.tif   detections per cell (~375 m, coarser for large areas)
    fires_frp.tif     summed fire radiative power (MW)
    fires_first_day.tif  day each cell was first detected burning (1 = start date): fire progression
  animation.gif     from `fire-cli animate` (.mp4 also supported)
```

FWI and FFDI are styled with their official danger classes (EFFIS for FWI,
McArthur ratings for FFDI); other indices get a continuous ramp. Each animation
frame shows the day's index classes over a muted basemap, that day's detections
as dark dots, earlier detections as grey cells, and timelines of the area-mean
index and detections per day. Fire dates are UTC (FIRMS `acq_date`).

### Limits and timing

`fetch` prints an estimate and warnings before downloading (and asks before long jobs; `-y` skips that).

- **EWDS (indices):** one request per month. Processing takes ~2 min, but requests
  wait in a shared queue (measured: 1–2 h each) and often run one at a time per user,
  so a 5-month job can take several hours. Max 3720 fields per request, never
  reached by the monthly split. Finished months are cached, so you can stop and rerun.
- **FIRMS (fires):** at most 5 days per request, 5000 transactions per 10 min per key.
  Large areas over long periods return hundreds of thousands of detections.
- **OpenStreetMap:** at most 64 tiles per map (zoom chosen automatically), cached,
  following the [tile usage policy](https://operations.osmfoundation.org/policies/tiles/).
  Map data © OpenStreetMap contributors.

Everything is cached, so rerunning an interrupted job only downloads what is missing.

## Python API

```python
from fire_cli import BBox, DateRange
from fire_cli.sources import basemap, cems, firms

bbox = BBox(west=12.0, south=48.5, east=19.0, north=51.1)   # Czechia
dates = DateRange.parse("2022-07-20", "2022-08-10")

paths = cems.download_indices(bbox, dates, ["fwi", "ffdi"])
ds = cems.open_indices(paths)                     # xarray.Dataset (time, latitude, longitude)

fires = firms.download_detections(bbox, dates, ["viirs-snpp", "modis"])  # GeoDataFrame
fires.to_file("fires.gpkg", layer="fires")

for name, layer in basemap.download_basemap(bbox).items():
    layer.to_file("fires.gpkg", layer=f"basemap_{name}")
```

## Tests

```sh
.venv/bin/pytest
```
