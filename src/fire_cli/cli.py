"""fire-cli command line interface."""

from __future__ import annotations

import logging
import sys

import click

from fire_cli import __version__
from fire_cli.job import Job, JobError, estimate, example_names, example_text, fmt_minutes, load_job, run
from fire_cli.sources import cems, firms

EXAMPLES_HINT = """
\b
Examples:
  fire-cli examples                              list ready-made jobs
  fire-cli fetch --example ceske-svycarsko-2022  Czech wildfire, July 2022 (quick)
  fire-cli fetch --example australia-2019-20     Black Summer bushfires (large, hours)
  fire-cli examples australia-2019-20 > job.yaml then edit job.yaml and run:
    fire-cli fetch job.yaml
  fire-cli fetch --bbox 14,50.7,14.6,51 --start 2022-07-20 --end 2022-08-05
  fire-cli render fire-cli-output/ceske-svycarsko-2022     GeoTIFFs for QGIS
  fire-cli animate fire-cli-output/ceske-svycarsko-2022    day-by-day GIF
"""


def _split(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [s.strip() for s in value.split(",") if s.strip()]


@click.group(epilog=EXAMPLES_HINT)
@click.version_option(__version__)
@click.option("-v", "--verbose", count=True, help="More log output (-vv for debug).")
@click.option("-q", "--quiet", is_flag=True, help="Only warnings and errors.")
def main(verbose: int, quiet: bool) -> None:
    """Download fire danger indices and active-fire detections for any area and time
    interval, ready to open in QGIS.

    Data: CEMS fire danger indices (Copernicus EWDS), NASA FIRMS active fires,
    Natural Earth and OpenStreetMap basemaps.
    """
    level = logging.WARNING if quiet else logging.DEBUG if verbose > 1 else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    if verbose < 1:
        # Library chatter (HTTP, GDAL) only with -v.
        for noisy in ("urllib3", "pyogrio", "rasterio", "fiona", "multiurl"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


@main.command(epilog=EXAMPLES_HINT)
@click.argument("job_file", required=False, type=click.Path(exists=True, dir_okay=False))
@click.option("-e", "--example", help="Start from a built-in example job (see `fire-cli examples`).")
@click.option("--bbox", help="Area as west,south,east,north in degrees.")
@click.option("--area-file", type=click.Path(exists=True, dir_okay=False), help="Vector file whose extent is the area.")
@click.option("--start", help="First day, YYYY-MM-DD.")
@click.option("--end", help="Last day, YYYY-MM-DD (default: same as start).")
@click.option("-i", "--indices", help="Comma-separated fire danger indices, e.g. fwi,ffdi. Empty string skips them.")
@click.option("-s", "--sensors", help="Comma-separated FIRMS sensors, e.g. viirs-snpp,modis. Empty string skips them.")
@click.option("--grid", type=click.Choice(sorted(cems.GRIDS)), help="Fire danger index grid in degrees.")
@click.option("--basemap/--no-basemap", default=None, help="Natural Earth borders, rivers, towns.")
@click.option("--osm/--no-osm", default=None, help="OpenStreetMap background GeoTIFF.")
@click.option("--osm-zoom", type=click.IntRange(0, 16), help="OSM zoom level (default: automatic).")
@click.option("-o", "--output", type=click.Path(file_okay=False), help="Output directory.")
@click.option("-n", "--name", help="Job name (used for the default output directory).")
@click.option("--dry-run", is_flag=True, help="Only show what would be downloaded and how long it may take.")
@click.option("--render/--no-render", "render_", default=None, help="Also write GeoTIFFs after downloading.")
@click.option("--animation", help="Also write an animation, e.g. animation.gif or animation.mp4.")
@click.option("-y", "--yes", is_flag=True, help="Don't ask for confirmation before long downloads.")
def fetch(job_file, example, bbox, area_file, start, end, indices, sensors, grid, basemap, osm, osm_zoom,
          output, name, render_, animation, dry_run, yes):
    """Download data for an area and time interval.

    Settings come from JOB_FILE (YAML, see `fire-cli examples NAME` for the
    format) and/or an --example; any option given here overrides them.
    """
    overrides = {
        "name": name,
        "bbox": [float(v) for v in _split(bbox)] if bbox else None,
        "area_file": area_file,
        "start": start,
        "end": end,
        "indices": _split(indices),
        "sensors": _split(sensors),
        "grid": grid,
        "basemap": basemap,
        "osm": osm,
        "osm_zoom": osm_zoom,
        "output": output,
        "render": render_,
        "animation": animation,
    }
    if not (job_file or example or bbox or area_file):
        raise click.UsageError(
            "Give a job file, an --example, or --bbox/--area-file with --start.\n"
            "Try: fire-cli fetch --example ceske-svycarsko-2022 --dry-run"
        )
    try:
        job = load_job(job_file, example, overrides)
        job.validate()
        est = estimate(job)
    except (JobError, ValueError) as exc:
        raise click.UsageError(str(exc)) from exc

    _print_estimate(job, est)
    if job.render or job.animation:
        click.echo(f"  Then: {'GeoTIFFs' if job.render else ''}{' + ' if job.render and job.animation else ''}"
                   f"{job.animation or ''} (seconds to a few minutes)\n")
    if est.errors:
        for e in est.errors:
            click.secho(f"✗ {e}", fg="red", err=True)
        sys.exit(2)
    if dry_run:
        return
    if est.minutes[1] > 20 and not yes and sys.stdin.isatty():
        click.confirm("This may take a while. Continue?", default=True, abort=True)

    result = run(job)

    click.echo()
    for kind, path in result.outputs.items():
        click.secho(f"✓ {kind:8s} {path}", fg="green")
    for kind, err in result.failures.items():
        click.secho(f"✗ {kind:8s} {err}", fg="red", err=True)
    if "osm" in result.outputs:
        click.echo("  Map tiles © OpenStreetMap contributors. Natural Earth basemap is public domain.")
    if result.failures:
        click.echo("Rerun the same command to retry; finished parts are cached.", err=True)
        sys.exit(1)


def _print_estimate(job: Job, est) -> None:
    click.secho(f"{job.name}" + (f" — {job.description}" if job.description else ""), bold=True)
    for line in est.lines:
        click.echo(f"  • {line}")
    lo, hi = est.minutes
    click.echo(f"  Estimated time: {fmt_minutes(lo)} – {fmt_minutes(hi)} (cached parts are skipped)")
    click.echo(f"  Output: {job.output_dir}/")
    for w in est.warnings:
        click.secho(f"  ! {w}", fg="yellow")
    click.echo()


OUTPUT_ARG = click.argument("path", type=click.Path())


@main.command()
@OUTPUT_ARG
@click.option("-i", "--indices", help="Comma-separated indices to render (default: all downloaded).")
@click.option("--stats", default="max,mean", show_default=True, help="Period statistics: any of max,mean,min.")
@click.option("--per-day", is_flag=True, help="Also write one GeoTIFF per index per day.")
@click.option("--fire-res", type=float, help="Fire raster cell size in degrees (default ~375 m, coarser for big areas).")
@click.option("-o", "--out-dir", type=click.Path(file_okay=False), help="Where to write (default: PATH/rasters).")
def render(path, indices, stats, per_day, fire_res, out_dir):
    """Write GeoTIFFs from a fetch output directory (or its job file).

    \b
    Per index:  <index>_daily.tif (one band per day), <index>_max.tif, <index>_mean.tif
    Fires:      fires_count.tif, fires_frp.tif (fire radiative power), fires_first_day.tif
    Each comes with a .qml style that QGIS applies automatically.
    """
    from fire_cli import render as rendermod

    try:
        files = rendermod.render(path, indices=_split(indices), stats=tuple(_split(stats) or ()),
                                 per_day=per_day, fire_res=fire_res, out_dir=out_dir)
    except (FileNotFoundError, ValueError, JobError) as exc:
        raise click.UsageError(str(exc)) from exc
    if not files:
        raise click.ClickException("Nothing to render: no indices.nc or fire detections found")
    for f in files:
        click.secho(f"✓ {f}", fg="green")


@main.command()
@OUTPUT_ARG
@click.option("-o", "--out", type=click.Path(dir_okay=False),
              help="Output file, .gif or .mp4 (default: PATH/animation.gif).")
@click.option("-i", "--index", help="Index to show (default: fwi, else ffdi, else the first downloaded).")
@click.option("--fps", type=click.FloatRange(0.1, 30), default=2, show_default=True, help="Frames (days) per second.")
@click.option("--width", type=click.IntRange(400, 3000), default=1000, show_default=True, help="Frame width in pixels.")
@click.option("--basemap", type=click.Choice(["auto", "osm", "naturalearth", "none"]), default="auto",
              show_default=True, help="Background: OSM tiles if downloaded, else Natural Earth.")
@click.option("--frames-dir", type=click.Path(file_okay=False), help="Also save every frame as PNG here.")
def animate(path, out, index, fps, width, basemap, frames_dir):
    """Day-by-day animation of fire danger and fire detections.

    PATH is a fetch output directory or its job file. Long intervals are much
    smaller as .mp4 than as .gif.
    """
    from fire_cli import animate as animmod

    try:
        result = animmod.animate(path, out=out, index=index, fps=fps, width_px=width, basemap=basemap,
                                 frames_dir=frames_dir)
    except (FileNotFoundError, ValueError, JobError) as exc:
        raise click.UsageError(str(exc)) from exc
    click.secho(f"✓ {result}", fg="green")


@main.command()
@click.argument("name", required=False)
def examples(name):
    """List example jobs, or print one as a YAML job file to copy and edit."""
    if name:
        try:
            click.echo(example_text(name), nl=False)
        except JobError as exc:
            raise click.UsageError(str(exc)) from exc
        return
    for ex in example_names():
        job = load_job(example=ex)
        click.echo(f"  {ex:26s} {job.description}")
    click.echo("\nRun one:       fire-cli fetch --example NAME [--dry-run]")
    click.echo("Make your own: fire-cli examples NAME > job.yaml, edit it, then fire-cli fetch job.yaml")


@main.command()
def info():
    """Available indices and sensors, and whether the API keys are set."""
    import os

    from fire_cli.job import FIRMS_FIRST_DAY

    click.secho("Fire danger indices (CEMS, daily, 1940–present, --indices):", bold=True)
    system = None
    for short, idx in cems.INDICES.items():
        if idx.system != system:
            system = idx.system
            click.echo(f"  {system}")
        click.echo(f"    {short:6s} {idx.description}")

    click.secho("\nActive-fire sensors (NASA FIRMS, --sensors):", bold=True)
    for sensor in firms.SENSORS:
        click.echo(f"  {sensor:14s} from {FIRMS_FIRST_DAY[sensor]}")

    click.secho("\nAPI keys:", bold=True)
    from pathlib import Path

    ewds = os.environ.get("EWDS_API_KEY") or Path("~/.cdsapirc").expanduser().exists()
    for label, ok, how in [
        ("EWDS_API_KEY", ewds, "https://ewds.climate.copernicus.eu/profile (also accept the dataset licence)"),
        ("FIRMS_MAP_KEY", os.environ.get("FIRMS_MAP_KEY"), "https://firms.modaps.eosdis.nasa.gov/api/map_key/"),
    ]:
        if ok:
            click.secho(f"  ✓ {label}", fg="green")
        else:
            click.secho(f"  ✗ {label} not set — get one at {how}", fg="red")


if __name__ == "__main__":
    main()
