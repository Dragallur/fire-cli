"""Day-by-day animation of fire danger and active fires over a basemap.

Each frame shows, in Web Mercator:
- the fire danger index of the day as semi-transparent danger classes,
- that day's fire detections as dark dots, earlier days' detections as grey cells,
- a timeline of the area-mean index and detections per day, with the current day marked.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from fire_cli import styles
from fire_cli.render import VIIRS_PIXEL_DEG, Outputs

log = logging.getLogger(__name__)

DPI = 100
INDEX_ALPHA = 0.55
PREFERRED_INDICES = ("fwi", "ffdi")
ATTRIBUTION = "Fire danger: CEMS/ERA5 (Copernicus) · Fires: NASA FIRMS"


def _to_mercator(lon, lat):
    from pyproj import Transformer

    t = Transformer.from_crs(4326, 3857, always_xy=True)
    return t.transform(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))


def _edges(centers: np.ndarray) -> np.ndarray:
    step = np.diff(centers).mean() if len(centers) > 1 else 0.25
    return np.concatenate([centers - step / 2, [centers[-1] + step / 2]])


def _muted_osm(path: Path):
    """OSM mosaic as a light greyscale image so the data colours stand out."""
    import rasterio

    with rasterio.open(path) as src:
        rgb = src.read().astype("float32")
        b = src.bounds
    grey = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    grey = 0.55 * grey + 0.45 * 255
    return np.repeat(grey[..., None], 3, axis=2).astype("uint8"), (b.left, b.right, b.bottom, b.top)


def _day_keys(series: pd.Series) -> np.ndarray:
    return pd.to_datetime(series).dt.strftime("%Y-%m-%d").to_numpy()


class Animator:
    def __init__(self, outputs: Outputs, *, index: str | None = None, basemap: str = "auto",
                 width_px: int = 1000):
        self.o = outputs
        self.days = [d.isoformat() for d in outputs.dates]
        bbox = outputs.bbox
        (self.x0, self.x1), (self.y0, self.y1) = _to_mercator([bbox.west, bbox.east], [bbox.south, bbox.north])

        # Fire danger index
        ds = outputs.indices()
        self.index = None
        if ds is not None and ds.data_vars:
            if index and index not in ds.data_vars:
                raise ValueError(f"Index {index!r} not downloaded; available: {list(ds.data_vars)}")
            self.index = index or next((i for i in PREFERRED_INDICES if i in ds.data_vars), list(ds.data_vars)[0])
            da = ds[self.index].transpose("time", "latitude", "longitude")
            by_day = {str(t)[:10]: i for i, t in enumerate(ds.time.values)}
            self.cube = da.values.astype("float32")
            self.cube_day = by_day
            lon_e, lat_e = _edges(ds.longitude.values), _edges(ds.latitude.values)
            self.mesh_x, _ = _to_mercator(lon_e, np.zeros_like(lon_e))
            _, self.mesh_y = _to_mercator(np.zeros_like(lat_e), lat_e)
            inside_lon = (ds.longitude.values >= bbox.west) & (ds.longitude.values <= bbox.east)
            inside_lat = (ds.latitude.values >= bbox.south) & (ds.latitude.values <= bbox.north)
            sub = self.cube[:, inside_lat][:, :, inside_lon]
            sub = sub if sub.size else self.cube  # area smaller than a cell: use the nearest cells
            with np.errstate(all="ignore"):
                means = np.nanmean(sub.reshape(len(self.cube), -1), axis=1)
            self.index_mean = np.array([means[by_day[d]] if d in by_day else np.nan for d in self.days])
            self.classes = styles.DANGER_CLASSES.get(self.index)
            finite = self.cube[np.isfinite(self.cube)]
            self.vmax = float(np.percentile(finite, 98)) if finite.size else 1.0

        # Fire detections
        gdf = outputs.fires()
        if gdf is not None and not gdf.empty:
            self.fx, self.fy = _to_mercator(gdf["longitude"], gdf["latitude"])
            self.fday = _day_keys(gdf["acq_date"])
            inside = (self.fx >= self.x0) & (self.fx <= self.x1) & (self.fy >= self.y0) & (self.fy <= self.y1)
            self.fx, self.fy, self.fday = self.fx[inside], self.fy[inside], self.fday[inside]
        else:
            self.fx = self.fy = np.array([])
            self.fday = np.array([], dtype=object)
        self.counts = np.array([(self.fday == d).sum() for d in self.days])

        if self.index is None and not len(self.fx):
            raise ValueError(f"Nothing to animate in {outputs.dir}: no indices.nc and no fire detections")

        self.basemap = basemap
        self.width_px = width_px
        self._build_figure()

    # -- figure -----------------------------------------------------------------

    def _build_figure(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap

        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "axes.edgecolor": styles.BASELINE,
            "axes.labelcolor": styles.INK_SECONDARY,
            "xtick.color": styles.INK_MUTED,
            "ytick.color": styles.INK_MUTED,
            "font.size": 9,
        })
        w_in = self.width_px / DPI
        aspect = (self.y1 - self.y0) / (self.x1 - self.x0)
        left, right = 0.07, 0.93
        # The map spans the plot width; very tall or flat areas are clamped (the map then narrows).
        map_h = float(np.clip((right - left) * w_in * aspect, 0.35 * w_in, 1.0 * w_in))
        has_index, has_fires = self.index is not None, len(self.fx) > 0
        rows = [0.8, map_h] + [1.0] * (int(has_index) + int(has_fires))
        gap = 0.45  # inches between rows
        height = sum(rows) + gap * (len(rows) - 1) + 0.45
        fig = plt.figure(figsize=(w_in, height), dpi=DPI, facecolor=styles.SURFACE)
        gs = fig.add_gridspec(len(rows), 1, height_ratios=rows, hspace=gap / np.mean(rows),
                              left=left, right=right, top=1 - 0.15 / height, bottom=0.3 / height)
        self.fig = fig

        # Header
        head = fig.add_subplot(gs[0])
        head.axis("off")
        title = self.o.job.description or self.o.job.name
        head.text(0, 0.95, title, fontsize=14, weight="bold", color=styles.INK, va="top", transform=head.transAxes)
        self.subtitle = head.text(0, 0.45, "", fontsize=11, color=styles.INK_SECONDARY, va="top",
                                  transform=head.transAxes)
        self._legend(head)

        # Map
        ax = fig.add_subplot(gs[1])
        self.ax = ax
        ax.set_xlim(self.x0, self.x1)
        ax.set_ylim(self.y0, self.y1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(styles.BASELINE)
            s.set_linewidth(0.8)
        ax.set_facecolor(styles.LAND)

        attribution = ATTRIBUTION
        if self.basemap in ("auto", "osm") and self.o.osm:
            img, extent = _muted_osm(self.o.osm)
            ax.imshow(img, extent=extent, zorder=0, interpolation="bilinear")
            attribution = "Map © OpenStreetMap contributors · " + attribution
        elif self.basemap in ("auto", "naturalearth"):
            self._natural_earth(ax)
        ax.set_xlim(self.x0, self.x1)
        ax.set_ylim(self.y0, self.y1)
        ax.text(0.995, 0.005, attribution, transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
                color=styles.INK_SECONDARY, zorder=10,
                bbox={"facecolor": styles.SURFACE, "alpha": 0.8, "edgecolor": "none", "pad": 2})

        if has_index:
            if self.classes:
                cmap = ListedColormap(styles.HEAT)
                norm = BoundaryNorm([-1e9, *self.classes.bounds, 1e9], cmap.N)
            else:
                cmap = LinearSegmentedColormap.from_list("heat", styles.HEAT)
                from matplotlib.colors import Normalize

                norm = Normalize(0, self.vmax)
            self.mesh = ax.pcolormesh(self.mesh_x, self.mesh_y, np.full(self.cube.shape[1:], np.nan),
                                      cmap=cmap, norm=norm, alpha=INDEX_ALPHA, shading="flat", zorder=1,
                                      edgecolors="none")
            self.cmap, self.norm = cmap, norm

        # Fire marks: dot diameter ~ the 375 m VIIRS footprint, kept between 3 and 9 px.
        m_per_px = (self.x1 - self.x0) / (ax.get_position().width * self.width_px)
        diam_px = float(np.clip(VIIRS_PIXEL_DEG * 111_000 / m_per_px, 3, 9))
        self.trail_cell = max(VIIRS_PIXEL_DEG * 111_000, 2 * m_per_px)
        nx = max(1, int((self.x1 - self.x0) / self.trail_cell))
        ny = max(1, int((self.y1 - self.y0) / self.trail_cell))
        self.trail_bins = (np.linspace(self.x0, self.x1, nx + 1), np.linspace(self.y0, self.y1, ny + 1))
        self.trail = np.zeros((nx, ny), dtype=bool)
        self.trail_img = ax.imshow(np.ma.masked_all((ny, nx)), extent=(self.x0, self.x1, self.y0, self.y1),
                                   origin="lower", cmap=ListedColormap([styles.INK_MUTED]), vmin=0, vmax=1,
                                   alpha=0.75, zorder=2, interpolation="nearest")
        size_pt2 = (diam_px * 72 / DPI) ** 2
        self.today = ax.scatter([], [], s=size_pt2, c=styles.INK, edgecolors="white", linewidths=0.8, zorder=3)

        # Timelines: one measure per panel, each on its own axis.
        x = np.arange(len(self.days))
        self.cursors = []
        row = 2
        tick_idx = np.unique(np.linspace(0, len(self.days) - 1, min(8, len(self.days))).round().astype(int))
        tick_labels = [pd.Timestamp(self.days[i]).strftime("%-d %b") for i in tick_idx]
        if has_index:
            tl = fig.add_subplot(gs[row])
            row += 1
            self._timeline_axes(tl, f"{self.index.upper()} · area mean", tick_idx, tick_labels)
            finite = np.isfinite(self.index_mean)
            top = np.nanmax(self.index_mean) * 1.15 if finite.any() else 1
            tl.set_ylim(0, top)
            if self.classes:
                for bound, label in zip(self.classes.bounds, self.classes.labels[1:]):
                    if bound < top:
                        tl.axhline(bound, color=styles.GRID, lw=0.8, zorder=0)
                        tl.text(len(self.days) - 0.5, bound, f" {label}", va="bottom", ha="right", fontsize=7,
                                color=styles.INK_MUTED)
            tl.plot(x, self.index_mean, color=styles.SERIES, lw=2, solid_capstyle="round", zorder=2)
            self.index_dot = tl.scatter([], [], s=(8 * 72 / DPI) ** 2 * 1.6, c=styles.SERIES,
                                        edgecolors=styles.SURFACE, linewidths=2, zorder=3)
            self.cursors.append(tl.axvline(0, color=styles.INK_MUTED, lw=0.8, zorder=1))
        if has_fires:
            bl = fig.add_subplot(gs[row])
            self._timeline_axes(bl, "Fire detections per day", tick_idx, tick_labels)
            self.bars = bl.bar(x, self.counts, width=0.8, color=styles.SERIES_MUTED, zorder=2)
            for bar in self.bars:
                bar.set_capstyle("round")
            bl.set_ylim(0, max(1, self.counts.max()) * 1.15)
            self.cursors.append(bl.axvline(0, color=styles.INK_MUTED, lw=0.8, zorder=1))

    def _timeline_axes(self, ax, title: str, tick_idx, tick_labels) -> None:
        ax.set_xlim(-0.6, len(self.days) - 0.4)
        ax.set_title(title, loc="left", fontsize=9, color=styles.INK_SECONDARY, pad=3)
        ax.set_xticks(tick_idx, tick_labels)
        ax.tick_params(length=0, labelsize=8)
        ax.grid(axis="y", color=styles.GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(styles.BASELINE)
        ax.set_facecolor(styles.SURFACE)

    def _legend(self, head) -> None:
        from matplotlib.patches import Rectangle

        items = []
        if self.index is not None and self.classes:
            items += [(c, lbl, "box") for c, lbl in zip(styles.HEAT, self.classes.labels)]
        x = 1.0
        y = 0.12
        # Right-aligned legend row: index classes, then fire marks.
        entries = items + [(styles.INK, "Fires today", "dot"), (styles.INK_MUTED, "Earlier fires", "box")]
        for color, label, kind in reversed(entries):
            text = head.text(x, y, label, ha="right", va="center", fontsize=8, color=styles.INK_SECONDARY,
                             transform=head.transAxes)
            self.fig.canvas.draw() if not hasattr(self, "_renderer") else None
            self._renderer = self.fig.canvas.get_renderer()
            bb = text.get_window_extent(self._renderer).transformed(head.transAxes.inverted())
            x = bb.x0 - 0.008
            sw = 0.012
            if kind == "dot":
                head.scatter([x - sw / 2], [y], s=30, c=color, edgecolors="white", linewidths=0.8,
                             transform=head.transAxes, clip_on=False)
            else:
                # Same transparency as on the map, so the swatches match what is drawn.
                alpha = INDEX_ALPHA if items and color in styles.HEAT else 0.75
                head.add_patch(Rectangle((x - sw, y - 0.09), sw, 0.18, color=color, alpha=alpha,
                                         transform=head.transAxes, clip_on=False, lw=0))
            x -= sw + 0.022
        if self.index is not None and self.classes:
            head.text(x, y, f"{self.index.upper()} ({self.classes.source}):", ha="right", va="center", fontsize=8,
                      color=styles.INK_SECONDARY, weight="bold", transform=head.transAxes)

    def _natural_earth(self, ax) -> None:
        layers = self.o.basemap()

        def merc(gdf):
            return gdf.to_crs(3857)

        # Land under the index overlay; borders, water and towns above it as reference lines.
        if "countries" in layers:
            countries = merc(layers["countries"])
            countries.plot(ax=ax, color=styles.LAND, lw=0, zorder=0)
            countries.boundary.plot(ax=ax, color=styles.INK_SECONDARY, lw=0.9, zorder=1.5)
        if "lakes" in layers:
            merc(layers["lakes"]).plot(ax=ax, color=styles.WATER, edgecolor="#5598e7", lw=0.6, zorder=1.5)
        if "rivers" in layers:
            merc(layers["rivers"]).plot(ax=ax, color="#5598e7", lw=1.2, zorder=1.5)
        if "places" in layers:
            pl = merc(layers["places"])
            ax.scatter(pl.geometry.x, pl.geometry.y, s=10, c=styles.INK_SECONDARY, zorder=4)
            for geom, name in zip(pl.geometry, pl["name"]):
                ax.annotate(name, (geom.x, geom.y), xytext=(4, 2), textcoords="offset points", fontsize=8,
                            color=styles.INK_SECONDARY, zorder=4, clip_on=True)

    # -- frames -------------------------------------------------------------------

    def frame(self, i: int) -> np.ndarray:
        day = self.days[i]
        parts = [pd.Timestamp(day).strftime("%-d %B %Y")]

        if self.index is not None:
            if day in self.cube_day:
                self.mesh.set_array(self.cube[self.cube_day[day]].ravel())
            else:
                self.mesh.set_array(np.full(self.cube.shape[1:], np.nan).ravel())
            mean = self.index_mean[i]
            if np.isfinite(mean):
                label = f" ({self.classes.label_for(mean)})" if self.classes else ""
                parts.append(f"{self.index.upper()} {mean:.1f}{label}")
                self.index_dot.set_offsets([[i, mean]])
            else:
                parts.append(f"{self.index.upper()}: no data")
                self.index_dot.set_offsets(np.empty((0, 2)))

        if len(self.fx) or hasattr(self, "bars"):
            today = self.fday == day
            self.today.set_offsets(np.column_stack([self.fx[today], self.fy[today]]) if today.any()
                                   else np.empty((0, 2)))
            img = np.ma.masked_where(~self.trail.T, self.trail.T.astype(float))
            self.trail_img.set_data(img)
            parts.append(f"{int(today.sum())} fire detection{'s' if today.sum() != 1 else ''}")
            if hasattr(self, "bars"):
                for j, bar in enumerate(self.bars):
                    bar.set_color(styles.SERIES if j == i else styles.SERIES_MUTED)
            # Today's fires join the grey trail from the next frame on.
            if today.any():
                h, _, _ = np.histogram2d(self.fx[today], self.fy[today], bins=self.trail_bins)
                self.trail |= h > 0

        for c in self.cursors:
            c.set_xdata([i, i])
        self.subtitle.set_text("  ·  ".join(parts))
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()


def animate(path: str | Path, *, out: str | Path | None = None, index: str | None = None, fps: float = 2,
            width_px: int = 1000, basemap: str = "auto", frames_dir: str | Path | None = None,
            hold_last: float = 2.0) -> Path:
    """Write a GIF or MP4 (by ``out`` extension) with one frame per day."""
    import imageio.v2 as imageio

    outputs = Outputs.load(path)
    out = Path(out or outputs.dir / "animation.gif")
    anim = Animator(outputs, index=index, basemap=basemap, width_px=width_px)
    log.info("Rendering %d frames", len(anim.days))

    frames = []
    for i in range(len(anim.days)):
        f = anim.frame(i)
        # Video encoders need even dimensions.
        frames.append(f[: f.shape[0] // 2 * 2, : f.shape[1] // 2 * 2])
        if frames_dir:
            Path(frames_dir).mkdir(parents=True, exist_ok=True)
            imageio.imwrite(Path(frames_dir) / f"frame_{i:04d}_{anim.days[i]}.png", frames[-1])
    frames += [frames[-1]] * int(round(hold_last * fps))

    out.parent.mkdir(parents=True, exist_ok=True)
    suffix = out.suffix.lower()
    if suffix == ".gif":
        from PIL import Image

        imgs = [Image.fromarray(f).quantize(colors=255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
                for f in frames]
        imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0, optimize=True)
    elif suffix in (".mp4", ".webm", ".mkv", ".mov"):
        with imageio.get_writer(out, fps=fps, macro_block_size=1, quality=8, ffmpeg_log_level="error") as writer:
            for f in frames:
                writer.append_data(f)
    else:
        raise ValueError(f"Unsupported animation format {suffix!r}; use .gif or .mp4")
    return out
