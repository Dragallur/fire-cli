"""Spatial and temporal extent of a request."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class BBox:
    """Bounding box in WGS84 degrees."""

    west: float
    south: float
    east: float
    north: float

    def __post_init__(self) -> None:
        if not (-180 <= self.west <= 180 and -180 <= self.east <= 180):
            raise ValueError(f"Longitudes must be within [-180, 180]: {self}")
        if not (-90 <= self.south <= 90 and -90 <= self.north <= 90):
            raise ValueError(f"Latitudes must be within [-90, 90]: {self}")
        if self.south >= self.north:
            raise ValueError(f"South must be less than north: {self}")
        if self.west >= self.east:
            # FIRMS cannot query across the antimeridian; split such areas into two requests.
            raise ValueError(f"West must be less than east (antimeridian crossing is not supported): {self}")

    @classmethod
    def parse(cls, text: str) -> "BBox":
        """Parse ``"west,south,east,north"``."""
        parts = [p.strip() for p in text.split(",")]
        if len(parts) != 4:
            raise ValueError(f"Expected 'west,south,east,north', got {text!r}")
        return cls(*(float(p) for p in parts))

    @classmethod
    def from_file(cls, path: str | Path) -> "BBox":
        """Bounding box of all features in a vector file (GeoJSON, GeoPackage, Shapefile, ...)."""
        import geopandas as gpd

        gdf = gpd.read_file(path)
        if gdf.crs is not None:
            gdf = gdf.to_crs(4326)
        west, south, east, north = (float(v) for v in gdf.total_bounds)
        # A single point (or a vertical/horizontal line) has no area; give it a small one.
        pad = 0.05
        if west == east:
            west, east = max(-180.0, west - pad), min(180.0, east + pad)
        if south == north:
            south, north = max(-90.0, south - pad), min(90.0, north + pad)
        return cls(west, south, east, north)

    def buffered(self, min_size: float) -> "BBox":
        """Grow the box symmetrically so each side spans at least ``min_size`` degrees."""
        dx = max(0.0, min_size - (self.east - self.west)) / 2
        dy = max(0.0, min_size - (self.north - self.south)) / 2
        return BBox(
            max(-180.0, self.west - dx),
            max(-90.0, self.south - dy),
            min(180.0, self.east + dx),
            min(90.0, self.north + dy),
        )

    def as_firms(self) -> str:
        return f"{self.west:g},{self.south:g},{self.east:g},{self.north:g}"

    def as_cds_area(self) -> list[float]:
        return [self.north, self.west, self.south, self.east]

    def slug(self) -> str:
        """Short stable identifier used in cache file names."""
        return hashlib.sha1(self.as_firms().encode()).hexdigest()[:10]


@dataclass(frozen=True)
class DateRange:
    """Inclusive range of calendar days."""

    start: date
    end: date

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError(f"Start {self.start} is after end {self.end}")

    @classmethod
    def parse(cls, start: str, end: str | None = None) -> "DateRange":
        s = date.fromisoformat(start)
        return cls(s, date.fromisoformat(end) if end else s)

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1

    def __iter__(self) -> Iterator[date]:
        for i in range(self.days):
            yield self.start + timedelta(days=i)

    def chunks(self, max_days: int) -> Iterator["DateRange"]:
        """Consecutive sub-ranges of at most ``max_days`` days."""
        cur = self.start
        while cur <= self.end:
            stop = min(cur + timedelta(days=max_days - 1), self.end)
            yield DateRange(cur, stop)
            cur = stop + timedelta(days=1)

    def months(self) -> Iterator[tuple[int, int, list[int]]]:
        """``(year, month, [days])`` for every calendar month touched by the range."""
        groups: dict[tuple[int, int], list[int]] = {}
        for d in self:
            groups.setdefault((d.year, d.month), []).append(d.day)
        for (y, m), days in groups.items():
            yield y, m, days

    def clip(self, start: date | None, end: date | None) -> "DateRange | None":
        """Intersection with ``[start, end]`` (None means unbounded), or None if empty."""
        s = max(self.start, start) if start else self.start
        e = min(self.end, end) if end else self.end
        return DateRange(s, e) if s <= e else None
