"""Credentials and local paths, read from environment variables."""

from __future__ import annotations

import os
from pathlib import Path


class MissingCredentialsError(RuntimeError):
    pass


def ewds_api_key() -> str | None:
    """Copernicus EWDS personal access token (https://ewds.climate.copernicus.eu/profile).

    Returns None when unset so cdsapi can fall back to ``~/.cdsapirc``.
    """
    return os.environ.get("EWDS_API_KEY")


def firms_map_key() -> str:
    """NASA FIRMS MAP_KEY (https://firms.modaps.eosdis.nasa.gov/api/map_key/)."""
    key = os.environ.get("FIRMS_MAP_KEY")
    if not key:
        raise MissingCredentialsError(
            "FIRMS_MAP_KEY is not set. Request a free key at "
            "https://firms.modaps.eosdis.nasa.gov/api/map_key/ and export FIRMS_MAP_KEY=<key>."
        )
    return key


def cache_dir() -> Path:
    """Where raw downloads are kept so repeated requests are not downloaded again."""
    base = os.environ.get("FIRE_CLI_CACHE") or Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "fire-cli"
    path = Path(base)
    path.mkdir(parents=True, exist_ok=True)
    return path
