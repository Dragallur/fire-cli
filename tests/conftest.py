import pytest
import requests

from fire_cli.sources import cems


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path_factory):
    """No real HTTP and a private cache directory for every test."""
    monkeypatch.setenv("FIRE_CLI_CACHE", str(tmp_path_factory.mktemp("cache")))

    def blocked(*args, **kwargs):
        raise requests.ConnectionError("network disabled in tests")

    monkeypatch.setattr(requests.Session, "request", blocked)
    monkeypatch.setattr(cems, "_consolidated_months", lambda: frozenset({(2023, 7)}))
