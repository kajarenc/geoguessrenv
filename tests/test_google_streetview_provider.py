"""Tests for the Google Street View panorama provider."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from geoguess_env.providers.base import PanoramaMetadata
from geoguess_env.providers.google_streetview import GoogleStreetViewProvider


def test_find_nearest_panorama_requires_valid_coordinates() -> None:
    provider = GoogleStreetViewProvider()

    with pytest.raises(ValueError):
        provider.find_nearest_panorama(lat=200.0, lon=10.0)


def test_get_connected_panoramas_traverses_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata_map = {
        "root": PanoramaMetadata(
            pano_id="root",
            lat=0.0,
            lon=0.0,
            heading=0.0,
            links=[{"id": "a"}, {"id": "b"}],
        ),
        "a": PanoramaMetadata(
            pano_id="a",
            lat=1.0,
            lon=1.0,
            heading=0.0,
            links=[{"id": "c"}],
        ),
        "b": PanoramaMetadata(
            pano_id="b",
            lat=2.0,
            lon=2.0,
            heading=0.0,
            links=[],
        ),
        "c": PanoramaMetadata(
            pano_id="c",
            lat=3.0,
            lon=3.0,
            heading=0.0,
            links=[],
        ),
    }

    def fake_get_metadata(self, pano_id: str) -> PanoramaMetadata | None:
        return metadata_map.get(pano_id)

    monkeypatch.setattr(
        GoogleStreetViewProvider, "get_panorama_metadata", fake_get_metadata
    )

    provider = GoogleStreetViewProvider()
    connected = provider.get_connected_panoramas("root", max_depth=2)

    assert {metadata.pano_id for metadata in connected} == {"a", "b", "c"}


def test_fetch_metadata_formats_date_and_links(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = GoogleStreetViewProvider(max_retries=1)

    class Link:
        def __init__(self, pano_id: str, direction: float) -> None:
            self.pano = SimpleNamespace(id=pano_id)
            self.direction = direction

    pano = SimpleNamespace(
        id="root",
        lat=1.0,
        lon=2.0,
        heading=3.0,
        pitch=None,
        roll=None,
        date={"year": 2024, "month": 5},
        elevation=10.0,
        links=[Link("neighbor", 90.0)],
    )

    # Mock the underlying module directly (the property getter caches to this)
    mock_streetlevel = type(
        "MockStreetLevel", (), {"find_panorama_by_id": lambda self, pano_id: pano}
    )()
    provider._streetlevel_module = mock_streetlevel

    metadata = provider._fetch_metadata_with_retries("root")

    assert metadata is not None
    assert metadata.pano_id == "root"
    assert metadata.date == "2024-05"
    assert metadata.links == [{"id": "neighbor", "direction": 90.0}]
