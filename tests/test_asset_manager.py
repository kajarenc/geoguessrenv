import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from geoguess_env.asset_manager import AssetManager
from geoguess_env.providers.base import PanoramaMetadata, PanoramaProvider


class StubProvider(PanoramaProvider):
    """Simple provider that serves predefined metadata and images."""

    def __init__(self, metadata_map, image_color=(123, 200, 50)):
        super().__init__()
        self._metadata_map = metadata_map
        self._image_color = image_color
        self.find_calls = 0
        self.metadata_calls = 0
        self.download_calls = 0

    def find_nearest_panorama(self, lat: float, lon: float) -> str:
        self.find_calls += 1
        return "root"

    def get_panorama_metadata(self, pano_id: str) -> PanoramaMetadata:
        self.metadata_calls += 1
        return self._metadata_map[pano_id]

    def download_panorama_image(self, pano_id: str, output_path: Path) -> bool:
        self.download_calls += 1
        Image.new("RGB", (8, 4), color=self._image_color).save(output_path)
        return True

    def get_connected_panoramas(self, pano_id: str, max_depth: int = 1):
        return []

    @property
    def provider_name(self) -> str:
        return "stub"


class OfflineOnlyProvider(PanoramaProvider):
    """Provider that should never be contacted during offline runs."""

    def find_nearest_panorama(self, lat: float, lon: float) -> str:
        raise AssertionError("offline run should not query provider")

    def get_panorama_metadata(self, pano_id: str) -> PanoramaMetadata:
        raise AssertionError("offline run should not query provider")

    def download_panorama_image(self, pano_id: str, output_path: Path) -> bool:
        raise AssertionError("offline run should not download images")

    def get_connected_panoramas(self, pano_id: str, max_depth: int = 1):
        return []

    @property
    def provider_name(self) -> str:
        return "offline-only"


@pytest.fixture
def root_metadata():
    return PanoramaMetadata(
        pano_id="root",
        lat=10.0,
        lon=20.0,
        heading=0.0,
        links=[
            {"id": "neighbor", "direction": 1.0},
            {"id": "outside", "direction": 2.0},
        ],
        date="2024-01",
    )


def test_prepare_graph_prunes_links_to_missing_nodes(tmp_path, root_metadata):
    provider = StubProvider({"root": root_metadata})
    manager = AssetManager(
        provider=provider, cache_root=tmp_path, max_connected_panoramas=1
    )

    result = manager.prepare_graph(10.0, 20.0, offline_mode=False)

    assert result.missing_assets == set()
    assert set(result.graph.keys()) == {"root"}
    assert result.graph["root"]["links"] == []
    assert provider.find_calls == 1
    assert provider.metadata_calls >= 1
    assert provider.download_calls >= 1


def test_prepare_graph_offline_uses_cached_metadata(tmp_path):
    metadata_dir = tmp_path / "metadata"
    images_dir = tmp_path / "images"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    nearest = metadata_dir / "nearest_pano_cache.json"
    nearest.write_text(json.dumps({"10.0,20.0": "root"}), encoding="utf-8")

    mini_file = metadata_dir / "root_mini.jsonl"
    mini_file.write_text(
        """
{"id": "root", "lat": 10.0, "lon": 20.0, "heading": 0.0, "links": [{"pano": {"id": "neighbor"}, "direction": 1.0}, {"pano": {"id": "ghost"}, "direction": 2.0}]}
{"id": "neighbor", "lat": 11.0, "lon": 21.0, "heading": 0.0, "links": [{"pano": {"id": "root"}, "direction": 3.14}]}
        """.strip(),
        encoding="utf-8",
    )

    for pano_id in ("root", "neighbor"):
        Image.new("RGB", (8, 4), color=(20, 40, 60)).save(images_dir / f"{pano_id}.jpg")

    manager = AssetManager(
        provider=OfflineOnlyProvider(), cache_root=tmp_path, max_connected_panoramas=8
    )

    result = manager.prepare_graph(10.0, 20.0, offline_mode=True)

    assert result.missing_assets == set()
    assert set(result.graph.keys()) == {"root", "neighbor"}
    assert {link["id"] for link in result.graph["root"]["links"]} == {"neighbor"}
    assert {link["id"] for link in result.graph["neighbor"]["links"]} == {"root"}


def test_get_image_array_uses_cache(tmp_path, root_metadata):
    neighbor_metadata = PanoramaMetadata(
        pano_id="neighbor",
        lat=11.0,
        lon=21.0,
        heading=0.0,
        links=[{"id": "root", "direction": 3.14}],
        date="2024-01",
    )

    provider = StubProvider(
        {"root": root_metadata, "neighbor": neighbor_metadata}, image_color=(10, 20, 30)
    )
    manager = AssetManager(
        provider=provider, cache_root=tmp_path, max_connected_panoramas=2
    )

    manager.prepare_graph(0.0, 0.0, offline_mode=False)

    first = manager.get_image_array("root")
    assert first is not None
    assert first.shape in {(4, 8, 3), (8, 4, 3)}

    image_path = manager.images_dir / "root.jpg"
    Image.new("RGB", (8, 4), color=(200, 10, 10)).save(image_path)

    second = manager.get_image_array("root")
    assert np.array_equal(first, second), (
        "Cache should prevent reload when entry is present"
    )

    manager._image_cache.clear()
    third = manager.get_image_array("root")
    assert not np.array_equal(first, third)
