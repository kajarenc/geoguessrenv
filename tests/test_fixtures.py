"""
Test fixtures and mocks for the GeoGuessr environment test suite.

This module provides reusable test fixtures, mocks, and utilities
for testing the refactored GeoGuessr environment components.
"""

import json
import math
import tempfile
from pathlib import Path
from typing import Dict, List
from unittest.mock import Mock

import numpy as np
import pytest

from geoguess_env.asset_manager import AssetManager
from geoguess_env.config import GeofenceConfig, GeoGuessrConfig
from geoguess_env.providers.base import PanoramaMetadata, PanoramaProvider


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def basic_config(temp_cache_dir):
    """Basic test configuration."""
    return GeoGuessrConfig(
        mode="offline",
        cache_root=temp_cache_dir,
        input_lat=47.620908,
        input_lon=-122.353508,
        max_steps=10,
    )


@pytest.fixture
def config_with_geofence(temp_cache_dir):
    """Configuration with circular geofence."""
    geofence = GeofenceConfig(
        type="circle", center={"lat": 47.620908, "lon": -122.353508}, radius_km=10.0
    )
    return GeoGuessrConfig(
        mode="online", cache_root=temp_cache_dir, geofence=geofence, max_steps=20
    )


@pytest.fixture
def sample_panorama_metadata():
    """Sample panorama metadata for testing."""
    return PanoramaMetadata(
        pano_id="test_pano_123",
        lat=47.620908,
        lon=-122.353508,
        heading=0.0,
        pitch=None,
        roll=None,
        date="2023-06",
        elevation=10.0,
        links=[
            {"id": "north_pano", "direction": 0.0},
            {"id": "east_pano", "direction": math.pi / 2},
            {"id": "south_pano", "direction": math.pi},
            {"id": "west_pano", "direction": 3 * math.pi / 2},
        ],
    )


@pytest.fixture
def mock_panorama_graph():
    """Mock panorama graph with connected nodes."""
    return {
        "center_pano": {
            "lat": 47.620908,
            "lon": -122.353508,
            "heading": 0.0,
            "links": [
                {"id": "north_pano", "direction": 0.0},
                {"id": "east_pano", "direction": math.pi / 2},
                {"id": "south_pano", "direction": math.pi},
                {"id": "west_pano", "direction": 3 * math.pi / 2},
            ],
        },
        "north_pano": {
            "lat": 47.621908,
            "lon": -122.353508,
            "heading": 180.0,
            "links": [{"id": "center_pano", "direction": math.pi}],
        },
        "east_pano": {
            "lat": 47.620908,
            "lon": -122.352508,
            "heading": 270.0,
            "links": [{"id": "center_pano", "direction": 3 * math.pi / 2}],
        },
        "south_pano": {
            "lat": 47.619908,
            "lon": -122.353508,
            "heading": 0.0,
            "links": [{"id": "center_pano", "direction": 0.0}],
        },
        "west_pano": {
            "lat": 47.620908,
            "lon": -122.354508,
            "heading": 90.0,
            "links": [{"id": "center_pano", "direction": math.pi / 2}],
        },
    }


@pytest.fixture
def test_image():
    """Create a test panorama image."""
    return np.zeros((512, 1024, 3), dtype=np.uint8)


@pytest.fixture
def mock_provider():
    """Create a mock panorama provider."""
    provider = Mock(spec=PanoramaProvider)
    provider.provider_name = "mock_provider"
    provider.max_retries = 3
    provider.rate_limit_qps = None
    provider.min_capture_year = None

    # Set up default return values
    provider.find_nearest_panorama.return_value = "test_pano_123"
    provider.get_panorama_metadata.return_value = PanoramaMetadata(
        pano_id="test_pano_123",
        lat=47.620908,
        lon=-122.353508,
        heading=0.0,
        links=[{"id": "neighbor_pano", "direction": math.pi / 2}],
    )
    provider.download_panorama_image.return_value = True
    provider.get_connected_panoramas.return_value = []
    provider.compute_image_hash.return_value = "abc123hash"
    provider.validate_coordinates.return_value = True

    return provider


@pytest.fixture
def asset_manager(mock_provider, temp_cache_dir):
    """Create an asset manager with mocked provider."""
    return AssetManager(
        provider=mock_provider, cache_root=temp_cache_dir, max_connected_panoramas=5
    )


class MockEnvironment:
    """Mock environment for testing without full initialization."""

    def __init__(self, config: GeoGuessrConfig):
        self.config = config
        self.current_pano_id = "test_pano"
        self.current_lat = config.input_lat
        self.current_lon = config.input_lon
        self.current_links = []
        self._steps = 0
        self._heading_rad = 0.0
        self._current_image = None
        self.observation_space = None
        self.action_space = None

    def get_info(self):
        """Get mock info dictionary."""
        return {
            "provider": "mock",
            "pano_id": self.current_pano_id,
            "gt_lat": self.current_lat,
            "gt_lon": self.current_lon,
            "steps": self._steps,
            "pose": {"yaw_deg": 0.0},
            "links": self.current_links,
        }


@pytest.fixture
def mock_environment(basic_config):
    """Create a mock environment for testing."""
    return MockEnvironment(basic_config)


def create_cached_metadata_file(cache_dir: Path, root_pano_id: str, graph_data: Dict):
    """Create a cached metadata file for testing."""
    metadata_dir = cache_dir / "metadata"
    metadata_dir.mkdir(exist_ok=True)

    mini_file = metadata_dir / f"{root_pano_id}_mini.jsonl"

    with open(mini_file, "w", encoding="utf-8") as f:
        for pano_id, data in graph_data.items():
            # Convert to cached format
            cache_entry = {
                "id": pano_id,
                "lat": data["lat"],
                "lon": data["lon"],
                "heading": data["heading"],
                "links": [],
            }

            for link in data.get("links", []):
                cache_entry["links"].append(
                    {"pano": {"id": link["id"]}, "direction": link["direction"]}
                )

            f.write(json.dumps(cache_entry) + "\n")


def create_cached_image_file(cache_dir: Path, pano_id: str, image: np.ndarray):
    """Create a cached image file for testing."""
    images_dir = cache_dir / "images"
    images_dir.mkdir(exist_ok=True)

    from PIL import Image

    image_path = images_dir / f"{pano_id}.jpg"
    Image.fromarray(image).save(image_path)


@pytest.fixture
def populated_cache(temp_cache_dir, mock_panorama_graph, test_image):
    """Create a cache directory with sample data."""
    # Create metadata file
    create_cached_metadata_file(temp_cache_dir, "center_pano", mock_panorama_graph)

    # Create image files for each panorama
    for pano_id in mock_panorama_graph.keys():
        create_cached_image_file(temp_cache_dir, pano_id, test_image)

    return temp_cache_dir


def create_test_action(operation: str, values: List[float]) -> Dict:
    """Create a test action dictionary."""
    return {
        "op": operation,
        operation: values,
        "value": values,  # For backward compatibility
    }


@pytest.fixture
def sample_click_action():
    """Sample click action."""
    return create_test_action("click", [512, 256])


@pytest.fixture
def sample_answer_action():
    """Sample answer action."""
    return create_test_action("answer", [47.620908, -122.353508])


class GeometryTestData:
    """Test data for geometry calculations."""

    # Seattle coordinates
    SEATTLE_LAT = 47.620908
    SEATTLE_LON = -122.353508

    # New York coordinates
    NYC_LAT = 40.7128
    NYC_LON = -74.0060

    # Known distance between Seattle and NYC (approximately 3876 km)
    SEATTLE_NYC_DISTANCE_KM = 3876.0

    # Test angles in radians
    NORTH_RAD = 0.0
    EAST_RAD = np.pi / 2
    SOUTH_RAD = np.pi
    WEST_RAD = 3 * np.pi / 2

    # Image dimensions for testing
    IMAGE_WIDTH = 1024
    IMAGE_HEIGHT = 512


@pytest.fixture
def geometry_test_data():
    """Geometry test data fixture."""
    return GeometryTestData()


def assert_coordinates_close(
    actual_lat: float,
    actual_lon: float,
    expected_lat: float,
    expected_lon: float,
    tolerance: float = 1e-6,
):
    """Assert that coordinates are close within tolerance."""
    assert abs(actual_lat - expected_lat) < tolerance, (
        f"Latitude difference too large: {actual_lat} vs {expected_lat}"
    )
    assert abs(actual_lon - expected_lon) < tolerance, (
        f"Longitude difference too large: {actual_lon} vs {expected_lon}"
    )


def assert_angle_close(actual_rad: float, expected_rad: float, tolerance: float = 1e-6):
    """Assert that angles are close within tolerance, handling wraparound."""
    from geoguess_env.geometry_utils import GeometryUtils

    # Normalize both angles
    actual_norm = GeometryUtils.normalize_angle(actual_rad)
    expected_norm = GeometryUtils.normalize_angle(expected_rad)

    # Calculate difference considering wraparound
    diff = abs(GeometryUtils.angle_difference(actual_norm, expected_norm))
    assert diff < tolerance, (
        f"Angle difference too large: {actual_rad} vs {expected_rad} (diff: {diff})"
    )


# Utility functions for test setup
def setup_mock_provider_responses(
    provider: Mock,
    pano_metadata: Dict[str, PanoramaMetadata],
    find_responses: Dict[str, str] = None,
):
    """Set up mock provider with specific responses."""
    if find_responses:
        provider.find_nearest_panorama.side_effect = (
            lambda lat, lon: find_responses.get(f"{lat},{lon}")
        )

    provider.get_panorama_metadata.side_effect = lambda pano_id: pano_metadata.get(
        pano_id
    )


def create_test_panorama_metadata(
    pano_id: str, lat: float, lon: float, heading: float = 0.0, links: List[Dict] = None
) -> PanoramaMetadata:
    """Create test panorama metadata with optional links."""
    return PanoramaMetadata(
        pano_id=pano_id, lat=lat, lon=lon, heading=heading, links=links or []
    )
