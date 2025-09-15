from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from geoguess_env.geoguessr_env import GeoGuessrEnv
from geoguess_env.geometry_utils import GeometryUtils


@pytest.fixture
def test_config():
    """Basic config for testing"""
    repo_root = Path(__file__).resolve().parents[1]
    cache_root = str(repo_root / "cache")
    return {
        "cache_root": cache_root,
        "input_lat": 47.620908,
        "input_lon": -122.353508,
        "arrow_hit_radius_px": 24,
        "max_steps": 5,
        "arrow_min_conf": 0.0,
    }


@pytest.fixture
def test_config_with_fixtures():
    """Config for tests that need real cached data from fixtures"""
    repo_root = Path(__file__).resolve().parents[1]
    cache_root = str(repo_root / "tests" / "fixtures")
    return {
        "cache_root": cache_root,
        "input_lat": 47.620908,
        "input_lon": -122.353508,
        "arrow_hit_radius_px": 24,
        "max_steps": 5,
        "arrow_min_conf": 0.0,
    }


@pytest.fixture
def env_with_mock_links(test_config):
    """Environment with mocked link data for consistent testing"""
    env = GeoGuessrEnv(config=test_config)

    # Mock the pano graph with known links at specific positions
    mock_graph = {
        "test_pano_1": {
            "lat": 47.620908,
            "lon": -122.353508,
            "heading": 0.0,
            "links": [
                {
                    "id": "pano_north",
                    "direction": 0.0,
                },  # North - will be at x=0 (or image_width)
                {
                    "id": "pano_east",
                    "direction": 90.0,
                },  # East - will be at x=256
                {"id": "pano_south", "direction": 180.0},  # South - will be at x=512
                {
                    "id": "pano_west",
                    "direction": 270.0,
                },  # West - will be at x=768
            ],
        },
        "pano_north": {
            "lat": 47.621908,
            "lon": -122.353508,
            "heading": 0.0,
            "links": [{"id": "test_pano_1", "direction": 180.0}],
        },
        "pano_east": {
            "lat": 47.620908,
            "lon": -122.352508,
            "heading": 0.0,
            "links": [{"id": "test_pano_1", "direction": 270.0}],
        },
        "pano_south": {
            "lat": 47.619908,
            "lon": -122.353508,
            "heading": 0.0,
            "links": [{"id": "test_pano_1", "direction": 0.0}],
        },
        "pano_west": {
            "lat": 47.620908,
            "lon": -122.354508,
            "heading": 0.0,
            "links": [{"id": "test_pano_1", "direction": 90.0}],
        },
    }

    # Mock the methods that load metadata
    env._pano_graph = mock_graph
    env.pano_root_id = "test_pano_1"

    # Mock the image loading to return a consistent test image
    test_image = np.zeros((512, 1024, 3), dtype=np.uint8)
    env._current_image = test_image
    env._image_height = 512
    env._image_width = 1024

    return env


def test_answer_action_terminates_episode(test_config_with_fixtures):
    env = GeoGuessrEnv(config=test_config_with_fixtures)
    try:
        obs, info = env.reset()
        assert (
            isinstance(obs, dict)
            and "image" in obs
            and isinstance(obs["image"], np.ndarray)
        )
        assert "gt_lat" in info and "gt_lon" in info

        # Issue an answer action; any lat/lon should terminate.
        action = {"op": "answer", "value": [42.0, 42.0]}
        obs2, reward, terminated, truncated, info2 = env.step(action)

        assert terminated is True
        assert truncated is False
        assert isinstance(reward, float)
        assert 0.0 <= reward <= 1.0
    finally:
        env.close()


def test_observation_format_and_types(test_config_with_fixtures):
    """Test observation format and data types"""
    env = GeoGuessrEnv(config=test_config_with_fixtures)
    try:
        obs, info = env.reset()

        # Test observation structure
        assert isinstance(obs, dict) and "image" in obs
        img = obs["image"]
        assert isinstance(img, np.ndarray)
        assert img.dtype == np.uint8
        assert len(img.shape) == 3
        assert img.shape[2] == 3  # RGB channels
        assert np.all(img >= 0) and np.all(img <= 255)

        # Test info structure
        required_info_keys = ["pano_id", "gt_lat", "gt_lon", "steps", "pose", "links"]
        for key in required_info_keys:
            assert key in info

        assert isinstance(info["pano_id"], str)
        assert isinstance(info["gt_lat"], (int, float))
        assert isinstance(info["gt_lon"], (int, float))
        assert isinstance(info["steps"], int)
        assert isinstance(info["pose"], dict)
        assert "yaw_deg" in info["pose"]
        assert isinstance(info["links"], list)

    finally:
        env.close()


def test_click_within_radius_selects_link():
    """Test clicks within arrow_hit_radius_px select the link"""
    config = {
        "cache_root": "/tmp",
        "input_lat": 47.620908,
        "input_lon": -122.353508,
        "max_steps": 10,
        "arrow_hit_radius_px": 24,
        "arrow_min_conf": 0.0,
    }

    env = GeoGuessrEnv(config=config)

    # Mock the pano graph and current state
    mock_graph = {
        "test_pano": {
            "lat": 47.620908,
            "lon": -122.353508,
            "heading": 0.0,
            "links": [
                {"id": "target_pano", "direction": 90.0}  # East direction
            ],
        },
        "target_pano": {
            "lat": 47.620908,
            "lon": -122.352508,
            "heading": 0.0,
            "links": [],
        },
    }

    env._pano_graph = mock_graph
    env.current_pano_id = "test_pano"
    env.current_lat = 47.620908
    env.current_lon = -122.353508
    env.current_links = mock_graph["test_pano"]["links"]
    env._image_height = 512
    env._image_width = 1024
    env._heading_rad = 0.0

    # Mock _get_observation to avoid file loading
    test_image = np.zeros((512, 1024, 3), dtype=np.uint8)
    with patch.object(env, "_get_observation", return_value={"image": test_image}):
        # Calculate where the east link should appear (direction=π/2, heading=0)
        # x = (π/2) / (2π) * 1024 = 256
        expected_x = 256
        expected_y = 256  # middle of image

        # Click within radius (should select a link)
        click_x = expected_x + 10  # Within 24px radius
        click_y = expected_y + 10
        action = {"op": "click", "value": [click_x, click_y]}

        obs, reward, terminated, truncated, info = env.step(action)

        # Should have moved to target pano
        assert env.current_pano_id == "target_pano"
        assert reward == 0.0  # Clicks should yield 0 reward
        assert not terminated
        assert not truncated


def test_click_outside_radius_no_op():
    """Test clicks outside arrow_hit_radius_px result in no-op"""
    config = {
        "cache_root": "/tmp",
        "input_lat": 47.620908,
        "input_lon": -122.353508,
        "max_steps": 10,
        "arrow_hit_radius_px": 24,
        "arrow_min_conf": 0.0,
    }

    env = GeoGuessrEnv(config=config)

    # Mock the pano graph and current state
    mock_graph = {
        "test_pano": {
            "lat": 47.620908,
            "lon": -122.353508,
            "heading": 0.0,
            "links": [
                {"id": "target_pano", "direction": 90.0}  # East direction
            ],
        },
        "target_pano": {
            "lat": 47.620908,
            "lon": -122.352508,
            "heading": 0.0,
            "links": [],
        },
    }

    env._pano_graph = mock_graph
    env.current_pano_id = "test_pano"
    env.current_lat = 47.620908
    env.current_lon = -122.353508
    env.current_links = mock_graph["test_pano"]["links"]
    env._current_image = np.zeros((512, 1024, 3), dtype=np.uint8)
    env._image_height = 512
    env._image_width = 1024
    env._heading_rad = 0.0

    # Calculate where the east link should appear
    expected_x = 256
    expected_y = 256

    initial_pano = env.current_pano_id

    # Click outside radius (should be no-op)
    click_x = expected_x + 50  # Outside 24px radius
    click_y = expected_y + 50
    action = {"op": "click", "value": [click_x, click_y]}

    obs, reward, terminated, truncated, info = env.step(action)

    # Should stay in the same pano
    assert env.current_pano_id == initial_pano
    assert reward == 0.0  # Clicks should yield 0 reward
    assert not terminated
    assert not truncated


def test_reward_semantics():
    """Test reward semantics: clicks = 0.0, answer based on distance"""
    config = {
        "cache_root": "/tmp",
        "input_lat": 47.620908,
        "input_lon": -122.353508,
        "max_steps": 10,
    }

    env = GeoGuessrEnv(config=config)

    # Mock basic state
    env.current_lat = 47.620908
    env.current_lon = -122.353508
    env._current_image = np.zeros((512, 1024, 3), dtype=np.uint8)

    # Test click reward
    action_click = {"op": "click", "value": [100, 100]}
    obs, reward, terminated, truncated, info = env.step(action_click)
    assert reward == 0.0
    assert not terminated

    # Test answer reward - perfect guess should give high reward
    action_answer = {"op": "answer", "value": [47.620907, -122.353507]}
    obs, reward, terminated, truncated, info = env.step(action_answer)
    assert reward > 0.9  # Should be close to 1.0 for a perfect guess
    assert terminated

    # Reset and test distant guess
    env.current_lat = 47.620908
    env.current_lon = -122.353508
    env._steps = 0

    # Guess far away (other side of world)
    action_bad = {"op": "answer", "value": [-47.620908, 57.646492]}
    obs, reward, terminated, truncated, info = env.step(action_bad)
    assert reward < 0.1  # Should be very low for distant guess
    assert terminated


def test_termination_conditions():
    """Test termination: answer terminates, max_steps truncates"""
    config = {
        "cache_root": "/tmp",
        "input_lat": 47.620908,
        "input_lon": -122.353508,
        "max_steps": 3,  # Very small for testing
    }

    env = GeoGuessrEnv(config=config)

    # Mock basic state
    env.current_lat = 47.620908
    env.current_lon = -122.353508
    env._current_image = np.zeros((512, 1024, 3), dtype=np.uint8)
    env._steps = 0

    # Test answer termination
    action = {"op": "answer", "value": [0.0, 0.0]}
    obs, reward, terminated, truncated, info = env.step(action)
    assert terminated
    assert not truncated

    # Reset for truncation test
    env._steps = 0

    # Take max_steps worth of clicks
    for i in range(3):
        action = {"op": "click", "value": [100, 100]}
        obs, reward, terminated, truncated, info = env.step(action)
        if i < 2:
            assert not terminated
            assert not truncated
        else:
            # On the 3rd step (index 2), should truncate
            assert not terminated
            assert truncated
            assert reward == 0.0  # Truncation gives 0 reward


def test_arrow_click_mapping_with_known_coordinates():
    """Test arrow click mapping with known links at specific coordinates"""
    config = {
        "cache_root": "/tmp",
        "input_lat": 47.620908,
        "input_lon": -122.353508,
        "max_steps": 10,
        "arrow_hit_radius_px": 24,
    }

    env = GeoGuessrEnv(config=config)

    # Set up known link positions
    mock_graph = {
        "center_pano": {
            "lat": 47.620908,
            "lon": -122.353508,
            "heading": 0.0,
            "links": [
                {"id": "north_pano", "direction": 0.0},  # North: x=0 or 1024, y=256
                {"id": "east_pano", "direction": 90.0},  # East: x=256, y=256
                {"id": "south_pano", "direction": 180.0},  # South: x=512, y=256
                {"id": "west_pano", "direction": 270.0},  # West: x=768, y=256
            ],
        }
    }

    for pano_id in ["north_pano", "east_pano", "south_pano", "west_pano"]:
        mock_graph[pano_id] = {
            "lat": 47.621,
            "lon": -122.353,
            "heading": 0.0,
            "links": [{"id": "center_pano", "direction": 0.0}],
        }

    env._pano_graph = mock_graph
    env.current_pano_id = "center_pano"
    env.current_lat = 47.620908
    env.current_lon = -122.353508
    env.current_links = mock_graph["center_pano"]["links"]
    env._image_height = 512
    env._image_width = 1024
    env._heading_rad = 0.0

    # Mock _get_observation to avoid file loading
    test_image = np.zeros((512, 1024, 3), dtype=np.uint8)
    with patch.object(env, "_get_observation", return_value={"image": test_image}):
        # Get the computed link screen positions
        links = env._compute_link_screens()
        assert len(links) == 4

        # Test clicking on each computed link position
        for link in links:
            env.current_pano_id = "center_pano"  # Reset position
            env.current_links = mock_graph["center_pano"]["links"]

            cx, cy = link["screen_xy"]
            expected_target = link["id"]

            # Click exactly at center
            action = {"op": "click", "value": [cx, cy]}
            obs, reward, terminated, truncated, info = env.step(action)

            assert env.current_pano_id == expected_target
            assert reward == 0.0
            assert not terminated


def test_geofence_sampling_deterministic():
    """Test that geofence sampling is deterministic with same seed"""
    geofence = {
        "type": "circle",
        "center": {"lat": 47.620908, "lon": -122.353508},
        "radius_km": 10.0,
    }

    config = {
        "cache_root": "/tmp",
        "mode": "online",
        "geofence": geofence,
        "max_steps": 5,
    }

    env = GeoGuessrEnv(config=config)

    # Sample with same seed should produce identical results
    seed = 42
    lat1, lon1 = env._sample_from_geofence(seed)
    lat2, lon2 = env._sample_from_geofence(seed)

    assert lat1 == lat2
    assert lon1 == lon2

    # Different seeds should produce different results (very high probability)
    lat3, lon3 = env._sample_from_geofence(seed + 1)
    assert lat3 != lat1 or lon3 != lon1


def test_geofence_sampling_within_bounds():
    """Test that geofence sampling produces coordinates within the specified circle"""
    center_lat, center_lon = 47.620908, -122.353508
    radius_km = 10.0

    geofence = {
        "type": "circle",
        "center": {"lat": center_lat, "lon": center_lon},
        "radius_km": radius_km,
    }

    config = {
        "cache_root": "/tmp",
        "mode": "online",
        "geofence": geofence,
        "max_steps": 5,
    }

    env = GeoGuessrEnv(config=config)

    # Test multiple samples to ensure they're all within bounds
    for i in range(10):
        lat, lon = env._sample_from_geofence(i)

        # Compute distance from center using Haversine formula
        distance_km = GeometryUtils.haversine_distance(center_lat, center_lon, lat, lon)

        # Should be within the radius (allowing for small numerical errors)
        assert distance_km <= radius_km + 0.001

        # Should be within valid lat/lon ranges
        assert -90.0 <= lat <= 90.0
        assert -180.0 <= lon <= 180.0


def test_geofence_sampling_in_reset():
    """Test that geofence sampling is used during environment reset"""
    geofence = {
        "type": "circle",
        "center": {"lat": 47.620908, "lon": -122.353508},
        "radius_km": 1.0,
    }

    config = {
        "cache_root": "/tmp",
        "mode": "online",
        "geofence": geofence,
        "max_steps": 5,
        "seed": 123,
    }

    env = GeoGuessrEnv(config=config)

    # Mock the asset manager to avoid actual data fetching
    mock_graph = {
        "test_pano": {
            "lat": 47.620908,
            "lon": -122.353508,
            "heading": 0.0,
            "links": [],
        }
    }

    with patch.object(
        env.asset_manager, "get_or_fetch_panorama_graph", return_value=mock_graph
    ):
        # Mock image for observation
        test_image = np.zeros((512, 1024, 3), dtype=np.uint8)
        with patch.object(env, "_get_observation", return_value={"image": test_image}):
            obs, info = env.reset()

            # Verify that sampling was called (coordinates should be different from defaults)
            # Since we're using geofence sampling, the lat/lon should be within the geofence
            gt_lat = info.get("gt_lat")
            gt_lon = info.get("gt_lon")

            if gt_lat is not None and gt_lon is not None:
                center_lat, center_lon = 47.620908, -122.353508
                distance_km = GeometryUtils.haversine_distance(
                    center_lat, center_lon, gt_lat, gt_lon
                )
                assert distance_km <= 1.0  # Within the geofence radius


def test_geofence_sampling_with_fallback():
    """Test that environment falls back to input coordinates when geofence not available"""
    input_lat, input_lon = 40.7128, -74.0060  # NYC coordinates

    config = {
        "cache_root": "/tmp",
        "mode": "online",
        "input_lat": input_lat,
        "input_lon": input_lon,
        "max_steps": 5,
    }

    env = GeoGuessrEnv(config=config)

    # Mock the asset manager to avoid actual data fetching
    mock_graph = {
        "test_pano": {
            "lat": input_lat,
            "lon": input_lon,
            "heading": 0.0,
            "links": [],
        }
    }

    with patch.object(
        env.asset_manager, "get_or_fetch_panorama_graph", return_value=mock_graph
    ):
        # Mock image for observation
        test_image = np.zeros((512, 1024, 3), dtype=np.uint8)
        with patch.object(env, "_get_observation", return_value={"image": test_image}):
            obs, info = env.reset()

            # Should have used the input coordinates
            assert env.current_lat == input_lat
            assert env.current_lon == input_lon


def test_geofence_invalid_type():
    """Test that invalid geofence type raises error"""
    geofence = {
        "type": "polygon",  # Polygon type without required polygon field
        "center": {"lat": 47.620908, "lon": -122.353508},
        "radius_km": 10.0,
    }

    config = {
        "cache_root": "/tmp",
        "mode": "online",
        "geofence": geofence,
        "max_steps": 5,
    }

    with pytest.raises(ValueError, match="Polygon geofence requires at least 3 points"):
        _env = GeoGuessrEnv(config=config)


def test_geofence_invalid_circular_config():
    """Test that invalid circular geofence config raises error"""
    geofence = {
        "type": "circle",
        "center": {"lat": 47.620908},  # Missing lon
        "radius_km": 10.0,
    }

    config = {
        "cache_root": "/tmp",
        "mode": "online",
        "geofence": geofence,
        "max_steps": 5,
    }

    with pytest.raises(
        ValueError, match="Circular geofence requires center with lat/lon"
    ):
        _env = GeoGuessrEnv(config=config)
