"""
Tests for unified continuous action space format.

Tests the new unified action format that is compatible with RL algorithms.
"""

import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from geoguess_env.geoguessr_env import GeoGuessrEnv


@pytest.fixture
def test_config():
    """Basic config for testing."""
    repo_root = Path(__file__).resolve().parents[1]
    cache_root = str(repo_root / "cache")
    return {
        "cache_root": cache_root,
        "input_lat": 47.620908,
        "input_lon": -122.353508,
        "max_steps": 5,
        "nav_config": {"arrow_hit_radius_px": 24, "arrow_min_conf": 0.0},
    }


@pytest.fixture
def env_with_mocks(test_config):
    """Environment with mocked data for testing."""
    env = GeoGuessrEnv(config=test_config)

    # Mock the pano graph
    mock_graph = {
        "test_pano": {
            "lat": 47.620908,
            "lon": -122.353508,
            "heading": 0.0,
            "links": [
                {"id": "target_pano", "direction": math.pi / 2}  # East
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
    env._heading_rad = 0.0

    # Mock image loading
    test_image = np.zeros((512, 1024, 3), dtype=np.uint8)
    with patch.object(env, "_get_observation", return_value={"image": test_image}):
        yield env


def test_action_space_is_box():
    """Test that action space is Box with correct shape."""
    config = {
        "cache_root": "/tmp",
        "input_lat": 47.620908,
        "input_lon": -122.353508,
        "max_steps": 5,
    }
    env = GeoGuessrEnv(config=config)

    # Check action space type and shape
    assert hasattr(env, "action_space")
    assert env.action_space.shape == (3,)
    assert env.action_space.dtype == np.float32

    # Check bounds
    assert np.allclose(env.action_space.low, [0.0, 0.0, 0.0])
    assert np.allclose(env.action_space.high, [1.0, 1.0, 1.0])

    env.close()


def test_action_space_sample_works():
    """Test that sampling from action space produces valid actions."""
    config = {
        "cache_root": "/tmp",
        "input_lat": 47.620908,
        "input_lon": -122.353508,
        "max_steps": 5,
    }
    env = GeoGuessrEnv(config=config)

    # Sample multiple actions
    for _ in range(10):
        action = env.action_space.sample()

        # Check shape and type
        assert action.shape == (3,)
        assert action.dtype == np.float32

        # Check bounds
        assert np.all(action >= 0.0) and np.all(action <= 1.0)

    env.close()


def test_unified_click_action():
    """Test click action in unified format."""
    config = {
        "cache_root": "/tmp",
        "input_lat": 47.620908,
        "input_lon": -122.353508,
        "max_steps": 10,
        "nav_config": {"arrow_hit_radius_px": 24, "arrow_min_conf": 0.0},
    }

    env = GeoGuessrEnv(config=config)

    # Mock setup
    mock_graph = {
        "test_pano": {
            "lat": 47.620908,
            "lon": -122.353508,
            "heading": 0.0,
            "links": [{"id": "target_pano", "direction": math.pi / 2}],
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
    env._heading_rad = 0.0

    # Mock image loading
    test_image = np.zeros((512, 1024, 3), dtype=np.uint8)
    with patch.object(env, "_get_observation", return_value={"image": test_image}):
        # Create a click action at the east link (x=256, y=256)
        # Normalized: x = 256/1024 = 0.25, y = 256/512 = 0.5
        action = np.array([0.25, 0.25, 0.5], dtype=np.float32)  # Click action

        obs, reward, terminated, truncated, info = env.step(action)

        # Should have navigated to target pano
        assert env.current_pano_id == "target_pano"
        assert reward == 0.0
        assert not terminated
        assert not truncated

    env.close()


def test_unified_answer_action():
    """Test answer action in unified format."""
    config = {
        "cache_root": "/tmp",
        "input_lat": 47.620908,
        "input_lon": -122.353508,
        "max_steps": 5,
    }

    env = GeoGuessrEnv(config=config)

    # Mock setup
    mock_graph = {
        "test_pano": {
            "lat": 47.620908,
            "lon": -122.353508,
            "heading": 0.0,
            "links": [],
        }
    }

    env._pano_graph = mock_graph
    env.current_pano_id = "test_pano"
    env.current_lat = 47.620908
    env.current_lon = -122.353508
    env.current_links = []
    env._heading_rad = 0.0

    # Mock image loading
    test_image = np.zeros((512, 1024, 3), dtype=np.uint8)
    with patch.object(env, "_get_observation", return_value={"image": test_image}):
        # Create an answer action with lat=47.6, lon=-122.3
        # Normalize: lat = (47.6 + 90) / 180 = 0.7644, lon = (-122.3 + 180) / 360 = 0.1603
        action = np.array([0.75, 0.7644, 0.1603], dtype=np.float32)  # Answer action

        obs, reward, terminated, truncated, info = env.step(action)

        # Should have terminated
        assert terminated
        assert not truncated
        assert isinstance(reward, float)
        assert 0.0 <= reward <= 1.0

        # Check info contains answer details
        assert "guess_lat" in info
        assert "guess_lon" in info
        assert "distance_km" in info
        assert "score" in info

    env.close()


def test_unified_action_threshold():
    """Test that action_type threshold at 0.5 works correctly."""
    config = {
        "cache_root": "/tmp",
        "input_lat": 47.620908,
        "input_lon": -122.353508,
        "max_steps": 5,
    }

    env = GeoGuessrEnv(config=config)

    # Mock setup
    mock_graph = {
        "test_pano": {
            "lat": 47.620908,
            "lon": -122.353508,
            "heading": 0.0,
            "links": [],
        }
    }

    env._pano_graph = mock_graph
    env.current_pano_id = "test_pano"
    env.current_lat = 47.620908
    env.current_lon = -122.353508
    env.current_links = []
    env._heading_rad = 0.0

    test_image = np.zeros((512, 1024, 3), dtype=np.uint8)
    with patch.object(env, "_get_observation", return_value={"image": test_image}):
        # Test action_type just below 0.5 (should be click)
        action_click = np.array([0.49, 0.5, 0.5], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action_click)
        assert not terminated  # Click doesn't terminate
        assert reward == 0.0  # Clicks give 0 reward

        # Reset step counter
        env._steps = 0

        # Test action_type exactly at 0.5 (should be answer)
        action_answer = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action_answer)
        assert terminated  # Answer terminates

    env.close()


def test_unified_action_clamping():
    """Test that action values are properly clamped to valid ranges."""
    config = {
        "cache_root": "/tmp",
        "input_lat": 47.620908,
        "input_lon": -122.353508,
        "max_steps": 5,
    }

    env = GeoGuessrEnv(config=config)

    # Mock setup
    mock_graph = {
        "test_pano": {
            "lat": 47.620908,
            "lon": -122.353508,
            "heading": 0.0,
            "links": [],
        }
    }

    env._pano_graph = mock_graph
    env.current_pano_id = "test_pano"
    env.current_lat = 47.620908
    env.current_lon = -122.353508
    env.current_links = []
    env._heading_rad = 0.0

    test_image = np.zeros((512, 1024, 3), dtype=np.uint8)
    with patch.object(env, "_get_observation", return_value={"image": test_image}):
        # Test with values that need clamping (simulate floating point errors)
        action = np.array([0.75, 1.1, -0.1], dtype=np.float32)  # Out of bounds

        # Should not crash and should clamp properly
        obs, reward, terminated, truncated, info = env.step(action)

        # Should still work (values clamped internally)
        assert terminated
        assert "guess_lat" in info
        assert "guess_lon" in info

        # Lat and lon should be within valid ranges
        assert -90.0 <= info["guess_lat"] <= 90.0
        assert -180.0 <= info["guess_lon"] <= 180.0

    env.close()


def test_backward_compatibility_with_dict_actions():
    """Test that legacy dict format still works for backward compatibility."""
    config = {
        "cache_root": "/tmp",
        "input_lat": 47.620908,
        "input_lon": -122.353508,
        "max_steps": 5,
    }

    env = GeoGuessrEnv(config=config)

    # Mock setup
    mock_graph = {
        "test_pano": {
            "lat": 47.620908,
            "lon": -122.353508,
            "heading": 0.0,
            "links": [],
        }
    }

    env._pano_graph = mock_graph
    env.current_pano_id = "test_pano"
    env.current_lat = 47.620908
    env.current_lon = -122.353508
    env.current_links = []
    env._heading_rad = 0.0

    test_image = np.zeros((512, 1024, 3), dtype=np.uint8)
    with patch.object(env, "_get_observation", return_value={"image": test_image}):
        # Test legacy dict format (should still work)
        legacy_action = {"op": "answer", "value": [47.6, -122.3]}

        obs, reward, terminated, truncated, info = env.step(legacy_action)

        # Should work just like before
        assert terminated
        assert "guess_lat" in info
        assert "guess_lon" in info

    env.close()


def test_unified_action_with_list_input():
    """Test that actions can be provided as lists (not just numpy arrays)."""
    config = {
        "cache_root": "/tmp",
        "input_lat": 47.620908,
        "input_lon": -122.353508,
        "max_steps": 5,
    }

    env = GeoGuessrEnv(config=config)

    # Mock setup
    mock_graph = {
        "test_pano": {
            "lat": 47.620908,
            "lon": -122.353508,
            "heading": 0.0,
            "links": [],
        }
    }

    env._pano_graph = mock_graph
    env.current_pano_id = "test_pano"
    env.current_lat = 47.620908
    env.current_lon = -122.353508
    env.current_links = []
    env._heading_rad = 0.0

    test_image = np.zeros((512, 1024, 3), dtype=np.uint8)
    with patch.object(env, "_get_observation", return_value={"image": test_image}):
        # Test with list (not numpy array)
        action_as_list = [0.75, 0.5, 0.5]

        obs, reward, terminated, truncated, info = env.step(action_as_list)

        # Should work
        assert terminated

    env.close()


def test_invalid_unified_action_shape():
    """Test handling of invalid action shapes."""
    config = {
        "cache_root": "/tmp",
        "input_lat": 47.620908,
        "input_lon": -122.353508,
        "max_steps": 5,
    }

    env = GeoGuessrEnv(config=config)

    # Mock setup
    mock_graph = {
        "test_pano": {
            "lat": 47.620908,
            "lon": -122.353508,
            "heading": 0.0,
            "links": [],
        }
    }

    env._pano_graph = mock_graph
    env.current_pano_id = "test_pano"
    env.current_lat = 47.620908
    env.current_lon = -122.353508
    env.current_links = []
    env._heading_rad = 0.0

    test_image = np.zeros((512, 1024, 3), dtype=np.uint8)
    with patch.object(env, "_get_observation", return_value={"image": test_image}):
        # Test with wrong shape (should fallback to center click)
        invalid_action = np.array([0.5, 0.5], dtype=np.float32)  # Only 2 elements

        obs, reward, terminated, truncated, info = env.step(invalid_action)

        # Should not crash, falls back to center click
        assert not terminated  # Click doesn't terminate
        assert reward == 0.0

    env.close()
