import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from gymnasium_env.envs import GeoGuessrWorldEnv


@pytest.fixture
def test_config():
    """Basic config for testing"""
    repo_root = Path(__file__).resolve().parents[1]
    cache_root = str(repo_root / "tempcache")
    return {
        "cache_root": cache_root,
        "input_lat": 47.620908,
        "input_lon": -122.353508,
        "max_steps": 5,
        "arrow_hit_radius_px": 24,
        "arrow_min_conf": 0.0,
    }


@pytest.fixture
def env_with_mock_links(test_config):
    """Environment with mocked link data for consistent testing"""
    env = GeoGuessrWorldEnv(config=test_config)

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
                    "direction": math.pi / 2,
                },  # East - will be at x=256
                {"id": "pano_south", "direction": math.pi},  # South - will be at x=512
                {
                    "id": "pano_west",
                    "direction": 3 * math.pi / 2,
                },  # West - will be at x=768
            ],
        },
        "pano_north": {
            "lat": 47.621908,
            "lon": -122.353508,
            "heading": 0.0,
            "links": [{"id": "test_pano_1", "direction": math.pi}],
        },
        "pano_east": {
            "lat": 47.620908,
            "lon": -122.352508,
            "heading": 0.0,
            "links": [{"id": "test_pano_1", "direction": 3 * math.pi / 2}],
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
            "links": [{"id": "test_pano_1", "direction": math.pi / 2}],
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


def test_answer_action_terminates_episode(test_config):
    env = GeoGuessrWorldEnv(config=test_config)
    try:
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
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


def test_observation_format_and_types(test_config):
    """Test observation format and data types"""
    env = GeoGuessrWorldEnv(config=test_config)
    try:
        obs, info = env.reset()

        # Test observation structure
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.uint8
        assert len(obs.shape) == 3
        assert obs.shape[2] == 3  # RGB channels
        assert np.all(obs >= 0) and np.all(obs <= 255)

        # Test info structure
        required_info_keys = ["pano_id", "gt_lat", "gt_lon", "steps", "pose", "links"]
        for key in required_info_keys:
            assert key in info

        assert isinstance(info["pano_id"], str)
        assert isinstance(info["gt_lat"], (int, float))
        assert isinstance(info["gt_lon"], (int, float))
        assert isinstance(info["steps"], int)
        assert isinstance(info["pose"], dict)
        assert "heading_deg" in info["pose"]
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

    env = GeoGuessrWorldEnv(config=config)

    # Mock the pano graph and current state
    mock_graph = {
        "test_pano": {
            "lat": 47.620908,
            "lon": -122.353508,
            "heading": 0.0,
            "links": [
                {"id": "target_pano", "direction": math.pi / 2}  # East direction
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
    with patch.object(env, "_get_observation", return_value=test_image):
        # Calculate where the east link should appear (direction=π/2, heading=0)
        # x = (π/2) / (2π) * 1024 = 256
        expected_x = 256
        expected_y = 256  # middle of image

        initial_pano = env.current_pano_id

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

    env = GeoGuessrWorldEnv(config=config)

    # Mock the pano graph and current state
    mock_graph = {
        "test_pano": {
            "lat": 47.620908,
            "lon": -122.353508,
            "heading": 0.0,
            "links": [
                {"id": "target_pano", "direction": math.pi / 2}  # East direction
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

    env = GeoGuessrWorldEnv(config=config)

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

    env = GeoGuessrWorldEnv(config=config)

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

    env = GeoGuessrWorldEnv(config=config)

    # Set up known link positions
    mock_graph = {
        "center_pano": {
            "lat": 47.620908,
            "lon": -122.353508,
            "heading": 0.0,
            "links": [
                {"id": "north_pano", "direction": 0.0},  # North: x=0 or 1024, y=256
                {"id": "east_pano", "direction": math.pi / 2},  # East: x=256, y=256
                {"id": "south_pano", "direction": math.pi},  # South: x=512, y=256
                {"id": "west_pano", "direction": 3 * math.pi / 2},  # West: x=768, y=256
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
    with patch.object(env, "_get_observation", return_value=test_image):
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
