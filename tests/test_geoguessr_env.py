import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from geoguess_env.asset_manager import PanoramaGraphResult
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
        "max_steps": 5,
        "nav_config": {"arrow_hit_radius_px": 24, "arrow_min_conf": 0.0},
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
        "max_steps": 5,
        "nav_config": {"arrow_hit_radius_px": 24, "arrow_min_conf": 0.0},
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
        "nav_config": {"arrow_hit_radius_px": 24, "arrow_min_conf": 0.0},
    }

    env = GeoGuessrEnv(config=config)

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
        "nav_config": {"arrow_hit_radius_px": 24, "arrow_min_conf": 0.0},
    }

    env = GeoGuessrEnv(config=config)

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


def test_max_steps_truncation_path(test_config_with_fixtures):
    """Ensure repeated clicks hit the max-step truncation guard."""
    config = dict(test_config_with_fixtures)
    config["max_steps"] = 3
    env = GeoGuessrEnv(config=config)

    try:
        _, info = env.reset()
        assert info["steps"] == 0

        # Use the first available link for navigation; fall back to center click.
        if info["links"]:
            click_xy = info["links"][0]["screen_xy"]
        else:
            click_xy = (env._image_width // 2, env._image_height // 2)

        for step_index in range(config["max_steps"]):
            action = env.action_parser.create_click_action(*click_xy)
            _, reward, terminated, truncated, info = env.step(action)

            assert reward == 0.0
            assert info["steps"] == step_index + 1

            if step_index < config["max_steps"] - 1:
                assert not terminated
                assert not truncated
            else:
                assert not terminated
                assert truncated

            if info["links"]:
                click_xy = info["links"][0]["screen_xy"]

    finally:
        env.close()


def test_answer_metadata_propagation(test_config_with_fixtures):
    """Answer steps should populate guess and scoring metadata in info."""
    env = GeoGuessrEnv(config=test_config_with_fixtures)

    try:
        _, info = env.reset()
        actual_lat = float(info["gt_lat"])
        actual_lon = float(info["gt_lon"])
        action = {"op": "answer", "value": [actual_lat, actual_lon]}

        _, reward, terminated, truncated, answer_info = env.step(action)

        assert terminated is True
        assert truncated is False
        assert reward == pytest.approx(1.0, rel=1e-6)

        expected_distance = GeometryUtils.haversine_distance(
            actual_lat, actual_lon, actual_lat, actual_lon
        )

        assert answer_info["guess_lat"] == pytest.approx(actual_lat)
        assert answer_info["guess_lon"] == pytest.approx(actual_lon)
        assert answer_info["distance_km"] == pytest.approx(expected_distance, abs=1e-6)
        assert answer_info["score"] == pytest.approx(reward, rel=1e-6)

    finally:
        env.close()


def test_arrow_click_mapping_with_known_coordinates():
    """Test arrow click mapping with known links at specific coordinates"""
    config = {
        "cache_root": "/tmp",
        "input_lat": 47.620908,
        "input_lon": -122.353508,
        "max_steps": 10,
        "nav_config": {"arrow_hit_radius_px": 24},
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
        "geofence": geofence,
        "max_steps": 5,
    }

    env = GeoGuessrEnv(config=config)

    # Sample with same seed should produce identical results when the episode RNG
    # is reset to the same state.
    seed = 42
    env._episode_rng = np.random.default_rng(seed)
    lat1, lon1 = env._sample_from_geofence()
    env._episode_rng = np.random.default_rng(seed)
    lat2, lon2 = env._sample_from_geofence()

    assert lat1 == lat2
    assert lon1 == lon2

    # Different seeds should produce different results (very high probability)
    env._episode_rng = np.random.default_rng(seed + 1)
    lat3, lon3 = env._sample_from_geofence()
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
        "geofence": geofence,
        "max_steps": 5,
    }

    env = GeoGuessrEnv(config=config)

    # Seed the episode RNG once for reproducible sampling
    env._episode_rng = np.random.default_rng(123)

    # Test multiple samples to ensure they're all within bounds
    for i in range(10):
        lat, lon = env._sample_from_geofence()

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

    graph_result = PanoramaGraphResult(
        root_id="test_pano", graph=mock_graph, missing_assets=set()
    )

    with patch.object(env.asset_manager, "prepare_graph", return_value=graph_result):
        test_image = np.zeros((512, 1024, 3), dtype=np.uint8)
        with patch.object(
            env.asset_manager, "get_image_array", return_value=test_image
        ):
            with patch.object(
                env.asset_manager, "resolve_nearest_panorama", return_value="test_pano"
            ):
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


def test_reset_seed_determinism(tmp_path):
    """Episodes with identical reset seeds should replay the same pano and pose."""
    geofence = {
        "type": "circle",
        "center": {"lat": 47.620908, "lon": -122.353508},
        "radius_km": 5.0,
    }

    config = {
        "cache_root": str(tmp_path),
        "geofence": geofence,
        "max_steps": 3,
    }

    env = GeoGuessrEnv(config=config)

    center_lat = geofence["center"]["lat"]
    center_lon = geofence["center"]["lon"]

    def deterministic_geofence_sample():
        base_seed = env._episode_seed or 0
        offset = (base_seed % 1000) * 1e-4
        return center_lat + offset, center_lon - offset

    # Override the sampler so seeds map to predictable coordinates for the test
    env._sample_from_geofence = deterministic_geofence_sample

    test_image = np.zeros((512, 1024, 3), dtype=np.uint8)
    heading_cache = {}

    def fake_resolve(lat, lon):
        return f"pano_{lat:.6f}_{lon:.6f}"

    def fake_prepare(root_lat, root_lon):
        pano_id = fake_resolve(root_lat, root_lon)
        if pano_id not in heading_cache:
            heading_cache[pano_id] = float(math.radians(len(heading_cache) * 45.0))
        graph = {
            pano_id: {
                "lat": root_lat,
                "lon": root_lon,
                "heading": heading_cache[pano_id],
                "links": [],
            }
        }
        return PanoramaGraphResult(
            root_id=pano_id,
            graph=graph,
            missing_assets=set(),
        )

    try:
        with patch.object(
            env.asset_manager,
            "resolve_nearest_panorama",
            side_effect=fake_resolve,
        ):
            with patch.object(
                env.asset_manager,
                "prepare_graph",
                side_effect=fake_prepare,
            ):
                with patch.object(
                    env.asset_manager,
                    "get_image_array",
                    side_effect=lambda pano_id: test_image,
                ):
                    obs1, info1 = env.reset(seed=123)
                    obs2, info2 = env.reset(seed=123)
                    obs3, info3 = env.reset(seed=321)
    finally:
        env.close()

    assert np.array_equal(obs1["image"], obs2["image"])
    assert info1["pano_id"] == info2["pano_id"]
    assert math.isclose(info1["gt_lat"], info2["gt_lat"])
    assert math.isclose(info1["gt_lon"], info2["gt_lon"])
    assert math.isclose(info1["pose"]["yaw_deg"], info2["pose"]["yaw_deg"])

    assert info3["pano_id"] != info1["pano_id"]
    assert not math.isclose(info3["gt_lat"], info1["gt_lat"])
    assert not math.isclose(info3["gt_lon"], info1["gt_lon"])
    assert not math.isclose(info3["pose"]["yaw_deg"], info1["pose"]["yaw_deg"])


def test_geofence_valid_sampling_reuses_rng():
    """The valid-coordinate sampler should respect the shared episode RNG state."""
    geofence = {
        "type": "circle",
        "center": {"lat": 47.620908, "lon": -122.353508},
        "radius_km": 10.0,
    }

    config = {
        "cache_root": "/tmp",
        "geofence": geofence,
        "max_steps": 5,
    }

    env = GeoGuessrEnv(config=config)

    with patch.object(
        env.asset_manager, "resolve_nearest_panorama", return_value="pano_id"
    ):
        env._episode_rng = np.random.default_rng(4242)
        lat1, lon1 = env._sample_valid_coordinates_from_geofence()

        env._episode_rng = np.random.default_rng(4242)
        lat2, lon2 = env._sample_valid_coordinates_from_geofence()

    assert lat1 == lat2
    assert lon1 == lon2


def test_geofence_sampling_with_fallback():
    """Test that environment falls back to input coordinates when geofence not available"""
    input_lat, input_lon = 40.7128, -74.0060  # NYC coordinates

    config = {
        "cache_root": "/tmp",
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

    graph_result = PanoramaGraphResult(
        root_id="test_pano", graph=mock_graph, missing_assets=set()
    )

    with patch.object(env.asset_manager, "prepare_graph", return_value=graph_result):
        test_image = np.zeros((512, 1024, 3), dtype=np.uint8)
        with patch.object(
            env.asset_manager, "get_image_array", return_value=test_image
        ):
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
        "geofence": geofence,
        "max_steps": 5,
    }

    with pytest.raises(
        ValueError, match="Circular geofence requires center with lat/lon"
    ):
        _env = GeoGuessrEnv(config=config)


def test_render_returns_numpy_array():
    """Test that render(mode='rgb_array') returns a numpy array"""
    config = {
        "cache_root": "/tmp",
        "input_lat": 47.620908,
        "input_lon": -122.353508,
        "max_steps": 5,
    }

    env = GeoGuessrEnv(config=config)

    # Set up a mock image state
    test_image = np.zeros((512, 1024, 3), dtype=np.uint8)
    env._current_image = test_image

    # Test render with rgb_array mode
    rendered = env.render(mode="rgb_array")

    # Verify it returns a numpy array
    assert isinstance(rendered, np.ndarray)
    assert rendered.dtype == np.uint8
    assert len(rendered.shape) == 3  # Should be 3D array (height, width, channels)
    assert rendered.shape[2] == 3  # Should have 3 color channels (RGB)
    assert np.all(rendered >= 0) and np.all(rendered <= 255)  # Values in valid range

    env.close()
