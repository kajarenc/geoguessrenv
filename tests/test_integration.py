"""
Integration tests for the refactored GeoGuessr environment.

Tests the complete environment workflow with the new modular architecture.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from geoguess_env.geoguessr_env import GeoGuessrEnv


class TestGeoGuessrEnvironmentIntegration:
    """Integration tests for the refactored environment."""

    def test_environment_initialization(self):
        """Test that environment can be initialized with new architecture."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "cache_root": temp_dir,
                "input_lat": 47.620908,
                "input_lon": -122.353508,
                "max_steps": 5,
            }

            env = GeoGuessrEnv(config=config)

            # Check that all components are properly initialized
            assert env.config is not None
            assert env.provider is not None
            assert env.asset_manager is not None
            assert env.action_parser is not None

            # Check configuration values
            assert env.config.max_steps == 5
            assert env.config.input_lat == 47.620908
            assert env.config.input_lon == -122.353508

            # Check that spaces are set up correctly
            assert env.observation_space is not None
            assert env.action_space is not None

            env.close()

    def test_environment_with_geofence_config(self):
        """Test environment initialization with geofence configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "cache_root": temp_dir,
                "max_steps": 10,
                "geofence": {
                    "type": "circle",
                    "center": {"lat": 47.620908, "lon": -122.353508},
                    "radius_km": 5.0,
                },
            }

            env = GeoGuessrEnv(config=config)

            # Check geofence configuration
            assert env.config.geofence is not None
            assert env.config.geofence.type == "circle"
            assert env.config.geofence.center["lat"] == 47.620908
            assert env.config.geofence.radius_km == 5.0

            env.close()

    @patch("geoguess_env.providers.google_streetview.streetlevel")
    @patch("geoguess_env.providers.google_streetview.search_panoramas")
    def test_environment_reset_with_mocked_data(self, mock_search, mock_streetlevel):
        """Test environment reset with mocked panorama data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set up mocks
            mock_pano = Mock()
            mock_pano.pano_id = "test_pano_123"
            mock_search.return_value = [mock_pano]

            mock_panorama = Mock()
            mock_panorama.id = "test_pano_123"
            mock_panorama.lat = 47.620908
            mock_panorama.lon = -122.353508
            mock_panorama.heading = 0.0
            mock_panorama.links = []
            mock_streetlevel.find_panorama_by_id.return_value = mock_panorama

            # Create test image
            test_image_path = Path(temp_dir) / "images" / "test_pano_123.jpg"
            test_image_path.parent.mkdir(parents=True, exist_ok=True)

            from PIL import Image

            test_image = Image.fromarray(np.zeros((512, 1024, 3), dtype=np.uint8))
            test_image.save(test_image_path)

            config = {
                "cache_root": temp_dir,
                "input_lat": 47.620908,
                "input_lon": -122.353508,
                "max_steps": 5,
            }

            env = GeoGuessrEnv(config=config)

            # Test reset
            obs, info = env.reset()

            # Verify observation structure
            assert isinstance(obs, dict)
            assert "image" in obs
            assert isinstance(obs["image"], np.ndarray)
            assert obs["image"].shape == (512, 1024, 3)
            assert obs["image"].dtype == np.uint8

            # Verify info structure
            assert isinstance(info, dict)
            required_keys = [
                "provider",
                "pano_id",
                "gt_lat",
                "gt_lon",
                "steps",
                "pose",
                "links",
            ]
            for key in required_keys:
                assert key in info

            assert info["provider"] == "google_streetview"
            assert info["steps"] == 0
            assert "yaw_deg" in info["pose"]

            env.close()

    def test_action_parser_integration(self):
        """Test action parsing integration with environment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "cache_root": temp_dir,
                "input_lat": 47.620908,
                "input_lon": -122.353508,
                "max_steps": 5,
            }

            env = GeoGuessrEnv(config=config)

            # Test action parsing through environment
            click_action = {"op": "click", "value": [100, 200]}
            op, values = env.action_parser.parse_action(click_action)
            assert op == 0
            assert values == (100.0, 200.0)

            answer_action = {"op": "answer", "value": [47.6, -122.3]}
            op, values = env.action_parser.parse_action(answer_action)
            assert op == 1
            assert values == (47.6, -122.3)

            # Test fallback behavior
            invalid_action = {"invalid": "action"}
            op, values = env.action_parser.parse_with_fallback(invalid_action)
            assert op == 0  # Falls back to center click
            assert values == (512.0, 256.0)  # Center of 1024x512 image

            env.close()

    def test_geometry_utils_integration(self):
        """Test geometry utilities integration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "cache_root": temp_dir,
                "input_lat": 47.620908,
                "input_lon": -122.353508,
                "max_steps": 5,
            }

            env = GeoGuessrEnv(config=config)

            # Test coordinate validation
            from geoguess_env.geometry_utils import GeometryUtils

            assert GeometryUtils.validate_coordinates(47.6, -122.3)
            assert not GeometryUtils.validate_coordinates(91.0, 0.0)

            # Test distance calculation
            distance = GeometryUtils.haversine_distance(
                47.620908,
                -122.353508,  # Seattle
                40.7128,
                -74.0060,  # NYC
            )
            assert 3800 < distance < 3900  # Approximately 3876 km

            # Test reward computation
            reward = GeometryUtils.compute_answer_reward(
                47.620908,
                -122.353508,  # Perfect guess
                47.620908,
                -122.353508,  # Actual location
            )
            assert reward == pytest.approx(1.0, rel=1e-10)

            env.close()

    def test_configuration_backward_compatibility(self):
        """Test that legacy configuration parameters still work."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Legacy configuration format
            legacy_config = {
                "cache_root": temp_dir,
                "input_lat": 47.620908,
                "input_lon": -122.353508,
                "max_steps": 5,
                "provider": "google_streetview",
                "arrow_hit_radius_px": 32,
                "arrow_min_conf": 0.8,
                "render_mode": "rgb_array",
            }

            env = GeoGuessrEnv(config=legacy_config)

            # Verify legacy parameters are mapped correctly
            assert env.config.provider_config.provider == "google_streetview"
            assert env.config.nav_config.arrow_hit_radius_px == 32
            assert env.config.nav_config.arrow_min_conf == 0.8
            assert env.config.render_config.render_mode == "rgb_array"

            env.close()

    def test_error_handling(self):
        """Test error handling in various scenarios."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test missing coordinates
            with pytest.raises(ValueError, match="No starting coordinates"):
                config = {
                    "cache_root": temp_dir,
                    "max_steps": 5,
                    # Missing input_lat and input_lon
                }
                env = GeoGuessrEnv(config=config)
                env.reset()

            # Test invalid coordinates
            with pytest.raises(ValueError, match="Invalid latitude"):
                config = {
                    "cache_root": temp_dir,
                    "input_lat": 91.0,  # Invalid latitude
                    "input_lon": -122.353508,
                    "max_steps": 5,
                }
                env = GeoGuessrEnv(config=config)
