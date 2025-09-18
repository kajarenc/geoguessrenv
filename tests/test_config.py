"""
Tests for configuration management.

Tests configuration validation, creation from dictionaries,
and backward compatibility with legacy parameter names.
"""

from pathlib import Path

import pytest

from geoguess_env.config import (
    GeofenceConfig,
    GeoGuessrConfig,
    NavigationConfig,
    RenderConfig,
)


class TestGeofenceConfig:
    """Test cases for GeofenceConfig class."""

    def test_circular_geofence_valid(self):
        """Test valid circular geofence configuration."""
        geofence = GeofenceConfig(
            type="circle", center={"lat": 40.7128, "lon": -74.0060}, radius_km=10.0
        )

        assert geofence.type == "circle"
        assert geofence.center["lat"] == 40.7128
        assert geofence.center["lon"] == -74.0060
        assert geofence.radius_km == 10.0

    def test_circular_geofence_missing_center(self):
        """Test circular geofence with missing center."""
        with pytest.raises(ValueError, match="center with lat/lon"):
            GeofenceConfig(type="circle", radius_km=10.0)

    def test_circular_geofence_invalid_center(self):
        """Test circular geofence with invalid center."""
        with pytest.raises(ValueError, match="center with lat/lon"):
            GeofenceConfig(
                type="circle",
                center={"lat": 40.7128},  # Missing lon
                radius_km=10.0,
            )

    def test_circular_geofence_missing_radius(self):
        """Test circular geofence with missing radius."""
        with pytest.raises(ValueError, match="positive radius_km"):
            GeofenceConfig(type="circle", center={"lat": 40.7128, "lon": -74.0060})

    def test_circular_geofence_negative_radius(self):
        """Test circular geofence with negative radius."""
        with pytest.raises(ValueError, match="positive radius_km"):
            GeofenceConfig(
                type="circle", center={"lat": 40.7128, "lon": -74.0060}, radius_km=-5.0
            )

    def test_polygon_geofence_valid(self):
        """Test valid polygon geofence configuration."""
        polygon_points = [[40.0, -74.0], [41.0, -74.0], [41.0, -73.0], [40.0, -73.0]]

        geofence = GeofenceConfig(type="polygon", polygon=polygon_points)

        assert geofence.type == "polygon"
        assert geofence.polygon == polygon_points

    def test_polygon_geofence_too_few_points(self):
        """Test polygon geofence with too few points."""
        with pytest.raises(ValueError, match="at least 3 points"):
            GeofenceConfig(type="polygon", polygon=[[40.0, -74.0], [41.0, -74.0]])

    def test_invalid_geofence_type(self):
        """Test invalid geofence type."""
        with pytest.raises(ValueError, match="Unsupported geofence type"):
            GeofenceConfig(type="invalid_type")


class TestGeoGuessrConfig:
    """Test cases for GeoGuessrConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = GeoGuessrConfig()

        assert config.max_steps == 40
        assert config.seed is None
        assert config.input_lat is None
        assert config.input_lon is None
        assert config.geofence is None
        assert config.cache_root == Path("cache")

        # Check nested configs
        assert config.provider_config.provider == "google_streetview"
        assert config.render_config.image_width == 1024
        assert config.render_config.image_height == 512
        assert config.nav_config.arrow_hit_radius_px == 24

    def test_config_with_coordinates(self):
        """Test configuration with valid coordinates."""
        config = GeoGuessrConfig(input_lat=40.7128, input_lon=-74.0060)

        assert config.input_lat == 40.7128
        assert config.input_lon == -74.0060

    def test_config_with_invalid_latitude(self):
        """Test configuration with invalid latitude."""
        with pytest.raises(ValueError, match="Invalid latitude"):
            GeoGuessrConfig(input_lat=91.0)

        with pytest.raises(ValueError, match="Invalid latitude"):
            GeoGuessrConfig(input_lat=-91.0)

    def test_config_with_invalid_longitude(self):
        """Test configuration with invalid longitude."""
        with pytest.raises(ValueError, match="Invalid longitude"):
            GeoGuessrConfig(input_lon=181.0)

        with pytest.raises(ValueError, match="Invalid longitude"):
            GeoGuessrConfig(input_lon=-181.0)

    def test_config_with_invalid_max_steps(self):
        """Test configuration with invalid max_steps."""
        with pytest.raises(ValueError, match="max_steps must be positive"):
            GeoGuessrConfig(max_steps=0)

        with pytest.raises(ValueError, match="max_steps must be positive"):
            GeoGuessrConfig(max_steps=-5)

    def test_config_with_invalid_image_dimensions(self):
        """Test configuration with invalid image dimensions."""
        render_config = RenderConfig(image_width=0, image_height=512)
        with pytest.raises(ValueError, match="Image dimensions must be positive"):
            GeoGuessrConfig(render_config=render_config)

    def test_config_with_invalid_arrow_radius(self):
        """Test configuration with invalid arrow hit radius."""
        nav_config = NavigationConfig(arrow_hit_radius_px=0)
        with pytest.raises(ValueError, match="arrow_hit_radius_px must be positive"):
            GeoGuessrConfig(nav_config=nav_config)

    def test_config_with_invalid_arrow_confidence(self):
        """Test configuration with invalid arrow confidence."""
        nav_config = NavigationConfig(arrow_min_conf=1.5)
        with pytest.raises(ValueError, match="arrow_min_conf must be between 0 and 1"):
            GeoGuessrConfig(nav_config=nav_config)

    def test_from_dict_empty(self):
        """Test creating config from empty dict."""
        config = GeoGuessrConfig.from_dict(None)
        assert config.max_steps == 40

        config = GeoGuessrConfig.from_dict({})
        assert config.max_steps == 40

    def test_from_dict_basic_params(self):
        """Test creating config from dict with basic parameters."""
        config_dict = {
            "mode": "online",
            "max_steps": 50,
            "input_lat": 40.7128,
            "input_lon": -74.0060,
            "cache_root": "/tmp/cache",
        }

        config = GeoGuessrConfig.from_dict(config_dict)

        assert config.max_steps == 50
        assert config.input_lat == 40.7128
        assert config.input_lon == -74.0060
        assert config.cache_root == Path("/tmp/cache")
        assert not hasattr(config, "mode")

    def test_from_dict_with_geofence(self):
        """Test creating config from dict with geofence."""
        config_dict = {
            "mode": "online",
            "geofence": {
                "type": "circle",
                "center": {"lat": 40.7128, "lon": -74.0060},
                "radius_km": 10.0,
            },
        }

        config = GeoGuessrConfig.from_dict(config_dict)

        assert config.geofence is not None
        assert config.geofence.type == "circle"
        assert config.geofence.center["lat"] == 40.7128
        assert config.geofence.radius_km == 10.0

    def test_from_dict_with_nested_configs(self):
        """Test creating config from dict with nested configurations."""
        config_dict = {
            "provider_config": {
                "provider": "mapillary",
                "rate_limit_qps": 2.0,
                "max_fetch_retries": 5,
            },
            "render_config": {
                "image_width": 2048,
                "image_height": 1024,
                "render_fps": 8,
            },
            "nav_config": {"arrow_hit_radius_px": 32, "arrow_min_conf": 0.8},
        }

        config = GeoGuessrConfig.from_dict(config_dict)

        assert config.provider_config.provider == "mapillary"
        assert config.provider_config.rate_limit_qps == 2.0
        assert config.provider_config.max_fetch_retries == 5

        assert config.render_config.image_width == 2048
        assert config.render_config.image_height == 1024
        assert config.render_config.render_fps == 8

        assert config.nav_config.arrow_hit_radius_px == 32
        assert config.nav_config.arrow_min_conf == 0.8

    def test_from_dict_legacy_parameters(self):
        """Test backward compatibility with legacy parameter names."""
        config_dict = {
            "provider": "mapillary",
            "rate_limit_qps": 1.5,
            "max_fetch_retries": 4,
            "min_capture_year": 2020,
            "render_mode": "human",
            "arrow_hit_radius_px": 30,
            "arrow_min_conf": 0.7,
        }

        config = GeoGuessrConfig.from_dict(config_dict)

        # Check that legacy parameters are mapped correctly
        assert config.provider_config.provider == "mapillary"
        assert config.provider_config.rate_limit_qps == 1.5
        assert config.provider_config.max_fetch_retries == 4
        assert config.provider_config.min_capture_year == 2020
        assert config.render_config.render_mode == "human"
        assert config.nav_config.arrow_hit_radius_px == 30
        assert config.nav_config.arrow_min_conf == 0.7

    def test_to_dict(self):
        """Test converting config to dictionary."""
        geofence = GeofenceConfig(
            type="circle", center={"lat": 40.7128, "lon": -74.0060}, radius_km=5.0
        )

        config = GeoGuessrConfig(
            max_steps=30,
            input_lat=40.7128,
            input_lon=-74.0060,
            geofence=geofence,
            cache_root="/custom/cache",
        )

        config_dict = config.to_dict()

        assert config_dict["max_steps"] == 30
        assert config_dict["input_lat"] == 40.7128
        assert config_dict["input_lon"] == -74.0060
        assert config_dict["cache_root"] == "/custom/cache"
        assert "mode" not in config_dict

        assert "geofence" in config_dict
        assert config_dict["geofence"]["type"] == "circle"
        assert config_dict["geofence"]["center"]["lat"] == 40.7128

        assert "provider_config" in config_dict
        assert "render_config" in config_dict
        assert "nav_config" in config_dict

    def test_cache_directory_properties(self):
        """Test cache directory properties."""
        config = GeoGuessrConfig(cache_root="/test/cache")

        assert config.images_dir == Path("/test/cache/images")
        assert config.metadata_dir == Path("/test/cache/metadata")
        assert config.replays_dir == Path("/test/cache/replays")

    def test_config_round_trip(self):
        """Test config to_dict -> from_dict round trip."""
        original_config = GeoGuessrConfig(
            max_steps=25,
            input_lat=47.6062,
            input_lon=-122.3321,
            cache_root="/tmp/test_cache",
        )

        config_dict = original_config.to_dict()
        restored_config = GeoGuessrConfig.from_dict(config_dict)

        assert restored_config.max_steps == original_config.max_steps
        assert restored_config.input_lat == original_config.input_lat
        assert restored_config.input_lon == original_config.input_lon
        assert restored_config.cache_root == original_config.cache_root
