"""
Tests for geometry utilities.

Tests coordinate transformations, distance calculations,
and navigation link projections.
"""

import math
import random

import pytest

from geoguess_env.geometry_utils import GeometryUtils
from tests.test_fixtures import (
    assert_angle_close,
)


class TestGeometryUtils:
    """Test cases for GeometryUtils class."""

    def test_haversine_distance_known_points(self, geometry_test_data):
        """Test Haversine distance calculation with known coordinates."""
        distance = GeometryUtils.haversine_distance(
            geometry_test_data.SEATTLE_LAT,
            geometry_test_data.SEATTLE_LON,
            geometry_test_data.NYC_LAT,
            geometry_test_data.NYC_LON,
        )

        # Allow 1% tolerance for the known distance
        expected = geometry_test_data.SEATTLE_NYC_DISTANCE_KM
        tolerance = expected * 0.01
        assert abs(distance - expected) < tolerance

    def test_haversine_distance_same_point(self):
        """Test distance calculation for the same point."""
        lat, lon = 40.7128, -74.0060
        distance = GeometryUtils.haversine_distance(lat, lon, lat, lon)
        assert distance == 0.0

    def test_haversine_distance_antipodal(self):
        """Test distance calculation for antipodal points."""
        # Points on opposite sides of Earth should be ~20015 km apart
        distance = GeometryUtils.haversine_distance(0, 0, 0, 180)
        expected = math.pi * GeometryUtils.EARTH_RADIUS_KM
        assert abs(distance - expected) < 1.0

    def test_normalize_angle(self):
        """Test angle normalization."""
        # Test various angles
        test_cases = [
            (0, 0),
            (math.pi, math.pi),
            (2 * math.pi, 0),
            (3 * math.pi, math.pi),
            (-math.pi / 2, 3 * math.pi / 2),
            (-2 * math.pi, 0),
        ]

        for input_angle, expected in test_cases:
            result = GeometryUtils.normalize_angle(input_angle)
            assert_angle_close(result, expected)

    def test_angle_difference(self):
        """Test angle difference calculation."""
        test_cases = [
            (0, 0, 0),
            (math.pi / 2, 0, math.pi / 2),
            (0, math.pi / 2, -math.pi / 2),
            (math.pi / 4, 3 * math.pi / 4, -math.pi / 2),
            (3 * math.pi / 4, math.pi / 4, math.pi / 2),
            (0.1, 2 * math.pi - 0.1, 0.2),  # Wraparound case
        ]

        for angle1, angle2, expected in test_cases:
            result = GeometryUtils.angle_difference(angle1, angle2)
            assert_angle_close(result, expected)

    def test_direction_to_screen_x(self, geometry_test_data):
        """Test direction to screen x-coordinate conversion."""
        width = geometry_test_data.IMAGE_WIDTH

        # With fixed formula: x = ((direction - heading) % 2π) / 2π * width
        # x=0 shows panorama heading direction
        test_cases = [
            (0, 0, 0),  # North direction, no pano heading -> x=0
            (math.pi / 2, 0, width // 4),  # East direction -> x=256
            (math.pi, 0, width // 2),  # South direction -> x=512
            (3 * math.pi / 2, 0, 3 * width // 4),  # West direction -> x=768
            (math.pi / 2, math.pi / 2, 0),  # East with East heading -> x=0
            (0, math.pi / 2, 3 * width // 4),  # North with East heading -> x=768
        ]

        for direction, pano_heading, expected_x in test_cases:
            result = GeometryUtils.direction_to_screen_x(direction, pano_heading, width)
            assert abs(result - expected_x) <= 1  # Allow 1 pixel tolerance

    def test_direction_to_screen_x_clamping(self):
        """Test that screen x coordinates are properly clamped."""
        width = 100

        # Test edge cases that might produce out-of-bounds coordinates
        x = GeometryUtils.direction_to_screen_x(0, 0, width)
        assert 0 <= x < width

        x = GeometryUtils.direction_to_screen_x(2 * math.pi, 0, width)
        assert 0 <= x < width

    def test_compute_link_screen_positions(self, geometry_test_data):
        """Test computation of link screen positions."""
        links = [
            {"id": "north", "direction": 0.0},
            {"id": "east", "direction": math.pi / 2},
            {"id": "south", "direction": math.pi},
            {"id": "west", "direction": 3 * math.pi / 2},
        ]

        screen_links = GeometryUtils.compute_link_screen_positions(
            links=links,
            pano_heading_rad=0.0,
            current_heading_rad=0.0,
            image_width=geometry_test_data.IMAGE_WIDTH,
            image_height=geometry_test_data.IMAGE_HEIGHT,
        )

        assert len(screen_links) == 4

        # Check that all links have required fields
        for link in screen_links:
            assert "id" in link
            assert "heading_deg" in link
            assert "screen_xy" in link
            assert "conf" in link
            assert "_rel_heading_deg" in link

            # Check screen coordinates are valid
            x, y = link["screen_xy"]
            assert 0 <= x < geometry_test_data.IMAGE_WIDTH
            assert 0 <= y < geometry_test_data.IMAGE_HEIGHT

            # Check heading is in valid range
            assert 0 <= link["heading_deg"] < 360

    def test_find_clicked_link_exact_hit(self):
        """Test finding clicked link with exact center hit."""
        screen_links = [
            {
                "id": "test_link",
                "screen_xy": [100, 200],
                "conf": 1.0,
                "_rel_heading_deg": 0.0,
            }
        ]

        result = GeometryUtils.find_clicked_link(
            click_x=100, click_y=200, screen_links=screen_links, hit_radius=25
        )

        assert result is not None
        assert result["id"] == "test_link"
        assert result["_distance_px"] == 0.0

    def test_find_clicked_link_within_radius(self):
        """Test finding clicked link within hit radius."""
        screen_links = [
            {
                "id": "close_link",
                "screen_xy": [100, 100],
                "conf": 1.0,
                "_rel_heading_deg": 10.0,
            },
            {
                "id": "far_link",
                "screen_xy": [200, 200],
                "conf": 1.0,
                "_rel_heading_deg": 20.0,
            },
        ]

        # Click near the first link
        result = GeometryUtils.find_clicked_link(
            click_x=110, click_y=110, screen_links=screen_links, hit_radius=25
        )

        assert result is not None
        assert result["id"] == "close_link"

    def test_find_clicked_link_outside_radius(self):
        """Test that clicks outside hit radius return None."""
        screen_links = [
            {
                "id": "test_link",
                "screen_xy": [100, 100],
                "conf": 1.0,
                "_rel_heading_deg": 0.0,
            }
        ]

        result = GeometryUtils.find_clicked_link(
            click_x=200, click_y=200, screen_links=screen_links, hit_radius=25
        )

        assert result is None

    def test_find_clicked_link_confidence_filter(self):
        """Test that low confidence links are filtered out."""
        screen_links = [
            {
                "id": "low_conf_link",
                "screen_xy": [100, 100],
                "conf": 0.3,
                "_rel_heading_deg": 0.0,
            }
        ]

        result = GeometryUtils.find_clicked_link(
            click_x=100,
            click_y=100,
            screen_links=screen_links,
            hit_radius=25,
            min_confidence=0.5,
        )

        assert result is None

    def test_find_clicked_link_sorting_priority(self):
        """Test that links are selected in correct priority order."""
        screen_links = [
            {
                "id": "far_better_heading",
                "screen_xy": [120, 120],  # Distance: ~28.28
                "conf": 1.0,
                "_rel_heading_deg": 5.0,  # Better heading
            },
            {
                "id": "close_worse_heading",
                "screen_xy": [110, 110],  # Distance: ~14.14
                "conf": 1.0,
                "_rel_heading_deg": 15.0,  # Worse heading
            },
        ]

        result = GeometryUtils.find_clicked_link(
            click_x=100, click_y=100, screen_links=screen_links, hit_radius=50
        )

        # Closer link should be selected despite worse heading
        assert result["id"] == "close_worse_heading"

    def test_sample_circular_geofence(self):
        """Test circular geofence sampling."""
        center_lat, center_lon = 47.620908, -122.353508
        radius_km = 10.0
        rng = random.Random(42)

        # Sample multiple points
        for _ in range(100):
            lat, lon = GeometryUtils.sample_circular_geofence(
                center_lat, center_lon, radius_km, rng
            )

            # Check coordinates are valid
            assert -90 <= lat <= 90
            assert -180 <= lon <= 180

            # Check point is within circle (with small tolerance for approximation)
            distance = GeometryUtils.haversine_distance(
                center_lat, center_lon, lat, lon
            )
            assert distance <= radius_km + 0.1

    def test_sample_circular_geofence_deterministic(self):
        """Test that geofence sampling is deterministic with same seed."""
        center_lat, center_lon = 40.7128, -74.0060
        radius_km = 5.0

        rng1 = random.Random(123)
        rng2 = random.Random(123)

        lat1, lon1 = GeometryUtils.sample_circular_geofence(
            center_lat, center_lon, radius_km, rng1
        )
        lat2, lon2 = GeometryUtils.sample_circular_geofence(
            center_lat, center_lon, radius_km, rng2
        )

        assert lat1 == lat2
        assert lon1 == lon2

    def test_point_in_polygon_simple(self):
        """Test point in polygon with simple square."""
        # Square polygon
        polygon = [(0, 0), (0, 10), (10, 10), (10, 0)]

        # Points inside
        assert GeometryUtils.point_in_polygon(5, 5, polygon)
        assert GeometryUtils.point_in_polygon(1, 1, polygon)
        assert GeometryUtils.point_in_polygon(9, 9, polygon)

        # Points outside
        assert not GeometryUtils.point_in_polygon(-1, 5, polygon)
        assert not GeometryUtils.point_in_polygon(15, 5, polygon)
        assert not GeometryUtils.point_in_polygon(5, -1, polygon)
        assert not GeometryUtils.point_in_polygon(5, 15, polygon)

        # Points on edges (implementation dependent)
        # assert not GeometryUtils.point_in_polygon(0, 5, polygon)

    def test_sample_polygon_geofence(self):
        """Test polygon geofence sampling."""
        # Triangle polygon
        polygon = [(0, 0), (0, 10), (10, 0)]
        rng = random.Random(42)

        # Sample multiple points
        for _ in range(50):
            lat, lon = GeometryUtils.sample_polygon_geofence(polygon, rng)

            # Check point is within polygon
            assert GeometryUtils.point_in_polygon(lat, lon, polygon)

    def test_sample_polygon_geofence_invalid(self):
        """Test polygon geofence sampling with invalid polygon."""
        # Polygon with too few points
        polygon = [(0, 0), (10, 10)]
        rng = random.Random(42)

        with pytest.raises(ValueError):
            GeometryUtils.sample_polygon_geofence(polygon, rng)

    def test_compute_answer_reward(self, geometry_test_data):
        """Test answer reward computation."""
        actual_lat = geometry_test_data.SEATTLE_LAT
        actual_lon = geometry_test_data.SEATTLE_LON

        # Perfect guess should give reward close to 1
        reward = GeometryUtils.compute_answer_reward(
            actual_lat, actual_lon, actual_lat, actual_lon
        )
        assert reward == pytest.approx(1.0, rel=1e-10)

        # Very far guess should give low reward
        reward = GeometryUtils.compute_answer_reward(
            geometry_test_data.NYC_LAT,
            geometry_test_data.NYC_LON,
            actual_lat,
            actual_lon,
        )
        assert reward < 0.01

        # Medium distance should give medium reward
        reward = GeometryUtils.compute_answer_reward(
            actual_lat + 1.0,  # ~111 km north
            actual_lon,
            actual_lat,
            actual_lon,
        )
        assert 0.1 < reward < 0.9

    def test_validate_coordinates(self):
        """Test coordinate validation."""
        # Valid coordinates
        assert GeometryUtils.validate_coordinates(0, 0)
        assert GeometryUtils.validate_coordinates(90, 180)
        assert GeometryUtils.validate_coordinates(-90, -180)
        assert GeometryUtils.validate_coordinates(45.5, -122.3)

        # Invalid coordinates
        assert not GeometryUtils.validate_coordinates(91, 0)
        assert not GeometryUtils.validate_coordinates(-91, 0)
        assert not GeometryUtils.validate_coordinates(0, 181)
        assert not GeometryUtils.validate_coordinates(0, -181)

    def test_clamp_coordinates(self):
        """Test coordinate clamping."""
        # Valid coordinates should remain unchanged
        lat, lon = GeometryUtils.clamp_coordinates(45.5, -122.3)
        assert lat == 45.5
        assert lon == -122.3

        # Invalid coordinates should be clamped
        lat, lon = GeometryUtils.clamp_coordinates(95, 185)
        assert lat == 90.0
        assert lon == 180.0

        lat, lon = GeometryUtils.clamp_coordinates(-95, -185)
        assert lat == -90.0
        assert lon == -180.0
