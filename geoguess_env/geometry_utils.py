"""
Geometry utilities for the GeoGuessr environment.

This module provides utilities for coordinate transformations,
distance calculations, and link projection for panorama navigation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import List, Optional, Tuple

import numpy as np

from .types import FloatPair, LinkScreen, NavigationLink


class GeometryUtils:
    """
    Utility class for geometric calculations in the GeoGuessr environment.

    Provides methods for coordinate transformations, distance calculations,
    and navigation link projections for equirectangular panorama images.
    """

    # Earth radius in kilometers (WGS84)
    EARTH_RADIUS_KM = 6371.0

    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the Haversine distance between two points on Earth.

        Args:
            lat1: Latitude of first point in degrees
            lon1: Longitude of first point in degrees
            lat2: Latitude of second point in degrees
            lon2: Longitude of second point in degrees

        Returns:
            Distance in kilometers
        """
        # Convert to radians
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)

        # Haversine formula
        a = (
            math.sin(dphi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return GeometryUtils.EARTH_RADIUS_KM * c

    @staticmethod
    def normalize_angle(angle_rad: float) -> float:
        """
        Normalize angle to [0, 2π) range.

        Args:
            angle_rad: Angle in radians

        Returns:
            Normalized angle in [0, 2π) range
        """
        tau = 2 * math.pi
        return angle_rad % tau

    @staticmethod
    def angle_difference(angle1_rad: float, angle2_rad: float) -> float:
        """
        Calculate the smallest signed difference between two angles.

        Args:
            angle1_rad: First angle in radians
            angle2_rad: Second angle in radians

        Returns:
            Signed difference in radians, in range [-π, π]
        """
        diff = (angle1_rad - angle2_rad + math.pi) % (2 * math.pi) - math.pi
        return diff

    @staticmethod
    def direction_to_screen_x(
        direction_rad: float, pano_heading_rad: float, image_width: int
    ) -> int:
        """
        Convert direction to x-coordinate in equirectangular image.

        Args:
            direction_rad: Link direction in radians (absolute compass bearing)
            pano_heading_rad: Panorama heading in radians
            image_width: Width of the image in pixels

        Returns:
            X-coordinate in image space (0 to image_width-1)
        """
        tau = 2 * math.pi

        # Calculate relative direction from panorama heading
        # x=0 corresponds to panorama heading direction
        # x=width/2 corresponds to opposite direction (heading + 180°)
        relative_direction = GeometryUtils.normalize_angle(
            direction_rad - pano_heading_rad
        )

        # Convert to x-coordinate
        x_float = (relative_direction / tau) * image_width
        x = int(round(x_float))

        # Clamp to valid range
        return max(0, min(image_width - 1, x))

    @staticmethod
    def compute_link_screen_positions(
        links: Sequence[NavigationLink],
        pano_heading_rad: float,
        current_heading_rad: float,
        image_width: int,
        image_height: int,
    ) -> List[LinkScreen]:
        """
        Compute screen positions for navigation links.

        Args:
            links: Link metadata entries with 'id' and 'direction' keys
            pano_heading_rad: Panorama heading in radians
            current_heading_rad: Current camera heading in radians
            image_width: Image width in pixels
            image_height: Image height in pixels

        Returns:
            List of link dictionaries with added screen position info
        """
        screen_links: List[LinkScreen] = []
        for link in links:
            link_id = link.get("id") if isinstance(link, dict) else None
            if not isinstance(link_id, str) or not link_id:
                continue

            direction_raw = link.get("direction") if isinstance(link, dict) else None
            try:
                direction_rad = (
                    float(direction_raw) if direction_raw is not None else 0.0
                )
            except (TypeError, ValueError):
                direction_rad = 0.0

            x = GeometryUtils.direction_to_screen_x(
                direction_rad, pano_heading_rad, image_width
            )
            y = image_height // 2

            abs_heading = GeometryUtils.normalize_angle(direction_rad)
            rel_heading_diff = GeometryUtils.angle_difference(
                direction_rad, current_heading_rad
            )

            screen_link: LinkScreen = {
                "id": link_id,
                "heading_deg": math.degrees(abs_heading) % 360.0,
                "screen_xy": [x, y],
                "conf": 1.0,
                "_distance_px": None,
                "_rel_heading_deg": abs(math.degrees(rel_heading_diff)),
            }

            screen_links.append(screen_link)

        return screen_links

    @staticmethod
    def find_clicked_link(
        click_x: int,
        click_y: int,
        screen_links: List[LinkScreen],
        hit_radius: int,
        min_confidence: float = 0.0,
    ) -> Optional[LinkScreen]:
        """
        Find the navigation link closest to a click position.

        Args:
            click_x: Click x-coordinate
            click_y: Click y-coordinate
            screen_links: List of links with screen positions
            hit_radius: Maximum distance in pixels for a valid click
            min_confidence: Minimum confidence threshold for links

        Returns:
            Selected link dictionary, or None if no valid link found
        """
        for link in screen_links:
            screen_xy = link.get("screen_xy")
            if not screen_xy or len(screen_xy) != 2:
                continue
            cx, cy = screen_xy
            dx = click_x - cx
            dy = click_y - cy
            link["_distance_px"] = math.hypot(dx, dy)

        candidates = [
            link
            for link in screen_links
            if (
                (distance := link.get("_distance_px")) is not None
                and distance <= hit_radius
                and link.get("conf", 0.0) >= min_confidence
            )
        ]

        if not candidates:
            return None

        # Sort by distance, then by relative heading, then by ID
        candidates.sort(
            key=lambda link: (
                link["_distance_px"],
                link["_rel_heading_deg"],
                link["id"],
            )
        )

        return candidates[0]

    @staticmethod
    def sample_circular_geofence(
        center_lat: float,
        center_lon: float,
        radius_km: float,
        rng: np.random.Generator,
    ) -> Tuple[float, float]:
        """
        Sample a random point within a circular geofence.

        Args:
            center_lat: Center latitude in degrees
            center_lon: Center longitude in degrees
            radius_km: Radius in kilometers
            rng: NumPy random number generator instance

        Returns:
            Tuple of (latitude, longitude) within the circle
        """
        # Sample uniformly within circle using sqrt for area correction
        r = radius_km * math.sqrt(rng.random())
        theta = 2 * math.pi * rng.random()

        # Convert to lat/lon offset (approximate)
        # 1 degree latitude ≈ 111 km
        # 1 degree longitude ≈ 111 km * cos(latitude)
        lat_offset = (r * math.cos(theta)) / 111.0
        lon_offset = (r * math.sin(theta)) / (
            111.0 * math.cos(math.radians(center_lat))
        )

        sample_lat = center_lat + lat_offset
        sample_lon = center_lon + lon_offset

        # Clamp to valid coordinate ranges
        sample_lat = max(-90.0, min(90.0, sample_lat))
        sample_lon = max(-180.0, min(180.0, sample_lon))

        return sample_lat, sample_lon

    @staticmethod
    def point_in_polygon(
        lat: float, lon: float, polygon: List[Tuple[float, float]]
    ) -> bool:
        """
        Check if a point is inside a polygon using ray casting algorithm.

        Args:
            lat: Point latitude
            lon: Point longitude
            polygon: List of (lat, lon) tuples defining polygon vertices

        Returns:
            True if point is inside polygon, False otherwise
        """
        x, y = lon, lat
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    @staticmethod
    def sample_polygon_geofence(
        polygon: List[Tuple[float, float]],
        rng: np.random.Generator,
        max_attempts: int = 1000,
    ) -> Tuple[float, float]:
        """
        Sample a random point within a polygon geofence.

        Uses rejection sampling within the polygon's bounding box.

        Args:
            polygon: List of (lat, lon) tuples defining polygon vertices
            rng: NumPy random number generator instance
            max_attempts: Maximum sampling attempts before giving up

        Returns:
            Tuple of (latitude, longitude) within the polygon

        Raises:
            ValueError: If no valid point found within max_attempts
        """
        if len(polygon) < 3:
            raise ValueError("Polygon must have at least 3 vertices")

        # Find bounding box
        lats = [p[0] for p in polygon]
        lons = [p[1] for p in polygon]
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)

        # Rejection sampling
        for _ in range(max_attempts):
            sample_lat = rng.uniform(min_lat, max_lat)
            sample_lon = rng.uniform(min_lon, max_lon)

            if GeometryUtils.point_in_polygon(sample_lat, sample_lon, polygon):
                return sample_lat, sample_lon

        raise ValueError(
            f"Could not find valid point in polygon after {max_attempts} attempts"
        )

    @staticmethod
    def compute_answer_reward(
        guess_lat: float,
        guess_lon: float,
        actual_lat: float,
        actual_lon: float,
        scale_factor: float = 400.0,
    ) -> float:
        """
        Compute reward for an answer based on distance from actual location.

        Args:
            guess_lat: Guessed latitude
            guess_lon: Guessed longitude
            actual_lat: Actual latitude
            actual_lon: Actual longitude
            scale_factor: Distance scale factor for reward computation

        Returns:
            Reward score between 0 and 1
        """
        distance_km = GeometryUtils.haversine_distance(
            actual_lat, actual_lon, guess_lat, guess_lon
        )
        return math.exp(-distance_km / scale_factor)

    @staticmethod
    def validate_coordinates(lat: float, lon: float) -> bool:
        """
        Validate latitude and longitude coordinates.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees

        Returns:
            True if coordinates are valid, False otherwise
        """
        return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0

    @staticmethod
    def clamp_coordinates(lat: float, lon: float) -> FloatPair:
        """
        Clamp coordinates to valid ranges.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees

        Returns:
            Tuple of (clamped_lat, clamped_lon)
        """
        clamped_lat = max(-90.0, min(90.0, lat))
        clamped_lon = max(-180.0, min(180.0, lon))
        return clamped_lat, clamped_lon
