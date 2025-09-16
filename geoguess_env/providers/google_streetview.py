"""
Google Street View provider implementation.

This module implements the PanoramaProvider interface for Google Street View
using the streetview and streetlevel libraries.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional

from streetlevel import streetview as streetlevel
from streetview import search_panoramas

from ..load_panorama_helper import load_single_panorama
from .base import PanoramaMetadata, PanoramaProvider


class GoogleStreetViewProvider(PanoramaProvider):
    """
    Google Street View provider implementation.

    Uses the streetview and streetlevel libraries to fetch panorama
    data from Google Street View.
    """

    @property
    def provider_name(self) -> str:
        """Get the name of this provider."""
        return "google_streetview"

    @property
    def attribution_info(self) -> Dict[str, str]:
        return {
            "provider": "Google Street View",
            "license": "Imagery © Google",
            "source": "https://www.google.com/streetview/",
        }

    def find_nearest_panorama(self, lat: float, lon: float) -> Optional[str]:
        """
        Find the nearest panorama ID for given coordinates.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees

        Returns:
            Panorama ID if found, None otherwise

        Raises:
            ValueError: If coordinates are invalid
        """
        if not self.validate_coordinates(lat, lon):
            raise ValueError(f"Invalid coordinates: lat={lat}, lon={lon}")

        attempt = 0
        while attempt < self.max_retries:
            try:
                # Apply rate limiting if specified
                if self.rate_limit_qps:
                    time.sleep(1.0 / self.rate_limit_qps)

                panos = search_panoramas(lat=lat, lon=lon)

                if not panos:
                    return None

                # Filter by minimum capture year if specified
                if self.min_capture_year:
                    panos = self._filter_by_capture_year(panos)

                if not panos:
                    return None

                # Sort by date descending (latest first)
                panos_sorted = sorted(panos, key=self._get_sort_key, reverse=True)

                # Return first valid panorama ID
                for pano in panos_sorted:
                    if pano.pano_id is not None:
                        return pano.pano_id

                return None

            except Exception as e:
                attempt += 1
                if attempt >= self.max_retries:
                    print(
                        f"Failed to find panorama after {self.max_retries} attempts: {e}"
                    )
                    return None

                # Exponential backoff
                wait_time = (2**attempt) * 0.1
                time.sleep(wait_time)

        return None

    def get_panorama_metadata(self, pano_id: str) -> Optional[PanoramaMetadata]:
        """
        Get metadata for a specific panorama.

        Args:
            pano_id: Panorama identifier

        Returns:
            Panorama metadata if found, None otherwise
        """
        attempt = 0
        while attempt < self.max_retries:
            try:
                # Apply rate limiting if specified
                if self.rate_limit_qps:
                    time.sleep(1.0 / self.rate_limit_qps)

                pano = streetlevel.find_panorama_by_id(pano_id)

                if not pano:
                    return None

                # Convert links to simplified format
                links = []
                if hasattr(pano, "links") and pano.links:
                    for link in pano.links:
                        if hasattr(link, "pano") and hasattr(link.pano, "id"):
                            links.append(
                                {
                                    "id": link.pano.id,
                                    "direction": getattr(link, "direction", 0.0),
                                }
                            )

                return PanoramaMetadata(
                    pano_id=pano.id,
                    lat=pano.lat,
                    lon=pano.lon,
                    heading=pano.heading,
                    pitch=getattr(pano, "pitch", None),
                    roll=getattr(pano, "roll", None),
                    date=self._format_date(getattr(pano, "date", None)),
                    elevation=getattr(pano, "elevation", None),
                    links=links if links else None,
                )

            except Exception as e:
                attempt += 1
                if attempt >= self.max_retries:
                    print(
                        f"Failed to get metadata for {pano_id} after {self.max_retries} attempts: {e}"
                    )
                    return None

                # Exponential backoff
                wait_time = (2**attempt) * 0.1
                time.sleep(wait_time)

        return None

    def download_panorama_image(self, pano_id: str, output_path: Path) -> bool:
        """
        Download panorama image to specified path.

        Args:
            pano_id: Panorama identifier
            output_path: Path where image should be saved

        Returns:
            True if download successful, False otherwise
        """
        # Create parent directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        attempt = 0
        while attempt < self.max_retries:
            try:
                # Apply rate limiting if specified
                if self.rate_limit_qps:
                    time.sleep(1.0 / self.rate_limit_qps)

                load_single_panorama(pano_id, str(output_path))

                # Verify file was created and is not empty
                if output_path.exists() and output_path.stat().st_size > 0:
                    return True
                else:
                    print(f"Downloaded file for {pano_id} is empty or missing")
                    return False

            except Exception as e:
                attempt += 1
                if attempt >= self.max_retries:
                    print(
                        f"Failed to download image for {pano_id} after {self.max_retries} attempts: {e}"
                    )
                    return False

                # Exponential backoff
                wait_time = (2**attempt) * 0.1
                time.sleep(wait_time)

        return False

    def get_connected_panoramas(
        self, pano_id: str, max_depth: int = 1
    ) -> List[PanoramaMetadata]:
        """
        Get metadata for panoramas connected to the given panorama.

        Args:
            pano_id: Starting panorama identifier
            max_depth: Maximum search depth (1 = immediate neighbors only)

        Returns:
            List of connected panorama metadata
        """
        connected = []
        visited = set()
        queue = [(pano_id, 0)]

        while queue:
            current_id, depth = queue.pop(0)

            if current_id in visited or depth > max_depth:
                continue

            visited.add(current_id)

            # Get metadata for current panorama
            metadata = self.get_panorama_metadata(current_id)
            if not metadata:
                continue

            if depth > 0:  # Don't include the starting panorama
                connected.append(metadata)

            # Add connected panoramas to queue if we haven't reached max depth
            if depth < max_depth and metadata.links:
                for link in metadata.links:
                    link_id = link.get("id")
                    if link_id and link_id not in visited:
                        queue.append((link_id, depth + 1))

        return connected

    def _filter_by_capture_year(self, panos):
        """Filter panoramas by minimum capture year."""
        if not self.min_capture_year:
            return panos

        filtered = []
        for pano in panos:
            date_str = getattr(pano, "date", None)
            if date_str:
                try:
                    year = int(date_str.split("-")[0])
                    if year >= self.min_capture_year:
                        filtered.append(pano)
                except (ValueError, IndexError):
                    # Include panoramas with unparseable dates
                    filtered.append(pano)
            else:
                # Include panoramas without dates
                filtered.append(pano)

        return filtered

    def _get_sort_key(self, pano):
        """Get sort key for panorama (date-based)."""
        date_str = getattr(pano, "date", None)
        if isinstance(date_str, str):
            try:
                year_str, month_str = date_str.split("-")[:2]
                return (int(year_str), int(month_str))
            except (ValueError, IndexError):
                pass
        # Put undated or malformed entries at the beginning
        return (-1, -1)

    def _format_date(self, date_obj):
        """Format date object to string."""
        if date_obj is None:
            return None

        if isinstance(date_obj, str):
            return date_obj

        if isinstance(date_obj, dict):
            year = date_obj.get("year")
            month = date_obj.get("month")
            day = date_obj.get("day")

            if year and month:
                if day:
                    return f"{year}-{month:02d}-{day:02d}"
                else:
                    return f"{year}-{month:02d}"
            else:
                return str(date_obj)

        return str(date_obj)
