"""
Google Street View provider implementation.

This module implements the PanoramaProvider interface for Google Street View
using the streetview and streetlevel libraries.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from tenacity import after_log, retry, stop_after_attempt, wait_exponential

from ..load_panorama_helper import load_single_panorama
from .base import PanoramaMetadata, PanoramaProvider

logger = logging.getLogger(__name__)


class GoogleStreetViewProvider(PanoramaProvider):
    """
    Google Street View provider implementation.

    Uses the streetview and streetlevel libraries to fetch panorama
    data from Google Street View.
    """

    def __init__(
        self,
        max_retries: int = 3,
        min_capture_year: Optional[int] = None,
    ):
        super().__init__(
            max_retries=max_retries,
            min_capture_year=min_capture_year,
        )
        # Keep metadata obtained during nearest pano lookup so we can
        # reuse it when the asset manager asks for details shortly after.
        self._prefetched_metadata: Dict[str, PanoramaMetadata] = {}
        # Lazy-loaded modules (initialized on first access)
        self._streetlevel_module: Any | None = None
        self._search_function: Callable[..., Any] | None = None

    @property
    def provider_name(self) -> str:
        """Get the name of this provider."""
        return "google_streetview"

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

        return self._find_nearest_with_retry(lat, lon)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.1, min=0.2, max=10),
        before_sleep=after_log(logger, logging.WARNING),
        reraise=False,
    )
    def _find_nearest_with_retry(self, lat: float, lon: float) -> Optional[str]:
        """Find nearest panorama with automatic retry on failure."""
        panos = self._search_panoramas(lat=lat, lon=lon)
        if not panos:
            return None

        if self.min_capture_year:
            panos = [p for p in panos if self._meets_year_requirement(p)]

        if not panos:
            return None

        panos_sorted = sorted(panos, key=self._get_capture_date, reverse=True)
        return self._find_valid_panorama(panos_sorted)

    def get_panorama_metadata(self, pano_id: str) -> Optional[PanoramaMetadata]:
        """
        Get metadata for a specific panorama.

        Args:
            pano_id: Panorama identifier

        Returns:
            Panorama metadata if found, None otherwise
        """
        if pano_id in self._prefetched_metadata:
            return self._prefetched_metadata.pop(pano_id)

        return self._fetch_metadata_with_retries(pano_id)

    def download_panorama_image(self, pano_id: str, output_path: Path) -> bool:
        """
        Download panorama image to specified path.

        Args:
            pano_id: Panorama identifier
            output_path: Path where image should be saved

        Returns:
            True if download successful, False otherwise
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return self._download_with_retry(pano_id, output_path)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.1, min=0.2, max=10),
        before_sleep=after_log(logger, logging.WARNING),
        reraise=False,
    )
    def _download_with_retry(self, pano_id: str, output_path: Path) -> bool:
        """Download panorama image with automatic retry on failure."""
        load_single_panorama(pano_id, str(output_path))

        if not (output_path.exists() and output_path.stat().st_size > 0):
            logger.warning("Downloaded file for %s is empty or missing", pano_id)
            return False
        return True

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

    def _find_valid_panorama(self, panos) -> Optional[str]:
        """Find first panorama with valid metadata."""
        for pano in panos:
            if pano_id := pano.pano_id:
                metadata = self._prefetched_metadata.get(
                    pano_id
                ) or self._fetch_metadata_with_retries(pano_id)
                if metadata:
                    self._prefetched_metadata[pano_id] = metadata
                    return pano_id
        return None

    def _meets_year_requirement(self, pano) -> bool:
        """Check if panorama meets minimum capture year requirement."""
        if not (date_str := getattr(pano, "date", None)):
            return True  # Include panoramas without dates
        try:
            year = int(date_str.split("-")[0])
            return year >= self.min_capture_year
        except (ValueError, IndexError):
            return True  # Include panoramas with unparseable dates

    def _get_capture_date(self, pano) -> tuple[int, int]:
        """Extract capture date for sorting."""
        if date_str := getattr(pano, "date", None):
            try:
                parts = date_str.split("-")
                return (int(parts[0]), int(parts[1]))
            except (ValueError, IndexError):
                pass
        return (-1, -1)

    def _extract_links(self, pano) -> List[Dict[str, Any]] | None:
        """Extract navigation links from panorama object."""
        if not (hasattr(pano, "links") and pano.links):
            return None

        links = [
            {"id": link.pano.id, "direction": getattr(link, "direction", 0.0)}
            for link in pano.links
            if hasattr(link, "pano") and hasattr(link.pano, "id")
        ]
        return links if links else None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.1, min=0.2, max=10),
        before_sleep=after_log(logger, logging.WARNING),
        reraise=False,
    )
    def _fetch_metadata_with_retries(self, pano_id: str) -> Optional[PanoramaMetadata]:
        """Fetch metadata with retry/backoff handling."""
        pano = self._streetlevel.find_panorama_by_id(pano_id)
        if not pano:
            logger.warning("Panorama %s not found when fetching metadata", pano_id)
            return None

        return PanoramaMetadata(
            pano_id=pano.id,
            lat=pano.lat,
            lon=pano.lon,
            heading=pano.heading,
            pitch=getattr(pano, "pitch", None),
            roll=getattr(pano, "roll", None),
            date=self._format_date(getattr(pano, "date", None)),
            elevation=getattr(pano, "elevation", None),
            links=self._extract_links(pano),
        )

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

    @property
    def _streetlevel(self) -> Any:
        """Lazily import and return the streetlevel module."""
        if self._streetlevel_module is None:
            self._streetlevel_module = importlib.import_module("streetlevel.streetview")
        return self._streetlevel_module

    @property
    def _search_panoramas(self) -> Callable[..., Any]:
        """Lazily import and return the search_panoramas function."""
        if self._search_function is None:
            streetview_module = importlib.import_module("streetview")
            self._search_function = getattr(streetview_module, "search_panoramas")
        return self._search_function
