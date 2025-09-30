"""
Google Street View provider implementation.

This module implements the PanoramaProvider interface for Google Street View
using the streetview and streetlevel libraries.
"""

from __future__ import annotations

import importlib
import logging
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

from ..load_panorama_helper import load_single_panorama
from ..types import NavigationLink
from .base import PanoramaMetadata, PanoramaProvider

logger = logging.getLogger(__name__)


# Street View dependencies are optional in test environments. Provide
# lightweight placeholders that tests can monkeypatch while avoiding
# importing the heavy native modules up-front (which can segfault on CI).
streetlevel: Any = SimpleNamespace(__lazy_stub__=True)
search_panoramas: Callable[..., Any] | None = None
_STREETVIEW_IMPORT_ERROR: Exception | None = None


class GoogleStreetViewProvider(PanoramaProvider):
    """
    Google Street View provider implementation.

    Uses the streetview and streetlevel libraries to fetch panorama
    data from Google Street View.
    """

    def __init__(
        self,
        rate_limit_qps: Optional[float] = None,
        max_retries: int = 3,
        min_capture_year: Optional[int] = None,
    ):
        super().__init__(
            rate_limit_qps=rate_limit_qps,
            max_retries=max_retries,
            min_capture_year=min_capture_year,
        )
        # Keep metadata obtained during nearest pano lookup so we can
        # reuse it when the asset manager asks for details shortly after.
        self._prefetched_metadata: Dict[str, PanoramaMetadata] = {}

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

        attempt = 0
        while attempt < self.max_retries:
            try:
                # Apply rate limiting if specified
                if self.rate_limit_qps:
                    time.sleep(1.0 / self.rate_limit_qps)

                search = self._get_search_function()
                panos = search(lat=lat, lon=lon)

                if not panos:
                    return None

                # Filter by minimum capture year if specified
                if self.min_capture_year:
                    panos = self._filter_by_capture_year(panos)

                if not panos:
                    return None

                # Sort by date descending (latest first)
                panos_sorted = sorted(panos, key=self._get_sort_key, reverse=True)

                # Return first panorama whose metadata can be retrieved
                for pano in panos_sorted:
                    pano_id = pano.pano_id
                    if pano_id is None:
                        continue

                    # Reuse metadata obtained earlier during candidate checks
                    metadata = self._prefetched_metadata.get(pano_id)
                    if metadata is None:
                        metadata = self._fetch_metadata_with_retries(pano_id)

                    if metadata:
                        self._prefetched_metadata[pano_id] = metadata
                        return pano_id

                return None

            except Exception as e:
                attempt += 1
                if attempt >= self.max_retries:
                    logger.warning(
                        "Failed to find panorama after %d attempts: %s",
                        self.max_retries,
                        e,
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
                    logger.warning(
                        "Downloaded file for %s is empty or missing", pano_id
                    )
                    return False

            except Exception as e:
                attempt += 1
                if attempt >= self.max_retries:
                    logger.warning(
                        "Failed to download image for %s after %d attempts: %s",
                        pano_id,
                        self.max_retries,
                        e,
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

    def _fetch_metadata_with_retries(self, pano_id: str) -> Optional[PanoramaMetadata]:
        """Fetch metadata with retry/backoff handling."""
        attempt = 0
        while attempt < self.max_retries:
            try:
                if self.rate_limit_qps:
                    time.sleep(1.0 / self.rate_limit_qps)

                streetlevel_module = self._get_streetlevel_module()
                pano = streetlevel_module.find_panorama_by_id(pano_id)
                if not pano:
                    logger.warning(
                        "Panorama %s not found when fetching metadata", pano_id
                    )
                    return None

                links: List[NavigationLink] = []
                if hasattr(pano, "links") and pano.links:
                    for link in pano.links:
                        if hasattr(link, "pano") and hasattr(link.pano, "id"):
                            links.append(
                                {
                                    "id": link.pano.id,
                                    "direction": getattr(link, "direction", 0.0),
                                }
                            )

                metadata = PanoramaMetadata(
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
                return metadata

            except Exception as e:
                attempt += 1
                if attempt >= self.max_retries:
                    logger.warning(
                        "Failed to fetch metadata for %s after %d attempts: %s",
                        pano_id,
                        self.max_retries,
                        e,
                    )
                    return None

                wait_time = (2**attempt) * 0.1
                time.sleep(wait_time)

        return None

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

    def _get_streetlevel_module(self) -> Any:
        """Return the lazily imported streetlevel module."""

        global streetlevel

        if getattr(streetlevel, "__lazy_stub__", False):
            if hasattr(streetlevel, "find_panorama_by_id"):
                return streetlevel
            streetlevel = self._import_streetview_modules()[0]

        return streetlevel

    def _get_search_function(self) -> Callable[..., Any]:
        """Return the search_panoramas callable, importing if necessary."""

        global search_panoramas

        if search_panoramas is not None:
            return search_panoramas

        return self._import_streetview_modules()[1]

    def _import_streetview_modules(self) -> tuple[Any, Callable[..., Any]]:
        """Attempt to import Street View dependencies and cache them."""

        global streetlevel, search_panoramas, _STREETVIEW_IMPORT_ERROR

        if _STREETVIEW_IMPORT_ERROR is not None:
            raise RuntimeError(
                "Street View dependencies failed to import"
            ) from _STREETVIEW_IMPORT_ERROR

        try:
            streetlevel_module = importlib.import_module("streetlevel.streetview")
            streetview_module = importlib.import_module("streetview")
            search_fn = getattr(streetview_module, "search_panoramas")
        except Exception as exc:  # pragma: no cover - optional dependency
            _STREETVIEW_IMPORT_ERROR = exc
            raise RuntimeError("Street View dependencies are unavailable") from exc

        streetlevel = streetlevel_module
        search_panoramas = search_fn
        return streetlevel_module, search_fn
