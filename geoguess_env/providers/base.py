"""
Base interface for panorama data providers.

This module defines the abstract interface that all panorama providers
must implement to work with the GeoGuessr environment.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from ..types import NavigationLink


@dataclass
class PanoramaMetadata:
    """Metadata for a single panorama."""

    pano_id: str
    lat: float
    lon: float
    heading: float
    pitch: Optional[float] = None
    roll: Optional[float] = None
    date: Optional[str] = None
    elevation: Optional[float] = None
    links: Optional[List[NavigationLink]] = None


@dataclass
class PanoramaAsset:
    """Complete panorama asset including metadata and image."""

    metadata: PanoramaMetadata
    image_path: Path
    image_hash: str


class PanoramaProvider(ABC):
    """
    Abstract base class for panorama data providers.

    Defines the interface for fetching panorama metadata and images
    from various street-level imagery providers (Google Street View,
    Mapillary, KartaView, etc.).
    """

    def __init__(
        self,
        max_retries: int = 3,
        min_capture_year: Optional[int] = None,
    ):
        """
        Initialize the provider.

        Args:
            max_retries: Maximum number of retry attempts for failed requests
            min_capture_year: Minimum capture year for panorama filtering
        """
        self.max_retries = max_retries
        self.min_capture_year = min_capture_year

    @abstractmethod
    def find_nearest_panorama(self, lat: float, lon: float) -> Optional[str]:
        """
        Find the nearest panorama ID for given coordinates.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees

        Returns:
            Panorama ID if found, None otherwise
        """
        pass

    @abstractmethod
    def get_panorama_metadata(self, pano_id: str) -> Optional[PanoramaMetadata]:
        """
        Get metadata for a specific panorama.

        Args:
            pano_id: Panorama identifier

        Returns:
            Panorama metadata if found, None otherwise
        """
        pass

    @abstractmethod
    def download_panorama_image(self, pano_id: str, output_path: Path) -> bool:
        """
        Download panorama image to specified path.

        Args:
            pano_id: Panorama identifier
            output_path: Path where image should be saved

        Returns:
            True if download successful, False otherwise
        """
        pass

    @abstractmethod
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
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get the name of this provider."""
        pass

    def compute_image_hash(self, image_path: Path) -> str:
        """
        Compute SHA256 hash of an image file.

        Args:
            image_path: Path to image file

        Returns:
            Hexadecimal SHA256 hash string
        """
        if not image_path.exists():
            return ""

        sha256_hash = hashlib.sha256()
        with open(image_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def validate_coordinates(self, lat: float, lon: float) -> bool:
        """
        Validate latitude and longitude coordinates.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees

        Returns:
            True if coordinates are valid, False otherwise
        """
        return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0
