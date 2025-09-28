"""
Asset manager for handling panorama data caching and retrieval.

This module provides the AssetManager class that orchestrates panorama
data fetching, caching, and validation for the GeoGuessr environment.
"""

import json
import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
from PIL import Image

from .providers.base import PanoramaAsset, PanoramaMetadata, PanoramaProvider

logger = logging.getLogger(__name__)


class RootPanoramaUnavailableError(Exception):
    """Raised when a root panorama cannot be prepared due to download failures."""


@dataclass
class PanoramaGraphResult:
    """Result of preparing a panorama graph for an episode."""

    root_id: str
    graph: Dict[str, Dict]
    missing_assets: Set[str]


class AssetManager:
    """
    Manages panorama asset caching and retrieval.

    Handles fetching panorama data from providers, caching to local storage,
    and ensuring data integrity through hash validation.
    """

    def __init__(
        self,
        provider: PanoramaProvider,
        cache_root: Path,
        max_connected_panoramas: int = 8,
    ):
        """
        Initialize the asset manager.

        Args:
            provider: Panorama data provider
            cache_root: Root directory for cached assets
            max_connected_panoramas: Maximum number of connected panoramas to fetch
        """
        self.provider = provider
        self.cache_root = Path(cache_root)
        self.max_connected_panoramas = max_connected_panoramas

        # Create cache directories
        self.images_dir = self.cache_root / "images"
        self.metadata_dir = self.cache_root / "metadata"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        # Cache for in-memory metadata
        self._metadata_cache: Dict[str, PanoramaMetadata] = {}
        self._image_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._image_cache_capacity = max(8, max_connected_panoramas * 2)

        # Failed panoramas blocklist
        self._failed_panoramas: Set[str] = set()
        self._load_failed_panoramas()

    def get_panorama_asset(self, pano_id: str) -> Optional[PanoramaAsset]:
        """
        Get a complete panorama asset (metadata + image).

        Args:
            pano_id: Panorama identifier

        Returns:
            PanoramaAsset if available, None otherwise
        """
        # Get metadata
        metadata = self._get_cached_metadata(pano_id)
        if not metadata:
            return None

        # Check for image
        image_path = self.images_dir / f"{pano_id}.jpg"
        if not image_path.exists():
            return None

        # Compute and validate hash
        image_hash = self.provider.compute_image_hash(image_path)
        if not image_hash:
            return None

        return PanoramaAsset(
            metadata=metadata, image_path=image_path, image_hash=image_hash
        )

    def preload_assets(
        self, pano_ids: Set[str], skip_existing: bool = True
    ) -> Dict[str, bool]:
        """
        Preload assets for multiple panoramas.

        Args:
            pano_ids: Set of panorama IDs to preload
            skip_existing: If True, skip panoramas that are already cached

        Returns:
            Dictionary mapping panorama IDs to success status
        """
        results = {}

        for pano_id in pano_ids:
            if skip_existing and self._is_asset_cached(pano_id):
                results[pano_id] = True
                continue

            success = self._fetch_and_cache_asset(pano_id)
            results[pano_id] = success

        return results

    def validate_cache_integrity(self) -> Dict[str, List[str]]:
        """
        Validate integrity of cached assets.

        Returns:
            Dictionary with 'valid', 'invalid', and 'missing' lists of panorama IDs
        """
        result = {"valid": [], "invalid": [], "missing": []}

        # Check all cached metadata files
        for metadata_file in self.metadata_dir.glob("*_mini.jsonl"):
            try:
                with open(metadata_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            pano_id = data.get("id")
                            if not pano_id:
                                continue

                            image_path = self.images_dir / f"{pano_id}.jpg"
                            if image_path.exists():
                                # Validate image hash if available
                                stored_hash = self._get_stored_hash(pano_id)
                                computed_hash = self.provider.compute_image_hash(
                                    image_path
                                )

                                if stored_hash and stored_hash != computed_hash:
                                    result["invalid"].append(pano_id)
                                else:
                                    result["valid"].append(pano_id)
                            else:
                                result["missing"].append(pano_id)
            except Exception as e:
                print(f"Error validating {metadata_file}: {e}")

        return result

    def _load_failed_panoramas(self):
        """Load the list of failed panoramas from disk."""
        blocklist_file = self.metadata_dir / "failed_panoramas.json"
        if blocklist_file.exists():
            try:
                with open(blocklist_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._failed_panoramas = set(data.get("failed_ids", []))
                    logger.info(
                        "Loaded %d failed panorama IDs from blocklist",
                        len(self._failed_panoramas),
                    )
            except Exception as e:
                logger.warning("Failed to load panorama blocklist: %s", e)
                self._failed_panoramas = set()
        else:
            self._failed_panoramas = set()

    def _save_failed_panoramas(self):
        """Save the list of failed panoramas to disk."""
        blocklist_file = self.metadata_dir / "failed_panoramas.json"
        try:
            with open(blocklist_file, "w", encoding="utf-8") as f:
                json.dump({"failed_ids": list(self._failed_panoramas)}, f)
                logger.debug(
                    "Saved %d failed panorama IDs to blocklist",
                    len(self._failed_panoramas),
                )
        except Exception as e:
            logger.warning("Failed to save panorama blocklist: %s", e)

    def _add_to_blocklist(self, pano_id: str):
        """Add a panorama ID to the blocklist."""
        self._failed_panoramas.add(pano_id)
        self._save_failed_panoramas()
        logger.info("Added panorama %s to blocklist", pano_id)

    def is_blocklisted(self, pano_id: str) -> bool:
        """Check if a panorama ID is blocklisted."""
        return pano_id in self._failed_panoramas

    def clear_cache(self, pano_ids: Optional[Set[str]] = None):
        """
        Clear cached assets.

        Args:
            pano_ids: If specified, only clear these panoramas. Otherwise clear all.
        """
        if pano_ids is None:
            # Clear everything
            import shutil

            if self.cache_root.exists():
                shutil.rmtree(self.cache_root)
                self.images_dir.mkdir(parents=True, exist_ok=True)
                self.metadata_dir.mkdir(parents=True, exist_ok=True)
            self._metadata_cache.clear()
            self._image_cache.clear()
        else:
            # Clear specific panoramas
            for pano_id in pano_ids:
                # Remove from memory cache
                self._metadata_cache.pop(pano_id, None)
                self._image_cache.pop(pano_id, None)

                # Remove image file
                image_path = self.images_dir / f"{pano_id}.jpg"
                if image_path.exists():
                    image_path.unlink()

                # Note: We don't remove from metadata files as they may contain
                # multiple panoramas. In a production system, we'd want more
                # sophisticated metadata management.

    def resolve_nearest_panorama(self, lat: float, lon: float) -> Optional[str]:
        """Public wrapper for nearest-panorama lookup."""

        return self._get_or_find_nearest_panorama(lat, lon)

    def prepare_graph(self, root_lat: float, root_lon: float) -> PanoramaGraphResult:
        """Prepare a panorama graph with metadata and images hydrated."""

        root_pano_id = self._get_or_find_nearest_panorama(root_lat, root_lon)

        if not root_pano_id:
            raise ValueError(f"No panorama found for {root_lat}, {root_lon}")

        # Check if root panorama is blocklisted
        if self.is_blocklisted(root_pano_id):
            logger.warning(
                "Root panorama %s is blocklisted, cannot prepare graph", root_pano_id
            )
            raise RootPanoramaUnavailableError(
                f"Root panorama {root_pano_id} at ({root_lat}, {root_lon}) is blocklisted due to previous download failures"
            )

        raw_graph = self._load_panorama_graph(root_pano_id)

        if not raw_graph:
            raw_graph = self._fetch_and_build_graph(root_pano_id)

            if not raw_graph:
                raise ValueError(
                    f"No panorama graph available for coordinates {root_lat}, {root_lon}"
                )

        sanitized_graph: Dict[str, Dict] = {}
        missing_assets: Set[str] = set()

        for pano_id in raw_graph.keys():
            metadata = self._get_or_fetch_metadata(pano_id)
            if not metadata:
                missing_assets.add(pano_id)
                continue

            if not self._is_asset_cached(pano_id):
                success = self._fetch_and_cache_asset(pano_id)
                if not success or not self._is_asset_cached(pano_id):
                    missing_assets.add(pano_id)
                    continue

            sanitized_graph[pano_id] = {
                "lat": self._coerce_scalar(metadata.lat),
                "lon": self._coerce_scalar(metadata.lon),
                "heading": self._coerce_scalar(metadata.heading),
                "date": metadata.date if isinstance(metadata.date, str) else None,
                "links": self._normalize_links(metadata.links),
            }

        if missing_assets:
            logger.warning(
                "Missing assets for panoramas: %s",
                ", ".join(sorted(missing_assets)),
            )

        if root_pano_id not in sanitized_graph:
            # Root panorama failed - add to blocklist and raise special exception
            self._add_to_blocklist(root_pano_id)
            message = f"Root panorama {root_pano_id} unavailable after preparation (download failed)"
            raise RootPanoramaUnavailableError(message)

        if sanitized_graph:
            valid_nodes = set(sanitized_graph.keys())
            for node_id, data in sanitized_graph.items():
                pruned_links = []
                for link in data.get("links", []) or []:
                    target_id = link.get("id")
                    if target_id in valid_nodes:
                        pruned_links.append(link)
                data["links"] = pruned_links

            self._save_graph_to_cache(root_pano_id, sanitized_graph)

        return PanoramaGraphResult(
            root_id=root_pano_id, graph=sanitized_graph, missing_assets=missing_assets
        )

    def get_image_array(self, pano_id: str) -> Optional[np.ndarray]:
        """Return RGB image array for a panorama, caching in memory."""

        cached = self._image_cache.get(pano_id)
        if cached is not None:
            self._image_cache.move_to_end(pano_id)
            return cached

        image_path = self.images_dir / f"{pano_id}.jpg"
        if not image_path.exists():
            return None

        try:
            with Image.open(image_path) as img:
                array = np.array(img.convert("RGB"), dtype=np.uint8)
        except Exception as exc:
            logger.warning("Failed to load image for %s: %s", pano_id, exc)
            return None

        self._image_cache[pano_id] = array
        if len(self._image_cache) > self._image_cache_capacity:
            self._image_cache.popitem(last=False)

        return array

    def _get_or_find_nearest_panorama(self, lat: float, lon: float) -> Optional[str]:
        """Get panorama ID from cache or find nearest."""
        # Round coordinates to 6 decimal places for consistent cache handling
        rounded_lat = round(lat, 6)
        rounded_lon = round(lon, 6)

        # Check cache first
        cache_file = self.metadata_dir / "nearest_pano_cache.json"
        cache_key = f"{rounded_lat},{rounded_lon}"

        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
                    if cache_key in cache_data:
                        cached_pano_id = cache_data[cache_key]
                        if cached_pano_id:
                            # Check if cached ID is blocklisted
                            if self.is_blocklisted(cached_pano_id):
                                logger.info(
                                    "Cached panorama %s for (%.6f, %.6f) is blocklisted, will fetch new",
                                    cached_pano_id,
                                    rounded_lat,
                                    rounded_lon,
                                )
                                # Remove from cache
                                del cache_data[cache_key]
                                with open(cache_file, "w", encoding="utf-8") as f:
                                    json.dump(cache_data, f)
                            else:
                                logger.info(
                                    "Nearest panorama for (%.6f, %.6f) resolved from cache: %s",
                                    rounded_lat,
                                    rounded_lon,
                                    cached_pano_id,
                                )
                                return cached_pano_id
                        else:
                            return cached_pano_id
            except Exception:
                pass

        # Fetch from provider using rounded coordinates for consistency
        pano_id = self.provider.find_nearest_panorama(rounded_lat, rounded_lon)

        # Check if the found panorama is blocklisted
        if pano_id and self.is_blocklisted(pano_id):
            logger.warning(
                "Nearest panorama %s for (%.6f, %.6f) is blocklisted",
                pano_id,
                rounded_lat,
                rounded_lon,
            )
            return None

        if pano_id:
            logger.info(
                "Nearest panorama for (%.6f, %.6f) fetched from provider: %s",
                rounded_lat,
                rounded_lon,
                pano_id,
            )
        else:
            logger.warning(
                "No panorama found near coordinates (%.6f, %.6f)",
                rounded_lat,
                rounded_lon,
            )

        # Update cache
        if pano_id:
            try:
                cache_data = {}
                if cache_file.exists():
                    with open(cache_file, "r", encoding="utf-8") as f:
                        cache_data = json.load(f)

                cache_data[cache_key] = pano_id

                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(cache_data, f)
            except Exception:
                pass

        return pano_id

    def _load_panorama_graph(self, root_pano_id: str) -> Dict[str, Dict]:
        """Load panorama graph from cache or fetch if needed."""
        graph = self._load_cached_graph(root_pano_id)

        if graph:
            return graph

        return self._fetch_and_build_graph(root_pano_id)

    def _load_cached_graph(self, root_pano_id: str) -> Dict[str, Dict]:
        """Load panorama graph from cached metadata."""
        mini_file = self.metadata_dir / f"{root_pano_id}_mini.jsonl"
        if not mini_file.exists():
            return {}

        graph = {}
        try:
            with open(mini_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        pano_id = data.get("id")
                        if pano_id:
                            # Convert to environment format
                            links = []
                            if data.get("links"):
                                for link in data["links"]:
                                    if isinstance(link, dict) and "pano" in link:
                                        pano_data = link["pano"]
                                        if (
                                            isinstance(pano_data, dict)
                                            and "id" in pano_data
                                        ):
                                            links.append(
                                                {
                                                    "id": pano_data["id"],
                                                    "direction": link.get(
                                                        "direction", 0.0
                                                    ),
                                                }
                                            )

                            graph[pano_id] = {
                                "lat": data.get("lat"),
                                "lon": data.get("lon"),
                                "heading": data.get("heading"),
                                "date": data.get("date"),
                                "links": links,
                            }
        except Exception as e:
            print(f"Error loading cached graph: {e}")
            return {}

        return graph

    def _fetch_and_build_graph(self, root_pano_id: str) -> Dict[str, Dict]:
        """Fetch panorama data and build graph."""
        graph = {}
        visited = set()
        queue = [root_pano_id]
        queued: Set[str] = {root_pano_id}
        processed_count = 0

        while queue and processed_count < self.max_connected_panoramas:
            pano_id = queue.pop(0)
            queued.discard(pano_id)
            print(f"FETCHING PANORAMA: {pano_id}")

            if pano_id in visited:
                continue

            visited.add(pano_id)

            # Fetch complete asset (metadata + image)
            asset_success = self._fetch_and_cache_asset(pano_id)
            if not asset_success:
                continue

            # Get metadata (should now be cached)
            metadata = self._get_or_fetch_metadata(pano_id)
            if not metadata:
                continue

            # Add to graph
            links = []
            if metadata.links:
                for link in metadata.links:
                    link_id = link.get("id")
                    if not link_id:
                        continue

                    links.append(link)

                    # Only queue panoramas we haven't processed yet. Track a
                    # separate queued set so we don't enqueue duplicates while
                    # still retaining back-links in the stored graph.
                    if link_id not in visited and link_id not in queued:
                        queue.append(link_id)
                        queued.add(link_id)

            graph[pano_id] = {
                "lat": metadata.lat,
                "lon": metadata.lon,
                "heading": metadata.heading,
                "date": metadata.date,
                "links": links,
            }

            processed_count += 1

        # Save to cache
        self._save_graph_to_cache(root_pano_id, graph)

        return graph

    def _get_or_fetch_metadata(self, pano_id: str) -> Optional[PanoramaMetadata]:
        """Get metadata from cache or fetch from provider."""
        # Check memory cache
        if pano_id in self._metadata_cache:
            return self._metadata_cache[pano_id]

        # Check disk cache
        metadata = self._get_cached_metadata(pano_id)
        if metadata:
            self._metadata_cache[pano_id] = metadata
            return metadata

        metadata = self.provider.get_panorama_metadata(pano_id)
        if metadata:
            self._metadata_cache[pano_id] = metadata
            logger.info("Downloaded metadata for panorama %s", pano_id)
        else:
            logger.warning("Failed to download metadata for panorama %s", pano_id)

        return metadata

    def _get_cached_metadata(self, pano_id: str) -> Optional[PanoramaMetadata]:
        """Get metadata from disk cache."""
        metadata = self._get_cached_metadata_from_legacy(pano_id)
        return metadata

    def _fetch_and_cache_asset(self, pano_id: str) -> bool:
        """Fetch and cache a complete panorama asset."""
        # Skip if blocklisted
        if self.is_blocklisted(pano_id):
            logger.debug("Skipping blocklisted panorama %s", pano_id)
            return False

        # Get or fetch metadata
        metadata = self._get_or_fetch_metadata(pano_id)
        if not metadata:
            return False

        # Download image if not cached
        image_path = self.images_dir / f"{pano_id}.jpg"
        if not image_path.exists():
            success = self.provider.download_panorama_image(pano_id, image_path)
            if success:
                logger.info(
                    "Downloaded panorama image for %s to %s", pano_id, image_path
                )
            else:
                logger.warning(
                    "Failed to download panorama image for %s to %s - adding to blocklist",
                    pano_id,
                    image_path,
                )
                # Add to the blocklist on download failure
                self._add_to_blocklist(pano_id)
                return False
        else:
            logger.debug("Panorama image already cached for %s", pano_id)

        self._image_cache.pop(pano_id, None)

        # Compute and store hash
        image_hash = self.provider.compute_image_hash(image_path)
        self._store_hash(pano_id, image_hash)

        return True

    def _is_asset_cached(self, pano_id: str) -> bool:
        """Check if asset is fully cached."""
        metadata = self._metadata_cache.get(pano_id)
        if metadata is None:
            metadata = self._get_cached_metadata(pano_id)
        if not metadata:
            return False

        image_path = self.images_dir / f"{pano_id}.jpg"
        return image_path.exists()

    def _save_graph_to_cache(self, root_pano_id: str, graph: Dict[str, Dict]):
        """Save graph to cache files."""
        # Save raw metadata (for compatibility)
        mini_file = self.metadata_dir / f"{root_pano_id}_mini.jsonl"

        try:
            with open(mini_file, "w", encoding="utf-8") as f:
                for pano_id, data in graph.items():
                    # Convert back to detailed format for caching
                    metadata = self._metadata_cache.get(pano_id)
                    if metadata:
                        cache_data = {
                            "id": metadata.pano_id,
                            "lat": self._coerce_scalar(metadata.lat),
                            "lon": self._coerce_scalar(metadata.lon),
                            "heading": self._coerce_scalar(metadata.heading),
                            "pitch": self._coerce_scalar(metadata.pitch),
                            "roll": self._coerce_scalar(metadata.roll),
                            "date": metadata.date
                            if isinstance(metadata.date, str)
                            else None,
                            "elevation": self._coerce_scalar(metadata.elevation),
                            "links": [],
                        }

                        if metadata.links:
                            for link in metadata.links:
                                link_id = link.get("id")
                                if link_id is None and isinstance(
                                    link.get("pano"), dict
                                ):
                                    link_id = link["pano"].get("id")

                                if not link_id:
                                    continue

                                cache_data["links"].append(
                                    {
                                        "pano": {"id": link_id},
                                        "direction": link.get("direction"),
                                    }
                                )

                        f.write(json.dumps(cache_data) + "\n")
        except Exception as e:
            print(f"Error saving graph to cache: {e}")

    def _get_stored_hash(self, pano_id: str) -> Optional[str]:
        """Get stored hash for panorama image."""
        # In a production system, we'd store hashes in a dedicated file
        # For now, return None to skip hash validation
        return None

    def _store_hash(self, pano_id: str, image_hash: str):
        """Store hash for panorama image."""
        # In a production system, we'd store hashes in a dedicated file
        # For now, this is a no-op
        pass

    def _metadata_from_dict(self, data: Dict[str, Any]) -> PanoramaMetadata:
        """Construct PanoramaMetadata from stored dict."""

        pano_id = data.get("pano_id") or data.get("id")
        if not pano_id:
            raise ValueError("Metadata payload missing pano_id")

        links = data.get("links") or []

        # Extract values with proper type checking
        lat = float(data.get("lat", 0.0))
        lon = float(data.get("lon", 0.0))
        heading = float(data.get("heading", 0.0))
        pitch = float(data.get("pitch", 0.0)) if data.get("pitch") is not None else None
        roll = float(data.get("roll", 0.0)) if data.get("roll") is not None else None
        date = str(data.get("date")) if data.get("date") is not None else None
        elevation = (
            float(data.get("elevation", 0.0))
            if data.get("elevation") is not None
            else None
        )

        return PanoramaMetadata(
            pano_id=pano_id,
            lat=lat,
            lon=lon,
            heading=heading,
            pitch=pitch,
            roll=roll,
            date=date,
            elevation=elevation,
            links=links,
        )

    def _get_cached_metadata_from_legacy(
        self, pano_id: str
    ) -> Optional[PanoramaMetadata]:
        """Fallback metadata lookup scanning legacy mini files."""

        for metadata_file in self.metadata_dir.glob("*_mini.jsonl"):
            try:
                with open(metadata_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        data = json.loads(line)
                        if data.get("id") == pano_id:
                            try:
                                return self._metadata_from_dict(data)
                            except Exception:
                                return None
            except Exception:
                continue

        return None

    def _normalize_links(self, links: Optional[List[Dict]]) -> List[Dict]:
        """Normalize link payload into env-friendly structure."""

        normalized: List[Dict] = []
        if not links:
            return normalized

        for link in links:
            if not isinstance(link, dict):
                continue

            link_id = link.get("id")
            if link_id is None and isinstance(link.get("pano"), dict):
                link_id = link["pano"].get("id")

            if not link_id:
                continue

            direction = link.get("direction")
            normalized.append({"id": link_id, "direction": direction})

        return normalized

    def _coerce_scalar(self, value):
        """Convert provider values into JSON-friendly scalars."""

        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        try:
            return float(value)
        except (TypeError, ValueError):
            return None
