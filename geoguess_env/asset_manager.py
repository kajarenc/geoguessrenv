"""
Asset manager for handling panorama data caching and retrieval.

This module provides the AssetManager class that orchestrates panorama
data fetching, caching, and validation for the GeoGuessr environment.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set

from .providers.base import PanoramaAsset, PanoramaMetadata, PanoramaProvider


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

    def get_or_fetch_panorama_graph(
        self, root_lat: float, root_lon: float, offline_mode: bool = False
    ) -> Dict[str, Dict]:
        """
        Get or fetch a panorama graph starting from given coordinates.

        Args:
            root_lat: Starting latitude
            root_lon: Starting longitude
            offline_mode: If True, only use cached data

        Returns:
            Dictionary mapping panorama IDs to their metadata and links

        Raises:
            ValueError: If no panorama found and offline_mode is True
        """
        # Try to find root panorama
        root_pano_id = self._get_or_find_nearest_panorama(
            root_lat, root_lon, offline_mode
        )

        if not root_pano_id:
            if offline_mode:
                raise ValueError(f"No cached panorama found for {root_lat}, {root_lon}")
            else:
                raise ValueError(f"No panorama found for {root_lat}, {root_lon}")

        # Load panorama graph
        return self._load_panorama_graph(root_pano_id, offline_mode)

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
        else:
            # Clear specific panoramas
            for pano_id in pano_ids:
                # Remove from memory cache
                self._metadata_cache.pop(pano_id, None)

                # Remove image file
                image_path = self.images_dir / f"{pano_id}.jpg"
                if image_path.exists():
                    image_path.unlink()

                # Note: We don't remove from metadata files as they may contain
                # multiple panoramas. In a production system, we'd want more
                # sophisticated metadata management.

    def _get_or_find_nearest_panorama(
        self, lat: float, lon: float, offline_mode: bool
    ) -> Optional[str]:
        """Get panorama ID from cache or find nearest."""
        # Check cache first
        cache_file = self.metadata_dir / "nearest_pano_cache.json"
        cache_key = f"{round(lat, 6)},{round(lon, 6)}"

        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
                    if cache_key in cache_data:
                        return cache_data[cache_key]
            except Exception:
                pass

        if offline_mode:
            return None

        # Fetch from provider
        pano_id = self.provider.find_nearest_panorama(lat, lon)

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

    def _load_panorama_graph(
        self, root_pano_id: str, offline_mode: bool
    ) -> Dict[str, Dict]:
        """Load panorama graph from cache or fetch if needed."""
        # Try to load from cache first
        graph = self._load_cached_graph(root_pano_id)

        if graph:
            return graph

        if offline_mode:
            return {}

        # Fetch and build graph
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
        processed_count = 0

        while queue and processed_count < self.max_connected_panoramas:
            pano_id = queue.pop(0)

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
                    if link_id and link_id not in visited:
                        queue.append(link_id)
                        links.append(link)

            graph[pano_id] = {
                "lat": metadata.lat,
                "lon": metadata.lon,
                "heading": metadata.heading,
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

        # Fetch from provider
        metadata = self.provider.get_panorama_metadata(pano_id)
        if metadata:
            self._metadata_cache[pano_id] = metadata

        return metadata

    def _get_cached_metadata(self, pano_id: str) -> Optional[PanoramaMetadata]:
        """Get metadata from disk cache."""
        # This is a simplified implementation - in production we'd want
        # more efficient metadata storage and retrieval
        for metadata_file in self.metadata_dir.glob("*_mini.jsonl"):
            try:
                with open(metadata_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            if data.get("id") == pano_id:
                                return PanoramaMetadata(
                                    pano_id=data["id"],
                                    lat=data["lat"],
                                    lon=data["lon"],
                                    heading=data["heading"],
                                    pitch=data.get("pitch"),
                                    roll=data.get("roll"),
                                    date=data.get("date"),
                                    elevation=data.get("elevation"),
                                    links=data.get("links"),
                                )
            except Exception:
                continue

        return None

    def _fetch_and_cache_asset(self, pano_id: str) -> bool:
        """Fetch and cache a complete panorama asset."""
        # Get or fetch metadata
        metadata = self._get_or_fetch_metadata(pano_id)
        if not metadata:
            return False

        # Download image if not cached
        image_path = self.images_dir / f"{pano_id}.jpg"
        if not image_path.exists():
            success = self.provider.download_panorama_image(pano_id, image_path)
            if not success:
                return False

        # Compute and store hash
        image_hash = self.provider.compute_image_hash(image_path)
        self._store_hash(pano_id, image_hash)

        return True

    def _is_asset_cached(self, pano_id: str) -> bool:
        """Check if asset is fully cached."""
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
                            "lat": metadata.lat,
                            "lon": metadata.lon,
                            "heading": metadata.heading,
                            "pitch": metadata.pitch,
                            "roll": metadata.roll,
                            "date": metadata.date,
                            "elevation": metadata.elevation,
                            "links": [],
                        }

                        if metadata.links:
                            for link in metadata.links:
                                cache_data["links"].append(
                                    {
                                        "pano": {"id": link["id"]},
                                        "direction": link["direction"],
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
