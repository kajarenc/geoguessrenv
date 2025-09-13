"""Mode handlers for online and offline operation of GeoGuessr environment."""

import os
from abc import ABC, abstractmethod
from typing import Dict, Optional

from gymnasium_env.envs.helpers import (
    download_images,
    download_metadata,
    get_nearest_pano_id,
)
from gymnasium_env.replay import ReplayManager


class EpisodeData:
    """Container for episode initialization data."""

    def __init__(
        self,
        pano_id: str,
        gt_lat: float,
        gt_lon: float,
        provider: str = "gsv",
        initial_yaw_deg: float = 0.0,
    ):
        self.pano_id = pano_id
        self.gt_lat = gt_lat
        self.gt_lon = gt_lon
        self.provider = provider
        self.initial_yaw_deg = initial_yaw_deg

    def to_dict(self) -> Dict:
        """Convert to dictionary for compatibility."""
        return {
            "provider": self.provider,
            "gt_lat": self.gt_lat,
            "gt_lon": self.gt_lon,
            "initial_yaw_deg": self.initial_yaw_deg,
        }


class ModeHandler(ABC):
    """Abstract base class for environment mode handlers."""

    def __init__(
        self,
        cache_root: str,
        metadata_dir: str,
        images_dir: str,
        provider: str = "gsv",
    ):
        self.cache_root = cache_root
        self.metadata_dir = metadata_dir
        self.images_dir = images_dir
        self.provider = provider

    @abstractmethod
    def initialize_episode(
        self,
        seed: Optional[int] = None,
        **kwargs,
    ) -> EpisodeData:
        """Initialize a new episode and return episode data."""
        pass

    @abstractmethod
    def load_pano_graph(self, pano_id: str) -> Dict[str, Dict]:
        """Load panorama graph for the given panorama ID."""
        pass

    def _load_minimetadata(self, jsonl_path: str) -> Dict[str, Dict]:
        """Load minimetadata from JSONL file."""
        import json

        graph: Dict[str, Dict] = {}
        if not os.path.isfile(jsonl_path):
            raise FileNotFoundError(f"Metadata file not found: {jsonl_path}")

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                pano_id = data.get("id")
                if not pano_id:
                    continue

                lat = data.get("lat")
                lon = data.get("lon")
                heading = data.get("heading")
                links_raw = data.get("links", []) or []
                links = []

                for link in links_raw:
                    link_pano = (link or {}).get("pano") or {}
                    link_id = link_pano.get("id")
                    direction = link.get("direction")
                    if isinstance(link_id, str) and isinstance(direction, (int, float)):
                        links.append({"id": link_id, "direction": float(direction)})

                graph[pano_id] = {
                    "lat": lat,
                    "lon": lon,
                    "heading": heading,
                    "links": links,
                }

        # Prune links pointing to non-existent nodes
        valid_ids = set(graph.keys())
        for node in graph.values():
            raw_links = node.get("links", []) or []
            node["links"] = [
                link
                for link in raw_links
                if isinstance(link, dict)
                and isinstance(link.get("id"), str)
                and link["id"] in valid_ids
            ]

        return graph


class OnlineModeHandler(ModeHandler):
    """Handler for online mode - downloads data and optionally records sessions."""

    def __init__(
        self,
        cache_root: str,
        metadata_dir: str,
        images_dir: str,
        provider: str = "gsv",
        geofence: Optional[Dict] = None,
        input_lat: Optional[float] = None,
        input_lon: Optional[float] = None,
        freeze_run_path: Optional[str] = None,
        rate_limit_qps: Optional[float] = None,
        max_fetch_retries: Optional[int] = None,
        min_capture_year: Optional[int] = None,
    ):
        super().__init__(cache_root, metadata_dir, images_dir, provider)
        self.geofence = geofence
        self.input_lat = input_lat
        self.input_lon = input_lon
        self.freeze_run_path = freeze_run_path
        self.rate_limit_qps = rate_limit_qps
        self.max_fetch_retries = max_fetch_retries
        self.min_capture_year = min_capture_year

        self._replay_manager = ReplayManager(cache_root)

    def initialize_episode(
        self,
        seed: Optional[int] = None,
        **kwargs,
    ) -> EpisodeData:
        """Initialize episode in online mode."""
        # Start recording session if seed provided
        if seed is not None:
            self._replay_manager.start_recording_session(seed, self.geofence)

        # Determine coordinates
        if self.input_lat is not None and self.input_lon is not None:
            lat, lon = self.input_lat, self.input_lon
        elif seed is not None and hasattr(self._replay_manager, "_geofence_sampler"):
            lat, lon = self._replay_manager.generate_episode_from_geofence()
        else:
            raise ValueError(
                "input_lat and input_lon must be provided when not using geofence sampling"
            )

        # Find nearest panorama
        pano_id = get_nearest_pano_id(lat, lon, self.metadata_dir)
        if pano_id is None:
            raise RuntimeError(f"No panorama found near coordinates ({lat}, {lon})")

        # Download data
        download_metadata(pano_id, self.metadata_dir)
        download_images(pano_id, self.metadata_dir, self.images_dir)

        # Load metadata to get ground truth coordinates
        metadata_path = os.path.join(self.metadata_dir, f"{pano_id}_mini.jsonl")
        pano_graph = self._load_minimetadata(metadata_path)
        node = pano_graph.get(pano_id, {})
        gt_lat = node.get("lat", lat)
        gt_lon = node.get("lon", lon)

        # Record episode if session is active
        if self._replay_manager.current_session is not None:
            self._replay_manager.add_episode_to_session(
                provider=self.provider,
                pano_id=pano_id,
                gt_lat=gt_lat,
                gt_lon=gt_lon,
                initial_yaw_deg=0.0,
            )

            # Save session if freeze_run_path specified
            if self.freeze_run_path:
                self._replay_manager.save_session(self.freeze_run_path)

        return EpisodeData(
            pano_id=pano_id,
            gt_lat=gt_lat,
            gt_lon=gt_lon,
            provider=self.provider,
            initial_yaw_deg=0.0,
        )

    def load_pano_graph(self, pano_id: str) -> Dict[str, Dict]:
        """Load panorama graph (data should already be downloaded)."""
        metadata_path = os.path.join(self.metadata_dir, f"{pano_id}_mini.jsonl")
        return self._load_minimetadata(metadata_path)


class OfflineModeHandler(ModeHandler):
    """Handler for offline mode - loads from replay sessions."""

    def __init__(
        self,
        cache_root: str,
        metadata_dir: str,
        images_dir: str,
        replay_session_path: str,
        provider: str = "gsv",
    ):
        super().__init__(cache_root, metadata_dir, images_dir, provider)
        self.replay_session_path = replay_session_path
        self._replay_manager = ReplayManager(cache_root)
        self._session_loaded = False

    def initialize_episode(
        self,
        seed: Optional[int] = None,
        **kwargs,
    ) -> EpisodeData:
        """Initialize episode in offline mode from replay session."""
        # Load session if not already loaded
        if not self._session_loaded:
            self._replay_manager.load_session(self.replay_session_path)
            self._session_loaded = True

        # Get next episode from session
        episode = self._replay_manager.get_next_episode()
        if episode is None:
            raise RuntimeError("No episodes available in replay session")

        # Verify cached data exists (no network calls in offline mode)
        metadata_path = os.path.join(self.metadata_dir, f"{episode.pano_id}_mini.jsonl")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Cached metadata not found for offline mode: {metadata_path}"
            )

        # Verify image exists
        image_path = os.path.join(self.images_dir, f"{episode.pano_id}.jpg")
        if not os.path.exists(image_path):
            raise FileNotFoundError(
                f"Cached image not found for offline mode: {image_path}"
            )

        return EpisodeData(
            pano_id=episode.pano_id,
            gt_lat=episode.gt_lat,
            gt_lon=episode.gt_lon,
            provider=episode.provider,
            initial_yaw_deg=episode.initial_yaw_deg,
        )

    def load_pano_graph(self, pano_id: str) -> Dict[str, Dict]:
        """Load panorama graph from cached data only."""
        metadata_path = os.path.join(self.metadata_dir, f"{pano_id}_mini.jsonl")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Cached metadata not found for offline mode: {metadata_path}"
            )
        return self._load_minimetadata(metadata_path)

    def reset_episode_index(self):
        """Reset episode index to replay from the beginning."""
        if self._session_loaded:
            self._replay_manager.reset_episode_index()


def create_mode_handler(
    mode: str,
    cache_root: str,
    metadata_dir: str,
    images_dir: str,
    **config,
) -> ModeHandler:
    """Factory function to create appropriate mode handler."""
    if mode == "online":
        return OnlineModeHandler(
            cache_root=cache_root,
            metadata_dir=metadata_dir,
            images_dir=images_dir,
            provider=config.get("provider", "gsv"),
            geofence=config.get("geofence"),
            input_lat=config.get("input_lat"),
            input_lon=config.get("input_lon"),
            freeze_run_path=config.get("freeze_run_path"),
            rate_limit_qps=config.get("rate_limit_qps"),
            max_fetch_retries=config.get("max_fetch_retries"),
            min_capture_year=config.get("min_capture_year"),
        )
    elif mode == "offline":
        replay_session_path = config.get("replay_session_path")
        if not replay_session_path:
            raise ValueError("replay_session_path is required for offline mode")

        return OfflineModeHandler(
            cache_root=cache_root,
            metadata_dir=metadata_dir,
            images_dir=images_dir,
            replay_session_path=replay_session_path,
            provider=config.get("provider", "gsv"),
        )
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'online' or 'offline'")

