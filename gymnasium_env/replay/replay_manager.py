"""Replay manager for handling session recording and playback."""

import random
from pathlib import Path
from typing import Dict, Optional, Tuple

from .models import ReplayEpisode, ReplaySession


class GeofenceSampler:
    """Simple geofence sampler for generating random coordinates."""

    def __init__(self, geofence: Optional[Dict] = None, seed: Optional[int] = None):
        """Initialize geofence sampler.

        Args:
            geofence: Geofence configuration (for now, uses world bounds if None)
            seed: Random seed for deterministic sampling
        """
        self.geofence = geofence or self._default_world_geofence()
        self.rng = random.Random(seed)

    def _default_world_geofence(self) -> Dict:
        """Default world geofence with basic land areas."""
        return {
            "type": "world_basic",
            "regions": [
                {
                    "lat_min": 40.0,
                    "lat_max": 50.0,
                    "lon_min": -125.0,
                    "lon_max": -65.0,
                },  # North America
                {
                    "lat_min": 35.0,
                    "lat_max": 60.0,
                    "lon_min": -10.0,
                    "lon_max": 30.0,
                },  # Europe
                {
                    "lat_min": -40.0,
                    "lat_max": -10.0,
                    "lon_min": 110.0,
                    "lon_max": 155.0,
                },  # Australia
            ],
        }

    def sample_coordinates(self) -> Tuple[float, float]:
        """Sample random coordinates within geofence."""
        regions = self.geofence.get("regions", [])
        if not regions:
            # Fallback to simple world bounds
            lat = self.rng.uniform(-60.0, 60.0)
            lon = self.rng.uniform(-180.0, 180.0)
            return lat, lon

        # Choose random region
        region = self.rng.choice(regions)
        lat = self.rng.uniform(region["lat_min"], region["lat_max"])
        lon = self.rng.uniform(region["lon_min"], region["lon_max"])
        return lat, lon


class ReplayManager:
    """Manages session recording and playback for the GeoGuessr environment."""

    def __init__(self, cache_root: str):
        """Initialize replay manager.

        Args:
            cache_root: Root directory for caching (replays will be stored in cache_root/replays/)
        """
        self.cache_root = Path(cache_root)
        self.replays_dir = self.cache_root / "replays"
        self.replays_dir.mkdir(parents=True, exist_ok=True)

        # Current session state
        self.current_session: Optional[ReplaySession] = None
        self.current_episode_index: int = 0

    def start_recording_session(
        self, seed: int, geofence: Optional[Dict] = None
    ) -> None:
        """Start recording a new session.

        Args:
            seed: Random seed for deterministic episode generation
            geofence: Geofence configuration for coordinate sampling
        """
        self.current_session = ReplaySession(seed=seed, episodes=[])
        self.current_episode_index = 0
        self._geofence_sampler = GeofenceSampler(geofence, seed)

    def add_episode_to_session(
        self,
        provider: str,
        pano_id: str,
        gt_lat: float,
        gt_lon: float,
        initial_yaw_deg: float = 0.0,
    ) -> None:
        """Add an episode to the current recording session.

        Args:
            provider: Panorama provider (e.g., "gsv")
            pano_id: Panorama ID
            gt_lat: Ground truth latitude
            gt_lon: Ground truth longitude
            initial_yaw_deg: Initial camera yaw in degrees
        """
        if self.current_session is None:
            raise RuntimeError(
                "No recording session started. Call start_recording_session() first."
            )

        episode = ReplayEpisode(
            provider=provider,
            pano_id=pano_id,
            gt_lat=gt_lat,
            gt_lon=gt_lon,
            initial_yaw_deg=initial_yaw_deg,
        )
        self.current_session.episodes.append(episode)

    def save_session(self, session_name: Optional[str] = None) -> str:
        """Save the current session to a file.

        Args:
            session_name: Optional custom name for the session file

        Returns:
            Path to the saved session file
        """
        if self.current_session is None:
            raise RuntimeError(
                "No session to save. Call start_recording_session() first."
            )

        if session_name is None:
            session_name = f"session_{self.current_session.seed}.json"

        session_path = self.replays_dir / session_name
        self.current_session.save_to_file(str(session_path))
        return str(session_path)

    def load_session(self, session_path: str) -> ReplaySession:
        """Load a replay session from file.

        Args:
            session_path: Path to the session file

        Returns:
            Loaded replay session
        """
        session = ReplaySession.load_from_file(session_path)
        self.current_session = session
        self.current_episode_index = 0
        return session

    def get_next_episode(self) -> Optional[ReplayEpisode]:
        """Get the next episode from the current session.

        Returns:
            Next episode or None if no more episodes
        """
        if self.current_session is None:
            return None

        if self.current_episode_index >= len(self.current_session.episodes):
            return None

        episode = self.current_session.episodes[self.current_episode_index]
        self.current_episode_index += 1
        return episode

    def reset_episode_index(self) -> None:
        """Reset episode index to start from the beginning."""
        self.current_episode_index = 0

    def generate_episode_from_geofence(
        self, provider: str = "gsv"
    ) -> Tuple[float, float]:
        """Generate a new episode by sampling from geofence.

        Args:
            provider: Panorama provider

        Returns:
            Tuple of (lat, lon) coordinates
        """
        if not hasattr(self, "_geofence_sampler"):
            # Fallback if no sampler is set up
            self._geofence_sampler = GeofenceSampler()

        lat, lon = self._geofence_sampler.sample_coordinates()
        return lat, lon

    def get_session_file_path(self, seed: int) -> str:
        """Get the file path for a session with the given seed.

        Args:
            seed: Session seed

        Returns:
            Path to the session file
        """
        return str(self.replays_dir / f"session_{seed}.json")
