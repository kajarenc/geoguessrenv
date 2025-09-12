"""Data models for session replay functionality."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class ReplayEpisode:
    """Represents a single episode in a replay session."""

    provider: str
    pano_id: str
    gt_lat: float
    gt_lon: float
    initial_yaw_deg: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "provider": self.provider,
            "pano_id": self.pano_id,
            "gt_lat": self.gt_lat,
            "gt_lon": self.gt_lon,
            "initial_yaw_deg": self.initial_yaw_deg,
        }


@dataclass
class ReplaySession:
    """Represents a complete replay session with multiple episodes."""

    seed: int
    episodes: List[ReplayEpisode]

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "seed": self.seed,
            "episodes": [episode.to_dict() for episode in self.episodes],
        }

    def save_to_file(self, file_path: str) -> None:
        """Save replay session to JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load_from_file(cls, file_path: str) -> "ReplaySession":
        """Load replay session from JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        episodes = []
        for episode_data in data.get("episodes", []):
            episodes.append(ReplayEpisode(**episode_data))

        return cls(
            seed=data["seed"],
            episodes=episodes,
        )

    @classmethod
    def load_from_dict(cls, data: Dict) -> "ReplaySession":
        """Load replay session from dictionary."""
        episodes = []
        for episode_data in data.get("episodes", []):
            episodes.append(ReplayEpisode(**episode_data))

        return cls(
            seed=data["seed"],
            episodes=episodes,
        )
