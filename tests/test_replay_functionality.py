"""Tests for session replay functionality."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from gymnasium_env.envs import GeoGuessrWorldEnv
from gymnasium_env.replay import ReplayEpisode, ReplayManager, ReplaySession


class TestReplayModels:
    """Test replay data models."""

    def test_replay_episode_creation(self):
        """Test ReplayEpisode creation and serialization."""
        episode = ReplayEpisode(
            provider="gsv",
            pano_id="test_pano_123",
            gt_lat=47.620908,
            gt_lon=-122.353508,
            initial_yaw_deg=90.0,
        )

        assert episode.provider == "gsv"
        assert episode.pano_id == "test_pano_123"
        assert episode.gt_lat == 47.620908
        assert episode.gt_lon == -122.353508
        assert episode.initial_yaw_deg == 90.0

        # Test serialization
        data = episode.to_dict()
        expected = {
            "provider": "gsv",
            "pano_id": "test_pano_123",
            "gt_lat": 47.620908,
            "gt_lon": -122.353508,
            "initial_yaw_deg": 90.0,
        }
        assert data == expected

    def test_replay_session_creation(self):
        """Test ReplaySession creation and serialization."""
        episodes = [
            ReplayEpisode("gsv", "pano1", 47.6, -122.3, 0.0),
            ReplayEpisode("gsv", "pano2", 48.6, -123.3, 45.0),
        ]
        session = ReplaySession(seed=123, episodes=episodes)

        assert session.seed == 123
        assert len(session.episodes) == 2
        assert session.episodes[0].pano_id == "pano1"
        assert session.episodes[1].pano_id == "pano2"

    def test_replay_session_file_operations(self):
        """Test saving and loading replay sessions."""
        episodes = [
            ReplayEpisode("gsv", "pano1", 47.6, -122.3, 0.0),
        ]
        original_session = ReplaySession(seed=456, episodes=episodes)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            # Save session
            original_session.save_to_file(temp_path)

            # Load session
            loaded_session = ReplaySession.load_from_file(temp_path)

            # Verify loaded session matches original
            assert loaded_session.seed == original_session.seed
            assert len(loaded_session.episodes) == len(original_session.episodes)
            assert (
                loaded_session.episodes[0].pano_id
                == original_session.episodes[0].pano_id
            )
            assert (
                loaded_session.episodes[0].gt_lat == original_session.episodes[0].gt_lat
            )
            assert (
                loaded_session.episodes[0].gt_lon == original_session.episodes[0].gt_lon
            )

        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestReplayManager:
    """Test replay manager functionality."""

    def test_replay_manager_initialization(self):
        """Test ReplayManager initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ReplayManager(temp_dir)

            assert manager.cache_root == Path(temp_dir)
            assert manager.replays_dir.exists()
            assert manager.current_session is None
            assert manager.current_episode_index == 0

    def test_session_recording_and_saving(self):
        """Test session recording workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ReplayManager(temp_dir)

            # Start recording
            manager.start_recording_session(seed=789)
            assert manager.current_session is not None
            assert manager.current_session.seed == 789
            assert len(manager.current_session.episodes) == 0

            # Add episodes
            manager.add_episode_to_session("gsv", "pano1", 47.6, -122.3, 0.0)
            manager.add_episode_to_session("gsv", "pano2", 48.6, -123.3, 45.0)

            assert len(manager.current_session.episodes) == 2

            # Save session
            session_path = manager.save_session()
            assert Path(session_path).exists()

            # Verify saved content
            with open(session_path, "r") as f:
                saved_data = json.load(f)

            assert saved_data["seed"] == 789
            assert len(saved_data["episodes"]) == 2
            assert saved_data["episodes"][0]["pano_id"] == "pano1"

    def test_session_loading_and_playback(self):
        """Test session loading and episode playback."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create and save a session
            episodes = [
                ReplayEpisode("gsv", "pano1", 47.6, -122.3, 0.0),
                ReplayEpisode("gsv", "pano2", 48.6, -123.3, 45.0),
            ]
            session = ReplaySession(seed=999, episodes=episodes)

            session_path = Path(temp_dir) / "test_session.json"
            session.save_to_file(str(session_path))

            # Load session with manager
            manager = ReplayManager(temp_dir)
            loaded_session = manager.load_session(str(session_path))

            assert loaded_session.seed == 999
            assert len(loaded_session.episodes) == 2

            # Test episode playback
            episode1 = manager.get_next_episode()
            assert episode1 is not None
            assert episode1.pano_id == "pano1"

            episode2 = manager.get_next_episode()
            assert episode2 is not None
            assert episode2.pano_id == "pano2"

            # No more episodes
            episode3 = manager.get_next_episode()
            assert episode3 is None

            # Reset and try again
            manager.reset_episode_index()
            episode1_again = manager.get_next_episode()
            assert episode1_again is not None
            assert episode1_again.pano_id == "pano1"


class TestEnvironmentReplayIntegration:
    """Test replay functionality integrated with GeoGuessrWorldEnv."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Provide a temporary cache directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    def test_online_mode_session_recording(self, temp_cache_dir):
        """Test that online mode can record sessions."""
        # Create a simple replay session file for testing
        episodes = [
            ReplayEpisode("gsv", "test_pano_123", 47.620908, -122.353508, 0.0),
        ]
        session = ReplaySession(seed=123, episodes=episodes)
        session_path = Path(temp_cache_dir) / "replays" / "test_session.json"
        session.save_to_file(str(session_path))

        config = {
            "mode": "online",
            "cache_root": temp_cache_dir,
            "input_lat": 47.620908,
            "input_lon": -122.353508,
            "freeze_run_path": "test_freeze.json",
        }

        env = GeoGuessrWorldEnv(config=config)

        # Mock the download functions to avoid network calls
        with (
            patch(
                "gymnasium_env.envs.geoguessr_world.get_nearest_pano_id"
            ) as mock_get_pano,
            patch("gymnasium_env.envs.geoguessr_world.download_metadata"),
            patch("gymnasium_env.envs.geoguessr_world.download_images"),
        ):
            mock_get_pano.return_value = "test_pano_123"

            # Create minimal metadata file
            metadata_dir = Path(temp_cache_dir) / "metadata"
            metadata_dir.mkdir(exist_ok=True)
            metadata_file = metadata_dir / "test_pano_123_mini.jsonl"

            with open(metadata_file, "w") as f:
                f.write(
                    '{"id": "test_pano_123", "lat": 47.620908, "lon": -122.353508, "heading": 0.0, "links": []}\n'
                )

            # Create minimal image file
            images_dir = Path(temp_cache_dir) / "images"
            images_dir.mkdir(exist_ok=True)
            image_file = images_dir / "test_pano_123.jpg"

            # Create a minimal 1x1 RGB image
            from PIL import Image

            img = Image.new("RGB", (1024, 512), color="red")
            img.save(image_file)

            try:
                obs, info = env.reset(seed=123)

                # Verify environment state
                assert info["provider"] == "gsv"
                assert info["pano_id"] == "test_pano_123"
                assert info["gt_lat"] == 47.620908
                assert info["gt_lon"] == -122.353508

                # Verify observation format
                assert "image" in obs
                assert obs["image"].shape == (512, 1024, 3)

            finally:
                env.close()

    def test_offline_mode_replay(self, temp_cache_dir):
        """Test that offline mode can replay from session files."""
        # Create a replay session
        episodes = [
            ReplayEpisode("gsv", "offline_pano_456", 48.620908, -123.353508, 45.0),
        ]
        session = ReplaySession(seed=456, episodes=episodes)
        session_path = Path(temp_cache_dir) / "replays" / "offline_session.json"
        session.save_to_file(str(session_path))

        # Create required cached metadata and image files
        metadata_dir = Path(temp_cache_dir) / "metadata"
        metadata_dir.mkdir(exist_ok=True)
        metadata_file = metadata_dir / "offline_pano_456_mini.jsonl"

        with open(metadata_file, "w") as f:
            f.write(
                '{"id": "offline_pano_456", "lat": 48.620908, "lon": -123.353508, "heading": 45.0, "links": []}\n'
            )

        images_dir = Path(temp_cache_dir) / "images"
        images_dir.mkdir(exist_ok=True)
        image_file = images_dir / "offline_pano_456.jpg"

        from PIL import Image

        img = Image.new("RGB", (1024, 512), color="blue")
        img.save(image_file)

        config = {
            "mode": "offline",
            "cache_root": temp_cache_dir,
            "replay_session_path": str(session_path),
        }

        env = GeoGuessrWorldEnv(config=config)

        try:
            obs, info = env.reset()

            # Verify environment loaded from replay
            assert info["provider"] == "gsv"
            assert info["pano_id"] == "offline_pano_456"
            assert info["gt_lat"] == 48.620908
            assert info["gt_lon"] == -123.353508
            assert (
                info["pose"]["heading_deg"] == 45.0
            )  # Should use initial_yaw_deg from replay

            # Verify observation format
            assert "image" in obs
            assert obs["image"].shape == (512, 1024, 3)

        finally:
            env.close()

    def test_deterministic_replay(self, temp_cache_dir):
        """Test that replays are deterministic."""
        # Create replay session with specific seed
        episodes = [
            ReplayEpisode("gsv", "deterministic_pano", 49.0, -124.0, 30.0),
        ]
        session = ReplaySession(seed=777, episodes=episodes)
        session_path = Path(temp_cache_dir) / "replays" / "deterministic_session.json"
        session.save_to_file(str(session_path))

        # Create cached files
        metadata_dir = Path(temp_cache_dir) / "metadata"
        metadata_dir.mkdir(exist_ok=True)
        metadata_file = metadata_dir / "deterministic_pano_mini.jsonl"

        with open(metadata_file, "w") as f:
            f.write(
                '{"id": "deterministic_pano", "lat": 49.0, "lon": -124.0, "heading": 30.0, "links": []}\n'
            )

        images_dir = Path(temp_cache_dir) / "images"
        images_dir.mkdir(exist_ok=True)
        image_file = images_dir / "deterministic_pano.jpg"

        from PIL import Image

        img = Image.new("RGB", (1024, 512), color="green")
        img.save(image_file)

        config = {
            "mode": "offline",
            "cache_root": temp_cache_dir,
            "replay_session_path": str(session_path),
        }

        # Run environment twice and verify identical results
        results = []
        for i in range(2):
            env = GeoGuessrWorldEnv(config=config)
            try:
                obs, info = env.reset(seed=777)  # Same seed
                results.append(
                    {
                        "pano_id": info["pano_id"],
                        "gt_lat": info["gt_lat"],
                        "gt_lon": info["gt_lon"],
                        "heading": info["pose"]["heading_deg"],
                    }
                )
            finally:
                env.close()

        # Verify results are identical
        assert results[0] == results[1]
        assert results[0]["pano_id"] == "deterministic_pano"
        assert results[0]["gt_lat"] == 49.0
        assert results[0]["gt_lon"] == -124.0
        # Use approximate comparison for floating point heading
        assert abs(results[0]["heading"] - 30.0) < 1e-10
