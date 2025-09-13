"""Tests for mode handlers."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from gymnasium_env.envs.mode_handlers import (
    EpisodeData,
    OfflineModeHandler,
    OnlineModeHandler,
    create_mode_handler,
)
from gymnasium_env.replay import ReplayEpisode, ReplaySession


class TestEpisodeData:
    """Test EpisodeData container."""

    def test_episode_data_creation(self):
        """Test EpisodeData creation and conversion."""
        episode_data = EpisodeData(
            pano_id="test_pano",
            gt_lat=47.6,
            gt_lon=-122.3,
            provider="gsv",
            initial_yaw_deg=45.0,
        )

        assert episode_data.pano_id == "test_pano"
        assert episode_data.gt_lat == 47.6
        assert episode_data.gt_lon == -122.3
        assert episode_data.provider == "gsv"
        assert episode_data.initial_yaw_deg == 45.0

        # Test conversion to dict
        data_dict = episode_data.to_dict()
        expected = {
            "provider": "gsv",
            "gt_lat": 47.6,
            "gt_lon": -122.3,
            "initial_yaw_deg": 45.0,
        }
        assert data_dict == expected


class TestOnlineModeHandler:
    """Test OnlineModeHandler functionality."""

    def test_online_handler_initialization(self):
        """Test OnlineModeHandler initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_root = temp_dir
            metadata_dir = str(Path(temp_dir) / "metadata")
            images_dir = str(Path(temp_dir) / "images")

            handler = OnlineModeHandler(
                cache_root=cache_root,
                metadata_dir=metadata_dir,
                images_dir=images_dir,
                provider="gsv",
                input_lat=47.6,
                input_lon=-122.3,
            )

            assert handler.cache_root == cache_root
            assert handler.metadata_dir == metadata_dir
            assert handler.images_dir == images_dir
            assert handler.provider == "gsv"
            assert handler.input_lat == 47.6
            assert handler.input_lon == -122.3

    def test_online_handler_episode_initialization(self):
        """Test episode initialization in online mode."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_root = temp_dir
            metadata_dir = str(Path(temp_dir) / "metadata")
            images_dir = str(Path(temp_dir) / "images")

            # Create directories
            Path(metadata_dir).mkdir(exist_ok=True)
            Path(images_dir).mkdir(exist_ok=True)

            handler = OnlineModeHandler(
                cache_root=cache_root,
                metadata_dir=metadata_dir,
                images_dir=images_dir,
                provider="gsv",
                input_lat=47.6,
                input_lon=-122.3,
            )

            # Mock the download functions
            with (
                patch(
                    "gymnasium_env.envs.mode_handlers.get_nearest_pano_id"
                ) as mock_get_pano,
                patch("gymnasium_env.envs.mode_handlers.download_metadata"),
                patch("gymnasium_env.envs.mode_handlers.download_images"),
            ):
                mock_get_pano.return_value = "test_pano_123"

                # Create minimal metadata file
                metadata_file = Path(metadata_dir) / "test_pano_123_mini.jsonl"
                with open(metadata_file, "w") as f:
                    f.write(
                        '{"id": "test_pano_123", "lat": 47.6, "lon": -122.3, "heading": 0.0, "links": []}\n'
                    )

                episode_data = handler.initialize_episode(seed=123)

                assert episode_data.pano_id == "test_pano_123"
                assert episode_data.gt_lat == 47.6
                assert episode_data.gt_lon == -122.3
                assert episode_data.provider == "gsv"
                assert episode_data.initial_yaw_deg == 0.0


class TestOfflineModeHandler:
    """Test OfflineModeHandler functionality."""

    def test_offline_handler_initialization(self):
        """Test OfflineModeHandler initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_root = temp_dir
            metadata_dir = str(Path(temp_dir) / "metadata")
            images_dir = str(Path(temp_dir) / "images")
            session_path = str(Path(temp_dir) / "session.json")

            handler = OfflineModeHandler(
                cache_root=cache_root,
                metadata_dir=metadata_dir,
                images_dir=images_dir,
                replay_session_path=session_path,
                provider="gsv",
            )

            assert handler.cache_root == cache_root
            assert handler.metadata_dir == metadata_dir
            assert handler.images_dir == images_dir
            assert handler.replay_session_path == session_path
            assert handler.provider == "gsv"

    def test_offline_handler_episode_initialization(self):
        """Test episode initialization in offline mode."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_root = temp_dir
            metadata_dir = str(Path(temp_dir) / "metadata")
            images_dir = str(Path(temp_dir) / "images")

            # Create directories
            Path(metadata_dir).mkdir(exist_ok=True)
            Path(images_dir).mkdir(exist_ok=True)

            # Create replay session
            episodes = [
                ReplayEpisode("gsv", "offline_pano", 48.6, -123.3, 30.0),
            ]
            session = ReplaySession(seed=789, episodes=episodes)
            session_path = str(Path(temp_dir) / "session.json")
            session.save_to_file(session_path)

            # Create cached metadata and image files
            metadata_file = Path(metadata_dir) / "offline_pano_mini.jsonl"
            with open(metadata_file, "w") as f:
                f.write(
                    '{"id": "offline_pano", "lat": 48.6, "lon": -123.3, "heading": 30.0, "links": []}\n'
                )

            from PIL import Image

            image_file = Path(images_dir) / "offline_pano.jpg"
            img = Image.new("RGB", (1024, 512), color="blue")
            img.save(image_file)

            handler = OfflineModeHandler(
                cache_root=cache_root,
                metadata_dir=metadata_dir,
                images_dir=images_dir,
                replay_session_path=session_path,
                provider="gsv",
            )

            episode_data = handler.initialize_episode(seed=789)

            assert episode_data.pano_id == "offline_pano"
            assert episode_data.gt_lat == 48.6
            assert episode_data.gt_lon == -123.3
            assert episode_data.provider == "gsv"
            assert episode_data.initial_yaw_deg == 30.0

    def test_offline_handler_missing_cache_error(self):
        """Test that offline handler raises error when cache is missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_root = temp_dir
            metadata_dir = str(Path(temp_dir) / "metadata")
            images_dir = str(Path(temp_dir) / "images")

            # Create directories
            Path(metadata_dir).mkdir(exist_ok=True)
            Path(images_dir).mkdir(exist_ok=True)

            # Create replay session
            episodes = [
                ReplayEpisode("gsv", "missing_pano", 48.6, -123.3, 30.0),
            ]
            session = ReplaySession(seed=789, episodes=episodes)
            session_path = str(Path(temp_dir) / "session.json")
            session.save_to_file(session_path)

            # Note: NOT creating cached files

            handler = OfflineModeHandler(
                cache_root=cache_root,
                metadata_dir=metadata_dir,
                images_dir=images_dir,
                replay_session_path=session_path,
                provider="gsv",
            )

            # Should raise error due to missing cache
            with pytest.raises(FileNotFoundError, match="Cached metadata not found"):
                handler.initialize_episode(seed=789)


class TestModeHandlerFactory:
    """Test mode handler factory function."""

    def test_create_online_handler(self):
        """Test creating online mode handler."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_root = temp_dir
            metadata_dir = str(Path(temp_dir) / "metadata")
            images_dir = str(Path(temp_dir) / "images")

            handler = create_mode_handler(
                mode="online",
                cache_root=cache_root,
                metadata_dir=metadata_dir,
                images_dir=images_dir,
                provider="gsv",
                input_lat=47.6,
                input_lon=-122.3,
            )

            assert isinstance(handler, OnlineModeHandler)
            assert handler.provider == "gsv"
            assert handler.input_lat == 47.6
            assert handler.input_lon == -122.3

    def test_create_offline_handler(self):
        """Test creating offline mode handler."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_root = temp_dir
            metadata_dir = str(Path(temp_dir) / "metadata")
            images_dir = str(Path(temp_dir) / "images")
            session_path = str(Path(temp_dir) / "session.json")

            handler = create_mode_handler(
                mode="offline",
                cache_root=cache_root,
                metadata_dir=metadata_dir,
                images_dir=images_dir,
                replay_session_path=session_path,
                provider="gsv",
            )

            assert isinstance(handler, OfflineModeHandler)
            assert handler.provider == "gsv"
            assert handler.replay_session_path == session_path

    def test_create_invalid_mode_error(self):
        """Test that invalid mode raises error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ValueError, match="Unknown mode: invalid"):
                create_mode_handler(
                    mode="invalid",
                    cache_root=temp_dir,
                    metadata_dir=str(Path(temp_dir) / "metadata"),
                    images_dir=str(Path(temp_dir) / "images"),
                )

    def test_offline_mode_missing_session_path_error(self):
        """Test that offline mode without session path raises error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ValueError, match="replay_session_path is required"):
                create_mode_handler(
                    mode="offline",
                    cache_root=temp_dir,
                    metadata_dir=str(Path(temp_dir) / "metadata"),
                    images_dir=str(Path(temp_dir) / "images"),
                )
