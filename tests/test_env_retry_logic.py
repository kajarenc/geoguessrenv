"""Tests for environment retry logic and data loading flow with blocklist."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from geoguess_env.asset_manager import AssetManager, RootPanoramaUnavailableError
from geoguess_env.geoguessr_env import GeoGuessrEnv
from geoguess_env.providers.google_streetview import GoogleStreetViewProvider


class TestEnvironmentRetryLogic:
    """Test suite for environment reset retry logic with download failures."""

    def test_reset_retries_on_root_panorama_unavailable(self):
        """Test that reset retries with new coordinates when root panorama is unavailable."""
        config = {
            "cache_root": tempfile.mkdtemp(),
            "geofence": {
                "type": "circle",
                "center": {"lat": 47.620908, "lon": -122.353508},
                "radius_km": 10.0,
            },
            "max_steps": 5,
            "seed": 42,
        }

        env = GeoGuessrEnv(config=config)

        # Mock asset manager to simulate failures then success
        call_count = 0
        attempted_coordinates = []

        def mock_prepare(root_lat, root_lon):
            nonlocal call_count
            call_count += 1
            attempted_coordinates.append((root_lat, root_lon))

            if call_count <= 2:
                # First two attempts fail
                raise RootPanoramaUnavailableError(
                    f"Simulated failure at ({root_lat}, {root_lon})"
                )
            else:
                # Third attempt succeeds
                return Mock(
                    root_id="success_pano",
                    graph={
                        "success_pano": {
                            "lat": root_lat,
                            "lon": root_lon,
                            "heading": 0.0,
                            "links": [],
                        }
                    },
                    missing_assets=set(),
                )

        with patch.object(env.asset_manager, "prepare_graph", side_effect=mock_prepare):
            with patch.object(
                env.asset_manager,
                "get_image_array",
                return_value=np.zeros((512, 1024, 3), dtype=np.uint8),
            ):
                obs, info = env.reset()

                # Should have tried 3 times with different coordinates
                assert call_count == 3
                assert len(attempted_coordinates) == 3

                # Each attempt should have different coordinates (due to geofence sampling)
                coords_set = set(attempted_coordinates)
                assert len(coords_set) == 3, "Should have sampled different coordinates"

                # Should succeed with the third panorama
                assert info["pano_id"] == "success_pano"

        env.close()

    def test_reset_fails_after_max_attempts(self):
        """Test that reset raises error after maximum retry attempts."""
        config = {
            "cache_root": tempfile.mkdtemp(),
            "geofence": {
                "type": "circle",
                "center": {"lat": 47.620908, "lon": -122.353508},
                "radius_km": 10.0,
            },
            "max_steps": 5,
            "seed": 42,
        }

        env = GeoGuessrEnv(config=config)

        # Mock to always fail
        def mock_prepare(root_lat, root_lon):
            raise RootPanoramaUnavailableError(
                f"Always fails at ({root_lat}, {root_lon})"
            )

        with patch.object(env.asset_manager, "prepare_graph", side_effect=mock_prepare):
            with pytest.raises(ValueError) as exc_info:
                env.reset()

            assert "Failed to find valid panorama location after 5 attempts" in str(
                exc_info.value
            )

        env.close()

    def test_reset_without_geofence_fails_immediately(self):
        """Test that reset without geofence fails immediately on root panorama unavailable."""
        config = {
            "cache_root": tempfile.mkdtemp(),
            "input_lat": 47.620908,
            "input_lon": -122.353508,
            "max_steps": 5,
            "seed": 42,
        }

        env = GeoGuessrEnv(config=config)

        # Mock to fail on first attempt
        call_count = 0

        def mock_prepare(root_lat, root_lon):
            nonlocal call_count
            call_count += 1
            raise RootPanoramaUnavailableError(f"Fails at ({root_lat}, {root_lon})")

        with patch.object(env.asset_manager, "prepare_graph", side_effect=mock_prepare):
            with pytest.raises(ValueError) as exc_info:
                env.reset()

            # Should only try once (no retry without geofence)
            assert call_count == 1
            assert (
                "Root panorama unavailable and no geofence configured for retry"
                in str(exc_info.value)
            )

        env.close()

    def test_sample_valid_coordinates_checks_blocklist(self):
        """Test that _sample_valid_coordinates_from_geofence respects blocklist."""
        config = {
            "cache_root": tempfile.mkdtemp(),
            "geofence": {
                "type": "circle",
                "center": {"lat": 47.620908, "lon": -122.353508},
                "radius_km": 10.0,
            },
            "max_steps": 5,
            "seed": 42,
        }

        env = GeoGuessrEnv(config=config)

        # Mock resolve_nearest_panorama to return different IDs
        panorama_sequence = [
            None,  # First coordinate has no panorama
            "good_pano",  # Second has good panorama (blocklisted ones return None)
        ]
        call_index = 0

        def mock_resolve(lat, lon):
            nonlocal call_index
            if call_index < len(panorama_sequence):
                result = panorama_sequence[call_index]
                call_index += 1
                return result
            return None

        with patch.object(
            env.asset_manager, "resolve_nearest_panorama", side_effect=mock_resolve
        ):
            lat, lon = env._sample_valid_coordinates_from_geofence()

            # Should have tried twice (first None, then good_pano)
            assert call_index == 2

        env.close()


class TestDataLoadingFlow:
    """Test the complete data loading flow with blocklist integration."""

    def test_download_failure_propagates_through_flow(self):
        """Test that download failure is properly handled through the entire flow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Set up provider that will fail image download
            provider = MagicMock(spec=GoogleStreetViewProvider)
            provider.provider_name = "test_provider"
            provider.find_nearest_panorama.return_value = "fail_download_pano"

            metadata = MagicMock(
                pano_id="fail_download_pano",
                lat=47.0,
                lon=-122.0,
                heading=0.0,
                pitch=None,
                roll=None,
                date="2023-01",
                elevation=None,
                links=[],  # No links so only root is fetched
            )
            provider.get_panorama_metadata.return_value = metadata
            provider.download_panorama_image.return_value = False  # Simulate failure

            manager = AssetManager(provider=provider, cache_root=tmpdir)

            # Attempt to prepare graph - will fail with ValueError since fetch returns empty dict
            with pytest.raises(ValueError) as exc_info:
                manager.prepare_graph(47.0, -122.0)

            # Verify the panorama was blocklisted
            assert manager.is_blocklisted("fail_download_pano")

            # Verify error message
            assert "No panorama graph available" in str(exc_info.value)

    def test_blocklisted_panorama_in_graph_excluded(self):
        """Test that blocklisted panoramas in graph are excluded from results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = MagicMock(spec=GoogleStreetViewProvider)
            provider.provider_name = "test_provider"
            provider.find_nearest_panorama.return_value = "root_pano"

            # Create metadata for root and linked panoramas
            root_metadata = MagicMock(
                pano_id="root_pano",
                lat=47.0,
                lon=-122.0,
                heading=0.0,
                links=[
                    {"id": "good_link", "direction": 0},
                    {"id": "bad_link", "direction": 1.57},
                ],
            )
            good_metadata = MagicMock(
                pano_id="good_link",
                lat=47.001,
                lon=-122.001,
                heading=0.0,
                links=[{"id": "root_pano", "direction": 3.14}],
            )
            bad_metadata = MagicMock(
                pano_id="bad_link",
                lat=47.002,
                lon=-122.002,
                heading=0.0,
                links=[{"id": "root_pano", "direction": 3.14}],
            )

            def mock_get_metadata(pano_id):
                return {
                    "root_pano": root_metadata,
                    "good_link": good_metadata,
                    "bad_link": bad_metadata,
                }.get(pano_id)

            provider.get_panorama_metadata.side_effect = mock_get_metadata

            # Root and good_link download successfully, bad_link fails
            def mock_download(pano_id, path):
                if pano_id == "bad_link":
                    return False
                # Create dummy file for successful downloads
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
                return True

            provider.download_panorama_image.side_effect = mock_download
            provider.compute_image_hash.return_value = "dummy_hash"

            manager = AssetManager(
                provider=provider, cache_root=tmpdir, max_connected_panoramas=3
            )

            # Prepare graph
            result = manager.prepare_graph(47.0, -122.0)

            # bad_link should not be in the final graph
            assert "root_pano" in result.graph
            assert "good_link" in result.graph
            assert "bad_link" not in result.graph

            # bad_link should be blocklisted
            assert manager.is_blocklisted("bad_link")

    def test_cached_panorama_becomes_blocklisted(self):
        """Test handling when a cached panorama becomes blocklisted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "metadata"
            cache_dir.mkdir(parents=True, exist_ok=True)

            # Create nearest_pano_cache with an entry
            cache_file = cache_dir / "nearest_pano_cache.json"
            initial_cache = {"47.5,-122.3": "cached_pano"}
            with open(cache_file, "w") as f:
                json.dump(initial_cache, f)

            provider = MagicMock(spec=GoogleStreetViewProvider)
            provider.find_nearest_panorama.return_value = "new_pano"

            manager = AssetManager(provider=provider, cache_root=tmpdir)

            # Add the cached panorama to blocklist
            manager._add_to_blocklist("cached_pano")

            # Request the same coordinates
            result = manager._get_or_find_nearest_panorama(47.5, -122.3)

            # Should get new panorama since cached one is blocklisted
            assert result == "new_pano"

            # Cache should be updated
            with open(cache_file, "r") as f:
                updated_cache = json.load(f)
                assert updated_cache["47.5,-122.3"] == "new_pano"

    def test_blocklist_prevents_repeated_download_attempts(self):
        """Test that blocklist prevents repeated download attempts for known-bad panoramas."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = MagicMock(spec=GoogleStreetViewProvider)
            provider.provider_name = "test_provider"

            # Track download attempts
            download_attempts = []

            def mock_download(pano_id, path):
                download_attempts.append(pano_id)
                return False  # Always fail

            provider.download_panorama_image.side_effect = mock_download
            provider.get_panorama_metadata.return_value = MagicMock(
                pano_id="bad_pano",
                lat=47.0,
                lon=-122.0,
                heading=0.0,
                links=[],
            )

            manager = AssetManager(provider=provider, cache_root=tmpdir)

            # First attempt - should try to download
            result1 = manager._fetch_and_cache_asset("bad_pano")
            assert result1 is False
            assert len(download_attempts) == 1

            # Second attempt - should skip due to blocklist
            result2 = manager._fetch_and_cache_asset("bad_pano")
            assert result2 is False
            assert len(download_attempts) == 1  # No additional attempt

            # Verify it's blocklisted
            assert manager.is_blocklisted("bad_pano")

    def test_integration_full_retry_flow(self):
        """Integration test of the complete retry flow with real components."""
        config = {
            "cache_root": tempfile.mkdtemp(),
            "geofence": {
                "type": "circle",
                "center": {"lat": 47.620908, "lon": -122.353508},
                "radius_km": 5.0,
            },
            "max_steps": 5,
            "seed": 123,
        }

        env = GeoGuessrEnv(config=config)

        # Track the full flow
        prepare_calls = []
        resolve_calls = []

        def track_prepare(root_lat, root_lon):
            prepare_calls.append((root_lat, root_lon))
            if len(prepare_calls) == 1:
                # First call fails
                raise RootPanoramaUnavailableError("First attempt fails")
            # Second call succeeds
            return Mock(
                root_id="success_pano",
                graph={
                    "success_pano": {
                        "lat": root_lat,
                        "lon": root_lon,
                        "heading": 0.0,
                        "links": [],
                    }
                },
                missing_assets=set(),
            )

        def track_resolve(lat, lon):
            resolve_calls.append((lat, lon))
            return f"pano_at_{lat}_{lon}"

        with patch.object(
            env.asset_manager, "prepare_graph", side_effect=track_prepare
        ):
            with patch.object(
                env.asset_manager, "resolve_nearest_panorama", side_effect=track_resolve
            ):
                with patch.object(
                    env.asset_manager,
                    "get_image_array",
                    return_value=np.zeros((512, 1024, 3), dtype=np.uint8),
                ):
                    obs, info = env.reset()

                    # Should have sampled valid coordinates multiple times
                    assert len(resolve_calls) > 0

                    # Should have tried prepare_graph twice
                    assert len(prepare_calls) == 2

                    # Different coordinates for each attempt
                    assert prepare_calls[0] != prepare_calls[1]

                    # Should succeed with second attempt
                    assert info["pano_id"] == "success_pano"

        env.close()
