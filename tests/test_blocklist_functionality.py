"""Tests for panorama blocklist and retry functionality."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from geoguess_env.asset_manager import AssetManager
from geoguess_env.providers.google_streetview import GoogleStreetViewProvider


class TestBlocklistFunctionality:
    """Test suite for blocklist functionality in AssetManager."""

    def test_blocklist_initialization(self):
        """Test that blocklist initializes correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = MagicMock(spec=GoogleStreetViewProvider)
            manager = AssetManager(provider=provider, cache_root=Path(tmpdir))

            # Should start with empty blocklist
            assert len(manager._failed_panoramas) == 0
            assert not manager.is_blocklisted("test_pano_id")

    def test_add_to_blocklist(self):
        """Test adding panorama IDs to blocklist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = MagicMock(spec=GoogleStreetViewProvider)
            manager = AssetManager(provider=provider, cache_root=Path(tmpdir))

            # Add a panorama to blocklist
            manager._add_to_blocklist("bad_pano_1")
            assert manager.is_blocklisted("bad_pano_1")
            assert not manager.is_blocklisted("good_pano")

            # Verify it's persisted to disk
            blocklist_file = Path(tmpdir) / "metadata" / "failed_panoramas.json"
            assert blocklist_file.exists()

            with open(blocklist_file, "r") as f:
                data = json.load(f)
                assert "bad_pano_1" in data["failed_ids"]

    def test_blocklist_persistence(self):
        """Test that blocklist persists across AssetManager instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = MagicMock(spec=GoogleStreetViewProvider)

            # First instance - add to blocklist
            manager1 = AssetManager(provider=provider, cache_root=Path(tmpdir))
            manager1._add_to_blocklist("persistent_bad_pano")
            assert manager1.is_blocklisted("persistent_bad_pano")

            # Second instance - should load blocklist from disk
            manager2 = AssetManager(provider=provider, cache_root=Path(tmpdir))
            assert manager2.is_blocklisted("persistent_bad_pano")

    def test_blocklisted_panorama_skipped_in_fetch(self):
        """Test that blocklisted panoramas are skipped during fetch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = MagicMock(spec=GoogleStreetViewProvider)
            manager = AssetManager(provider=provider, cache_root=Path(tmpdir))

            # Add to blocklist
            manager._add_to_blocklist("blocked_pano")

            # Try to fetch - should return False immediately
            result = manager._fetch_and_cache_asset("blocked_pano")
            assert result is False

            # Provider should not have been called
            provider.get_panorama_metadata.assert_not_called()
            provider.download_panorama_image.assert_not_called()

    def test_failed_download_adds_to_blocklist(self):
        """Test that failed downloads are automatically added to blocklist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = MagicMock(spec=GoogleStreetViewProvider)
            provider.provider_name = "test_provider"

            # Mock successful metadata fetch but failed image download
            provider.get_panorama_metadata.return_value = MagicMock(
                pano_id="failing_pano", lat=47.0, lon=-122.0, heading=0.0, links=[]
            )
            provider.download_panorama_image.return_value = False

            manager = AssetManager(provider=provider, cache_root=Path(tmpdir))

            # Attempt to fetch asset
            result = manager._fetch_and_cache_asset("failing_pano")
            assert result is False

            # Should now be blocklisted
            assert manager.is_blocklisted("failing_pano")

            # Verify persisted
            blocklist_file = Path(tmpdir) / "metadata" / "failed_panoramas.json"
            with open(blocklist_file, "r") as f:
                data = json.load(f)
                assert "failing_pano" in data["failed_ids"]

    def test_root_panorama_unavailable_error_on_blocklisted(self):
        """Test that prepare_graph returns None for blocklisted root panorama."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = MagicMock(spec=GoogleStreetViewProvider)
            provider.find_nearest_panorama.return_value = "blocklisted_root"

            manager = AssetManager(provider=provider, cache_root=Path(tmpdir))
            manager._add_to_blocklist("blocklisted_root")

            # Should raise ValueError since no panorama can be found (blocklisted one is skipped)
            with pytest.raises(ValueError) as exc_info:
                manager.prepare_graph(47.0, -122.0)

            assert "No panorama found for" in str(exc_info.value)

    def test_cached_blocklisted_panorama_removed_from_cache(self):
        """Test that blocklisted panoramas are removed from nearest_pano_cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "metadata"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / "nearest_pano_cache.json"

            # Create initial cache with a panorama
            initial_cache = {"47.0,-122.0": "soon_to_be_blocked"}
            with open(cache_file, "w") as f:
                json.dump(initial_cache, f)

            # Create blocklist with that panorama
            blocklist_file = cache_dir / "failed_panoramas.json"
            with open(blocklist_file, "w") as f:
                json.dump({"failed_ids": ["soon_to_be_blocked"]}, f)

            provider = MagicMock(spec=GoogleStreetViewProvider)
            provider.find_nearest_panorama.return_value = "new_good_pano"

            manager = AssetManager(provider=provider, cache_root=Path(tmpdir))

            # Request the same coordinates
            result = manager._get_or_find_nearest_panorama(47.0, -122.0)

            # Should get new panorama, not the blocklisted one
            assert result == "new_good_pano"

            # Cache should be updated
            with open(cache_file, "r") as f:
                updated_cache = json.load(f)
                assert updated_cache.get("47.0,-122.0") == "new_good_pano"

    def test_find_nearest_skips_blocklisted(self):
        """Test that _get_or_find_nearest_panorama skips blocklisted panoramas."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = MagicMock(spec=GoogleStreetViewProvider)

            # First call returns blocklisted, second returns good
            provider.find_nearest_panorama.return_value = "blocklisted_pano"

            manager = AssetManager(provider=provider, cache_root=Path(tmpdir))
            manager._add_to_blocklist("blocklisted_pano")

            # Should return None since the found panorama is blocklisted
            result = manager._get_or_find_nearest_panorama(47.0, -122.0)
            assert result is None

    def test_multiple_panoramas_in_blocklist(self):
        """Test handling multiple panoramas in blocklist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = MagicMock(spec=GoogleStreetViewProvider)
            manager = AssetManager(provider=provider, cache_root=Path(tmpdir))

            # Add multiple panoramas
            bad_ids = ["bad1", "bad2", "bad3"]
            for pano_id in bad_ids:
                manager._add_to_blocklist(pano_id)

            # All should be blocklisted
            for pano_id in bad_ids:
                assert manager.is_blocklisted(pano_id)

            # Good panoramas should not be blocklisted
            assert not manager.is_blocklisted("good_pano")

            # Verify persistence
            blocklist_file = Path(tmpdir) / "metadata" / "failed_panoramas.json"
            with open(blocklist_file, "r") as f:
                data = json.load(f)
                assert set(data["failed_ids"]) == set(bad_ids)

    def test_corrupt_blocklist_file_handled_gracefully(self):
        """Test that corrupt blocklist file is handled without crashing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "metadata"
            cache_dir.mkdir(parents=True, exist_ok=True)
            blocklist_file = cache_dir / "failed_panoramas.json"

            # Write corrupt JSON
            with open(blocklist_file, "w") as f:
                f.write("{ this is not valid json }")

            provider = MagicMock(spec=GoogleStreetViewProvider)

            # Should not crash, should start with empty blocklist
            manager = AssetManager(provider=provider, cache_root=Path(tmpdir))
            assert len(manager._failed_panoramas) == 0

    def test_prepare_graph_adds_failed_root_to_blocklist(self):
        """Test that prepare_graph adds root panorama to blocklist on download failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = MagicMock(spec=GoogleStreetViewProvider)
            provider.provider_name = "test_provider"
            provider.find_nearest_panorama.return_value = "root_that_fails"

            # Mock metadata exists but image download fails
            metadata = MagicMock(
                pano_id="root_that_fails", lat=47.0, lon=-122.0, heading=0.0, links=[]
            )
            provider.get_panorama_metadata.return_value = metadata
            provider.download_panorama_image.return_value = (
                False  # Simulate download failure
            )

            manager = AssetManager(provider=provider, cache_root=Path(tmpdir))

            # The first attempt will fail during _fetch_and_build_graph
            # But it should try to fetch the asset which will add it to blocklist
            # Since _fetch_and_build_graph returns empty dict when all panoramas fail,
            # we should get ValueError instead of RootPanoramaUnavailableError
            with pytest.raises(ValueError) as exc_info:
                manager.prepare_graph(47.0, -122.0)

            assert "No panorama graph available" in str(exc_info.value)

            # Root should now be blocklisted due to download failure
            assert manager.is_blocklisted("root_that_fails")
