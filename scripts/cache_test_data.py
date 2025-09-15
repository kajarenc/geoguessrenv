#!/usr/bin/env python3
"""
Script to fetch and cache panorama data for test coordinates.

This script runs the environment in online mode to fetch and cache
panorama data for the coordinates used in tests, so that tests can
run in CI without needing internet access.
"""

import sys
from pathlib import Path

# Add the project root to sys.path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from geoguess_env.geoguessr_env import GeoGuessrEnv  # noqa: E402


def cache_test_data():
    """Cache panorama data for test coordinates."""
    # Test coordinates from tests/test_geoguessr_env.py
    test_lat = 47.620908
    test_lon = -122.353508

    print(f"Fetching panorama data for {test_lat}, {test_lon}...")

    # Configure environment for online mode to fetch data
    config = {
        "cache_root": str(repo_root / "cache"),
        "mode": "online",  # Enable online fetching
        "input_lat": test_lat,
        "input_lon": test_lon,
        "max_steps": 5,
        "arrow_hit_radius_px": 24,
        "arrow_min_conf": 0.0,
    }

    try:
        # Create environment and reset to trigger data fetching
        env = GeoGuessrEnv(config=config)
        print("Environment created, starting reset to fetch data...")

        obs, info = env.reset()
        print("Successfully cached panorama data!")
        print(f"  Root panorama ID: {info.get('pano_id')}")
        print(f"  Ground truth: {info.get('gt_lat')}, {info.get('gt_lon')}")
        print(f"  Image shape: {obs['image'].shape}")

        env.close()

        # List cached files
        cache_root = Path(config["cache_root"])
        metadata_files = list(cache_root.glob("metadata/*.jsonl"))
        image_files = list(cache_root.glob("images/*.jpg"))

        print("\nCached files:")
        print(f"  Metadata files: {len(metadata_files)}")
        print(f"  Image files: {len(image_files)}")

        if metadata_files:
            print(f"  Sample metadata: {metadata_files[0].name}")
        if image_files:
            print(f"  Sample image: {image_files[0].name}")

        return True

    except Exception as e:
        print(f"Error caching data: {e}")
        return False


if __name__ == "__main__":
    success = cache_test_data()
    if success:
        print("\n✓ Test data cached successfully!")
        sys.exit(0)
    else:
        print("\n✗ Failed to cache test data")
        sys.exit(1)
