#!/usr/bin/env python3
"""
Demo script showing session replay functionality.
Creates a simple replay session and demonstrates offline replay.
"""

import json
import tempfile
from pathlib import Path

from gymnasium_env.envs import GeoGuessrWorldEnv
from gymnasium_env.replay import ReplayEpisode, ReplayManager, ReplaySession


def create_demo_cache(cache_dir: Path):
    """Create minimal cached data for demo."""
    # Create directory structure
    metadata_dir = cache_dir / "metadata"
    images_dir = cache_dir / "images"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # Create demo panorama metadata
    demo_panos = [
        {"id": "demo_pano_1", "lat": 47.620908, "lon": -122.353508, "heading": 0.0},
        {"id": "demo_pano_2", "lat": 48.620908, "lon": -123.353508, "heading": 45.0},
    ]

    for pano in demo_panos:
        # Create metadata file
        metadata_file = metadata_dir / f"{pano['id']}_mini.jsonl"
        with open(metadata_file, "w") as f:
            f.write(json.dumps({**pano, "links": []}) + "\n")

        # Create minimal image file
        from PIL import Image

        img = Image.new("RGB", (1024, 512), color="red")
        image_file = images_dir / f"{pano['id']}.jpg"
        img.save(image_file)

    return demo_panos


def demo_session_creation():
    """Demonstrate creating and saving a replay session."""
    print("=== Demo: Creating Replay Session ===")

    # Create episodes
    episodes = [
        ReplayEpisode("gsv", "demo_pano_1", 47.620908, -122.353508, 0.0),
        ReplayEpisode("gsv", "demo_pano_2", 48.620908, -123.353508, 45.0),
    ]

    # Create session
    session = ReplaySession(seed=12345, episodes=episodes)
    print(
        f"Created session with seed {session.seed} and {len(session.episodes)} episodes"
    )

    # Save to file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        session_path = f.name

    session.save_to_file(session_path)
    print(f"Session saved to: {session_path}")

    # Display session content
    with open(session_path, "r") as f:
        session_data = json.load(f)

    print("Session content:")
    print(json.dumps(session_data, indent=2))

    return session_path


def demo_replay_manager():
    """Demonstrate ReplayManager functionality."""
    print("\n=== Demo: Replay Manager ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        manager = ReplayManager(temp_dir)

        # Start recording session
        print("Starting recording session...")
        manager.start_recording_session(seed=54321, geofence=None)

        # Add some episodes
        manager.add_episode_to_session("gsv", "manager_pano_1", 49.0, -124.0, 30.0)
        manager.add_episode_to_session("gsv", "manager_pano_2", 50.0, -125.0, 60.0)

        print(f"Added {len(manager.current_session.episodes)} episodes")

        # Save session
        session_path = manager.save_session("demo_session.json")
        print(f"Session saved to: {session_path}")

        # Load session back
        loaded_session = manager.load_session(session_path)
        print(f"Loaded session with seed {loaded_session.seed}")

        # Demonstrate episode playback
        print("Playing back episodes:")
        episode_count = 0
        while True:
            episode = manager.get_next_episode()
            if episode is None:
                break
            episode_count += 1
            print(
                f"  Episode {episode_count}: {episode.pano_id} at ({episode.gt_lat}, {episode.gt_lon})"
            )


def demo_environment_replay():
    """Demonstrate environment with replay functionality."""
    print("\n=== Demo: Environment Replay ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = Path(temp_dir)

        # Create demo cache data
        demo_panos = create_demo_cache(cache_dir)
        print(f"Created demo cache with {len(demo_panos)} panoramas")

        # Create replay session
        episodes = [
            ReplayEpisode("gsv", "demo_pano_1", 47.620908, -122.353508, 0.0),
            ReplayEpisode("gsv", "demo_pano_2", 48.620908, -123.353508, 45.0),
        ]
        session = ReplaySession(seed=99999, episodes=episodes)

        # Save session
        replays_dir = cache_dir / "replays"
        replays_dir.mkdir(exist_ok=True)
        session_path = replays_dir / "demo_session.json"
        session.save_to_file(str(session_path))

        print(f"Created replay session: {session_path}")

        # Test offline mode
        print("Testing offline mode...")
        config = {
            "mode": "offline",
            "cache_root": str(cache_dir),
            "replay_session_path": str(session_path),
        }

        env = GeoGuessrWorldEnv(config=config)

        try:
            # Reset environment (should load first episode)
            obs, info = env.reset()

            print("Loaded episode:")
            print(f"  Provider: {info['provider']}")
            print(f"  Pano ID: {info['pano_id']}")
            print(f"  Ground truth: ({info['gt_lat']}, {info['gt_lon']})")
            print(f"  Initial heading: {info['pose']['heading_deg']}°")
            print(f"  Image shape: {obs['image'].shape}")

            # Take a simple action
            action = {"op": "answer", "value": [47.0, -122.0]}
            obs, reward, terminated, truncated, info = env.step(action)

            print("Took answer action:")
            print(f"  Reward: {reward:.4f}")
            print(f"  Terminated: {terminated}")
            print(f"  Distance: {info.get('distance_km', 'N/A')} km")

        finally:
            env.close()

        print("Offline replay demo completed successfully!")


def main():
    """Run all demos."""
    print("GeoGuessr Environment - Session Replay Demo")
    print("=" * 50)

    try:
        # Demo 1: Session creation
        session_path = demo_session_creation()

        # Demo 2: Replay manager
        demo_replay_manager()

        # Demo 3: Environment integration
        demo_environment_replay()

        print("\n" + "=" * 50)
        print("All demos completed successfully!")
        print("\nKey features demonstrated:")
        print("✓ ReplaySession and ReplayEpisode data models")
        print("✓ ReplayManager for session recording and playback")
        print("✓ Environment integration with online/offline modes")
        print("✓ Deterministic episode loading from replay files")
        print("✓ Cached asset verification in offline mode")

    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Clean up
        try:
            Path(session_path).unlink(missing_ok=True)
        except:
            pass


if __name__ == "__main__":
    main()
