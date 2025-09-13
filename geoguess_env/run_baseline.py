"""
CLI runner for baseline agent with online/offline modes.
Implements the command-line interface specified in TaskDescription.md.
"""

import argparse
import csv
import json
import os
from typing import Dict, List, Optional

import gymnasium as gym
from gymnasium.envs.registration import register

from .baseline_agent import BaselineAgent

ENV_ID = "GeoGuessr-v0"


def ensure_registered() -> None:
    """Ensure the environment is registered."""
    try:
        register(
            id=ENV_ID,
            entry_point="geoguess_env.geoguessr_env:GeoGuessrEnv",
        )
    except Exception:
        # Already registered
        pass


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run baseline agent in GeoGuessr environment"
    )

    # Mode and provider
    parser.add_argument(
        "--mode",
        choices=["online", "offline"],
        required=True,
        help="Run mode: online (sample & cache) or offline (replay)",
    )
    parser.add_argument(
        "--provider", default="gsv", help="Provider for online mode (default: gsv)"
    )

    # Episodes and sampling
    parser.add_argument(
        "--episodes",
        type=int,
        default=30,
        help="Number of episodes to run (default: 30)",
    )
    parser.add_argument("--geofence", help="Geofence JSON file for sampling locations")
    parser.add_argument(
        "--seed", type=int, default=123, help="Random seed (default: 123)"
    )

    # Caching
    parser.add_argument(
        "--cache", default="./cache", help="Cache directory (default: ./cache)"
    )
    parser.add_argument(
        "--freeze-run", help="Path to save session replay JSON (online mode)"
    )

    # Offline replay
    parser.add_argument("--replay", help="Path to replay session JSON (offline mode)")

    # Output
    parser.add_argument("--out", required=True, help="Output CSV file path")

    # Agent configuration
    parser.add_argument(
        "--max-nav-steps",
        type=int,
        default=20,
        help="Maximum navigation steps per episode (default: 20)",
    )

    return parser.parse_args()


def load_geofence(geofence_path: Optional[str]) -> Optional[Dict]:
    """Load geofence configuration from JSON file."""
    if not geofence_path:
        return None

    try:
        with open(geofence_path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load geofence from {geofence_path}: {e}")
        return None


def load_replay_session(replay_path: str) -> Dict:
    """Load replay session from JSON file."""
    with open(replay_path, "r") as f:
        return json.load(f)


def save_replay_session(episodes_data: List[Dict], seed: int, output_path: str) -> None:
    """Save episode data as a replay session."""
    replay_data = {"seed": seed, "episodes": episodes_data}

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(replay_data, f, indent=2)


def run_online_episodes(args) -> List[Dict]:
    """
    Run episodes in online mode, sampling and caching as we go.

    Returns:
        List of episode results for potential replay saving
    """
    ensure_registered()

    # Load geofence
    geofence = load_geofence(args.geofence)

    # For online mode, we need to sample starting coordinates
    # For now, use a default location (Seattle) if no geofence is provided
    default_lat, default_lon = 47.620908, -122.353508

    # Create environment config
    env_config = {
        "provider": args.provider,
        "mode": "online",
        "geofence": geofence,
        "input_lat": default_lat,
        "input_lon": default_lon,
        "cache_root": args.cache,
        "seed": args.seed,
        "max_steps": 40,  # Environment max steps
    }

    env = gym.make(ENV_ID, config=env_config)
    agent = BaselineAgent(max_nav_steps=args.max_nav_steps, seed=args.seed)

    results = []
    episodes_data = []  # For replay saving

    print(f"Running {args.episodes} episodes in online mode...")

    for episode_idx in range(args.episodes):
        print(f"Episode {episode_idx + 1}/{args.episodes}")

        # Reset environment and agent
        observation, info = env.reset()
        agent.reset()

        # Store episode start data for replay
        episode_data = {
            "provider": args.provider,
            "pano_id": info.get("pano_id"),
            "gt_lat": info.get("gt_lat"),
            "gt_lon": info.get("gt_lon"),
            "initial_yaw_deg": info.get("pose", {}).get("heading_deg", 0.0),
        }

        # Run episode
        done = False
        while not done:
            action = agent.act(observation, info)
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # Store results
        result = {
            "episode": episode_idx,
            "pano_id": episode_data["pano_id"],
            "gt_lat": episode_data["gt_lat"],
            "gt_lon": episode_data["gt_lon"],
            "guess_lat": info.get("guess_lat"),
            "guess_lon": info.get("guess_lon"),
            "distance_km": info.get("distance_km"),
            "score": info.get("score", 0.0),
            "steps": info.get("steps", 0),
        }
        results.append(result)
        episodes_data.append(episode_data)

    env.close()

    # Save replay session if requested
    if args.freeze_run:
        save_replay_session(episodes_data, args.seed, args.freeze_run)
        print(f"Saved replay session to {args.freeze_run}")

    return results


def run_offline_episodes(args) -> List[Dict]:
    """
    Run episodes in offline mode from replay session.

    Returns:
        List of episode results
    """
    ensure_registered()

    # Load replay session
    if not args.replay:
        raise ValueError("--replay is required for offline mode")

    replay_session = load_replay_session(args.replay)
    episodes_data = replay_session.get("episodes", [])

    print(f"Running {len(episodes_data)} episodes in offline mode from {args.replay}")

    # Create environment config for offline mode
    env_config = {
        "mode": "offline",
        "cache_root": args.cache,
        "max_steps": 40,
    }

    env = gym.make(ENV_ID, config=env_config)
    agent = BaselineAgent(
        max_nav_steps=args.max_nav_steps, seed=replay_session.get("seed", 123)
    )

    results = []

    for episode_idx, episode_data in enumerate(episodes_data):
        print(f"Episode {episode_idx + 1}/{len(episodes_data)}")

        # Set up environment for this specific episode from replay data
        # Update config with episode-specific data
        episode_config = env_config.copy()
        episode_config.update(
            {
                "input_lat": episode_data.get("gt_lat"),
                "input_lon": episode_data.get("gt_lon"),
                "provider": episode_data.get("provider"),
            }
        )

        # Create new environment instance for this episode with specific config
        env_episode = gym.make(ENV_ID, config=episode_config)
        observation, info = env_episode.reset()
        agent.reset()

        # Run episode
        done = False
        while not done:
            action = agent.act(observation, info)
            observation, reward, terminated, truncated, info = env_episode.step(action)
            done = terminated or truncated

        # Store results
        result = {
            "episode": episode_idx,
            "pano_id": episode_data.get("pano_id", info.get("pano_id")),
            "gt_lat": episode_data.get("gt_lat", info.get("gt_lat")),
            "gt_lon": episode_data.get("gt_lon", info.get("gt_lon")),
            "guess_lat": info.get("guess_lat"),
            "guess_lon": info.get("guess_lon"),
            "distance_km": info.get("distance_km"),
            "score": info.get("score", 0.0),
            "steps": info.get("steps", 0),
        }
        results.append(result)

        # Close episode environment
        env_episode.close()

    # Original env is no longer used
    env.close()
    return results


def save_results_csv(results: List[Dict], output_path: str) -> None:
    """Save results to CSV file."""
    if not results:
        print("No results to save")
        return

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Only create directory if there is one
        os.makedirs(output_dir, exist_ok=True)

    fieldnames = [
        "episode",
        "pano_id",
        "gt_lat",
        "gt_lon",
        "guess_lat",
        "guess_lon",
        "distance_km",
        "score",
        "steps",
    ]

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to {output_path}")


def main() -> None:
    """Main entry point."""
    args = parse_args()

    try:
        if args.mode == "online":
            results = run_online_episodes(args)
        elif args.mode == "offline":
            results = run_offline_episodes(args)
        else:
            raise ValueError(f"Unknown mode: {args.mode}")

        # Save results to CSV
        save_results_csv(results, args.out)

        # Print summary statistics
        if results:
            avg_score = sum(r.get("score", 0) for r in results) / len(results)
            avg_distance = sum(r.get("distance_km", 0) for r in results) / len(results)
            avg_steps = sum(r.get("steps", 0) for r in results) / len(results)

            print(f"\nSummary for {len(results)} episodes:")
            print(f"Average score: {avg_score:.4f}")
            print(f"Average distance: {avg_distance:.2f} km")
            print(f"Average steps: {avg_steps:.1f}")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
