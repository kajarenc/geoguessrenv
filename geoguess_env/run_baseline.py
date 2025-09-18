"""CLI runner for the baseline agent operating in online mode only."""

import argparse
import csv
import json
import logging
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

    parser.add_argument(
        "--provider", default="gsv", help="Provider for sampling panoramas"
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


def run_episodes(args) -> List[Dict]:
    """Run episodes, sampling and caching panoramas as we go."""
    ensure_registered()

    # Load geofence
    geofence = load_geofence(args.geofence)

    # For online mode, we need to sample starting coordinates
    # For now, use a default location (Seattle) if no geofence is provided
    default_lat, default_lon = 47.620908, -122.353508

    # Create base environment config
    base_env_config = {
        "provider": args.provider,
        "geofence": geofence,
        "cache_root": args.cache,
        "seed": args.seed,
        "max_steps": 40,  # Environment max steps
    }

    # If geofence is provided, don't set default coordinates - let the environment sample them
    if not geofence:
        base_env_config["input_lat"] = default_lat
        base_env_config["input_lon"] = default_lon

    agent = BaselineAgent(max_nav_steps=args.max_nav_steps, seed=args.seed)

    results = []

    print(f"Running {args.episodes} episodes...")

    for episode_idx in range(args.episodes):
        print(f"Episode {episode_idx + 1}/{args.episodes}")

        # Create environment for this episode
        episode_config = base_env_config.copy()

        # For each episode, set a unique seed to get different locations from geofence
        if geofence:
            episode_seed = args.seed + episode_idx if args.seed else episode_idx
            episode_config["seed"] = episode_seed

        env = gym.make(ENV_ID, config=episode_config)

        # Reset environment and agent
        observation, info = env.reset()
        agent.reset()

        # Store episode start data for replay
        episode_data = {
            "provider": args.provider,
            "pano_id": info.get("pano_id"),
            "gt_lat": info.get("gt_lat"),
            "gt_lon": info.get("gt_lon"),
            "initial_yaw_deg": info.get("pose", {}).get("yaw_deg", 0.0),
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

        # Close episode environment
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
    logging.basicConfig(level=logging.INFO)
    try:
        results = run_episodes(args)

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
