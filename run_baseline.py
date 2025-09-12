#!/usr/bin/env python3
"""
Baseline runner for GeoGuessr environment with session replay support.
Implements the CLI interface specified in the task description.
"""

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional

from gymnasium_env.envs import GeoGuessrWorldEnv


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run baseline agent on GeoGuessr environment"
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["online", "offline"],
        required=True,
        help="Run mode: online (sample & cache) or offline (replay)",
    )

    # Online mode arguments
    parser.add_argument(
        "--provider", default="gsv", help="Panorama provider (default: gsv)"
    )
    parser.add_argument(
        "--episodes", type=int, default=1, help="Number of episodes to run (default: 1)"
    )
    parser.add_argument("--geofence", type=str, help="Path to geofence JSON file")
    parser.add_argument(
        "--seed", type=int, default=123, help="Random seed (default: 123)"
    )
    parser.add_argument(
        "--freeze-run", type=str, help="Path to save replay session for offline use"
    )

    # Offline mode arguments
    parser.add_argument(
        "--replay",
        type=str,
        help="Path to replay session file (required for offline mode)",
    )

    # Common arguments
    parser.add_argument("--cache", type=str, required=True, help="Cache directory path")
    parser.add_argument("--out", type=str, required=True, help="Output CSV file path")

    # Environment parameters
    parser.add_argument(
        "--max-steps",
        type=int,
        default=40,
        help="Maximum steps per episode (default: 40)",
    )

    return parser.parse_args()


class SimpleBaselineAgent:
    """
    Simple baseline agent that follows arrows by clicking and then answers with a heuristic.

    Strategy:
    1. Click at different screen positions to explore (sweep headings)
    2. Stop if we detect looping (same pano_id seen before)
    3. Answer with a simple global heuristic (continent centroids)
    """

    def __init__(self, max_nav_steps: int = 10):
        self.max_nav_steps = max_nav_steps
        self.visited_panos = set()
        self.step_count = 0

        # Simple continent centroids for heuristic guessing
        self.continent_centroids = [
            (39.8283, -98.5795),  # North America (USA center)
            (54.5260, 15.2551),  # Europe
            (-25.2744, 133.7751),  # Australia
            (20.5937, 78.9629),  # Asia (India)
            (-8.7832, -55.4915),  # South America (Brazil)
        ]

    def reset(self):
        """Reset agent state for new episode."""
        self.visited_panos.clear()
        self.step_count = 0

    def act(self, observation: Dict, info: Dict) -> Dict:
        """
        Decide on next action based on observation and info.

        Args:
            observation: Environment observation with 'image' key
            info: Environment info with pano_id, links, etc.

        Returns:
            Action dictionary with 'op' and 'value' keys
        """
        self.step_count += 1
        current_pano = info.get("pano_id")
        links = info.get("links", [])

        # Check if we've been here before (loop detection)
        if current_pano in self.visited_panos:
            return self._make_answer()

        self.visited_panos.add(current_pano)

        # If we've explored enough or no links available, make an answer
        if self.step_count >= self.max_nav_steps or not links:
            return self._make_answer()

        # Try to click on a link to navigate
        # Simple strategy: click on the first available link position
        if links:
            link = links[0]
            screen_xy = link.get("screen_xy", [512, 256])  # Fallback to center
            return {"op": "click", "value": screen_xy}

        # Fallback: make an answer
        return self._make_answer()

    def _make_answer(self) -> Dict:
        """Make an answer using simple heuristic."""
        # Simple heuristic: randomly pick a continent centroid
        import random

        lat, lon = random.choice(self.continent_centroids)
        return {"op": "answer", "value": [lat, lon]}


def load_geofence(geofence_path: str) -> Optional[Dict]:
    """Load geofence configuration from JSON file."""
    if not geofence_path or not os.path.exists(geofence_path):
        return None

    try:
        with open(geofence_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load geofence from {geofence_path}: {e}")
        return None


def run_episodes(
    env: GeoGuessrWorldEnv, agent: SimpleBaselineAgent, num_episodes: int
) -> List[Dict]:
    """Run multiple episodes and collect results."""
    results = []

    for episode_idx in range(num_episodes):
        print(f"Running episode {episode_idx + 1}/{num_episodes}...")

        agent.reset()
        obs, info = env.reset()

        episode_result = {
            "episode": episode_idx,
            "pano_id": info.get("pano_id"),
            "gt_lat": info.get("gt_lat"),
            "gt_lon": info.get("gt_lon"),
            "guess_lat": None,
            "guess_lon": None,
            "distance_km": None,
            "score": None,
            "steps": 0,
        }

        # Run episode until termination
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = agent.act(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)

            episode_result["steps"] = info.get("steps", 0)

            # If episode ended with an answer, record the results
            if terminated and "guess_lat" in info:
                episode_result.update(
                    {
                        "guess_lat": info["guess_lat"],
                        "guess_lon": info["guess_lon"],
                        "distance_km": info.get("distance_km"),
                        "score": info.get("score"),
                    }
                )

        results.append(episode_result)
        print(
            f"Episode {episode_idx + 1} completed: score={episode_result.get('score', 0):.4f}"
        )

    return results


def save_results_csv(results: List[Dict], output_path: str):
    """Save results to CSV file."""
    if not results:
        print("No results to save.")
        return

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

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


def main():
    """Main entry point."""
    args = parse_args()

    # Validate arguments
    if args.mode == "offline" and not args.replay:
        print("Error: --replay is required for offline mode")
        sys.exit(1)

    # Load geofence if provided
    geofence = load_geofence(args.geofence) if args.geofence else None

    # Create environment configuration
    env_config = {
        "mode": args.mode,
        "provider": args.provider,
        "cache_root": args.cache,
        "max_steps": args.max_steps,
        "geofence": geofence,
        "seed": args.seed,
    }

    # Add mode-specific configuration
    if args.mode == "offline":
        env_config["replay_session_path"] = args.replay
    elif args.mode == "online" and args.freeze_run:
        env_config["freeze_run_path"] = args.freeze_run

    # For online mode without geofence, we need specific coordinates
    # This is a limitation of the current implementation
    if args.mode == "online" and not geofence:
        print(
            "Warning: No geofence provided for online mode, using default coordinates"
        )
        env_config.update(
            {
                "input_lat": 47.620908,  # Seattle as default
                "input_lon": -122.353508,
            }
        )

    print(f"Starting {args.mode} mode with {args.episodes} episodes...")
    print(f"Cache directory: {args.cache}")
    print(f"Output file: {args.out}")

    # Create environment and agent
    env = GeoGuessrWorldEnv(config=env_config)
    agent = SimpleBaselineAgent(max_nav_steps=min(10, args.max_steps - 5))

    try:
        # Run episodes
        results = run_episodes(env, agent, args.episodes)

        # Save results
        save_results_csv(results, args.out)

        # Print summary
        if results:
            scores = [r.get("score", 0) for r in results if r.get("score") is not None]
            if scores:
                avg_score = sum(scores) / len(scores)
                print("\nSummary:")
                print(f"Episodes completed: {len(results)}")
                print(f"Average score: {avg_score:.4f}")
                print(f"Best score: {max(scores):.4f}")
                print(f"Worst score: {min(scores):.4f}")

    finally:
        env.close()


if __name__ == "__main__":
    main()
