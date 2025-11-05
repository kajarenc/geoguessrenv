#!/usr/bin/env python3
"""
Demo of the new unified continuous action space.

This demo shows how to use the new unified action format which is
compatible with RL algorithms like PPO, SAC, and DQN.
"""

import logging
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register

ENV_ID = "GeoGuessr-v0"


def ensure_registered() -> None:
    """Register the environment if not already registered."""
    try:
        register(
            id=ENV_ID,
            entry_point="geoguess_env.geoguessr_env:GeoGuessrEnv",
        )
    except Exception:
        pass  # Already registered


def main() -> None:
    """Run the demo."""
    logging.basicConfig(level=logging.INFO)
    ensure_registered()

    # Setup cache directory
    script_path = Path(__file__).resolve()
    cache_root = script_path.parent / "cache"

    # Create environment
    input_lat, input_lon = 51.481610, -0.163400
    env = gym.make(
        ENV_ID,
        render_mode="human",
        config={
            "provider_config": {"provider": "gsv"},
            "cache_root": str(cache_root),
            "input_lat": input_lat,
            "input_lon": input_lon,
            "max_steps": 10,
        },
    )

    print("=" * 80)
    print("UNIFIED ACTION SPACE DEMO")
    print("=" * 80)
    print()

    # Check action space
    print(f"Action Space: {env.action_space}")
    print(f"  Shape: {env.action_space.shape}")
    print(f"  Dtype: {env.action_space.dtype}")
    print(f"  Low: {env.action_space.low}")
    print(f"  High: {env.action_space.high}")
    print()

    # Reset environment
    observation, info = env.reset()
    print(f"Episode started at panorama: {info.get('pano_id')}")
    print(f"Ground truth: lat={info['gt_lat']:.6f}, lon={info['gt_lon']:.6f}")
    print(f"Available links: {len(info.get('links', []))}")
    print()

    print("=" * 80)
    print("ACTION FORMAT EXPLANATION")
    print("=" * 80)
    print()
    print("Unified action: [action_type, param1, param2]")
    print("  - action_type: 0.0-1.0 (< 0.5 = click, >= 0.5 = answer)")
    print("  - param1: 0.0-1.0 (for click: x/width, for answer: (lat+90)/180)")
    print("  - param2: 0.0-1.0 (for click: y/height, for answer: (lon+180)/360)")
    print()

    steps = 0
    total_reward = 0.0
    done = False

    print("=" * 80)
    print("EPISODE EXECUTION")
    print("=" * 80)
    print()

    while not done and steps < 10:
        print(f"Step {steps + 1}:")

        if steps < 8:
            # Click actions - navigate around
            if steps == 0:
                # Click at center
                action = np.array([0.25, 0.5, 0.5], dtype=np.float32)
                print("  Action: Click at center (512, 256)")
            elif steps == 1:
                # Click to the right
                action = np.array([0.25, 0.75, 0.5], dtype=np.float32)
                print("  Action: Click right (768, 256)")
            elif steps == 2:
                # Click to the left
                action = np.array([0.25, 0.25, 0.5], dtype=np.float32)
                print("  Action: Click left (256, 256)")
            else:
                # Random click
                action = env.action_space.sample()
                # Force it to be a click
                action[0] = 0.25
                x = int(action[1] * 1024)
                y = int(action[2] * 512)
                print(f"  Action: Random click ({x}, {y})")
        else:
            # Answer action on last step
            # Guess somewhere near the actual location
            # Target: lat=51.48, lon=-0.16
            # Normalized: lat = (51.48 + 90) / 180 = 0.786, lon = (-0.16 + 180) / 360 = 0.4996
            action = np.array([0.75, 0.786, 0.4996], dtype=np.float32)
            print("  Action: Answer (lat=51.48, lon=-0.16)")

        # Take action
        observation, reward, terminated, truncated, info = env.step(action)

        print(f"  Reward: {reward:.4f}")
        print(f"  Terminated: {terminated}, Truncated: {truncated}")

        if terminated:
            print(f"  Guess: lat={info.get('guess_lat', 0):.6f}, lon={info.get('guess_lon', 0):.6f}")
            print(f"  Distance: {info.get('distance_km', 0):.2f} km")
            print(f"  Final score: {info.get('score', 0):.4f}")

        # Render
        env.render()

        total_reward += float(reward)
        steps += 1
        done = terminated or truncated
        print()

    print("=" * 80)
    print(f"Episode finished after {steps} steps")
    print(f"Total reward: {total_reward:.4f}")
    print("=" * 80)

    env.close()


if __name__ == "__main__":
    main()
