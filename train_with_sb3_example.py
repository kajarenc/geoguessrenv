#!/usr/bin/env python3
"""
Example: Training with Stable-Baselines3

This example demonstrates how to train a PPO agent on the GeoGuessr environment
using the new unified action space.

Requirements:
    pip install stable-baselines3

Usage:
    python train_with_sb3_example.py
"""

import logging
from pathlib import Path

import gymnasium as gym
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
        pass


def make_env(rank: int = 0):
    """Factory function to create environment instances."""

    def _init():
        script_path = Path(__file__).resolve()
        cache_root = script_path.parent / "cache"

        env = gym.make(
            ENV_ID,
            config={
                "provider_config": {"provider": "gsv"},
                "cache_root": str(cache_root),
                "input_lat": 47.620908,  # Seattle
                "input_lon": -122.353508,
                "max_steps": 10,
            },
        )
        return env

    return _init


def main():
    """Train a PPO agent on the GeoGuessr environment."""
    logging.basicConfig(level=logging.INFO)
    ensure_registered()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError:
        print("=" * 80)
        print("ERROR: Stable-Baselines3 not installed")
        print("=" * 80)
        print()
        print("Please install stable-baselines3:")
        print("  pip install stable-baselines3")
        print()
        print("Or install with all RL dependencies:")
        print("  pip install stable-baselines3[extra]")
        return

    print("=" * 80)
    print("TRAINING GEOGUESSR WITH STABLE-BASELINES3 PPO")
    print("=" * 80)
    print()

    # Create environment
    print("Creating environment...")
    env = make_env()()

    # Check environment compatibility
    print("Checking environment compatibility with Gymnasium API...")
    try:
        check_env(env, warn=True)
        print("✓ Environment is compatible with Gymnasium API")
    except Exception as e:
        print(f"✗ Environment check failed: {e}")
        return
    finally:
        env.close()
    print()

    # Create vectorized environment (required for SB3)
    print("Creating vectorized environment...")
    vec_env = DummyVecEnv([make_env(i) for i in range(1)])
    print(f"✓ Vectorized environment created with 1 worker")
    print()

    # Verify action space
    print("Action Space Information:")
    print(f"  Type: {vec_env.action_space}")
    print(f"  Shape: {vec_env.action_space.shape}")
    print(f"  Dtype: {vec_env.action_space.dtype}")
    print()

    # Create PPO model
    print("Creating PPO model...")
    print("  Policy: MlpPolicy (Multi-Layer Perceptron)")
    print("  Algorithm: Proximal Policy Optimization (PPO)")
    print()

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log="./tensorboard_logs/",
    )

    print("✓ PPO model created")
    print()

    # Print model info
    print("Model Configuration:")
    print(f"  Learning rate: {model.learning_rate}")
    print(f"  Steps per update: {model.n_steps}")
    print(f"  Batch size: {model.batch_size}")
    print(f"  Epochs per update: {model.n_epochs}")
    print()

    # Train the model
    print("=" * 80)
    print("TRAINING")
    print("=" * 80)
    print()
    print("Starting training for 10,000 steps...")
    print("(This is just a demo - real training would require much more steps)")
    print()

    try:
        model.learn(total_timesteps=10000, progress_bar=True)
        print()
        print("✓ Training completed successfully!")
    except KeyboardInterrupt:
        print()
        print("Training interrupted by user")
    except Exception as e:
        print()
        print(f"Training failed: {e}")
        raise
    print()

    # Save the model
    model_path = "./geoguessr_ppo_model"
    print(f"Saving model to {model_path}...")
    model.save(model_path)
    print("✓ Model saved")
    print()

    # Test the trained model
    print("=" * 80)
    print("TESTING TRAINED MODEL")
    print("=" * 80)
    print()

    test_env = make_env()()
    obs, info = test_env.reset()

    print("Running one episode with trained policy...")
    print()

    total_reward = 0
    done = False
    step = 0

    while not done and step < 10:
        # Predict action using trained policy
        action, _states = model.predict(obs, deterministic=True)

        # Take action
        obs, reward, terminated, truncated, info = test_env.step(action)

        print(f"Step {step + 1}: reward={reward:.4f}")

        total_reward += reward
        done = terminated or truncated
        step += 1

    print()
    print(f"Episode finished: {step} steps, total reward={total_reward:.4f}")

    if "distance_km" in info:
        print(f"Final guess distance: {info['distance_km']:.2f} km")

    test_env.close()
    vec_env.close()

    print()
    print("=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Train for longer (e.g., 1M+ steps)")
    print("  2. Tune hyperparameters")
    print("  3. Use vectorized environments for faster training")
    print("  4. Add observation preprocessing (normalize images)")
    print("  5. Experiment with different algorithms (SAC, DQN, etc.)")
    print()
    print("To view training progress:")
    print("  tensorboard --logdir ./tensorboard_logs/")
    print()


if __name__ == "__main__":
    main()
