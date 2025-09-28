import argparse
import logging
from pathlib import Path

import gymnasium as gym
from gymnasium.envs.registration import register

from agents.base import AgentConfig
from agents.openai_agent import OpenAIVisionAgent

ENV_ID = "GeoGuessrWorld-v0"


def ensure_registered() -> None:
    try:
        register(
            id=ENV_ID,
            entry_point="geoguess_env.geoguessr_env:GeoGuessrEnv",
        )
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run OpenAI agent in GeoGuessrWorldEnv")
    p.add_argument("--model", default="gpt-4o")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_nav_steps", type=int, default=10)
    p.add_argument("--image_width", type=int, default=1024)
    p.add_argument("--image_height", type=int, default=512)
    p.add_argument("--input_lat", type=float, default=51.481610)
    p.add_argument("--input_lon", type=float, default=-0.163400)
    p.add_argument("--cache_root", type=str, default=None)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--render", action="store_true")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    # Load .env for OPENAI_API_KEY if present
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
    except Exception:
        pass
    ensure_registered()

    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    cache_root = Path(args.cache_root) if args.cache_root else project_root / "cache"

    render_config = {
        "image_width": args.image_width,
        "image_height": args.image_height,
    }
    if args.render:
        render_config["render_mode"] = "human"

    env_config = {
        "cache_root": str(cache_root),
        "input_lat": args.input_lat,
        "input_lon": args.input_lon,
        "render_config": render_config,
    }

    env = gym.make(
        ENV_ID,
        render_mode="human" if args.render else None,
        config=env_config,
    )

    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    agent_cfg = AgentConfig(
        model=args.model,
        temperature=args.temperature,
        max_nav_steps=args.max_nav_steps,
        image_width=args.image_width,
        image_height=args.image_height,
        cache_dir=cache_dir,
    )
    agent = OpenAIVisionAgent(agent_cfg)

    agent.reset()
    obs, info = env.reset()

    steps = 0
    total_reward = 0.0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        if steps >= agent_cfg.max_nav_steps:
            # Force answer if we hit step budget
            # Use a naive guess at (0,0); the agent should generally answer earlier
            parser = getattr(env, "action_parser", None)
            if parser is None and hasattr(env, "unwrapped"):
                parser = getattr(env.unwrapped, "action_parser", None)
            if parser is not None:
                action = parser.create_answer_action(0.0, 0.0)
            else:
                action = {"op": "answer", "value": [0.0, 0.0]}
        else:
            action = agent.act(obs, info)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        if args.render:
            env.render()

        if terminated or truncated:
            break

    print(
        {
            "steps": steps,
            "score": info.get("score"),
            "distance_km": info.get("distance_km"),
            "pano_id": info.get("pano_id"),
        }
    )
    env.close()


if __name__ == "__main__":
    main()
