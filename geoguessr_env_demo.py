import logging
import os

import gymnasium as gym
from gymnasium.envs.registration import register

ENV_ID = "GeoGuessr-v0"


def ensure_registered() -> None:
    try:
        # Safe to call repeatedly; ignore if already registered
        register(
            id=ENV_ID,
            entry_point="geoguess_env.geoguessr_env:GeoGuessrEnv",
        )
    except Exception:
        # Likely already registered in this interpreter session
        pass


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ensure_registered()

    # Use cache directory aligned with TaskDescription.md spec
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_root = os.path.join(script_dir, "cache")

    input_lat, input_lon = 51.481610, -0.163400
    env = gym.make(
        ENV_ID,
        render_mode="human",
        config={
            "provider_config": {"provider": "gsv"},
            "cache_root": cache_root,
            "input_lat": input_lat,
            "input_lon": input_lon,
        },
    )
    observation, info = env.reset()
    lat = info["gt_lat"]
    lon = info["gt_lon"]
    guess_lat, guess_lon = 51.506843, -0.113710
    links = info.get("links", [])
    print(
        f"pano_id={info.get('pano_id')} gt_lat={lat:.6f} gt_lon={lon:.6f} links={len(links)}"
    )

    steps = 0
    total_reward = 0.0
    done = False
    while not done and steps < 11:
        # On the 3rd step, click at screen position (x=740, y=256)
        if steps == 2:
            action = {"op": "click", "value": [740, 256]}
        elif steps == 10:
            action = {"op": "answer", "value": [guess_lat, guess_lon]}
        else:
            action = {"op": "click", "value": [512, 256]}
        observation, reward, terminated, truncated, info = env.step(action)
        print(info)
        print("---------")
        env.render()
        total_reward += float(reward)
        steps += 1
        done = terminated or truncated

    print(f"Episode finished after {steps} steps. Total reward: {total_reward}")
    env.close()


if __name__ == "__main__":
    main()
