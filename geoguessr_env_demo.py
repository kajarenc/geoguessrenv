import gymnasium as gym
from gymnasium.envs.registration import register


ENV_ID = "GeoGuessrWorld-v0"


def ensure_registered() -> None:
    try:
        # Safe to call repeatedly; ignore if already registered
        register(
            id=ENV_ID,
            entry_point="gymnasium_env.envs.geoguessr_world:GeoGuessrWorldEnv",
        )
    except Exception:
        # Likely already registered in this interpreter session
        pass


def main() -> None:
    ensure_registered()

    env = gym.make(ENV_ID, render_mode="human")
    observation, info = env.reset()
    lat = info["gt_lat"]
    lon = info["gt_lon"]
    guess_lat, guess_lon = 51.506843, -0.113710
    links = info.get("links", [])
    print(f"pano_id={info.get('pano_id')} gt_lat={lat:.6f} gt_lon={lon:.6f} links={len(links)}")

    steps = 0
    total_reward = 0.0
    done = False
    while not done and steps < 11:
        # On the 3th step, click at screen position (x=567, y=256)
        if steps == 2:
            action = {"op": "click", "value": [737, 256]}
        elif steps == 10:
            action = {"op": "answer", "value": [guess_lat, guess_lon]}
        else:
            action = {"op": "click", "value": [0, 0]}
        observation, reward, terminated, truncated, info = env.step(action)
        print(info)
        print('---------')
        env.render()
        total_reward += reward
        steps += 1
        done = terminated or truncated

    print(f"Episode finished after {steps} steps. Total reward: {total_reward}")
    env.close()


if __name__ == "__main__":
    main()


