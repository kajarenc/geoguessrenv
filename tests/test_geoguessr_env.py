from pathlib import Path

import numpy as np

from gymnasium_env.envs import GeoGuessrWorldEnv


def test_answer_action_terminates_episode():
    repo_root = Path(__file__).resolve().parents[1]
    cache_root = str(repo_root / "tempcache")

    # Choose any coordinates present in tempcache/metadata/nearest_pano_cache.json
    # so that reset() runs entirely offline using cached assets.
    config = {
        "cache_root": cache_root,
        "input_lat": 47.620908,
        "input_lon": -122.353508,
        "max_steps": 5,
    }

    env = GeoGuessrWorldEnv(config=config)
    try:
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert "gt_lat" in info and "gt_lon" in info

        # Issue an answer action; any lat/lon should terminate.
        action = {"op": "answer", "value": [info["gt_lat"], info["gt_lon"]]}
        obs2, reward, terminated, truncated, info2 = env.step(action)

        assert terminated is True
        assert truncated is False
        assert isinstance(reward, float)
        assert 0.0 <= reward <= 1.0
    finally:
        env.close()


