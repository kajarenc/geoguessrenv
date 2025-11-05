"""
Gymnasium wrappers for the GeoGuessr environment.

Provides wrappers for action/observation transformations and backward compatibility.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class LegacyActionWrapper(gym.ActionWrapper):
    """Wrapper for backward compatibility with legacy dict action format.

    Converts the old Dict action space format to the new unified Box format.
    This allows existing code using the old format to continue working.

    Legacy format:
        {'op': 0 or 1, 'click': [x, y], 'answer': [lat, lon]}

    Unified format:
        [action_type, param1, param2] where values are in [0, 1]

    Example:
        >>> env = gym.make("GeoGuessr-v0")
        >>> env = LegacyActionWrapper(env)
        >>> # Now you can use the old dict format
        >>> action = {'op': 0, 'click': [512, 256], 'answer': [0, 0]}
        >>> env.step(action)
    """

    def __init__(self, env: gym.Env):
        """Initialize the wrapper.

        Args:
            env: The environment to wrap
        """
        super().__init__(env)

        # Override action space to show legacy Dict format
        self.action_space = spaces.Dict(
            {
                "op": spaces.Discrete(2),
                "click": spaces.Box(
                    low=np.array([0, 0], dtype=np.int32),
                    high=np.array([1023, 511], dtype=np.int32),
                    shape=(2,),
                    dtype=np.int32,
                ),
                "answer": spaces.Box(
                    low=np.array([-90.0, -180.0], dtype=np.float32),
                    high=np.array([90.0, 180.0], dtype=np.float32),
                    shape=(2,),
                    dtype=np.float32,
                ),
            }
        )

    def action(self, action: dict) -> np.ndarray:
        """Convert legacy dict format to unified array format.

        Args:
            action: Dict with keys 'op', 'click', and 'answer'

        Returns:
            Unified action array [action_type, param1, param2]
        """
        op = action.get("op", 0)

        if op == 0:
            # Click action
            click = action.get("click", [512, 256])
            x, y = float(click[0]), float(click[1])

            # Normalize to [0, 1]
            x_norm = x / 1024.0
            y_norm = y / 512.0

            return np.array([0.25, x_norm, y_norm], dtype=np.float32)
        else:
            # Answer action
            answer = action.get("answer", [0.0, 0.0])
            lat, lon = float(answer[0]), float(answer[1])

            # Normalize to [0, 1]
            lat_norm = (lat + 90.0) / 180.0
            lon_norm = (lon + 180.0) / 360.0

            return np.array([0.75, lat_norm, lon_norm], dtype=np.float32)


class NormalizeObservation(gym.ObservationWrapper):
    """Normalize image observations from [0, 255] to [0, 1] range.

    Many RL algorithms expect normalized observations. This wrapper converts
    uint8 images to float32 in the [0, 1] range.

    Example:
        >>> env = gym.make("GeoGuessr-v0")
        >>> env = NormalizeObservation(env)
        >>> obs, _ = env.reset()
        >>> assert obs["image"].dtype == np.float32
        >>> assert obs["image"].min() >= 0.0 and obs["image"].max() <= 1.0
    """

    def __init__(self, env: gym.Env):
        """Initialize the wrapper.

        Args:
            env: The environment to wrap
        """
        super().__init__(env)

        # Update observation space to float32
        # Type assertion: we know the observation space is Dict
        assert isinstance(env.observation_space, spaces.Dict), (
            "Observation space must be Dict"
        )
        old_image_space = env.observation_space["image"]  # type: ignore[index]
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=old_image_space.shape,
                    dtype=np.float32,
                )
            }
        )

    def observation(self, obs: dict) -> dict:
        """Normalize the image observation.

        Args:
            obs: Observation dict with 'image' key

        Returns:
            Observation with normalized image
        """
        return {"image": obs["image"].astype(np.float32) / 255.0}


class FlattenObservation(gym.ObservationWrapper):
    """Flatten the image observation to a 1D vector.

    Some algorithms (e.g., simple feedforward networks) expect flattened
    observations. This wrapper reshapes the image from (H, W, C) to (H*W*C,).

    Example:
        >>> env = gym.make("GeoGuessr-v0")
        >>> env = FlattenObservation(env)
        >>> obs, _ = env.reset()
        >>> # Image is now flattened
        >>> assert obs["image"].shape == (512 * 1024 * 3,)
    """

    def __init__(self, env: gym.Env):
        """Initialize the wrapper.

        Args:
            env: The environment to wrap
        """
        super().__init__(env)

        # Update observation space to flattened
        # Type assertion: we know the observation space is Dict
        assert isinstance(env.observation_space, spaces.Dict), (
            "Observation space must be Dict"
        )
        old_image_space = env.observation_space["image"]  # type: ignore[index]
        flat_dim = int(np.prod(old_image_space.shape))

        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(flat_dim,),
                    dtype=old_image_space.dtype,
                )
            }
        )

    def observation(self, obs: dict) -> dict:
        """Flatten the image observation.

        Args:
            obs: Observation dict with 'image' key

        Returns:
            Observation with flattened image
        """
        return {"image": obs["image"].flatten()}


class GrayscaleObservation(gym.ObservationWrapper):
    """Convert RGB images to grayscale.

    Reduces the observation space by converting RGB (3 channels) to grayscale
    (1 channel) using standard luminance weights.

    Example:
        >>> env = gym.make("GeoGuessr-v0")
        >>> env = GrayscaleObservation(env)
        >>> obs, _ = env.reset()
        >>> # Image is now grayscale
        >>> assert obs["image"].shape == (512, 1024, 1)
    """

    def __init__(self, env: gym.Env):
        """Initialize the wrapper.

        Args:
            env: The environment to wrap
        """
        super().__init__(env)

        # Update observation space to grayscale
        # Type assertion: we know the observation space is Dict
        assert isinstance(env.observation_space, spaces.Dict), (
            "Observation space must be Dict"
        )
        old_image_space = env.observation_space["image"]  # type: ignore[index]
        h, w, _ = old_image_space.shape

        self.observation_space = spaces.Dict(
            {"image": spaces.Box(low=0, high=255, shape=(h, w, 1), dtype=np.uint8)}
        )

    def observation(self, obs: dict) -> dict:
        """Convert RGB to grayscale.

        Args:
            obs: Observation dict with RGB 'image'

        Returns:
            Observation with grayscale image
        """
        rgb_image = obs["image"]

        # Convert to grayscale using luminance weights
        gray = np.dot(rgb_image[..., :3], [0.2989, 0.5870, 0.1140])
        gray = gray.astype(np.uint8)

        return {"image": gray[..., np.newaxis]}


class ResizeObservation(gym.ObservationWrapper):
    """Resize image observations to a different resolution.

    Useful for reducing computational requirements or matching specific
    model input sizes.

    Example:
        >>> env = gym.make("GeoGuessr-v0")
        >>> env = ResizeObservation(env, width=512, height=256)
        >>> obs, _ = env.reset()
        >>> assert obs["image"].shape == (256, 512, 3)
    """

    def __init__(self, env: gym.Env, width: int, height: int):
        """Initialize the wrapper.

        Args:
            env: The environment to wrap
            width: Target width
            height: Target height
        """
        super().__init__(env)

        self.width = width
        self.height = height

        # Update observation space
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0, high=255, shape=(height, width, 3), dtype=np.uint8
                )
            }
        )

    def observation(self, obs: dict) -> dict:
        """Resize the image observation.

        Args:
            obs: Observation dict with 'image' key

        Returns:
            Observation with resized image
        """
        from PIL import Image

        image = obs["image"]

        # Convert to PIL Image, resize, and convert back
        pil_image = Image.fromarray(image)
        resized = pil_image.resize((self.width, self.height), Image.BILINEAR)

        return {"image": np.array(resized)}


class ActionRepeat(gym.Wrapper):
    """Repeat actions for multiple steps.

    Each action is repeated for N steps, and only the final observation
    is returned. Rewards are accumulated.

    This can speed up exploration and reduce the effective episode length.

    Example:
        >>> env = gym.make("GeoGuessr-v0")
        >>> env = ActionRepeat(env, repeat=4)
        >>> # Each action is now repeated 4 times
    """

    def __init__(self, env: gym.Env, repeat: int = 4):
        """Initialize the wrapper.

        Args:
            env: The environment to wrap
            repeat: Number of times to repeat each action
        """
        super().__init__(env)
        self.repeat = repeat

    def step(self, action):
        """Repeat action and accumulate rewards.

        Args:
            action: Action to repeat

        Returns:
            Final observation, accumulated reward, terminated, truncated, info
        """
        total_reward = 0.0
        terminated = False
        truncated = False

        for _ in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info
