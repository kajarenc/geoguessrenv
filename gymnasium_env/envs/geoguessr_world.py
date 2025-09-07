from gymnasium import spaces
import pygame
import numpy as np
import gymnasium as gym
import os
import json
from typing import Dict, List, Optional, Tuple
from PIL import Image


class GeoGuessrWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        # Config (MVP: fixed root pano and pre-downloaded assets)
        self.render_mode = render_mode
        self.size = size
        self.pano_root_id: str = "B8Yw_SheqPArDjOk4WL4yw"

        # Resolve project paths
        # This file lives at gymnasium_env/envs/geoguessr_world.py
        # Images are at gymnasium_env/load/images/<pano_id>.jpg
        # Metadata JSONL is at gymnasium_env/load/metadata/<root>_minimetadata.jsonl
        env_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(env_dir)
        self.images_dir = os.path.join(project_root, "load", "images")
        self.metadata_path = os.path.join(
            project_root, "load", "metadata", f"{self.pano_root_id}_minimetadata.jsonl"
        )

        # Load metadata graph into memory
        self._pano_graph: Dict[str, Dict] = self._load_minimetadata(self.metadata_path)

        # Determine observation space from root image
        root_image_path = os.path.join(self.images_dir, f"{self.pano_root_id}.jpg")
        with Image.open(root_image_path) as img:
            img = img.convert("RGB")
            width, height = img.size
        self._image_width = width
        self._image_height = height
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self._image_height, self._image_width, 3),
            dtype=np.uint8,
        )
        # Placeholder action space (no-op)
        self.action_space = spaces.Discrete(1)

        # Runtime state
        self.current_pano_id: Optional[str] = None
        self.current_lat: Optional[float] = None
        self.current_lon: Optional[float] = None
        self.current_links: List[Dict] = []
        self._current_image: Optional[np.ndarray] = None
        self._steps: int = 0

        # Pygame render state
        self._screen = None
        self._clock = None


    def _get_info(self):
        return {
            "pano_id": self.current_pano_id,
            "lat": self.current_lat,
            "lon": self.current_lon,
            "links": self.current_links,
            "steps": self._steps,
        }


    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self._steps = 0
        # Initialize to fixed root pano
        self._set_current_pano(self.pano_root_id)

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    # --- Minimal step: no-op, reward 0.0 ---
    def step(self, action):
        self._steps += 1
        obs = self._get_observation()
        reward = 0.0
        terminated = False
        truncated = False
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    # --- Rendering ---
    def render(self):
        if self.render_mode == "rgb_array":
            return self._get_observation()
        if self.render_mode == "human":
            if self._screen is None:
                pygame.init()
                self._screen = pygame.display.set_mode((self._image_width, self._image_height))
                pygame.display.set_caption("GeoGuessrWorldEnv")
                self._clock = pygame.time.Clock()

            # Pump events to keep window responsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pass

            frame = self._get_observation()
            # pygame expects (width, height, 3) and surfaces are transposed
            surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
            self._screen.blit(surf, (0, 0))
            pygame.display.flip()
            # Cap to metadata fps
            if self._clock is not None:
                self._clock.tick(self.metadata.get("render_fps", 4))
            return None
        # If no render mode specified, do nothing
        return None

    def close(self):
        if self._screen is not None:
            try:
                pygame.display.quit()
                pygame.quit()
            except Exception:
                pass
            self._screen = None
            self._clock = None

    # --- Helpers ---
    def _load_minimetadata(self, jsonl_path: str) -> Dict[str, Dict]:
        graph: Dict[str, Dict] = {}
        if not os.path.isfile(jsonl_path):
            raise FileNotFoundError(f"Metadata file not found: {jsonl_path}")
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                pano_id = data.get("id")
                if not pano_id:
                    continue
                lat = data.get("lat")
                lon = data.get("lon")
                links_raw = data.get("links", []) or []
                links: List[Dict[str, object]] = []
                for link in links_raw:
                    link_pano = (link or {}).get("pano") or {}
                    link_id = link_pano.get("id")
                    direction = link.get("direction")
                    if isinstance(link_id, str) and isinstance(direction, (int, float)):
                        links.append({"id": link_id, "direction": float(direction)})
                graph[pano_id] = {"lat": lat, "lon": lon, "links": links}
        return graph

    def _set_current_pano(self, pano_id: str) -> None:
        node = self._pano_graph.get(pano_id)
        if node is None:
            raise KeyError(f"Pano id not found in metadata: {pano_id}")
        self.current_pano_id = pano_id
        self.current_lat = float(node.get("lat")) if node.get("lat") is not None else None
        self.current_lon = float(node.get("lon")) if node.get("lon") is not None else None
        self.current_links = list(node.get("links", []))
        self._current_image = None  # force reload on next observation

    def _get_observation(self) -> np.ndarray:
        if self._current_image is None:
            image_path = os.path.join(self.images_dir, f"{self.current_pano_id}.jpg")
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                np_img = np.array(img, dtype=np.uint8)
            self._current_image = np_img
        return self._current_image