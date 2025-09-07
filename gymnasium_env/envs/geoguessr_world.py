from gymnasium import spaces
import pygame
import numpy as np
import gymnasium as gym
import os
import json
import math
from typing import Dict, List, Optional, Tuple
from PIL import Image


class GeoGuessrWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        # Config (MVP: fixed root pano and pre-downloaded assets)
        self.render_mode = render_mode
        self.size = size
        self.pano_root_id: str = "moy3EWiKMN8DvB9Zu3kITg" # London
        # self.pano_root_id: str = "B8Yw_SheqPArDjOk4WL4yw"

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

        # Controls (click mapping)
        self.arrow_hit_radius_px: int = 24
        self.arrow_min_conf: float = 0.0

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
        # Action space
        # op: 0 = click, 1 = answer
        # value: two floats; for click -> (x, y) in pixels (will be clamped), for answer -> (lat, lon)
        self.action_space = spaces.Dict(
            {
                "op": spaces.Discrete(2),
                "value": spaces.Box(
                    low=np.array([-90.0, -180.0], dtype=np.float32),
                    high=np.array([90.0, 180.0], dtype=np.float32),
                    shape=(2,),
                    dtype=np.float32,
                ),
            }
        )

        # Runtime state
        self.current_pano_id: Optional[str] = None
        self.current_lat: Optional[float] = None
        self.current_lon: Optional[float] = None
        self.current_links: List[Dict] = []
        self._current_image: Optional[np.ndarray] = None
        self._steps: int = 0
        self._heading_rad: float = 0.0  # current camera heading in radians

        # Pygame render state
        self._screen = None
        self._clock = None
        self._font = None


    def _get_info(self):
        links_with_screen = self._compute_link_screens()
        return {
            "pano_id": self.current_pano_id,
            "gt_lat": self.current_lat,
            "gt_lon": self.current_lon,
            "steps": self._steps,
            "pose": {"heading_deg": math.degrees(self._heading_rad) % 360.0},
            "links": links_with_screen,
        }


    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self._steps = 0
        # Initialize to fixed root pano
        self._set_current_pano(self.pano_root_id)
        # Initialize camera heading to current pano heading if available
        node = self._pano_graph.get(self.current_pano_id, {})
        heading = node.get("heading")
        if isinstance(heading, (int, float)):
            self._heading_rad = float(heading)
        else:
            self._heading_rad = 0.0

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    # --- Step with click and answer actions ---
    def step(self, action):
        self._steps += 1

        op, value = self._parse_action(action)

        terminated = False
        truncated = False
        reward = 0.0

        if op == 0:
            # click: value is (x, y) pixels
            x, y = value
            self._handle_click(x, y)
        elif op == 1:
            # answer: value is (lat, lon)
            guess_lat, guess_lon = value
            reward = self._compute_answer_reward(guess_lat, guess_lon)
            terminated = True

        obs = self._get_observation()
        info = self._get_info()
        if op == 1:
            info["guess_lat"] = float(value[0])
            info["guess_lon"] = float(value[1])
            distance_km = self._haversine_km(
                float(self.current_lat), float(self.current_lon), float(value[0]), float(value[1])
            )
            info["distance_km"] = distance_km
            info["score"] = reward
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
                if self._font is None:
                    try:
                        self._font = pygame.font.SysFont("Arial", 14)
                    except Exception:
                        self._font = pygame.font.Font(None, 14)

            # Pump events to keep window responsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pass

            frame = self._get_observation()
            # pygame expects (width, height, 3) and surfaces are transposed
            surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
            self._screen.blit(surf, (0, 0))

            # Overlay link centers and pano ids for human debugging only
            links = self._compute_link_screens()
            radius = int(self.arrow_hit_radius_px)
            for l in links:
                cx, cy = l["screen_xy"]
                # Draw filled red circle
                pygame.draw.circle(self._screen, (255, 0, 0), (int(cx), int(cy)), radius, 0)
                # Draw white outline
                pygame.draw.circle(self._screen, (255, 255, 255), (int(cx), int(cy)), radius, 2)
                # Draw pano id below (or above if near bottom)
                if self._font is not None:
                    text_surf = self._font.render(str(l["id"]), True, (255, 255, 255))
                    text_rect = text_surf.get_rect()
                    text_rect.centerx = int(cx)
                    text_rect.top = int(cy) + radius + 4
                    if text_rect.bottom > self._image_height:
                        text_rect.bottom = int(cy) - radius - 4
                        text_rect.top = text_rect.bottom - text_rect.height
                    # simple shadow for readability
                    shadow = text_rect.copy()
                    shadow.x += 1
                    shadow.y += 1
                    self._screen.blit(text_surf, shadow)
                    self._screen.blit(text_surf, text_rect)
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
                heading = data.get("heading")
                links_raw = data.get("links", []) or []
                links: List[Dict[str, object]] = []
                for link in links_raw:
                    link_pano = (link or {}).get("pano") or {}
                    link_id = link_pano.get("id")
                    direction = link.get("direction")
                    if isinstance(link_id, str) and isinstance(direction, (int, float)):
                        links.append({"id": link_id, "direction": float(direction)})
                graph[pano_id] = {"lat": lat, "lon": lon, "heading": heading, "links": links}
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

    # --- Click handling and link mapping ---
    def _compute_link_screens(self) -> List[Dict[str, object]]:
        """
        Compute screen-space centers for current links. We project directions (radians)
        into equirectangular x positions, with y at image vertical center.
        """
        node = self._pano_graph.get(self.current_pano_id, {})
        pano_heading = node.get("heading")
        if not isinstance(pano_heading, (int, float)):
            pano_heading = 0.0
        links: List[Dict[str, object]] = []
        for link in self.current_links:
            direction = float(link.get("direction", 0.0))
            link_id = str(link.get("id"))
            x = self._direction_to_x(direction, self._image_width)
            y = self._image_height // 2
            # rel heading in degrees for tie-breakers: how far from current heading
            abs_heading = self._normalize_angle(direction + pano_heading)
            rel = self._angle_diff_rad(abs_heading, self._heading_rad)
            links.append(
                {
                    "id": link_id,
                    "heading_deg": float((math.degrees(abs_heading) % 360.0)),
                    "screen_xy": [int(x), int(y)],
                    "conf": 1.0,
                    "_distance_px": None,  # filled during hit-test
                    "_rel_heading_deg": float(abs(math.degrees(rel))),
                }
            )
        return links

    def _handle_click(self, x: float, y: float) -> None:
        # Clamp click to image bounds and round to int
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        xi = max(0, min(self._image_width - 1, xi))
        yi = max(0, min(self._image_height - 1, yi))

        links = self._compute_link_screens()
        if not links:
            return
        # Compute distances
        for l in links:
            cx, cy = l["screen_xy"]
            dx = xi - int(cx)
            dy = yi - int(cy)
            l["_distance_px"] = math.hypot(dx, dy)

        # Filter by radius and confidence
        candidates = [
            l for l in links if (l["_distance_px"] is not None and l["_distance_px"] <= float(self.arrow_hit_radius_px) and float(l["conf"]) >= float(self.arrow_min_conf))
        ]
        if not candidates:
            return  # no-op

        # Sort by distance, then smallest abs rel heading, then lexicographic pano id
        candidates.sort(key=lambda l: (float(l["_distance_px"]), float(l["_rel_heading_deg"]), str(l["id"])))
        chosen = candidates[0]
        next_id = str(chosen["id"])

        # Move to neighbor pano
        self._set_current_pano(next_id)
        # Update camera heading to new pano's heading
        node = self._pano_graph.get(self.current_pano_id, {})
        heading = node.get("heading")
        if isinstance(heading, (int, float)):
            self._heading_rad = float(heading)

    # --- Geometry helpers ---
    @staticmethod
    def _normalize_angle(a: float) -> float:
        tau = getattr(math, "tau", 2 * math.pi)
        return a % tau

    @staticmethod
    def _angle_diff_rad(a: float, b: float) -> float:
        """Smallest signed difference a-b in radians in [-pi, pi]."""
        d = (a - b + math.pi) % (2 * math.pi) - math.pi
        return d

    @staticmethod
    def _direction_to_x(direction: float, image_width: int) -> int:
        tau = getattr(math, "tau", 2 * math.pi)
        d = (direction + tau) % tau
        x_float = (d / tau) * float(image_width)
        x = int(round(x_float))
        if x < 0:
            x = 0
        if x >= image_width:
            x = image_width - 1
        return x

    # --- Answer reward ---
    def _compute_answer_reward(self, guess_lat: float, guess_lon: float) -> float:
        if self.current_lat is None or self.current_lon is None:
            return 0.0
        d_km = self._haversine_km(float(self.current_lat), float(self.current_lon), float(guess_lat), float(guess_lon))
        score = math.exp(-d_km / 400.0)
        return float(score)

    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371.0
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    # --- Action parsing ---
    def _parse_action(self, action) -> Tuple[int, Tuple[float, float]]:
        """
        Returns a pair (op, value) where:
        - op: 0 for click, 1 for answer
        - value: (x, y) for click; (lat, lon) for answer
        Accepts either a dict {"op": int or str, "value": [a,b]} or already-normalized tuple.
        """
        if isinstance(action, dict):
            op = action.get("op")
            if isinstance(op, str):
                op_norm = 0 if op == "click" else 1
            else:
                op_norm = int(op)
            val = action.get("value", [0, 0])
            if isinstance(val, (list, tuple)) and len(val) == 2:
                v0 = float(val[0])
                v1 = float(val[1])
            else:
                v0, v1 = 0.0, 0.0
            # Validate/clamp pixels for click based on actual image size
            if op_norm == 0:
                v0 = max(0.0, min(float(self._image_width - 1), v0))
                v1 = max(0.0, min(float(self._image_height - 1), v1))
            return op_norm, (v0, v1)
        # Fallback: treat as no-op click center
        return 0, (float(self._image_width // 2), float(self._image_height // 2))