import json
import math
import os
import random
from typing import Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from PIL import Image

from .helpers import (
    download_images,
    download_metadata,
    get_nearest_pano_id,
)


class GeoGuessrEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    def __init__(self, config: Optional[Dict] = None, render_mode=None):
        # Normalize config
        cfg = dict(config or {})
        # Public config (towards spec)
        self.provider: Optional[str] = cfg.get("provider")
        self.mode: Optional[str] = cfg.get("mode")  # {online, offline}
        self.geofence: Optional[Dict] = cfg.get("geofence")
        self.input_lat: Optional[float] = cfg.get("input_lat")
        self.input_lon: Optional[float] = cfg.get("input_lon")
        self.cache_root: Optional[str] = cfg.get("cache_root")
        self.max_steps: int = int(cfg.get("max_steps", 40))
        self._initial_seed: Optional[int] = cfg.get("seed")
        self.rate_limit_qps: Optional[float] = cfg.get("rate_limit_qps")
        self.max_fetch_retries: Optional[int] = cfg.get("max_fetch_retries")
        self.min_capture_year: Optional[int] = cfg.get("min_capture_year")

        # Rendering config
        self.render_mode = render_mode

        # Controls (click mapping)
        self.arrow_hit_radius_px: int = int(cfg.get("arrow_hit_radius_px", 24))
        self.arrow_min_conf: float = float(cfg.get("arrow_min_conf", 0.0))

        # Root pano selection (MVP: fixed root pano and pre-downloaded assets)
        self.pano_root_id: Optional[str] = None

        # Resolve project paths using cache_root when provided
        # images -> <cache_root>/images; metadata -> <cache_root>/metadata
        env_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(env_dir)
        cache_root = self.cache_root or os.path.join(project_root, "cache")
        self.images_dir = os.path.join(cache_root, "images")
        self.metadata_dir = os.path.join(cache_root, "metadata")

        # Defer loading metadata and reading image sizes to reset() after downloads
        self._pano_graph: Dict[str, Dict] = {}
        # Provide sensible defaults for dimensions until reset() initializes them
        self._image_width = 1024
        self._image_height = 512

        # Observation is a dictionary with an "image" key per spec
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(self._image_height, self._image_width, 3),
                    dtype=np.uint8,
                )
            }
        )
        # Action space
        # Two actions:
        # - op == 0: click → value in pixels (x, y) with explicit bounds [0,1024] x [0,512]
        # - op == 1: answer → value is (lat, lon) with standard bounds [-90,90] x [-180,180]
        self._click_low = np.array([0, 0], dtype=np.int32)
        self._click_high = np.array([1024, 512], dtype=np.int32)
        click_space = spaces.Box(
            low=self._click_low,
            high=self._click_high,
            shape=(2,),
            dtype=np.int32,
        )
        answer_space = spaces.Box(
            low=np.array([-90.0, -180.0], dtype=np.float32),
            high=np.array([90.0, 180.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )
        self.action_space = spaces.Dict(
            {
                "op": spaces.Discrete(2),
                "click": click_space,
                "answer": answer_space,
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
            "provider": self.provider,
            "pano_id": self.current_pano_id,
            "gt_lat": self.current_lat,
            "gt_lon": self.current_lon,
            "steps": self._steps,
            # Provide both yaw_deg (preferred) and heading_deg (compat) in degrees
            "pose": {
                "yaw_deg": math.degrees(self._heading_rad) % 360.0,
                "heading_deg": math.degrees(self._heading_rad) % 360.0,
            },
            "links": links_with_screen,
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        if seed is None and self._initial_seed is not None:
            seed = int(self._initial_seed)
        super().reset(seed=seed)

        # Sample coordinates from geofence if provided, otherwise use input coordinates
        if self.geofence and self.mode == "online":
            lat, lon = self._sample_from_geofence(seed)
        else:
            lat, lon = self.input_lat, self.input_lon

        # TODO[Karen]: improve pano fetching logic, don't fetch optimistically
        # TODO[Karen]: skip if malformed image appears
        self.pano_root_id = get_nearest_pano_id(lat, lon, self.metadata_dir)
        # Ensure directories exist
        os.makedirs(self.metadata_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        download_metadata(self.pano_root_id, self.metadata_dir)
        download_images(self.pano_root_id, self.metadata_dir, self.images_dir)

        # Now that metadata is available, load the graph
        metadata_path = os.path.join(
            self.metadata_dir, f"{self.pano_root_id}_mini.jsonl"
        )
        self._pano_graph = self._load_minimetadata(metadata_path)

        self._steps = 0
        # Initialize to fixed root pano
        self._set_current_pano(self.pano_root_id)
        # Initialize camera heading to the current pano heading if available
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
            print(f"CLICK: x: {x}, y: {y}")
            self._handle_click(x, y)
        elif op == 1:
            # answer: value is (lat, lon)
            guess_lat, guess_lon = value
            reward = self._compute_answer_reward(guess_lat, guess_lon)
            print(f"ANSWER: {guess_lat}, {guess_lon}")
            terminated = True

        # Truncation if exceeding max_steps (unless already terminated)
        if (
            not terminated
            and self.max_steps is not None
            and self._steps >= int(self.max_steps)
        ):
            truncated = True

        obs = self._get_observation()
        info = self._get_info()
        if op == 1:
            info["guess_lat"] = float(value[0])
            info["guess_lon"] = float(value[1])
            if self.current_lat is not None and self.current_lon is not None:
                distance_km = self._haversine_km(
                    float(self.current_lat),
                    float(self.current_lon),
                    float(value[0]),
                    float(value[1]),
                )
                info["distance_km"] = distance_km
            else:
                info["distance_km"] = float("inf")  # Unknown distance if no position
            info["score"] = reward
            info["steps"] = self._steps
        return obs, reward, terminated, truncated, info

    # --- Rendering ---
    def render(self, mode=None):
        # Allow override via argument; fall back to configured render_mode
        render_mode = mode if mode is not None else self.render_mode
        if render_mode == "rgb_array":
            return self._get_observation()["image"]
        if render_mode == "human":
            if self._screen is None:
                pygame.init()
                self._screen = pygame.display.set_mode(
                    (self._image_width, self._image_height)
                )
                pygame.display.set_caption("GeoGuessrEnv")
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

            frame = self._get_observation()["image"]
            # pygame expects (width, height, 3) and surfaces are transposed
            surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
            self._screen.blit(surf, (0, 0))

            # Overlay link centers and pano ids for human debugging only
            links = self._compute_link_screens()
            radius = int(self.arrow_hit_radius_px)
            for link in links:
                cx, cy = link["screen_xy"]
                # Draw filled red circle
                pygame.draw.circle(
                    self._screen, (255, 0, 0), (int(cx), int(cy)), radius, 0
                )
                # Draw white outline
                pygame.draw.circle(
                    self._screen, (255, 255, 255), (int(cx), int(cy)), radius, 2
                )
                # Draw pano id below (or above if near bottom)
                if self._font is not None:
                    text_surf = self._font.render(
                        str(link["id"]), True, (255, 255, 255)
                    )
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

    # --- Geofence sampling helpers ---
    def _sample_from_geofence(self, seed: Optional[int] = None) -> Tuple[float, float]:
        """
        Sample coordinates from the configured geofence using seeded RNG.

        Args:
            seed: Random seed for deterministic sampling

        Returns:
            Tuple of (latitude, longitude) within the geofence
        """
        if not self.geofence:
            raise ValueError("No geofence configured for sampling")

        # Use a separate random instance for geofence sampling to ensure determinism
        rng = random.Random(seed)

        geofence_type = self.geofence.get("type")
        if geofence_type == "circle":
            return self._sample_from_circular_geofence(self.geofence, rng)
        else:
            raise ValueError(f"Unsupported geofence type: {geofence_type}")

    def _sample_from_circular_geofence(
        self, geofence: Dict, rng: random.Random
    ) -> Tuple[float, float]:
        """
        Sample coordinates from a circular geofence.

        Args:
            geofence: Circular geofence configuration
            rng: Random number generator instance

        Returns:
            Tuple of (latitude, longitude) within the circle
        """
        center = geofence.get("center", {})
        center_lat = center.get("lat")
        center_lon = center.get("lon")
        radius_km = geofence.get("radius_km")

        if None in (center_lat, center_lon, radius_km):
            raise ValueError(
                "Invalid circular geofence: missing center lat/lon or radius_km"
            )

        # Sample a point uniformly within the circle
        # Use sqrt to get uniform distribution by area
        r = radius_km * math.sqrt(rng.random())
        theta = 2 * math.pi * rng.random()

        # Convert to lat/lon offset
        # Approximate conversion: 1 degree lat ≈ 111 km, 1 degree lon ≈ 111 km * cos(lat)
        lat_offset = (r * math.cos(theta)) / 111.0
        lon_offset = (r * math.sin(theta)) / (
            111.0 * math.cos(math.radians(center_lat))
        )

        sample_lat = center_lat + lat_offset
        sample_lon = center_lon + lon_offset

        # Clamp to valid ranges
        sample_lat = max(-90.0, min(90.0, sample_lat))
        sample_lon = max(-180.0, min(180.0, sample_lon))

        return sample_lat, sample_lon

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
                graph[pano_id] = {
                    "lat": lat,
                    "lon": lon,
                    "heading": heading,
                    "links": links,
                }
        # Prune links pointing to non-existent nodes
        valid_ids = set(graph.keys())
        for node in graph.values():
            raw_links = node.get("links", []) or []
            node["links"] = [
                link
                for link in raw_links
                if isinstance(link, dict)
                and isinstance(link.get("id"), str)
                and link["id"] in valid_ids
            ]
        return graph

    def _set_current_pano(self, pano_id: str) -> None:
        node = self._pano_graph.get(pano_id)
        if node is None:
            raise KeyError(f"Pano id not found in metadata: {pano_id}")
        self.current_pano_id = pano_id
        self.current_lat = (
            float(node.get("lat")) if node.get("lat") is not None else None
        )
        self.current_lon = (
            float(node.get("lon")) if node.get("lon") is not None else None
        )
        self.current_links = list(node.get("links", []))
        self._current_image = None  # force reload on next observation

    def _get_observation(self) -> Dict[str, np.ndarray]:
        if self._current_image is None:
            image_path = os.path.join(self.images_dir, f"{self.current_pano_id}.jpg")
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                np_img = np.array(img, dtype=np.uint8)
            self._current_image = np_img
        return {"image": self._current_image}

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
            x = self._direction_to_x(direction, float(pano_heading), self._image_width)
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
        for link in links:
            cx, cy = link["screen_xy"]
            dx = xi - int(cx)
            dy = yi - int(cy)
            link["_distance_px"] = math.hypot(dx, dy)

        # Filter by radius and confidence
        candidates = [
            link
            for link in links
            if (
                link["_distance_px"] is not None
                and link["_distance_px"] <= float(self.arrow_hit_radius_px)
                and float(link["conf"]) >= float(self.arrow_min_conf)
            )
        ]
        if not candidates:
            return  # no-op

        # Sort by distance, then smallest abs rel heading, then lexicographic pano id
        candidates.sort(
            key=lambda link: (
                float(link["_distance_px"]),
                float(link["_rel_heading_deg"]),
                str(link["id"]),
            )
        )
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
    def _direction_to_x(direction: float, heading: float, image_width: int) -> int:
        tau = getattr(math, "tau", 2 * math.pi)
        d = (direction + heading) % tau
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
        d_km = self._haversine_km(
            float(self.current_lat),
            float(self.current_lon),
            float(guess_lat),
            float(guess_lon),
        )
        score = math.exp(-d_km / 400.0)
        return float(score)

    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371.0
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = (
            math.sin(dphi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    # --- Action parsing ---
    def _parse_action(
        self, action: Union[Dict, str]
    ) -> Tuple[int, Tuple[float, float]]:
        """
        Returns a pair (op, value) where:
        - op: 0 for click, 1 for answer
        - value: (x, y) for click; (lat, lon) for answer
        Accepts multiple formats:
          - JSON string: '{"op":"click","value":[x,y]}' or '{"op":"answer","value":[lat,lon]}'
          - Dict with value: {"op": "click", "value": [x,y]}
          - Dict with explicit keys: {"op": "click", "click": [x,y], "answer": [lat,lon]}
        """
        # Handle JSON string input (from VLMBroker)
        if isinstance(action, str):
            try:
                action = json.loads(action.strip())
            except (json.JSONDecodeError, AttributeError):
                # Invalid JSON string, fallback to center click
                return 0, (
                    float(self._image_width // 2),
                    float(self._image_height // 2),
                )

        if isinstance(action, dict):
            op = action.get("op")
            if isinstance(op, str):
                op_norm = 0 if op == "click" else 1
            else:
                op_norm = int(op) if op is not None else 0

            # Prefer new explicit keys if present
            if (
                op_norm == 0
                and "click" in action
                and isinstance(action.get("click"), (list, tuple))
            ):
                val = action.get("click", [0, 0])
            elif (
                op_norm == 1
                and "answer" in action
                and isinstance(action.get("answer"), (list, tuple))
            ):
                val = action.get("answer", [0, 0])
            else:
                val = action.get("value", [0, 0])

            if isinstance(val, (list, tuple)) and len(val) == 2:
                if op_norm == 0:
                    # click expects ints
                    v0 = int(val[0])
                    v1 = int(val[1])
                else:
                    v0 = float(val[0])
                    v1 = float(val[1])
            else:
                if op_norm == 0:
                    v0, v1 = 0, 0
                else:
                    v0, v1 = 0.0, 0.0

            # Validate/clamp based on declared spaces
            if op_norm == 0:
                # click: bounds [0,1024] x [0,512] (note: should be exclusive upper bound)
                v0 = max(int(self._click_low[0]), min(int(self._click_high[0]) - 1, v0))
                v1 = max(int(self._click_low[1]), min(int(self._click_high[1]) - 1, v1))
            else:
                # answer: latitude/longitude bounds
                v0 = max(-90.0, min(90.0, v0))
                v1 = max(-180.0, min(180.0, v1))
            return op_norm, (v0, v1)

        # Fallback: treat as no-op click center
        return 0, (float(self._image_width // 2), float(self._image_height // 2))
