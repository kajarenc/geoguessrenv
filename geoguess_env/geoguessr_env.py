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

from .action_parser import ActionParser
from .asset_manager import AssetManager
from .config import GeoGuessrConfig
from .geometry_utils import GeometryUtils
from .providers.google_streetview import GoogleStreetViewProvider


class GeoGuessrEnv(gym.Env):
    """
    GeoGuessr-style Gymnasium environment for panorama navigation.

    This environment provides a street-level panorama navigation task where
    agents can click to navigate between connected panoramas and submit
    coordinate guesses for scoring.

    The environment uses a modular architecture with separate components for
    configuration management, asset loading, geometry calculations, and
    action parsing.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, config: Optional[Dict] = None, render_mode=None):
        """
        Initialize the GeoGuessr environment.

        Args:
            config: Environment configuration dictionary
            render_mode: Rendering mode ('human', 'rgb_array', or None)
        """
        # Parse and validate configuration
        self.config = GeoGuessrConfig.from_dict(config)

        # Override render mode if specified
        if render_mode is not None:
            self.config.render_config.render_mode = render_mode

        # Set up provider and asset manager
        self.provider = GoogleStreetViewProvider(
            rate_limit_qps=self.config.provider_config.rate_limit_qps,
            max_retries=self.config.provider_config.max_fetch_retries,
            min_capture_year=self.config.provider_config.min_capture_year,
        )

        self.asset_manager = AssetManager(
            provider=self.provider,
            cache_root=self.config.cache_root,
            max_connected_panoramas=self.config.nav_config.max_connected_panoramas,
        )

        # Set up action parser
        self.action_parser = ActionParser(
            image_width=self.config.render_config.image_width,
            image_height=self.config.render_config.image_height,
        )

        # Initialize environment state
        self.pano_root_id: Optional[str] = None
        self._pano_graph: Dict[str, Dict] = {}
        self._image_width = self.config.render_config.image_width
        self._image_height = self.config.render_config.image_height

        # Set up observation and action spaces
        self._setup_spaces()

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

    def _setup_spaces(self):
        """
        Set up observation and action spaces based on configuration.
        """
        # Observation space: dictionary with image
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

        # Action space: click and answer operations
        click_space = spaces.Box(
            low=np.array([0, 0], dtype=np.int32),
            high=np.array([self._image_width, self._image_height], dtype=np.int32),
            shape=(2,),
            dtype=np.int32,
        )
        # Store click bounds for action validation
        self._click_low = [0, 0]
        self._click_high = [self._image_width, self._image_height]
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

    def _get_info(self):
        """
        Get environment info dictionary.
        """
        links_with_screen = self._compute_link_screens()
        return {
            "provider": self.provider.provider_name,
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
        """
        Reset the environment to start a new episode.

        Args:
            seed: Random seed for episode generation
            options: Additional reset options (unused)

        Returns:
            Tuple of (observation, info)
        """
        # Initialize random number generator
        if seed is None and self.config.seed is not None:
            seed = self.config.seed
        super().reset(seed=seed)

        # Determine starting coordinates with retry logic for geofence sampling
        if self.config.geofence and self.config.mode == "online":
            lat, lon = self._sample_valid_coordinates_from_geofence(seed)
        else:
            lat, lon = self.config.input_lat, self.config.input_lon

        if lat is None or lon is None:
            raise ValueError(
                "No starting coordinates provided (either input_lat/lon or geofence required)"
            )

        # Get or fetch panorama graph using asset manager
        # Round coordinates to 6 decimal places for consistent cache handling
        lat = round(lat, 6)
        lon = round(lon, 6)
        offline_mode = self.config.mode == "offline"
        try:
            self._pano_graph = self.asset_manager.get_or_fetch_panorama_graph(
                root_lat=lat, root_lon=lon, offline_mode=offline_mode
            )
        except Exception as e:
            raise ValueError(f"Failed to load panorama data for {lat}, {lon}: {e}")

        if not self._pano_graph:
            raise ValueError(
                f"No panorama graph available for coordinates {lat}, {lon}"
            )

        # Find root panorama ID (first key in graph)
        self.pano_root_id = next(iter(self._pano_graph.keys()))

        # Preload images for all panoramas in the graph to ensure smooth navigation
        if not offline_mode:
            print(f"Prefetching images for {len(self._pano_graph)} panoramas...")
            pano_ids = set(self._pano_graph.keys())
            preload_results = self.asset_manager.preload_assets(
                pano_ids, skip_existing=True
            )
            successful_loads = sum(1 for success in preload_results.values() if success)
            print(
                f"Successfully prefetched {successful_loads}/{len(pano_ids)} panorama images"
            )

        # Reset episode state
        self._steps = 0
        self._set_current_pano(self.pano_root_id)

        # Initialize camera heading
        node = self._pano_graph.get(self.current_pano_id, {})
        heading = node.get("heading", 0.0)
        self._heading_rad = (
            math.radians(heading) if isinstance(heading, (int, float)) else 0.0
        )

        # Get initial observation and info
        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action):
        """
        Execute one environment step.

        Args:
            action: Action to execute (click or answer)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self._steps += 1

        # Parse action using robust action parser
        try:
            op, values = self.action_parser.parse_action(action)
        except Exception as e:
            print(f"Action parsing failed: {e}, using fallback")
            op, values = self.action_parser.parse_with_fallback(action)

        terminated = False
        truncated = False
        reward = 0.0

        if op == 0:
            # Click action: navigate to link
            x, y = values
            print(f"CLICK: x: {x}, y: {y}")
            self._handle_click(x, y)
        elif op == 1:
            # Answer action: submit coordinate guess
            guess_lat, guess_lon = values
            reward = GeometryUtils.compute_answer_reward(
                guess_lat, guess_lon, self.current_lat or 0.0, self.current_lon or 0.0
            )
            print(f"ANSWER: {guess_lat}, {guess_lon} (reward: {reward:.4f})")
            terminated = True

        # Check for truncation due to max steps
        if not terminated and self._steps >= self.config.max_steps:
            truncated = True

        # Get observation and info
        obs = self._get_observation()
        info = self._get_info()

        # Add answer-specific info
        if op == 1:
            info["guess_lat"] = float(values[0])
            info["guess_lon"] = float(values[1])

            if self.current_lat is not None and self.current_lon is not None:
                distance_km = GeometryUtils.haversine_distance(
                    self.current_lat,
                    self.current_lon,
                    float(values[0]),
                    float(values[1]),
                )
                info["distance_km"] = distance_km
            else:
                info["distance_km"] = float("inf")

            info["score"] = reward

        return obs, reward, terminated, truncated, info

    # --- Rendering ---
    def render(self, mode=None):
        # Allow override via argument; fall back to configured render_mode
        render_mode = (
            mode if mode is not None else self.config.render_config.render_mode
        )
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
            radius = int(self.config.nav_config.arrow_hit_radius_px)
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
        Sample coordinates from the configured geofence using GeometryUtils.

        Args:
            seed: Random seed for deterministic sampling

        Returns:
            Tuple of (latitude, longitude) within the geofence
        """
        if not self.config.geofence:
            raise ValueError("No geofence configured for sampling")

        # Use a separate random instance for geofence sampling to ensure determinism
        rng = random.Random(seed)

        geofence = self.config.geofence
        if geofence.type == "circle":
            return GeometryUtils.sample_circular_geofence(
                center_lat=geofence.center["lat"],
                center_lon=geofence.center["lon"],
                radius_km=geofence.radius_km,
                rng=rng,
            )
        elif geofence.type == "polygon":
            return GeometryUtils.sample_polygon_geofence(
                polygon=[(p[0], p[1]) for p in geofence.polygon], rng=rng
            )
        else:
            raise ValueError(f"Unsupported geofence type: {geofence.type}")

    def _sample_valid_coordinates_from_geofence(
        self, seed: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Sample coordinates from geofence with retry logic to find locations with panoramas.

        Args:
            seed: Random seed for deterministic sampling

        Returns:
            Tuple of (latitude, longitude) where a panorama exists

        Raises:
            ValueError: If no valid coordinates found after max attempts
        """
        max_attempts = 10

        for attempt in range(max_attempts):
            # Use the attempt number to vary the seed for each retry
            attempt_seed = (seed + attempt) if seed is not None else attempt
            lat, lon = self._sample_from_geofence(attempt_seed)

            # Round coordinates to check if panorama exists
            rounded_lat = round(lat, 6)
            rounded_lon = round(lon, 6)

            # Check if we can find a panorama at this location
            try:
                pano_id = self.asset_manager._get_or_find_nearest_panorama(
                    rounded_lat, rounded_lon, offline_mode=False
                )
                if pano_id:
                    print(
                        f"Found valid coordinates after {attempt + 1} attempts: {rounded_lat}, {rounded_lon}"
                    )
                    return rounded_lat, rounded_lon
            except Exception as e:
                print(
                    f"Attempt {attempt + 1}: No panorama at {rounded_lat}, {rounded_lon}: {e}"
                )

        raise ValueError(
            f"Could not find valid coordinates with panoramas after {max_attempts} attempts"
        )

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
        """
        Get current observation (panorama image).
        """
        if self._current_image is None:
            # Try to get image from asset manager first
            asset = self.asset_manager.get_panorama_asset(self.current_pano_id)
            if asset and asset.image_path.exists():
                with Image.open(asset.image_path) as img:
                    img = img.convert("RGB")
                    self._current_image = np.array(img, dtype=np.uint8)
            else:
                # Fallback to direct file access for backward compatibility
                image_path = self.config.images_dir / f"{self.current_pano_id}.jpg"
                if image_path.exists():
                    with Image.open(image_path) as img:
                        img = img.convert("RGB")
                        self._current_image = np.array(img, dtype=np.uint8)
                else:
                    # Create a placeholder image if none found
                    self._current_image = np.zeros(
                        (self._image_height, self._image_width, 3), dtype=np.uint8
                    )

        return {"image": self._current_image}

    # --- Click handling and link mapping ---
    def _compute_link_screens(self) -> List[Dict[str, object]]:
        """
        Compute screen-space centers for current links using GeometryUtils.
        """
        node = self._pano_graph.get(self.current_pano_id, {})
        pano_heading = node.get("heading", 0.0)
        current_heading_deg = math.degrees(self._heading_rad)

        return GeometryUtils.compute_link_screen_positions(
            links=self.current_links,
            pano_heading=pano_heading,
            current_heading=current_heading_deg,
            image_width=self._image_width,
            image_height=self._image_height,
        )

    def _handle_click(self, x: float, y: float) -> None:
        """
        Handle click action and navigate to appropriate link if found.
        """
        # Get screen links
        screen_links = self._compute_link_screens()
        if not screen_links:
            return

        # Find clicked link using GeometryUtils
        clicked_link = GeometryUtils.find_clicked_link(
            click_x=int(round(x)),
            click_y=int(round(y)),
            screen_links=screen_links,
            hit_radius=self.config.nav_config.arrow_hit_radius_px,
            min_confidence=self.config.nav_config.arrow_min_conf,
        )

        if clicked_link is None:
            return  # No valid link clicked

        # Navigate to the selected link
        next_id = clicked_link["id"]

        # Check if the target pano exists in the graph before navigating
        if next_id not in self._pano_graph:
            # Skip navigation if target pano is not loaded in the graph
            return

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
