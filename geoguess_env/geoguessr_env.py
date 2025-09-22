import logging
import math
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

from .action_parser import ActionParser
from .asset_manager import AssetManager
from .config import GeoGuessrConfig
from .geometry_utils import GeometryUtils
from .providers.google_streetview import GoogleStreetViewProvider

logger = logging.getLogger(__name__)


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
        self._episode_seed: Optional[int] = None
        self._episode_rng: Optional[np.random.Generator] = None

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
        # Capture the RNG configured by the base class for deterministic sampling
        self._episode_seed = self.np_random_seed
        self._episode_rng = self.np_random

        # Determine starting coordinates with retry logic for geofence sampling
        if self.config.geofence:
            lat, lon = self._sample_valid_coordinates_from_geofence()
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
        try:
            graph_result = self.asset_manager.prepare_graph(root_lat=lat, root_lon=lon)
        except Exception as e:
            raise ValueError(f"Failed to load panorama data for {lat}, {lon}: {e}")

        if not graph_result.graph:
            raise ValueError(
                f"No panorama graph available for coordinates {lat}, {lon}"
            )

        if graph_result.missing_assets:
            missing = ", ".join(sorted(graph_result.missing_assets))
            raise ValueError(
                f"Missing cached assets for panoramas ({missing}) at {lat}, {lon}"
            )

        # Store prepared graph
        self._pano_graph = graph_result.graph

        # Find root panorama ID (first key in graph)
        self.pano_root_id = graph_result.root_id

        root_metadata = self._pano_graph.get(self.pano_root_id, {})
        root_date = root_metadata.get("date")
        logger.info(
            "Starting episode at root pano %s (capture date: %s)",
            self.pano_root_id,
            root_date or "unknown",
        )

        # Verify image availability once to enforce runtime invariant
        for pano_id in self._pano_graph.keys():
            if self.asset_manager.get_image_array(pano_id) is None:
                raise RuntimeError(
                    f"Panorama image missing from cache for {pano_id} after preparation"
                )

        # Reset episode state
        self._steps = 0
        self._set_current_pano(self.pano_root_id)

        # Initialize camera heading
        node = self._pano_graph.get(self.current_pano_id, {})
        heading = node.get("heading", 0.0)
        self._heading_rad = float(heading) if isinstance(heading, (int, float)) else 0.0

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
    def _get_episode_rng(self) -> np.random.Generator:
        if self._episode_rng is None:
            self._episode_rng = self.np_random
        return self._episode_rng

    def _sample_from_geofence(self) -> Tuple[float, float]:
        """Sample coordinates from the configured geofence using the episode RNG."""
        if not self.config.geofence:
            raise ValueError("No geofence configured for sampling")

        rng = self._get_episode_rng()

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

    def _sample_valid_coordinates_from_geofence(self) -> Tuple[float, float]:
        """Sample geofence coordinates with retries until a cached panorama is found.

        Raises:
            ValueError: If no valid coordinates found after max attempts
        """
        max_attempts = 10

        for attempt in range(max_attempts):
            lat, lon = self._sample_from_geofence()

            # Round coordinates to check if panorama exists
            rounded_lat = round(lat, 6)
            rounded_lon = round(lon, 6)

            # Check if we can find a panorama at this location
            try:
                pano_id = self.asset_manager.resolve_nearest_panorama(
                    rounded_lat, rounded_lon
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
            image_array = self.asset_manager.get_image_array(self.current_pano_id)
            if image_array is None:
                raise RuntimeError(
                    f"Missing panorama image for {self.current_pano_id} during observation"
                )

            # Copy to decouple environment state from the shared cache
            self._current_image = np.array(image_array, copy=True)

        return {"image": self._current_image}

    # --- Click handling and link mapping ---
    def _compute_link_screens(self) -> List[Dict[str, object]]:
        """
        Compute screen-space centers for current links using GeometryUtils.
        """
        node = self._pano_graph.get(self.current_pano_id, {})
        pano_heading = node.get("heading", 0.0)
        current_heading_rad = self._heading_rad
        # Normalize link directions: metadata stores radians; pass through as radians.
        normalized_links: List[Dict[str, object]] = []
        for link in self.current_links or []:
            try:
                raw_dir = float(link.get("direction", 0.0))
            except Exception:
                raw_dir = 0.0

            # Direction is provided in radians in metadata; no conversion here.
            direction_rad = float(raw_dir)
            normalized_links.append(
                {
                    "id": link.get("id"),
                    "direction": direction_rad,
                }
            )

        return GeometryUtils.compute_link_screen_positions(
            links=normalized_links,
            pano_heading_rad=float(pano_heading)
            if isinstance(pano_heading, (int, float))
            else 0.0,
            current_heading_rad=current_heading_rad,
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
        # Update camera heading to the new pano's stored heading (already radians)
        node = self._pano_graph.get(self.current_pano_id, {})
        heading = node.get("heading")
        if isinstance(heading, (int, float)):
            self._heading_rad = float(heading)
