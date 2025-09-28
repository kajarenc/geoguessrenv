import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

from .action_parser import ActionParser
from .asset_manager import (
    AssetManager,
    PanoramaGraphResult,
    RootPanoramaUnavailableError,
)
from .config import GeoGuessrConfig
from .geometry_utils import GeometryUtils
from .providers.google_streetview import GoogleStreetViewProvider

logger = logging.getLogger(__name__)


class GeoGuessrEnv(gym.Env[Dict[str, np.ndarray], Dict[str, Any]]):
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

    def __init__(
        self, config: Optional[Dict[str, Any]] = None, render_mode: Optional[str] = None
    ) -> None:
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
            cache_root=Path(self.config.cache_root),
            max_connected_panoramas=self.config.nav_config.max_connected_panoramas,
        )

        # Set up action parser
        self.action_parser = ActionParser(
            image_width=self.config.render_config.image_width,
            image_height=self.config.render_config.image_height,
        )

        # Initialize environment state
        self.pano_root_id: Optional[str] = None
        self._pano_graph: Dict[str, Dict[str, Any]] = {}
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

    def _setup_spaces(self) -> None:
        """Configure observation and action spaces based on render settings."""

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

        click_space = spaces.Box(
            low=np.array([0, 0], dtype=np.int32),
            high=np.array([self._image_width, self._image_height], dtype=np.int32),
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

    def _get_info(self) -> Dict[str, Any]:
        """Assemble the metadata dictionary returned alongside observations."""

        heading_deg = math.degrees(self._heading_rad) % 360.0
        return {
            "provider": self.provider.provider_name,
            "pano_id": self.current_pano_id,
            "gt_lat": self.current_lat,
            "gt_lon": self.current_lon,
            "steps": self._steps,
            "pose": {"yaw_deg": heading_deg, "heading_deg": heading_deg},
            "links": self._compute_link_screens(),
        }

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment to start a fresh navigation episode."""

        if seed is None and self.config.seed is not None:
            seed = self.config.seed
        super().reset(seed=seed, options=options)
        self._capture_episode_rng()

        graph_result = self._prepare_graph_with_retries()
        self._pano_graph = graph_result.graph
        self.pano_root_id = graph_result.root_id

        root_metadata = self._pano_graph.get(self.pano_root_id, {})
        root_date = root_metadata.get("date")
        logger.info(
            "Starting episode at root pano %s (capture date: %s)",
            self.pano_root_id,
            root_date or "unknown",
        )

        self._validate_graph_assets()

        self._steps = 0
        self._set_current_pano(self.pano_root_id)
        self._heading_rad = self._get_heading_for_pano(self.current_pano_id)

        return self._get_observation(), self._get_info()

    def _capture_episode_rng(self) -> None:
        """Persist the RNG set up by the Gymnasium base class for later use."""

        self._episode_seed = self.np_random_seed
        self._episode_rng = self.np_random

    def _prepare_graph_with_retries(self) -> PanoramaGraphResult:
        """Load the panorama graph, retrying when the root panorama is unavailable."""

        max_attempts = 5
        last_unavailable_error: Optional[RootPanoramaUnavailableError] = None

        for attempt in range(max_attempts):
            lat, lon = self._select_start_coordinates(attempt)
            lat = round(lat, 6)
            lon = round(lon, 6)

            try:
                graph_result = self.asset_manager.prepare_graph(
                    root_lat=lat, root_lon=lon
                )
            except RootPanoramaUnavailableError as exc:
                last_unavailable_error = exc
                logger.warning(
                    "Attempt %d/%d: Root panorama unavailable at (%f, %f): %s",
                    attempt + 1,
                    max_attempts,
                    lat,
                    lon,
                    exc,
                )
                continue
            except Exception as exc:  # pragma: no cover - defensive guardrail
                raise ValueError(
                    f"Failed to load panorama data for {lat}, {lon}: {exc}"
                ) from exc

            if not graph_result.graph:
                raise ValueError(
                    f"No panorama graph available for coordinates {lat}, {lon}"
                )

            if graph_result.missing_assets:
                missing = ", ".join(sorted(graph_result.missing_assets))
                raise ValueError(
                    f"Missing cached assets for panoramas ({missing}) at {lat}, {lon}"
                )

            return graph_result

        if last_unavailable_error is not None:
            raise ValueError(
                f"Failed to find valid panorama location after {max_attempts} attempts"
            ) from last_unavailable_error

        raise ValueError(
            f"Failed to prepare panorama graph after {max_attempts} attempts"
        )

    def _select_start_coordinates(self, attempt: int) -> Tuple[float, float]:
        """Determine starting coordinates for the episode."""

        if self.config.geofence:
            return self._sample_valid_coordinates_from_geofence()

        if attempt > 0:
            raise ValueError(
                f"Failed to load panorama data for {self.config.input_lat}, {self.config.input_lon}: "
                "Root panorama unavailable and no geofence configured for retry"
            )

        lat = self.config.input_lat
        lon = self.config.input_lon
        if lat is None or lon is None:
            raise ValueError(
                "No starting coordinates provided (either input_lat/lon or geofence required)"
            )

        return float(lat), float(lon)

    def _validate_graph_assets(self) -> None:
        """Ensure images for the prepared graph are present in the cache."""

        for pano_id in self._pano_graph:
            if self.asset_manager.get_image_array(pano_id) is None:
                raise RuntimeError(
                    f"Panorama image missing from cache for {pano_id} after preparation"
                )

    def _get_heading_for_pano(self, pano_id: Optional[str]) -> float:
        """Return the stored heading for the given panorama, defaulting to zero."""

        if pano_id is None:
            return 0.0

        heading = self._pano_graph.get(pano_id, {}).get("heading", 0.0)
        return float(heading) if isinstance(heading, (int, float)) else 0.0

    def step(
        self, action: Dict[str, Any]
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute one environment step using a click or answer action."""

        self._steps += 1

        try:
            op, values = self.action_parser.parse_action(action)
        except Exception as exc:  # pragma: no cover - defensive guardrail
            logger.warning("Action parsing failed (%s); using fallback parser", exc)
            op, values = self.action_parser.parse_with_fallback(action)

        terminated = False
        truncated = False
        reward = 0.0

        if op == 0:
            x, y = values
            logger.debug("Handling click at (%.2f, %.2f)", x, y)
            self._handle_click(x, y)
        elif op == 1:
            guess_lat, guess_lon = values
            reward = GeometryUtils.compute_answer_reward(
                guess_lat, guess_lon, self.current_lat or 0.0, self.current_lon or 0.0
            )
            logger.debug(
                "Answer submitted lat=%.6f lon=%.6f reward=%.4f",
                guess_lat,
                guess_lon,
                reward,
            )
            terminated = True
        else:
            logger.error("Received unsupported op code: %s", op)

        if not terminated and self._steps >= self.config.max_steps:
            truncated = True

        obs = self._get_observation()
        info = self._get_info()

        if op == 1:
            guess_lat = float(values[0])
            guess_lon = float(values[1])
            info["guess_lat"] = guess_lat
            info["guess_lon"] = guess_lon

            if self.current_lat is not None and self.current_lon is not None:
                info["distance_km"] = GeometryUtils.haversine_distance(
                    self.current_lat,
                    self.current_lon,
                    guess_lat,
                    guess_lon,
                )
            else:
                info["distance_km"] = math.inf

            info["score"] = reward

        return obs, reward, terminated, truncated, info

    # --- Rendering ---
    def render(self, mode: Optional[str] = None) -> Optional[np.ndarray]:
        """Render the current frame either for human viewing or as an RGB array."""

        render_mode = mode or self.config.render_config.render_mode
        if render_mode == "rgb_array":
            return self._get_observation()["image"]

        if render_mode != "human":
            return None

        self._initialize_human_renderer()
        pygame.event.pump()

        if self._screen is None:
            logger.warning("Human renderer surface unavailable; skipping frame draw")
            return None

        frame = self._get_observation()["image"]
        surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        self._screen.blit(surface, (0, 0))

        radius = int(self.config.nav_config.arrow_hit_radius_px)
        for link in self._compute_link_screens():
            cx, cy = map(int, link["screen_xy"])
            pygame.draw.circle(self._screen, (255, 0, 0), (cx, cy), radius, 0)
            pygame.draw.circle(self._screen, (255, 255, 255), (cx, cy), radius, 2)
            if self._font is None:
                continue
            text_surface = self._font.render(str(link["id"]), True, (255, 255, 255))
            text_rect = text_surface.get_rect()
            text_rect.centerx = cx
            text_rect.top = cy + radius + 4
            if text_rect.bottom > self._image_height:
                text_rect.bottom = cy - radius - 4
                text_rect.top = text_rect.bottom - text_rect.height
            shadow_rect = text_rect.copy()
            shadow_rect.x += 1
            shadow_rect.y += 1
            self._screen.blit(text_surface, shadow_rect)
            self._screen.blit(text_surface, text_rect)

        pygame.display.flip()
        if self._clock is not None:
            self._clock.tick(self.metadata.get("render_fps", 4))

        return None

    def _initialize_human_renderer(self) -> None:
        """Lazy-initialize pygame surfaces required for human rendering."""

        if self._screen is not None:
            return

        pygame.init()
        self._screen = pygame.display.set_mode((self._image_width, self._image_height))
        pygame.display.set_caption("GeoGuessrEnv")
        self._clock = pygame.time.Clock()
        if self._font is None:
            try:
                self._font = pygame.font.SysFont("Arial", 14)
            except Exception:  # pragma: no cover - fallback for headless systems
                self._font = pygame.font.Font(None, 14)

    def close(self) -> None:
        if self._screen is not None:
            try:
                pygame.display.quit()
                pygame.quit()
            except Exception:  # pragma: no cover - defensive guardrail
                pass
            self._screen = None
            self._clock = None
            self._font = None

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
            if geofence.center is None:
                raise ValueError(
                    "Circle geofence requires a center with 'lat' and 'lon' values"
                )
            center_lat, center_lon = self._normalize_lat_lon(
                geofence.center,
                "Circle geofence center must provide 'lat' and 'lon' values",
            )
            radius_km = geofence.radius_km
            if radius_km is None:
                raise ValueError("Circle geofence requires a 'radius_km' value")
            return GeometryUtils.sample_circular_geofence(
                center_lat=center_lat,
                center_lon=center_lon,
                radius_km=float(radius_km),
                rng=rng,
            )

        if geofence.type == "polygon":
            polygon_points = geofence.polygon
            if not polygon_points:
                raise ValueError("Polygon geofence requires a non-empty 'polygon' list")
            normalized_polygon = [
                self._normalize_lat_lon(
                    point, "Polygon geofence points must provide 'lat' and 'lon' values"
                )
                for point in polygon_points
            ]
            if len(normalized_polygon) < 3:
                raise ValueError("Polygon geofence requires at least three points")
            return GeometryUtils.sample_polygon_geofence(
                polygon=normalized_polygon, rng=rng
            )

        raise ValueError(f"Unsupported geofence type: {geofence.type}")

    def _sample_valid_coordinates_from_geofence(self) -> Tuple[float, float]:
        """Sample geofence coordinates with retries until a cached panorama is found."""

        max_attempts = 10

        for attempt in range(max_attempts):
            lat, lon = self._sample_from_geofence()
            rounded_lat = round(lat, 6)
            rounded_lon = round(lon, 6)

            try:
                pano_id = self.asset_manager.resolve_nearest_panorama(
                    rounded_lat, rounded_lon
                )
                if pano_id:
                    logger.debug(
                        "Found valid coordinates after %d attempts: %.6f, %.6f",
                        attempt + 1,
                        rounded_lat,
                        rounded_lon,
                    )
                    return rounded_lat, rounded_lon
            except Exception as exc:  # pragma: no cover - defensive guardrail
                logger.debug(
                    "Attempt %d: no panorama at %.6f, %.6f (%s)",
                    attempt + 1,
                    rounded_lat,
                    rounded_lon,
                    exc,
                )

        raise ValueError(
            f"Could not find valid coordinates with panoramas after {max_attempts} attempts"
        )

    @staticmethod
    def _normalize_lat_lon(value: Any, error_message: str) -> Tuple[float, float]:
        """Extract latitude and longitude from supported container types."""

        if isinstance(value, dict):
            lat = value.get("lat")
            lon = value.get("lon")
        elif isinstance(value, (tuple, list)) and len(value) >= 2:
            lat, lon = value[0], value[1]
        else:
            lat = getattr(value, "lat", None)
            lon = getattr(value, "lon", None)

        if lat is None or lon is None:
            raise ValueError(error_message)

        return float(lat), float(lon)

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
        current_id = self.current_pano_id

        if self._current_image is None:
            if current_id is None:
                raise RuntimeError(
                    "Missing panorama image and no current pano ID to reload it"
                )

            image_array = self.asset_manager.get_image_array(current_id)
            if image_array is None:
                raise RuntimeError(
                    f"Missing panorama image for {current_id} during observation"
                )

            # Copy to decouple environment state from the shared cache
            self._current_image = np.array(image_array, copy=True)

        return {"image": self._current_image}

    # --- Click handling and link mapping ---
    def _compute_link_screens(self) -> List[Dict[str, Any]]:
        """
        Compute screen-space centers for current links using GeometryUtils.
        """
        current_pano_id = self.current_pano_id
        if current_pano_id is None:
            return []

        normalized_links: List[Dict[str, Any]] = []
        for link in self.current_links or []:
            direction_value = link.get("direction", 0.0)
            try:
                direction_rad = float(direction_value)
            except (TypeError, ValueError):
                direction_rad = 0.0
            normalized_links.append({"id": link.get("id"), "direction": direction_rad})

        return GeometryUtils.compute_link_screen_positions(
            links=normalized_links,
            pano_heading_rad=self._get_heading_for_pano(current_pano_id),
            current_heading_rad=self._heading_rad,
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
            logger.debug("Click ignored because no outgoing links are available")
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
            logger.debug("Click at (%.2f, %.2f) did not hit a link", x, y)
            return  # No valid link clicked

        # Navigate to the selected link
        next_id = clicked_link.get("id")
        if not next_id:
            logger.debug("Clicked link missing identifier; ignoring")
            return

        # Check if the target pano exists in the graph before navigating
        if next_id not in self._pano_graph:
            # Skip navigation if target pano is not loaded in the graph
            logger.debug("Link %s not present in preloaded graph; ignoring", next_id)
            return

        # Move to neighbor pano
        self._set_current_pano(next_id)
        # Update camera heading to the new pano's stored heading (already radians)
        current_id = self.current_pano_id
        if current_id is None:
            raise RuntimeError("Current pano ID unset after navigation")
        node = self._pano_graph.get(current_id, {})
        heading = node.get("heading")
        if isinstance(heading, (int, float)):
            self._heading_rad = float(heading)
