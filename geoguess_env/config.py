"""
Configuration management for the GeoGuessr environment.

This module provides structured configuration classes for the environment
with validation and type safety.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union


@dataclass
class GeofenceConfig:
    """Configuration for geographic boundaries."""

    type: str  # 'circle', 'polygon', etc.
    center: Optional[Dict[str, float]] = None  # {'lat': float, 'lon': float}
    radius_km: Optional[float] = None
    polygon: Optional[list] = None  # List of [lat, lon] points

    def __post_init__(self):
        """Validate geofence configuration."""
        if self.type == "circle":
            if not self.center or "lat" not in self.center or "lon" not in self.center:
                raise ValueError("Circular geofence requires center with lat/lon")
            if self.radius_km is None or self.radius_km <= 0:
                raise ValueError("Circular geofence requires positive radius_km")
        elif self.type == "polygon":
            if not self.polygon or len(self.polygon) < 3:
                raise ValueError("Polygon geofence requires at least 3 points")
        else:
            raise ValueError(f"Unsupported geofence type: {self.type}")


@dataclass
class ProviderConfig:
    """Configuration for panorama data providers."""

    provider: str = "google_streetview"
    rate_limit_qps: Optional[float] = None
    max_fetch_retries: int = 3
    min_capture_year: Optional[int] = None


@dataclass
class RenderConfig:
    """Configuration for environment rendering."""

    render_mode: Optional[str] = "rgb_array"
    image_width: int = 1024
    image_height: int = 512
    render_fps: int = 4


@dataclass
class NavigationConfig:
    """Configuration for navigation behavior."""

    arrow_hit_radius_px: int = 24
    arrow_min_conf: float = 0.0
    max_connected_panoramas: int = 8


@dataclass
class GeoGuessrConfig:
    """
    Complete configuration for the GeoGuessr environment.

    This dataclass provides type-safe configuration management with
    validation and sensible defaults for all environment parameters.
    """

    max_steps: int = 40
    seed: Optional[int] = None

    # Location specification
    input_lat: Optional[float] = None
    input_lon: Optional[float] = None
    geofence: Optional[GeofenceConfig] = None

    # Storage
    cache_root: Union[str, Path] = "cache"

    # Provider configuration
    provider_config: ProviderConfig = field(default_factory=ProviderConfig)

    # Rendering configuration
    render_config: RenderConfig = field(default_factory=RenderConfig)

    # Navigation configuration
    nav_config: NavigationConfig = field(default_factory=NavigationConfig)

    def __post_init__(self):
        """Validate and normalize configuration."""
        # Normalize cache_root to Path
        self.cache_root = Path(self.cache_root)

        # Validate coordinates if provided
        if self.input_lat is not None:
            if not (-90.0 <= self.input_lat <= 90.0):
                raise ValueError(f"Invalid latitude: {self.input_lat}")

        if self.input_lon is not None:
            if not (-180.0 <= self.input_lon <= 180.0):
                raise ValueError(f"Invalid longitude: {self.input_lon}")

        # Validate max_steps
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")

        # Validate image dimensions
        if self.render_config.image_width <= 0 or self.render_config.image_height <= 0:
            raise ValueError("Image dimensions must be positive")

        # Validate navigation parameters
        if self.nav_config.arrow_hit_radius_px <= 0:
            raise ValueError("arrow_hit_radius_px must be positive")

        if not (0.0 <= self.nav_config.arrow_min_conf <= 1.0):
            raise ValueError("arrow_min_conf must be between 0 and 1")

        if self.nav_config.max_connected_panoramas <= 0:
            raise ValueError("max_connected_panoramas must be positive")

    @classmethod
    def from_dict(cls, config_dict: Optional[Dict]) -> "GeoGuessrConfig":
        """
        Create configuration from dictionary.

        Args:
            config_dict: Configuration dictionary (can be None)

        Returns:
            GeoGuessrConfig instance
        """
        if not config_dict:
            return cls()

        config_data = dict(config_dict)

        allowed_top_level = {
            "max_steps",
            "seed",
            "input_lat",
            "input_lon",
            "geofence",
            "cache_root",
            "provider_config",
            "render_config",
            "nav_config",
        }
        ignored_keys = {"mode"}

        unexpected = set(config_data) - allowed_top_level - ignored_keys
        if unexpected:
            raise ValueError(
                "Unsupported configuration keys provided. "
                f"Remove legacy parameters: {sorted(unexpected)}"
            )

        # Extract nested configurations
        provider_config = ProviderConfig(**config_data.get("provider_config", {}))
        render_config = RenderConfig(**config_data.get("render_config", {}))
        nav_config = NavigationConfig(**config_data.get("nav_config", {}))

        # Handle geofence
        geofence = None
        if "geofence" in config_data:
            geofence_data = config_data["geofence"]
            if isinstance(geofence_data, GeofenceConfig):
                geofence = geofence_data
            elif geofence_data:
                geofence_params = {
                    key: value
                    for key, value in geofence_data.items()
                    if key in {"type", "center", "radius_km", "polygon"}
                }

                geofence_type = geofence_params.get("type")
                if not isinstance(geofence_type, str) or not geofence_type:
                    raise ValueError(
                        "Geofence configuration requires a string 'type' field"
                    )

                geofence = GeofenceConfig(
                    type=geofence_type,
                    center=geofence_params.get("center"),
                    radius_km=geofence_params.get("radius_km"),
                    polygon=geofence_params.get("polygon"),
                )

        # Create main config
        primary_keys = {"max_steps", "seed", "input_lat", "input_lon", "cache_root"}
        main_config = {
            key: config_data[key] for key in primary_keys if key in config_data
        }

        # Ignore "mode" for now if present without treating it as an error
        main_config.pop("mode", None)

        return cls(
            **main_config,
            geofence=geofence,
            provider_config=provider_config,
            render_config=render_config,
            nav_config=nav_config,
        )

    def to_dict(self) -> Dict:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        result = {
            "max_steps": self.max_steps,
            "seed": self.seed,
            "input_lat": self.input_lat,
            "input_lon": self.input_lon,
            "cache_root": str(self.cache_root),
        }

        if self.geofence:
            result["geofence"] = {
                "type": self.geofence.type,
                "center": self.geofence.center,
                "radius_km": self.geofence.radius_km,
                "polygon": self.geofence.polygon,
            }

        # Add nested configs
        result["provider_config"] = {
            "provider": self.provider_config.provider,
            "rate_limit_qps": self.provider_config.rate_limit_qps,
            "max_fetch_retries": self.provider_config.max_fetch_retries,
            "min_capture_year": self.provider_config.min_capture_year,
        }

        result["render_config"] = {
            "render_mode": self.render_config.render_mode,
            "image_width": self.render_config.image_width,
            "image_height": self.render_config.image_height,
            "render_fps": self.render_config.render_fps,
        }

        result["nav_config"] = {
            "arrow_hit_radius_px": self.nav_config.arrow_hit_radius_px,
            "arrow_min_conf": self.nav_config.arrow_min_conf,
            "max_connected_panoramas": self.nav_config.max_connected_panoramas,
        }

        return result

    @property
    def images_dir(self) -> Path:
        """Get images cache directory."""
        return self.cache_root / "images"

    @property
    def metadata_dir(self) -> Path:
        """Get metadata cache directory."""
        return self.cache_root / "metadata"

    @property
    def replays_dir(self) -> Path:
        """Get replays cache directory."""
        return self.cache_root / "replays"
