"""
Configuration management for the GeoGuessr environment.

This module provides structured configuration classes for the environment
with validation and type safety.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union, cast


@dataclass
class GeofenceConfig:
    """Configuration for geographic boundaries."""

    type: str  # 'circle', 'polygon', etc.
    center: Optional[Dict[str, float]] = None  # {'lat': float, 'lon': float}
    radius_km: Optional[float] = None
    polygon: Optional[List[List[float]]] = None  # List of [lat, lon] points

    def __post_init__(self) -> None:
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

    def __post_init__(self) -> None:
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
    def from_dict(cls, config_dict: Mapping[str, object] | None) -> "GeoGuessrConfig":
        """
        Create configuration from dictionary.

        Args:
            config_dict: Configuration mapping (can be None)

        Returns:
            GeoGuessrConfig instance
        """
        if not config_dict:
            return cls()

        config_data: Dict[str, object] = dict(config_dict)

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
        provider_config = ProviderConfig(
            **_extract_mapping(config_data, "provider_config")
        )
        render_config = RenderConfig(**_extract_mapping(config_data, "render_config"))
        nav_config = NavigationConfig(**_extract_mapping(config_data, "nav_config"))

        # Handle geofence
        geofence = None
        if "geofence" in config_data:
            geofence_data = config_data["geofence"]
            if isinstance(geofence_data, GeofenceConfig):
                geofence = geofence_data
            elif isinstance(geofence_data, Mapping):
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
                    center=_coerce_center(geofence_params.get("center")),
                    radius_km=_coerce_optional_float(geofence_params.get("radius_km")),
                    polygon=_coerce_polygon(geofence_params.get("polygon")),
                )
            elif geofence_data is not None:
                raise ValueError("Geofence configuration must be a mapping")

        # Create main config
        primary_keys = {"max_steps", "seed", "input_lat", "input_lon", "cache_root"}
        main_config = {
            key: config_data[key] for key in primary_keys if key in config_data
        }
        main_config.pop("mode", None)

        return cls(
            **main_config,
            geofence=geofence,
            provider_config=provider_config,
            render_config=render_config,
            nav_config=nav_config,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        result: Dict[str, Any] = {
            "max_steps": self.max_steps,
            "seed": self.seed,
            "input_lat": self.input_lat,
            "input_lon": self.input_lon,
            "cache_root": str(self.cache_root),
        }

        if self.geofence:
            result["geofence"] = {
                "type": self.geofence.type,
                "center": dict(self.geofence.center) if self.geofence.center else None,
                "radius_km": self.geofence.radius_km,
                "polygon": self.geofence.polygon,
            }

        # Add nested configs
        result["provider_config"] = {
            "provider": self.provider_config.provider,
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
        return Path(self.cache_root) / "images"

    @property
    def metadata_dir(self) -> Path:
        """Get metadata cache directory."""
        return Path(self.cache_root) / "metadata"

    @property
    def replays_dir(self) -> Path:
        """Get replays cache directory."""
        return Path(self.cache_root) / "replays"


def _extract_mapping(source: Mapping[str, object], key: str) -> Dict[str, object]:
    """Extract nested mapping from top-level configuration."""

    value = source.get(key)
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    raise ValueError(f"Configuration field '{key}' must be a mapping")


def _coerce_center(value: object) -> Dict[str, float] | None:
    """Coerce optional geofence center mapping to floats."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("Geofence center must be a mapping with 'lat' and 'lon'")

    mapping_value = cast(Mapping[str, object], value)

    if "lat" not in mapping_value or "lon" not in mapping_value:
        raise ValueError("Circular geofence requires center with lat/lon")

    lat_raw = mapping_value.get("lat")
    lon_raw = mapping_value.get("lon")
    if not isinstance(lat_raw, (int, float)) or not isinstance(lon_raw, (int, float)):
        raise ValueError("Geofence center coordinates must be numeric")

    return {"lat": float(lat_raw), "lon": float(lon_raw)}


def _coerce_optional_float(value: object) -> float | None:
    """Cast optional numeric configuration values to float."""

    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError("Geofence radius must be numeric")


def _coerce_polygon(value: object) -> List[List[float]] | None:
    """Validate optional polygon payload."""

    if value is None:
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        polygon: List[List[float]] = []
        for point in value:
            if not isinstance(point, Sequence) or len(point) < 2:
                raise ValueError("Polygon points must be sequences of two numbers")
            lat, lon = point[0], point[1]
            if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
                raise ValueError("Polygon coordinates must be numeric")
            polygon.append([float(lat), float(lon)])
        return polygon
    raise ValueError("Polygon must be a sequence of [lat, lon] pairs")
