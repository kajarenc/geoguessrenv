"""
Action parsing utilities for the GeoGuessr environment.

Simplified to a single action format per TaskDescription.md:

- Click:  {"op": "click", "value": [x, y]}
- Answer: {"op": "answer", "value": [lat, lon]}

Coordinates are validated and clamped. On parsing failure, a safe
center click is returned by the fallback API.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, cast

from .geometry_utils import GeometryUtils
from .types import (
    ActionCode,
    ActionPayload,
    AnswerAction,
    ClickAction,
    FloatPair,
    NumberLike,
)


class ActionParsingError(Exception):
    """Exception raised when action parsing fails."""

    pass


class ActionParser:
    """Single-format action parser for the GeoGuessr environment."""

    def __init__(self, image_width: int, image_height: int) -> None:
        """
        Initialize the action parser.

        Args:
            image_width: Width of the panorama image
            image_height: Height of the panorama image
        """
        self.image_width = image_width
        self.image_height = image_height

    def parse_action(
        self, action: ActionPayload | Mapping[str, object] | str
    ) -> tuple[ActionCode, FloatPair]:
        """
        Parse action in the single supported format.

        Supported inputs:
        - JSON string: '{"op":"click","value":[x,y]}' or '{"op":"answer","value":[lat,lon]}'
        - Mapping: {"op": "click"|"answer", "value": [..,..]}

        Returns (op_code, values): op_code is 0 for click, 1 for answer.
        """
        try:
            action_mapping = self._normalize_action(action)

            op_raw = action_mapping.get("op")
            if not isinstance(op_raw, str):
                raise ActionParsingError("Action missing 'op' field")

            op_lower = op_raw.lower()
            if op_lower not in ("click", "answer"):
                raise ActionParsingError(f"Invalid operation: {op_raw}")

            values_raw = action_mapping.get("value")
            if values_raw is None:
                raise ActionParsingError("Action missing 'value' field")

            if op_lower == "click":
                values = self._parse_click_values(values_raw)
                op_code: ActionCode = 0
            else:
                values = self._parse_answer_values(values_raw)
                op_code = 1

            return op_code, values

        except ActionParsingError:
            raise
        except Exception as exc:  # pragma: no cover - defensive guardrail
            raise ActionParsingError(f"Unexpected error parsing action: {exc}")

    def parse_with_fallback(
        self, action: ActionPayload | Mapping[str, object] | str
    ) -> tuple[ActionCode, FloatPair]:
        """
        Parse action with automatic fallback to safe default.

        Args:
            action: Action in various formats

        Returns:
            Tuple of (operation, values) - defaults to center click on error
        """
        try:
            return self.parse_action(action)
        except ActionParsingError:
            center_x = float(self.image_width // 2)
            center_y = float(self.image_height // 2)
            return cast(ActionCode, 0), (center_x, center_y)

    def _normalize_action(
        self, action: ActionPayload | Mapping[str, object] | str
    ) -> Mapping[str, object | Sequence[NumberLike]]:
        """Coerce supported action inputs into mapping form."""

        candidate: Any = action
        if isinstance(action, str):
            candidate = self._parse_json_string(action)

        if not isinstance(candidate, Mapping):
            raise ActionParsingError(
                f"Action must be mapping or JSON string, got {type(candidate)}"
            )

        # Copy to avoid mutating caller-provided mappings
        normalized: dict[str, object | Sequence[NumberLike]] = dict(candidate)
        return normalized

    def _parse_json_string(self, action_str: str) -> Mapping[str, object]:
        """Parse JSON string to dictionary."""
        try:
            payload = json.loads(action_str.strip())
        except json.JSONDecodeError as exc:
            stripped = action_str.strip()
            if stripped.startswith(("{", "[")):
                raise ActionParsingError(f"Invalid JSON string: {exc}")
            raise ActionParsingError("Action must be dict or JSON string")

        if not isinstance(payload, Mapping):
            raise ActionParsingError("Parsed JSON payload must be an object")

        return payload

    def _parse_click_values(self, values: object) -> FloatPair:
        """Parse click coordinates from value array."""
        coords = self._coerce_pair(values, name="click")
        if not self.validate_click_action(*coords):
            clamped_x = max(0.0, min(self.image_width - 1, coords[0]))
            clamped_y = max(0.0, min(self.image_height - 1, coords[1]))
            return clamped_x, clamped_y
        return coords

    def _parse_answer_values(self, values: object) -> FloatPair:
        """Parse answer coordinates from value array."""
        lat, lon = self._coerce_pair(values, name="answer")
        if not GeometryUtils.validate_coordinates(lat, lon):
            return GeometryUtils.clamp_coordinates(lat, lon)
        return lat, lon

    @staticmethod
    def _coerce_pair(values: object, *, name: str) -> FloatPair:
        """Coerce an arbitrary sequence into a pair of floats."""
        expected_format = "[lat, lon]" if name == "answer" else "[x, y]"

        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ActionParsingError(
                f"{name.capitalize()} values must be {expected_format} array"
            )
        if len(values) != 2:
            raise ActionParsingError(
                f"{name.capitalize()} values must be {expected_format} array"
            )
        try:
            first = float(values[0])
            second = float(values[1])
        except (TypeError, ValueError) as exc:
            raise ActionParsingError(
                f"{name.capitalize()} coordinates must be numeric: {exc}"
            ) from exc
        return first, second

    def validate_click_action(self, x: float, y: float) -> bool:
        """
        Validate click coordinates.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            True if coordinates are valid for the image
        """
        return 0 <= x < self.image_width and 0 <= y < self.image_height

    def validate_answer_action(self, lat: float, lon: float) -> bool:
        """
        Validate answer coordinates.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees

        Returns:
            True if coordinates are valid
        """
        return GeometryUtils.validate_coordinates(lat, lon)

    def create_click_action(self, x: float, y: float) -> ClickAction:
        """Create a click action in the single supported format."""
        clamped_x = max(0, min(self.image_width - 1, int(round(x))))
        clamped_y = max(0, min(self.image_height - 1, int(round(y))))
        return {"op": "click", "value": [clamped_x, clamped_y]}

    def create_answer_action(self, lat: float, lon: float) -> AnswerAction:
        """Create an answer action in the single supported format."""
        clamped_lat, clamped_lon = GeometryUtils.clamp_coordinates(lat, lon)
        return {"op": "answer", "value": [clamped_lat, clamped_lon]}

    def get_center_click(self) -> ClickAction:
        """Return a safe center click action."""
        return self.create_click_action(self.image_width // 2, self.image_height // 2)
