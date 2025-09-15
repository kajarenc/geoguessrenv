"""
Action parsing utilities for the GeoGuessr environment.

Simplified to a single action format per TaskDescription.md:

- Click:  {"op": "click", "value": [x, y]}
- Answer: {"op": "answer", "value": [lat, lon]}

Coordinates are validated and clamped. On parsing failure, a safe
center click is returned by the fallback API.
"""

import json
from typing import Dict, Tuple, Union

from .geometry_utils import GeometryUtils


class ActionParsingError(Exception):
    """Exception raised when action parsing fails."""

    pass


class ActionParser:
    """Single-format action parser for the GeoGuessr environment."""

    def __init__(self, image_width: int, image_height: int):
        """
        Initialize the action parser.

        Args:
            image_width: Width of the panorama image
            image_height: Height of the panorama image
        """
        self.image_width = image_width
        self.image_height = image_height

    def parse_action(self, action: Union[Dict, str]) -> Tuple[int, Tuple[float, float]]:
        """
        Parse action in the single supported format.

        Supported inputs:
        - JSON string: '{"op":"click","value":[x,y]}' or '{"op":"answer","value":[lat,lon]}'
        - Dict: {"op": "click"|"answer", "value": [..,..]}

        Returns (op_code, values): op_code is 0 for click, 1 for answer.
        """
        try:
            # Handle JSON string input
            if isinstance(action, str):
                action = self._parse_json_string(action)

            if not isinstance(action, dict):
                raise ActionParsingError(
                    f"Action must be dict or JSON string, got {type(action)}"
                )

            # Extract and normalize operation (string only)
            op = action.get("op")
            if not isinstance(op, str):
                raise ActionParsingError("Action missing 'op' field")

            op_lower = op.lower()
            if op_lower not in ("click", "answer"):
                raise ActionParsingError(f"Invalid operation: {op}")

            # Extract values based on operation from single 'value' key
            if "value" not in action:
                raise ActionParsingError("Action missing 'value' field")

            if op_lower == "click":
                values = self._parse_click_values(action["value"])
                op_code = 0
            else:
                values = self._parse_answer_values(action["value"])
                op_code = 1

            return op_code, values

        except ActionParsingError:
            raise
        except Exception as e:
            raise ActionParsingError(f"Unexpected error parsing action: {e}")

    def parse_with_fallback(
        self, action: Union[Dict, str]
    ) -> Tuple[int, Tuple[float, float]]:
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
            # Fallback to safe center click
            return 0, (float(self.image_width // 2), float(self.image_height // 2))

    def _parse_json_string(self, action_str: str) -> Dict:
        """Parse JSON string to dictionary."""
        try:
            return json.loads(action_str.strip())
        except json.JSONDecodeError as e:
            # Check if it looks like attempted JSON (starts with { or [)
            stripped = action_str.strip()
            if stripped.startswith(("{", "[")):
                raise ActionParsingError(f"Invalid JSON string: {e}")
            else:
                raise ActionParsingError("Action must be dict or JSON string")

    def _parse_click_values(self, values) -> Tuple[float, float]:
        """Parse click coordinates from value array."""
        if not isinstance(values, (list, tuple)) or len(values) != 2:
            raise ActionParsingError(f"Click values must be [x, y] array, got {values}")

        try:
            x, y = float(values[0]), float(values[1])
        except (ValueError, TypeError) as e:
            raise ActionParsingError(f"Click coordinates must be numeric: {e}")

        # Clamp to image bounds
        x = max(0, min(self.image_width - 1, x))
        y = max(0, min(self.image_height - 1, y))

        return x, y

    def _parse_answer_values(self, values) -> Tuple[float, float]:
        """Parse answer coordinates from value array."""
        if not isinstance(values, (list, tuple)) or len(values) != 2:
            raise ActionParsingError(
                f"Answer values must be [lat, lon] array, got {values}"
            )

        try:
            lat, lon = float(values[0]), float(values[1])
        except (ValueError, TypeError) as e:
            raise ActionParsingError(f"Answer coordinates must be numeric: {e}")

        # Validate and clamp coordinates
        if not GeometryUtils.validate_coordinates(lat, lon):
            lat, lon = GeometryUtils.clamp_coordinates(lat, lon)

        return lat, lon

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

    def create_click_action(self, x: float, y: float) -> Dict:
        """Create a click action in the single supported format."""
        x = max(0, min(self.image_width - 1, x))
        y = max(0, min(self.image_height - 1, y))
        return {"op": "click", "value": [int(x), int(y)]}

    def create_answer_action(self, lat: float, lon: float) -> Dict:
        """Create an answer action in the single supported format."""
        lat, lon = GeometryUtils.clamp_coordinates(lat, lon)
        return {"op": "answer", "value": [lat, lon]}

    def get_center_click(self) -> Dict:
        """Return a safe center click action."""
        return self.create_click_action(self.image_width // 2, self.image_height // 2)
