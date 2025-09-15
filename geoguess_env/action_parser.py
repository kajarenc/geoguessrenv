"""
Action parsing utilities for the GeoGuessr environment.

This module provides robust action parsing with validation,
error handling, and support for multiple action formats.
"""

import json
from typing import Dict, Tuple, Union

from .geometry_utils import GeometryUtils


class ActionParsingError(Exception):
    """Exception raised when action parsing fails."""

    pass


class ActionParser:
    """
    Robust action parser for the GeoGuessr environment.

    Handles parsing of click and answer actions from various input formats
    with comprehensive validation and error recovery.
    """

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
        Parse action from various input formats.

        Supports multiple input formats:
        - JSON string: '{"op":"click","value":[x,y]}'
        - Dict with value: {"op": "click", "value": [x,y]}
        - Dict with explicit keys: {"op": "click", "click": [x,y], "answer": [lat,lon]}

        Args:
            action: Action in various formats

        Returns:
            Tuple of (operation, values) where:
            - operation: 0 for click, 1 for answer
            - values: (x, y) for click, (lat, lon) for answer

        Raises:
            ActionParsingError: If action cannot be parsed
        """
        try:
            # Handle JSON string input
            if isinstance(action, str):
                action = self._parse_json_string(action)

            if not isinstance(action, dict):
                raise ActionParsingError(
                    f"Action must be dict or JSON string, got {type(action)}"
                )

            # Extract operation
            op = action.get("op")
            if op is None:
                raise ActionParsingError("Action missing 'op' field")

            # Normalize operation
            if isinstance(op, str):
                op_code = (
                    0
                    if op.lower() == "click"
                    else 1
                    if op.lower() == "answer"
                    else None
                )
                if op_code is None:
                    raise ActionParsingError(f"Invalid operation: {op}")
            else:
                op_code = int(op)
                if op_code not in (0, 1):
                    raise ActionParsingError(f"Operation must be 0 or 1, got {op_code}")

            # Extract values based on operation and available keys
            if op_code == 0:  # Click
                values = self._parse_click_values(action)
            else:  # Answer
                values = self._parse_answer_values(action)

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

    def _parse_click_values(self, action: Dict) -> Tuple[float, float]:
        """Parse click coordinates from action dictionary."""
        # Try explicit 'click' key first
        if "click" in action:
            values = action["click"]
        elif "value" in action:
            values = action["value"]
        else:
            raise ActionParsingError("Click action missing coordinate values")

        if not isinstance(values, (list, tuple)) or len(values) != 2:
            raise ActionParsingError(f"Click values must be [x, y] array, got {values}")

        try:
            x, y = float(values[0]), float(values[1])
        except (ValueError, TypeError) as e:
            raise ActionParsingError(f"Click coordinates must be numeric: {e}")

        # Validate and clamp coordinates
        x = max(0, min(self.image_width - 1, x))
        y = max(0, min(self.image_height - 1, y))

        return x, y

    def _parse_answer_values(self, action: Dict) -> Tuple[float, float]:
        """Parse answer coordinates from action dictionary."""
        # Try explicit 'answer' key first
        if "answer" in action:
            values = action["answer"]
        elif "value" in action:
            values = action["value"]
        else:
            raise ActionParsingError("Answer action missing coordinate values")

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
        """
        Create a properly formatted click action.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            Click action dictionary
        """
        # Clamp coordinates
        x = max(0, min(self.image_width - 1, x))
        y = max(0, min(self.image_height - 1, y))

        return {
            "op": "click",
            "click": [int(x), int(y)],
            "value": [int(x), int(y)],  # For backward compatibility
        }

    def create_answer_action(self, lat: float, lon: float) -> Dict:
        """
        Create a properly formatted answer action.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees

        Returns:
            Answer action dictionary
        """
        # Clamp coordinates
        lat, lon = GeometryUtils.clamp_coordinates(lat, lon)

        return {
            "op": "answer",
            "answer": [lat, lon],
            "value": [lat, lon],  # For backward compatibility
        }

    def get_center_click(self) -> Dict:
        """
        Get a safe center click action.

        Returns:
            Click action for center of image
        """
        return self.create_click_action(self.image_width // 2, self.image_height // 2)
