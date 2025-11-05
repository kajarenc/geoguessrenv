"""
VLMBroker for standardized I/O between vision language models and the GeoGuessr environment.
Handles prompt building and action parsing according to TaskDescription.md specification.
"""

import json
from typing import Dict, Optional

import numpy as np


class VLMBroker:
    """
    Broker for standardizing VLM interactions with GeoGuessr environment.
    Builds prompts and parses actions according to the specification.
    """

    def build_prompt(self, image: np.ndarray, pose: Dict) -> str:
        """
        Build a prompt explaining the task and allowed actions.

        Args:
            image: Current view image (H, W, 3)
            pose: Current pose information with yaw_deg

        Returns:
            Prompt string for the VLM
        """
        height, width = image.shape[:2]
        yaw_deg = pose.get("yaw_deg", 0.0)

        prompt = f"""You are playing GeoGuessr with a street-view panorama.

Goal: Navigate by clicking, then answer with your best latitude/longitude guess.

Context:
- Image size: {width}x{height} pixels
- Camera yaw: {yaw_deg:.1f}°

Allowed actions (single JSON schema ONLY):
- Click to navigate: {{"op":"click","value":[x,y]}}
- Answer to finish:  {{"op":"answer","value":[lat_deg,lon_deg]}}

Constraints:
- x ∈ [0,{width - 1}], y ∈ [0,{height - 1}] (pixels)
- lat ∈ [-90,90], lon ∈ [-180,180]
- Clicks give 0 reward; only the final answer is scored
- Navigation arrows are latent; click where you infer them

Return exactly ONE JSON object in the schema above, with no extra text."""

        return prompt

    def parse_action(
        self, text: str, image_width: int = 1024, image_height: int = 512
    ) -> Dict:
        """
        Parse VLM response text to extract action.

        Args:
            text: VLM response text
            image_width: Image width for clamping click coordinates
            image_height: Image height for clamping click coordinates

        Returns:
            Parsed action dict, or safe center click on parsing failure
        """
        # Find first JSON object in the text
        json_match = self._extract_first_json(text)

        if json_match is None:
            # Fallback to center click
            return self._safe_center_click(image_width, image_height)

        try:
            action = json.loads(json_match)

            if not isinstance(action, dict) or "op" not in action:
                return self._safe_center_click(image_width, image_height)

            op = action.get("op")
            value = action.get("value", [])

            if op == "click":
                return self._validate_click_action(value, image_width, image_height)
            elif op == "answer":
                return self._validate_answer_action(value)
            else:
                # Unknown operation, fallback to center click
                return self._safe_center_click(image_width, image_height)

        except (json.JSONDecodeError, ValueError, TypeError):
            return self._safe_center_click(image_width, image_height)

    def _extract_first_json(self, text: str) -> Optional[str]:
        """Extract the first JSON object from text."""
        # Look for JSON pattern starting with { and ending with }
        # Handle nested braces correctly
        start_idx = text.find("{")
        if start_idx == -1:
            return None

        brace_count = 0
        for i, char in enumerate(text[start_idx:], start_idx):
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    return text[start_idx : i + 1]

        return None

    def _validate_click_action(
        self, value, image_width: int, image_height: int
    ) -> Dict:
        """Validate and clamp click coordinates."""
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return self._safe_center_click(image_width, image_height)

        try:
            x, y = int(value[0]), int(value[1])
            # Clamp to image bounds
            x = max(0, min(image_width - 1, x))
            y = max(0, min(image_height - 1, y))

            return {"op": "click", "value": [x, y]}
        except (ValueError, TypeError):
            return self._safe_center_click(image_width, image_height)

    def _validate_answer_action(self, value) -> Dict:
        """Validate answer coordinates."""
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            # Default to equator/prime meridian
            return {"op": "answer", "value": [0.0, 0.0]}

        try:
            lat, lon = float(value[0]), float(value[1])
            # Clamp to valid coordinate ranges
            lat = max(-90.0, min(90.0, lat))
            lon = max(-180.0, min(180.0, lon))

            return {"op": "answer", "value": [lat, lon]}
        except (ValueError, TypeError):
            return {"op": "answer", "value": [0.0, 0.0]}

    def _safe_center_click(self, image_width: int, image_height: int) -> Dict:
        """Return a safe center click action."""
        return {"op": "click", "value": [image_width // 2, image_height // 2]}

    def action_to_unified(
        self, action_dict: Dict, image_width: int = 1024, image_height: int = 512
    ) -> np.ndarray:
        """
        Convert dict action format to unified continuous array format.

        This is useful when you want to convert VLM actions to the new
        unified format for compatibility with RL algorithms.

        Args:
            action_dict: Action in dict format {"op": "click"/"answer", "value": [...]}
            image_width: Image width for normalization
            image_height: Image height for normalization

        Returns:
            Unified action array [action_type, param1, param2] in [0, 1] range
        """
        op = action_dict.get("op", "click")
        value = action_dict.get("value", [image_width // 2, image_height // 2])

        if op == "click":
            x, y = float(value[0]), float(value[1])
            # Normalize to [0, 1]
            x_norm = np.clip(x / image_width, 0.0, 1.0)
            y_norm = np.clip(y / image_height, 0.0, 1.0)
            # Use 0.25 for action_type (< 0.5 means click)
            return np.array([0.25, x_norm, y_norm], dtype=np.float32)
        else:  # answer
            lat, lon = float(value[0]), float(value[1])
            # Normalize to [0, 1]
            lat_norm = (np.clip(lat, -90.0, 90.0) + 90.0) / 180.0
            lon_norm = (np.clip(lon, -180.0, 180.0) + 180.0) / 360.0
            # Use 0.75 for action_type (>= 0.5 means answer)
            return np.array([0.75, lat_norm, lon_norm], dtype=np.float32)

    def unified_to_action(
        self, action_array: np.ndarray, image_width: int = 1024, image_height: int = 512
    ) -> Dict:
        """
        Convert unified continuous array format to dict action format.

        This is useful for interpreting RL algorithm outputs as human-readable actions.

        Args:
            action_array: Unified action [action_type, param1, param2] in [0, 1] range
            image_width: Image width for denormalization
            image_height: Image height for denormalization

        Returns:
            Action in dict format {"op": "click"/"answer", "value": [...]}
        """
        action_type_val = float(action_array[0])
        param1 = float(action_array[1])
        param2 = float(action_array[2])

        if action_type_val < 0.5:
            # Click action
            x = int(np.clip(param1 * image_width, 0, image_width - 1))
            y = int(np.clip(param2 * image_height, 0, image_height - 1))
            return {"op": "click", "value": [x, y]}
        else:
            # Answer action
            lat = np.clip(param1 * 180.0 - 90.0, -90.0, 90.0)
            lon = np.clip(param2 * 360.0 - 180.0, -180.0, 180.0)
            return {"op": "answer", "value": [float(lat), float(lon)]}
