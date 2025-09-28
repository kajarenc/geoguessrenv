"""
Baseline agent that follows arrows by clicking, then answers with simple heuristics.
Implements the minimal arrow follower described in TaskDescription.md.
"""

import random
from typing import Dict, List, Optional, Set, Tuple

from .vlm_broker import VLMBroker


class BaselineAgent:
    """
    A minimal arrow follower that:
    1. Clicks positions to sweep headings looking for navigation arrows
    2. Follows arrows for K steps or until loop detection (pano_id repeats)
    3. Answers with simple heuristic (continent centroids or global prior)
    """

    # Continent centroids for fallback guessing
    CONTINENT_CENTROIDS = [
        (40.0, -100.0),  # North America
        (10.0, -60.0),  # South America
        (50.0, 10.0),  # Europe
        (20.0, 25.0),  # Africa
        (30.0, 100.0),  # Asia
        (-25.0, 135.0),  # Australia/Oceania
    ]

    def __init__(
        self,
        max_nav_steps: int = 20,
        sweep_positions: int = 8,
        seed: Optional[int] = None,
    ):
        """
        Initialize baseline agent.

        Args:
            max_nav_steps: Maximum navigation steps before answering
            sweep_positions: Number of positions to try when sweeping for arrows
            seed: Random seed for reproducible behavior
        """
        self.max_nav_steps = max_nav_steps
        self.sweep_positions = sweep_positions
        self.vlm_broker = VLMBroker()

        # Set random seed
        if seed is not None:
            random.seed(seed)

        # Episode state
        self.visited_panos: Set[str] = set()
        self.step_count: int = 0

    def reset(self) -> None:
        """Reset episode state."""
        self.visited_panos.clear()
        self.step_count = 0

    def act(self, observation: Dict, info: Dict) -> Dict:
        """
        Produce an action based on the baseline strategy.

        Args:
            observation: Environment observation with 'image' key
            info: Environment info including pano_id, steps, etc.

        Returns:
            Action dict compatible with environment
        """
        pano_id = info.get("pano_id")
        current_steps = info.get("steps", 0)

        # Track visited panoramas for loop detection
        if pano_id and pano_id not in self.visited_panos:
            self.visited_panos.add(pano_id)

        # Check termination conditions
        should_answer = (
            current_steps >= self.max_nav_steps  # Reached max steps
            or (
                pano_id
                and pano_id in self.visited_panos
                and len(self.visited_panos) > 1
            )  # Loop detected
        )

        if should_answer:
            return self._generate_answer_action()
        else:
            return self._generate_navigation_action(observation, info)

    def _generate_navigation_action(self, observation: Dict, info: Dict) -> Dict:
        """Generate a navigation click action."""
        image = observation.get("image")
        if image is None:
            # Fallback to center click
            return {"op": "click", "value": [512, 256]}

        height, width = image.shape[:2]
        current_steps = info.get("steps", 0)

        # Generate sweep positions across the image width
        # Focus on horizontal sweep at middle height
        sweep_clicks = self._generate_sweep_positions(width, height)

        # For this baseline, we'll try the first position in our sweep
        click_x, click_y = sweep_clicks[current_steps % len(sweep_clicks)]

        return {"op": "click", "value": [click_x, click_y]}

    def _generate_sweep_positions(
        self, width: int, height: int
    ) -> List[Tuple[int, int]]:
        """
        Generate positions for sweeping across the image to find arrows.

        Args:
            width: Image width
            height: Image height

        Returns:
            List of (x, y) positions to try
        """
        positions = []
        y_center = height // 2

        # Horizontal sweep at center height
        for i in range(self.sweep_positions):
            x = int((i + 0.5) * width / self.sweep_positions)
            positions.append((x, y_center))

        return positions

    def _generate_answer_action(self) -> Dict:
        """
        Generate an answer action with simple heuristic.

        Uses continent centroids as a simple prior for guessing locations.
        """
        # Simple heuristic: randomly select from continent centroids
        lat, lon = random.choice(self.CONTINENT_CENTROIDS)

        # Add some noise to avoid always guessing the exact same locations
        lat += random.uniform(-5.0, 5.0)
        lon += random.uniform(-10.0, 10.0)

        # Clamp to valid ranges
        lat = max(-90.0, min(90.0, lat))
        lon = max(-180.0, min(180.0, lon))

        return {"op": "answer", "value": [lat, lon]}


class ImprovedBaselineAgent(BaselineAgent):
    """
    Slightly improved baseline that uses more sophisticated heuristics.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Track movement success to learn which positions work
        self.successful_positions: List[Tuple[int, int]] = []
        self.last_pano_id = None

    def act(self, observation: Dict, info: Dict) -> Dict:
        """Enhanced action generation with learning from successful clicks."""
        pano_id = info.get("pano_id")

        # Check if we moved (successful navigation)
        if self.last_pano_id and pano_id != self.last_pano_id:
            # Previous click was successful, remember it
            # (This would require tracking the last click position)
            pass

        self.last_pano_id = pano_id

        return super().act(observation, info)

    def _generate_answer_action(self) -> Dict:
        """
        Enhanced answer generation that could use image analysis or other cues.
        For now, uses the same continent centroid approach.
        """
        # Could be enhanced with:
        # - Text detection in images (street signs, language)
        # - Architectural style analysis
        # - Vegetation/climate analysis
        # - Sun position analysis

        return super()._generate_answer_action()
