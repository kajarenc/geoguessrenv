from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class AgentAction:
    op: str  # "click" or "answer"
    click: Optional[Tuple[int, int]] = None
    answer: Optional[Tuple[float, float]] = None


@dataclass
class AgentConfig:
    model: str = "gpt-5"
    temperature: float = 0.0
    max_nav_steps: int = 20
    image_width: int = 1024
    image_height: int = 512
    request_timeout_s: float = 60.0
    cache_dir: Optional[str] = None


class BaseAgent:
    def __init__(self, config: Optional[AgentConfig] = None) -> None:
        self.config = config or AgentConfig()

    def reset(self) -> None:
        """Reset any episodic state (override in subclasses)."""
        return None

    def act(self, observation, info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Produce an action compatible with the environment's action space:
          {"op": "click"|"answer", "click": [x,y], "answer": [lat,lon]}
        """
        raise NotImplementedError


