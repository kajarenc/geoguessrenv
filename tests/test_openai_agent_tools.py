import os

import numpy as np
import pytest

from agents.base import AgentConfig
from agents.openai_agent import OpenAIVisionAgent


class StubAgent(OpenAIVisionAgent):
    def __init__(self, cfg: AgentConfig, stub_response):
        super().__init__(cfg)
        self._stub_response = stub_response

    def _chat_completions(  # type: ignore[override]
        self, messages, tools, tool_choice
    ):
        return self._stub_response


def _dummy_obs(h: int = 512, w: int = 1024):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _info_with_links(steps: int = 0):
    return {
        "pano_id": "TEST_PANO",
        "steps": steps,
        "pose": {"heading_deg": 180.0},
        "links": [
            {"id": "A", "heading_deg": 330.0, "screen_xy": [942, 256]},
            {"id": "B", "heading_deg": 150.0, "screen_xy": [427, 256]},
        ],
    }


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ, reason="OPENAI_API_KEY is not set"
)
def test_act_click_snaps_to_nearest_link():
    cfg = AgentConfig(cache_dir=None)
    stub = {"op": "click", "click": {"x": 960, "y": 240}}
    agent = StubAgent(cfg, stub)

    obs = _dummy_obs()
    info = _info_with_links(steps=0)
    action = agent.act(obs, info)

    # Should snap to the nearest provided link center (942, 256)
    assert action["op"] == "click"
    assert action["value"] == [942, 256]


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ, reason="OPENAI_API_KEY is not set"
)
def test_act_answer_is_clamped():
    cfg = AgentConfig(cache_dir=None)
    stub = {"op": "answer", "answer": {"lat": 123.4, "lon": -200.0}}
    agent = StubAgent(cfg, stub)

    obs = _dummy_obs()
    info = _info_with_links(steps=0)
    action = agent.act(obs, info)

    assert action["op"] == "answer"
    # Expect clamping to [-90, 90] and [-180, 180]
    assert action["value"] == [90.0, -180.0]


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ, reason="OPENAI_API_KEY is not set"
)
def test_fallback_click_then_answer():
    cfg = AgentConfig(cache_dir=None)

    # Case 1: No valid tool call data; links present -> click first link
    agent_click = StubAgent(cfg, {})
    obs = _dummy_obs()
    info = _info_with_links(steps=0)
    action_click = agent_click.act(obs, info)
    assert action_click["op"] == "click"
    assert action_click["value"] == [942, 256]

    # Case 2: No valid tool call data; no links -> answer (0,0)
    agent_answer = StubAgent(cfg, {})
    info2 = {
        "pano_id": "TEST_PANO",
        "steps": 0,
        "pose": {"heading_deg": 0.0},
        "links": [],
    }
    action_answer = agent_answer.act(obs, info2)
    assert action_answer["op"] == "answer"
    assert action_answer["value"] == [0.0, 0.0]
