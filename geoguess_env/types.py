"""Shared type aliases and structured payload definitions for GeoGuessr env."""

from __future__ import annotations

from typing import Literal, Sequence, Tuple, TypedDict, Union

import numpy as np
import numpy.typing as npt

NumberLike = Union[int, float]
FloatPair = Tuple[float, float]

ActionCode = Literal[0, 1]
ActionOp = Literal["click", "answer"]


class ActionPayload(TypedDict):
    """Structured action payload accepted by the action parser."""

    op: ActionOp
    value: Sequence[NumberLike]


class ClickAction(ActionPayload):
    """Click action with screen coordinates."""

    op: Literal["click"]


class AnswerAction(ActionPayload):
    """Answer action with latitude/longitude values."""

    op: Literal["answer"]


ObservationArray = npt.NDArray[np.uint8]


class Observation(TypedDict):
    """Observation returned by the environment."""

    image: ObservationArray


class PoseInfo(TypedDict):
    """Pose metadata provided by the environment."""

    yaw_deg: float
    heading_deg: float


class LinkScreen(TypedDict, total=False):
    """Screen-space representation of a navigation link."""

    id: str
    screen_xy: list[int]
    heading_deg: float
    conf: float
    _distance_px: float | None
    _rel_heading_deg: float


class EnvInfo(TypedDict, total=False):
    """Information dictionary returned with each observation."""

    provider: str
    pano_id: str | None
    gt_lat: float | None
    gt_lon: float | None
    steps: int
    pose: PoseInfo
    links: list[LinkScreen]
    guess_lat: float
    guess_lon: float
    distance_km: float
    score: float


class NavigationLink(TypedDict, total=False):
    """Navigation link metadata stored in the panorama graph."""

    id: str
    direction: float | None


class PanoramaNode(TypedDict, total=False):
    """Panorama metadata stored in the prepared graph."""

    lat: float | None
    lon: float | None
    heading: float | None
    date: str | None
    links: list[NavigationLink]


PanoramaGraph = dict[str, PanoramaNode]


def ensure_float_pair(values: Sequence[NumberLike]) -> FloatPair:
    """Helper converting a numeric sequence into a float pair."""

    if len(values) != 2:
        raise ValueError("Expected a sequence of two numeric values")

    first, second = float(values[0]), float(values[1])
    return first, second


def clamp_coordinates(lat: float, lon: float) -> FloatPair:
    """Clamp latitude/longitude pair to valid ranges."""

    clamped_lat = max(-90.0, min(90.0, lat))
    clamped_lon = max(-180.0, min(180.0, lon))
    return clamped_lat, clamped_lon


__all__ = [
    "ActionCode",
    "ActionOp",
    "ActionPayload",
    "AnswerAction",
    "ClickAction",
    "ObservationArray",
    "Observation",
    "PoseInfo",
    "LinkScreen",
    "EnvInfo",
    "NavigationLink",
    "PanoramaNode",
    "PanoramaGraph",
    "FloatPair",
    "NumberLike",
    "ensure_float_pair",
    "clamp_coordinates",
]
