# src/phase5/missions.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Mission:
    name: str
    start_xy: Tuple[float, float]  # (x,y) in pixels
    goal_xy: Tuple[float, float]   # (x,y) in pixels
    timeout_s: float = 20.0
    goal_tol_px: float = 20.0


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def make_mission_left_to_right(
    H: int,
    W: int,
    *,
    margin_px: int = 20,
    y_frac: float = 0.50,
    timeout_s: float = 20.0,
    goal_tol_px: float = 20.0,
) -> Mission:
    """
    Left->Right across the scene at a chosen y fraction (default mid-height).
    """
    m = int(max(1, margin_px))
    y = _clamp(y_frac * (H - 1), 0, H - 1)

    start = (float(m), float(y))
    goal = (float(W - 1 - m), float(y))

    return Mission(
        name="mission_left_to_right",
        start_xy=start,
        goal_xy=goal,
        timeout_s=float(timeout_s),
        goal_tol_px=float(goal_tol_px),
    )


def make_mission_right_to_left(
    H: int,
    W: int,
    *,
    margin_px: int = 20,
    y_frac: float = 0.50,
    timeout_s: float = 20.0,
    goal_tol_px: float = 20.0,
) -> Mission:
    """
    Right->Left across the scene at a chosen y fraction (default mid-height).
    """
    m = int(max(1, margin_px))
    y = _clamp(y_frac * (H - 1), 0, H - 1)

    start = (float(W - 1 - m), float(y))
    goal = (float(m), float(y))

    return Mission(
        name="mission_right_to_left",
        start_xy=start,
        goal_xy=goal,
        timeout_s=float(timeout_s),
        goal_tol_px=float(goal_tol_px),
    )


__all__ = ["Mission", "make_mission_left_to_right", "make_mission_right_to_left"]
