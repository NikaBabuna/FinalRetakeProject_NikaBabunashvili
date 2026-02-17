# src/phase5/vt_dynamics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple
import numpy as np

from src.phase5.repulsion import repulsion_accel

Vec2 = Tuple[float, float]


def bilinear_sample_flow(flow_hw2: np.ndarray, x: float, y: float) -> np.ndarray:
    """
    Bilinear sample of a dense flow field at subpixel (x,y).
    flow_hw2: (H,W,2), where flow[...,0]=u (x-dir), flow[...,1]=v (y-dir)
    """
    H, W = flow_hw2.shape[:2]
    if H <= 0 or W <= 0:
        return np.zeros((2,), dtype=np.float32)

    # clamp coordinates to valid range
    x = float(np.clip(x, 0.0, W - 1.0))
    y = float(np.clip(y, 0.0, H - 1.0))

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, W - 1)
    y1 = min(y0 + 1, H - 1)

    fx = x - x0
    fy = y - y0

    v00 = flow_hw2[y0, x0].astype(np.float32, copy=False)
    v10 = flow_hw2[y0, x1].astype(np.float32, copy=False)
    v01 = flow_hw2[y1, x0].astype(np.float32, copy=False)
    v11 = flow_hw2[y1, x1].astype(np.float32, copy=False)

    top = (1.0 - fx) * v00 + fx * v10
    bot = (1.0 - fx) * v01 + fx * v11
    v = (1.0 - fy) * top + fy * bot
    return v.astype(np.float32, copy=False)


@dataclass
class VTParams:
    kv: float = 2.5          # how fast v tracks v_des
    kd: float = 0.25         # linear damping on v
    rsafe: float = 40.0      # pixels
    krep: float = 250.0      # repulsion gain
    rep_gamma: float = 2.0   # repulsion nonlinearity
    max_rep_accel: float = 2000.0
    # optional goal bias (used later for missions)
    goal_weight: float = 0.0
    goal_speed: float = 0.0  # px/s when goal_weight>0


def vt_derivative(
    state: np.ndarray,                 # [x,y,vx,vy]
    flow_t: np.ndarray,                # (H,W,2) in px/sec (or consistent units)
    ped_points_t: Optional[Sequence],  # list/array of (x,y) for this timestep
    params: VTParams,
    goal_xy: Optional[Vec2] = None,
) -> np.ndarray:
    """
    Returns d/dt [x,y,vx,vy] = [vx,vy,ax,ay]
    """
    x, y, vx, vy = map(float, state)
    v = np.array([vx, vy], dtype=np.float32)

    v_flow = bilinear_sample_flow(flow_t, x, y)  # local desired velocity from flow

    v_des = v_flow.copy()

    # Optional goal bias (you'll use this for the "missions" checkbox)
    if params.goal_weight > 0.0 and goal_xy is not None and params.goal_speed > 0.0:
        gx, gy = float(goal_xy[0]), float(goal_xy[1])
        to_goal = np.array([gx - x, gy - y], dtype=np.float32)
        d = float(np.linalg.norm(to_goal))
        if d > 1e-6:
            v_goal = (to_goal / d) * float(params.goal_speed)
            v_des = v_des + float(params.goal_weight) * v_goal

    # Repulsion acceleration from pedestrians
    a_rep = repulsion_accel(
        robot_xy=(x, y),
        points=ped_points_t,
        rsafe=float(params.rsafe),
        krep=float(params.krep),
        gamma=float(params.rep_gamma),
        max_accel=float(params.max_rep_accel),
    )

    # Velocity tracking + damping + repulsion
    a = float(params.kv) * (v_des - v) - float(params.kd) * v + a_rep

    return np.array([vx, vy, float(a[0]), float(a[1])], dtype=np.float32)


def rk2_step(
    state: np.ndarray,
    dt: float,
    deriv_fn: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """
    Midpoint RK2 for state integration.
    """
    k1 = deriv_fn(state)
    mid = state + 0.5 * float(dt) * k1
    k2 = deriv_fn(mid)
    return state + float(dt) * k2


__all__ = ["VTParams", "vt_derivative", "rk2_step", "bilinear_sample_flow"]
