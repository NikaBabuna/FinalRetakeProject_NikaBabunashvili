# src/phase5/repulsion.py
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union
import numpy as np

Vec2 = Tuple[float, float]


def _as_points_xy(points: Union[np.ndarray, Sequence, None]) -> np.ndarray:
    """
    Convert input points into an (N,2) float32 array.

    Accepts:
      - np.ndarray shape (N,2)
      - list of (x,y)
      - list of dicts/objects with (x,y) or (cx,cy) or pos/xy fields
    """
    if points is None:
        return np.zeros((0, 2), dtype=np.float32)

    if isinstance(points, np.ndarray):
        arr = points.astype(np.float32, copy=False)
        if arr.ndim == 2 and arr.shape[1] == 2:
            return arr
        if arr.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        raise ValueError(f"points ndarray must be (N,2), got {arr.shape}")

    out = []
    for p in points:
        if p is None:
            continue

        # tuple/list (x,y)
        if isinstance(p, (tuple, list)) and len(p) >= 2:
            out.append((float(p[0]), float(p[1])))
            continue

        # dict-like
        if isinstance(p, dict):
            if "pos" in p and p["pos"] is not None and len(p["pos"]) >= 2:
                out.append((float(p["pos"][0]), float(p["pos"][1])))
                continue
            if "xy" in p and p["xy"] is not None and len(p["xy"]) >= 2:
                out.append((float(p["xy"][0]), float(p["xy"][1])))
                continue
            if "cx" in p and "cy" in p:
                out.append((float(p["cx"]), float(p["cy"])))
                continue
            if "x" in p and "y" in p:
                out.append((float(p["x"]), float(p["y"])))
                continue

        # object attributes
        if hasattr(p, "pos"):
            pos = getattr(p, "pos")
            if pos is not None and len(pos) >= 2:
                out.append((float(pos[0]), float(pos[1])))
                continue
        if hasattr(p, "xy"):
            xy = getattr(p, "xy")
            if xy is not None and len(xy) >= 2:
                out.append((float(xy[0]), float(xy[1])))
                continue
        if hasattr(p, "cx") and hasattr(p, "cy"):
            out.append((float(getattr(p, "cx")), float(getattr(p, "cy"))))
            continue
        if hasattr(p, "x") and hasattr(p, "y"):
            out.append((float(getattr(p, "x")), float(getattr(p, "y"))))
            continue

    if not out:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(out, dtype=np.float32)


def min_distance(robot_xy: Vec2, points: Union[np.ndarray, Sequence, None]) -> float:
    pts = _as_points_xy(points)
    if pts.shape[0] == 0:
        return float("inf")
    r = np.asarray(robot_xy, dtype=np.float32)
    d = np.linalg.norm(pts - r[None, :], axis=1)
    return float(np.min(d))


def repulsion_force(
    robot_xy: Vec2,
    points: Union[np.ndarray, Sequence, None],
    rsafe: float,
    krep: float = 250.0,
    gamma: float = 2.0,
    eps: float = 1e-6,
    max_force: Optional[float] = 2000.0,
    **kwargs,
) -> np.ndarray:
    """
    Backward-compatible name expected by some tests: repulsion_force(...)

    The "force" here is intended to be *added to dv/dt* (so effectively an acceleration term
    in your VT model). It pushes the robot away from pedestrians inside rsafe.

    Extra kwargs are accepted so older call-sites don't break (e.g., k_rep, max_accel, etc).
    """
    # Accept a few common legacy names if passed via kwargs
    if "k_rep" in kwargs and kwargs["k_rep"] is not None:
        krep = float(kwargs["k_rep"])
    if "power" in kwargs and kwargs["power"] is not None:
        gamma = float(kwargs["power"])
    if "max_accel" in kwargs and kwargs["max_accel"] is not None:
        max_force = float(kwargs["max_accel"])

    pts = _as_points_xy(points)
    if pts.shape[0] == 0 or rsafe <= 0:
        return np.zeros((2,), dtype=np.float32)

    r = np.asarray(robot_xy, dtype=np.float32)
    dif = r[None, :] - pts
    d = np.linalg.norm(dif, axis=1)

    mask = d < float(rsafe)
    if not np.any(mask):
        return np.zeros((2,), dtype=np.float32)

    d_m = d[mask]
    dif_m = dif[mask]

    # weight ramps up as you get closer (0 at rsafe, 1 at d=0)
    w = ((float(rsafe) - d_m) / float(rsafe))
    w = np.clip(w, 0.0, 1.0) ** float(gamma)

    inv = 1.0 / np.maximum(d_m, float(eps))
    contrib = (dif_m * inv[:, None]) * w[:, None]
    F = float(krep) * np.sum(contrib, axis=0).astype(np.float32)

    if max_force is not None:
        mag = float(np.linalg.norm(F))
        if mag > float(max_force) and mag > 1e-9:
            F = F * (float(max_force) / mag)

    return F


def repulsion_accel(
    robot_xy: Vec2,
    points: Union[np.ndarray, Sequence, None],
    rsafe: float,
    krep: float = 250.0,
    gamma: float = 2.0,
    eps: float = 1e-6,
    max_accel: Optional[float] = 2000.0,
) -> np.ndarray:
    """
    Same as repulsion_force, but with a clearer name for the VT dynamics module.
    """
    return repulsion_force(
        robot_xy=robot_xy,
        points=points,
        rsafe=rsafe,
        krep=krep,
        gamma=gamma,
        eps=eps,
        max_force=max_accel,
    )


__all__ = ["repulsion_force", "repulsion_accel", "min_distance"]
