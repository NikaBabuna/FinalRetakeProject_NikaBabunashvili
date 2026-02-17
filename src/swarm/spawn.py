# src/swarm/spawn.py

import numpy as np
from src import config

# Optional: corridor validation (nice to have, not required)
try:
    from src.path.constraints import is_inside_path
except Exception:
    is_inside_path = None


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(2,)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.array([1.0, 0.0], dtype=float)
    return v / n


def _normal_from_tangent(t: np.ndarray) -> np.ndarray:
    t = _unit(t)
    return np.array([-t[1], t[0]], dtype=float)


def _half_width_inner() -> float:
    # corridor half-width minus robot radius (and optional spawn margin)
    w = float(getattr(config, "PATH_WIDTH_PIX", getattr(config, "PATH_WIDTH", 40.0)))
    r = float(getattr(config, "ROBOT_RADIUS", 6.0))
    margin = float(getattr(config, "SPAWN_MARGIN", 2.0))
    return max(0.0, 0.5 * w - r - margin)


def _min_sep() -> float:
    # Prevent collisions at t=0
    r = float(getattr(config, "ROBOT_RADIUS", 6.0))
    return float(getattr(config, "SPAWN_MIN_SEP", 2.0 * r + 0.5))


def _spawn_points_near_s_range(spline, s_lo, s_hi, n_points, rng, max_tries=120) -> np.ndarray:
    """
    Spawns points near a segment of the spline [s_lo, s_hi],
    randomized inside the corridor width (normal offsets),
    AND enforces a minimum separation between points.
    """
    s_lo = float(max(0.0, s_lo))
    s_hi = float(min(float(spline.length), s_hi))
    if s_hi < s_lo:
        s_lo, s_hi = s_hi, s_lo

    half_inner = _half_width_inner()
    pts = np.zeros((n_points, 2), dtype=float)

    min_sep = _min_sep()
    placed = []  # list of (2,) points already accepted

    for i in range(n_points):
        ok = False

        for _ in range(max_tries):
            s = float(rng.uniform(s_lo, s_hi))
            base = np.asarray(spline.p(s), dtype=float).reshape(2,)
            tan = np.asarray(spline.tangent(s), dtype=float).reshape(2,)
            nrm = _normal_from_tangent(tan)

            off = float(rng.uniform(-half_inner, +half_inner))
            cand = base + off * nrm

            # Corridor check (if available)
            if is_inside_path is not None and (not bool(is_inside_path(cand))):
                continue

            # Separation check
            if placed:
                d = np.linalg.norm(np.asarray(placed) - cand, axis=1)
                if float(np.min(d)) < min_sep:
                    continue

            pts[i] = cand
            placed.append(cand)
            ok = True
            break

        if not ok:
            # Fallback: deterministic offsets so we still place something and keep it spread.
            s = float(np.clip(0.5 * (s_lo + s_hi), 0.0, float(spline.length)))
            base = np.asarray(spline.p(s), dtype=float).reshape(2,)
            tan = np.asarray(spline.tangent(s), dtype=float).reshape(2,)
            nrm = _normal_from_tangent(tan)

            sign = -1.0 if (i % 2 == 0) else 1.0
            mag = min(half_inner, (i // 2 + 1) * (0.6 * min_sep))
            cand = base + sign * mag * nrm

            pts[i] = cand
            placed.append(cand)

    return pts


def spawn_swarm_state(spline, N: int | None = None, seed: int | None = None):
    """
    Canonical Phase-4 initializer expected by tests.

    Returns:
      positions: (N,2)
      velocities: (N,2)  all zeros
      groups: (N,)      +1 (A->B) then -1 (B->A)
    """
    if N is None:
        N = int(getattr(config, "N_ROBOTS", 24))
    N = int(N)

    if seed is None:
        seed = int(getattr(config, "SEED", 0))

    rng = np.random.default_rng(seed)

    half = N // 2
    rest = N - half

    L = float(spline.length)

    # how far along the spline we allow the spawn region to extend from endpoints
    span = float(getattr(config, "SPAWN_S_SPAN", min(60.0, 0.12 * L)))

    pts_A = _spawn_points_near_s_range(spline, 0.0, span, half, rng)
    pts_B = _spawn_points_near_s_range(spline, max(0.0, L - span), L, rest, rng)

    positions = np.vstack([pts_A, pts_B]).astype(float)
    velocities = np.zeros_like(positions, dtype=float)

    groups = np.ones((N,), dtype=int)
    groups[half:] = -1

    return positions, velocities, groups


def spawn_swarm_positions(spline, N: int | None = None, seed: int | None = None):
    """
    Returns only (positions, groups).
    """
    positions, velocities, groups = spawn_swarm_state(spline, N=N, seed=seed)
    return positions, groups


def spawn_swarm_near_A_B(spline, N: int | None = None, seed: int | None = None, return_velocities: bool = True):
    """
    Runner-friendly alias.

    Returns either:
      (positions, velocities, groups) if return_velocities=True
      (positions, groups)            if return_velocities=False
    """
    positions, velocities, groups = spawn_swarm_state(spline, N=N, seed=seed)
    if return_velocities:
        return positions, velocities, groups
    return positions, groups
