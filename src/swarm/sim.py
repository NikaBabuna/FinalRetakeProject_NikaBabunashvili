# src/swarm/sim.py

import numpy as np
from src import config
from src.core.core import step_rk2
from src.path.path_controller import PathController
from src.path.border_forces import border_repulsion_force
from src.swarm.collisions import detect_collisions


class ReversedSpline:
    """
    Wraps a spline so s=0 corresponds to original end (B),
    s=L corresponds to original start (A).
    """
    def __init__(self, spline):
        self._spline = spline
        self.length = float(spline.length)

    def p(self, s: float) -> np.ndarray:
        s = float(s)
        return np.asarray(self._spline.p(self.length - s), dtype=float)

    def pos(self, s: float) -> np.ndarray:
        return self.p(s)

    def tangent(self, s: float) -> np.ndarray:
        s = float(s)
        return -np.asarray(self._spline.tangent(self.length - s), dtype=float)

    def closest_s(self, pos: np.ndarray) -> float:
        return float(self.length - self._spline.closest_s(pos))


def simulate_swarm_oneway_A_to_B(spline, positions, velocities, steps, dt=None):
    if dt is None:
        dt = float(getattr(config, "DT", 0.05))

    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)

    N = positions.shape[0]
    traj = np.zeros((steps + 1, N, 2), dtype=float)
    traj[0] = positions

    controllers = [PathController(spline) for _ in range(N)]
    kp = float(getattr(config, "KP", 1.2))

    for k in range(1, steps + 1):
        targets = np.zeros_like(positions)
        for i in range(N):
            targets[i] = controllers[i].update(positions[i])

        u = kp * (targets - positions)
        u = u + border_repulsion_force(positions, spline)

        positions, velocities = step_rk2(positions, velocities, u, dt)
        traj[k] = positions

    return traj


def simulate_swarm_oneway_B_to_A(spline, positions, velocities, steps, dt=None):
    rev = ReversedSpline(spline)
    return simulate_swarm_oneway_A_to_B(rev, positions, velocities, steps, dt=dt)


def simulate_swarm_twoway(
    spline,
    positions,
    velocities,
    groups,
    steps,
    dt=None,
    log_collisions: bool = False,          # ✅ old tests use this
    return_collision_log: bool = False,    # ✅ newer code can use this
):
    """
    Runs BOTH directions together starting at the same timestep.

    groups[i]:
      +1 for A->B robots
      -1 for B->A robots

    Returns:
      traj: (steps+1, N, 2)

    If log_collisions=True OR return_collision_log=True:
      returns (traj, collision_log)
      where collision_log is a list of (timestep, pairs)
    """
    if dt is None:
        dt = float(getattr(config, "DT", 0.05))

    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    groups = np.asarray(groups, dtype=int)

    N = positions.shape[0]
    traj = np.zeros((steps + 1, N, 2), dtype=float)
    traj[0] = positions

    rev = ReversedSpline(spline)

    controllers = []
    for i in range(N):
        controllers.append(PathController(spline if groups[i] == 1 else rev))

    kp = float(getattr(config, "KP", 1.2))

    collision_log = []
    want_log = bool(log_collisions or return_collision_log)

    for k in range(1, steps + 1):
        targets = np.zeros_like(positions)
        for i in range(N):
            targets[i] = controllers[i].update(positions[i])

        u = kp * (targets - positions)
        u = u + border_repulsion_force(positions, spline)

        positions, velocities = step_rk2(positions, velocities, u, dt)
        traj[k] = positions

        if want_log:
            pairs = detect_collisions(positions)
            if pairs:
                collision_log.append((k, pairs))

    if want_log:
        return traj, collision_log
    return traj
