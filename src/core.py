import numpy as np
from src import config


# ==================================================
# STATE
# ==================================================

def initialize_robots(n_robots: int):
    """
    Returns:
        positions  (N,2)
        velocities (N,2)
    """
    positions = np.zeros((n_robots, 2), dtype=float)
    velocities = np.zeros((n_robots, 2), dtype=float)
    return positions, velocities


# ==================================================
# CORE HELPERS
# ==================================================

def clip_speed(velocities: np.ndarray, vmax: float) -> np.ndarray:
    """
    Clips each robot's speed so ||v|| <= vmax.
    velocities: (N,2)
    """
    speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
    speeds = np.maximum(speeds, 1e-8)          # avoid division by zero
    scale = np.minimum(1.0, vmax / speeds)     # <= 1.0 if speeding
    return velocities * scale


# ==================================================
# FORCES
# ==================================================

def attraction_force(positions: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    force = KP * (target - position)
    positions: (N,2), target: (2,)
    """
    return config.KP * (target - positions)


def damping_force(velocities: np.ndarray) -> np.ndarray:
    """
    force = -KD * velocity
    velocities: (N,2)
    """
    return -config.KD * velocities


def repulsion_force(positions: np.ndarray) -> np.ndarray:
    """
    Pairwise repulsion if robots are closer than RSAFE.
    positions: (N,2)
    returns: (N,2)
    """
    n = positions.shape[0]
    forces = np.zeros_like(positions)

    rsafe = config.RSAFE
    krep = config.KREP

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            diff = positions[i] - positions[j]
            dist = np.linalg.norm(diff)

            if dist < 1e-8:
                continue

            if dist < rsafe:
                direction = diff / dist
                magnitude = krep * (rsafe - dist)
                forces[i] += direction * magnitude

    return forces


def compute_acceleration(positions: np.ndarray,
                         velocities: np.ndarray,
                         target: np.ndarray) -> np.ndarray:
    """
    a = attraction + repulsion + damping
    """
    return (
        attraction_force(positions, target)
        + repulsion_force(positions)
        + damping_force(velocities)
    )


# ==================================================
# INTEGRATOR (RK2 MIDPOINT)
# ==================================================

def step_rk2(positions: np.ndarray,
             velocities: np.ndarray,
             target: np.ndarray,
             dt: float):
    """
    RK2 midpoint for the system:
      dx/dt = v
      dv/dt = a(x, v)

    Returns:
      positions_new, velocities_new
    """

    # k1 at current state
    a1 = compute_acceleration(positions, velocities, target)
    v_half = velocities + 0.5 * dt * a1
    x_half = positions + 0.5 * dt * velocities  # midpoint position uses current v

    # k2 at midpoint state
    a2 = compute_acceleration(x_half, v_half, target)

    # full update
    velocities_new = velocities + dt * a2
    velocities_new = clip_speed(velocities_new, config.VMAX)

    positions_new = positions + dt * v_half

    return positions_new, velocities_new


# ==================================================
# SIMULATION LOOP
# ==================================================

def run_simulation(target: np.ndarray,
                   n_steps: int = 300,
                   dt: float = config.DT,
                   n_robots: int = 1,
                   start_positions: np.ndarray | None = None,
                   start_velocities: np.ndarray | None = None,
                   store_velocities: bool = False):
    """
    Returns:
      trajectory: (n_steps+1, N, 2)
      velocities_history (optional): (n_steps+1, N, 2)
    """

    if start_positions is None or start_velocities is None:
        positions, velocities = initialize_robots(n_robots)
    else:
        positions = start_positions.copy()
        velocities = start_velocities.copy()

    trajectory = np.zeros((n_steps + 1, n_robots, 2), dtype=float)
    trajectory[0] = positions

    vel_hist = None
    if store_velocities:
        vel_hist = np.zeros((n_steps + 1, n_robots, 2), dtype=float)
        vel_hist[0] = velocities

    for k in range(1, n_steps + 1):
        positions, velocities = step_rk2(positions, velocities, target, dt)
        trajectory[k] = positions
        if store_velocities:
            vel_hist[k] = velocities

    if store_velocities:
        return trajectory, vel_hist
    return trajectory

