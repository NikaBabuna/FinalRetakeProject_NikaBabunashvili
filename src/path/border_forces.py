import numpy as np
from src import config


def border_repulsion_force(positions: np.ndarray, spline) -> np.ndarray:
    """
    Returns a per-robot force (N,2) that pushes robots back toward the spline centerline
    when they are too close to the corridor border.

    Logic:
      - Find closest point on spline for each robot
      - Compute radial vector from centerline to robot
      - If distance > allowed_radius: push inward along that radial direction
    """
    positions = np.asarray(positions, dtype=float)
    N = positions.shape[0]
    forces = np.zeros((N, 2), dtype=float)

    path_width = float(getattr(config, "PATH_WIDTH_PIX", getattr(config, "PATH_WIDTH", 60.0)))
    half_w = 0.5 * path_width

    robot_r = float(getattr(config, "ROBOT_RADIUS", 0.0))
    margin = float(getattr(config, "BORDER_MARGIN", 6.0))
    k_border = float(getattr(config, "K_BORDER", 40.0))
    fmax = float(getattr(config, "BORDER_FORCE_MAX", 200.0))

    # “Allowed” radius from centerline before border push kicks in
    allowed = max(0.0, half_w - robot_r - margin)

    for i in range(N):
        x = positions[i]

        s = float(spline.closest_s(x))
        c = spline.p(s)  # centerline point
        dvec = x - c
        d = float(np.linalg.norm(dvec))

        if d < 1e-9:
            continue  # exactly on centerline

        if d <= allowed:
            continue  # safely inside

        # outward normal direction from centerline -> robot
        n = dvec / d

        # push inward proportional to penetration
        pen = d - allowed
        f = -k_border * pen * n

        # clamp magnitude
        mag = float(np.linalg.norm(f))
        if mag > fmax:
            f = f * (fmax / max(1e-9, mag))

        forces[i] = f

    return forces
