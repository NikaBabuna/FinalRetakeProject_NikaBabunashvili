import numpy as np
from src import config


def detect_collisions(positions: np.ndarray, threshold: float | None = None):
    """
    Detect pairwise collisions for positions (N,2).

    Returns:
      pairs: list of (i, j, dist) where i < j and dist < threshold
    """
    pos = np.asarray(positions, dtype=float)
    N = pos.shape[0]

    if threshold is None:
        threshold = float(getattr(config, "ACCIDENT_DIST", 2.0 * getattr(config, "ROBOT_RADIUS", 0.0)))

    pairs = []

    for i in range(N):
        for j in range(i + 1, N):
            d = float(np.linalg.norm(pos[i] - pos[j]))
            if d < threshold:
                pairs.append((i, j, d))

    return pairs


def count_collisions(positions: np.ndarray, threshold: float | None = None) -> int:
    return len(detect_collisions(positions, threshold))
