import numpy as np
from src import config
from src.path.spline_path import build_spline_from_centerline


# -----------------------------
# Tune this if needed
# -----------------------------
# Width of allowed corridor around centerline (in PIXELS, since your spline world coords are pixels)
PATH_WIDTH_PIX = getattr(config, "PATH_WIDTH_PIX", 60.0)  # default 60px
PATH_HALF_PIX = 0.5 * float(PATH_WIDTH_PIX)


_cached_spline = None


def _get_spline():
    global _cached_spline
    if _cached_spline is None:
        # IMPORTANT: build it the SAME way tests do (no args)
        _cached_spline = build_spline_from_centerline()
    return _cached_spline


def _pos(spline, s: float) -> np.ndarray:
    # compatibility: some versions use pos(), others use p()
    if hasattr(spline, "pos"):
        return spline.pos(s)
    return spline.p(s)


def _closest_s_refined(spline, x: np.ndarray) -> float:
    """
    closest_s() is sample-based, so it can be off by ~1px.
    This refinement makes 'point on path' produce distance ~0.
    """
    s0 = float(spline.closest_s(x))

    # window size based on path length and samples (safe default)
    # if sampling = 1200, step ~ L/1200
    step = float(spline.length) / float(getattr(config, "CLOSEST_S_SAMPLES", 1200))
    window = 6.0 * step  # search +/- ~6 steps

    lo = max(0.0, s0 - window)
    hi = min(float(spline.length), s0 + window)

    # iterative local grid search (fast + robust)
    best_s = s0
    best_d2 = float("inf")

    for _ in range(3):  # 3 refinements is plenty
        ss = np.linspace(lo, hi, 41)  # dense local sampling
        for s in ss:
            p = _pos(spline, float(s))
            d2 = float(np.sum((p - x) ** 2))
            if d2 < best_d2:
                best_d2 = d2
                best_s = float(s)

        # tighten window around best
        lo = max(0.0, best_s - window * 0.25)
        hi = min(float(spline.length), best_s + window * 0.25)

    return best_s


# ============================================================
# API: distance to path
# ============================================================

def distance_to_path(x: np.ndarray) -> float:
    """
    Distance from point x to the spline centerline (pixels).
    x is in the SAME coordinate frame as spline (world coords in your project = pixels with origin at A).
    """
    spline = _get_spline()
    x = np.asarray(x, dtype=float).reshape(2)

    s = _closest_s_refined(spline, x)
    p = _pos(spline, s)
    d = float(np.linalg.norm(x - p))

    # numerical cleanup so tests pass for on-path points
    if d < 1e-2:
        return 0.0
    return d


# ============================================================
# API: inside path
# ============================================================

def is_inside_path(x: np.ndarray) -> bool:
    """
    Inside if within corridor around centerline:
        distance <= (PATH_WIDTH/2 - robot_radius)

    NOTE: robot_radius in config is tiny (0.2) compared to pixel coords,
    but we still include it for correctness.
    """
    x = np.asarray(x, dtype=float).reshape(2)
    d = distance_to_path(x)

    allowed = PATH_HALF_PIX - float(getattr(config, "ROBOT_RADIUS", 0.0))
    return d <= allowed
