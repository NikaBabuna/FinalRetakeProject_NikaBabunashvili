import numpy as np
from scipy.interpolate import CubicSpline

# --- spline caching (for backwards-compatible constraint calls) ---
_CACHED_SPLINE = None

def get_spline():
    global _CACHED_SPLINE
    if _CACHED_SPLINE is None:
        raise TypeError("No cached spline yet. Build a spline first.")
    return _CACHED_SPLINE



# ============================================================
# SPLINE OBJECT
# ============================================================

class SplinePath:
    """
    Continuous parametric path p(s)
    s = arc length (0 → L)
    """

    def __init__(self, points: np.ndarray):
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Centerline must be (N,2)")
        if len(points) < 4:
            raise ValueError("Need >=4 points for spline")

        pts = np.asarray(points, dtype=float)

        # --- arc length parameter ---
        diffs = pts[1:] - pts[:-1]
        seg = np.linalg.norm(diffs, axis=1)
        s = np.zeros(len(pts), dtype=float)
        s[1:] = np.cumsum(seg)

        self.s = s
        self.length = float(s[-1])
        self.points = pts

        if self.length <= 1e-6:
            raise RuntimeError("Centerline too short")

        # splines
        self.sx = CubicSpline(s, pts[:, 0])
        self.sy = CubicSpline(s, pts[:, 1])

        # --- auto-cache newest spline for old APIs like distance_to_path(x) ---
        global _CACHED_SPLINE
        _CACHED_SPLINE = self
        try:
            from src.path import constraints as _constraints
            _constraints.set_default_spline(self)
        except Exception:
            pass

    # ------------------------------------------------
    def clamp(self, s: float) -> float:
        return float(np.clip(s, 0.0, self.length))

    # ------------------------------------------------
    def p(self, s: float) -> np.ndarray:
        s = self.clamp(s)
        return np.array([self.sx(s), self.sy(s)], dtype=float)

    def pos(self, s: float) -> np.ndarray:
        """Compatibility alias for tests"""
        return self.p(s)

    # ------------------------------------------------
    def tangent(self, s: float) -> np.ndarray:
        s = self.clamp(s)
        dx = float(self.sx.derivative()(s))
        dy = float(self.sy.derivative()(s))
        v = np.array([dx, dy], dtype=float)
        n = np.linalg.norm(v)
        if n < 1e-9:
            return np.array([1.0, 0.0])
        return v / n

    def sample(self, n: int = 200) -> np.ndarray:
        """
        Returns n sampled points along spline.
        Used for plotting/debug/tests.
        """
        ss = np.linspace(0.0, self.length, n)
        xs = self.sx(ss)
        ys = self.sy(ss)
        return np.stack([xs, ys], axis=1)

    # ------------------------------------------------
    def closest_s(self, pos: np.ndarray, samples: int = 400) -> float:
        """
        Brute-force closest point search (robust).
        """
        pos = np.asarray(pos, dtype=float).reshape(2)

        ss = np.linspace(0.0, self.length, samples)
        xs = self.sx(ss)
        ys = self.sy(ss)

        dx = xs - pos[0]
        dy = ys - pos[1]
        d2 = dx * dx + dy * dy
        idx = int(np.argmin(d2))

        return float(ss[idx])


# ============================================================
# BUILDER
# ============================================================

def build_spline_from_centerline(centerline: np.ndarray | None = None) -> SplinePath:
    """
    Main API used by system + tests.

    If centerline is None:
        rebuild full pipeline automatically.
    If provided:
        build spline directly from given centerline.
    """

    # ------------------------------------------------
    # If no centerline provided → rebuild pipeline
    # ------------------------------------------------
    if centerline is None:
        from src.path.centerline import extract_centerline_points
        centerline = extract_centerline_points()

    centerline = np.asarray(centerline, dtype=float)

    if centerline is None or len(centerline) < 4:
        raise RuntimeError("Invalid centerline for spline")

    return SplinePath(centerline)


