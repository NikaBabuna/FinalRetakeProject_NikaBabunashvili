# src/path/constraints.py

import numpy as np
from src import config

_DEFAULT_SPLINE = None


def set_default_spline(spline) -> None:
    global _DEFAULT_SPLINE
    _DEFAULT_SPLINE = spline


def _is_spline(obj) -> bool:
    return (
        hasattr(obj, "closest_s")
        and hasattr(obj, "tangent")
        and (hasattr(obj, "p") or hasattr(obj, "pos"))
    )


def _looks_like_point(x) -> bool:
    try:
        a = np.asarray(x, dtype=float)
        return a.shape == (2,)
    except Exception:
        return False


def _get_default_spline():
    global _DEFAULT_SPLINE
    if _DEFAULT_SPLINE is not None:
        return _DEFAULT_SPLINE

    # Preferred: spline_path.get_spline() (we add this in spline_path.py)
    try:
        from src.path.spline_path import get_spline
        s = get_spline()
        if _is_spline(s):
            _DEFAULT_SPLINE = s
            return s
    except Exception:
        pass

    raise TypeError(
        "Missing spline. Use distance_to_path(spline, x) / is_inside_path(spline, x), "
        "or ensure spline_path.py caches a spline via get_spline()."
    )


def _parse_spline_x(args, spline=None, x=None):
    # Explicit keyword usage
    if spline is not None and x is not None:
        return spline, np.asarray(x, dtype=float)

    # Two-arg usage: accept BOTH orders (spline, x) OR (x, spline)
    if len(args) == 2:
        a0, a1 = args
        if _is_spline(a0) and _looks_like_point(a1):
            return a0, np.asarray(a1, dtype=float)
        if _is_spline(a1) and _looks_like_point(a0):
            return a1, np.asarray(a0, dtype=float)
        # fall back: assume (spline, x)
        return a0, np.asarray(a1, dtype=float)

    # One-arg usage: interpreted as x; spline is default/cached
    if len(args) == 1:
        if x is None:
            x = args[0]
        if spline is None:
            spline = _get_default_spline()
        return spline, np.asarray(x, dtype=float)

    raise TypeError("Bad call. Use (x) or (spline, x) or (x, spline).")

import numpy as np

def _spline_pos(spline, s: float) -> np.ndarray:
    """Return position on spline, supporting either .pos(s) or .p(s)."""
    if hasattr(spline, "pos"):
        return np.asarray(spline.pos(float(s)), dtype=float)
    return np.asarray(spline.p(float(s)), dtype=float)

def _closest_s_refined(spline, x: np.ndarray) -> float:
    """
    Robust 1D projection onto spline:
    - coarse global sampling to get a good bracket
    - bounded scalar minimization inside bracket (SciPy)
    """
    x = np.asarray(x, dtype=float)
    L = float(getattr(spline, "length", 1.0))
    L = max(L, 1e-9)

    # coarse global search
    M = 400
    ss = np.linspace(0.0, L, M)
    d2 = np.empty(M, dtype=float)
    for i, s in enumerate(ss):
        p = _spline_pos(spline, s)
        v = p - x
        d2[i] = float(v[0]*v[0] + v[1]*v[1])
    i0 = int(np.argmin(d2))
    s0 = float(ss[i0])

    # bracket around best coarse point
    ds = float(ss[1] - ss[0]) if M > 1 else L
    lo = max(0.0, s0 - 2.0 * ds)
    hi = min(L, s0 + 2.0 * ds)
    if hi - lo < 1e-6:
        lo = max(0.0, s0 - 1.0)
        hi = min(L, s0 + 1.0)

    # refine via 1D optimization
    try:
        from scipy.optimize import minimize_scalar

        def obj(s):
            p = _spline_pos(spline, s)
            v = p - x
            return float(v[0]*v[0] + v[1]*v[1])

        res = minimize_scalar(
            obj,
            bounds=(lo, hi),
            method="bounded",
            options={"xatol": 1e-8},
        )
        if bool(getattr(res, "success", False)):
            return float(res.x)
    except Exception:
        pass

    return float(s0)



def _spline_pos(spline, s: float) -> np.ndarray:
    if hasattr(spline, "p"):
        return np.asarray(spline.p(float(s)), dtype=float)
    return np.asarray(spline.pos(float(s)), dtype=float)


def _spline_tangent_unit(spline, s: float) -> np.ndarray:
    t = np.asarray(spline.tangent(float(s)), dtype=float)
    n = float(np.linalg.norm(t)) + 1e-12
    return t / n


def _perp(v: np.ndarray) -> np.ndarray:
    return np.array([-v[1], v[0]], dtype=float)


def _get_path_width_pix() -> float:
    return float(getattr(config, "PATH_WIDTH_PIX", getattr(config, "PATH_WIDTH", 80.0)))


def _get_robot_radius() -> float:
    return float(getattr(config, "ROBOT_RADIUS", 6.0))


def distance_to_path(*args, spline=None, x=None) -> float:
    spline, x = _parse_spline_x(args, spline=spline, x=x)
    x = np.asarray(x, dtype=float)

    s_star = _closest_s_refined(spline, x)
    p_star = _spline_pos(spline, s_star)
    return float(np.linalg.norm(x - p_star))




def is_inside_path(*args, spline=None, x=None, path_width_pix=None, robot_radius=None) -> bool:
    spline, x = _parse_spline_x(args, spline=spline, x=x)

    W = float(_get_path_width_pix() if path_width_pix is None else path_width_pix)
    rr = float(_get_robot_radius() if robot_radius is None else robot_radius)
    inner = 0.5 * W - rr

    d = distance_to_path(spline, x)
    return bool(d <= inner)


def border_repulsion_forces(positions: np.ndarray, spline) -> np.ndarray:
    positions = np.asarray(positions, dtype=float)
    forces = np.zeros_like(positions)

    W = float(getattr(config, "PATH_WIDTH_PIX", getattr(config, "PATH_WIDTH", 80.0)))
    rr = float(getattr(config, "ROBOT_RADIUS", 6.0))
    k_border = float(getattr(config, "K_BORDER", 200.0))

    inner = 0.5 * W - rr
    if inner <= 1e-9:
        return forces  # corridor too thin -> don't explode

    # margin: MUST be < inner, or your dead zone disappears and "inside" gets force.
    margin_cfg = float(getattr(config, "BORDER_MARGIN_PIX", 3.0 * rr))
    margin = min(margin_cfg, 0.45 * inner)  # hard cap prevents "always on"
    margin = max(margin, 1.0)

    start = inner - margin  # <= start => zero force (dead zone)

    for i in range(positions.shape[0]):
        x = positions[i]

        # projection
        s = _closest_s_refined(spline, x)
        p = _spline_pos(spline, s)

        # tangent -> unit
        t = np.asarray(spline.tangent(s), dtype=float)
        tn = float(np.linalg.norm(t))
        if tn < 1e-12:
            continue
        t = t / tn

        # normal (one of the two)
        n = np.array([-t[1], t[0]], dtype=float)

        # signed lateral offset
        offset = float(np.dot(x - p, n))
        d = abs(offset)

        # dead zone: EXACT zero inside
        if d <= start:
            continue

        # direction toward centerline
        if abs(offset) < 1e-12:
            continue  # exactly on centerline -> no force
        dir_in = -np.sign(offset) * n

        # smooth ramp in the last 'margin' band
        penetration = d - start  # 0..margin (inside corridor)
        alpha = penetration / max(margin, 1e-9)
        mag = k_border * (alpha ** 2)

        forces[i] = dir_in * mag

    return forces


def border_repulsion_force(positions: np.ndarray, spline) -> np.ndarray:
    positions = np.asarray(positions, dtype=float)
    if positions.ndim == 1:
        return border_repulsion_forces(positions[None, :], spline)[0]
    return border_repulsion_forces(positions, spline)

