# src/path/border_forces.py

import numpy as np
from src.path.constraints import border_repulsion_force as _brf
from src.path.constraints import border_repulsion_forces as _brfs
from src.path.constraints import border_repulsion_forces, border_repulsion_force



def _is_spline(obj) -> bool:
    return (
        hasattr(obj, "closest_s")
        and hasattr(obj, "tangent")
        and (hasattr(obj, "p") or hasattr(obj, "pos"))
    )


def border_repulsion_force(a, b=None, **kwargs):
    """
    Flexible wrapper that supports both call styles:
      border_repulsion_force(spline, x)
      border_repulsion_force(x, spline)

    And supports x as:
      shape (2,)  -> returns (2,)
      shape (N,2) -> returns (N,2)
    """
    if b is None:
        raise TypeError("border_repulsion_force requires (spline, x) or (x, spline).")

    spline, x = (a, b) if _is_spline(a) else (b, a)
    x = np.asarray(x, dtype=float)

    if x.ndim == 1:
        return _brf(spline, x, **kwargs)

    if x.ndim == 2 and x.shape[1] == 2:
        return _brfs(x, spline, **kwargs)

    raise ValueError("x must be shape (2,) or (N,2).")


def border_repulsion_forces(positions, spline, **kwargs):
    return border_repulsion_force(positions, spline, **kwargs)
