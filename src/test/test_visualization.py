import os
import numpy as np

from src.test._test_utils import TestContext
from src.path.spline_path import build_spline_from_centerline

TEST_ID = "P3_STEP7_001"
TEST_NAME = "Visualization layer validation"
TEST_DESCRIPTION = "Renders map + spline + borders + trajectory and saves an image."


def _import_renderer():
    """
    Support either location:
      - src.core.render_path_following
      - src.viz.render_path_following
    """
    try:
        from src.core.core import render_path_following
        return render_path_following
    except Exception:
        pass

    try:
        from src.visualization.viz import render_path_following
        return render_path_following
    except Exception:
        pass

    raise ImportError(
        "Could not import render_path_following from src.core or src.viz. "
        "Put render_path_following(spline, trajectory, save_path) in one of them."
    )


def run(results_dir: str) -> bool:
    ctx = TestContext(results_dir)
    ok = True

    ctx.info(f"{TEST_ID} - {TEST_NAME}")
    ctx.info(TEST_DESCRIPTION)

    # 1) Build spline from pipeline
    try:
        spline = build_spline_from_centerline()
        ctx.pass_(f"Spline built (length={spline.length:.2f})")
    except Exception as e:
        ctx.fail(f"Spline build failed: {e}")
        return False

    # 2) Create a valid trajectory (T,2) in same coordinate frame as spline
    # Easiest reliable trajectory for visualization: follow the spline itself.
    s_vals = np.linspace(0.0, spline.length, 250)
    traj = np.array([spline.p(s) for s in s_vals], dtype=float)

    # small offset so it’s not perfectly identical (optional)
    traj = traj + np.array([2.0, -1.0])

    if traj.ndim == 2 and traj.shape[1] == 2 and np.isfinite(traj).all():
        ctx.pass_("Trajectory prepared (T,2) and finite")
    else:
        ctx.fail("Trajectory invalid shape or contains NaN/inf")
        return False

    # 3) Render
    try:
        render_path_following = _import_renderer()
        save_path = ctx.path("final_visualization.gif")
        render_path_following(spline, traj, save_path)
        ctx.pass_("render_path_following() executed")
    except Exception as e:
        ctx.fail(f"Rendering failed: {e}")
        return False

    # 4) Validate output file
    if os.path.isfile(save_path) and os.path.getsize(save_path) > 10_000:
        ctx.pass_(f"Visualization saved: {save_path}")
    else:
        ctx.fail("Visualization file missing or too small (likely blank/failed save)")
        ok = False

    return ok
