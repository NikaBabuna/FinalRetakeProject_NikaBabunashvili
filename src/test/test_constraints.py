import numpy as np
import matplotlib.pyplot as plt

from src.test._test_utils import TestContext
from src.path.constraints import distance_to_path, is_inside_path
from src.path.spline_path import build_spline_from_centerline
from src import config


TEST_ID = "P3_STEP5_001"
TEST_NAME = "Path constraint validation"
TEST_DESCRIPTION = "Checks distance-to-path and inside-path logic."


def run(results_dir: str) -> bool:
    ctx = TestContext(results_dir=results_dir)
    ok = True

    ctx.info(f"{TEST_ID} - {TEST_NAME}")
    ctx.info(TEST_DESCRIPTION)

    # ------------------------------------------------
    # Build spline
    # ------------------------------------------------
    try:
        spline = build_spline_from_centerline()
        ctx.pass_(f"Spline built (length={spline.length:.2f})")
    except Exception as e:
        ctx.fail(f"Spline build failed: {e}")
        return False

    # ------------------------------------------------
    # Pick a point ON path
    # ------------------------------------------------
    s_mid = spline.length * 0.5
    p_on = spline.pos(s_mid)

    d = distance_to_path(p_on)

    if d < 1e-3:
        ctx.pass_("Distance near zero for point on path")
    else:
        ctx.fail(f"Distance not near zero: {d}")
        ok = False

    if is_inside_path(p_on):
        ctx.pass_("Point on path is inside")
    else:
        ctx.fail("Point on path reported outside")
        ok = False

    # ------------------------------------------------
    # Point slightly offset but still inside
    # ------------------------------------------------
    offset_inside = p_on + np.array([5.0, 0.0])
    d2 = distance_to_path(offset_inside)

    ctx.info(f"Distance (offset inside) = {d2:.2f}")

    if is_inside_path(offset_inside):
        ctx.pass_("Offset point still inside")
    else:
        ctx.fail("Offset point incorrectly outside")
        ok = False

    # ------------------------------------------------
    # Point clearly outside
    # ------------------------------------------------
    offset_out = p_on + np.array([config.PATH_WIDTH * 2.0, 0.0])

    if not is_inside_path(offset_out):
        ctx.pass_("Far point correctly detected outside")
    else:
        ctx.fail("Far point incorrectly inside")
        ok = False

    # ------------------------------------------------
    # Debug plot
    # ------------------------------------------------
    pts = spline.sample(400)

    plt.figure(figsize=(6, 6))
    plt.plot(pts[:, 0], pts[:, 1], label="centerline")

    plt.scatter(*p_on, label="on path")
    plt.scatter(*offset_inside, label="inside")
    plt.scatter(*offset_out, label="outside")

    plt.axis("equal")
    plt.legend()
    plt.grid(True)

    save_path = ctx.path("constraints_debug.png")
    plt.savefig(save_path, dpi=200)
    plt.close()

    ctx.info(f"Saved debug plot: {save_path}")
    return ok
