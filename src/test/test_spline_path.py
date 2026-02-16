import numpy as np
import matplotlib.pyplot as plt

from src.test._test_utils import TestContext
from src.path.spline_path import build_spline_from_centerline


TEST_ID = "P3_STEP4_001"
TEST_NAME = "Spline path functional validation"
TEST_DESCRIPTION = "Builds spline from extracted centerline and validates p(s), tangent(s), closest_s(x)."


def run(results_dir: str) -> bool:
    ctx = TestContext(results_dir=results_dir)
    ok = True

    ctx.info(f"{TEST_ID} - {TEST_NAME}")
    ctx.info(TEST_DESCRIPTION)

    # Build spline from current map + A/B + mask + centerline
    try:
        from src.test._test_utils import load_centerline_from_previous_step

        centerline = load_centerline_from_previous_step()
        path = build_spline_from_centerline(centerline)

        ctx.pass_(f"Spline built (length={path.length:.2f})")
    except Exception as e:
        ctx.fail(f"Spline build failed: {e}")
        return False

    if path.length > 10:
        ctx.pass_("Spline length looks reasonable")
    else:
        ctx.fail("Spline length too small (unexpected)")
        ok = False

    # Check position endpoints
    p0 = path.pos(0.0)
    pL = path.pos(path.length)

    if np.all(np.isfinite(p0)) and np.all(np.isfinite(pL)):
        ctx.pass_("Endpoint positions are finite")
    else:
        ctx.fail("Endpoint positions contain NaN/inf")
        ok = False

    # Tangent normalization check
    for s in [0.0, path.length * 0.25, path.length * 0.5, path.length * 0.9]:
        t = path.tangent(s)
        n = float(np.linalg.norm(t))
        if abs(n - 1.0) < 1e-3:
            ctx.pass_(f"Tangent normalized at s={s:.2f}")
        else:
            ctx.fail(f"Tangent not normalized at s={s:.2f} (norm={n})")
            ok = False
            break

    # closest_s sanity: pick a point on the path, offset it, project back
    s_test = path.length * 0.6
    x_on = path.pos(s_test)
    x_off = x_on + np.array([15.0, -10.0])
    s_proj = path.closest_s(x_off)

    if 0.0 <= s_proj <= path.length:
        ctx.pass_("closest_s returns valid s in [0, L]")
    else:
        ctx.fail("closest_s returned s outside [0, L]")
        ok = False

    # Debug plot: sampled spline in world coords
    pts = path.sample(400)

    plt.figure()
    plt.plot(pts[:, 0], pts[:, 1])
    plt.scatter([p0[0]], [p0[1]], marker="o")
    plt.scatter([pL[0]], [pL[1]], marker="x")
    plt.title("Spline path (world coords, origin at A)")
    plt.axis("equal")
    plt.grid(True)

    save_path = ctx.path("spline_path_world.png")
    plt.savefig(save_path, dpi=200)
    plt.close()

    ctx.info(f"Saved spline plot: {save_path}")
    return ok
