import numpy as np

from src.test._test_utils import TestContext
from src.path.spline_path import build_spline_from_centerline
from src.path.border_forces import border_repulsion_force
from src import config


TEST_ID = "P4_STEP3_002"
TEST_NAME = "Border repulsion points inward"
TEST_DESCRIPTION = "Checks border force is ~0 inside the corridor and pushes inward near the border."


def run(results_dir: str) -> bool:
    ctx = TestContext(results_dir)
    ok = True

    ctx.info(f"{TEST_ID} - {TEST_NAME}")
    ctx.info(TEST_DESCRIPTION)

    spline = build_spline_from_centerline()
    L = float(spline.length)

    path_width = float(getattr(config, "PATH_WIDTH_PIX", getattr(config, "PATH_WIDTH", 60.0)))
    half_w = 0.5 * path_width

    # choose a mid-s location
    s = 0.5 * L
    c = spline.p(s)
    t = spline.tangent(s)
    n = np.array([-t[1], t[0]], dtype=float)  # unit normal

    # one point safely inside, one near border (outside allowed zone)
    inside = c + 0.2 * half_w * n
    near_border = c + 0.95 * half_w * n

    positions = np.vstack([inside, near_border])

    f = border_repulsion_force(positions, spline)

    fin = f[0]
    fb = f[1]

    ctx.info(f"Force inside: {fin}")
    ctx.info(f"Force near border: {fb}")

    # inside should be basically zero
    if float(np.linalg.norm(fin)) < 1e-6:
        ctx.pass_("Inside point gets ~0 border force")
    else:
        ctx.fail("Inside point got non-zero border force")
        ok = False

    # near border force should point inward (dot with outward normal should be negative)
    dot = float(np.dot(fb, n))
    ctx.info(f"dot(force, outward_normal) = {dot:.6f}")

    if dot < 0.0 and float(np.linalg.norm(fb)) > 1e-6:
        ctx.pass_("Near-border point gets inward push")
    else:
        ctx.fail("Near-border force did not push inward as expected")
        ok = False

    return ok
