import numpy as np
from src.path.path_controller import PathController
from src.path.spline_path import build_spline_from_centerline
from src.test._test_utils import TestContext, load_centerline_from_previous_step

TEST_ID = "P3_STEP5_001"
TEST_NAME = "Path controller functional validation"
TEST_DESCRIPTION = "Ensures controller returns valid path targets and progress behaves correctly."


def run(results_dir: str) -> bool:
    ctx = TestContext(results_dir)
    ok = True

    ctx.info(f"{TEST_ID} - {TEST_NAME}")

    # rebuild centerline + spline fresh
    centerline = load_centerline_from_previous_step()
    spline = build_spline_from_centerline(centerline)

    controller = PathController(spline)

    # place robot slightly off start of path
    robot = spline.p(0.0) + np.array([30.0, 0.0])
    target = controller.update(robot)

    # ---- checks ----

    if np.isfinite(target).all():
        ctx.pass_("Target is valid numeric")
    else:
        ctx.fail("Target contains invalid values")
        ok = False

    if isinstance(controller.s, float) or isinstance(controller.s, (int, np.floating)):
        ctx.pass_("Progress parameter valid")
    else:
        ctx.fail("Progress parameter invalid type")
        ok = False

    if controller.s >= 0.0:
        ctx.pass_("Progress initialized correctly")
    else:
        ctx.fail("Progress negative")
        ok = False

    ctx.info("Controller functional test complete")
    return ok
