import numpy as np

from src.test._test_utils import TestContext
from src.path.spline_path import build_spline_from_centerline
from src.path.path_controller import PathController
from src.core.core import step_rk2
from src import config


TEST_ID = "P4_STEP3_001"
TEST_NAME = "Swarm repulsion turns on"
TEST_DESCRIPTION = "Places two robots too close together and verifies repulsion increases separation."


def run(results_dir: str) -> bool:
    ctx = TestContext(results_dir)
    ok = True

    ctx.info(f"{TEST_ID} - {TEST_NAME}")
    ctx.info(TEST_DESCRIPTION)

    # 1) Build spline
    try:
        spline = build_spline_from_centerline()
        ctx.pass_(f"Spline built (length={spline.length:.2f})")
    except Exception as e:
        ctx.fail(f"Spline build failed: {e}")
        return False

    # 2) Basic config sanity
    if float(getattr(config, "KREP", 0.0)) <= 0.0:
        ctx.fail("config.KREP must be > 0 to enable repulsion")
        return False
    if float(getattr(config, "RSAFE", 0.0)) <= 0.0:
        ctx.fail("config.RSAFE must be > 0 to enable repulsion")
        return False

    # 3) Two robots: same target, only repulsion should separate them
    s0 = 0.55 * float(spline.length)
    p = spline.p(s0)

    rsafe = float(getattr(config, "RSAFE", 10.0))
    sep0 = 0.35 * rsafe  # deliberately too close (< RSAFE)

    positions = np.zeros((2, 2), dtype=float)
    velocities = np.zeros((2, 2), dtype=float)

    positions[0] = p + np.array([-0.5 * sep0, 0.0])
    positions[1] = p + np.array([+0.5 * sep0, 0.0])

    controller = PathController(spline)
    kp = float(getattr(config, "KP", 1.2))
    dt = float(getattr(config, "DT", 0.05))

    dist_start = float(np.linalg.norm(positions[0] - positions[1]))
    ctx.info(f"Start distance: {dist_start:.3f} (RSAFE={rsafe:.3f})")

    # 4) Sim a bit
    steps = 80
    for _ in range(steps):
        # same target for both (so attraction doesn't separate them)
        target = controller.update(positions[0])
        targets = np.vstack([target, target])

        u = kp * (targets - positions)          # (N,2) external forces
        positions, velocities = step_rk2(positions, velocities, u, dt)

    dist_end = float(np.linalg.norm(positions[0] - positions[1]))
    ctx.info(f"End distance:   {dist_end:.3f}")

    # 5) Pass condition: separation increases noticeably
    if dist_end > dist_start + 0.25 * rsafe:
        ctx.pass_("Repulsion increased separation (frep is ON)")
    else:
        ctx.fail("Separation did not increase enough — repulsion too weak or not applied")
        ok = False

    return ok
