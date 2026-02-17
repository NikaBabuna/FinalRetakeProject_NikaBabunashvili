import numpy as np

from src.test._test_utils import TestContext
from src.path.spline_path import build_spline_from_centerline
from src.swarm.spawn import spawn_swarm_state


TEST_ID = "P4_STEP1_002"
TEST_NAME = "Swarm initial velocities are zero"
TEST_DESCRIPTION = "Ensures swarm initialization sets all robot velocities to exactly zero."


def run(results_dir: str) -> bool:
    ctx = TestContext(results_dir)
    ok = True

    ctx.info(f"{TEST_ID} - {TEST_NAME}")
    ctx.info(TEST_DESCRIPTION)

    try:
        spline = build_spline_from_centerline()
        ctx.pass_(f"Spline built (length={spline.length:.2f})")
    except Exception as e:
        ctx.fail(f"Spline build failed: {e}")
        return False

    try:
        positions, velocities, groups = spawn_swarm_state(spline)
        ctx.pass_(f"Spawned positions {positions.shape}, velocities {velocities.shape}, groups {groups.shape}")
    except Exception as e:
        ctx.fail(f"spawn_swarm_state failed: {e}")
        return False

    # shape checks
    if velocities.shape == positions.shape and velocities.ndim == 2 and velocities.shape[1] == 2:
        ctx.pass_("Velocity shape matches (N,2)")
    else:
        ctx.fail(f"Velocity shape invalid: {velocities.shape}")
        ok = False

    # zero check
    max_abs = float(np.max(np.abs(velocities)))
    ctx.info(f"Max abs initial velocity: {max_abs:.6f}")

    if max_abs == 0.0:
        ctx.pass_("All initial velocities are exactly zero")
    else:
        ctx.fail("Initial velocities are not all zero")
        ok = False

    return ok
