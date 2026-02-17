import numpy as np
from src.test._test_utils import TestContext
from src.swarm.collisions import detect_collisions
from src import config


TEST_ID = "P4_STEP4_001"
TEST_NAME = "Collision detection works"
TEST_DESCRIPTION = "Creates two robots closer than 2*robot_radius and verifies a collision is detected."


def run(results_dir: str) -> bool:
    ctx = TestContext(results_dir)
    ok = True

    r = float(getattr(config, "ROBOT_RADIUS", 6.0))
    thresh = 2.0 * r

    # two robots too close + one far
    positions = np.array([
        [0.0, 0.0],
        [0.9 * thresh, 0.0],   # collision with robot 0
        [10.0 * thresh, 0.0],  # far away
    ], dtype=float)

    pairs = detect_collisions(positions, threshold=thresh)

    ctx.info(f"Detected pairs: {pairs}")

    if len(pairs) == 1 and pairs[0][0] == 0 and pairs[0][1] == 1:
        ctx.pass_("Detected exactly the expected collision pair (0,1)")
    else:
        ctx.fail("Collision detection did not match expectation")
        ok = False

    return ok
