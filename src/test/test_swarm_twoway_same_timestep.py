import numpy as np
import matplotlib.pyplot as plt

from src.test._test_utils import TestContext
from src.path.spline_path import build_spline_from_centerline
from src.swarm.spawn import spawn_swarm_state
from src.swarm.sim import simulate_swarm_twoway


TEST_ID = "P4_STEP2_003"
TEST_NAME = "Two-way swarm starts at same timestep"
TEST_DESCRIPTION = "Runs a single simulation step and confirms both A-group and B-group moved at t=0->1."


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

    positions, velocities, groups = spawn_swarm_state(spline)

    # run ONLY 1 step - this test is specifically about "same timestep"
    traj, collision_log = simulate_swarm_twoway(spline, positions, velocities, groups, steps=1, log_collisions=True)

    p0 = traj[0]
    p1 = traj[1]

    moved = np.linalg.norm(p1 - p0, axis=1)
    moved_A = moved[groups == 1]
    moved_B = moved[groups == -1]

    ctx.info(f"Mean move A-group: {float(np.mean(moved_A)):.6f}")
    ctx.info(f"Mean move B-group: {float(np.mean(moved_B)):.6f}")

    if np.all(moved_A > 0):
        ctx.pass_("All A->B robots moved on first step")
    else:
        ctx.fail("Some A->B robots did NOT move on first step")
        ok = False

    if np.all(moved_B > 0):
        ctx.pass_("All B->A robots moved on first step")
    else:
        ctx.fail("Some B->A robots did NOT move on first step")
        ok = False

    # quick debug plot: start and after 1 step
    plt.figure(figsize=(7, 5))
    path_pts = spline.sample(600)
    plt.plot(path_pts[:, 0], path_pts[:, 1], label="spline")

    plt.scatter(p0[groups == 1, 0], p0[groups == 1, 1], label="A-group t0")
    plt.scatter(p0[groups == -1, 0], p0[groups == -1, 1], label="B-group t0")

    plt.scatter(p1[groups == 1, 0], p1[groups == 1, 1], marker="x", label="A-group t1")
    plt.scatter(p1[groups == -1, 0], p1[groups == -1, 1], marker="x", label="B-group t1")

    plt.title("Two-way start: t0 vs t1")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    save_path = ctx.path("swarm_twoway_t0_t1.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    ctx.info(f"Saved: {save_path}")

    return ok
