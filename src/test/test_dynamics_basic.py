import numpy as np

from src.core.core import run_simulation
from src.visualization.viz import plot_trajectory
from src import config
from src.test._test_utils import TestContext

TEST_ID = "P2_CORE_001"
TEST_NAME = "Core simulation functional validation"
TEST_DESCRIPTION = "Validates basic motion, speed limits, multi-robot interaction, and trajectory sanity."


def run(results_dir: str) -> bool:
    ctx = TestContext(results_dir=results_dir)
    ok = True

    ctx.info(f"{TEST_ID} - {TEST_NAME}")
    ctx.info(TEST_DESCRIPTION)

    # =========================================================
    # TEST 1 — Single robot moves toward target
    # =========================================================
    target = np.array([10.0, 0.0], dtype=float)

    traj, vel = run_simulation(
        target=target,
        n_steps=200,
        n_robots=1,
        store_velocities=True
    )

    start = traj[0, 0]
    end = traj[-1, 0]

    dist_start = float(np.linalg.norm(target - start))
    dist_end = float(np.linalg.norm(target - end))

    if dist_end < dist_start:
        ctx.pass_(f"Robot approaches target ({dist_start:.3f} → {dist_end:.3f})")
    else:
        ctx.fail("Robot did not approach target")
        ok = False

    # trajectory sanity
    if np.all(np.isfinite(traj)):
        ctx.pass_("Trajectory contains valid numeric values")
    else:
        ctx.fail("Trajectory contains NaN/inf")
        ok = False

    plot_trajectory(traj, target, save_dir=results_dir, filename="core_single_robot.png")

    # =========================================================
    # TEST 2 — Speed constraint respected globally
    # =========================================================
    speeds = np.linalg.norm(vel[:, 0, :], axis=1)
    max_speed = float(np.max(speeds))

    ctx.info(f"Max speed observed: {max_speed:.6f}")

    if max_speed <= config.VMAX + 1e-9:
        ctx.pass_("Global speed limit respected")
    else:
        ctx.fail("Speed limit violated")
        ok = False

    # =========================================================
    # TEST 3 — Multi-robot repulsion prevents collapse
    # =========================================================
    start_positions = np.array([
        [0.0, 0.0],
        [0.2, 0.0]
    ], dtype=float)

    start_velocities = np.zeros((2, 2), dtype=float)

    traj2, _ = run_simulation(
        target=np.array([5.0, 0.0], dtype=float),
        n_steps=120,
        n_robots=2,
        start_positions=start_positions,
        start_velocities=start_velocities,
        store_velocities=True
    )

    d0 = float(np.linalg.norm(traj2[0, 0] - traj2[0, 1]))
    d1 = float(np.linalg.norm(traj2[-1, 0] - traj2[-1, 1]))

    ctx.info(f"Initial separation: {d0:.3f}")
    ctx.info(f"Final separation:   {d1:.3f}")

    if d1 > d0:
        ctx.pass_("Repulsion increases separation")
    else:
        ctx.fail("Repulsion failed to separate robots")
        ok = False

    # collision sanity
    min_dist = np.min(
        np.linalg.norm(traj2[:, 0, :] - traj2[:, 1, :], axis=1)
    )

    ctx.info(f"Minimum distance observed: {min_dist:.3f}")

    if min_dist > 0:
        ctx.pass_("No numerical collapse between robots")
    else:
        ctx.fail("Robots numerically collapsed")
        ok = False

    plot_trajectory(traj2, np.array([5.0, 0.0]), save_dir=results_dir, filename="core_multi_robot.png")

    # =========================================================
    # TEST 4 — Engine stability under longer run
    # =========================================================
    traj_long = run_simulation(
        target=np.array([20.0, 10.0], dtype=float),
        n_steps=400,
        n_robots=3
    )

    if np.all(np.isfinite(traj_long)):
        ctx.pass_("Engine stable over longer simulation")
    else:
        ctx.fail("Numerical instability detected")
        ok = False

    ctx.info("Core simulation functional test complete")

    return ok
