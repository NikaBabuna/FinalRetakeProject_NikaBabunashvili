import numpy as np
from src.core import run_simulation
from src.viz import plot_trajectory
from src import config
from src.test.testing_utils import TestContext

TEST_ID = "P2_CORE_001"
TEST_NAME = "Phase 2 core validation"
TEST_DESCRIPTION = "Checks: moves toward target, respects VMAX (true velocities), repulsion increases separation."


def run(results_dir: str) -> bool:
    ctx = TestContext(results_dir=results_dir)
    ok = True

    ctx.info(f"{TEST_ID} - {TEST_NAME}")
    ctx.info(TEST_DESCRIPTION)

    # ----------------------------
    # TEST 1: single robot moves toward target
    # ----------------------------
    target = np.array([10.0, 0.0], dtype=float)
    traj, vel = run_simulation(target=target, n_steps=200, n_robots=1, store_velocities=True)

    start = traj[0, 0]
    end = traj[-1, 0]
    dist_start = float(np.linalg.norm(target - start))
    dist_end = float(np.linalg.norm(target - end))

    ctx.metrics["dist_start"] = dist_start
    ctx.metrics["dist_end"] = dist_end

    if dist_end < dist_start:
        ctx.pass_(f"Robot moved toward target (dist {dist_start:.3f} -> {dist_end:.3f})")
    else:
        ctx.fail(f"Robot did NOT move toward target (dist {dist_start:.3f} -> {dist_end:.3f})")
        ok = False

    plot_trajectory(traj, target, save_dir=results_dir, filename="test1_single_robot.png")

    # ----------------------------
    # TEST 2: speed never exceeds VMAX (use true velocities)
    # ----------------------------
    speeds = np.linalg.norm(vel[:, 0, :], axis=1)
    max_speed = float(np.max(speeds))
    ctx.metrics["max_speed"] = max_speed
    ctx.metrics["vmax"] = float(config.VMAX)

    if max_speed <= config.VMAX + 1e-9:
        ctx.pass_(f"Speed limit respected (max {max_speed:.6f} <= VMAX {config.VMAX:.6f})")
    else:
        ctx.fail(f"Speed exceeded VMAX (max {max_speed:.6f} > VMAX {config.VMAX:.6f})")
        ok = False

    # ----------------------------
    # TEST 3: repulsion increases separation
    # ----------------------------
    start_positions = np.array([[0.0, 0.0], [0.2, 0.0]], dtype=float)
    start_velocities = np.zeros((2, 2), dtype=float)

    traj2, _vel2 = run_simulation(
        target=np.array([5.0, 0.0], dtype=float),
        n_steps=100,
        n_robots=2,
        start_positions=start_positions,
        start_velocities=start_velocities,
        store_velocities=True
    )

    d0 = float(np.linalg.norm(traj2[0, 0] - traj2[0, 1]))
    d1 = float(np.linalg.norm(traj2[-1, 0] - traj2[-1, 1]))
    ctx.metrics["initial_sep"] = d0
    ctx.metrics["final_sep"] = d1

    if d1 > d0:
        ctx.pass_(f"Repulsion increased separation ({d0:.3f} -> {d1:.3f})")
    else:
        ctx.fail(f"Repulsion did NOT increase separation ({d0:.3f} -> {d1:.3f})")
        ok = False

    plot_trajectory(traj2, np.array([5.0, 0.0], dtype=float), save_dir=results_dir, filename="test2_two_robots.png")

    # Final metrics print (nice for report)
    ctx.info(f"METRICS: {ctx.metrics}")

    return ok
