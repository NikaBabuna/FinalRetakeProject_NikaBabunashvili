import numpy as np
import matplotlib.pyplot as plt

from src.test._test_utils import TestContext
from src.path.spline_path import build_spline_from_centerline
from src.swarm.spawn import spawn_swarm_state
from src.swarm.sim import simulate_swarm_oneway_B_to_A
from src import config


TEST_ID = "P4_STEP2_002"
TEST_NAME = "Swarm one-way B->A progress"
TEST_DESCRIPTION = "Ensures B-side robots make backward progress toward A using reversed spline controller."


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

    # spawn full swarm, but ONLY simulate the B-side half for this checkbox
    positions, velocities, groups = spawn_swarm_state(spline)
    N = positions.shape[0]
    half = N // 2

    posB = positions[half:]
    velB = velocities[half:]

    dt = float(getattr(config, "DT", 0.05))
    vmax = float(getattr(config, "VMAX", 2.0))

    steps = int(np.ceil(0.5 * spline.length / max(1e-9, dt * vmax)))
    steps = max(600, min(6000, steps))

    ctx.info(f"Simulating B->A: robots={N-half}, dt={dt:.3f}, steps={steps}")

    traj = simulate_swarm_oneway_B_to_A(spline, posB, velB, steps=steps, dt=dt)

    # progress check in ORIGINAL spline coordinates: mean closest_s should DECREASE a lot
    s0 = np.array([spline.closest_s(p) for p in traj[0]], dtype=float)
    s1 = np.array([spline.closest_s(p) for p in traj[-1]], dtype=float)

    mean0 = float(np.mean(s0))
    mean1 = float(np.mean(s1))

    ctx.info(f"Mean s (original): start={mean0:.2f} -> end={mean1:.2f} (L={spline.length:.2f})")

    if mean1 < mean0 - 0.25 * spline.length:
        ctx.pass_("B-side robots make strong backward progress toward A")
    else:
        ctx.fail("B-side robots did not make enough backward progress")
        ok = False

    # Save a plot (visual proof)
    path_pts = spline.sample(600)

    plt.figure(figsize=(7, 5))
    plt.plot(path_pts[:, 0], path_pts[:, 1], label="spline")

    for i in range(traj.shape[1]):
        plt.plot(traj[:, i, 0], traj[:, i, 1], linewidth=1)

    plt.title("Swarm one-way B->A (B-half only)")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    save_path = ctx.path("swarm_oneway_B_to_A.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    ctx.info(f"Saved: {save_path}")

    return ok
