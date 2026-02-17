import numpy as np
import matplotlib.pyplot as plt

from src.test._test_utils import TestContext
from src.path.spline_path import build_spline_from_centerline
from src.swarm.spawn import spawn_swarm_state
from src.swarm.sim import simulate_swarm_oneway_A_to_B
from src import config


TEST_ID = "P4_STEP2_001"
TEST_NAME = "Swarm one-way A->B progress"
TEST_DESCRIPTION = "Uses per-robot spline targets and RK2 to ensure A-side robots make forward progress toward B."


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

    # spawn full swarm, but ONLY simulate the A-side half for this checkbox
    positions, velocities, groups = spawn_swarm_state(spline)
    N = positions.shape[0]
    half = N // 2

    posA = positions[:half]
    velA = velocities[:half]

    # choose steps so they should noticeably advance along the path
    dt = float(getattr(config, "DT", 0.05))
    vmax = float(getattr(config, "VMAX", 2.0))

    # rough travel estimate: distance per step ~ dt*vmax
    steps = int(np.ceil(0.5 * spline.length / max(1e-9, dt * vmax)))  # ~50% of the path
    steps = max(600, min(6000, steps))  # keep runtime reasonable

    ctx.info(f"Simulating A->B: robots={half}, dt={dt:.3f}, steps={steps}")

    traj = simulate_swarm_oneway_A_to_B(spline, posA, velA, steps=steps, dt=dt)

    # progress check: mean closest_s should increase a lot
    s0 = np.array([spline.closest_s(p) for p in traj[0]], dtype=float)
    s1 = np.array([spline.closest_s(p) for p in traj[-1]], dtype=float)

    mean0 = float(np.mean(s0))
    mean1 = float(np.mean(s1))

    ctx.info(f"Mean s: start={mean0:.2f} -> end={mean1:.2f} (L={spline.length:.2f})")

    if mean1 > mean0 + 0.25 * spline.length:
        ctx.pass_("A-side robots make strong forward progress toward B")
    else:
        ctx.fail("A-side robots did not make enough forward progress")
        ok = False

    # Save a plot (quick visual proof)
    path_pts = spline.sample(600)

    plt.figure(figsize=(7, 5))
    plt.plot(path_pts[:, 0], path_pts[:, 1], label="spline")

    for i in range(traj.shape[1]):
        plt.plot(traj[:, i, 0], traj[:, i, 1], linewidth=1)

    plt.title("Swarm one-way A->B (A-half only)")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    save_path = ctx.path("swarm_oneway_A_to_B.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    ctx.info(f"Saved: {save_path}")

    return ok
