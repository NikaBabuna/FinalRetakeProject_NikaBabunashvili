import numpy as np
import matplotlib.pyplot as plt

from src.test._test_utils import TestContext
from src.path.spline_path import build_spline_from_centerline
from src.swarm.sim import simulate_swarm_twoway
from src import config

try:
    from src.path.constraints import is_inside_path
except Exception:
    is_inside_path = None

try:
    from src.swarm.spawn import spawn_swarm_positions as _spawn
except Exception:
    try:
        from src.swarm.spawn import spawn_swarm_near_A_B as _spawn
    except Exception:
        _spawn = None


TEST_ID = "P4_STEP4_002"
TEST_NAME = "Two-way swarm: stays in corridor"
TEST_DESCRIPTION = "Runs a short two-way sim and checks most robot samples are inside corridor."


def run(results_dir: str) -> bool:
    ctx = TestContext(results_dir)
    ok = True

    ctx.info(f"{TEST_ID} - {TEST_NAME}")
    ctx.info(TEST_DESCRIPTION)

    if is_inside_path is None:
        ctx.fail("is_inside_path not available from src.path.constraints")
        return False
    if _spawn is None:
        ctx.fail("Could not import spawn function from src.swarm.spawn")
        return False

    spline = build_spline_from_centerline()
    L = float(spline.length)
    ctx.pass_(f"Spline built (length={L:.2f})")

    N = int(getattr(config, "N_ROBOTS", 24))
    dt = float(getattr(config, "DT", 0.05))
    vmax = float(getattr(config, "VMAX", 2.0))

    steps = int((L / max(1e-6, vmax)) / dt)  # one crossing time
    steps = int(np.clip(steps, 300, 6000))
    ctx.info(f"Sim: N={N}, dt={dt:.3f}, steps={steps}")

    spawned = _spawn(spline, N=N) if "N" in _spawn.__code__.co_varnames else _spawn(spline)
    if isinstance(spawned, (tuple, list)) and len(spawned) == 3:
        positions, velocities, groups = spawned
    else:
        positions, groups = spawned
        velocities = np.zeros_like(positions)

    traj = simulate_swarm_twoway(spline, positions, velocities, groups, steps=steps, dt=dt)
    traj = np.asarray(traj, dtype=float)

    # sample a limited number of frames
    frame_stride = max(1, steps // 200)
    inside = 0
    total = 0

    outside_counts_per_frame = []

    for t in range(0, steps + 1, frame_stride):
        pos = traj[t]
        flags = [bool(is_inside_path(p)) for p in pos]
        inside_t = int(sum(flags))
        total_t = len(flags)

        inside += inside_t
        total += total_t
        outside_counts_per_frame.append(total_t - inside_t)

    frac_inside = inside / max(1, total)
    ctx.info(f"Inside ratio (sampled): {frac_inside*100:.2f}%")

    # allow small tolerance because corridor logic can be strict at edges
    if frac_inside >= 0.98:
        ctx.pass_("Swarm stays inside corridor (>=98% of sampled points inside)")
    else:
        ctx.fail("Too many outside-corridor samples")
        ok = False

    # plot outside count per sampled frame
    plt.figure(figsize=(7, 4))
    plt.plot(outside_counts_per_frame)
    plt.grid(True)
    plt.title("Outside count per sampled frame")
    plt.xlabel("sample index")
    plt.ylabel("# robots outside")

    save_path = ctx.path("twoway_outside_counts.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    ctx.info(f"Saved: {save_path}")

    return ok
