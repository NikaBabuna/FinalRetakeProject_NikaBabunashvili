import os
import numpy as np

from src.test._test_utils import TestContext
from src.path.spline_path import build_spline_from_centerline
from src.swarm.sim import simulate_swarm_twoway
from src import config

try:
    from src.swarm.viz_swarm import render_swarm_twoway_gif
except Exception:
    render_swarm_twoway_gif = None

try:
    from src.swarm.spawn import spawn_swarm_positions as _spawn
except Exception:
    try:
        from src.swarm.spawn import spawn_swarm_near_A_B as _spawn
    except Exception:
        _spawn = None


TEST_ID = "P4_STEP4_003"
TEST_NAME = "Two-way swarm: GIF export"
TEST_DESCRIPTION = "Creates a short two-way swarm GIF and checks it exists and is not tiny."


def run(results_dir: str) -> bool:
    ctx = TestContext(results_dir)
    ok = True

    ctx.info(f"{TEST_ID} - {TEST_NAME}")
    ctx.info(TEST_DESCRIPTION)

    if render_swarm_twoway_gif is None:
        ctx.fail("render_swarm_twoway_gif not importable from src.swarm.viz_swarm")
        return False
    if _spawn is None:
        ctx.fail("Could not import spawn function from src.swarm.spawn")
        return False

    spline = build_spline_from_centerline()
    L = float(spline.length)
    ctx.pass_(f"Spline built (length={L:.2f})")

    # keep this test light
    N = int(min(24, getattr(config, "N_ROBOTS", 24)))
    dt = float(getattr(config, "DT", 0.05))
    vmax = float(getattr(config, "VMAX", 2.0))

    steps = int((L / max(1e-6, vmax)) / dt)  # one crossing time
    steps = int(np.clip(steps, 250, 2500))
    ctx.info(f"Sim: N={N}, dt={dt:.3f}, steps={steps}")

    spawned = _spawn(spline, N=N) if "N" in _spawn.__code__.co_varnames else _spawn(spline)
    if isinstance(spawned, (tuple, list)) and len(spawned) == 3:
        positions, velocities, groups = spawned
    else:
        positions, groups = spawned
        velocities = np.zeros_like(positions)

    traj = simulate_swarm_twoway(spline, positions, velocities, groups, steps=steps, dt=dt)

    gif_path = ctx.path(f"twoway_gif_export_N{N}.gif")
    render_swarm_twoway_gif(
        spline=spline,
        traj=traj,
        groups=groups,
        save_path=gif_path,
        stride=max(2, steps // 200),  # cap frames
        fps=20,
        trail=35,
        show_corridor=True,
        show_map=True,
        show_axes=False,
        title=f"Two-way swarm (N={N})"
    )

    if os.path.isfile(gif_path):
        size = os.path.getsize(gif_path)
        ctx.info(f"GIF size: {size} bytes")
        if size > 50_000:
            ctx.pass_("GIF exported and looks non-trivial (size > 50KB)")
        else:
            ctx.fail("GIF exists but is too small (might be blank)")
            ok = False
    else:
        ctx.fail("GIF not created")
        ok = False

    return ok
