# src/test/test_swarm_spawn.py

import numpy as np
import matplotlib.pyplot as plt

from src.test._test_utils import TestContext
from src.path.spline_path import build_spline_from_centerline
from src.swarm.spawn import spawn_swarm_positions

# Optional: corridor validation (if available)
try:
    from src.path.constraints import is_inside_path
except Exception:
    is_inside_path = None


TEST_ID = "P4_STEP1_001"
TEST_NAME = "Swarm spawn near A and B"
TEST_DESCRIPTION = "Spawns N/2 robots near start and N/2 near end of spline, with small randomization, and saves a debug plot."


def run(results_dir: str) -> bool:
    ctx = TestContext(results_dir)
    ok = True

    ctx.info(f"{TEST_ID} - {TEST_NAME}")
    ctx.info(TEST_DESCRIPTION)

    # Build spline
    try:
        spline = build_spline_from_centerline()
        ctx.pass_(f"Spline built (length={spline.length:.2f})")
    except Exception as e:
        ctx.fail(f"Spline build failed: {e}")
        return False

    # Spawn
    try:
        positions, groups = spawn_swarm_positions(spline)
        ctx.pass_(f"Spawned positions: {positions.shape}, groups: {groups.shape}")
    except Exception as e:
        ctx.fail(f"Spawn failed: {e}")
        return False

    # Shape checks
    N = positions.shape[0]
    if positions.ndim == 2 and positions.shape[1] == 2 and groups.shape == (N,):
        ctx.pass_("Shapes valid")
    else:
        ctx.fail("Invalid shapes for positions/groups")
        ok = False

    # Group split checks
    half = N // 2
    if np.all(groups[:half] == 1) and np.all(groups[half:] == -1):
        ctx.pass_("Groups correct (+1 then -1)")
    else:
        ctx.fail("Groups array incorrect")
        ok = False

    # Proximity sanity: first half closer to A than B, second half closer to B than A
    pA = spline.p(0.0)
    pB = spline.p(spline.length)

    dA = np.linalg.norm(positions - pA, axis=1)
    dB = np.linalg.norm(positions - pB, axis=1)

    if float(np.mean(dA[:half])) < float(np.mean(dB[:half])):
        ctx.pass_("First half is closer to A than B")
    else:
        ctx.fail("First half does not look closer to A")
        ok = False

    if float(np.mean(dB[half:])) < float(np.mean(dA[half:])):
        ctx.pass_("Second half is closer to B than A")
    else:
        ctx.fail("Second half does not look closer to B")
        ok = False

    # New: variation check (so we know randomization actually happened)
    spread = float(np.mean(np.std(positions, axis=0)))
    ctx.info(f"Spawn spread (mean std in x/y): {spread:.2f}")

    if spread > 1.0:
        ctx.pass_("Spawn has positional variation (not degenerate)")
    else:
        ctx.fail("Spawn variation too small (looks degenerate)")
        ok = False

    # Optional: corridor check (if available)
    if is_inside_path is not None:
        inside_flags = [bool(is_inside_path(p)) for p in positions]
        inside_count = int(sum(inside_flags))
        ctx.info(f"Inside-path count: {inside_count}/{N}")

        # We expect all of them to be inside; allow 1 to fail if your corridor is strict
        if inside_count >= N - 1:
            ctx.pass_("Spawn points are inside path corridor (or nearly all)")
        else:
            ctx.fail("Too many spawn points outside corridor")
            ok = False
    else:
        ctx.info("is_inside_path not available; skipping corridor validation")

    # Save debug plot (spline + spawns)
    pts = spline.sample(600)
    plt.figure(figsize=(7, 5))
    plt.plot(pts[:, 0], pts[:, 1], label="spline")

    plt.scatter(positions[:half, 0], positions[:half, 1], label="spawn A-side (+1)")
    plt.scatter(positions[half:, 0], positions[half:, 1], label="spawn B-side (-1)")

    plt.scatter([pA[0]], [pA[1]], marker="o", label="A")
    plt.scatter([pB[0]], [pB[1]], marker="x", label="B")

    plt.legend()
    plt.axis("equal")
    plt.grid(True)

    save_path = ctx.path("swarm_spawn_debug.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    ctx.info(f"Saved: {save_path}")

    return ok
