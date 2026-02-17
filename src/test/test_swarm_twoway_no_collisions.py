import os
import numpy as np
import matplotlib.pyplot as plt

from src.test._test_utils import TestContext
from src.path.spline_path import build_spline_from_centerline
from src.swarm.sim import simulate_swarm_twoway

# spawn function name differs across versions, so we support both
try:
    from src.swarm.spawn import spawn_swarm_positions as _spawn
except Exception:
    try:
        from src.swarm.spawn import spawn_swarm_near_A_B as _spawn
    except Exception:
        _spawn = None

from src import config
from src.swarm.collisions import detect_collisions


TEST_ID = "P4_STEP4_001"
TEST_NAME = "Two-way swarm: progress + no collisions"
TEST_DESCRIPTION = "Runs A->B and B->A together. Checks both groups move in correct direction and no collisions occur."


def run(results_dir: str) -> bool:
    ctx = TestContext(results_dir)
    ok = True

    ctx.info(f"{TEST_ID} - {TEST_NAME}")
    ctx.info(TEST_DESCRIPTION)

    if _spawn is None:
        ctx.fail("Could not import a swarm spawn function from src.swarm.spawn")
        return False

    # Build spline
    try:
        spline = build_spline_from_centerline()
        L = float(spline.length)
        ctx.pass_(f"Spline built (length={L:.2f})")
    except Exception as e:
        ctx.fail(f"Spline build failed: {e}")
        return False

    # Choose a reasonable N + sim length
    N = int(getattr(config, "N_ROBOTS", 24))
    dt = float(getattr(config, "DT", 0.05))
    vmax = float(getattr(config, "VMAX", 2.0))

    # steps ~ enough for them to meet and pass
    steps = int((L / max(1e-6, vmax)) / dt) * 2
    steps = int(np.clip(steps, 400, 9000))

    ctx.info(f"Sim: N={N}, dt={dt:.3f}, steps={steps}")

    # Spawn
    spawned = _spawn(spline, N=N) if "N" in _spawn.__code__.co_varnames else _spawn(spline)
    if isinstance(spawned, (tuple, list)) and len(spawned) == 3:
        positions, velocities, groups = spawned
    else:
        positions, groups = spawned
        velocities = np.zeros_like(positions)

    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    groups = np.asarray(groups, dtype=int)

    if positions.shape != (N, 2) or velocities.shape != (N, 2) or groups.shape != (N,):
        ctx.fail(f"Bad spawn shapes: pos={positions.shape}, vel={velocities.shape}, groups={groups.shape}")
        return False

    # Run two-way sim
    try:
        traj = simulate_swarm_twoway(
            spline=spline,
            positions=positions,
            velocities=velocities,
            groups=groups,
            steps=steps,
            dt=dt
        )
        traj = np.asarray(traj, dtype=float)
        ctx.pass_(f"Simulated traj: {traj.shape}")
    except Exception as e:
        ctx.fail(f"simulate_swarm_twoway failed: {e}")
        return False

    if traj.ndim != 3 or traj.shape[1:] != (N, 2):
        ctx.fail(f"Trajectory has wrong shape: {traj.shape}")
        return False

    # Progress check (mean s increases for +1, decreases for -1)
    def mean_s(frame_positions, idx):
        ss = [spline.closest_s(frame_positions[i]) for i in idx]
        return float(np.mean(ss)) if len(ss) else 0.0

    idx_pos = np.where(groups == 1)[0]
    idx_neg = np.where(groups == -1)[0]

    s_pos_0 = mean_s(traj[0], idx_pos)
    s_pos_1 = mean_s(traj[-1], idx_pos)
    s_neg_0 = mean_s(traj[0], idx_neg)
    s_neg_1 = mean_s(traj[-1], idx_neg)

    ctx.info(f"Mean s (+1): start={s_pos_0:.2f} -> end={s_pos_1:.2f} (L={L:.2f})")
    ctx.info(f"Mean s (-1): start={s_neg_0:.2f} -> end={s_neg_1:.2f} (L={L:.2f})")

    if s_pos_1 > s_pos_0 + 0.15 * L:
        ctx.pass_("A->B group made strong forward progress")
    else:
        ctx.fail("A->B group progress is too small")
        ok = False

    if s_neg_1 < s_neg_0 - 0.15 * L:
        ctx.pass_("B->A group made strong backward progress")
    else:
        ctx.fail("B->A group progress is too small")
        ok = False

    # Collision check (sample frames to keep it fast)
    check_stride = max(1, steps // 300)  # ~300 checks max
    total_collisions = 0
    min_pair_dist = float("inf")
    t_min = 0

    for t in range(0, steps + 1, check_stride):
        pos = traj[t]
        pairs = detect_collisions(pos)
        total_collisions += len(pairs)

        # track min distance (from reported pairs if any, else skip)
        for (_, _, d) in pairs:
            if d < min_pair_dist:
                min_pair_dist = float(d)
                t_min = t

    ctx.info(f"Collision events (sampled): {total_collisions}")
    if total_collisions == 0:
        ctx.pass_("No collisions detected (sampled frames)")
    else:
        ctx.fail(f"Collisions detected! min d={min_pair_dist:.2f} at t={t_min}")
        ok = False

    # Save a quick plot: mean s over time (sampled)
    ts = list(range(0, steps + 1, check_stride))
    s_pos_series = []
    s_neg_series = []

    for t in ts:
        s_pos_series.append(mean_s(traj[t], idx_pos))
        s_neg_series.append(mean_s(traj[t], idx_neg))

    plt.figure(figsize=(7, 4))
    plt.plot(ts, s_pos_series, label="mean s (+1)")
    plt.plot(ts, s_neg_series, label="mean s (-1)")
    plt.axhline(0, linewidth=1)
    plt.axhline(L, linewidth=1)
    plt.grid(True)
    plt.legend()
    plt.title("Two-way progress (mean s)")

    save_path = ctx.path("twoway_progress_mean_s.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    ctx.info(f"Saved: {save_path}")

    return ok
