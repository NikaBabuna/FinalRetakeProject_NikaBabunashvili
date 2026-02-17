import numpy as np

from src import config
from src.path.spline_path import build_spline_from_centerline
from src.swarm.spawn import spawn_swarm_near_A_B
from src.swarm.sim import simulate_swarm_twoway
from src.utils.io_utils import ensure_output_root, create_new_results_dir
from src.visualization.viz import render_path_following  # you already have this


def required_steps(length: float, dt: float, vmax: float, safety: float = 1.35) -> int:
    max_per_step = max(1e-9, dt * vmax)
    base = int(np.ceil(length / max_per_step))
    return int(np.ceil(base * safety)) + 200


def main():
    # Pick your “max safe N” here after tuning finishes
    N_SAFE = int(getattr(config, "N_SAFE", 12))

    out_root = ensure_output_root("output")
    session_dir = create_new_results_dir(out_root, prefix="results_")
    print(f"[INFO] Results folder: {session_dir}")

    spline = build_spline_from_centerline()
    dt = float(getattr(config, "DT", 0.05))
    vmax = float(getattr(config, "VMAX", 2.0))
    steps = required_steps(float(spline.length), dt, vmax)

    positions, velocities, groups = spawn_swarm_near_A_B(spline, N=N_SAFE)

    traj, collision_log = simulate_swarm_twoway(
        spline, positions, velocities, groups, steps=steps, dt=dt, log_collisions=True
    )

    total_pairs = sum(len(pairs) for _, pairs in collision_log)
    print(f"[RESULT] N={N_SAFE} total_collision_pairs={total_pairs}")

    # Save a simple text summary
    summary_path = session_dir + "/phase4_demo_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"N_SAFE={N_SAFE}\n")
        f.write(f"DT={dt}\nVMAX={vmax}\n")
        f.write(f"KP={getattr(config,'KP',None)} KD={getattr(config,'KD',None)} KREP={getattr(config,'KREP',None)} RSAFE={getattr(config,'RSAFE',None)}\n")
        f.write(f"total_collision_pairs={total_pairs}\n")
        if collision_log:
            f.write(f"first_collision_step={collision_log[0][0]}\n")
        else:
            f.write("first_collision_step=-1\n")

    print(f"[INFO] Saved summary: {summary_path}")

    # Visualization:
    # We'll render ONE representative robot trajectory as GIF for quick proof.
    # (Swarm GIF is heavier; we can do it next.)
    rep_robot = traj[:, 0, :]  # robot 0 path
    save_path = session_dir + "/phase4_demo_robot0.gif"
    render_path_following(spline, rep_robot, save_path)
    print(f"[INFO] Saved gif: {save_path}")


if __name__ == "__main__":
    main()
