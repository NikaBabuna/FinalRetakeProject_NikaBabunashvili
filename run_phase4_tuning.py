import numpy as np

from src import config
from src.path.spline_path import build_spline_from_centerline
from src.swarm.spawn import spawn_swarm_near_A_B
from src.swarm.sim import simulate_swarm_twoway
from src.utils.io_utils import ensure_output_root, create_new_results_dir


def required_steps(length: float, dt: float, vmax: float, safety: float = 1.35) -> int:
    # Rough steps to traverse path with a safety margin
    max_per_step = max(1e-9, dt * vmax)
    base = int(np.ceil(length / max_per_step))
    return int(np.ceil(base * safety)) + 200


def main():
    # results folder
    out_root = ensure_output_root("output")
    session_dir = create_new_results_dir(out_root, prefix="results_")
    save_dir = session_dir  # keep simple
    print(f"[INFO] Results folder: {session_dir}")

    # build spline
    spline = build_spline_from_centerline()
    L = float(spline.length)
    dt = float(getattr(config, "DT", 0.05))
    vmax = float(getattr(config, "VMAX", 2.0))
    steps = required_steps(L, dt, vmax)

    print(f"[INFO] Spline length={L:.2f}, dt={dt}, vmax={vmax} -> steps={steps}")
    print(f"[INFO] Using KREP={config.KREP}, KD={config.KD}, RSAFE={config.RSAFE}, ROBOT_RADIUS={config.ROBOT_RADIUS}")

    # sweep N
    N_list = [4, 6, 8, 10, 12, 16, 20, 24, 28, 32]
    best_safe = 0

    report_lines = []
    report_lines.append("N,collisions,first_collision_step")

    for N in N_list:
        # spawn
        pos, vel, groups = spawn_swarm_near_A_B(spline, N=N)

        # simulate
        traj, collision_log = simulate_swarm_twoway(
            spline, pos, vel, groups, steps=steps, dt=dt, log_collisions=True
        )

        n_coll_steps = len(collision_log)
        first_step = collision_log[0][0] if collision_log else -1

        # total pair count across all timesteps (rough severity)
        total_pairs = sum(len(pairs) for _, pairs in collision_log)

        print(f"[N={N:2d}] collision_steps={n_coll_steps:4d}  total_pairs={total_pairs:6d}  first={first_step}")

        report_lines.append(f"{N},{total_pairs},{first_step}")

        if total_pairs == 0:
            best_safe = N
        else:
            # once it fails, you can stop early if you want
            pass

    # save CSV report
    csv_path = save_dir + "/phase4_capacity.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("\n===================================")
    print(f"[RESULT] Max safe N (zero collisions): {best_safe}")
    print(f"[INFO] Saved: {csv_path}")
    print("===================================\n")


if __name__ == "__main__":
    main()
