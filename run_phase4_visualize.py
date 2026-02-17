# run_phase4_visualize.py

import os
import numpy as np

from src import config

from src.path.spline_path import build_spline_from_centerline
from src.swarm.spawn import spawn_swarm_near_A_B
from src.swarm.sim import simulate_swarm_twoway
from src.swarm.viz_swarm import render_swarm_twoway_gif


def get_project_root() -> str:
    # this script lives in project root, next to /src
    return os.path.dirname(os.path.abspath(__file__))


def ensure_output_root_local() -> str:
    """
    Your project sometimes uses /output and sometimes /outputs.
    We’ll prefer /output if it exists, otherwise create it.
    """
    root = get_project_root()

    out1 = os.path.join(root, "output")
    out2 = os.path.join(root, "outputs")

    if os.path.isdir(out1) or not os.path.isdir(out2):
        os.makedirs(out1, exist_ok=True)
        return out1

    os.makedirs(out2, exist_ok=True)
    return out2


def create_new_results_dir_local(output_root: str) -> str:
    """
    Creates:
      output/results_0001
      output/results_0002
      ...
    """
    i = 1
    while True:
        results_dir = os.path.join(output_root, f"results_{i:04d}")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)
            return results_dir
        i += 1


def main():
    output_root = ensure_output_root_local()
    results_dir = create_new_results_dir_local(output_root)

    out_dir = os.path.join(results_dir, "phase4")
    os.makedirs(out_dir, exist_ok=True)

    # build spline exactly like your tests do
    spline = build_spline_from_centerline()

    # Pick N for visualization
    N = int(getattr(config, "N_ROBOTS", 32))

    # Spawn
    positions, velocities, groups = spawn_swarm_near_A_B(spline, N=N)

    # Sim time: ~2 crossing times so they meet
    dt = float(getattr(config, "DT", 0.05))
    vmax = float(getattr(config, "VMAX", 2.0))
    steps = int((float(spline.length) / vmax) / dt) * 2

    traj = simulate_swarm_twoway(
        spline=spline,
        positions=positions,
        velocities=velocities,
        groups=groups,
        steps=steps,
        dt=dt
    )

    gif_path = os.path.join(out_dir, f"swarm_twoway_N{N}.gif")
    render_swarm_twoway_gif(
        spline=spline,
        traj=traj,
        groups=groups,
        save_path=gif_path,
        stride=2,
        fps=20,
        trail=35,
        show_corridor=True,
        title=f"Two-way swarm (N={N})"
    )

    print(f"[OK] Saved GIF: {gif_path}")


if __name__ == "__main__":
    main()
