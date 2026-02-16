import os
import numpy as np
import matplotlib.pyplot as plt

from src.path.spline_path import build_spline_from_centerline
from src.path.path_controller import PathController
from src.path.constraints import is_inside_path, distance_to_path
from src.core.core import step_rk2, initialize_robots
from src import config

TEST_NAME = "P3_STEP8_001"
TEST_DESC = "Formal path-following validation"


def _distance_to_path(p, spline):
    try:
        return float(distance_to_path(p, spline))
    except TypeError:
        return float(distance_to_path(p))


def _is_inside_path(p, spline):
    try:
        return bool(is_inside_path(p, spline))
    except TypeError:
        return bool(is_inside_path(p))


def _required_steps(spline_length: float, dt: float, vmax: float, safety: float = 1.25) -> int:
    """
    With VMAX clipping, max distance per step ≈ dt * vmax.
    Steps needed ≈ length / (dt*vmax). Add safety margin.
    """
    max_per_step = max(1e-9, dt * vmax)
    base = int(np.ceil(spline_length / max_per_step))
    return int(np.ceil(base * safety)) + 200  # + buffer


def simulate_path_following(spline, dt: float, steps: int):
    controller = PathController(spline)

    pos_batch, vel_batch = initialize_robots(1)

    pos_batch[0] = spline.p(0.0)
    vel_batch[0] = np.zeros(2)

    traj = [pos_batch[0].copy()]

    for _ in range(steps):
        pos = pos_batch[0]
        target = controller.update(pos)  # returns target point (2,)

        pos_batch, vel_batch = step_rk2(pos_batch, vel_batch, target, dt)
        traj.append(pos_batch[0].copy())

    return np.array(traj)


def run(results_dir):
    print(f"[INFO] {TEST_NAME} - {TEST_DESC}")

    try:
        spline = build_spline_from_centerline()
        print(f"[PASS] Spline built (length={spline.length:.2f})")

        dt = float(getattr(config, "DT", 0.05))
        vmax = float(getattr(config, "VMAX", 2.0))
        steps = _required_steps(spline.length, dt, vmax)

        print(f"[INFO] Using dt={dt:.3f}, VMAX={vmax:.3f} -> steps={steps}")

        traj = simulate_path_following(spline, dt=dt, steps=steps)

        if traj.ndim != 2 or traj.shape[1] != 2:
            raise RuntimeError(f"Trajectory shape invalid: {traj.shape}")
        if not np.all(np.isfinite(traj)):
            raise RuntimeError("Trajectory has NaNs/Infs")
        print("[PASS] Trajectory valid")

        final_pos = traj[-1]
        final_s = spline.closest_s(final_pos)
        progress_ratio = final_s / spline.length

        # must reach at least 95% of arc-length
        reached_goal_region = progress_ratio >= 0.95

        # border check
        path_width = float(getattr(config, "PATH_WIDTH_PIX", config.PATH_WIDTH))
        half_allowed = 0.5 * path_width

        max_violation = 0.0
        outside_count = 0

        for p in traj:
            d = _distance_to_path(p, spline)
            inside = _is_inside_path(p, spline)
            if not inside:
                outside_count += 1
            max_violation = max(max_violation, max(0.0, d - half_allowed))

        print(f"[INFO] Path progress: {progress_ratio*100:.1f}%")
        print(f"[INFO] Max border violation: {max_violation:.2f}")
        print(f"[INFO] Outside count: {outside_count}")

        passed = True

        if not reached_goal_region:
            print("[FAIL] Robot did not reach end of path")
            passed = False
        else:
            print("[PASS] Robot reached end of path")

        if max_violation > 5.0:
            print("[FAIL] Robot left path borders")
            passed = False
        else:
            print("[PASS] Stayed inside borders")

        # Save plot
        save_dir = os.path.join(results_dir, "tests")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "path_following_validation.png")

        s_vals = np.linspace(0, spline.length, 600)
        center = np.array([spline.p(s) for s in s_vals])

        plt.figure(figsize=(8, 5))
        plt.plot(center[:, 0], center[:, 1], label="centerline")
        plt.plot(traj[:, 0], traj[:, 1], label="robot")
        plt.scatter([final_pos[0]], [final_pos[1]], label="final")
        plt.legend()
        plt.title("Path Following Validation")
        plt.axis("equal")
        plt.grid(True)
        plt.savefig(save_path, dpi=200)
        plt.close()

        print(f"[INFO] Saved: {save_path}")
        return passed

    except Exception as e:
        print(f"[FAIL] Exception: {e}")
        return False
