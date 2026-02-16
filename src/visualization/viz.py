import matplotlib
matplotlib.use("Agg")

import os
import numpy as np
import matplotlib.pyplot as plt
import imageio


def plot_trajectory(trajectory: np.ndarray,
                    target: np.ndarray,
                    save_dir: str,
                    filename: str = "trajectory.png"):
    """
    Saves trajectory plot into save_dir/filename
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    _, n, _ = trajectory.shape

    plt.figure()
    for i in range(n):
        xs = trajectory[:, i, 0]
        ys = trajectory[:, i, 1]
        plt.plot(xs, ys)
        plt.scatter(xs[0], ys[0], marker="o")

    plt.scatter(target[0], target[1], marker="x")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Robot Trajectory")
    plt.axis("equal")
    plt.grid(True)

    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"Saved to: {save_path}")

import os
import numpy as np
import matplotlib.pyplot as plt
import imageio

from src.map_tools.map_loader import load_map_image
from src.config import PATH_WIDTH


def render_path_following(spline, trajectory, save_path):
    """
    FINAL SUBMISSION VISUALIZATION (GIF)

    Shows:
    - map background
    - spline centerline
    - path borders
    - animated robot following path
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    gif_path = save_path.replace(".png", ".gif")

    # -----------------------------
    # load map
    # -----------------------------
    img = load_map_image()

    # -----------------------------
    # precompute spline + borders
    # -----------------------------
    s_vals = np.linspace(0, spline.length, 600)
    center = np.array([spline.p(s) for s in s_vals])

    half_w = PATH_WIDTH / 2.0
    left = []
    right = []

    for s in s_vals:
        p = spline.p(s)
        t = spline.tangent(s)
        n = np.array([-t[1], t[0]])
        left.append(p + half_w * n)
        right.append(p - half_w * n)

    left = np.array(left)
    right = np.array(right)

    # -----------------------------
    # animation frames
    # -----------------------------
    frames = []
    traj = np.array(trajectory)

    for i in range(0, len(traj), 3):  # skip frames for speed
        plt.figure(figsize=(10, 6))

        plt.imshow(img)
        plt.axis("off")

        # spline + borders
        plt.plot(center[:, 0], center[:, 1], linewidth=2, label="center")
        plt.plot(left[:, 0], left[:, 1], "--", linewidth=1)
        plt.plot(right[:, 0], right[:, 1], "--", linewidth=1)

        # trajectory so far
        plt.plot(traj[:i+1, 0], traj[:i+1, 1], color="orange", linewidth=3)

        # robot
        plt.scatter(traj[i, 0], traj[i, 1], s=80)

        plt.title("Robot Path Following")
        plt.legend()

        # save frame to memory
        plt.tight_layout()

        canvas = plt.gcf().canvas
        canvas.draw()

        frame = np.asarray(canvas.buffer_rgba())
        frames.append(frame.copy())

        plt.close()
    # -----------------------------
    # save gif
    # -----------------------------
    imageio.mimsave(gif_path, frames, fps=15)

    print(f"[INFO] GIF saved → {gif_path}")



