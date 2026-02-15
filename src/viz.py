import os
import numpy as np
import matplotlib.pyplot as plt


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
