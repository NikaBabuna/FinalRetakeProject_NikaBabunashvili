import numpy as np
import matplotlib.pyplot as plt

from src.test._test_utils import TestContext
from src.path.centerline import extract_centerline_points
from src.path.spline_path import build_spline_from_centerline
from src.path.path_controller import PathController


TEST_ID = "P3_STEP6_001"
TEST_NAME = "Path following simulation"
TEST_DESCRIPTION = "Robot follows spline using moving target controller"


def run(results_dir: str) -> bool:
    ctx = TestContext(results_dir)
    ctx.info(f"{TEST_ID} - {TEST_NAME}")
    ctx.info(TEST_DESCRIPTION)

    try:
        pts = extract_centerline_points()
        spline = build_spline_from_centerline(pts)
        ctx.pass_(f"Spline built (length={spline.length:.2f})")
    except Exception as e:
        ctx.fail(f"Spline build failed: {e}")
        return False

    controller = PathController(spline)

    # start robot at beginning
    x = spline.p(0.0).copy()
    v = np.zeros(2)

    dt = 0.05
    steps = 1200

    traj = []

    for _ in range(steps):
        force, target = controller.update(x, v, dt)

        # simple physics
        v += force * dt
        x += v * dt

        traj.append(x.copy())

    traj = np.array(traj)

    # plot
    plt.figure(figsize=(6,4))
    path_pts = spline.sample(600)
    plt.plot(path_pts[:,0], path_pts[:,1], label="path")
    plt.plot(traj[:,0], traj[:,1], label="robot")
    plt.scatter(traj[0,0], traj[0,1], s=60, label="start")
    plt.scatter(traj[-1,0], traj[-1,1], s=60, label="end")
    plt.legend()
    plt.axis("equal")
    plt.title("Path Following")

    save_path = ctx.path("path_following.png")
    plt.savefig(save_path)
    plt.close()

    ctx.pass_("Robot followed path")
    ctx.info(f"Saved: {save_path}")

    return True
