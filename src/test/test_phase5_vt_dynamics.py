# src/test/test_phase5_vt_dynamics.py
import numpy as np

from src.test._test_utils import TestContext
from src.phase5.vt_dynamics import VTParams, vt_derivative, rk2_step

TEST_ID = "P5_STEP9_001"
TEST_NAME = "IVP VT dynamics integrates stably and tracks a velocity field"
TEST_DESCRIPTION = "Synthetic constant flow field: robot velocity converges toward flow; repulsion pushes away when close."


def run(results_dir: str) -> bool:
    ctx = TestContext(results_dir)
    ok = True

    ctx.info(f"{TEST_ID} - {TEST_NAME}")
    ctx.info(TEST_DESCRIPTION)

    # Synthetic field: constant rightward velocity (px/s)
    H, W = 120, 160
    flow = np.zeros((H, W, 2), dtype=np.float32)
    flow[..., 0] = 3.0  # u = +3 px/s
    flow[..., 1] = 0.0

    params = VTParams(kv=3.0, kd=0.2, rsafe=20.0, krep=400.0, rep_gamma=2.0, max_rep_accel=3000.0)

    # Start at center, zero velocity
    state = np.array([W / 2, H / 2, 0.0, 0.0], dtype=np.float32)
    dt = 0.05

    # No pedestrians: should converge vx -> ~3
    for _ in range(120):
        deriv_fn = lambda s: vt_derivative(s, flow, ped_points_t=None, params=params, goal_xy=None)
        state = rk2_step(state, dt, deriv_fn)

        if not np.isfinite(state).all():
            ctx.fail("State blew up (NaN/inf) during integration")
            return False

    vx = float(state[2])
    if abs(vx - 3.0) < 0.35:
        ctx.pass_(f"Velocity tracks flow: vx≈{vx:.3f} close to 3.0")
    else:
        ctx.fail(f"Velocity did not track flow well: vx={vx:.3f} (expected near 3.0)")
        ok = False

    # With a pedestrian slightly to the right: repulsion should push left (negative ax component)
    state2 = np.array([80.0, 60.0, 0.0, 0.0], dtype=np.float32)
    ped = [(82.0, 60.0)]  # close on the right
    d = vt_derivative(state2, flow, ped_points_t=ped, params=params, goal_xy=None)
    ax = float(d[2])

    if ax < 0.0:
        ctx.pass_(f"Repulsion pushes away from pedestrian (ax={ax:.3f} < 0)")
    else:
        ctx.fail(f"Repulsion not pushing away as expected (ax={ax:.3f} >= 0)")
        ok = False

    return ok
