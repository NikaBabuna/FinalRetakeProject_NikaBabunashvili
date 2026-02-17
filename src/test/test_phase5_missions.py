# src/test/test_phase5_missions.py

import cv2
import numpy as np

from src.test._test_utils import TestContext
from src.phase5.video_frames import extract_frames
from src.phase5.missions import make_mission_left_to_right, make_mission_right_to_left
from src import config

TEST_ID = "P5_STEP8_001"
TEST_NAME = "Two missions exist and have correct directions (L->R, R->L)"
TEST_DESCRIPTION = "Creates two missions based on extracted frame size. Validates in-bounds, non-identical, and saves a debug overlay."


def _in_bounds(xy, H, W) -> bool:
    x, y = xy
    return (0 <= x < W) and (0 <= y < H)


def run(results_dir: str) -> bool:
    ctx = TestContext(results_dir)
    ok = True

    ctx.info(f"{TEST_ID} - {TEST_NAME}")
    ctx.info(TEST_DESCRIPTION)

    meta = extract_frames(
        results_dir=results_dir,
        max_seconds=float(getattr(config, "P5_MAX_SECONDS", 1.0) or 1.0),
        stride=int(getattr(config, "P5_FRAME_STRIDE", 10) or 10),
        resize_width=int(getattr(config, "P5_RESIZE_WIDTH", 640) or 640),
        keep_aspect=bool(getattr(config, "P5_KEEP_ASPECT", True)),
        overwrite=True,
    )

    paths = meta["frame_paths"]
    if len(paths) < 1:
        ctx.fail("Need at least 1 extracted frame")
        return False

    frame = cv2.imread(paths[0])
    if frame is None:
        ctx.fail("Failed to read extracted frame")
        return False

    H, W = frame.shape[:2]
    margin = int(getattr(config, "P5_MISSION_MARGIN_PX", 20) or 20)
    y_frac = float(getattr(config, "P5_MISSION_Y_FRAC", 0.50) or 0.50)

    m1 = make_mission_left_to_right(H, W, margin_px=margin, y_frac=y_frac)
    m2 = make_mission_right_to_left(H, W, margin_px=margin, y_frac=y_frac)

    # in-bounds checks
    if _in_bounds(m1.start_xy, H, W) and _in_bounds(m1.goal_xy, H, W):
        ctx.pass_("Mission 1 start/goal in bounds")
    else:
        ctx.fail(f"Mission 1 out of bounds: start={m1.start_xy}, goal={m1.goal_xy}, HW={(H,W)}")
        ok = False

    if _in_bounds(m2.start_xy, H, W) and _in_bounds(m2.goal_xy, H, W):
        ctx.pass_("Mission 2 start/goal in bounds")
    else:
        ctx.fail(f"Mission 2 out of bounds: start={m2.start_xy}, goal={m2.goal_xy}, HW={(H,W)}")
        ok = False

    # direction checks
    if m1.goal_xy[0] > m1.start_xy[0]:
        ctx.pass_("Mission 1 direction OK (L->R)")
    else:
        ctx.fail("Mission 1 direction wrong (expected goal_x > start_x)")
        ok = False

    if m2.goal_xy[0] < m2.start_xy[0]:
        ctx.pass_("Mission 2 direction OK (R->L)")
    else:
        ctx.fail("Mission 2 direction wrong (expected goal_x < start_x)")
        ok = False

    # non-identical
    if (np.linalg.norm(np.array(m1.start_xy) - np.array(m2.start_xy)) > 1e-3) or \
       (np.linalg.norm(np.array(m1.goal_xy) - np.array(m2.goal_xy)) > 1e-3):
        ctx.pass_("Missions are not identical")
    else:
        ctx.fail("Missions appear identical")
        ok = False

    # Save debug overlay (start/goal markers)
    dbg = frame.copy()

    # Mission 1: white circles
    s1 = tuple(map(int, m1.start_xy))
    g1 = tuple(map(int, m1.goal_xy))
    cv2.circle(dbg, s1, 8, (255, 255, 255), 2)
    cv2.circle(dbg, g1, 8, (255, 255, 255), 2)
    cv2.putText(dbg, "M1 start", (s1[0] + 10, s1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(dbg, "M1 goal",  (g1[0] + 10, g1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    # Mission 2: gray circles
    s2 = tuple(map(int, m2.start_xy))
    g2 = tuple(map(int, m2.goal_xy))
    cv2.circle(dbg, s2, 8, (180, 180, 180), 2)
    cv2.circle(dbg, g2, 8, (180, 180, 180), 2)
    cv2.putText(dbg, "M2 start", (s2[0] + 10, s2[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)
    cv2.putText(dbg, "M2 goal",  (g2[0] + 10, g2[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)

    save_path = ctx.path("phase5_missions_debug.png")
    cv2.imwrite(save_path, dbg)
    ctx.info(f"Saved debug overlay: {save_path}")

    return ok
