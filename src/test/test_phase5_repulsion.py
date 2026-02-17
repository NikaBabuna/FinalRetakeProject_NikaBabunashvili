# src/test/test_phase5_repulsion.py

import cv2
import numpy as np

from src.test._test_utils import TestContext
from src.phase5.video_frames import extract_frames
from src.phase5.ped_detect import PedestrianDetectorMOG2, boxes_to_centroids
from src.phase5.repulsion import repulsion_force
from src import config

TEST_ID = "P5_STEP7_001"
TEST_NAME = "Repulsion from pedestrian positions xj(t) behaves sensibly"
TEST_DESCRIPTION = "Uses detected pedestrian centroids as repulsion points, checks force increases when closer, and saves a debug overlay."


def run(results_dir: str) -> bool:
    ctx = TestContext(results_dir)
    ok = True

    ctx.info(f"{TEST_ID} - {TEST_NAME}")
    ctx.info(TEST_DESCRIPTION)

    meta = extract_frames(
        results_dir=results_dir,
        max_seconds=float(getattr(config, "P5_MAX_SECONDS", 2.0) or 2.0),
        stride=int(getattr(config, "P5_FRAME_STRIDE", 3) or 3),
        resize_width=int(getattr(config, "P5_RESIZE_WIDTH", 640) or 640),
        keep_aspect=bool(getattr(config, "P5_KEEP_ASPECT", True)),
        overwrite=True,
    )

    paths = meta["frame_paths"]
    if len(paths) < 3:
        ctx.fail("Need at least a few frames")
        return False

    # Use your tuned-ish detector defaults
    det = PedestrianDetectorMOG2(
        history=300,
        var_threshold=25.0,
        detect_shadows=False,
        min_area=80,
        morph_k=5,
    )

    # Find a frame that actually has detections
    frame = None
    cents = None
    for i in range(min(20, len(paths))):
        frame = cv2.imread(paths[i])
        if frame is None:
            continue
        boxes = det.detect(frame)
        cents = boxes_to_centroids(boxes)
        if cents.shape[0] > 0:
            ctx.info(f"Using frame i={i} with {cents.shape[0]} repulsion points")
            break

    if frame is None or cents is None or cents.shape[0] == 0:
        ctx.fail("No pedestrian centroids found to test repulsion")
        return False

    H, W = frame.shape[:2]
    rsafe = float(getattr(config, "P5_RSAFE_PIX", 45.0) or 45.0)
    krep = float(getattr(config, "P5_KREP_PED", 1.0) or 1.0)

    p0 = cents[0]  # pick one repulsion point

    # Robot near (inside rsafe)
    near = np.array([p0[0] + 0.25 * rsafe, p0[1]], dtype=np.float32)
    near[0] = np.clip(near[0], 0, W - 1)
    near[1] = np.clip(near[1], 0, H - 1)

    # Robot far (outside rsafe)
    far = np.array([p0[0] + 2.0 * rsafe, p0[1]], dtype=np.float32)
    far[0] = np.clip(far[0], 0, W - 1)
    far[1] = np.clip(far[1], 0, H - 1)

    F_near = repulsion_force(near, cents, rsafe=rsafe, k_rep=krep)
    F_far = repulsion_force(far, cents, rsafe=rsafe, k_rep=krep)

    n_near = float(np.linalg.norm(F_near))
    n_far = float(np.linalg.norm(F_far))

    ctx.info(f"Repulsion magnitudes: near={n_near:.4f}, far={n_far:.4f}, rsafe={rsafe:.1f}")

    if np.isfinite(F_near).all() and np.isfinite(F_far).all():
        ctx.pass_("Repulsion force is finite")
    else:
        ctx.fail("Repulsion force contains NaN/inf")
        ok = False

    if n_near > n_far + 1e-3:
        ctx.pass_("Repulsion is stronger when closer (near > far)")
    else:
        ctx.fail("Repulsion did not increase when closer")
        ok = False

    # Far should be near-zero (not exactly zero if there are other centroids close)
    # We'll just require it's "small compared to near"
    if n_near > 1e-6 and (n_far / n_near) < 0.25:
        ctx.pass_("Far repulsion is relatively small")
    else:
        ctx.fail("Far repulsion not small enough (tune rsafe/krep or detection noise)")
        ok = False

    # Save debug overlay
    dbg = frame.copy()

    # draw centroids
    for (cx, cy) in cents.astype(int):
        cv2.circle(dbg, (int(cx), int(cy)), 3, (0, 0, 255), -1)

    # draw robot near + arrow
    cv2.circle(dbg, (int(near[0]), int(near[1])), 6, (255, 255, 255), 2)
    end_near = near + 8.0 * F_near  # scale for visibility
    cv2.arrowedLine(
        dbg,
        (int(near[0]), int(near[1])),
        (int(end_near[0]), int(end_near[1])),
        (255, 255, 255),
        2,
        tipLength=0.3,
    )

    # draw robot far + arrow
    cv2.circle(dbg, (int(far[0]), int(far[1])), 6, (200, 200, 200), 2)
    end_far = far + 8.0 * F_far
    cv2.arrowedLine(
        dbg,
        (int(far[0]), int(far[1])),
        (int(end_far[0]), int(end_far[1])),
        (200, 200, 200),
        2,
        tipLength=0.3,
    )

    save_path = ctx.path("phase5_repulsion_debug.png")
    cv2.imwrite(save_path, dbg)
    ctx.info(f"Saved debug overlay: {save_path}")

    return ok
