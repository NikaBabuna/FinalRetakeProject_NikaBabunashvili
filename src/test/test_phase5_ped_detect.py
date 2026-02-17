# src/test/test_phase5_ped_detect.py

import cv2
import numpy as np

from src.test._test_utils import TestContext
from src.phase5.video_frames import extract_frames
from src.phase5.ped_detect import PedestrianDetectorMOG2, boxes_to_centroids
from src import config

TEST_ID = "P5_STEP6_001"
TEST_NAME = "Pedestrian detection (background subtraction) returns reasonable blobs"
TEST_DESCRIPTION = "Runs MOG2-based detection on a few frames, checks outputs are in-bounds, and saves a debug overlay."


def _cfg(name: str, default):
    return getattr(config, name, default)


def run(results_dir: str) -> bool:
    ctx = TestContext(results_dir)
    ok = True

    ctx.info(f"{TEST_ID} - {TEST_NAME}")
    ctx.info(TEST_DESCRIPTION)

    # keep it fast
    meta = extract_frames(
        results_dir=results_dir,
        max_seconds=float(_cfg("P5_MAX_SECONDS", 2.0) or 2.0),
        stride=int(_cfg("P5_FRAME_STRIDE", 3) or 3),
        resize_width=int(_cfg("P5_RESIZE_WIDTH", 640) or 640),
        keep_aspect=bool(_cfg("P5_KEEP_ASPECT", True)),
        overwrite=True,
    )

    paths = meta["frame_paths"]
    if len(paths) < 6:
        ctx.fail("Need at least a few frames for detection (>= 6 recommended)")
        return False

    # --- Better defaults (tunable via config if you add them) ---
    # Why these changes:
    # - detect_shadows=False keeps mask cleaner (shadows often merge people)
    # - min_area lower catches more pedestrians in overhead / small-person videos
    # - var_threshold slightly higher reduces flicker/noise blobs
    det = PedestrianDetectorMOG2(
        history=int(_cfg("P5_MOG2_HISTORY", 300)),
        var_threshold=float(_cfg("P5_MOG2_VAR_THRESHOLD", 25.0)),
        detect_shadows=bool(_cfg("P5_MOG2_DETECT_SHADOWS", False)),
        min_area=int(_cfg("P5_DET_MIN_AREA", 80)),
        morph_k=int(_cfg("P5_DET_MORPH_K", 5)),
    )

    # --- Warm up background model first (super important) ---
    warmup = int(_cfg("P5_DET_WARMUP_FRAMES", 12))
    warmup = min(warmup, max(0, len(paths) - 3))

    warmed = 0
    for i in range(warmup):
        frame = cv2.imread(paths[i])
        if frame is None:
            continue
        _ = det.detect(frame)  # don't evaluate, just update model
        warmed += 1

    ctx.info(f"Warmed up detector on {warmed} frames")

    # Now evaluate on a handful of frames AFTER warmup
    start = warmup
    remaining = len(paths) - start
    sample_count = min(12, remaining)
    if sample_count <= 0:
        ctx.fail("Not enough frames after warmup to evaluate")
        return False

    # evenly spaced sample indices
    if sample_count == 1:
        sample_idxs = [start]
    else:
        sample_idxs = np.linspace(start, len(paths) - 1, sample_count).astype(int).tolist()

    any_found = False
    last_debug = None

    # optional sanity (prevents “everything is one blob” from passing silently)
    max_boxes_reasonable = int(_cfg("P5_DET_MAX_BOXES_REASONABLE", 400))

    for i in sample_idxs:
        frame = cv2.imread(paths[i])
        if frame is None:
            continue

        H, W = frame.shape[:2]
        boxes = det.detect(frame)
        cents = boxes_to_centroids(boxes)

        if len(boxes) > 0:
            any_found = True

        if len(boxes) > max_boxes_reasonable:
            ctx.fail(f"Too many blobs ({len(boxes)}) on frame {i} — likely noisy mask; increase min_area or var_threshold")
            ok = False

        # bounds checks
        for (x, y, w, h) in boxes:
            if not (0 <= x < W and 0 <= y < H and w > 0 and h > 0):
                ctx.fail(f"Bad box: {(x, y, w, h)} for frame size {(W, H)}")
                ok = False

            # "giant blob" guard (optional but helpful)
            area = float(w * h)
            if area > 0.60 * float(W * H):
                ctx.fail("Detected a giant blob covering most of the frame — tune detector (shadows/threshold/min_area)")
                ok = False

        if cents.size > 0:
            if not np.isfinite(cents).all():
                ctx.fail("Centroids contain NaN/inf")
                ok = False
            if not (
                np.all(cents[:, 0] >= 0) and np.all(cents[:, 0] < W) and
                np.all(cents[:, 1] >= 0) and np.all(cents[:, 1] < H)
            ):
                ctx.fail("Some centroids out of bounds")
                ok = False

        # keep one debug frame (last sampled)
        if i == sample_idxs[-1]:
            dbg = frame.copy()
            for (x, y, w, h) in boxes:
                cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 0), 2)
            for (cx, cy) in cents.astype(int):
                cv2.circle(dbg, (int(cx), int(cy)), 4, (0, 0, 255), -1)

            cv2.putText(
                dbg,
                f"boxes={len(boxes)} min_area={int(_cfg('P5_DET_MIN_AREA', 80))} varThr={float(_cfg('P5_MOG2_VAR_THRESHOLD', 25.0))}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            last_debug = dbg

    if any_found:
        ctx.pass_("Detector returned at least one blob on sampled frames")
    else:
        ctx.fail("No blobs detected — lower min_area, lower stride, or check if the camera is fixed")
        ok = False

    if last_debug is not None:
        save_path = ctx.path("phase5_ped_detect_debug.png")
        cv2.imwrite(save_path, last_debug)
        ctx.info(f"Saved debug overlay: {save_path}")

    return ok
