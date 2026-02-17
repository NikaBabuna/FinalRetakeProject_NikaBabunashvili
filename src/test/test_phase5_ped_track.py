# src/test/test_phase5_ped_track.py

import cv2
import numpy as np

from src.test._test_utils import TestContext
from src.phase5.video_frames import extract_frames
from src.phase5.ped_detect import PedestrianDetectorMOG2, boxes_to_centroids
from src.phase5.ped_track import NearestNeighborTracker
from src import config

TEST_ID = "P5_STEP7_001"
TEST_NAME = "Pedestrian tracking (rough) keeps IDs stable across frames"
TEST_DESCRIPTION = "Runs MOG2 detection + nearest-neighbor tracking and checks at least one track persists; saves debug overlay."


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
    if len(paths) < 6:
        ctx.fail("Need at least ~6 frames to test tracking")
        return False

    # Detection (use your tuned params)
    det = PedestrianDetectorMOG2(
        history=300,
        var_threshold=25.0,
        detect_shadows=False,
        min_area=80,
        morph_k=5,
        open_iters=1,
        close_iters=1,
        split_large_blobs=True,
    )

    # Tracking knobs (can override via config if you want)
    tracker = NearestNeighborTracker(
        max_dist=float(getattr(config, "P5_TRACK_MAX_DIST", 45.0) or 45.0),
        max_missed=int(getattr(config, "P5_TRACK_MAX_MISSED", 4) or 4),
        min_hits=int(getattr(config, "P5_TRACK_MIN_HITS", 2) or 2),
        ema_alpha=float(getattr(config, "P5_TRACK_EMA_ALPHA", 0.6) or 0.6),
    )

    # Track persistence stats
    seen_frames_per_id = {}  # id -> count
    last_dbg = None

    # sample first ~20 frames max
    sample_paths = paths[: min(20, len(paths))]

    for i, p in enumerate(sample_paths):
        frame = cv2.imread(p)
        if frame is None:
            continue
        H, W = frame.shape[:2]

        boxes = det.detect(frame)
        cents = boxes_to_centroids(boxes)

        tracks = tracker.update(boxes, cents)

        # record appearances
        for t in tracks:
            seen_frames_per_id[t.track_id] = seen_frames_per_id.get(t.track_id, 0) + 1

            # basic sanity: in-bounds + finite
            cx, cy = float(t.centroid[0]), float(t.centroid[1])
            if not np.isfinite([cx, cy]).all():
                ctx.fail("Track centroid has NaN/inf")
                ok = False
            if not (0 <= cx < W and 0 <= cy < H):
                ctx.fail(f"Track centroid out of bounds: {(cx, cy)} for {(W, H)}")
                ok = False

        # save debug overlay on the last sampled frame
        if i == len(sample_paths) - 1:
            dbg = frame.copy()
            # detections
            for (x, y, w, h) in boxes:
                cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # tracks
            for t in tracks:
                cx, cy = int(t.centroid[0]), int(t.centroid[1])
                cv2.circle(dbg, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(
                    dbg,
                    f"id={t.track_id}",
                    (cx + 5, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            last_dbg = dbg

    if len(seen_frames_per_id) == 0:
        ctx.fail("No tracks were created — detector might be returning nothing")
        return False

    max_persist = max(seen_frames_per_id.values())
    ctx.info(f"Tracker created {tracker.total_created} tracks; best persistence = {max_persist} frames")

    # Checkbox requirement: "even rough" but at least one ID persists
    if max_persist >= 5:
        ctx.pass_("At least one track persists >= 5 sampled frames (IDs not resetting every frame)")
    else:
        ctx.fail("Tracks are too unstable (no ID persists >= 5 frames) — increase max_dist or max_missed")
        ok = False

    if last_dbg is not None:
        save_path = ctx.path("phase5_ped_track_debug.png")
        cv2.imwrite(save_path, last_dbg)
        ctx.info(f"Saved debug overlay: {save_path}")

    return ok
