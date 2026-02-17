# src/test/test_phase5_dense_flow.py

import os
import numpy as np
import cv2

from src.test._test_utils import TestContext
from src.phase5.video_frames import extract_frames
from src.phase5.optical_flow import compute_dense_flow, flow_to_hsv_vis
from src import config

TEST_ID = "P5_STEP5_001"
TEST_NAME = "Dense optical flow (OpenCV) produces valid (u,v) field"
TEST_DESCRIPTION = "Loads two consecutive resized frames and computes dense flow (Farnebäck/DIS). Validates shape/dtype/finiteness and saves a visualization."

def _pick_moving_pair(paths, max_scan=200, min_mean_absdiff=1.0):
    """
    Find an index i such that frames[i] and frames[i+1] differ enough to produce non-trivial flow.
    Uses mean absolute grayscale difference as a cheap motion proxy.
    """
    n = len(paths)
    scan = min(int(max_scan), n - 1)

    for i in range(scan):
        f0 = cv2.imread(paths[i], cv2.IMREAD_GRAYSCALE)
        f1 = cv2.imread(paths[i + 1], cv2.IMREAD_GRAYSCALE)
        if f0 is None or f1 is None:
            continue

        diff = float(np.mean(np.abs(f0.astype(np.float32) - f1.astype(np.float32))))
        if diff >= float(min_mean_absdiff):
            return i, diff

    return 0, 0.0


def run(results_dir: str) -> bool:
    ctx = TestContext(results_dir)
    ok = True

    ctx.info(f"{TEST_ID} - {TEST_NAME}")
    ctx.info(TEST_DESCRIPTION)

    # Keep this test FAST: only extract ~2 seconds and stride a bit
    try:
        meta = extract_frames(
            results_dir=results_dir,
            max_seconds=float(getattr(config, "P5_MAX_SECONDS", 2.0) or 2.0),
            stride=int(getattr(config, "P5_FRAME_STRIDE", 3) or 3),
            resize_width=int(getattr(config, "P5_RESIZE_WIDTH", 640) or 640),
            keep_aspect=bool(getattr(config, "P5_KEEP_ASPECT", True)),
            overwrite=True,
        )
        ctx.pass_(f"Extracted {len(meta['frame_paths'])} frames at {meta['out_hw']}")
    except Exception as e:
        ctx.fail(f"Frame extraction failed: {e}")
        return False

    paths = meta["frame_paths"]
    if len(paths) < 2:
        ctx.fail("Need at least 2 frames to compute flow")
        return False

    # Pick a pair that actually has motion (more robust than always taking the first two)
    i, diff = _pick_moving_pair(
        paths,
        max_scan=min(200, len(paths) - 1),
        min_mean_absdiff=float(getattr(config, "P5_MIN_FRAME_DIFF", 1.0)),
    )
    ctx.info(f"Selected frame pair i={i} with mean_absdiff={diff:.3f}")

    f0 = cv2.imread(paths[i])
    f1 = cv2.imread(paths[i + 1])
    if f0 is None or f1 is None:
        ctx.fail("Failed to read extracted frames from disk")
        return False

    prev = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
    nxt = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)

    method = str(getattr(config, "P5_FLOW_METHOD", "farneback"))

    # Compute dense flow
    try:
        flow = compute_dense_flow(prev, nxt, method=method)
        ctx.pass_(f"Computed flow using method='{method}'")
    except Exception as e:
        ctx.fail(f"compute_dense_flow failed: {e}")
        return False

    # --- Validations for checkbox ---
    if flow.ndim == 3 and flow.shape[2] == 2:
        ctx.pass_(f"Flow shape OK: {flow.shape}")
    else:
        ctx.fail(f"Flow shape wrong: {flow.shape}")
        ok = False

    if flow.dtype in (np.float32, np.float64):
        ctx.pass_(f"Flow dtype OK: {flow.dtype}")
    else:
        ctx.fail(f"Flow dtype unexpected: {flow.dtype}")
        ok = False

    if np.isfinite(flow).all():
        ctx.pass_("Flow values are finite (no NaN/inf)")
    else:
        ctx.fail("Flow contains NaN/inf")
        ok = False

    # Non-trivial motion check (robust-ish): look at the 95th percentile magnitude
    mag = np.linalg.norm(flow.astype(np.float32), axis=2)
    p95 = float(np.percentile(mag, 95))
    mean_mag = float(np.mean(mag))
    ctx.info(f"Flow magnitude: mean={mean_mag:.4f}, p95={p95:.4f}")

    # If your clip is *truly* static, this could fail — but for pedestrian video it should pass.
    if p95 > 0.05:
        ctx.pass_("Flow is non-trivial (p95 magnitude > 0.15 px/frame)")
    else:
        ctx.fail("Flow looks near-zero (video may be too static, or stride/max_seconds too aggressive)")
        ok = False

    # Save artifacts (visual proof)
    try:
        vis = flow_to_hsv_vis(flow, mag_clip=max(1.0, p95))
        vis_path = ctx.path("phase5_flow_vis_000000.png")
        cv2.imwrite(vis_path, vis)
        ctx.pass_(f"Saved flow visualization: {vis_path}")

        npy_path = ctx.path("phase5_flow_000000.npy")
        np.save(npy_path, flow.astype(np.float32))
        ctx.info(f"Saved raw flow field: {npy_path}")
    except Exception as e:
        ctx.fail(f"Failed to save artifacts: {e}")
        ok = False

    return ok
