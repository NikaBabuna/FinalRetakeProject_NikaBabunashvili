# src/test/test_phase5_optical_flow.py

import os
import cv2
import numpy as np

TEST_ID = "P5_FLOW_001"
TEST_NAME = "Phase 5 - Dense optical flow + saturation + storage"
TEST_DESCRIPTION = (
    "Computes dense flow from frames, saves per-step .npy flow fields, "
    "checks shape/finite values, and verifies saturation."
)


def _try_import_test_context():
    try:
        from src.test._test_utils import TestContext
        return TestContext
    except Exception:
        return None


def _make_synthetic_video(video_path: str, w: int = 320, h: int = 240, fps: int = 20, n_frames: int = 40) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("OpenCV VideoWriter could not be opened (codec support issue).")

    for i in range(n_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Moving blob (this ensures optical flow is not all-zero)
        x = int((i / max(1, n_frames - 1)) * (w - 60) + 30)
        y = h // 2
        cv2.circle(frame, (x, y), 18, (255, 255, 255), -1)

        writer.write(frame)

    writer.release()


def run(results_dir: str) -> bool:
    ok = True

    TestContext = _try_import_test_context()
    ctx = TestContext(results_dir=results_dir) if TestContext else None

    def info(msg): print(f"  [INFO] {msg}") if ctx is None else ctx.info(msg)
    def pass_(msg): print(f"  [PASS] {msg}") if ctx is None else ctx.pass_(msg)
    def fail(msg):
        nonlocal ok
        ok = False
        print(f"  [FAIL] {msg}") if ctx is None else ctx.fail(msg)

    info(f"{TEST_ID} - {TEST_NAME}")
    info(TEST_DESCRIPTION)

    # Imports
    try:
        from src.phase5.video_frames import extract_frames
        from src.phase5.optical_flow import compute_and_save_flow_sequence
        from src.utils.io_utils import get_project_root
        from src import config
    except Exception as e:
        fail(f"Import failed: {e}")
        return False

    # Choose video:
    project_root = get_project_root()
    real_video = os.path.join(project_root, "data", getattr(config, "P5_VIDEO_FILENAME", "pedestrians.mp4"))

    if os.path.exists(real_video):
        video_path = real_video
        info(f"Using real video: {video_path}")
        max_seconds = 2.0
        stride = 5
    else:
        video_path = os.path.join(results_dir, "synthetic_flow.avi")
        info("Real video not found in /data. Generating synthetic test video.")
        try:
            _make_synthetic_video(video_path)
        except Exception as e:
            fail(f"Failed to generate synthetic video: {e}")
            return False
        max_seconds = None
        stride = 1

    # Extract frames into results_dir
    try:
        fmeta = extract_frames(
            results_dir=results_dir,
            video_path=video_path,
            resize_width=320,
            keep_aspect=True,
            max_seconds=max_seconds,
            stride=stride,
            ext="png",
            overwrite=True,
        )
        frames_dir = fmeta["frames_dir"]
        frame_paths = fmeta["frame_paths"]
    except Exception as e:
        fail(f"Frame extraction failed: {e}")
        return False

    if len(frame_paths) < 3:
        fail(f"Need >= 3 frames for this flow test, got {len(frame_paths)}")
        return False
    pass_(f"Frames ready: {len(frame_paths)} frames")

    # Compute flow sequence
    flow_vmax = 6.0  # pixels/frame (test saturation threshold)
    try:
        meta = compute_and_save_flow_sequence(
            results_dir=results_dir,
            frames_dir=frames_dir,
            method=getattr(config, "P5_FLOW_METHOD", "farneback"),
            flow_vmax=flow_vmax,
            overwrite=True,
            save_visualizations=True,
            vis_every=5,
        )
    except Exception as e:
        fail(f"Optical flow computation crashed: {e}")
        return False

    # Checks
    try:
        flow_paths = meta["flow_paths"]
        if len(flow_paths) != len(frame_paths) - 1:
            fail(f"Expected {len(frame_paths)-1} flows, got {len(flow_paths)}")
        else:
            pass_(f"Saved correct number of flow fields: {len(flow_paths)}")

        # Load one flow and validate shape & finiteness
        flow0 = np.load(flow_paths[0])
        if flow0.ndim != 3 or flow0.shape[2] != 2:
            fail(f"Flow has wrong shape: {flow0.shape} (expected HxWx2)")
        else:
            pass_(f"Flow shape OK: {flow0.shape}")

        if not np.isfinite(flow0).all():
            fail("Flow contains non-finite values (nan/inf)")
        else:
            pass_("Flow values are finite")

        # Ensure flow isn't all zero (synthetic should have motion; real video should too)
        mag0 = np.linalg.norm(flow0, axis=2)
        if float(mag0.mean()) < 0.02:
            fail(f"Flow magnitude too small (mean={mag0.mean():.4f}) - looks like no motion")
        else:
            pass_(f"Flow has motion (mean magnitude={mag0.mean():.4f})")

        # Saturation check: max magnitude <= flow_vmax + epsilon
        max_mag = float(mag0.max())
        if max_mag > flow_vmax + 1e-3:
            fail(f"Saturation failed: max_mag={max_mag:.4f} > vmax={flow_vmax}")
        else:
            pass_(f"Saturation OK: max_mag={max_mag:.4f} <= vmax={flow_vmax}")

        # Verify visualization artifact exists
        vis_dir = meta["vis_dir"]
        vis_paths = meta["vis_paths"]
        if vis_dir is None or len(vis_paths) == 0:
            fail("No flow visualization images were saved")
        else:
            missing = [p for p in vis_paths[:2] if not os.path.exists(p)]
            if missing:
                fail(f"Missing visualization files: {missing}")
            else:
                pass_(f"Flow visualizations saved: {len(vis_paths)} images")

    except Exception as e:
        fail(f"Validation logic crashed: {e}")

    return ok

