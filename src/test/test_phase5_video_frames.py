# src/test/test_phase5_video_frames.py

import os
import cv2
import numpy as np

TEST_ID = "P5_VIDEO_001"
TEST_NAME = "Phase 5 - Video frame extraction + resizing"
TEST_DESCRIPTION = (
    "Validates that frames are extracted into results_dir, resizing works, "
    "and at least a few frames are produced."
)


def _try_import_test_context():
    try:
        from src.test._test_utils import TestContext
        return TestContext
    except Exception:
        return None


def _make_synthetic_video(video_path: str, w: int = 640, h: int = 360, fps: int = 20, n_frames: int = 60) -> None:
    """
    Creates a tiny deterministic synthetic video so this test can run even if the
    real pedestrians.mp4 isn't present.

    Uses MJPG + AVI because it's typically the most portable codec combo for OpenCV.
    """
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("OpenCV VideoWriter could not be opened (codec support issue).")

    rng = np.random.default_rng(0)

    for i in range(n_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Moving dot
        x = int((i / max(1, n_frames - 1)) * (w - 1))
        y = h // 2
        cv2.circle(frame, (x, y), 12, (255, 255, 255), -1)

        # A little noise to make frames distinct
        noise = rng.integers(0, 30, size=(h, w, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)

        writer.write(frame)

    writer.release()


def run(results_dir: str) -> bool:
    ok = True

    # Optional standardized printer
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

    # ------------------------------------------------------------
    # Import the Phase 5 frame extraction function
    # ------------------------------------------------------------
    try:
        from src.phase5.video_frames import extract_frames
        from src.utils.io_utils import get_project_root
        from src import config
    except Exception as e:
        fail(f"Import failed: {e}")
        return False

    # ------------------------------------------------------------
    # Decide which video to use:
    # - Prefer real assignment video in /data
    # - Otherwise synthesize a tiny test video inside results_dir
    # ------------------------------------------------------------
    project_root = get_project_root()
    real_video = os.path.join(project_root, "data", getattr(config, "P5_VIDEO_FILENAME", "pedestrians.mp4"))

    if os.path.exists(real_video):
        video_path = real_video
        info(f"Using real video: {video_path}")
        # Keep test light even if video is huge
        max_seconds = 2.0
        stride = 5
    else:
        video_path = os.path.join(results_dir, "synthetic_p5.avi")
        info("Real video not found in /data. Generating synthetic test video.")
        try:
            _make_synthetic_video(video_path)
        except Exception as e:
            fail(f"Failed to generate synthetic video: {e}")
            return False
        max_seconds = None
        stride = 2

    # ------------------------------------------------------------
    # Run extraction WITH resizing (this checkbox)
    # ------------------------------------------------------------
    resize_width = 320  # manageable resolution for testing
    try:
        meta = extract_frames(
            results_dir=results_dir,
            video_path=video_path,
            resize_width=resize_width,
            keep_aspect=True,
            max_seconds=max_seconds,
            stride=stride,
            ext="png",
            overwrite=True,
        )
    except Exception as e:
        fail(f"extract_frames crashed: {e}")
        return False

    # ------------------------------------------------------------
    # Checks (PASS/FAIL conditions)
    # ------------------------------------------------------------
    try:
        frames_dir = meta["frames_dir"]
        frame_paths = meta["frame_paths"]

        # 1) frames_dir exists and is inside results_dir
        if not os.path.isdir(frames_dir):
            fail(f"frames_dir missing: {frames_dir}")
        else:
            pass_(f"frames_dir exists: {frames_dir}")

        abs_results = os.path.abspath(results_dir)
        abs_frames = os.path.abspath(frames_dir)
        if not abs_frames.startswith(abs_results):
            fail("frames_dir is not inside results_dir (tests must write only inside results_dir).")
        else:
            pass_("frames are saved inside results_dir (good test hygiene)")

        # 2) produce at least 2 frames
        if len(frame_paths) < 2:
            fail(f"Too few frames extracted: {len(frame_paths)}")
        else:
            pass_(f"Extracted {len(frame_paths)} frames")

        # 3) sample existence checks
        for idx in [0, len(frame_paths) // 2, len(frame_paths) - 1]:
            if not os.path.exists(frame_paths[idx]):
                fail(f"Frame file missing: {frame_paths[idx]}")
        if ok:
            pass_("Sampled frame files exist")

        # 4) resizing check: first saved frame width == resize_width
        img0 = cv2.imread(frame_paths[0])
        if img0 is None:
            fail("Could not read first saved frame with cv2.imread")
        else:
            got_w = img0.shape[1]
            if got_w != int(resize_width):
                fail(f"Resize failed: got width={got_w}, expected={resize_width}")
            else:
                pass_(f"Resize OK: frame width={got_w}")

        # 5) metadata sanity
        fps = float(meta.get("fps", 0.0))
        if fps <= 0.0:
            fail(f"Bad fps reported: {fps}")
        else:
            pass_(f"fps looks valid: {fps:.2f}")

    except Exception as e:
        fail(f"Validation logic crashed: {e}")

    return ok
