# src/phase5/video_frames.py

from __future__ import annotations

import os
from typing import List, Tuple, Optional, Dict, Any

import cv2

from src import config
from src.utils.io_utils import get_project_root


def _resolve_video_path(video_path: Optional[str] = None) -> str:
    """
    Resolve the pedestrian video path. If video_path is None, uses:
      <project_root>/data/<config.P5_VIDEO_FILENAME>
    """
    if video_path is not None:
        # allow absolute or relative
        if os.path.isabs(video_path):
            return video_path
        return os.path.join(get_project_root(), video_path)

    return os.path.join(get_project_root(), "data", getattr(config, "P5_VIDEO_FILENAME", "pedestrians.mp4"))


def _resize_frame(frame, resize_width: Optional[int], keep_aspect: bool = True):
    """
    Resize frame to a fixed width while preserving aspect ratio by default.
    """
    if resize_width is None:
        return frame

    h, w = frame.shape[:2]
    if w == 0 or h == 0:
        return frame

    if w == int(resize_width):
        return frame

    new_w = int(resize_width)
    if keep_aspect:
        scale = new_w / float(w)
        new_h = max(1, int(round(h * scale)))
    else:
        new_h = h

    # INTER_AREA is best for downscaling
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def extract_frames(
    results_dir: str,
    video_path: Optional[str] = None,
    resize_width: Optional[int] = None,
    keep_aspect: Optional[bool] = None,
    max_seconds: Optional[float] = None,
    stride: Optional[int] = None,
    ext: Optional[str] = None,
    overwrite: bool = True,
) -> Dict[str, Any]:
    """
    Extract frames from video and save them into:
      <results_dir>/phase5/frames/frame_000000.png

    Returns metadata dict:
      {
        "video_path": ...,
        "frames_dir": ...,
        "frame_paths": [...],
        "fps": float,
        "orig_hw": (H, W),
        "out_hw": (H, W),
        "stride": int,
        "max_seconds": float|None
      }
    """
    os.makedirs(results_dir, exist_ok=True)

    if resize_width is None:
        resize_width = getattr(config, "P5_RESIZE_WIDTH", 640)
    if keep_aspect is None:
        keep_aspect = bool(getattr(config, "P5_KEEP_ASPECT", True))
    if max_seconds is None:
        max_seconds = getattr(config, "P5_MAX_SECONDS", 30.0)
    if stride is None:
        stride = int(getattr(config, "P5_FRAME_STRIDE", 1))
    if ext is None:
        ext = getattr(config, "P5_FRAME_EXT", "png")

    stride = max(1, int(stride))

    vid_path = _resolve_video_path(video_path)
    if not os.path.exists(vid_path):
        raise FileNotFoundError(f"Phase 5 video not found at: {vid_path}")

    out_dir = os.path.join(results_dir, "phase5", "frames")
    if overwrite and os.path.isdir(out_dir):
        # remove old frames
        for fn in os.listdir(out_dir):
            if fn.lower().endswith("." + ext.lower()):
                try:
                    os.remove(os.path.join(out_dir, fn))
                except Exception:
                    pass
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {vid_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    # If fps is missing/0 in some files, assume a reasonable default
    if fps <= 1e-6:
        fps = 30.0

    max_frames = None
    if max_seconds is not None:
        max_frames = int(max_seconds * fps)

    frame_paths: List[str] = []

    orig_hw: Optional[Tuple[int, int]] = None
    out_hw: Optional[Tuple[int, int]] = None

    read_idx = 0
    saved_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if orig_hw is None:
            h0, w0 = frame.shape[:2]
            orig_hw = (h0, w0)

        # stop if time limit reached (based on read frames)
        if max_frames is not None and read_idx >= max_frames:
            break

        # apply stride (save every k-th frame)
        if (read_idx % stride) != 0:
            read_idx += 1
            continue

        frame_out = _resize_frame(frame, resize_width, keep_aspect=keep_aspect)

        if out_hw is None:
            h1, w1 = frame_out.shape[:2]
            out_hw = (h1, w1)

        fname = f"frame_{saved_idx:06d}.{ext}"
        fpath = os.path.join(out_dir, fname)

        # Write as BGR (cv2 standard)
        ok_write = cv2.imwrite(fpath, frame_out)
        if not ok_write:
            raise RuntimeError(f"Failed to write frame: {fpath}")

        frame_paths.append(fpath)

        saved_idx += 1
        read_idx += 1

    cap.release()

    if len(frame_paths) < 2:
        raise RuntimeError(
            f"Extracted too few frames ({len(frame_paths)}). "
            f"Check video content / max_seconds / stride."
        )

    # ensure metadata is not None
    if orig_hw is None:
        raise RuntimeError("No frames read from video.")
    if out_hw is None:
        out_hw = orig_hw

    return {
        "video_path": vid_path,
        "frames_dir": out_dir,
        "frame_paths": frame_paths,
        "fps": fps,
        "orig_hw": orig_hw,
        "out_hw": out_hw,
        "stride": stride,
        "max_seconds": max_seconds,
        "resize_width": resize_width,
        "keep_aspect": keep_aspect,
    }


def quick_check_resize(meta: Dict[str, Any]) -> None:
    """
    Small sanity check:
      - loads first frame and checks width == config.P5_RESIZE_WIDTH (if enabled)
    """
    paths = meta["frame_paths"]
    first = paths[0]
    img = cv2.imread(first)
    if img is None:
        raise RuntimeError(f"Failed to read saved frame: {first}")

    want_w = getattr(config, "P5_RESIZE_WIDTH", None)
    if want_w is not None:
        got_w = img.shape[1]
        if got_w != int(want_w):
            raise AssertionError(f"Resize check failed: got width={got_w}, expected={want_w}")