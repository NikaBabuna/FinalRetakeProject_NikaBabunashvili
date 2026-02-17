# src/phase5/optical_flow.py

import os
import glob
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def _sorted_frame_paths(frames_dir: str, exts=("png", "jpg", "jpeg")) -> List[str]:
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(frames_dir, f"*.{ext}")))
    paths = sorted(paths)
    if len(paths) == 0:
        raise FileNotFoundError(f"No frames found in: {frames_dir}")
    return paths


def saturate_flow(flow: np.ndarray, vmax: float) -> np.ndarray:
    """
    Clip per-pixel flow magnitude to vmax (pixels/frame).

    flow: (H, W, 2)
    """
    if vmax is None or vmax <= 0:
        return flow

    mag = np.linalg.norm(flow, axis=2)  # (H,W)
    # Avoid division by zero
    scale = np.ones_like(mag, dtype=np.float32)
    mask = mag > vmax
    scale[mask] = (vmax / (mag[mask] + 1e-12)).astype(np.float32)

    flow_sat = flow.copy().astype(np.float32)
    flow_sat[..., 0] *= scale
    flow_sat[..., 1] *= scale
    return flow_sat


def flow_to_hsv_vis(flow: np.ndarray, mag_clip: Optional[float] = None) -> np.ndarray:
    """
    Visualize dense flow:
      hue = direction
      value = magnitude (scaled by mag_clip)
    Returns BGR uint8 image.

    NOTE: We avoid cv2.normalize(NORM_MINMAX) because if magnitude has low/zero
    dynamic range, it can produce an all-black image.
    """
    fx = flow[..., 0].astype(np.float32)
    fy = flow[..., 1].astype(np.float32)
    mag, ang = cv2.cartToPolar(fx, fy, angleInDegrees=True)

    # Choose a stable scaling reference
    if mag_clip is None or mag_clip <= 0:
        # robust fallback: ignore extreme outliers
        mag_clip = float(np.percentile(mag, 99))
        mag_clip = max(mag_clip, 1e-6)

    mag01 = np.clip(mag / float(mag_clip), 0.0, 1.0)
    val = (mag01 * 255.0).astype(np.uint8)

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = (ang / 2).astype(np.uint8)  # OpenCV hue range is [0..179]
    hsv[..., 1] = 255
    hsv[..., 2] = val

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr



def compute_dense_flow(prev_gray: np.ndarray, next_gray: np.ndarray, method: str = "farneback") -> np.ndarray:
    """
    Compute dense optical flow between two grayscale frames.
    Returns flow (H, W, 2) in pixels/frame.

    method:
      - "farneback" (default, most portable)
      - "dis" (faster sometimes, if available in your OpenCV build)
    """
    method = (method or "farneback").lower().strip()

    if method == "dis":
        # DIS is not available in some OpenCV builds; fall back if needed
        if not hasattr(cv2, "DISOpticalFlow_create"):
            method = "farneback"
        else:
            dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            flow = dis.calc(prev_gray, next_gray, None)
            return flow.astype(np.float32)

    # Farneback (portable)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, next_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )
    return flow.astype(np.float32)


def compute_and_save_flow_sequence(
    results_dir: str,
    frames_dir: str,
    method: str = "farneback",
    flow_vmax: float = 8.0,
    overwrite: bool = True,
    save_visualizations: bool = True,
    vis_every: int = 10,
) -> Dict:
    """
    Compute dense optical flow for each consecutive frame pair in frames_dir and save to disk.

    Saves:
      - flows: phase5/flow/flow_000000.npy  (H,W,2) float32 (pixels/frame)
      - optional visualizations: phase5/flow_vis/vis_000000.png

    Returns metadata dict with paths + counts.
    """
    os.makedirs(results_dir, exist_ok=True)

    out_flow_dir = os.path.join(results_dir, "phase5", "flow")
    out_vis_dir = os.path.join(results_dir, "phase5", "flow_vis")
    os.makedirs(out_flow_dir, exist_ok=True)
    if save_visualizations:
        os.makedirs(out_vis_dir, exist_ok=True)

    frame_paths = _sorted_frame_paths(frames_dir)
    n_frames = len(frame_paths)
    if n_frames < 2:
        raise ValueError("Need at least 2 frames to compute optical flow.")

    flow_paths: List[str] = []
    vis_paths: List[str] = []

    for i in range(n_frames - 1):
        flow_path = os.path.join(out_flow_dir, f"flow_{i:06d}.npy")
        do_compute = overwrite or (not os.path.exists(flow_path))

        if do_compute:
            prev = cv2.imread(frame_paths[i], cv2.IMREAD_GRAYSCALE)
            nxt = cv2.imread(frame_paths[i + 1], cv2.IMREAD_GRAYSCALE)
            if prev is None or nxt is None:
                raise RuntimeError(f"Failed to read frames {frame_paths[i]} or {frame_paths[i+1]}")

            flow = compute_dense_flow(prev, nxt, method=method)
            flow = saturate_flow(flow, flow_vmax)

            np.save(flow_path, flow)

            if save_visualizations and (i % max(1, int(vis_every)) == 0):
                vis = flow_to_hsv_vis(flow, mag_clip=flow_vmax)
                vis_path = os.path.join(out_vis_dir, f"vis_{i:06d}.png")
                cv2.imwrite(vis_path, vis)
                vis_paths.append(vis_path)

        flow_paths.append(flow_path)

    meta = {
        "frames_dir": frames_dir,
        "n_frames": n_frames,
        "n_flows": len(flow_paths),
        "flow_dir": out_flow_dir,
        "flow_paths": flow_paths,
        "vis_dir": out_vis_dir if save_visualizations else None,
        "vis_paths": vis_paths,
        "method": method,
        "flow_vmax": float(flow_vmax),
    }
    return meta
