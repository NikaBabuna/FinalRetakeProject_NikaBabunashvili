# src/swarm/viz_swarm.py

import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import imageio.v2 as imageio

from src import config
from src.swarm.collisions import detect_collisions

# Map loader (your project has this)
try:
    from src.map_tools.map_loader import load_map_image
except Exception:
    load_map_image = None


def _sample_spline(spline, n=450):
    L = float(spline.length)
    ss = np.linspace(0.0, L, n)
    pts = np.array([spline.p(float(s)) for s in ss], dtype=float)  # (n,2)
    tans = np.array([spline.tangent(float(s)) for s in ss], dtype=float)
    norms = np.linalg.norm(tans, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-9, None)
    tans = tans / norms
    return ss, pts, tans


def _corridor_bounds(pts, tans, half_width):
    n = np.stack([-tans[:, 1], tans[:, 0]], axis=1)  # (n,2)
    left = pts + half_width * n
    right = pts - half_width * n
    return left, right


def _unwrap_traj(traj):
    if isinstance(traj, (tuple, list)) and len(traj) > 0:
        return traj[0]
    return traj


def _load_map_rgb(downsample: float = 1.0):
    """
    Returns (rgb_image, H, W) or (None, 0, 0) if not available.
    """
    if load_map_image is None:
        return None, 0, 0

    img_bgr = load_map_image()
    if img_bgr is None:
        return None, 0, 0

    # Optional downsample (speeds up rendering if map is large)
    if downsample is not None and float(downsample) != 1.0:
        import cv2
        ds = float(downsample)
        new_w = max(1, int(img_bgr.shape[1] * ds))
        new_h = max(1, int(img_bgr.shape[0] * ds))
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    img_rgb = img_bgr[..., ::-1].copy()  # BGR -> RGB
    h, w = img_rgb.shape[:2]
    return img_rgb, h, w


def render_swarm_twoway_gif(
    spline,
    traj,                # (T,N,2) or (traj, extra)
    groups,              # (N,)
    save_path,
    stride=2,
    fps=20,
    trail=40,
    show_corridor=True,
    show_map=True,       # ✅ NEW
    map_alpha=1.0,       # ✅ NEW
    map_downsample=1.0,  # ✅ NEW (0.5 is faster if needed)
    show_axes=False,     # ✅ NEW (usually nicer OFF)
    title="Two-way swarm (A↔B)",
    max_frames=600,
):
    traj = _unwrap_traj(traj)
    traj = np.asarray(traj, dtype=float)
    groups = np.asarray(groups, dtype=int)

    if traj.ndim != 3 or traj.shape[2] != 2:
        raise ValueError(f"Expected traj shape (T,N,2) but got {traj.shape}")

    T, N, _ = traj.shape
    if groups.shape != (N,):
        raise ValueError(f"groups must be shape (N,), got {groups.shape} for N={N}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Auto-adjust stride so GIF doesn’t explode
    frames = list(range(0, T, stride))
    if len(frames) > max_frames:
        stride = int(np.ceil(T / max_frames))
        frames = list(range(0, T, stride))
        print(f"[INFO] Too many frames → auto stride={stride}, frames={len(frames)}")

    # Spline + corridor geometry
    _, pts, tans = _sample_spline(spline, n=600)
    half_width = 0.5 * float(getattr(config, "PATH_WIDTH_PIX", getattr(config, "PATH_WIDTH", 40.0)))
    robot_r = float(getattr(config, "ROBOT_RADIUS", 6.0))
    half_width_inner = max(0.0, half_width - robot_r)
    left, right = _corridor_bounds(pts, tans, half_width_inner)

    # Figure
    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    # Determine view bounds from traj (so it looks tight and nice)
    all_xy = traj.reshape(-1, 2)
    xmin, ymin = np.min(all_xy, axis=0)
    xmax, ymax = np.max(all_xy, axis=0)
    pad = 40

    # ✅ Map background
    if show_map:
        rgb, H, W = _load_map_rgb(downsample=map_downsample)
        if rgb is not None:
            # imshow with origin="upper" matches pixel coordinates (y down)
            ax.imshow(rgb, origin="upper", alpha=float(map_alpha), zorder=0)

            # If map is downsampled, coordinates are still in original pixels.
            # Two options:
            # 1) keep map_downsample=1.0 (recommended)
            # 2) if you downsample, your overlay coordinates will not align.
            # So: only downsample if you ALSO scale your spline/traj (not recommended now).
            if float(map_downsample) != 1.0:
                print("[WARN] map_downsample != 1.0 will misalign overlays unless you scale traj/spline too.")

    # Overlays
    ax.plot(pts[:, 0], pts[:, 1], linewidth=2.2, label="spline", zorder=2)

    if show_corridor:
        ax.plot(left[:, 0], left[:, 1], linestyle="--", linewidth=1.2, label="corridor", zorder=2)
        ax.plot(right[:, 0], right[:, 1], linestyle="--", linewidth=1.2, zorder=2)

    # Split by direction
    idx_pos = np.where(groups == 1)[0]
    idx_neg = np.where(groups == -1)[0]

    p0 = traj[0]
    scat_pos = ax.scatter(p0[idx_pos, 0], p0[idx_pos, 1], s=28, label="A->B (+1)", zorder=3)
    scat_neg = ax.scatter(p0[idx_neg, 0], p0[idx_neg, 1], s=28, label="B->A (-1)", zorder=3)

    # Collision rings
    scat_col = ax.scatter([], [], s=170, facecolors="none", edgecolors="red", linewidths=2, label="collision", zorder=4)

    # Trails
    trail_lines = []
    for _ in range(N):
        ln, = ax.plot([], [], linewidth=1.2, alpha=0.6, zorder=2.5)
        trail_lines.append(ln)

    txt = ax.text(
        0.02, 0.98, "",
        transform=ax.transAxes,
        ha="left", va="top",
        bbox=dict(boxstyle="round", alpha=0.6, facecolor="white", edgecolor="none")
    )

    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")

    if show_axes:
        ax.grid(True, alpha=0.25)
    else:
        ax.axis("off")

    ax.legend(loc="upper right")

    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)

    cumulative_collision_events = 0

    with imageio.get_writer(save_path, mode="I", fps=fps) as writer:
        for fi, frame_idx in enumerate(frames):
            pos = traj[frame_idx]

            scat_pos.set_offsets(pos[idx_pos])
            scat_neg.set_offsets(pos[idx_neg])

            pairs = detect_collisions(pos)
            if pairs:
                involved = sorted(set([i for (i, j, d) in pairs] + [j for (i, j, d) in pairs]))
                scat_col.set_offsets(pos[involved])
                cumulative_collision_events += len(pairs)
            else:
                scat_col.set_offsets(np.zeros((0, 2)))

            start = max(0, frame_idx - trail * stride)
            for i in range(N):
                seg = traj[start:frame_idx + 1, i, :]
                trail_lines[i].set_data(seg[:, 0], seg[:, 1])

            txt.set_text(
                f"frame: {frame_idx}/{T-1}\n"
                f"collisions this frame: {len(pairs)}\n"
                f"collision events (cumulative): {cumulative_collision_events}"
            )

            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            rgb_frame = rgba[..., :3]
            writer.append_data(rgb_frame)

            if (fi + 1) % 50 == 0:
                print(f"[INFO] wrote {fi+1}/{len(frames)} frames...")

    plt.close(fig)
    return save_path
