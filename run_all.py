# run_all.py
from __future__ import annotations

import os
import json
import argparse
import numpy as np
import cv2

from src import config
from src.utils.io_utils import ensure_output_root, create_new_results_dir

from src.map_tools.map_loader import load_map_image
from src.map_tools.map_click_ab import ensure_AB_points

from src.path.path_extraction import extract_path_mask
from src.path.centerline import extract_centerline_points
from src.path.spline_path import build_spline_from_centerline

from src.path.path_controller import PathController
from src.core.core import step_rk2, initialize_robots

from src.visualization.viz import render_path_following

from src.swarm.spawn import spawn_swarm_state
from src.swarm.sim import simulate_swarm_twoway
from src.swarm.viz_swarm import render_swarm_twoway_gif
from src.swarm.collisions import detect_collisions


# -------------------------
# helpers
# -------------------------
def _mkdir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def _save_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _safe_extract_mask() -> np.ndarray:
    """
    Your extract_path_mask sometimes returns (mask, gray, wall_mask) when debug=True.
    Sometimes returns mask only. This wrapper handles both.
    """
    out = extract_path_mask(debug=True) if "debug" in extract_path_mask.__code__.co_varnames else extract_path_mask()
    if isinstance(out, (tuple, list)):
        mask = out[0]
    else:
        mask = out
    # normalize to boolean-like mask
    mask = (mask.astype(np.uint8) > 0)
    return mask


def _draw_centerline_on_map(img_bgr: np.ndarray, pts: np.ndarray, A, B, save_path: str) -> None:
    dbg = img_bgr.copy()
    for p in pts.astype(int):
        x, y = int(p[0]), int(p[1])
        if 0 <= x < dbg.shape[1] and 0 <= y < dbg.shape[0]:
            cv2.circle(dbg, (x, y), 1, (0, 0, 255), -1)
    cv2.circle(dbg, (int(A[0]), int(A[1])), 8, (0, 0, 255), -1)
    cv2.circle(dbg, (int(B[0]), int(B[1])), 8, (255, 0, 0), -1)
    cv2.imwrite(save_path, dbg)


def _overlay_mask(img_bgr: np.ndarray, mask_bool: np.ndarray, save_path: str) -> None:
    overlay = img_bgr.copy()
    overlay[mask_bool] = (0, 255, 0)
    blended = cv2.addWeighted(img_bgr, 0.7, overlay, 0.3, 0.0)
    cv2.imwrite(save_path, blended)


def _controller_target_only(controller: PathController, x: np.ndarray, v: np.ndarray, dt: float) -> np.ndarray:
    """
    Your PathController.update is used in different ways across your tests:
      - target = controller.update(pos)
      - force, target = controller.update(x, v, dt)

    We standardize to "target only".
    """
    try:
        out = controller.update(x, v, dt)
    except TypeError:
        out = controller.update(x)

    if isinstance(out, (tuple, list)) and len(out) == 2:
        # (force, target)
        return np.asarray(out[1], dtype=float)
    return np.asarray(out, dtype=float)


def _simulate_single_robot_follow(spline, steps: int, dt: float) -> np.ndarray:
    """
    Uses YOUR engine step_rk2 (like your test_subproblem_1_validation).
    Returns traj: (T,2)
    """
    controller = PathController(spline)

    pos_batch, vel_batch = initialize_robots(1)
    pos_batch[0] = spline.p(0.0).copy()
    vel_batch[0] = np.zeros(2)

    traj = [pos_batch[0].copy()]

    for _ in range(steps):
        pos = pos_batch[0]
        vel = vel_batch[0]
        target = _controller_target_only(controller, pos, vel, dt)

        # step_rk2 in your project accepts a single target (2,) for n_robots=1
        pos_batch, vel_batch = step_rk2(pos_batch, vel_batch, target, dt)
        traj.append(pos_batch[0].copy())

    return np.asarray(traj, dtype=float)


def _required_steps_for_path(spline_length: float, dt: float, vmax: float, safety: float = 1.25) -> int:
    max_per_step = max(1e-9, dt * vmax)
    base = int(np.ceil(spline_length / max_per_step))
    return int(np.ceil(base * safety)) + 200


# -------------------------
# SUBPROBLEM 1
# -------------------------
def run_subproblem1(base_dir: str, force_reselect_ab: bool) -> dict:
    sub1_dir = _mkdir(os.path.join(base_dir, "subproblem1"))

    # allow reselect on demand
    if force_reselect_ab and hasattr(config, "FORCE_AB_RESELECT"):
        config.FORCE_AB_RESELECT = True

    img = load_map_image()
    H, W = img.shape[:2]

    A, B = ensure_AB_points()
    A = np.array(A, dtype=int)
    B = np.array(B, dtype=int)

    # save A/B overlay
    ab_dbg = img.copy()
    cv2.circle(ab_dbg, (int(A[0]), int(A[1])), 8, (0, 0, 255), -1)   # A red
    cv2.circle(ab_dbg, (int(B[0]), int(B[1])), 8, (255, 0, 0), -1)   # B blue
    ab_path = os.path.join(sub1_dir, "map_with_A_B.png")
    cv2.imwrite(ab_path, ab_dbg)

    # pipeline: mask -> centerline -> spline
    mask = _safe_extract_mask()
    mask_path = os.path.join(sub1_dir, "path_mask_overlay.png")
    _overlay_mask(img, mask, mask_path)

    centerline = extract_centerline_points(mask=mask, A=A, B=B)
    centerline = np.asarray(centerline, dtype=float)

    centerline_dbg_path = os.path.join(sub1_dir, "centerline_debug.png")
    _draw_centerline_on_map(img, centerline, A, B, centerline_dbg_path)

    spline = build_spline_from_centerline(centerline)

    # simulate
    dt = float(getattr(config, "DT", 0.05))
    vmax = float(getattr(config, "VMAX", 2.0))
    steps = _required_steps_for_path(float(spline.length), dt, vmax)

    traj = _simulate_single_robot_follow(spline, steps=steps, dt=dt)

    # save planned-vs-actual using your renderer (produces GIF)
    gif_path = os.path.join(sub1_dir, "sub1_path_following.gif")
    render_path_following(spline, traj, gif_path)  # your function converts .png to .gif internally

    # metrics
    final_pos = traj[-1]
    final_s = float(spline.closest_s(final_pos))
    progress = final_s / max(1e-9, float(spline.length))

    metrics = {
        "A": [int(A[0]), int(A[1])],
        "B": [int(B[0]), int(B[1])],
        "map_hw": [int(H), int(W)],
        "spline_length": float(spline.length),
        "dt": float(dt),
        "vmax": float(vmax),
        "steps": int(steps),
        "progress_ratio": float(progress),
        "artifacts": {
            "map_with_A_B": ab_path,
            "path_mask_overlay": mask_path,
            "centerline_debug": centerline_dbg_path,
            "path_following_gif": gif_path.replace(".png", ".gif"),
        },
    }
    _save_json(os.path.join(sub1_dir, "sub1_metrics.json"), metrics)
    return metrics


# -------------------------
# SUBPROBLEM 2
# -------------------------
def run_subproblem2(base_dir: str, N: int, gif_max_frames: int = 500) -> dict:
    sub2_dir = _mkdir(os.path.join(base_dir, "subproblem2"))

    # build spline from current pipeline (same A/B as Sub1)
    mask = _safe_extract_mask()
    A, B = ensure_AB_points()
    centerline = extract_centerline_points(mask=mask, A=A, B=B)
    spline = build_spline_from_centerline(np.asarray(centerline, dtype=float))

    # spawn + simulate
    positions, velocities, groups = spawn_swarm_state(spline, N=N, seed=int(getattr(config, "SEED", 0) or 0))

    dt = float(getattr(config, "DT", 0.05))
    vmax = float(getattr(config, "VMAX", 2.0))
    L = float(spline.length)

    steps = int((L / max(1e-6, vmax)) / dt) * 2
    steps = int(np.clip(steps, 450, 6000))

    sim_out = simulate_swarm_twoway(
        spline=spline,
        positions=positions,
        velocities=velocities,
        groups=groups,
        steps=steps,
        dt=dt,
    )

    # simulate_swarm_twoway sometimes returns (traj, collision_log)
    if isinstance(sim_out, (tuple, list)) and len(sim_out) >= 1:
        traj = np.asarray(sim_out[0], dtype=float)
    else:
        traj = np.asarray(sim_out, dtype=float)

    # collision metrics (sampled)
    check_stride = max(1, steps // 300)
    collisions = 0
    min_pair_dist = float("inf")

    for t in range(0, steps + 1, check_stride):
        pos = traj[t]
        pairs = detect_collisions(pos)
        collisions += len(pairs)
        # track min dist among reported pairs (good enough)
        for (_, _, d) in pairs:
            min_pair_dist = min(min_pair_dist, float(d))

    # render GIF (this is your official visual proof)
    gif_path = os.path.join(sub2_dir, f"sub2_twoway_swarm_N{int(N)}.gif")
    stride = max(2, int(np.ceil((steps + 1) / max(1, gif_max_frames))))
    render_swarm_twoway_gif(
        spline=spline,
        traj=traj,
        groups=groups,
        save_path=gif_path,
        stride=stride,
        fps=20,
        trail=35,
        show_corridor=True,
        show_map=True,
        map_alpha=1.0,
        map_downsample=1.0,
        show_axes=False,
        title=f"Two-way swarm (N={int(N)})",
        max_frames=gif_max_frames,
    )

    metrics = {
        "N": int(N),
        "dt": float(dt),
        "vmax": float(vmax),
        "steps": int(steps),
        "collision_events_sampled": int(collisions),
        "min_pair_dist_from_collisions_px": (None if min_pair_dist == float("inf") else float(min_pair_dist)),
        "gif_path": gif_path,
        "gif_stride_used": int(stride),
    }
    _save_json(os.path.join(sub2_dir, "sub2_metrics.json"), metrics)
    return metrics


# -------------------------
# SUBPROBLEM 3
# -------------------------
def run_subproblem3(base_dir: str, max_seconds: float | None, stride: int, resize_width: int) -> dict:
    """
    Runs your Phase 5 missions runner inside subproblem3 folder.
    Videos STOPPING at goal requires the tiny patch below in src/phase5/run_missions.py.
    If patch isn't present, it will still run but won't early-stop.
    """
    sub3_dir = _mkdir(os.path.join(base_dir, "subproblem3"))

    # Try to use your phase5 runner
    from src.phase5.run_missions import run as run_phase5_missions

    # Set knobs for this run (support both naming styles used across your project)
    if hasattr(config, "P5_RUN_MAX_SECONDS"):
        config.P5_RUN_MAX_SECONDS = max_seconds
    if hasattr(config, "P5_RUN_FRAME_STRIDE"):
        config.P5_RUN_FRAME_STRIDE = int(stride)
    if hasattr(config, "P5_RUN_RESIZE_WIDTH"):
        config.P5_RUN_RESIZE_WIDTH = int(resize_width)

    # also support the older names your tests use
    if hasattr(config, "P5_MAX_SECONDS"):
        config.P5_MAX_SECONDS = max_seconds
    if hasattr(config, "P5_FRAME_STRIDE"):
        config.P5_FRAME_STRIDE = int(stride)
    if hasattr(config, "P5_RESIZE_WIDTH"):
        config.P5_RESIZE_WIDTH = int(resize_width)

    out = run_phase5_missions(sub3_dir)
    _save_json(os.path.join(sub3_dir, "sub3_outputs.json"), out)
    return out


# -------------------------
# MAIN
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=None, help="Optional existing output/results_XXXX folder.")
    ap.add_argument("--reselect-ab", action="store_true", help="Force reselect A/B on this run.")
    ap.add_argument("--skip-sub1", action="store_true")
    ap.add_argument("--skip-sub2", action="store_true")
    ap.add_argument("--skip-sub3", action="store_true")

    ap.add_argument("--swarm-n", type=int, default=int(getattr(config, "N_ROBOTS", 24)))
    ap.add_argument("--gif-max-frames", type=int, default=500)

    ap.add_argument("--p5-seconds", type=float, default=10.0, help="Phase5 runtime seconds. Use --p5-full for full video.")
    ap.add_argument("--p5-full", action="store_true", help="Use full pedestrians.mp4 for phase5.")
    ap.add_argument("--p5-stride", type=int, default=2)
    ap.add_argument("--p5-resize", type=int, default=int(getattr(config, "P5_RESIZE_WIDTH", 640) or 640))

    args = ap.parse_args()

    if args.results_dir is None:
        out_root = ensure_output_root("output")
        results_dir = create_new_results_dir(out_root)
    else:
        results_dir = args.results_dir
        os.makedirs(results_dir, exist_ok=True)

    run_all_dir = _mkdir(os.path.join(results_dir, "run_all"))

    print("===============================================")
    print("RUN ALL — Subproblem 1 + 2 + 3")
    print(f"Results dir: {results_dir}")
    print(f"Artifacts:  {run_all_dir}")
    print("===============================================")

    summary = {"results_dir": results_dir, "run_all_dir": run_all_dir, "ok": True}

    if not args.skip_sub1:
        print("\n--- SUBPROBLEM 1 ---")
        try:
            summary["subproblem1"] = run_subproblem1(run_all_dir, force_reselect_ab=bool(args.reselect_ab))
        except Exception as e:
            summary["ok"] = False
            summary["subproblem1"] = {"error": str(e)}
            print(f"[FAIL] Subproblem 1 crashed: {e}")

    if not args.skip_sub2:
        print("\n--- SUBPROBLEM 2 ---")
        try:
            summary["subproblem2"] = run_subproblem2(run_all_dir, N=int(args.swarm_n), gif_max_frames=int(args.gif_max_frames))
        except Exception as e:
            summary["ok"] = False
            summary["subproblem2"] = {"error": str(e)}
            print(f"[FAIL] Subproblem 2 crashed: {e}")

    if not args.skip_sub3:
        print("\n--- SUBPROBLEM 3 ---")
        try:
            max_seconds = None if args.p5_full else float(args.p5_seconds)
            summary["subproblem3"] = run_subproblem3(
                run_all_dir,
                max_seconds=max_seconds,
                stride=int(args.p5_stride),
                resize_width=int(args.p5_resize),
            )
        except Exception as e:
            summary["ok"] = False
            summary["subproblem3"] = {"error": str(e)}
            print(f"[FAIL] Subproblem 3 crashed: {e}")

    _save_json(os.path.join(run_all_dir, "run_all_summary.json"), summary)

    print("\n===============================================")
    print("DONE")
    print(f"Summary: {os.path.join(run_all_dir, 'run_all_summary.json')}")
    print("===============================================")


if __name__ == "__main__":
    main()
